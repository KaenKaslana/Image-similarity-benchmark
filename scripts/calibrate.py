"""Calibrate ``score_floors`` / ``score_gamma`` / weights of configs/shape.yaml on sample pairs.

Two stages, both free of AI-generation credits:

1. ``measure``: for every sample pair, auto-orient the candidate, render the
   views, run the shape pipeline and cache the RAW metric scores in
   ``outputs/calibration/raw.json`` (slow; parallel; resumable).
2. ``fit``: grid-search floors, gamma and weights against target score bands
   per pair class and print the per-class score distribution.

Pair classes
    identical   a model against itself                         -> 100
    mild/medium/strong
                a model against a deformed copy of itself (non-uniform
                scale, bend, twist) - stand-ins for replicas of decreasing
                fidelity
    ai_image / ai_text
                real AI replicas listed in ``AI_REPLICAS`` (only used if the
                files exist locally)
    same        two different models of the same category
    unrelated   models from different categories

Models are Sketchfab uids cached in ``models/`` (download them first with
``fetch-sketchfab``; missing ones are skipped).

Usage::

    python scripts/calibrate.py measure --workers 5
    python scripts/calibrate.py fit
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MODELS = PROJECT_ROOT / "models"
RAW = PROJECT_ROOT / "outputs" / "calibration" / "raw.json"
VIEWS = ("front", "side", "top")

# category -> Sketchfab uids (single, clean objects; screened by eye)
SAMPLES: dict[str, list[str]] = {
    "chair": [
        "6479a1900b614b59b26784c3a7922eb3",  # Victorian chair
        "003b00c7cdd348888296b85c40275a50",  # Worn Leather Office Chair
        "675f34f7304e4d92812a41e9750539aa",  # Office Chair Modern
        "f16281407afc45ceb189b5d870afb849",  # Mercury Chair Regency Period
        "3f95ffdc194047ada2d9391828a5d234",  # Old Wooden Chair
    ],
    "character": ["a9e99a47d2a24957ab6c08c9e57ad48c", "6cdd055b4afa41eb9360dbbfe75c7f10"],
    "dragon": ["ecf63885166c40e2bbbcdf11cd14e65f"],
    "mug": ["5f5ccee1514c440887c072fae8e0d699"],
    "sword": ["b2662f2666a844e8a1bd0e7c4a7672d8", "09ff916bd580422395bec19b8a9cbad6"],
    "car": ["61766c1e6e1e4b2b803c174f33433a74", "3fa89e6bf78b49aab111af4195efb6fe"],
    "guitar": [
        "a7ffc570d3fe41c89b9dde195ab0faea",
        "2007af7561fe46958d1f7e92dff8a40d",
        "daeadd913644438aa3096bb357a02016",
    ],
    "table": [
        "0dc1c7d6cbab4d74bef7c4f82abf2caf",
        "1132fa2850a24917892733566bd68e74",
        "e53a5ef2faf84edcaddf49d4d27f366c",
    ],
    "teapot": [
        "9b087673052446299e6decda222a42da",
        "252c28595d3c45fb87095faf71466113",
        "27f2ac58f7614f2796630bdc6f18ee2f",
    ],
    "sneaker": ["05e79a2dffbc4356bba7e0751fada08f"],
    "jet": [
        "7a13c91f07e042a685b4d265644fdc06",
        "ba8c9ca82b204222834997996dfc1e1e",
        "e91124b750b74223bd5cb18d03a254cb",
    ],
}

# (class, reference uid, generated file under models/generated/)
AI_REPLICAS = [
    ("ai_image", "6479a1900b614b59b26784c3a7922eb3", "tripo_519287de-a375-44c2-8de8-dcc6157cb760.glb"),
    ("ai_image", "a9e99a47d2a24957ab6c08c9e57ad48c", "tripo_20bae81a-0cbb-471b-9717-46a11520f827.glb"),
    ("ai_text", "6479a1900b614b59b26784c3a7922eb3", "tripo_3aec1d1e-a4c5-4f47-af03-6b01dd364947.glb"),
    ("ai_text", "a9e99a47d2a24957ab6c08c9e57ad48c", "tripo_874456d6-c428-4444-9336-a744d61db289.glb"),
    ("ai_text", "ecf63885166c40e2bbbcdf11cd14e65f", "tripo_923186f8-4cc9-4919-8c51-bb27a2cb6212.glb"),
]

# deformation levels: (non-uniform scale xyz, bend, twist degrees, noise)
DEFORM = {
    "mild": ((1.04, 0.97, 1.03), 0.00, 0.0, 0.003),
    "medium": ((1.12, 0.92, 1.08), 0.08, 8.0, 0.006),
    "strong": ((1.25, 0.82, 1.15), 0.18, 25.0, 0.010),
}


def model_path(uid: str) -> Path | None:
    for ext in (".glb", ".zip"):
        p = MODELS / f"{uid}{ext}"
        if p.is_file():
            if ext == ".zip":
                hits = sorted((MODELS / uid).rglob("*.gltf"))
                return hits[0] if hits else None
            return p
    return None


def build_pairs(seed: int = 7, n_unrelated: int = 30, n_deform_refs: int = 9) -> list[dict]:
    rng = random.Random(seed)
    have = {c: [u for u in uids if model_path(u)] for c, uids in SAMPLES.items()}
    pairs: list[dict] = []
    refs = [u for uids in have.values() for u in uids]
    for u in refs[:: max(1, len(refs) // 4)][:4]:
        pairs.append({"cls": "identical", "ref": u, "cand": u})
    deform_refs = [uids[0] for uids in have.values() if uids][:n_deform_refs]
    for u in deform_refs:
        for level in DEFORM:
            pairs.append({"cls": level, "ref": u, "cand": f"deform:{level}:{u}"})
    for cls, ref, fname in AI_REPLICAS:
        if model_path(ref) and (MODELS / "generated" / fname).is_file():
            pairs.append({"cls": cls, "ref": ref, "cand": f"file:generated/{fname}"})
    for cat, uids in have.items():
        combos = list(itertools.combinations(uids, 2))
        rng.shuffle(combos)
        for a, b in combos[:4]:
            pairs.append({"cls": "same", "ref": a, "cand": b, "category": cat})
    cross = [(a, b) for (ca, ua), (cb, ub) in itertools.combinations(have.items(), 2) for a in ua for b in ub]
    rng.shuffle(cross)
    for a, b in cross[:n_unrelated]:
        pairs.append({"cls": "unrelated", "ref": a, "cand": b})
    for p in pairs:
        p["key"] = f"{p['cls']}|{p['ref']}|{p['cand']}"
    return pairs


def deformed_mesh(uid: str, level: str, out_dir: Path) -> Path:
    import trimesh

    scale, bend, twist, noise = DEFORM[level]
    mesh = trimesh.load(str(model_path(uid)), force="mesh")
    v = np.asarray(mesh.vertices, dtype=np.float64)
    lo, hi = v.min(0), v.max(0)
    v = (v - (lo + hi) / 2) / (hi - lo).max()  # unit box, centred
    v = v * np.array(scale)
    up = int(np.argmax(hi - lo))  # deform along the longest axis
    a, b = [i for i in range(3) if i != up]
    t = v[:, up]
    v[:, a] += bend * (t * 2) ** 2 / 2
    ang = np.deg2rad(twist) * t * 2
    va, vb = v[:, a].copy(), v[:, b].copy()
    v[:, a], v[:, b] = va * np.cos(ang) - vb * np.sin(ang), va * np.sin(ang) + vb * np.cos(ang)
    rng = np.random.default_rng(0)
    v += noise * np.sin(v[:, [1, 2, 0]] * 9 + rng.uniform(0, 6, 3))  # smooth low-frequency wobble
    path = out_dir / f"{uid}_{level}.glb"
    trimesh.Trimesh(v, mesh.faces, process=False).export(path)
    return path


_RUNNER = None


def measure_pair(pair: dict) -> dict:
    """Raw metric scores of one pair with the shape pipeline (floors ignored)."""
    global _RUNNER
    from PIL import Image

    from src.benchmark import BenchmarkRunner
    from src.config import load_config
    from src.orient import apply_orientation, auto_orient
    from src.render import RenderOptions, load_mesh, render_view

    if _RUNNER is None:
        cfg = load_config(PROJECT_ROOT / "configs" / "shape.yaml")
        cfg.metrics.lpips.device = "cpu"
        cfg.output.save_preprocessed = cfg.output.save_comparisons = False
        _RUNNER = BenchmarkRunner(cfg)
    with tempfile.TemporaryDirectory(prefix="imgsim_cal_") as tmp:
        tmp = Path(tmp)
        ref_path = model_path(pair["ref"])
        cand = pair["cand"]
        if cand.startswith("deform:"):
            _, level, uid = cand.split(":")
            cand_path = deformed_mesh(uid, level, tmp)
        elif cand.startswith("file:"):
            cand_path = MODELS / cand[5:]
        else:
            cand_path = model_path(cand)
        opts = RenderOptions(size=512, views=VIEWS)
        ref_mesh = load_mesh(ref_path)
        base = load_mesh(cand_path)
        best = auto_orient(base, ref_mesh, VIEWS)
        cand_mesh = apply_orientation(base, best.up, best.front, best.yaw)
        views = {}
        for v in VIEWS:
            for side, mesh in (("ref", ref_mesh), ("cand", cand_mesh)):
                Image.fromarray(render_view(mesh, v, opts), mode="RGBA").save(tmp / f"{side}_{v}.png")
            r = _RUNNER.compare_pair(tmp / f"ref_{v}.png", tmp / f"cand_{v}.png", name=v)
            views[v] = {
                "silhouette": r.silhouette_score,
                "edge": r.edge_score,
                "lpips": r.lpips_score,
                "ssim": r.ssim_score,
            }
    return {**pair, "views": views, "orient": {"up": best.up, "front": best.front, "yaw": best.yaw}}


# ---------------------------------------------------------------------------
# fitting
# ---------------------------------------------------------------------------
METRICS = ("silhouette", "edge", "lpips")

# class -> (low, high) target band for the OVERALL score; None = unbounded
TARGETS = {
    "identical": (99.5, None),
    "mild": (90, None),
    "medium": (75, 95),
    "strong": (50, 80),
    "ai_image": (82, None),
    "ai_text": (50, 62),
    "unrelated": (None, 12),
}


def power_mean(vals: list[float], power: float) -> float:
    """Power mean of view scores; 1 = arithmetic mean, smaller leans towards the weakest view."""
    arr = np.clip(np.asarray(vals, dtype=np.float64), 0.0, None)
    if power == 1.0:
        return float(arr.mean())
    return float(np.mean(arr**power) ** (1.0 / power))


def overall(rec: dict, floors: dict, gamma: float, weights: dict, power: float = 1.0) -> float:
    vals = []
    for v in rec["views"].values():
        total = wsum = 0.0
        for m in METRICS:
            s = v.get(m)
            if s is None or weights[m] <= 0:
                continue
            x = min(1.0, max(0.0, (s - floors[m]) / (100.0 - floors[m])))
            total += weights[m] * 100.0 * x**gamma
            wsum += weights[m]
        vals.append(total / wsum if wsum else 0.0)
    return power_mean(vals, power)


def loss(scores: dict[str, list[float]]) -> float:
    out = 0.0
    for cls, (lo, hi) in TARGETS.items():
        vals = scores.get(cls) or []
        if not vals:
            continue
        err = [max(0.0, (lo or -1e9) - s) + max(0.0, s - (hi if hi is not None else 1e9)) for s in vals]
        out += float(np.mean(np.square(err)))  # every class counts equally, whatever its size
    return out


def fit(records: list[dict]) -> None:
    by_cls: dict[str, list[dict]] = {}
    for r in records:
        by_cls.setdefault(r["cls"], []).append(r)
    grid = itertools.product(
        (30, 35, 40, 45, 50, 55),  # silhouette floor
        (20, 25, 30, 35, 40, 45),  # edge floor
        (65, 70, 75, 80),  # lpips floor
        (0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0),  # gamma
        ((0.55, 0.25, 0.20), (0.50, 0.30, 0.20), (0.60, 0.20, 0.20), (0.45, 0.25, 0.30), (0.70, 0.30, 0.0)),
        (1.0, 0.5, 0.25),  # view aggregation power
    )
    results = []
    for fs, fe, fl, g, w, pw in grid:
        floors = {"silhouette": fs, "edge": fe, "lpips": fl}
        weights = dict(zip(METRICS, w))
        scores = {c: [overall(r, floors, g, weights, pw) for r in rs] for c, rs in by_cls.items()}
        results.append((loss(scores), floors, g, weights, scores, pw))
    results.sort(key=lambda t: t[0])

    def show(title: str, floors: dict, g: float, weights: dict, scores: dict, pw: float = 1.0) -> None:
        print(f"\n{title}\n  floors {floors}  gamma {g}  weights {weights}  view_power {pw}")
        print(f"  {'class':<10} {'n':>3} {'min':>6} {'median':>7} {'max':>6}   target")
        for cls in ("identical", "mild", "medium", "strong", "ai_image", "ai_text", "same", "unrelated"):
            vals = scores.get(cls)
            if vals:
                print(f"  {cls:<10} {len(vals):>3} {min(vals):6.1f} {np.median(vals):7.1f} {max(vals):6.1f}   {TARGETS.get(cls, '-')}")

    from src.config import load_config

    cur = load_config(PROJECT_ROOT / "configs" / "shape.yaml")
    cur_f = {m: cur.score_floors[m] for m in METRICS}
    cur_w = {m: cur.weights[m] for m in METRICS}
    cur_scores = {c: [overall(r, cur_f, cur.score_gamma, cur_w, cur.view_power) for r in rs] for c, rs in by_cls.items()}
    show(f"CURRENT configs/shape.yaml (loss {loss(cur_scores):.1f})", cur_f, cur.score_gamma, cur_w, cur_scores, cur.view_power)
    for rank, (l, floors, g, weights, scores, pw) in enumerate(results[:3], 1):
        show(f"BEST #{rank} (loss {l:.1f})", floors, g, weights, scores, pw)
    for pw in (1.0, 0.5, 0.25):
        l, floors, g, weights, scores, _ = next(r for r in results if r[5] == pw)
        print(f"\nbest loss with view_power {pw}: {l:.1f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["measure", "fit", "list"])
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    pairs = build_pairs()
    if args.command == "list":
        for p in pairs:
            print(p["key"])
        print(len(pairs), "pairs")
        return 0
    RAW.parent.mkdir(parents=True, exist_ok=True)
    cache: dict[str, dict] = json.loads(RAW.read_text()) if RAW.is_file() else {}
    if args.command == "measure":
        todo = [p for p in pairs if p["key"] not in cache]
        print(f"{len(pairs)} pairs, {len(todo)} to measure with {args.workers} workers")
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(measure_pair, p): p for p in todo}
            for i, fut in enumerate(as_completed(futures), 1):
                p = futures[fut]
                try:
                    cache[p["key"]] = fut.result()
                except Exception as exc:  # keep going; report at the end
                    print(f"  FAILED {p['key']}: {exc}")
                    continue
                RAW.write_text(json.dumps(cache, indent=1))
                print(f"  [{i}/{len(todo)}] {p['key'][:70]}", flush=True)
        return 0
    records = [cache[p["key"]] for p in pairs if p["key"] in cache]
    print(f"{len(records)} measured pairs of {len(pairs)}")
    fit(records)
    return 0


if __name__ == "__main__":
    sys.exit(main())
