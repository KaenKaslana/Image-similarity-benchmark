"""Multi-object test cases: six different objects, each with front/side/top views.

Every candidate carries a different kind of deviation, so these tests check
that each metric reacts to the right effect in the right views, and that
per-object group scores are computed.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest

from src.benchmark import BenchmarkRunner, PairResult, compute_group_scores, group_name
from src.config import BenchmarkConfig, config_from_dict
from src.objects import VIEWS, object_test_cases, render_object_views, write_object_set

OBJECTS = ("robot", "mug", "table", "bottle", "house", "lamp", "mismatch")


def _cfg(**pre) -> BenchmarkConfig:
    return config_from_dict({"preprocessing": {"canvas_size": 160, **pre}})


@pytest.fixture(scope="module")
def object_run(tmp_path_factory: pytest.TempPathFactory) -> dict[str, PairResult]:
    """Run the whole object set once and index results by file name."""
    root = tmp_path_factory.mktemp("objects")
    ref, cand = root / "reference", root / "candidate"
    names = write_object_set(ref, cand, size=192)
    assert len(names) == len(OBJECTS) * 3
    result = BenchmarkRunner(_cfg()).run(ref, cand, None)
    assert all(p.ok for p in result.pairs)
    by_name = {p.name: p for p in result.pairs}
    by_name["__result__"] = result  # type: ignore[assignment]
    return by_name


def _views(run: dict[str, PairResult], obj: str) -> dict[str, PairResult]:
    return {v: run[f"{obj}_{v}.png"] for v in VIEWS}


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def test_every_object_renders_three_distinct_views() -> None:
    for name, spec in object_test_cases().items():
        views = render_object_views(spec["reference"], size=96)
        assert set(views) == {"front.png", "side.png", "top.png"}, name
        for img in views.values():
            assert img.mode == "RGBA" and img.size == (96, 96)
            alpha = img.getchannel("A")
            assert 0 < sum(1 for a in alpha.getdata() if a > 0) < 96 * 96, f"{name}: view is empty or full"


def test_objects_are_different_from_each_other() -> None:
    from src.metrics import compute_silhouette_iou
    from src.preprocessing import preprocess_pil

    cfg = _cfg().preprocessing
    fronts = {n: preprocess_pil(render_object_views(s["reference"], size=128)["front.png"], cfg)
              for n, s in object_test_cases().items()}
    names = list(fronts)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            iou = compute_silhouette_iou(fronts[a].mask, fronts[b].mask)
            assert iou is not None and iou < 0.8, f"{a} vs {b} look too similar (IoU {iou:.2f})"


def test_write_object_set_names_and_modes(tmp_path: Path) -> None:
    names = write_object_set(tmp_path / "r", tmp_path / "c", size=64, mode="RGB")
    assert names[:3] == ["robot_front.png", "robot_side.png", "robot_top.png"]
    for n in names:
        assert (tmp_path / "r" / n).is_file() and (tmp_path / "c" / n).is_file()


# ---------------------------------------------------------------------------
# Per-object expectations
# ---------------------------------------------------------------------------
def test_robot_identical_scores_100(object_run) -> None:
    for view, p in _views(object_run, "robot").items():
        assert p.pair_score > 99.5, (view, p.pair_score)
        assert p.silhouette_iou > 0.999 and p.lpips_distance < 1e-4


def test_mug_missing_handle_hits_front_and_top_not_side(object_run) -> None:
    v = _views(object_run, "mug")
    assert v["side"].pair_score > 99.5  # handle is hidden behind the body in the side view
    for view in ("front", "top"):
        assert v[view].silhouette_iou < 0.95, view
        assert v[view].edge_score < 90, view
        assert v[view].pair_score < v["side"].pair_score - 3, view


def test_table_proportion_change_hits_front_and_side_not_top(object_run) -> None:
    v = _views(object_run, "table")
    assert v["top"].pair_score > 99.5
    for view in ("front", "side"):
        assert v[view].silhouette_iou < 0.85, view
        assert v[view].pair_score < 90, view


def test_bottle_neck_change_is_moderate(object_run) -> None:
    v = _views(object_run, "bottle")
    for view in ("front", "side"):
        assert 0.7 < v[view].silhouette_iou < 0.98, view
        assert 70 < v[view].pair_score < 99, view
    assert v["top"].pair_score > 95  # neck radius change only alters the tiny cap circle


def test_house_colour_change_hits_lpips_and_ssim(object_run) -> None:
    v = _views(object_run, "house")
    for view, p in v.items():
        assert p.lpips_distance > 0.05, view
        assert p.ssim < 0.97, view
        assert p.silhouette_iou > 0.85, view  # geometry almost unchanged (only the chimney)
    # chimney is visible in front and side, not in top
    assert v["front"].silhouette_iou < 0.995 and v["side"].silhouette_iou < 0.995
    assert v["top"].silhouette_iou > 0.995


def test_lamp_misalignment_is_corrected_by_default_alignment(object_run, tmp_path: Path) -> None:
    # Default run (phase_correlation alignment): the shifted lamp is geometrically
    # identical, so silhouette / edge must be ~perfect and the shift recorded.
    v = _views(object_run, "lamp")
    for view, p in v.items():
        assert p.alignment["method"] == "phase_correlation" and p.alignment["applied"], view
        dx, dy = p.alignment["shift_px"]
        assert dx < -2 and dy < -2, (view, p.alignment)  # candidate was moved +5 %, +3 %
        # The shift is integer-pixel and the lamp pole is only ~5 px wide at this
        # test resolution, so a 1 px residual costs a few IoU points (0.97 at 512 px).
        assert p.silhouette_iou > 0.9, (view, p.silhouette_iou)
        assert p.edge_score > 90, (view, p.edge_score)
        # Absolute pair score is limited by the synthetic texture noise, which is
        # fixed to canvas coordinates and therefore does not move with the object.
        assert p.pair_score > 85, (view, p.pair_score)

    # Without alignment the same data is penalised heavily.
    spec = object_test_cases()["lamp"]
    ref, cand = tmp_path / "r", tmp_path / "c"
    write_object_set(ref, cand, size=192, objects={"lamp": spec})
    none = {p.name: p for p in BenchmarkRunner(_cfg(alignment="none")).run(ref, cand, None).pairs}
    for name, p in none.items():
        assert p.silhouette_iou < 0.9, name
        assert p.pair_score < 95, name
        assert p.pair_score < v[name.split("_", 1)[1].replace(".png", "")].pair_score - 5, name


def test_mismatch_two_different_objects_scores_low_everywhere(object_run) -> None:
    v = _views(object_run, "mismatch")
    for view, p in v.items():
        assert p.edge_score < 50, (view, p.edge_score)
        assert p.lpips_distance > 0.15, (view, p.lpips_distance)
    # Front and side views separate a chair from a lamp very clearly.
    for view in ("front", "side"):
        assert v[view].silhouette_iou < 0.4, (view, v[view].silhouette_iou)
        assert v[view].pair_score < 65, (view, v[view].pair_score)
    # From above both objects are a centred blob (seat vs shade), so the top
    # view alone is genuinely ambiguous: lower than a matching object, but not tiny.
    assert v["top"].silhouette_iou < 0.85
    assert v["top"].pair_score < 85


def test_score_ordering_across_objects(object_run) -> None:
    result = object_run["__result__"]
    g = result.group_scores
    assert set(g) == set(OBJECTS)
    assert g["robot"]["score"] > 99.5
    assert g["robot"]["score"] > g["bottle"]["score"] > g["table"]["score"]
    assert g["robot"]["score"] > g["mug"]["score"]
    assert g["robot"]["score"] > g["house"]["score"]
    assert g["robot"]["score"] > g["lamp"]["score"]
    # two completely different objects must be the worst group by a clear margin
    others = [g[o]["score"] for o in OBJECTS if o != "mismatch"]
    assert g["mismatch"]["score"] < min(others) - 5


def test_one_report_per_object_plus_summary(tmp_path: Path) -> None:
    from PIL import Image

    ref, cand = tmp_path / "r", tmp_path / "c"
    cases = object_test_cases()
    write_object_set(ref, cand, size=96, objects={"robot": cases["robot"], "mismatch": cases["mismatch"]})
    result = BenchmarkRunner(_cfg()).run(ref, cand, tmp_path / "out")
    names = sorted(p.name for p in result.run_dir.glob("report*.png"))
    # exactly one report per object (three views inside) + one summary table
    assert names == ["report.png", "report_mismatch.png", "report_robot.png"]
    per_object = Image.open(result.run_dir / "report_robot.png")
    summary = Image.open(result.run_dir / "report.png")
    assert per_object.height > 2 * summary.height  # three image rows vs. a compact table
    assert not (result.run_dir / "reports").exists()

    # Multi-page per-object report only when pairs_per_page is smaller than the view count.
    cfg3 = _cfg()
    cfg3.output.report_pairs_per_page = 2
    result3 = BenchmarkRunner(cfg3).run(ref, cand, tmp_path / "out3")
    names3 = sorted(p.name for p in result3.run_dir.glob("report_robot*.png"))
    assert names3 == ["report_robot.png", "report_robot_page02.png"]


def test_single_object_without_prefix_gets_one_report(tmp_path: Path) -> None:
    ref, cand = tmp_path / "r", tmp_path / "c"
    ref.mkdir(), cand.mkdir()
    for view_file, img in render_object_views(object_test_cases()["mug"]["reference"], size=96).items():
        img.save(ref / view_file)
        img.save(cand / view_file)
    result = BenchmarkRunner(_cfg()).run(ref, cand, tmp_path / "out")
    assert result.group_scores == {}
    assert sorted(p.name for p in result.run_dir.glob("report*.png")) == ["report.png"]


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------
def test_group_name_parsing() -> None:
    assert group_name("mug_front.png", "_") == "mug"
    assert group_name("Mug_Front.PNG", "_") == "Mug"
    assert group_name("front.png", "_") is None
    assert group_name("mug_front.png", "") is None
    assert group_name("_front.png", "_") is None
    assert group_name("a-b_c.png", "-") == "a"


def test_group_scores_average_valid_pairs_only() -> None:
    def pr(name: str, score: float | None, error: str | None = None) -> PairResult:
        return PairResult(name=name, reference_path="", candidate_path="", pair_score=score, error=error)

    results = [pr("mug_front.png", 80.0), pr("mug_side.png", 90.0), pr("mug_top.png", None, "broken"),
               pr("cup_front.png", 50.0), pr("loose.png", 10.0)]
    groups = compute_group_scores(results, "_")
    assert list(groups) == ["cup", "mug"]
    assert groups["mug"] == {"score": 85.0, "num_pairs": 2, "pairs": ["mug_front.png", "mug_side.png"]}
    assert groups["cup"]["score"] == 50.0
    assert compute_group_scores(results, "") == {}


def test_group_scores_in_json_csv_and_console(tmp_path: Path) -> None:
    from src.reporting import format_summary_table

    ref, cand = tmp_path / "r", tmp_path / "c"
    cases = object_test_cases()
    write_object_set(ref, cand, size=96, objects={"robot": cases["robot"], "mug": cases["mug"]})
    result = BenchmarkRunner(_cfg()).run(ref, cand, tmp_path / "out")
    assert set(result.group_scores) == {"robot", "mug"}
    data = json.loads((result.run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert math.isclose(data["group_scores"]["robot"]["score"], result.group_scores["robot"]["score"])
    with (result.run_dir / "metrics.csv").open(encoding="utf-8", newline="") as fh:
        names = [r["name"] for r in csv.DictReader(fh)]
    assert names[-3:] == ["__group__:mug", "__group__:robot", "__overall__"]
    table = format_summary_table(result)
    assert "[robot] (3 views)" in table and "[mug] (3 views)" in table
    # a two-page report is produced for 6 pairs with the default 6 per page -> exactly one page
    assert (result.run_dir / "report.png").is_file()
