"""End-to-end tests: pairing, running, outputs, CLI and config loading."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest
import yaml
from PIL import Image

from src.benchmark import BenchmarkRunner, PairingError, compute_overall_score, discover_images, pair_images
from src.cli import main
from src.config import BenchmarkConfig, ConfigError, config_from_dict, load_config
from src.synthetic import adjust_brightness, draw_shape, write_sample_set

EXTS = [".png", ".jpg", ".jpeg"]


def _small_config(**overrides) -> BenchmarkConfig:
    cfg = config_from_dict({"preprocessing": {"canvas_size": 128, **overrides}})
    return cfg


@pytest.fixture
def dataset(tmp_path: Path) -> tuple[Path, Path]:
    ref, cand = tmp_path / "reference", tmp_path / "candidate"
    write_sample_set(ref, cand, size=128)
    return ref, cand


# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------
def test_discover_images_filters_and_is_case_insensitive(tmp_path: Path) -> None:
    draw_shape(16).save(tmp_path / "A.PNG")
    draw_shape(16, mode="RGB").save(tmp_path / "b.jpg")
    (tmp_path / "notes.txt").write_text("x")
    found = discover_images(tmp_path, EXTS)
    assert set(found) == {"a.png", "b.jpg"}


def test_discover_rejects_case_collision(tmp_path: Path) -> None:
    sub = tmp_path / "s"
    sub.mkdir()
    draw_shape(16).save(sub / "a.png")
    # Windows file systems are case-insensitive, so simulate the collision via a Linux-only check.
    draw_shape(16).save(sub / "A.png")
    names = {p.name for p in sub.iterdir()}
    if len(names) == 1:
        pytest.skip("case-insensitive file system cannot hold both a.png and A.png")
    with pytest.raises(PairingError, match="differ only by case"):
        discover_images(sub, EXTS)


def test_pairing_matches_by_case_insensitive_name(tmp_path: Path) -> None:
    ref, cand = tmp_path / "r", tmp_path / "c"
    ref.mkdir(), cand.mkdir()
    draw_shape(16).save(ref / "Front.PNG")
    draw_shape(16).save(cand / "front.png")
    pairs, skipped = pair_images(ref, cand, EXTS)
    assert len(pairs) == 1 and pairs[0].name == "Front.PNG"
    assert skipped == {"reference_without_candidate": [], "candidate_without_reference": []}


def test_missing_pair_is_a_clear_error(dataset: tuple[Path, Path]) -> None:
    ref, cand = dataset
    (cand / "top.png").unlink()
    draw_shape(16).save(cand / "extra.png")
    with pytest.raises(PairingError) as exc:
        pair_images(ref, cand, EXTS, skip_unmatched=False)
    msg = str(exc.value)
    assert "top.png" in msg and "extra.png" in msg
    assert "reference images with no candidate" in msg
    assert "candidate images with no reference" in msg
    assert "skip_unmatched" in msg


def test_skip_unmatched_option(dataset: tuple[Path, Path]) -> None:
    ref, cand = dataset
    (cand / "top.png").unlink()
    pairs, skipped = pair_images(ref, cand, EXTS, skip_unmatched=True)
    assert {p.name for p in pairs} == {"front.png", "side.png", "back.png"}
    assert skipped["reference_without_candidate"] == ["top.png"]


def test_empty_or_missing_folders(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(PairingError, match="No supported images"):
        pair_images(empty, empty, EXTS)
    with pytest.raises(PairingError, match="does not exist"):
        pair_images(tmp_path / "nope", empty, EXTS)


# ---------------------------------------------------------------------------
# Full run
# ---------------------------------------------------------------------------
def test_full_run_writes_all_outputs(dataset: tuple[Path, Path], tmp_path: Path) -> None:
    ref, cand = dataset
    out_root = tmp_path / "outputs"
    result = BenchmarkRunner(_small_config()).run(ref, cand, out_root)

    assert result.run_dir is not None and result.run_dir.parent == out_root
    assert result.run_dir.name.startswith("run_")
    for rel in ("metrics.json", "metrics.csv", "report.png"):
        assert (result.run_dir / rel).is_file(), rel
    for name in ("front", "side", "top", "back"):
        assert (result.run_dir / "preprocessed" / "reference" / f"{name}.png").is_file()
        assert (result.run_dir / "preprocessed" / "candidate" / f"{name}.png").is_file()
        assert (result.run_dir / "comparisons" / f"{name}.png").is_file()

    data = json.loads((result.run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert set(data["pairs"]) == {"front.png", "side.png", "top.png", "back.png"}
    front = data["pairs"]["front.png"]
    for key in ("ssim", "ssim_score", "lpips_distance", "lpips_score", "silhouette_iou",
                "silhouette_score", "edge_score", "pair_score"):
        assert key in front
    assert front["pair_score"] > 99.5  # identical image
    assert data["pairs"]["side.png"]["pair_score"] < front["pair_score"]  # shifted
    assert data["pairs"]["top.png"]["pair_score"] < data["pairs"]["side.png"]["pair_score"]  # reshaped
    assert math.isclose(data["overall_score"], result.overall_score)
    assert math.isclose(
        result.overall_score, sum(p["pair_score"] for p in data["pairs"].values()) / 4
    )
    assert "weights" in data["configuration"]

    with (result.run_dir / "metrics.csv").open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert [r["name"] for r in rows] == ["back.png", "front.png", "side.png", "top.png", "__overall__"]
    assert math.isclose(float(rows[-1]["pair_score"]), result.overall_score)

    # Raw floats are not rounded in JSON.
    assert front["ssim_score"] == 100.0 * min(max(front["ssim"], 0.0), 1.0)
    assert math.isclose(front["lpips_score"], 100.0 * math.exp(-front["lpips_distance"]))


def test_run_without_output(dataset: tuple[Path, Path]) -> None:
    ref, cand = dataset
    result = BenchmarkRunner(_small_config()).run(ref, cand, None)
    assert result.run_dir is None and result.overall_score is not None


def test_only_same_names_are_compared(dataset: tuple[Path, Path]) -> None:
    ref, cand = dataset
    result = BenchmarkRunner(_small_config()).run(ref, cand, None)
    for p in result.pairs:
        assert Path(p.reference_path).name.lower() == Path(p.candidate_path).name.lower()


def test_silhouette_not_available_without_mask_and_weights_renormalised(tmp_path: Path) -> None:
    ref, cand = tmp_path / "r", tmp_path / "c"
    ref.mkdir(), cand.mkdir()
    draw_shape(128, "circle", mode="RGB").save(ref / "a.png")
    draw_shape(128, "circle", mode="RGB", offset=(4, 2)).save(cand / "a.png")
    cfg = _small_config(mask_mode="alpha")  # RGB inputs -> no mask
    result = BenchmarkRunner(cfg).run(ref, cand, None)
    p = result.pairs[0]
    assert p.silhouette_iou is None and p.silhouette_score is None
    assert p.unavailable_metrics == ["silhouette"]
    assert math.isclose(sum(p.effective_weights.values()), 1.0)
    expected = (0.4 * p.lpips_score + 0.3 * p.ssim_score + 0.1 * p.edge_score) / 0.8
    assert math.isclose(p.pair_score, expected)
    d = result.to_dict()["pairs"]["a.png"]
    assert d["silhouette_iou"] is None  # serialised as null, no fake value


def test_multiple_image_modes_in_one_run(tmp_path: Path) -> None:
    ref, cand = tmp_path / "r", tmp_path / "c"
    ref.mkdir(), cand.mkdir()
    for name, mode in (("rgba.png", "RGBA"), ("rgb.png", "RGB"), ("gray.png", "L")):
        draw_shape(96, "square", mode=mode).save(ref / name)
        draw_shape(96, "square", mode=mode).save(cand / name)
    draw_shape(96, "square", mode="RGB").save(ref / "photo.jpg", quality=95)
    draw_shape(96, "square", mode="RGB").save(cand / "photo.jpg", quality=95)
    result = BenchmarkRunner(_small_config()).run(ref, cand, None)
    assert len(result.pairs) == 4
    for p in result.pairs:
        assert p.ok and p.pair_score > 95, p


def test_corrupt_image_is_reported_not_crashing(dataset: tuple[Path, Path]) -> None:
    ref, cand = dataset
    (cand / "front.png").write_bytes(b"not an image")
    result = BenchmarkRunner(_small_config()).run(ref, cand, None)
    bad = next(p for p in result.pairs if p.name == "front.png")
    assert bad.error is not None and "Cannot read image" in bad.error
    assert not bad.ok
    assert len(result.valid_pairs) == 3
    assert result.overall_score == compute_overall_score(result.pairs)


def test_perturbations_lower_pair_score(tmp_path: Path) -> None:
    ref, cand = tmp_path / "r", tmp_path / "c"
    ref.mkdir(), cand.mkdir()
    base = draw_shape(128, "circle")
    variants = {
        "same.png": base,
        "bright.png": adjust_brightness(base, 0.6),
        "color.png": draw_shape(128, "circle", color=(40, 90, 200)),
        "shift.png": draw_shape(128, "circle", offset=(8, 4)),
        "shape.png": draw_shape(128, "triangle"),
    }
    for name, img in variants.items():
        base.save(ref / name)
        img.save(cand / name)
    # alignment="none": a positional shift must be visible as a lower score here.
    result = BenchmarkRunner(_small_config(alignment="none")).run(ref, cand, None)
    scores = {p.name: p for p in result.pairs}
    assert scores["same.png"].pair_score > 99.5
    for name in ("bright.png", "color.png", "shift.png", "shape.png"):
        assert scores[name].pair_score < scores["same.png"].pair_score - 1, name
    assert scores["shape.png"].silhouette_score < 90 and scores["shape.png"].edge_score < 70
    assert scores["shift.png"].silhouette_score < 95

    # With the default alignment the shifted copy is no longer penalised on
    # geometry (silhouette / edge), while the other perturbations still are.
    aligned = {p.name: p for p in BenchmarkRunner(_small_config()).run(ref, cand, None).pairs}
    assert aligned["shift.png"].silhouette_score > 99 and aligned["shift.png"].edge_score > 95
    assert aligned["shift.png"].alignment["applied"] and aligned["shift.png"].alignment["shift_px"] != [0, 0]
    assert aligned["same.png"].alignment["shift_px"] == [0, 0]
    for name in ("bright.png", "color.png", "shape.png"):
        assert aligned[name].pair_score < aligned["same.png"].pair_score - 1, name


def test_foreground_bbox_makes_shift_irrelevant(tmp_path: Path) -> None:
    ref, cand = tmp_path / "r", tmp_path / "c"
    ref.mkdir(), cand.mkdir()
    draw_shape(160, "square", scale=0.4).save(ref / "a.png")
    draw_shape(160, "square", scale=0.4, offset=(20, -15)).save(cand / "a.png")
    none = BenchmarkRunner(_small_config(crop_mode="none", alignment="none")).run(ref, cand, None).pairs[0]
    bbox = BenchmarkRunner(_small_config(crop_mode="foreground_bbox", alignment="none")).run(ref, cand, None).pairs[0]
    assert bbox.silhouette_iou > 0.97 > none.silhouette_iou
    assert bbox.pair_score > none.pair_score


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def test_default_yaml_loads_and_weights_sum_to_one() -> None:
    cfg = load_config(Path(__file__).resolve().parent.parent / "configs" / "default.yaml")
    assert math.isclose(sum(cfg.weights.values()), 1.0)
    assert cfg.weights == {"lpips": 0.40, "ssim": 0.30, "silhouette": 0.20, "edge": 0.10}
    assert cfg.preprocessing.canvas_size == 512


def test_invalid_weights_in_yaml(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump({"weights": {"lpips": 0.5, "ssim": 0.5, "silhouette": 0.2, "edge": 0.1}}))
    with pytest.raises(ConfigError, match="sum to 1"):
        load_config(p)
    p.write_text(yaml.safe_dump({"weights": {"lpips": -0.1, "ssim": 0.9, "silhouette": 0.1, "edge": 0.1}}))
    with pytest.raises(ConfigError, match=">= 0"):
        load_config(p)


def test_unknown_config_key_and_bad_enum(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="unknown key"):
        config_from_dict({"preprocessing": {"canvas_sizee": 512}})
    with pytest.raises(ConfigError, match="crop_mode"):
        config_from_dict({"preprocessing": {"crop_mode": "magic"}})
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "missing.yaml")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def test_cli_compare(dataset: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    ref, cand = dataset
    out = tmp_path / "out"
    code = main(["compare", "--reference", str(ref), "--candidate", str(cand), "--output", str(out),
                 "--canvas-size", "128", "--log-level", "WARNING"])
    assert code == 0
    captured = capsys.readouterr().out
    assert "overall_score" in captured and "front.png" in captured
    runs = list(out.glob("run_*"))
    assert len(runs) == 1 and (runs[0] / "report.png").is_file()


def test_cli_compare_missing_pair_exits_nonzero(dataset: tuple[Path, Path], tmp_path: Path, capsys) -> None:
    ref, cand = dataset
    (cand / "top.png").unlink()
    code = main(["compare", "--reference", str(ref), "--candidate", str(cand), "--output", str(tmp_path / "o"),
                 "--canvas-size", "128", "--log-level", "WARNING"])
    assert code == 3
    assert "top.png" in capsys.readouterr().err
    code = main(["compare", "--reference", str(ref), "--candidate", str(cand), "--output", str(tmp_path / "o"),
                 "--canvas-size", "128", "--log-level", "WARNING", "--skip-unmatched"])
    assert code == 0


def test_cli_compare_pair(dataset: tuple[Path, Path], tmp_path: Path, capsys) -> None:
    ref, cand = dataset
    out = tmp_path / "out"
    code = main(["compare-pair", "--reference", str(ref / "front.png"), "--candidate", str(cand / "front.png"),
                 "--output", str(out), "--canvas-size", "128", "--log-level", "WARNING"])
    assert code == 0
    printed = capsys.readouterr().out
    payload = json.loads(printed[printed.index("{"): printed.rindex("}") + 1])
    assert payload["pair"]["pair_score"] > 99.5
    run = next(out.glob("run_*"))
    assert (run / "metrics.json").is_file() and (run / "comparisons" / "front.png").is_file()


def test_cli_bad_config_exit_code(dataset: tuple[Path, Path], tmp_path: Path, capsys) -> None:
    ref, cand = dataset
    bad = tmp_path / "bad.yaml"
    bad.write_text("weights: {lpips: 1, ssim: 1, silhouette: 0, edge: 0}\n")
    code = main(["compare", "--reference", str(ref), "--candidate", str(cand), "--config", str(bad), "--no-save"])
    assert code == 2
    assert "Configuration error" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Three-view drawings (front / side / top of one object)
# ---------------------------------------------------------------------------
def test_three_view_set_each_view_compared_only_with_itself(tmp_path: Path) -> None:
    from src.synthetic import write_three_view_set

    ref, cand = tmp_path / "r", tmp_path / "c"
    names = write_three_view_set(ref, cand, size=256)
    assert names == ["front.png", "side.png", "top.png"]
    result = BenchmarkRunner(_small_config()).run(ref, cand, None)
    assert {p.name for p in result.pairs} == set(names)
    for p in result.pairs:
        assert p.ok
        assert Path(p.reference_path).name == Path(p.candidate_path).name
        # Candidate object has different dimensions -> every view is < 100 but still similar.
        assert 50 < p.pair_score < 99, p
        assert p.silhouette_iou is not None and p.silhouette_iou < 0.99
    assert math.isclose(result.overall_score, sum(p.pair_score for p in result.pairs) / 3)


def test_three_view_identical_object_scores_100(tmp_path: Path) -> None:
    from src.synthetic import draw_three_views

    ref, cand = tmp_path / "r", tmp_path / "c"
    ref.mkdir(), cand.mkdir()
    for name, img in draw_three_views(256).items():
        img.save(ref / name)
        img.save(cand / name)
    result = BenchmarkRunner(_small_config()).run(ref, cand, None)
    assert all(p.pair_score > 99.5 for p in result.pairs)
