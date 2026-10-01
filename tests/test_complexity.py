"""Face-count (mesh complexity) term of compare-models."""
import pytest

from src.complexity import MeshStats, blend_with_shape_score, complexity_score, mesh_complexity
from src.config import ConfigError, MeshComplexityConfig, config_from_dict


@pytest.mark.parametrize(
    "ratio, expected",
    [
        (1.0, 100.0),  # identical face count
        (2.0, 100.0),  # within the free band (factor 2)
        (0.5, 100.0),  # symmetric
        (8.0, 50.0),  # log2 = 3 -> halfway between free (1) and zero (5)
        (1 / 8, 50.0),
        (32.0, 0.0),  # log2 = 5 -> zero
        (1000.0, 0.0),
        (0.0, 0.0),  # degenerate candidate
    ],
)
def test_complexity_score_is_symmetric_on_log2_scale(ratio: float, expected: float) -> None:
    assert complexity_score(ratio, free_log2=1.0, zero_log2=5.0) == pytest.approx(expected)


def test_mesh_complexity_dict_and_blend() -> None:
    cfg = MeshComplexityConfig(weight=0.2)
    cfg.validate()
    info = mesh_complexity(MeshStats(faces=1000, vertices=600), MeshStats(faces=8000, vertices=4200), cfg)
    assert info["reference"] == {"faces": 1000, "vertices": 600}
    assert info["candidate"]["faces"] == 8000
    assert info["face_ratio"] == pytest.approx(8.0)
    assert info["log2_face_ratio"] == pytest.approx(3.0)
    assert info["score"] == pytest.approx(50.0)
    # overall = shape * (1 - w * (1 - mesh/100)) = 90 * (1 - 0.2 * 0.5)
    assert blend_with_shape_score(90.0, info) == pytest.approx(81.0)
    # matching face count: the shape score is untouched whatever the weight
    same = mesh_complexity(MeshStats(1000, 600), MeshStats(1500, 900), cfg)
    assert blend_with_shape_score(90.0, same) == pytest.approx(90.0)
    # weight 0 -> the shape score is returned untouched; None stays None
    info0 = mesh_complexity(MeshStats(1000, 600), MeshStats(8000, 4200), MeshComplexityConfig(weight=0.0))
    assert blend_with_shape_score(90.0, info0) == 90.0
    assert blend_with_shape_score(None, info) is None
    assert blend_with_shape_score(90.0, None) == 90.0


def test_mesh_complexity_config_validation() -> None:
    cfg = config_from_dict({"mesh_complexity": {"weight": 0.3, "free_log2": 0.5, "zero_log2": 4}})
    assert cfg.mesh_complexity.weight == 0.3 and cfg.mesh_complexity.zero_log2 == 4.0
    assert config_from_dict({}).mesh_complexity.weight == 0.0  # default: report only
    with pytest.raises(ConfigError):
        config_from_dict({"mesh_complexity": {"weight": 1.5}})
    with pytest.raises(ConfigError):
        config_from_dict({"mesh_complexity": {"free_log2": 3, "zero_log2": 2}})
    with pytest.raises(ConfigError):
        config_from_dict({"mesh_complexity": {"weight": "lots"}})


@pytest.mark.parametrize(
    "ratio, score, bonus",
    [
        (1.0, 100.0, 0.0),  # same face count: nothing happens
        (2.0, 100.0, 0.0),  # up to 2x more: free
        (8.0, 50.0, 0.0),  # 8x more: halfway to zero
        (32.0, 0.0, 0.0),
        (0.5, 100.0, 0.5),  # half the faces: no penalty, half the bonus
        (0.25, 100.0, 1.0),  # a quarter: full bonus
        (1 / 64, 100.0, 1.0),  # fewer still: capped
    ],
)
def test_fewer_is_better_mode(ratio: float, score: float, bonus: float) -> None:
    from src.complexity import efficiency_bonus

    assert complexity_score(ratio, 1.0, 5.0, "fewer_is_better") == pytest.approx(score)
    assert efficiency_bonus(ratio, 2.0) == pytest.approx(bonus)


def test_fewer_is_better_blend_is_capped() -> None:
    cfg = MeshComplexityConfig(mode="fewer_is_better", weight=0.15, bonus_weight=0.05, bonus_log2=2.0)
    cfg.validate()
    lean = mesh_complexity(MeshStats(2304, 1200), MeshStats(576, 300), cfg)  # a quarter of the faces
    assert lean["mode"] == "fewer_is_better" and lean["score"] == 100.0 and lean["bonus"] == pytest.approx(1.0)
    assert blend_with_shape_score(90.0, lean) == pytest.approx(94.5)  # +5 %
    assert blend_with_shape_score(98.0, lean) == 100.0  # capped
    assert blend_with_shape_score(3.0, lean) == pytest.approx(3.15)  # a poor shape stays poor
    fat = mesh_complexity(MeshStats(2304, 1200), MeshStats(7164, 3600), cfg)  # 3.1x more faces
    assert fat["bonus"] == 0.0 and 80 < fat["score"] < 90
    assert blend_with_shape_score(84.8, fat) == pytest.approx(84.8 * (1 - 0.15 * (1 - fat["score"] / 100)))
    # symmetric mode never gives a bonus, even when configured
    sym = MeshComplexityConfig(mode="symmetric", weight=0.15, bonus_weight=0.05)
    sym.validate()
    assert mesh_complexity(MeshStats(2304, 1200), MeshStats(576, 300), sym)["bonus_weight"] == 0.0
    with pytest.raises(ConfigError):
        config_from_dict({"mesh_complexity": {"mode": "bigger_is_better"}})
