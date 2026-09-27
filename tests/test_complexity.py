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
