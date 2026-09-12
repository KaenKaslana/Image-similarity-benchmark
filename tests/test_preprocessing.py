"""Tests for image loading and preprocessing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.config import PreprocessingConfig
from src.preprocessing import (
    ImageLoadError,
    build_foreground_mask,
    compute_crop_box,
    load_image,
    preprocess_image,
    preprocess_pil,
    save_preprocessed,
    split_rgb_alpha,
)
from src.synthetic import draw_shape


def _cfg(**overrides) -> PreprocessingConfig:
    cfg = PreprocessingConfig(canvas_size=128, **overrides)
    cfg.validate()
    return cfg


# ---------------------------------------------------------------------------
# Loading / modes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["RGB", "RGBA", "L"])
def test_supported_modes_produce_rgb_canvas(tmp_path: Path, mode: str) -> None:
    path = tmp_path / f"img_{mode}.png"
    draw_shape(96, "circle", mode=mode).save(path)
    out = preprocess_image(path, _cfg())
    assert out.rgb.shape == (128, 128, 3)
    assert out.rgb.dtype == np.uint8
    assert out.meta["original_mode"] == mode
    assert out.meta["has_alpha"] == (mode == "RGBA")


def test_jpeg_is_supported(tmp_path: Path) -> None:
    path = tmp_path / "img.jpg"
    draw_shape(96, "square", mode="RGB").save(path, quality=90)
    out = preprocess_image(path, _cfg())
    assert out.rgb.shape == (128, 128, 3)


def test_grayscale_and_16bit_modes() -> None:
    gray = Image.new("L", (10, 10), 128)
    rgb, alpha = split_rgb_alpha(gray)
    assert rgb.shape == (10, 10, 3) and alpha is None and rgb[0, 0].tolist() == [128, 128, 128]

    i16 = Image.fromarray(np.linspace(0, 65535, 100, dtype=np.uint16).reshape(10, 10), mode="I;16")
    rgb16, alpha16 = split_rgb_alpha(i16)
    assert rgb16.shape == (10, 10, 3) and alpha16 is None
    assert rgb16.min() == 0 and rgb16.max() == 255


def test_palette_with_transparency_has_alpha() -> None:
    rgba = draw_shape(32, "circle", mode="RGBA", seed=None)
    pal = rgba.convert("P")
    rgb, alpha = split_rgb_alpha(rgba)
    assert alpha is not None and alpha.max() == 255 and alpha.min() == 0
    assert pal.mode == "P"


def test_corrupt_file_raises_clear_error(tmp_path: Path) -> None:
    bad = tmp_path / "broken.png"
    bad.write_bytes(b"\x89PNG\r\n\x1a\n" + b"garbage" * 10)
    with pytest.raises(ImageLoadError, match="Cannot read image"):
        load_image(bad)
    with pytest.raises(ImageLoadError, match="not found"):
        load_image(tmp_path / "missing.png")


def test_truncated_file_raises(tmp_path: Path) -> None:
    good = tmp_path / "good.png"
    draw_shape(64, "circle").save(good)
    data = good.read_bytes()
    truncated = tmp_path / "trunc.png"
    truncated.write_bytes(data[: len(data) // 2])
    with pytest.raises(ImageLoadError):
        load_image(truncated)


def test_exif_orientation_is_applied(tmp_path: Path) -> None:
    # 40x20 image, top-left quadrant red; orientation tag 6 = rotate 270 CW -> becomes 20x40.
    img = Image.new("RGB", (40, 20), (255, 255, 255))
    for x in range(20):
        for y in range(10):
            img.putpixel((x, y), (255, 0, 0))
    exif = img.getexif()
    exif[0x0112] = 6
    path = tmp_path / "exif.jpg"
    img.save(path, exif=exif.tobytes(), quality=100)
    loaded = load_image(path)
    assert loaded.size == (20, 40)


# ---------------------------------------------------------------------------
# Alpha handling and masks
# ---------------------------------------------------------------------------
def test_transparent_pixels_composited_on_white() -> None:
    img = draw_shape(64, "circle", mode="RGBA", seed=None)
    out = preprocess_pil(img, _cfg())
    corner = out.rgb[2, 2]
    assert corner.tolist() == [255, 255, 255]
    assert out.mask is not None and out.mask_source == "alpha" and out.mask_reliable
    assert not out.mask[2, 2] and out.mask[64, 64]


def test_custom_background_color() -> None:
    img = draw_shape(64, "circle", mode="RGBA", seed=None)
    out = preprocess_pil(img, _cfg(background_color=[0, 0, 0]))
    assert out.rgb[2, 2].tolist() == [0, 0, 0]


def test_mask_from_background_threshold_when_no_alpha() -> None:
    img = draw_shape(64, "circle", mode="RGB", seed=None)
    out = preprocess_pil(img, _cfg(mask_mode="auto"))
    assert out.mask_source == "background_threshold" and out.mask_reliable
    assert out.mask is not None and 0.1 < out.mask.mean() < 0.5


def test_mask_mode_alpha_gives_no_mask_for_rgb() -> None:
    img = draw_shape(64, "circle", mode="RGB", seed=None)
    out = preprocess_pil(img, _cfg(mask_mode="alpha"))
    assert out.mask is None and not out.has_mask and out.mask_source == "none"


def test_thresholded_mask_rejected_for_photo_like_image() -> None:
    rng = np.random.default_rng(1)
    noisy = Image.fromarray(rng.integers(0, 200, size=(64, 64, 3), dtype=np.uint8), mode="RGB")
    out = preprocess_pil(noisy, _cfg(mask_mode="auto"))
    assert out.mask_source == "background_threshold" and not out.mask_reliable and not out.has_mask


def test_build_foreground_mask_alpha_threshold() -> None:
    rgb = np.zeros((4, 4, 3), np.uint8)
    alpha = np.array([[0, 10, 20, 255]] * 4, np.uint8)
    mask, source, reliable = build_foreground_mask(rgb, alpha, _cfg(alpha_threshold=16))
    assert source == "alpha" and reliable
    assert mask[0].tolist() == [False, False, True, True]


# ---------------------------------------------------------------------------
# Geometry: aspect ratio, canvas, crops
# ---------------------------------------------------------------------------
def test_aspect_ratio_preserved_and_centered() -> None:
    wide = Image.new("RGB", (200, 50), (0, 0, 255))
    out = preprocess_pil(wide, _cfg())
    assert out.meta["resized_size"] == [128, 32]
    assert out.meta["offset"] == [0, 48]
    assert out.rgb[64, 64].tolist() == [0, 0, 255]  # centre is blue
    assert out.rgb[5, 64].tolist() == [255, 255, 255]  # top padding is white
    assert out.rgb[122, 64].tolist() == [255, 255, 255]  # bottom padding is white


def test_same_rules_for_reference_and_candidate() -> None:
    cfg = _cfg()
    a = preprocess_pil(Image.new("RGB", (300, 100), (10, 10, 10)), cfg)
    b = preprocess_pil(Image.new("RGB", (300, 100), (20, 20, 20)), cfg)
    assert a.meta["scale"] == b.meta["scale"]
    assert a.meta["offset"] == b.meta["offset"]
    assert a.rgb.shape == b.rgb.shape


def test_center_crop_box() -> None:
    box = compute_crop_box((100, 300), None, False, _cfg(crop_mode="center_crop"))
    assert box == (100, 0, 200, 100)
    assert compute_crop_box((100, 300), None, False, _cfg(crop_mode="none")) is None


def test_center_crop_end_to_end() -> None:
    wide = Image.new("RGB", (200, 50), (0, 200, 0))
    out = preprocess_pil(wide, _cfg(crop_mode="center_crop"))
    assert out.meta["crop_box"] == [75, 0, 125, 50]
    assert out.meta["resized_size"] == [128, 128]
    assert (out.rgb == np.array([0, 200, 0], np.uint8)).all()


def test_foreground_bbox_crop_centres_object_with_padding() -> None:
    # Small object in the corner of a large transparent canvas.
    img = draw_shape(256, "square", offset=(-80, -80), scale=0.2, mode="RGBA", seed=None)
    cfg = _cfg(crop_mode="foreground_bbox", foreground_padding=0.25)
    out = preprocess_pil(img, cfg)
    assert out.meta["crop_box"] is not None
    assert out.mask is not None
    ys, xs = np.where(out.mask)
    cy, cx = ys.mean(), xs.mean()
    assert abs(cy - 63.5) < 2 and abs(cx - 63.5) < 2  # centred
    side = xs.max() - xs.min() + 1
    # object side = canvas / (1 + 2*padding) = 128 / 1.5 ~ 85
    assert abs(side - 128 / 1.5) < 4


def test_foreground_bbox_falls_back_without_mask(caplog: pytest.LogCaptureFixture) -> None:
    img = draw_shape(64, "circle", mode="RGB", seed=None)
    cfg = _cfg(crop_mode="foreground_bbox", mask_mode="alpha")
    with caplog.at_level("WARNING"):
        out = preprocess_pil(img, cfg)
    assert out.meta["crop_box"] is None
    assert "falling back" in caplog.text


def test_save_preprocessed_writes_rgb_and_mask(tmp_path: Path) -> None:
    out = preprocess_pil(draw_shape(64, "circle", mode="RGBA"), _cfg())
    p = save_preprocessed(out, tmp_path, "front.png")
    assert p.exists() and (tmp_path / "front_mask.png").exists()
    reloaded = np.asarray(Image.open(p))
    assert reloaded.shape == (128, 128, 3)
