from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

from deerui.domain.models import DetectionResult

_logger = logging.getLogger(__name__)


def build_detection_label(detection: DetectionResult) -> str:
    name = (
        detection.display_zh
        or detection.display_name
        or detection.display_en
        or detection.action_key
        or ""
    )
    if detection.confidence is not None and name:
        return f"{name} {detection.confidence:.2f}"
    return name


def draw_detection_overlay_bgr(
    frame_bgr,
    detection: DetectionResult,
    *,
    color_bgr: tuple[int, int, int],
    cv2_module=None,
    text_size: int = 18,
    thickness: int = 2,
):
    if frame_bgr is None:
        return frame_bgr
    try:
        overlay = frame_bgr.copy()
    except Exception:  # noqa: BLE001
        return frame_bgr
    if detection.bbox_xyxy is None:
        return overlay

    cv2 = cv2_module
    if cv2 is None:
        try:
            import cv2 as cv2_import  # type: ignore[import-not-found]
        except ImportError:
            cv2_import = None
        cv2 = cv2_import

    x1, y1, x2, y2 = (int(value) for value in detection.bbox_xyxy)
    if cv2 is not None:
        try:
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color_bgr, thickness)
        except Exception:  # noqa: BLE001
            _logger.debug("failed to draw detection rectangle", exc_info=True)

    label = build_detection_label(detection)
    if not label:
        return overlay

    origin = (x1, max(2, y1 - text_size - 4))
    if _needs_unicode_text(label):
        return _draw_unicode_text_bgr(overlay, label, origin, color_bgr, text_size)

    if cv2 is None:
        return _draw_unicode_text_bgr(overlay, label, origin, color_bgr, text_size)
    try:
        cv2.putText(
            overlay,
            label,
            (origin[0], max(origin[1] + text_size, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color_bgr,
            thickness,
            getattr(cv2, "LINE_AA", 16),
        )
    except Exception:  # noqa: BLE001
        return _draw_unicode_text_bgr(overlay, label, origin, color_bgr, text_size)
    return overlay


def _needs_unicode_text(text: str) -> bool:
    return any(ord(char) > 127 for char in text)


def _draw_unicode_text_bgr(
    frame_bgr,
    text: str,
    origin: tuple[int, int],
    color_bgr: tuple[int, int, int],
    font_size: int,
):
    try:
        import numpy as np
        from PIL import Image, ImageDraw
    except ImportError:
        return frame_bgr

    try:
        rgb = frame_bgr[:, :, ::-1].copy()
        image = Image.fromarray(rgb)
        draw = ImageDraw.Draw(image)
        color_rgb = (int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0]))
        draw.text(
            (max(0, int(origin[0])), max(0, int(origin[1]))),
            text,
            font=_load_font(font_size),
            fill=color_rgb,
        )
        result_rgb = np.asarray(image)
        return result_rgb[:, :, ::-1].copy()
    except Exception:  # noqa: BLE001
        _logger.debug("failed to draw unicode overlay text", exc_info=True)
        return frame_bgr


@lru_cache(maxsize=16)
def _load_font(font_size: int):
    from PIL import ImageFont

    for path in _font_candidates():
        try:
            if path.exists():
                return ImageFont.truetype(str(path), font_size)
        except Exception:  # noqa: BLE001
            continue
    return ImageFont.load_default()


def _font_candidates() -> tuple[Path, ...]:
    windows_fonts = Path("C:/Windows/Fonts")
    return (
        windows_fonts / "msyh.ttc",
        windows_fonts / "simhei.ttf",
        windows_fonts / "simsun.ttc",
        windows_fonts / "STXIHEI.TTF",
        windows_fonts / "HYZhongHeiTi-197.ttf",
    )
