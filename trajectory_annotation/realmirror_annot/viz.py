"""Overlay drawing for annotation review images."""
from __future__ import annotations

import cv2
import numpy as np

ROLE_COLORS = {  # BGR
    "target": (0, 255, 0),
    "destination": (0, 200, 255),
    "destination_basket": (0, 160, 255),
    "source_support": (255, 200, 0),
    "distractor": (200, 200, 200),
}
HAND_COLORS = {"left": (255, 80, 255), "right": (255, 255, 0)}
BOX_EDGES = [(0, 1), (0, 2), (1, 3), (2, 3), (4, 5), (4, 6), (5, 7), (6, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
MIN_LABEL_PX = 1  # draw every object that has at least this many visible pixels
MIN_VISIBLE_FRACTION = 0.3  # below this visible share, draw only the visible part


def _id_palette(n=70000, seed=3):
    rng = np.random.default_rng(seed)
    pal = rng.integers(40, 255, size=(n, 3), dtype=np.uint8)
    pal[0] = 0
    return pal


_PALETTE = _id_palette()


FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.45


def _put_label(img, text, x, y, color, filled=True):
    """Label on a filled colour tag (black or white text for contrast); (x, y) = tag bottom-left."""
    (tw, th), base = cv2.getTextSize(text, FONT, FONT_SCALE, 1)
    x = int(min(max(0, x), img.shape[1] - tw - 4))
    y = int(y)
    if y - th - base - 2 < 0:
        y = th + base + 2
    if filled:
        cv2.rectangle(img, (x, y - th - base - 2), (x + tw + 4, y), color, -1)
        lum = 0.114 * color[0] + 0.587 * color[1] + 0.299 * color[2]
        txt = (0, 0, 0) if lum > 140 else (255, 255, 255)
    else:
        txt = color
    cv2.putText(img, text, (x + 2, y - base), FONT, FONT_SCALE, txt, 1, cv2.LINE_AA)


def _draw_wire(img, corners, color, s):
    if corners is None:
        return
    h, w = img.shape[:2]
    pts = np.array(corners, dtype=float) * s
    for a, b in BOX_EDGES:
        pa, pb = pts[a], pts[b]
        if (pa < 0).all() or (pb < 0).all() or abs(pa).max() > 10 * max(w, h) or abs(pb).max() > 10 * max(w, h):
            continue
        cv2.line(img, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), color, 1, cv2.LINE_AA)


def _draw_entity(img, view: dict, color, label: str, s: int, wire=True, thick=2, background=False):
    # Full projected extent (occlusion ignored, clipped to the image) when at least
    # MIN_VISIBLE_FRACTION of the projected area is visible; otherwise the box of the
    # visible pixels only (a chair peeking over a counter should not get a huge box).
    if not view.get("visible", view.get("in_frustum", False)):
        return
    occl = view.get("occluded_fraction_approx")
    mostly_hidden = occl is not None and occl > 1.0 - MIN_VISIBLE_FRACTION
    if background or mostly_hidden:
        box = view.get("bbox_visible_xyxy_px") or view.get("bbox_xyxy_px")
    else:
        box = view.get("bbox_xyxy_px") or view.get("bbox_visible_xyxy_px")
    if box is None:
        return
    if wire:
        _draw_wire(img, view.get("corners_px"), color, s)
    x0, y0, x1, y1 = [int(round(v * s)) for v in box]
    cv2.rectangle(img, (x0, y0), (x1, y1), color, thick)
    _put_label(img, label, x0, y0 - 1, color)


def draw_step(images: dict, rec: dict, meta: dict, scale: int = 3, id_maps: dict | None = None, mask_overlay: bool = False, wire: bool = False, alpha: float = 0.3) -> np.ndarray:
    """Review image: 2D boxes only by default (mask tint / 3D wireframes are opt-in)."""
    skip_ids = {0} | {so["mask_id"] for so in meta.get("scene_objects", []) if so.get("background")} | {meta.get("mask_ids", {}).get("robot_body", 60003)}
    tiles = []
    for cam_name, img in images.items():
        canvas = np.ascontiguousarray(img.copy())
        if mask_overlay and id_maps is not None and id_maps.get(cam_name) is not None:
            ids = id_maps[cam_name]
            col = _PALETTE[np.clip(ids, 0, len(_PALETTE) - 1)]
            m = ~np.isin(ids, list(skip_ids))
            canvas[m] = (canvas[m] * (1 - alpha) + col[m] * alpha).astype(np.uint8)
        if scale != 1:
            canvas = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        # static scene objects: thin boxes
        for so in rec.get("scene_objects", []):
            v = so["views"].get(cam_name)
            if not v or v.get("visible_px", 0) < MIN_LABEL_PX:
                continue
            color = (150, 150, 150) if so.get("background") else tuple(int(c) for c in _PALETTE[so["mask_id"]])
            _draw_entity(canvas, v, color, so["name"], scale, wire=False, thick=1, background=bool(so.get("background")))
        for obj in rec["objects"]:
            if not obj.get("active"):
                continue
            color = ROLE_COLORS.get(obj["role"], (200, 200, 200))
            _draw_entity(canvas, obj["views"][cam_name], color, obj["name"] + ("*" if obj["role"] == "target" else ""), scale, wire=wire)
        for side, g in rec["grippers"].items():
            _draw_entity(canvas, g["views"][cam_name], HAND_COLORS[side], f"{side}_hand {'closed' if g['closed'] else 'open'}", scale, wire=wire)
        _put_label(canvas, f"{cam_name}  step {rec['step']}  success={int(rec['success'])}", 4, 20, (40, 40, 40))
        tiles.append(canvas)
    out = np.concatenate(tiles, axis=1)
    bar = np.zeros((26, out.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, f"{meta['task_name']} | {rec.get('instruction') or ''}", (6, 18), FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return np.concatenate([bar, out], axis=0)


def write_video(path, frames, fps=10):
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        vw.write(f)
    vw.release()
