"""One-time extraction of object bounds from the RealMirror scene USDs.

For every tracked prim we store the axis-aligned bound in the prim's own
(untransformed) frame together with the prim's world scale, so that at run
time a world-space 3D box is ``p(t) + R(t) @ (scale * corner)``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _usd_matrix_to_rst(M):
    rows = np.array([[M[r][c] for c in range(3)] for r in range(3)], dtype=float)
    t = np.array([M[3][c] for c in range(3)], dtype=float)
    scale = np.linalg.norm(rows, axis=1)
    R = (rows / scale[:, None]).T  # column-vector rotation
    return R, scale, t


def build_object_table(scene_usd: str | Path, prim_paths: list[str], scene_root: str = "/scene") -> dict:
    from pxr import Usd, UsdGeom
    from .camera import matrix_to_quat_wxyz

    stage = Usd.Stage.Open(str(scene_usd))
    default_prim = stage.GetDefaultPrim()
    if not default_prim:
        raise RuntimeError(f"{scene_usd} has no default prim")
    bc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy"], useExtentsHint=True)
    xc = UsdGeom.XformCache()
    table = {}
    for pp in prim_paths:
        if not pp.startswith(scene_root):
            raise ValueError(f"prim path {pp} does not start with {scene_root}")
        path = default_prim.GetPath().pathString + pp[len(scene_root):]
        prim = stage.GetPrimAtPath(path)
        if not prim:
            table[pp] = {"error": f"prim not found: {path}"}
            continue
        R, scale, t = _usd_matrix_to_rst(xc.GetLocalToWorldTransform(prim))
        rng = bc.ComputeUntransformedBound(prim).ComputeAlignedRange()
        lo, hi = np.array(rng.GetMin(), dtype=float), np.array(rng.GetMax(), dtype=float)
        if not np.isfinite(lo).all() or not np.isfinite(hi).all():
            table[pp] = {"error": f"empty bound for {path}"}
            continue
        table[pp] = {
            "usd_path": path,
            "prim_type": prim.GetTypeName(),
            "local_aabb_min": lo.tolist(),
            "local_aabb_max": hi.tolist(),
            "world_scale": scale.tolist(),
            "size_m": ((hi - lo) * scale).tolist(),
            "usd_position": t.tolist(),
            "usd_quat_wxyz": matrix_to_quat_wxyz(R).tolist(),
        }
    return table


def load_object_table(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def object_local_corners(entry: dict) -> np.ndarray:
    """8 corners of the object's box in its rigid-body frame (scale applied)."""
    lo = np.asarray(entry["local_aabb_min"]) * np.asarray(entry["world_scale"])
    hi = np.asarray(entry["local_aabb_max"]) * np.asarray(entry["world_scale"])
    return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])


def build_object_points(scene_usd: str | Path, prim_paths: list[str], scene_root: str = "/scene", max_points: int = 3000, seed: int = 0) -> dict:
    """Sub-sampled mesh vertices of every tracked prim, expressed in the prim's
    rigid-body frame with the world scale already applied (meters)."""
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(scene_usd))
    default_prim = stage.GetDefaultPrim()
    xc = UsdGeom.XformCache()
    rng = np.random.default_rng(seed)
    out = {}
    for pp in prim_paths:
        path = default_prim.GetPath().pathString + pp[len(scene_root):]
        prim = stage.GetPrimAtPath(path)
        if not prim:
            continue
        _, scale, _ = _usd_matrix_to_rst(xc.GetLocalToWorldTransform(prim))
        pts = []
        for mp in Usd.PrimRange(prim):
            if not mp.IsA(UsdGeom.Mesh):
                continue
            if UsdGeom.Imageable(mp).ComputePurpose() not in ("default", "render"):
                continue
            p = UsdGeom.Mesh(mp).GetPointsAttr().Get()
            if not p:
                continue
            p = np.array(p, dtype=float)
            if mp != prim:
                rel, _ = xc.ComputeRelativeTransform(mp, prim)
                rows = np.array([[rel[r][c] for c in range(3)] for r in range(3)])
                t = np.array([rel[3][c] for c in range(3)])
                p = p @ rows + t  # USD row-vector convention
            pts.append(p)
        if not pts:
            continue
        allp = np.concatenate(pts)
        if len(allp) > max_points:
            allp = allp[rng.choice(len(allp), max_points, replace=False)]
        out[pp] = (allp * scale).astype(np.float32)
    return out
