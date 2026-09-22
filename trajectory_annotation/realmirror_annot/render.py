"""Instance-ID rendering of the whole scene (static objects + tracked objects +
robot) with pyrender, giving exact per-pixel visibility for every object.

Requires an OpenGL context: run with PYOPENGL_PLATFORM=egl on a GPU node.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .camera import PinholeCamera
from .kinematics import URDFKinematics
from .scene import SceneObject, load_scene_cache

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

ROBOT_ID_BASE = 60000  # ids >= this are robot links; hands get fixed ids below
LEFT_HAND_ID = 60001
RIGHT_HAND_ID = 60002
ROBOT_BODY_ID = 60003


_LUT_R = np.random.default_rng(12345).permutation(256).astype(np.int32)
_LUT_G = np.random.default_rng(54321).permutation(256).astype(np.int32)


def _checksum(r, g):
    # random lookup tables: a colour blended from two valid colours at an edge
    # passes this check with probability ~1/256 only
    return _LUT_R[np.asarray(r)] ^ _LUT_G[np.asarray(g)]


def id_to_color(i: int):
    r, g = i & 255, (i >> 8) & 255
    return [r, g, int(_checksum(r, g))]


def color_to_id(img: np.ndarray):
    """Returns (ids int32 HxW, valid bool HxW). ids must be < 65536."""
    img = img.astype(np.int32)
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    ids = r + (g << 8)
    valid = b == _checksum(r, g)
    return ids, valid


def world_axes_to_gl(position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Isaac world-axes camera (x fwd, y left, z up) -> OpenGL camera (looks -z, +y up)."""
    R = rotation
    Rgl = np.stack([-R[:, 1], R[:, 2], -R[:, 0]], axis=1)
    T = np.eye(4)
    T[:3, :3] = Rgl
    T[:3, 3] = position
    return T


class SceneRenderer:
    def __init__(self, task: str, kin: URDFKinematics, mesh_root: Path, hand_links: dict, width: int = 256, height: int = 256,
                 cache_dir: str | Path = "data/scene_cache", root: str | Path = "."):
        import pyrender
        import trimesh

        self.pyrender = pyrender
        self.width, self.height = width, height
        self.objects: list[SceneObject] = load_scene_cache(task, cache_dir, root)
        self.scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[1.0, 1.0, 1.0])
        self.seg_map = {}
        self.obj_nodes = {}
        for o in self.objects:
            tm = trimesh.Trimesh(vertices=o.V, faces=o.F, process=False)
            mesh = pyrender.Mesh.from_trimesh(tm, smooth=False)
            node = self.scene.add(mesh, pose=np.eye(4), name=f"obj{o.obj_id}")
            self.obj_nodes[o.obj_id] = node
            self.seg_map[node] = id_to_color(o.obj_id)
        # robot links
        self.link_nodes = {}
        hand_of = {}
        for side, links in hand_links.items():
            for l in links:
                hand_of[l] = side
        cache = {}
        for link in kin.links:
            vis = kin.visuals.get(link, [])
            if not vis:
                continue
            parts = []
            for v in vis:
                key = (v.filename, tuple(v.scale))
                if key not in cache:
                    cache[key] = trimesh.load(kin.resolve_mesh_path(v.filename, mesh_root), force="mesh")
                m = cache[key].copy()
                m.apply_scale(v.scale)
                m.apply_transform(v.origin)
                parts.append(m)
            tm = trimesh.util.concatenate(parts) if len(parts) > 1 else parts[0]
            node = self.scene.add(pyrender.Mesh.from_trimesh(tm, smooth=False), pose=np.eye(4), name=f"link:{link}")
            self.link_nodes[link] = node
            lid = LEFT_HAND_ID if hand_of.get(link) == "left" else RIGHT_HAND_ID if hand_of.get(link) == "right" else ROBOT_BODY_ID
            self.seg_map[node] = id_to_color(lid)
        self.cam_node = self.scene.add(pyrender.IntrinsicsCamera(1, 1, 1, 1, znear=0.01, zfar=50.0), pose=np.eye(4))
        self.renderer = pyrender.OffscreenRenderer(width, height)
        self.valid_ids = np.array(sorted({0, LEFT_HAND_ID, RIGHT_HAND_ID, ROBOT_BODY_ID} | {o.obj_id for o in self.objects}), dtype=np.int32)
        # sub-sampled vertices of every static object (world frame) for cheap projections
        rng = np.random.default_rng(0)
        self.static_points = {}
        for o in self.objects:
            if o.frame == "world":
                V = o.V
                self.static_points[o.obj_id] = V[rng.choice(len(V), 300, replace=False)] if len(V) > 300 else V

    def set_tracked_pose(self, tracked_prim: str, R: np.ndarray, p: np.ndarray):
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = p
        for o in self.objects:
            if o.tracked_prim == tracked_prim:
                self.scene.set_pose(self.obj_nodes[o.obj_id], T)

    def hide_tracked(self, tracked_prim: str):
        T = np.eye(4)
        T[:3, 3] = [999.0, 999.0, 999.0]
        for o in self.objects:
            if o.tracked_prim == tracked_prim:
                self.scene.set_pose(self.obj_nodes[o.obj_id], T)

    def set_robot(self, link_poses: dict):
        for link, node in self.link_nodes.items():
            if link in link_poses:
                self.scene.set_pose(node, link_poses[link])

    def render_ids(self, cam: PinholeCamera):
        """Returns (id_map int32 HxW, depth float32 HxW)."""
        pr = self.pyrender
        self.cam_node.camera = pr.IntrinsicsCamera(cam.fx, cam.fy, cam.cx, cam.cy, znear=0.01, zfar=50.0)
        self.scene.set_pose(self.cam_node, world_axes_to_gl(cam.position, cam.rotation))
        color, depth = self.renderer.render(self.scene, flags=pr.RenderFlags.SEG, seg_node_map=self.seg_map)
        ids, valid = color_to_id(color)
        return clean_ids(ids, self.valid_ids, valid), depth

    def close(self):
        self.renderer.delete()


_NEIGHBOUR_OFFSETS = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]


def _shift(a: np.ndarray, dy: int, dx: int, fill):
    out = np.full_like(a, fill)
    h, w = a.shape
    ys, ye = max(dy, 0), h + min(dy, 0)
    xs, xe = max(dx, 0), w + min(dx, 0)
    out[ys:ye, xs:xe] = a[ys - dy:ye - dy, xs - dx:xe - dx]
    return out


def clean_ids(ids: np.ndarray, valid_ids: np.ndarray, checksum_ok: np.ndarray | None = None, max_passes: int = 3) -> np.ndarray:
    """Pixels whose decoded id is not a known instance (colour blending at
    edges) take the id of a valid 8-neighbour (vectorised, a few passes)."""
    ids = ids.copy()
    bad = ~np.isin(ids, valid_ids)
    if checksum_ok is not None:
        bad |= ~checksum_ok
    for _ in range(max_passes):
        if not bad.any():
            break
        for dy, dx in _NEIGHBOUR_OFFSETS:
            nid = _shift(ids, dy, dx, 0)
            nbad = _shift(bad, dy, dx, True)
            take = bad & ~nbad
            ids[take] = nid[take]
            bad &= ~take
    ids[bad] = 0
    return ids


def mask_stats(id_map: np.ndarray, obj_id: int):
    m = id_map == obj_id
    n = int(m.sum())
    if n == 0:
        return None
    ys, xs = np.nonzero(m)
    return {"visible_px": n, "bbox_visible_xyxy_px": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]}
