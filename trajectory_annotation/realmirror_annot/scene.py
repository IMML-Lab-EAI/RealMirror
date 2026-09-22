"""Whole-scene inventory + triangle-mesh cache extracted from the RealMirror scene USDs.

Every "object" (see configs/scene_objects.yaml for the split rules) is stored
with a triangulated, optionally decimated mesh:

* static objects  -> vertices directly in world coordinates (frame="world")
* tracked objects -> vertices in the frame of the prim whose pose is recorded
                     in trajectory.hdf5, with the USD scale baked in, so that
                     world(t) = R(t) @ v + p(t)          (frame="tracked")
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

SCENE_ROOT = "/scene"


def auto_label(prim_name: str) -> str:
    n = re.sub(r"^(model_|SM_P_|E_|P_)", "", prim_name)
    n = re.sub(r"_\d+$", "", n)
    n = re.sub(r"^\d+_?", "", n)
    n = re.sub(r"(?<=[a-z])(?=[A-Z])", "_", n)
    return n.lower().strip("_") or prim_name.lower()


def _usd_mat(M):
    rows = np.array([[M[r][c] for c in range(3)] for r in range(3)], dtype=float)
    t = np.array([M[3][c] for c in range(3)], dtype=float)
    return rows, t


def _triangulate(counts: np.ndarray, indices: np.ndarray) -> np.ndarray:
    tris = []
    k = 0
    for c in counts:
        if c >= 3:
            base = indices[k]
            for i in range(1, c - 1):
                tris.append((base, indices[k + i], indices[k + i + 1]))
        k += c
    return np.asarray(tris, dtype=np.int64).reshape(-1, 3)


def _decimate(V: np.ndarray, F: np.ndarray, target: int):
    if len(F) <= target:
        return V, F
    import fast_simplification

    v, f = fast_simplification.simplify(V.astype(np.float32), F.astype(np.int32), target_count=int(target))
    return np.asarray(v, dtype=np.float32), np.asarray(f, dtype=np.int64)


@dataclass
class SceneObject:
    obj_id: int
    prim_path: str          # relative to scene root, e.g. "/model_chopping_board/E_part1_2"
    name: str
    category: str
    background: bool
    frame: str              # "world" | "tracked"
    tracked_prim: str | None  # "/scene/..." path as recorded in trajectory.hdf5
    usd_position: list
    aabb_world_min: list
    aabb_world_max: list
    num_faces: int
    num_faces_original: int
    V: np.ndarray = field(repr=False)
    F: np.ndarray = field(repr=False)

    def meta(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k not in ("V", "F")}
        return d


class SceneCacheBuilder:
    def __init__(self, scene_cfg_path="configs/scene_objects.yaml", object_cfg_path="configs/realmirror_objects.yaml", asset_root="data/realmirror-asset", root="."):
        self.root = Path(root)
        self.scene_cfg = yaml.safe_load(open(self.root / scene_cfg_path))
        self.object_cfg = yaml.safe_load(open(self.root / object_cfg_path))
        self.asset_root = self.root / asset_root

    # --------------------------------------------------------------
    def _label_for(self, task_cfg: dict, rel_path: str, prim_name: str, tracked_meta: dict | None):
        labels = task_cfg.get("labels", {}) or {}
        if rel_path in labels:
            l = labels[rel_path]
            return l["name"], l.get("category", "object"), bool(l.get("background", False))
        for prefix, l in (task_cfg.get("prefix_labels", {}) or {}).items():
            if rel_path.startswith(prefix):
                return l["name"], l.get("category", "object"), bool(l.get("background", False))
        if tracked_meta:
            return tracked_meta["name"], tracked_meta.get("category", "object"), False
        return auto_label(prim_name), "object", False

    def build(self, task: str, out_dir: str | Path = "data/scene_cache") -> list[SceneObject]:
        from pxr import Usd, UsdGeom

        task_cfg = self.scene_cfg["tasks"][task]
        defaults = self.scene_cfg["defaults"]
        usd_path = self.asset_root / "scenes" / task / self.object_cfg["scene_usd"][task]
        tracked = self.object_cfg["objects"][task]  # "/scene/..." -> meta
        stage = Usd.Stage.Open(str(usd_path))
        dp = stage.GetDefaultPrim()
        root_path = dp.GetPath().pathString
        xc = UsdGeom.XformCache()
        split = set(task_cfg.get("split", []) or [])

        # enumerate object prims
        def enumerate_objects(prim, rel):
            for c in prim.GetChildren():
                if c.GetTypeName() in ("Material", "Shader", "Scope") or c.GetName() in ("Looks", "materials"):
                    continue
                crel = rel + "/" + c.GetName()
                if crel in split:
                    yield from enumerate_objects(c, crel)
                elif c.IsA(UsdGeom.Imageable):
                    yield crel, c

        objects: list[SceneObject] = []

        def gather(prim, only_under=None, exclude_under=None):
            """Triangles of the visible meshes under `prim` (world space)."""
            Vs, Fs, off = [], [], 0
            for mp in Usd.PrimRange(prim):
                if not mp.IsA(UsdGeom.Mesh):
                    continue
                mpath = mp.GetPath().pathString
                if only_under is not None and not (mpath == only_under or mpath.startswith(only_under + "/")):
                    continue
                if exclude_under is not None and (mpath == exclude_under or mpath.startswith(exclude_under + "/")):
                    continue
                img = UsdGeom.Imageable(mp)
                if img.ComputeVisibility() == UsdGeom.Tokens.invisible or img.ComputePurpose() not in ("default", "render"):
                    continue
                mesh = UsdGeom.Mesh(mp)
                pts = mesh.GetPointsAttr().Get()
                cnt = mesh.GetFaceVertexCountsAttr().Get()
                idx = mesh.GetFaceVertexIndicesAttr().Get()
                if not pts or not cnt or not idx:
                    continue
                rows, t = _usd_mat(xc.GetLocalToWorldTransform(mp))
                V = np.asarray(pts, dtype=float) @ rows + t
                F = _triangulate(np.asarray(cnt), np.asarray(idx))
                if len(F) == 0:
                    continue
                Vs.append(V)
                Fs.append(F + off)
                off += len(V)
            if not Vs:
                return None, None
            return np.concatenate(Vs).astype(np.float32), np.concatenate(Fs)

        def add_object(rel, prim, V, F, name, category, background, tracked_prim):
            n_orig = len(F)
            cap = defaults["max_faces_background"] if background else defaults["max_faces_per_object"]
            V, F = _decimate(V, F, cap)
            aabb_min, aabb_max = V.min(axis=0), V.max(axis=0)
            usd_pos = _usd_mat(xc.GetLocalToWorldTransform(prim))[1]
            frame = "world"
            if tracked_prim is not None:
                tprim = stage.GetPrimAtPath(root_path + tracked_prim[len(SCENE_ROOT):])
                rows, t = _usd_mat(xc.GetLocalToWorldTransform(tprim))
                scale = np.linalg.norm(rows, axis=1)
                R = (rows / scale[:, None]).T
                V = ((V - t) @ R).astype(np.float32)  # R^T (p - t): tracked frame, scale baked in
                frame = "tracked"
            objects.append(SceneObject(len(objects) + 1, rel, name, category, background, frame, tracked_prim,
                                       usd_pos.round(5).tolist(), aabb_min.round(4).tolist(), aabb_max.round(4).tolist(),
                                       int(len(F)), n_orig, V, F.astype(np.int32)))

        for rel, prim in enumerate_objects(dp, ""):
            # tracked prims (pose recorded in trajectory.hdf5) that live inside this object
            inside = []
            for tp, meta in tracked.items():
                trel = tp[len(SCENE_ROOT):]
                if trel == rel or rel.startswith(trel + "/"):
                    inside.append((tp, meta, None))          # the whole object IS the tracked body
                elif trel.startswith(rel + "/"):
                    inside.append((tp, meta, root_path + trel))  # tracked body is a strict descendant
            name, category, background = self._label_for(task_cfg, rel, prim.GetName(), inside[0][1] if inside else None)
            if not inside:
                V, F = gather(prim)
                if V is not None:
                    add_object(rel, prim, V, F, name, category, background, None)
                continue
            # tracked bodies carved out as their own objects (only the rigid body's own subtree moves)
            carved = []
            for tp, meta, tpath in inside:
                if tpath is None:
                    V, F = gather(prim)
                    if V is not None:
                        add_object(rel, prim, V, F, meta["name"], meta.get("category", category), False, tp)
                    carved = None
                    break
                V, F = gather(prim, only_under=tpath)
                if V is not None:
                    add_object(tp[len(SCENE_ROOT):], stage.GetPrimAtPath(tpath), V, F, meta["name"], meta.get("category", "object"), False, tp)
                    carved.append(tpath)
            if carved:
                # remainder of the group stays static
                V, F = _gather_excluding(gather, prim, carved)
                if V is not None and len(F) >= 12:  # skip leftover slivers (e.g. a label quad)
                    rest_name = name if name not in {o.name for o in objects} else name + "_other"
                    add_object(rel, prim, V, F, rest_name, category, background, None)
        out = self.root / out_dir / task
        out.mkdir(parents=True, exist_ok=True)
        (out / "inventory.json").write_text(json.dumps({"task": task, "scene_usd": str(usd_path), "objects": [o.meta() for o in objects]}, indent=1))
        np.savez_compressed(out / "meshes.npz", **{f"V{o.obj_id}": o.V for o in objects}, **{f"F{o.obj_id}": o.F for o in objects})
        return objects


def _gather_excluding(gather, prim, excluded):
    """gather() over `prim` skipping every subtree in `excluded` (list of prim paths)."""
    import numpy as _np
    from pxr import Usd, UsdGeom

    # reuse gather's mesh handling by calling it per top-level child not in excluded; simpler: gather all then
    # drop excluded meshes -> implemented by passing exclude_under one at a time is not enough for >1 subtree,
    # so we walk manually here.
    Vs, Fs, off = [], [], 0
    for mp in Usd.PrimRange(prim):
        if not mp.IsA(UsdGeom.Mesh):
            continue
        mpath = mp.GetPath().pathString
        if any(mpath == e or mpath.startswith(e + "/") for e in excluded):
            continue
        V, F = gather(mp)  # single mesh
        if V is None:
            continue
        Vs.append(V)
        Fs.append(F + off)
        off += len(V)
    if not Vs:
        return None, None
    return _np.concatenate(Vs).astype(_np.float32), _np.concatenate(Fs)


def load_scene_cache(task: str, cache_dir: str | Path = "data/scene_cache", root: str | Path = ".") -> list[SceneObject]:
    d = Path(root) / cache_dir / task
    inv = json.loads((d / "inventory.json").read_text())
    data = np.load(d / "meshes.npz")
    objs = []
    for m in inv["objects"]:
        objs.append(SceneObject(V=data[f"V{m['obj_id']}"], F=data[f"F{m['obj_id']}"], **m))
    return objs
