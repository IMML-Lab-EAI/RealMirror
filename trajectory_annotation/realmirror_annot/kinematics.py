"""Minimal URDF forward kinematics (fixed / revolute / continuous / prismatic joints).

Gives the world pose (4x4) of every link for a joint-angle dictionary, plus the
per-link visual-mesh AABB read from the URDF mesh files so a 3D box can be
attached to any link (used for the hands / grippers).
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def rpy_matrix(r: float, p: float, y: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def axis_angle_matrix(axis, angle: float) -> np.ndarray:
    a = np.asarray(axis, dtype=float)
    a = a / (np.linalg.norm(a) + 1e-12)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)


def make_T(R=None, t=None) -> np.ndarray:
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if t is not None:
        T[:3, 3] = t
    return T


@dataclass
class Joint:
    name: str
    kind: str
    parent: str
    child: str
    origin: np.ndarray  # 4x4
    axis: np.ndarray
    lower: float | None
    upper: float | None


@dataclass
class VisualMesh:
    filename: str
    origin: np.ndarray  # 4x4 link -> mesh
    scale: np.ndarray


class URDFKinematics:
    def __init__(self, urdf_path: str | Path):
        self.urdf_path = Path(urdf_path)
        root = ET.parse(self.urdf_path).getroot()
        self.joints: dict[str, Joint] = {}
        self.child_to_joint: dict[str, Joint] = {}
        self.links: list[str] = [l.get("name") for l in root.findall("link")]
        self.visuals: dict[str, list[VisualMesh]] = {}
        for l in root.findall("link"):
            vis = []
            for v in l.findall("visual"):
                mesh = v.find("geometry/mesh")
                if mesh is None:
                    continue
                o = v.find("origin")
                xyz = np.fromstring(o.get("xyz", "0 0 0"), sep=" ") if o is not None else np.zeros(3)
                rpy = np.fromstring(o.get("rpy", "0 0 0"), sep=" ") if o is not None else np.zeros(3)
                scale = np.fromstring(mesh.get("scale", "1 1 1"), sep=" ")
                vis.append(VisualMesh(mesh.get("filename"), make_T(rpy_matrix(*rpy), xyz), scale))
            self.visuals[l.get("name")] = vis
        for j in root.findall("joint"):
            kind = j.get("type")
            o = j.find("origin")
            xyz = np.fromstring(o.get("xyz", "0 0 0"), sep=" ") if o is not None else np.zeros(3)
            rpy = np.fromstring(o.get("rpy", "0 0 0"), sep=" ") if o is not None else np.zeros(3)
            ax = j.find("axis")
            axis = np.fromstring(ax.get("xyz"), sep=" ") if ax is not None else np.array([1.0, 0, 0])
            lim = j.find("limit")
            lower = float(lim.get("lower")) if lim is not None and lim.get("lower") else None
            upper = float(lim.get("upper")) if lim is not None and lim.get("upper") else None
            joint = Joint(j.get("name"), kind, j.find("parent").get("link"), j.find("child").get("link"),
                          make_T(rpy_matrix(*rpy), xyz), axis, lower, upper)
            self.joints[joint.name] = joint
            self.child_to_joint[joint.child] = joint
        children = {j.child for j in self.joints.values()}
        roots = [l for l in self.links if l not in children]
        if len(roots) != 1:
            raise ValueError(f"expected one root link, got {roots}")
        self.root_link = roots[0]

    def joint_transform(self, j: Joint, q: float) -> np.ndarray:
        if j.kind in ("revolute", "continuous"):
            return j.origin @ make_T(axis_angle_matrix(j.axis, q))
        if j.kind == "prismatic":
            return j.origin @ make_T(t=j.axis / (np.linalg.norm(j.axis) + 1e-12) * q)
        return j.origin

    def link_poses(self, angles: dict[str, float], base_T: np.ndarray | None = None) -> dict[str, np.ndarray]:
        """4x4 pose of every link in the frame given by ``base_T`` (root link pose)."""
        poses = {self.root_link: np.eye(4) if base_T is None else np.asarray(base_T, dtype=float)}
        # links are listed in a topological order in the URDF; be safe and iterate until done
        pending = [j for j in self.joints.values()]
        while pending:
            rest = []
            for j in pending:
                if j.parent in poses:
                    poses[j.child] = poses[j.parent] @ self.joint_transform(j, float(angles.get(j.name, 0.0)))
                else:
                    rest.append(j)
            if len(rest) == len(pending):
                raise ValueError("URDF joint tree is disconnected")
            pending = rest
        return poses

    def limit_violations(self, angles: dict[str, float], tol: float = 0.05) -> list[str]:
        bad = []
        for n, v in angles.items():
            j = self.joints.get(n)
            if j and j.lower is not None and not (j.lower - tol <= v <= j.upper + tol):
                bad.append(n)
        return bad

    def resolve_mesh_path(self, filename: str, mesh_root: Path) -> Path:
        # URDF filenames look like "A2_t2d0_flagship/meshes/raise_a2_t2d0_flagship/body.obj"
        rel = Path(filename.replace("package://", ""))
        for cand in (mesh_root / rel, mesh_root / rel.relative_to(rel.parts[0]) if len(rel.parts) > 1 else mesh_root / rel,
                     mesh_root / Path(*rel.parts[2:]) if len(rel.parts) > 2 else mesh_root / rel):
            if cand.exists():
                return cand
        raise FileNotFoundError(f"mesh {filename} not found under {mesh_root}")

    def link_local_aabb(self, link: str, mesh_root: Path, cache: dict | None = None) -> np.ndarray | None:
        """Axis-aligned bounds (2,3) of the link's visual meshes in the link frame."""
        import trimesh

        pts = []
        for v in self.visuals.get(link, []):
            key = (v.filename, tuple(v.scale))
            if cache is not None and key in cache:
                verts = cache[key]
            else:
                m = trimesh.load(self.resolve_mesh_path(v.filename, mesh_root), force="mesh")
                verts = np.asarray(m.vertices, dtype=float)
                if cache is not None:
                    cache[key] = verts
            verts = verts * v.scale
            verts_link = verts @ v.origin[:3, :3].T + v.origin[:3, 3]
            pts.append(verts_link)
        if not pts:
            return None
        allp = np.concatenate(pts)
        return np.stack([allp.min(axis=0), allp.max(axis=0)])


def aabb_corners(aabb: np.ndarray) -> np.ndarray:
    lo, hi = aabb
    return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])


def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return np.asarray(pts) @ T[:3, :3].T + T[:3, 3]


def link_local_points(kin: URDFKinematics, link: str, mesh_root: Path, max_points: int = 400, cache: dict | None = None, seed: int = 0) -> np.ndarray | None:
    """Sub-sampled visual-mesh vertices of a link in the link frame (meters)."""
    import trimesh

    rng = np.random.default_rng(seed)
    pts = []
    for v in kin.visuals.get(link, []):
        key = (v.filename, tuple(v.scale))
        if cache is not None and key in cache:
            verts = cache[key]
        else:
            m = trimesh.load(kin.resolve_mesh_path(v.filename, mesh_root), force="mesh")
            verts = np.asarray(m.vertices, dtype=float)
            if cache is not None:
                cache[key] = verts
        verts = verts * v.scale
        pts.append(verts @ v.origin[:3, :3].T + v.origin[:3, 3])
    if not pts:
        return None
    allp = np.concatenate(pts)
    if len(allp) > max_points:
        allp = allp[rng.choice(len(allp), max_points, replace=False)]
    return allp
