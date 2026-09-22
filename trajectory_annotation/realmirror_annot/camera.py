"""Pinhole camera model matching Isaac Sim's ``omni.isaac.sensor.Camera``.

Isaac's ``Camera.set_world_pose`` (camera_axes="world") uses a pose frame in
which the camera looks along +X, +Y points to the image left and +Z up.
Pixel origin is top-left.  Intrinsics follow the USD camera model:
``fx = width * focal_length / horizontal_aperture``; Isaac sets the vertical
aperture so that ``fy == fx``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# UsdGeom.Camera default focalLength; RealMirror only overrides horizontalAperture.
# Verified on Task1 episode 0: lid-centroid fit gives 48.5 mm, basket bbox matches at 50.
ISAAC_DEFAULT_FOCAL_LENGTH_MM = 50.0


def quat_wxyz_to_matrix(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    n = np.sqrt(w * w + x * x + y * y + z * z)
    if n == 0:
        raise ValueError("zero quaternion")
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def matrix_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    m = np.asarray(R, dtype=float)
    t = np.trace(m)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w, x, y, z = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
    return np.array([w, x, y, z])


@dataclass
class PinholeCamera:
    name: str
    width: int
    height: int
    horizontal_aperture: float
    focal_length: float = ISAAC_DEFAULT_FOCAL_LENGTH_MM
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))  # world <- camera(world-axes)

    @property
    def fx(self) -> float:
        return self.width * self.focal_length / self.horizontal_aperture

    @property
    def fy(self) -> float:
        return self.fx  # Isaac sets vertical aperture = horizontal * H / W

    @property
    def cx(self) -> float:
        return self.width / 2.0

    @property
    def cy(self) -> float:
        return self.height / 2.0

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])

    def set_pose(self, position, quat_wxyz) -> None:
        self.position = np.asarray(position, dtype=float)
        self.rotation = quat_wxyz_to_matrix(quat_wxyz)

    def set_pose_matrix(self, T: np.ndarray) -> None:
        self.position = np.asarray(T[:3, 3], dtype=float)
        self.rotation = np.asarray(T[:3, :3], dtype=float)

    def world_to_camera(self, points_world) -> np.ndarray:
        """Returns (N,3) in the Isaac world-axes camera frame: x forward, y left, z up."""
        p = np.atleast_2d(np.asarray(points_world, dtype=float))
        return (p - self.position) @ self.rotation  # R^T (p - t)

    def project(self, points_world):
        """Returns pixel coords (N,2) [u,v] and forward depth (N,). Points with
        depth <= 0 get NaN pixels."""
        c = self.world_to_camera(points_world)
        depth = c[:, 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = self.cx - self.fx * c[:, 1] / depth
            v = self.cy - self.fy * c[:, 2] / depth
        uv = np.stack([u, v], axis=1)
        uv[depth <= 1e-6] = np.nan
        return uv, depth

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "width": self.width,
            "height": self.height,
            "focal_length_mm": self.focal_length,
            "horizontal_aperture_mm": self.horizontal_aperture,
            "K": self.K.round(4).tolist(),
            "position_world": self.position.round(5).tolist(),
            "quat_wxyz_world": matrix_to_quat_wxyz(self.rotation).round(6).tolist(),
            "axes_convention": "isaac_world(+x forward,+y left,+z up)",
        }


def bbox_from_points(uv: np.ndarray, depth: np.ndarray, width: int, height: int):
    """2D AABB (pixels, xyxy) of projected points that lie in front of the camera.

    Returns (bbox_xyxy_clipped, in_front_fraction, in_frame) or (None, frac, False).
    """
    valid = depth > 1e-6
    frac = float(valid.mean()) if len(valid) else 0.0
    if not valid.any():
        return None, frac, False
    pts = uv[valid]
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    in_frame = x1 >= 0 and y1 >= 0 and x0 <= width - 1 and y0 <= height - 1
    clipped = [
        float(np.clip(x0, 0, width - 1)),
        float(np.clip(y0, 0, height - 1)),
        float(np.clip(x1, 0, width - 1)),
        float(np.clip(y1, 0, height - 1)),
    ]
    return clipped, frac, bool(in_frame)
