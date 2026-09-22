"""Reader for one RealMirror evaluation episode directory (trajectory.hdf5 + json)."""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

CAMERA_KEYS = {"head_camera": "head_camera_bgr", "left_wrist_camera": "left_wrist_camera_bgr", "right_wrist_camera": "right_wrist_camera_bgr"}


class Episode:
    def __init__(self, episode_dir: str | Path):
        self.dir = Path(episode_dir)
        self.metadata = json.loads((self.dir / "metadata.json").read_text())
        self.task_config = json.loads((self.dir / "task_config.json").read_text())
        self.h5 = h5py.File(self.dir / "trajectory.hdf5", "r")
        self.num_steps = int(self.h5["step_index"].shape[0])
        self.prim_paths = [p.decode() if isinstance(p, bytes) else str(p) for p in self.h5["objects/prim_paths"][:]]
        self.dof_names = [d.decode() if isinstance(d, bytes) else str(d) for d in self.h5["robot/dof_names"][:]]
        self.task_name = self.metadata["task_name"]
        self.success = bool(self.metadata.get("success", False))
        self.scene = self.task_config["scene"][0]

    def close(self):
        self.h5.close()

    # ---- per-step accessors -------------------------------------------------
    def info(self, t: int) -> dict:
        raw = self.h5["infos/json"][t]
        return json.loads(raw.decode() if isinstance(raw, bytes) else raw)

    def active_task(self, t: int = 0) -> dict:
        return self.info(t).get("active_task", {})

    def object_positions(self, t: int) -> np.ndarray:
        return self.h5["states/objects/positions"][t]

    def object_quaternions(self, t: int) -> np.ndarray:
        return self.h5["states/objects/quaternions_wxyz"][t]

    def object_linear_velocities(self, t: int) -> np.ndarray:
        return self.h5["states/objects/linear_velocities"][t]

    def robot_root(self, t: int):
        return self.h5["states/robot/root_position"][t], self.h5["states/robot/root_quaternion_wxyz"][t]

    def joint_positions(self, t: int) -> np.ndarray:
        return self.h5["states/robot/joint_positions"][t]

    def state26(self, t: int) -> np.ndarray:
        return self.h5["observations/state"][t]

    def step_success(self, t: int) -> bool:
        return bool(self.h5["success"][t])

    def image(self, camera: str, t: int) -> np.ndarray:
        """BGR uint8 frame as stored (camera_color_order = BGR)."""
        return self.h5[f"observations/{CAMERA_KEYS[camera]}"][t]

    def image_shape(self, camera: str):
        return tuple(self.h5[f"observations/{CAMERA_KEYS[camera]}"].shape[1:3])
