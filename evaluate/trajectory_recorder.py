"""Streaming per-step trajectory recording for RealMirror evaluation."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping
import warnings

import cv2
import h5py
import numpy as np


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _as_array(value: Any, *, dtype=None) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    return np.copy(array)


class RealMirrorTrajectoryRecorder:
    """Write one HDF5 file per rollout without buffering the full episode."""

    FORMAT_VERSION = 1

    def __init__(
        self,
        run_dir: str | Path,
        task_name: str,
        model_type: str,
        video_fps: int = 20,
        save_video: bool = True,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.task_name = task_name
        self.model_type = model_type
        self.video_fps = video_fps
        self.save_video = save_video

        self._active = False
        self._step_count = 0
        self._episode_dir: Path | None = None
        self._h5: h5py.File | None = None
        self._metadata: dict[str, Any] = {}
        self._robot = None
        self._tracked_prims: list[tuple[str, Any]] = []

    def start_episode(
        self,
        rollout_index: int,
        robot: Any,
        tracked_prims: Mapping[str, Any],
        episode_metadata: Mapping[str, Any] | None = None,
        task_config: Mapping[str, Any] | None = None,
    ) -> Path:
        """Open an episode file and capture the initial simulator state."""
        if self._active:
            self.close(status="interrupted_by_next_rollout")

        self._robot = robot
        self._tracked_prims = list(tracked_prims.items())
        self._step_count = 0
        self._episode_dir = self._allocate_episode_dir(rollout_index)
        self._episode_dir.mkdir(parents=True, exist_ok=False)
        self._h5 = h5py.File(self._episode_dir / "trajectory.hdf5", "w")
        self._active = True

        created_at = datetime.now(timezone.utc).isoformat()
        self._metadata = {
            "format_version": self.FORMAT_VERSION,
            "task_name": self.task_name,
            "model_type": self.model_type,
            "rollout_index": rollout_index,
            "created_at_utc": created_at,
            "status": "recording",
            "num_steps": 0,
            "camera_color_order": "BGR",
            "quaternion_convention": "wxyz",
            "episode_metadata": _jsonable(dict(episode_metadata or {})),
        }
        self._h5.attrs.update(
            {
                "format_version": self.FORMAT_VERSION,
                "task_name": self.task_name,
                "model_type": self.model_type,
                "rollout_index": rollout_index,
                "created_at_utc": created_at,
                "status": "recording",
                "num_steps": 0,
                "camera_color_order": "BGR",
                "quaternion_convention": "wxyz",
            }
        )

        string_dtype = h5py.string_dtype(encoding="utf-8")
        object_paths = [path for path, _ in self._tracked_prims]
        self._h5.create_dataset("objects/prim_paths", data=object_paths, dtype=string_dtype)

        dof_names = list(robot.get_dof_names() or [])
        self._h5.create_dataset("robot/dof_names", data=dof_names, dtype=string_dtype)

        initial_snapshot = self._capture_snapshot()
        for key, value in initial_snapshot.items():
            self._h5.create_dataset(f"initial_state/{key}", data=value)

        if task_config is not None:
            (self._episode_dir / "task_config.json").write_text(
                json.dumps(_jsonable(dict(task_config)), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        self._write_metadata()
        self._h5.flush()
        return self._episode_dir

    def record_step(
        self,
        *,
        step_index: int,
        observation: Mapping[str, Any],
        policy_actions: Any,
        applied_joint_positions: Any,
        success: bool,
        info: Mapping[str, Any] | None = None,
    ) -> None:
        """Append one evaluation-loop transition and simulator snapshot."""
        if not self._active or self._h5 is None:
            raise RuntimeError("Trajectory recorder received a step before start_episode().")

        self._append("step_index", step_index)
        for key, value in observation.items():
            self._append(f"observations/{key}", value)
        self._append("actions/policy_chunk", policy_actions)
        self._append("actions/applied_joint_positions", applied_joint_positions)
        self._append("success", success)
        self._append("infos/json", json.dumps(_jsonable(dict(info or {})), ensure_ascii=False))

        for key, value in self._capture_snapshot().items():
            self._append(f"states/{key}", value)

        self._step_count += 1
        self._h5.attrs["num_steps"] = self._step_count
        self._h5.flush()

    def end_episode(self, result: Mapping[str, Any]) -> Path | None:
        """Finalize an episode and optionally encode its head-camera preview."""
        if not self._active:
            return None

        episode_dir = self._episode_dir
        result_dict = dict(result)
        success = bool(result_dict.get("success", False))
        self._metadata.update(
            {
                "status": "complete",
                "num_steps": self._step_count,
                "success": success,
                "result": _jsonable(result_dict),
            }
        )
        if self._h5 is not None:
            self._h5.attrs["status"] = "complete"
            self._h5.attrs["success"] = success
            self._h5.attrs["num_steps"] = self._step_count
            self._h5.flush()
            self._h5.close()
            self._h5 = None
        self._write_metadata()

        if self.save_video and episode_dir is not None and self._step_count:
            try:
                self._write_preview_video(episode_dir)
            except Exception as exc:
                warnings.warn(f"Could not write trajectory preview video: {exc}")

        print(f"[trajectory] saved {self._step_count} steps to {episode_dir}")
        self._active = False
        return episode_dir

    def close(self, status: str = "interrupted") -> None:
        """Close a partial episode while preserving every flushed step."""
        if not self._active:
            return
        self._metadata.update({"status": status, "num_steps": self._step_count})
        if self._h5 is not None:
            self._h5.attrs["status"] = status
            self._h5.attrs["num_steps"] = self._step_count
            self._h5.flush()
            self._h5.close()
            self._h5 = None
        self._write_metadata()
        print(f"[trajectory] preserved partial episode at {self._episode_dir}")
        self._active = False

    def _allocate_episode_dir(self, rollout_index: int) -> Path:
        base = self.run_dir / f"episode-{rollout_index:06d}"
        if not base.exists():
            return base
        attempt = 1
        while True:
            candidate = self.run_dir / f"episode-{rollout_index:06d}-attempt-{attempt:02d}"
            if not candidate.exists():
                return candidate
            attempt += 1

    def _capture_snapshot(self) -> dict[str, np.ndarray]:
        joint_positions = self._robot.get_joint_positions()
        joint_velocities = self._robot.get_joint_velocities()
        root_position, root_quaternion = self._robot.get_world_pose()
        root_linear_velocity = self._robot.get_linear_velocity()
        root_angular_velocity = self._robot.get_angular_velocity()

        object_positions = []
        object_quaternions = []
        object_linear_velocities = []
        object_angular_velocities = []
        for _, prim in self._tracked_prims:
            try:
                position, quaternion = prim.get_world_pose()
                linear_velocity = prim.get_linear_velocity()
                angular_velocity = prim.get_angular_velocity()
                object_positions.append(_as_array(position, dtype=np.float64))
                object_quaternions.append(_as_array(quaternion, dtype=np.float64))
                object_linear_velocities.append(_as_array(linear_velocity, dtype=np.float64))
                object_angular_velocities.append(_as_array(angular_velocity, dtype=np.float64))
            except Exception:
                object_positions.append(np.full(3, np.nan, dtype=np.float64))
                object_quaternions.append(np.full(4, np.nan, dtype=np.float64))
                object_linear_velocities.append(np.full(3, np.nan, dtype=np.float64))
                object_angular_velocities.append(np.full(3, np.nan, dtype=np.float64))

        return {
            "robot/joint_positions": _as_array(joint_positions, dtype=np.float64),
            "robot/joint_velocities": _as_array(joint_velocities, dtype=np.float64),
            "robot/root_position": _as_array(root_position, dtype=np.float64),
            "robot/root_quaternion_wxyz": _as_array(root_quaternion, dtype=np.float64),
            "robot/root_linear_velocity": _as_array(root_linear_velocity, dtype=np.float64),
            "robot/root_angular_velocity": _as_array(root_angular_velocity, dtype=np.float64),
            "objects/positions": np.asarray(object_positions, dtype=np.float64).reshape(-1, 3),
            "objects/quaternions_wxyz": np.asarray(object_quaternions, dtype=np.float64).reshape(-1, 4),
            "objects/linear_velocities": np.asarray(object_linear_velocities, dtype=np.float64).reshape(-1, 3),
            "objects/angular_velocities": np.asarray(object_angular_velocities, dtype=np.float64).reshape(-1, 3),
        }

    def _append(self, path: str, value: Any) -> None:
        if self._h5 is None:
            raise RuntimeError("Trajectory HDF5 file is not open.")

        if isinstance(value, str):
            array = np.asarray(value, dtype=h5py.string_dtype(encoding="utf-8"))
            compression = None
        else:
            array = np.asarray(value)
            if array.dtype.kind in {"U", "O"}:
                array = np.asarray(
                    json.dumps(_jsonable(value), ensure_ascii=False),
                    dtype=h5py.string_dtype(encoding="utf-8"),
                )
            compression = "lzf" if array.ndim > 0 and array.nbytes >= 1024 else None

        if path not in self._h5:
            self._h5.create_dataset(
                path,
                shape=(0, *array.shape),
                maxshape=(None, *array.shape),
                chunks=(1, *array.shape),
                dtype=array.dtype,
                compression=compression,
            )
        dataset = self._h5[path]
        if dataset.shape[1:] != array.shape:
            raise ValueError(
                f"Trajectory value shape changed for {path}: "
                f"expected {dataset.shape[1:]}, got {array.shape}"
            )
        dataset.resize(dataset.shape[0] + 1, axis=0)
        dataset[-1] = array

    def _write_metadata(self) -> None:
        if self._episode_dir is not None:
            (self._episode_dir / "metadata.json").write_text(
                json.dumps(self._metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    def _write_preview_video(self, episode_dir: Path) -> None:
        with h5py.File(episode_dir / "trajectory.hdf5", "r") as trajectory:
            if "observations/head_camera_bgr" not in trajectory:
                return
            frames = trajectory["observations/head_camera_bgr"]
            if len(frames) == 0:
                return

            first_frame = np.asarray(frames[0])
            height, width = first_frame.shape[:2]
            video_path = episode_dir / "preview.mp4"
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                float(self.video_fps),
                (width, height),
            )
            if not writer.isOpened():
                raise RuntimeError(f"Could not open video writer for {video_path}")
            try:
                for frame_bgr in frames:
                    writer.write(np.ascontiguousarray(frame_bgr))
            finally:
                writer.release()
