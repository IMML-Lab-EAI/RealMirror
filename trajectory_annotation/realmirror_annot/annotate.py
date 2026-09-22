"""Build the per-step annotation dict for a RealMirror evaluation episode.

Two levels of information are produced for every step and camera:

* geometric (always): 3D poses/boxes of tracked objects and both hands,
  projected 2D boxes (tight = projected mesh vertices, loose = 3D-box corners),
  frustum visibility.
* rendered (``render=True``, needs a GPU/EGL context): an instance-ID image of
  the whole scene (static scene objects + tracked objects + robot) that gives
  pixel-exact masks, visible bounding boxes, occlusion, and a list of every
  scene object that is actually visible in each camera.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from .camera import ISAAC_DEFAULT_FOCAL_LENGTH_MM, PinholeCamera, bbox_from_points, matrix_to_quat_wxyz, quat_wxyz_to_matrix
from .episode import Episode
from .extents import load_object_table, object_local_corners
from .kinematics import URDFKinematics, aabb_corners, link_local_points, make_T, transform_points

FAR_AWAY = 100.0  # inactive objects are parked at (999, 999, 999)
CAMERAS = ["head_camera", "left_wrist_camera", "right_wrist_camera"]
LEFT_HAND_ID, RIGHT_HAND_ID, ROBOT_BODY_ID = 60001, 60002, 60003
MIN_VISIBLE_PX = 1


def review_steps_for(num_steps: int, count: int) -> list[int]:
    """`count` evenly spaced step indices (first, ..., last)."""
    if num_steps <= 0 or count <= 0:
        return []
    return sorted(set(int(round(v)) for v in np.linspace(0, num_steps - 1, min(count, num_steps))))


def _r(x, nd=4):
    return np.round(np.asarray(x, dtype=float), nd).tolist()


def usd_camera_to_world_axes(T_usd: np.ndarray) -> np.ndarray:
    """USD camera frame (looks -Z, +Y up, +X right) -> Isaac 'world axes'
    frame (+X forward, +Y left, +Z up) with the same origin."""
    R = T_usd[:3, :3]
    Rw = np.stack([-R[:, 2], -R[:, 0], R[:, 1]], axis=1)
    return make_T(Rw, T_usd[:3, 3])


def _norm_box(box, cam):
    return None if box is None else _r([box[0] / cam.width, box[1] / cam.height, box[2] / cam.width, box[3] / cam.height], 4)


def _hull_area_px(uv: np.ndarray, depth: np.ndarray, width: int, height: int) -> int:
    """Pixels covered by the convex hull of the projected points, clipped to the image."""
    ok = depth > 1e-6
    if ok.sum() < 3:
        return 0
    pts = uv[ok]
    pts = pts[np.isfinite(pts).all(axis=1)]
    pts = np.clip(pts, [-4 * width, -4 * height], [5 * width, 5 * height])
    if len(pts) < 3:
        return 0
    hull = cv2.convexHull(pts.astype(np.float32))
    canvas = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(canvas, np.round(hull).astype(np.int32), 1)
    return int(canvas.sum())


def _view_entry(cam: PinholeCamera, corners_world: np.ndarray, center_world: np.ndarray, points_world: np.ndarray | None = None) -> dict:
    """2D information of one entity in one camera (geometric part).

    ``bbox_xyxy_px`` is the tight box of the projected mesh vertices when
    ``points_world`` is given (else of the 3D-box corners); ``bbox_box3d_xyxy_px``
    is always the box of the projected 3D-box corners.
    """
    uv, depth = cam.project(corners_world)
    box3d_2d, frac_front, in_frame3d = bbox_from_points(uv, depth, cam.width, cam.height)
    hull_px = None
    if points_world is not None and len(points_world):
        puv, pdepth = cam.project(points_world)
        box, pfrac, in_frame = bbox_from_points(puv, pdepth, cam.width, cam.height)
        inside = (pdepth > 1e-6) & (puv[:, 0] >= 0) & (puv[:, 0] <= cam.width - 1) & (puv[:, 1] >= 0) & (puv[:, 1] <= cam.height - 1)
        in_frame_fraction = float(inside.mean())
        if in_frame:
            hull_px = _hull_area_px(puv, pdepth, cam.width, cam.height)
    else:
        box, in_frame, in_frame_fraction = box3d_2d, in_frame3d, float("nan")
    c_uv, c_depth = cam.project(center_world[None])
    entry = {
        "in_frustum": bool(box is not None and in_frame),
        "points_in_frame_fraction": None if np.isnan(in_frame_fraction) else round(in_frame_fraction, 3),
        "corners_in_front_fraction": round(frac_front, 3),
        "depth_m": round(float(c_depth[0]), 4),
        "center_px": None if np.isnan(c_uv[0]).any() else _r(c_uv[0], 1),
        "bbox_xyxy_px": None if box is None else _r(box, 1),
        "bbox_xyxy_norm": _norm_box(box, cam),
        "bbox_box3d_xyxy_px": None if box3d_2d is None else _r(box3d_2d, 1),
        "corners_px": None if box3d_2d is None else _r(np.nan_to_num(uv, nan=-1.0), 1),
        "unoccluded_area_px_approx": hull_px,
    }
    if box is not None:
        entry["area_px"] = round(float((box[2] - box[0]) * (box[3] - box[1])), 1)
    return entry


def _add_mask_stats(entry: dict, id_map: np.ndarray | None, mask_id: int):
    """Rendered part: pixel-exact visibility from the instance-ID image."""
    if id_map is None:
        entry["visible"] = entry.get("in_frustum", False)
        entry["visibility_source"] = "frustum_only"
        return
    m = id_map == mask_id
    n = int(m.sum())
    entry["visible_px"] = n
    entry["visible"] = bool(n >= MIN_VISIBLE_PX)
    entry["visibility_source"] = "render"
    if n:
        ys, xs = np.nonzero(m)
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        entry["bbox_visible_xyxy_px"] = [x0, y0, x1, y1]
        h, w = id_map.shape
        entry["bbox_visible_xyxy_norm"] = _r([x0 / w, y0 / h, x1 / w, y1 / h], 4)
        entry["mask_centroid_px"] = [round(float(xs.mean()), 1), round(float(ys.mean()), 1)]
        # largest connected component (slivers seen through gaps inflate the full box)
        crop = m[y0:y1 + 1, x0:x1 + 1].astype(np.uint8)
        ncomp, lab, stats, _ = cv2.connectedComponentsWithStats(crop, connectivity=8)
        entry["n_components"] = int(ncomp - 1)
        if ncomp > 2:
            k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            sx, sy, sw, sh, sa = (int(v) for v in stats[k])
            entry["visible_px_main"] = sa
            entry["bbox_visible_main_xyxy_px"] = [x0 + sx, y0 + sy, x0 + sx + sw - 1, y0 + sy + sh - 1]
        else:
            entry["visible_px_main"] = n
            entry["bbox_visible_main_xyxy_px"] = [x0, y0, x1, y1]
    hull = entry.get("unoccluded_area_px_approx")
    if hull:
        entry["occluded_fraction_approx"] = round(max(0.0, 1.0 - n / hull), 3)


class Annotator:
    def __init__(self, config_path: str | Path = "configs/realmirror_objects.yaml", object_table_dir: str | Path = "data/object_tables",
                 root: str | Path = ".", render: bool = False, scene_cache_dir: str | Path = "data/scene_cache"):
        self.root = Path(root)
        self.cfg = yaml.safe_load(open(self.root / config_path))
        rc = self.cfg["robot"]
        self.kin = URDFKinematics(self.root / rc["urdf"])
        self.mesh_root = self.root / rc["mesh_root"]
        self.hand_links = rc["hand_links"]
        self.palm_link = rc["palm_link"]
        self.fingertip_links = rc["fingertip_links"]
        self.closure_joint = rc["closure_joint"]
        self.closure_closed = float(rc["closure_closed_deg"])
        self.wrist_cfg = rc["wrist_camera"]
        self.object_table_dir = self.root / object_table_dir
        self.render = render
        self.scene_cache_dir = scene_cache_dir
        cache = {}
        self.link_points = {}
        for side in ("left", "right"):
            for link in self.hand_links[side]:
                pts = link_local_points(self.kin, link, self.mesh_root, max_points=400, cache=cache)
                if pts is not None:
                    self.link_points[link] = pts
        self._tables = {}
        self._points = {}
        self._renderers = {}

    # ------------------------------------------------------------------
    def object_table(self, task: str) -> dict:
        if task not in self._tables:
            self._tables[task] = load_object_table(self.object_table_dir / f"{task}.json")
        return self._tables[task]

    def object_points(self, task: str) -> dict:
        if task not in self._points:
            f = self.object_table_dir / f"{task}_points.npz"
            self._points[task] = {k.replace("|", "/"): v for k, v in np.load(f).items()} if f.exists() else {}
        return self._points[task]

    def static_scene(self, task: str):
        """Static (non-tracked) scene objects with sub-sampled world points; no GPU needed."""
        if not hasattr(self, "_static"):
            self._static = {}
        if task not in self._static:
            from .scene import load_scene_cache

            rng = np.random.default_rng(0)
            objs = []
            for o in load_scene_cache(task, self.scene_cache_dir, self.root):
                if o.frame != "world":
                    continue
                pts = o.V[rng.choice(len(o.V), 300, replace=False)] if len(o.V) > 300 else o.V
                objs.append((o, pts))
            self._static[task] = objs
        return self._static[task]

    def renderer(self, task: str, width: int, height: int):
        if not self.render:
            return None
        if task not in self._renderers:
            from .render import SceneRenderer

            self._renderers[task] = SceneRenderer(task, self.kin, self.mesh_root, self.hand_links, width, height, self.scene_cache_dir, self.root)
        return self._renderers[task]

    def make_cameras(self, ep: Episode) -> dict[str, PinholeCamera]:
        cams = {}
        hc = ep.scene["camera_configs"]["head_camera"]
        h, w = ep.image_shape("head_camera")
        head = PinholeCamera("head_camera", w, h, horizontal_aperture=float(hc.get("horizontal_aperture", 20.955)),
                             focal_length=float(hc["focal_length"]) if hc.get("focal_length") else ISAAC_DEFAULT_FOCAL_LENGTH_MM)
        head.set_pose(hc["position"], hc["orientation"])
        cams["head_camera"] = head
        for side in ("left", "right"):
            name = f"{side}_wrist_camera"
            h, w = ep.image_shape(name)
            cams[name] = PinholeCamera(name, w, h, horizontal_aperture=float(self.wrist_cfg["horizontal_aperture_mm"]),
                                       focal_length=float(self.wrist_cfg["focal_length_mm"]))
        return cams

    def _wrist_camera_pose(self, side: str, link_poses: dict) -> np.ndarray:
        wc = self.wrist_cfg[side]
        T_parent = link_poses[wc["parent_link"]]
        T_cam_usd = T_parent @ make_T(quat_wxyz_to_matrix(wc["local_rot0_wxyz"]), wc["local_pos0"])
        return usd_camera_to_world_axes(T_cam_usd)

    # ------------------------------------------------------------------
    def episode_meta(self, ep: Episode, cams: dict, renderer=None) -> dict:
        table = self.object_table(ep.task_name)
        meta = {
            "episode_dir": str(ep.dir),
            "task_name": ep.task_name,
            "model_type": ep.metadata.get("model_type"),
            "rollout_index": ep.metadata.get("rollout_index"),
            "success": ep.success,
            "num_steps": ep.num_steps,
            "instruction": ep.active_task(0).get("task_name"),
            "active_task": ep.active_task(0),
            "result": ep.metadata.get("result"),
            "success_criteria": ep.scene.get("eval_cfg", {}).get("success_criteria"),
            "camera_intrinsics": {n: c.to_dict() for n, c in cams.items()},
            "tracked_objects": [
                {"prim_path": pp, **{k: v for k, v in table.get(pp, {}).items() if k in ("name", "category", "size_m", "local_aabb_min", "local_aabb_max", "world_scale")}}
                for pp in ep.prim_paths
            ],
            "dof_names": ep.dof_names,
            "hand_links": self.hand_links,
            "mask_ids": {"left_hand": LEFT_HAND_ID, "right_hand": RIGHT_HAND_ID, "robot_body": ROBOT_BODY_ID, "background": 0},
            "coordinate_conventions": {
                "world": "Isaac Sim world frame, meters, z-up; robot root at origin",
                "quaternions": "wxyz",
                "bbox_xyxy_px": "tight box of projected mesh vertices, top-left origin pixels, [x_min, y_min, x_max, y_max]",
                "bbox_box3d_xyxy_px": "box of the projected 3D-box corners (looser)",
                "bbox_visible_xyxy_px": "box of the pixels actually visible in the rendered instance-ID image",
                "box3d_corners": "8 corners ordered (x-,y-,z-),(x-,y-,z+),(x-,y+,z-),... in the object/hand frame AABB",
                "id_maps": "masks.h5 dataset id_map[step, camera_index, y, x] (uint16); camera_index follows 'cameras' order; 0 = nothing",
                "occluded_fraction_approx": "1 - visible_px / convex-hull area of the projected mesh vertices",
            },
            "cameras": CAMERAS,
            "source": "sim_gt",
            "rendered": bool(renderer is not None),
        }
        if renderer is not None:
            meta["scene_objects"] = [
                {"mask_id": o.obj_id, "name": o.name, "category": o.category, "background": o.background, "prim_path": o.prim_path,
                 "frame": o.frame, "tracked_prim": o.tracked_prim, "usd_position": o.usd_position,
                 "aabb_world_min": o.aabb_world_min, "aabb_world_max": o.aabb_world_max}
                for o in renderer.objects
            ]
        return meta

    def object_roles(self, ep: Episode) -> dict[str, str]:
        at = ep.active_task(0)
        roles = {pp: "distractor" for pp in ep.prim_paths}
        for key, role in (("cylinder_prim_path", "target"), ("target_prim_path", "destination"), ("plate_prim_path", "source_support"), ("basket_prim_path", "destination_basket")):
            p = at.get(key)
            if p and p in roles:
                roles[p] = role
        return roles

    def annotate_step(self, ep: Episode, t: int, cams: dict, roles: dict, renderer=None):
        table = self.object_table(ep.task_name)
        points = self.object_points(ep.task_name)
        root_p, root_q = ep.robot_root(t)
        root_T = make_T(quat_wxyz_to_matrix(root_q), root_p)
        joints = ep.joint_positions(t)
        angles = dict(zip(ep.dof_names, joints.tolist()))
        link_poses = self.kin.link_poses(angles, root_T)
        for side in ("left", "right"):
            cams[f"{side}_wrist_camera"].set_pose_matrix(self._wrist_camera_pose(side, link_poses))

        pos = ep.object_positions(t)
        quat = ep.object_quaternions(t)
        vel = ep.object_linear_velocities(t)
        active = [bool(np.abs(p).max() < FAR_AWAY and np.isfinite(p).all()) for p in pos]

        # ---- render instance ids -----------------------------------------
        id_maps = {n: None for n in cams}
        if renderer is not None:
            renderer.set_robot(link_poses)
            for i, pp in enumerate(ep.prim_paths):
                if active[i]:
                    renderer.set_tracked_pose(pp, quat_wxyz_to_matrix(quat[i]), pos[i])
                else:
                    renderer.hide_tracked(pp)
            for n, c in cams.items():
                id_maps[n], _ = renderer.render_ids(c)
        mask_id_of_prim = {}
        if renderer is not None:
            for o in renderer.objects:
                if o.tracked_prim:
                    mask_id_of_prim[o.tracked_prim] = o.obj_id

        # ---- grippers ---------------------------------------------------
        grippers = {}
        palm_pos = {}
        for side, mask_id in (("left", LEFT_HAND_ID), ("right", RIGHT_HAND_ID)):
            palm_T = link_poses[self.palm_link[side]]
            pts = np.concatenate([transform_points(link_poses[link], self.link_points[link]) for link in self.hand_links[side] if link in self.link_points])
            local = transform_points(np.linalg.inv(palm_T), pts)
            lo, hi = local.min(axis=0), local.max(axis=0)
            corners_world = transform_points(palm_T, aabb_corners(np.stack([lo, hi])))
            center_world = transform_points(palm_T, ((lo + hi) / 2)[None])[0]
            palm_pos[side] = palm_T[:3, 3]
            cj = self.closure_joint[side]
            closure_deg = float(np.degrees(angles.get(cj, 0.0)))
            finger_joints = {n: round(float(angles[n]), 5) for n in ep.dof_names if n.startswith(("L_" if side == "left" else "R_"))}
            views = {}
            for n, c in cams.items():
                v = _view_entry(c, corners_world, center_world, pts)
                _add_mask_stats(v, id_maps[n], mask_id)
                views[n] = v
            grippers[side] = {
                "mask_id": mask_id,
                "palm_link": self.palm_link[side],
                "palm_position_world": _r(palm_T[:3, 3]),
                "palm_quat_wxyz": _r(matrix_to_quat_wxyz(palm_T[:3, :3]), 6),
                "wrist_position_world": _r(link_poses[f"{side}_arm_link07"][:3, 3]),
                "fingertips_world": {f: _r(link_poses[l][:3, 3]) for f, l in self.fingertip_links[side].items()},
                "box3d_center_world": _r(center_world),
                "box3d_size_m": _r(hi - lo),
                "box3d_corners_world": _r(corners_world),
                "closure_joint": cj,
                "closure_deg": round(closure_deg, 2),
                "closure_ratio": round(closure_deg / self.closure_closed, 3),
                "closed": bool(closure_deg >= 0.5 * self.closure_closed),
                "finger_joint_angles_rad": finger_joints,
                "views": views,
            }

        # ---- tracked objects --------------------------------------------
        objects = []
        for i, pp in enumerate(ep.prim_paths):
            entry = table.get(pp, {})
            p, q = pos[i], quat[i]
            obj = {
                "name": entry.get("name", pp.split("/")[-1]),
                "prim_path": pp,
                "mask_id": mask_id_of_prim.get(pp),
                "category": entry.get("category", "object"),
                "role": roles.get(pp, "distractor"),
                "active": active[i],
            }
            if active[i] and "local_aabb_min" in entry:
                R = quat_wxyz_to_matrix(q)
                corners_world = p + object_local_corners(entry) @ R.T
                center_world = corners_world.mean(axis=0)
                pts_world = p + points[pp] @ R.T if pp in points else None
                views = {}
                invisible_in_sim = renderer is not None and obj["mask_id"] is None  # e.g. Task3 holder plates: visibility=invisible in the USD
                obj["visible_in_sim"] = not invisible_in_sim
                for n, c in cams.items():
                    v = _view_entry(c, corners_world, center_world, pts_world)
                    if invisible_in_sim:
                        v["visible"] = False
                        v["visible_px"] = 0
                        v["visibility_source"] = "invisible_in_sim"
                    elif obj["mask_id"]:
                        _add_mask_stats(v, id_maps[n], obj["mask_id"])
                    else:
                        _add_mask_stats(v, None, 0)
                    views[n] = v
                obj.update({
                    "position_world": _r(p),
                    "quat_wxyz": _r(q, 6),
                    "linear_velocity": _r(vel[i]),
                    "size_m": _r(entry["size_m"], 4),
                    "box3d_center_world": _r(center_world),
                    "box3d_corners_world": _r(corners_world),
                    "dist_to_left_palm_m": round(float(np.linalg.norm(center_world - palm_pos["left"])), 4),
                    "dist_to_right_palm_m": round(float(np.linalg.norm(center_world - palm_pos["right"])), 4),
                    "views": views,
                })
            else:
                obj["views"] = {n: {"visible": False, "in_frustum": False} for n in cams}
                obj["reason"] = "inactive_or_out_of_scene" if not active[i] else "no_bounds"
            objects.append(obj)

        # ---- every other (static) scene object in some camera --------------
        scene_objects = []
        if renderer is not None:
            tracked_ids = set(mask_id_of_prim.values())
            counts = {n: np.bincount(id_maps[n].ravel(), minlength=max(o.obj_id for o in renderer.objects) + 1) for n in cams}
            static_iter = [(o, renderer.static_points.get(o.obj_id)) for o in renderer.objects if o.obj_id not in tracked_ids]
        else:
            static_iter = self.static_scene(ep.task_name)
        for o, pts_world in static_iter:
            lo, hi = np.asarray(o.aabb_world_min), np.asarray(o.aabb_world_max)
            corners_world = aabb_corners(np.stack([lo, hi]))
            center_world = (lo + hi) / 2
            views = {}
            for n, c in cams.items():
                if renderer is not None:
                    if not (o.obj_id < len(counts[n]) and counts[n][o.obj_id] >= MIN_VISIBLE_PX):
                        continue
                    v = _view_entry(c, corners_world, center_world, pts_world)
                    _add_mask_stats(v, id_maps[n], o.obj_id)
                else:
                    v = _view_entry(c, corners_world, center_world, pts_world)
                    if not v["in_frustum"]:
                        continue
                    _add_mask_stats(v, None, o.obj_id)
                views[n] = v
            if views:
                scene_objects.append({
                    "mask_id": o.obj_id, "name": o.name, "category": o.category, "background": o.background,
                    "prim_path": o.prim_path, "static": True,
                    "position_world": _r(center_world), "size_m": _r(hi - lo),
                    "box3d_corners_world": _r(corners_world),
                    "dist_to_left_palm_m": round(float(np.linalg.norm(center_world - palm_pos["left"])), 4),
                    "dist_to_right_palm_m": round(float(np.linalg.norm(center_world - palm_pos["right"])), 4),
                    "views": views,
                })

        rec = {
            "step": int(t),
            "success": ep.step_success(t),
            "instruction": ep.active_task(t).get("task_name"),
            "cameras": {n: {"position_world": _r(c.position), "quat_wxyz_world": _r(matrix_to_quat_wxyz(c.rotation), 6)} for n, c in cams.items()},
            "robot": {
                "root_position": _r(root_p),
                "root_quat_wxyz": _r(root_q, 6),
                "joint_positions": _r(joints, 5),
                "state26": _r(ep.state26(t), 5),
            },
            "grippers": grippers,
            "objects": objects,
            "scene_objects": scene_objects,
        }
        return rec, id_maps

    # ------------------------------------------------------------------
    def annotate_episode(self, episode_dir, out_dir, review_every: int = 0, review_video: bool = False, steps=None, review_count: int = 0) -> Path:
        import h5py

        from .viz import draw_step, write_video

        ep = Episode(episode_dir)
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        cams = self.make_cameras(ep)
        h, w = ep.image_shape("head_camera")
        renderer = self.renderer(ep.task_name, w, h)
        roles = self.object_roles(ep)
        meta = self.episode_meta(ep, cams, renderer)
        (out / "episode_meta.json").write_text(json.dumps(meta, indent=2))
        frames = []
        step_range = range(ep.num_steps) if steps is None else [t for t in steps if 0 <= t < ep.num_steps]
        review_steps = set(review_steps_for(len(step_range), review_count)) if review_count else set()
        if review_every:
            review_steps |= {t for t in step_range if t % review_every == 0}
        masks_h5 = None
        if renderer is not None:
            masks_h5 = h5py.File(out / "masks.h5", "w")
            dset = masks_h5.create_dataset("id_map", shape=(ep.num_steps, len(CAMERAS), h, w), dtype=np.uint16,
                                           chunks=(1, len(CAMERAS), h, w), compression="gzip", compression_opts=4)
            masks_h5.attrs["cameras"] = json.dumps(CAMERAS)
        import time as _time

        t_start = _time.time()
        t_step = 0.0
        with open(out / "annotations.jsonl", "w") as fh:
            for t in step_range:
                _t0 = _time.time()
                rec, id_maps = self.annotate_step(ep, t, cams, roles, renderer)
                t_step += _time.time() - _t0
                fh.write(json.dumps(rec) + "\n")
                if masks_h5 is not None:
                    dset[t] = np.stack([id_maps[n] for n in CAMERAS]).astype(np.uint16)
                want_img = (t in review_steps) or review_video
                if want_img:
                    imgs = {n: ep.image(n, t) for n in CAMERAS}
                    canvas = draw_step(imgs, rec, meta, id_maps=id_maps)
                    if t in review_steps:
                        (out / "review").mkdir(exist_ok=True)
                        cv2.imwrite(str(out / "review" / f"step_{t:06d}.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    if review_video:
                        frames.append(canvas)
        if masks_h5 is not None:
            masks_h5.close()
        if review_video and frames:
            write_video(out / "review.mp4", frames, fps=10)
        n = max(1, len(step_range))
        print(f"[timing] {ep.task_name} {ep.dir.name}: {n} steps, annotate_step {t_step / n:.3f} s/step, total {(_time.time() - t_start) / n:.3f} s/step", flush=True)
        ep.close()
        return out
