import numpy as np
from realmirror_annot.camera import PinholeCamera, quat_wxyz_to_matrix, matrix_to_quat_wxyz, bbox_from_points


def make_head():
    cam = PinholeCamera("head", 256, 256, horizontal_aperture=31.0)
    cam.set_pose([0.0, 0.05, 1.4], [0.81915204, 0.0, 0.57357644, 0.0])
    return cam


def test_head_camera_looks_down_forward():
    cam = make_head()
    fwd = cam.rotation[:, 0]
    assert np.allclose(fwd, [0.342, 0.0, -0.94], atol=1e-3)


def test_point_on_optical_axis_projects_to_center():
    cam = make_head()
    p = cam.position + 1.0 * cam.rotation[:, 0]
    uv, d = cam.project(p)
    assert np.allclose(uv[0], [128, 128]) and np.isclose(d[0], 1.0)


def test_left_of_camera_maps_to_smaller_u_and_up_to_smaller_v():
    cam = make_head()
    p_left = cam.position + cam.rotation[:, 0] + 0.1 * cam.rotation[:, 1]
    p_up = cam.position + cam.rotation[:, 0] + 0.1 * cam.rotation[:, 2]
    (u_l, v_l), _ = cam.project(p_left)[0][0], None
    (u_u, v_u), _ = cam.project(p_up)[0][0], None
    assert u_l < 128 and np.isclose(v_l, 128)
    assert v_u < 128 and np.isclose(u_u, 128)


def test_behind_camera_is_nan():
    cam = make_head()
    uv, d = cam.project(cam.position - cam.rotation[:, 0])
    assert np.isnan(uv).all() and d[0] < 0


def test_quat_roundtrip():
    q = np.array([0.81915204, 0.0, 0.57357644, 0.0])
    R = quat_wxyz_to_matrix(q)
    assert np.allclose(matrix_to_quat_wxyz(R), q, atol=1e-6)


def test_bbox_from_points_clips_and_flags():
    uv = np.array([[-10.0, 5.0], [50.0, 60.0], [np.nan, np.nan]])
    depth = np.array([1.0, 1.0, -1.0])
    box, frac, in_frame = bbox_from_points(uv, depth, 256, 256)
    assert box == [0.0, 5.0, 50.0, 60.0] and np.isclose(frac, 2 / 3) and in_frame
