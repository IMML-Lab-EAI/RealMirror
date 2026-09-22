from pathlib import Path
import numpy as np
import pytest
from realmirror_annot.kinematics import URDFKinematics, aabb_corners

URDF = Path("data/realmirror-asset/robot/A2_t2d0_flagship_opensource_urdf/urdf/raise_a2_t2d0_flagship/model_no_col.urdf")
MESH_ROOT = Path("data/realmirror-asset/robot/A2_t2d0_flagship_opensource_urdf")

pytestmark = pytest.mark.skipif(not URDF.exists(), reason="asset not downloaded")


def test_zero_pose_matches_usd_link07_position():
    kin = URDFKinematics(URDF)
    poses = kin.link_poses({})
    assert np.allclose(poses["left_arm_link07"][:3, 3], [-0.0278, 0.7005, 0.3421], atol=2e-3)
    assert np.allclose(poses["right_arm_link07"][:3, 3], [-0.0278, -0.7005, 0.3421], atol=2e-3)


def test_zero_pose_matches_usd_link07_rotation():
    kin = URDFKinematics(URDF)
    R = kin.link_poses({})["left_arm_link07"][:3, :3]
    # USD row-vector matrix rows were [[0,-1,0],[0,0,1],[-1,0,0]] -> columns of R
    assert np.allclose(R[:, 0], [0, -1, 0], atol=1e-3) and np.allclose(R[:, 1], [0, 0, 1], atol=1e-3)


def test_moving_arm_joint_moves_hand():
    kin = URDFKinematics(URDF)
    p0 = kin.link_poses({})["left_hand"][:3, 3]
    p1 = kin.link_poses({"idx14_left_arm_joint2": 1.2})["left_hand"][:3, 3]
    assert np.linalg.norm(p1 - p0) > 0.05


def test_hand_aabb_is_hand_sized():
    kin = URDFKinematics(URDF)
    box = kin.link_local_aabb("left_hand", MESH_ROOT)
    size = box[1] - box[0]
    assert (size > 0.01).all() and (size < 0.3).all()
    assert aabb_corners(box).shape == (8, 3)
