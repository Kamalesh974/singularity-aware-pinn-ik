"""
The MuJoCo scene is generated from the same DH table as robot.py. This checks
the two agree: for random joint angles, the end-effector position reported by
MuJoCo must equal robot.forward_kinematics (both arms of a 2-arm scene).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from pinn_ik import robot
from pinn_ik.mujoco_scene import ArmScene


def test_mujoco_matches_pytorch_fk():
    torch.manual_seed(0)
    scene = ArmScene(n_arms=2)
    worst = 0.0
    for _ in range(200):
        q = robot.sample_random_joints(1, dtype=torch.float64)
        ref, _ = robot.forward_kinematics(q)
        for arm in range(2):
            scene.set_joints(arm, q[0].numpy())
        scene.forward()
        for arm in range(2):
            worst = max(worst, float(np.abs(scene.ee_local(arm) - ref[0].numpy()).max()))
    assert worst < 1e-6, f"MuJoCo vs PyTorch FK mismatch: {worst:.3e} m"
    print(f"OK: MuJoCo end-effector matches PyTorch FK on 200 random poses x 2 arms (max diff {worst:.2e} m)")


if __name__ == "__main__":
    test_mujoco_matches_pytorch_fk()
