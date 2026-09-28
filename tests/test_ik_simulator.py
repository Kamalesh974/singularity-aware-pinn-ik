"""
Tests for the interactive simulator's non-GUI logic (run: py tests/test_ik_simulator.py).
The GUI itself is exercised end to end by:  py -m ik_simulator --selftest some_folder
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from pinn_ik import robot, robot_config as cfg
from ik_simulator import settings, trajectory
from ik_simulator.kinematics import ee_position, ee_positions, joint_limits
from ik_simulator.pinn_solver import PINNSolver
from ik_simulator.workspace import WorkspaceModel


def test_numpy_fk_matches_pytorch_fk():
    rng = np.random.default_rng(0)
    lim = joint_limits()
    Q = rng.uniform(lim[:, 0], lim[:, 1], size=(200, cfg.N_JOINTS))
    ref, _ = robot.forward_kinematics(torch.tensor(Q, dtype=torch.float64))
    single = np.array([ee_position(q) for q in Q])
    batch = ee_positions(Q)
    assert np.abs(single - ref.numpy()).max() < 1e-6
    assert np.abs(batch - ref.numpy()).max() < 1e-6
    print("OK: numpy forward kinematics (used for animation) matches the PINN's torch FK")


def test_trajectory_is_smooth_and_exact():
    q0 = np.array(cfg.HOME_POSITION)
    q1 = q0 + np.array([0.5, -0.3, 0.8, -0.6, 0.4, 0.9, -1.0])
    for profile in trajectory.PROFILES:
        tr = trajectory.plan(q0, q1, duration=2.0, max_speed=5.0, profile=profile)
        t, Q = tr.sample(60)
        assert np.allclose(Q[0], q0) and np.allclose(Q[-1], q1), "must start and end exactly on the requested poses"
        assert np.allclose(tr.velocity(0.0), 0) and np.allclose(tr.velocity(tr.duration), 0), "zero velocity at both ends"
        step = np.abs(np.diff(Q, axis=0)).max()
        assert step < 0.3 * np.abs(q1 - q0).max(), "no sudden jumps between animation frames"
        assert np.all(np.diff(Q[:, 0]) * np.sign(q1[0] - q0[0]) >= -1e-12), "joint moves monotonically (no overshoot)"
    slow = trajectory.plan(q0, q1, duration=0.5, max_speed=1.0)
    assert slow.peak_speed <= 1.0 + 1e-9, "duration is stretched so the joint speed limit is respected"
    print("OK: trajectories start/end exactly, are smooth, and respect the speed limit")


def test_workspace_reachability_messages():
    ws = WorkspaceModel(n_samples=60_000)
    ws.build()
    assert ws.check(ee_position(np.array(cfg.HOME_POSITION))).reachable
    far = ws.check([3.0, 0.0, 1.0])
    assert not far.reachable and "maximum reach" in far.message
    floor = ws.check([0.4, 0.0, -0.2])
    assert not floor.reachable and "floor" in floor.message
    ws.check([0.0, 0.0, 0.0])  # awkward point (the base itself): must return a verdict, not raise
    rng = np.random.default_rng(1)
    assert all(ws.check(ws.random_reachable_point(rng)).reachable for _ in range(50))
    print("OK: reachability check accepts real workspace points and explains why it rejects the others")


def test_solver_never_fabricates_solutions():
    s = PINNSolver(settings.DEFAULT_CHECKPOINT)
    q_home = np.array(cfg.HOME_POSITION)
    target = ee_position(np.array([0.3, -0.4, 0.2, -1.8, 0.1, 1.9, 0.5]))

    res = s.solve(target, q_home)
    lim = joint_limits()
    assert 1 <= len(res.solutions) <= settings.MAX_SOLUTIONS
    for sol in res.solutions:
        assert np.linalg.norm(ee_position(sol.q) - target) * 1000 <= res.tolerance_mm + 1e-6, "independent numpy FK agrees"
        assert np.all(sol.q >= lim[:, 0] - 1e-6) and np.all(sol.q <= lim[:, 1] + 1e-6)
    for a in range(len(res.solutions)):
        for b in range(a):
            gap = np.abs(res.solutions[a].q - res.solutions[b].q).max()
            assert gap >= settings.MIN_SOLUTION_SEPARATION_RAD - 1e-9, "solutions must be genuinely different"

    impossible = s.solve(target, q_home, tol_mm=1e-4)          # no PINN is that exact
    assert len(impossible.solutions) == 0 and "No valid IK solution" in impossible.message
    two = s.solve(target, q_home, max_solutions=2)
    assert len(two.solutions) <= 2
    print(f"OK: solver returned {len(res.solutions)} verified solutions, 0 for an impossible tolerance, and never pads")


def test_wrong_robot_is_detected():
    original = list(cfg.DH_PARAMETERS)
    try:
        cfg.DH_PARAMETERS[3] = (0.1, original[3][1], original[3][2], original[3][3])
        s = PINNSolver(settings.DEFAULT_CHECKPOINT)
        assert not s.robot_matches and "different robot" in s.warning
    finally:
        cfg.DH_PARAMETERS[:] = original
    assert PINNSolver(settings.DEFAULT_CHECKPOINT).robot_matches
    print("OK: changing the DH table makes the app warn that the saved PINN was trained for another robot")


if __name__ == "__main__":
    test_numpy_fk_matches_pytorch_fk()
    test_trajectory_is_smooth_and_exact()
    test_workspace_reachability_messages()
    test_solver_never_fabricates_solutions()
    test_wrong_robot_is_detected()
    print("All simulator tests passed.")
