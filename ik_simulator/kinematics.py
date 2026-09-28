"""
NumPy forward kinematics for the animation loop (one call per frame, so it must be light).

Reads the same pinn_ik/robot_config.py as the PINN's torch FK; tests/test_ik_simulator.py checks
the two agree to ~1e-6 m.
"""
import numpy as np

from pinn_ik import robot_config as cfg


def _dh(alpha, a, theta, d):
    ct, st, ca, sa = np.cos(theta), np.sin(theta), np.cos(alpha), np.sin(alpha)
    return np.array([[ct, -st, 0.0, a],
                     [st * ca, ct * ca, -sa, -sa * d],
                     [st * sa, ct * sa, ca, ca * d],
                     [0.0, 0.0, 0.0, 1.0]])


def frame_transforms(q):
    """
    World<-frame transforms for joints 1..N and the tool point, shape (N+1, 4, 4).
    Entry i (i < N) is the frame that joint i+1 rotates; the last entry is the tool/end-effector.
    """
    T = np.eye(4)
    frames = []
    for i, (a, alpha, d, offset) in enumerate(cfg.DH_PARAMETERS):
        T = T @ _dh(alpha, a, q[i] + offset, d)
        frames.append(T)
    tool = np.eye(4)
    tool[2, 3] = cfg.TOOL_OFFSET_Z
    frames.append(T @ tool)
    return np.array(frames)


def ee_position(q):
    """End-effector (tool point) position in the base frame, metres."""
    return frame_transforms(q)[-1][:3, 3]


def ee_positions(Q):
    """Vectorised ee_position for a whole trajectory: Q (M, N) -> (M, 3)."""
    Q = np.asarray(Q, dtype=float)
    M = Q.shape[0]
    T = np.tile(np.eye(4), (M, 1, 1))
    for i, (a, alpha, d, offset) in enumerate(cfg.DH_PARAMETERS):
        th = Q[:, i] + offset
        ct, st, ca, sa = np.cos(th), np.sin(th), np.cos(alpha), np.sin(alpha)
        Ti = np.zeros((M, 4, 4))
        Ti[:, 0, 0], Ti[:, 0, 1], Ti[:, 0, 3] = ct, -st, a
        Ti[:, 1, 0], Ti[:, 1, 1], Ti[:, 1, 2], Ti[:, 1, 3] = st * ca, ct * ca, -sa, -sa * d
        Ti[:, 2, 0], Ti[:, 2, 1], Ti[:, 2, 2], Ti[:, 2, 3] = st * sa, ct * sa, ca, ca * d
        Ti[:, 3, 3] = 1.0
        T = T @ Ti
    return T[:, :3, 3] + T[:, :3, 2] * cfg.TOOL_OFFSET_Z


def joint_limits():
    """(N, 2) array of joint limits in radians."""
    return np.array(cfg.JOINT_LIMITS_RAD, dtype=float)
