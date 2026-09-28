"""
Module 2: Classical numeric IK baseline (Damped Least Squares), used to
benchmark the PINN solver on both accuracy and behavior near singularities.

Position-only DLS: uses the linear-velocity (top 3 rows) block of the
geometric Jacobian, matching the position-only target space in the roadmap.
"""
import torch

from . import robot


def solve_dls(target_pos, theta_init, max_iters=150, damping=0.05, tol=1e-4, step_scale=1.0,
              adaptive=True, w_thresh=0.05, lambda_max=0.15):
    """
    target_pos: (B, 3), theta_init: (B, 7).
    Returns final theta (B, 7).

    adaptive=True uses Nakamura & Hanafusa's variable-damping DLS: damping
    is 0 away from singularities and ramps up smoothly as manipulability
    w(theta) drops below w_thresh, capped at lambda_max. This keeps the
    baseline from either (a) oscillating/failing near singularities with a
    fixed small damping, or (b) being sluggish everywhere with a fixed large
    one -- it's the standard fix, used here so the classical baseline isn't
    a strawman in PINN-vs-classical comparisons.
    """
    theta = theta_init.clone()
    B = theta.shape[0]
    I3 = torch.eye(3, device=theta.device, dtype=theta.dtype)

    for _ in range(max_iters):
        pos, _rot = robot.forward_kinematics(theta)
        err = target_pos - pos  # (B, 3)
        if torch.max(torch.linalg.norm(err, dim=-1)) < tol:
            break

        J = robot.geometric_jacobian(theta)[:, :3, :]  # (B, 3, 7)

        if adaptive:
            w = robot.manipulability(theta, position_only=True)  # (B,)
            ratio = torch.clamp(1.0 - w / w_thresh, min=0.0)
            lambda_sq = (lambda_max * ratio) ** 2  # (B,)
            damping_sq_I = lambda_sq.view(B, 1, 1) * I3
        else:
            damping_sq_I = damping ** 2 * I3

        JJt = torch.bmm(J, J.transpose(1, 2)) + damping_sq_I  # (B, 3, 3)
        JJt_inv_err = torch.linalg.solve(JJt, err.unsqueeze(-1))  # (B, 3, 1)
        d_theta = step_scale * torch.bmm(J.transpose(1, 2), JJt_inv_err).squeeze(-1)  # (B, 7)

        theta = robot.clamp_to_limits(theta + d_theta)

    return theta
