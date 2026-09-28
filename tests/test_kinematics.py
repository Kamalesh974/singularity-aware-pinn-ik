"""
Sanity checks for Module 1. The most important one is
test_geometric_jacobian_matches_autograd: it cross-validates the analytic
geometric Jacobian against torch.autograd.functional.jacobian applied to the
FK position function. If these two independently-derived Jacobians agree to
numerical precision, that's strong evidence both the FK chain and the
geometric-Jacobian construction are implemented correctly -- independent of
whether the specific DH numbers match the physical Panda exactly.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from pinn_ik import robot


def test_geometric_jacobian_matches_autograd():
    torch.manual_seed(0)
    B = 5
    theta = robot.sample_random_joints(B)

    J_analytic = robot.geometric_jacobian(theta)  # (B, 6, 7)

    def fk_pos_flat(theta_flat):
        # single-sample position function for autograd.functional.jacobian
        pos, _ = robot.forward_kinematics(theta_flat.unsqueeze(0))
        return pos.squeeze(0)

    for b in range(B):
        J_auto_pos = torch.autograd.functional.jacobian(fk_pos_flat, theta[b])  # (3, 7)
        J_analytic_pos = J_analytic[b, :3, :]
        assert torch.allclose(J_auto_pos, J_analytic_pos, atol=1e-5), \
            f"Jacobian mismatch at sample {b}: max diff {(J_auto_pos - J_analytic_pos).abs().max().item()}"

    print("OK: analytic geometric Jacobian matches autograd Jacobian")


def test_manipulability_nonnegative_and_finite():
    theta = robot.sample_random_joints(50)
    w = robot.manipulability(theta)
    assert torch.all(w >= 0)
    assert torch.all(torch.isfinite(w))
    print("OK: manipulability values are non-negative and finite")


def test_reachable_position_within_arm_extent():
    theta = robot.sample_random_joints(50)
    pos, _ = robot.forward_kinematics(theta)
    dist_from_base = torch.linalg.norm(pos, dim=-1)
    # Panda's max reach is ~0.855m; generous upper bound as a smoke check.
    assert torch.all(dist_from_base < 1.2), f"max dist {dist_from_base.max().item()}"
    print("OK: FK positions stay within a plausible arm-extent bound")


if __name__ == "__main__":
    test_geometric_jacobian_matches_autograd()
    test_manipulability_nonnegative_and_finite()
    test_reachable_position_within_arm_extent()
    print("All sanity checks passed.")
