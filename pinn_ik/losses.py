"""
Module 3: Physics-constrained loss terms.

L_total = L_position + lambda1 * L_joint_limits + lambda2 * L_singularity

Singularity loss has two variants -- see the docstring on singularity_loss
for why the naive 1/(w+delta) barrier from the original roadmap is replaced
by a squared-hinge barrier as the default.
"""
import torch


def position_loss(pred_pos, target_pos, mode="sq"):
    """
    Position error, (B,3) vs (B,3).
    mode="sq": mean squared distance (gradient shrinks as the error shrinks).
    mode="l2": mean plain distance (constant-magnitude gradient, keeps
               pushing on small residual errors).
    """
    d2 = torch.sum((pred_pos - target_pos) ** 2, dim=-1)
    if mode == "sq":
        return torch.mean(d2)
    if mode == "l2":
        return torch.mean(torch.sqrt(d2 + 1e-8))
    raise ValueError(f"unknown position loss mode: {mode}")


def joint_limit_loss(theta, limits):
    """Quadratic penalty outside [min, max], zero inside."""
    limits = limits.to(theta.device, theta.dtype)
    below = torch.relu(limits[:, 0] - theta)
    above = torch.relu(theta - limits[:, 1])
    return torch.mean(below ** 2 + above ** 2)


def singularity_loss(w, eps=0.02, mode="hinge", delta=1e-3, w_floor=1e-3):
    """
    Penalize manipulability w(theta) dropping below threshold eps.

    mode="hinge" (default, recommended):
        L = mean( relu(eps - w)^2 )
        Zero when w >= eps (arm is comfortably away from a singularity).
        Below eps it grows quadratically in (eps - w), so the gradient is
        d/dw = -2*(eps-w), which is bounded and -> 0 exactly at w=0 instead
        of blowing up. This is the fix for the "gradient blow-up near
        w -> 0" concern: the original 1/(w+delta) barrier has gradient
        -1/(w+delta)^2, which diverges right in the region the network is
        supposed to learn to navigate, causing loss spikes / unstable
        training. The hinge trades that off against having zero gradient
        once w has already crossed eps and kept dropping toward 0 -- it
        stops pushing harder once you're deep in the singular region rather
        than pushing infinitely hard, which in practice is fine because the
        network reaches "already violated the margin" territory rarely if
        the margin is trained against consistently.

    mode="clamped_inverse":
        L = mean( 1 / (clamp(w, min=w_floor) + delta) )
        Keeps the classic inverse-barrier shape (always some repulsive
        gradient) but caps the maximum loss/gradient by flooring w before
        the divide, so it can't diverge. Use this if you want a
        nonzero gradient even deep inside a singular region (e.g. to help
        an optimizer escape rather than plateau), at the cost of a slightly
        less clean "zero when safe" property.
    """
    if mode == "hinge":
        return torch.mean(torch.relu(eps - w) ** 2)
    elif mode == "clamped_inverse":
        return torch.mean(1.0 / (torch.clamp(w, min=w_floor) + delta))
    else:
        raise ValueError(f"unknown mode: {mode}")


def total_loss(pred_pos, target_pos, theta_pred, limits, w,
               lambda_limits=1.0, lambda_singularity=1.0, singularity_kwargs=None,
               position_mode="sq"):
    singularity_kwargs = singularity_kwargs or {}
    l_pos = position_loss(pred_pos, target_pos, mode=position_mode)
    l_lim = joint_limit_loss(theta_pred, limits)
    l_sing = singularity_loss(w, **singularity_kwargs)
    total = l_pos + lambda_limits * l_lim + lambda_singularity * l_sing
    return total, {"position": l_pos.item(), "joint_limits": l_lim.item(), "singularity": l_sing.item()}
