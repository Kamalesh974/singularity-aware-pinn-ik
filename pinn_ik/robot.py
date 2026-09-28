"""
Module 1: Differentiable Forward Kinematics + Jacobian for a 7-DOF redundant
manipulator (Franka Emika Panda), built on modified (Craig) DH parameters.

Everything here is batched torch and differentiable end-to-end, so autograd
can backprop through FK straight into a network's joint-angle predictions
(this is what makes the position loss self-supervised -- no labeled
(theta, x) pairs are needed, only sampled target positions).

NOTE ON THE DH TABLE: the values below are the commonly published modified-DH
parameters for the Panda used across several robotics papers. Cross-check
against Franka Emika's official URDF/documentation before trusting them for
anything beyond simulation -- treat this as "known to be self-consistent"
(verified against autograd in tests/test_kinematics.py), not "verified
against the physical robot."
"""
import torch

from . import robot_config as _cfg

N_JOINTS = _cfg.N_JOINTS

# All robot numbers come from pinn_ik/robot_config.py (modified DH: a_{i-1}, alpha_{i-1}, d_i,
# plus a constant theta offset). theta_i is the free joint variable, supplied at call time.
_A = torch.tensor([row[0] for row in _cfg.DH_PARAMETERS])
_ALPHA = torch.tensor([row[1] for row in _cfg.DH_PARAMETERS])
_D = torch.tensor([row[2] for row in _cfg.DH_PARAMETERS])
_THETA_OFFSET = torch.tensor(_cfg.joint_offsets())

# Fixed flange offset from joint 7's frame to the end-effector point.
_FLANGE_D = _cfg.TOOL_OFFSET_Z

JOINT_LIMITS = torch.tensor(_cfg.JOINT_LIMITS_RAD)


def _modified_dh_transform(alpha, a, theta, d):
    """Batched modified-DH transform. theta: (B,), alpha/a/d: scalars."""
    B = theta.shape[0]
    device, dtype = theta.device, theta.dtype
    cos_t, sin_t = torch.cos(theta), torch.sin(theta)
    cos_a = torch.cos(torch.as_tensor(alpha, device=device, dtype=dtype))
    sin_a = torch.sin(torch.as_tensor(alpha, device=device, dtype=dtype))

    T = torch.zeros(B, 4, 4, device=device, dtype=dtype)
    T[:, 0, 0] = cos_t
    T[:, 0, 1] = -sin_t
    T[:, 0, 3] = a
    T[:, 1, 0] = sin_t * cos_a
    T[:, 1, 1] = cos_t * cos_a
    T[:, 1, 2] = -sin_a
    T[:, 1, 3] = -sin_a * d
    T[:, 2, 0] = sin_t * sin_a
    T[:, 2, 1] = cos_t * sin_a
    T[:, 2, 2] = cos_a
    T[:, 2, 3] = cos_a * d
    T[:, 3, 3] = 1.0
    return T


def forward_kinematics(theta, return_chain=False):
    """
    theta: (B, 7) joint angles.
    Returns end-effector position (B, 3) and rotation matrix (B, 3, 3).
    If return_chain=True, also returns the per-joint origins (B, 7, 3) and
    z-axes (B, 7, 3) needed for the geometric Jacobian.
    """
    B = theta.shape[0]
    device, dtype = theta.device, theta.dtype
    T = torch.eye(4, device=device, dtype=dtype).expand(B, 4, 4).contiguous()

    # Modified (Craig) DH: theta_i is the rotation about z_i (the OUTGOING
    # frame's z-axis), not z_{i-1}. So each joint's axis/origin for the
    # Jacobian must be captured AFTER that joint's own transform is applied,
    # not before -- capturing before is a classic off-by-one for this
    # convention (verified against autograd in tests/test_kinematics.py).
    origins = []
    z_axes = []
    for i in range(N_JOINTS):
        Ti = _modified_dh_transform(_ALPHA[i].item(), _A[i].item(), theta[:, i] + _THETA_OFFSET[i].item(),
                                    _D[i].item())
        T = torch.bmm(T, Ti)
        origins.append(T[:, :3, 3])
        z_axes.append(T[:, :3, 2])

    # Fixed flange offset along the last frame's z-axis.
    T_flange = torch.eye(4, device=device, dtype=dtype).expand(B, 4, 4).contiguous()
    T_flange = T_flange.clone()
    T_flange[:, 2, 3] = _FLANGE_D
    T = torch.bmm(T, T_flange)

    pos = T[:, :3, 3]
    rot = T[:, :3, :3]

    if return_chain:
        origins = torch.stack(origins, dim=1)  # (B, 7, 3)
        z_axes = torch.stack(z_axes, dim=1)     # (B, 7, 3)
        return pos, rot, origins, z_axes
    return pos, rot


def geometric_jacobian(theta):
    """
    Analytic geometric Jacobian (B, 6, 7) for revolute joints:
      J_v_i = z_{i-1} x (p_e - p_{i-1})
      J_w_i = z_{i-1}
    Differentiable (built from the same autograd-tracked FK chain).
    """
    pos_e, _rot_e, origins, z_axes = forward_kinematics(theta, return_chain=True)
    p_diff = pos_e.unsqueeze(1) - origins       # (B, 7, 3)
    Jv = torch.cross(z_axes, p_diff, dim=-1)    # (B, 7, 3)
    Jw = z_axes                                 # (B, 7, 3)
    J = torch.cat([Jv, Jw], dim=-1)             # (B, 7, 6)
    return J.transpose(1, 2)                    # (B, 6, 7)


def manipulability(theta, jacobian=None, position_only=True):
    """
    Yoshikawa manipulability index w(theta) = product of singular values of J.
    Computed via SVD (numerically stabler near-singularity than det(J J^T)).
    position_only=True uses the 3x7 linear-velocity block, matching a
    position-only IK target space.
    """
    J = jacobian if jacobian is not None else geometric_jacobian(theta)
    J = J[:, :3, :] if position_only else J
    s = torch.linalg.svdvals(J)
    return torch.prod(s, dim=-1)


def clamp_to_limits(theta, limits=None):
    limits = JOINT_LIMITS if limits is None else limits
    limits = limits.to(theta.device, theta.dtype)
    return torch.max(torch.min(theta, limits[:, 1]), limits[:, 0])


def sample_random_joints(batch_size, limits=None, device=None, dtype=torch.float32):
    limits = JOINT_LIMITS if limits is None else limits
    limits = limits.to(device, dtype)
    u = torch.rand(batch_size, N_JOINTS, device=device, dtype=dtype)
    return limits[:, 0] + u * (limits[:, 1] - limits[:, 0])
