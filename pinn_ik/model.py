"""
Module 3: PINN architecture.

Seed-conditioned residual network (fix for IK non-uniqueness on a redundant
arm): it takes the current joint angles as input and predicts a small step,

    theta_next = theta + f(target_pos, theta, ...)

so for a given (target, current pose) there is one "nearest" branch to pick.

feedback=True additionally feeds the network the hand position FK(theta) and the
remaining error (target - FK(theta)), computed with the differentiable forward
kinematics. Without it the network has to learn the arm's forward kinematics
implicitly just to know how far off it is, which is what capped accuracy at
~40 mm. With it the task becomes "turn this remaining error into a joint step"
(a learned Newton/DLS-like update), and the network can be applied for several
passes (`solve`), each pass refining the last.
"""
import torch
import torch.nn as nn

from . import robot


class ConditionedIKNet(nn.Module):
    def __init__(self, hidden=256, n_layers=4, max_step=0.6, feedback=False):
        super().__init__()
        self.feedback = feedback
        self.passes = 1  # default number of refinement passes used by solve()
        in_dim = 3 + robot.N_JOINTS + (6 if feedback else 0)
        layers = [nn.Linear(in_dim, hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, robot.N_JOINTS)]
        self.net = nn.Sequential(*layers)
        self.max_step = max_step  # bounds the step (rad) per joint per pass

    def forward(self, target_pos, theta, pos=None):
        """One pass. `pos` = FK(theta) if the caller already has it (saves a recompute)."""
        feats = [target_pos, theta]
        if self.feedback:
            if pos is None:
                pos, _ = robot.forward_kinematics(theta)
            feats += [pos, (target_pos - pos) * 10.0]
        delta = torch.tanh(self.net(torch.cat(feats, dim=-1))) * self.max_step
        return theta + delta

    @torch.no_grad()
    def solve(self, target_pos, theta_seed, passes=None):
        """Inference: apply the network `passes` times, clamping to the joint limits each time."""
        theta = theta_seed
        for _ in range(passes or self.passes):
            theta = robot.clamp_to_limits(self(target_pos, theta))
        return theta


def load_model(path, device="cpu"):
    """Load a checkpoint written by train.train (older ones lack a config block)."""
    ck = torch.load(path, map_location=device)
    cfg = ck.get("config", {})
    model = ConditionedIKNet(hidden=cfg.get("hidden", 256), n_layers=cfg.get("n_layers", 4),
                             max_step=cfg.get("max_step", 0.6), feedback=cfg.get("feedback", False)).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.passes = cfg.get("unroll", 1)
    model.config = cfg
    model.dh_fingerprint = cfg.get("dh_fingerprint")  # None for checkpoints from before fingerprints existed
    model.eval()
    return model
