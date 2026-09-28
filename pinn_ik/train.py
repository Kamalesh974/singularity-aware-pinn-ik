"""
Module 4: Training loop.

Sampling strategy matters as much as the loss function here (see model.py
for why). theta_seed is uniformly randomized every batch, and target_pos is
FK of a small random perturbation of theta_seed -- so training already
covers the full reachable workspace (via the seed) while every example
stays a small step from a known pose, matching real trajectory tracking.

An earlier version of this file mixed in a minority of "global" samples
(target = FK of a fully unrelated random config, unrelated to theta_seed)
meant to add coverage of large reconfiguration moves. That was a bug: the
model's residual step is capped at max_step=0.6 rad/joint (~1.6 rad total
norm, see model.py) by design, but those global samples required joint-space
steps averaging ~5 rad -- architecturally unreachable in one step. They
produced a permanent loss floor (~110-120mm mean error) that no amount of
training or LR tuning could remove, since the model can never get those
samples right. Confirmed by breaking out local-only vs global-only eval
error (41mm vs 405mm) before removing them. local_max_step is kept well
inside max_step so every generated target stays solvable.

No labeled (theta, x) dataset is used anywhere -- targets are generated on
the fly via forward kinematics, and the position loss is purely
self-supervised through the differentiable FK chain.

BUG FOUND 2026-08-16: with theta_seed drawn uniformly at random, manipulability
w(theta) averages ~0.066 (median ~0.064) -- comfortably above the singularity
loss's default eps=0.02 threshold ~95% of the time. So the hinge barrier
relu(eps-w)^2 almost never activated during training (measured expected
loss ~2.8e-6, vs. position loss ~0.01-0.18): the singularity-avoidance term
was live in the code but contributed essentially no gradient signal. Fixed
two ways: (1) eps raised so the barrier engages at a realistic fraction of
configs, (2) a fraction of seeds are now deliberately biased toward
low-manipulability configs (via cheap oversample-and-filter, not an
expensive gradient search) so the model actually sees and has to learn to
handle near-singular starting poses, not just encounter them by accident.
"""
import argparse
import csv
import time
import torch

from . import robot, robot_config, losses
from .model import ConditionedIKNet


def sample_seeds(batch_size, singular_bias_fraction=0.3, oversample_factor=8, device=None, dtype=torch.float32):
    """
    Draw theta_seed with a fraction deliberately biased toward low
    manipulability: oversample random candidates, keep the lowest-w ones.
    Cheap (forward pass + sort only, no optimization loop) so it's fine to
    run every training step.
    """
    limits = robot.JOINT_LIMITS
    n_bias = int(batch_size * singular_bias_fraction)
    n_uniform = batch_size - n_bias

    parts = []
    if n_bias > 0:
        candidates = robot.sample_random_joints(n_bias * oversample_factor, limits, device, dtype)
        with torch.no_grad():
            w_cand = robot.manipulability(candidates)
        idx = torch.argsort(w_cand)[:n_bias]
        parts.append(candidates[idx])
    if n_uniform > 0:
        parts.append(robot.sample_random_joints(n_uniform, limits, device, dtype))

    theta_seed = torch.cat(parts, dim=0)
    return theta_seed[torch.randperm(batch_size, device=device)]


def sample_batch(batch_size, local_max_step=0.3, singular_bias_fraction=0.3, device=None, dtype=torch.float32):
    limits = robot.JOINT_LIMITS
    theta_seed = sample_seeds(batch_size, singular_bias_fraction, device=device, dtype=dtype)

    delta = (torch.rand(batch_size, robot.N_JOINTS, device=device, dtype=dtype) * 2 - 1) * local_max_step
    theta_target = robot.clamp_to_limits(theta_seed + delta, limits)

    target_pos, _ = robot.forward_kinematics(theta_target)
    return theta_seed, target_pos


@torch.no_grad()
def _mine_hard_tasks(model, batch_size, unroll, hard_fraction, oversample, singular_bias_fraction, device):
    """
    Learn from mistakes: try `oversample` x batch_size fresh tasks with the CURRENT model, keep the
    ones it gets most wrong (hard_fraction of the batch) plus random others so easy cases aren't forgotten.
    Returns seed poses, targets, and the mean error (mm) of the hard tasks it picked.
    """
    seed, tgt = sample_batch(batch_size * oversample, singular_bias_fraction=singular_bias_fraction, device=device)
    th = seed
    for _ in range(unroll):
        th = model(tgt, th)
    pos, _ = robot.forward_kinematics(th)
    err = torch.linalg.norm(pos - tgt, dim=-1)
    order = torch.argsort(err, descending=True)
    n_hard = int(batch_size * hard_fraction)
    hard, rest = order[:n_hard], order[n_hard:]
    rest = rest[torch.randperm(len(rest), device=device)[: batch_size - n_hard]]
    idx = torch.cat([hard, rest])
    return seed[idx], tgt[idx], err[hard].mean().item() * 1000


def train(steps=2000, batch_size=256, lr=1e-3, lr_min=None, device=None, log_every=200,
          eps=0.05, lambda_limits=1.0, lambda_singularity=5.0, singularity_mode="hinge",
          singular_bias_fraction=0.3, csv_path=None, checkpoint_path=None,
          hidden=256, n_layers=4, position_mode="sq", feedback=False, unroll=1,
          hard_fraction=0.0, mine_oversample=2):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = ConditionedIKNet(hidden=hidden, n_layers=n_layers, feedback=feedback).to(device)
    # later passes count more, so the final (refined) answer is what gets optimised hardest
    pass_weights = torch.tensor([2.0 ** k for k in range(unroll)], device=device)
    pass_weights = pass_weights / pass_weights.sum()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = None
    if lr_min is not None:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr_min)
    limits = robot.JOINT_LIMITS.to(device)

    history = []
    t_start = time.time()

    for step in range(1, steps + 1):
        if hard_fraction > 0:
            theta_seed, target_pos, hard_err = _mine_hard_tasks(
                model, batch_size, unroll, hard_fraction, mine_oversample, singular_bias_fraction, device)
        else:
            theta_seed, target_pos = sample_batch(batch_size, singular_bias_fraction=singular_bias_fraction, device=device)
            hard_err = 0.0

        # Unrolled refinement: apply the network `unroll` times, feeding each pass's output back
        # in, and grade every pass (the last one hardest). unroll=1 is the original single pass.
        theta, pred_pos = theta_seed, None
        l_pos = l_lim = 0.0
        for k in range(unroll):
            theta = model(target_pos, theta, pos=pred_pos)
            pred_pos, _ = robot.forward_kinematics(theta)
            l_pos = l_pos + pass_weights[k] * losses.position_loss(pred_pos, target_pos, mode=position_mode)
            l_lim = l_lim + losses.joint_limit_loss(theta, limits) / unroll
        theta_hat = theta
        w = robot.manipulability(theta_hat)
        l_sing = losses.singularity_loss(w, eps=eps, mode=singularity_mode)
        loss = l_pos + lambda_limits * l_lim + lambda_singularity * l_sing
        parts = {"position": l_pos.item(), "joint_limits": float(l_lim), "singularity": l_sing.item()}

        opt.zero_grad()
        loss.backward()
        opt.step()
        if scheduler is not None:
            scheduler.step()

        if step % log_every == 0 or step == 1 or step == steps:
            with torch.no_grad():
                pos_err_mm = torch.mean(torch.linalg.norm(pred_pos - target_pos, dim=-1)).item() * 1000
                mean_w = torch.mean(w).item()
            elapsed = time.time() - t_start
            cur_lr = opt.param_groups[0]["lr"]
            row = {"step": step, "elapsed_s": round(elapsed, 1), "loss": loss.item(),
                   "pos_err_mm": pos_err_mm, "loss_position": parts["position"],
                   "loss_joint_limits": parts["joint_limits"], "loss_singularity": parts["singularity"],
                   "mean_manipulability": mean_w, "lr": cur_lr, "hard_err_mm": hard_err}
            history.append(row)
            print(f"step {step:6d}/{steps} | {elapsed:6.1f}s | lr {cur_lr:.2e} | loss {loss.item():.5f} | pos_err {pos_err_mm:7.2f} mm "
                  f"| pos {parts['position']:.5f} | lim {parts['joint_limits']:.5f} "
                  f"| sing {parts['singularity']:.5f} | mean_w {mean_w:.4f}", flush=True)

    if csv_path:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)
        print(f"wrote training history to {csv_path}")

    if checkpoint_path:
        torch.save({"model_state_dict": model.state_dict(), "steps": steps, "batch_size": batch_size,
                    "config": {"hidden": hidden, "n_layers": n_layers, "max_step": model.max_step,
                               "position_mode": position_mode, "feedback": feedback, "unroll": unroll,
                               "lambda_singularity": lambda_singularity, "hard_fraction": hard_fraction,
                               "dh_fingerprint": robot_config.fingerprint()}},
                   checkpoint_path)
        print(f"wrote checkpoint to {checkpoint_path}")

    return model, history


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--csv-path", type=str, default=None)
    p.add_argument("--checkpoint-path", type=str, default=None)
    args = p.parse_args()
    train(steps=args.steps, batch_size=args.batch_size, lr=args.lr,
          csv_path=args.csv_path, checkpoint_path=args.checkpoint_path)
