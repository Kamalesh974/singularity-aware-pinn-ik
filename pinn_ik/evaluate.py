"""
Module 4/5 support: compare the trained PINN against the classical DLS
baseline specifically in low-manipulability (near-singular) regions -- the
scenario the whole project is motivated by.

Rather than relying on hand-recalled "known" Panda singular configurations
(risky to assert from memory, see robot.py's DH caveat), near-singular seed
configurations are found empirically: gradient descent directly on the
manipulability index w(theta) from random starts, picking the lowest-w
local minima found. This only depends on the FK/Jacobian implementation
already cross-validated against autograd in test_kinematics.py, not on any
externally-asserted fact about the real robot.
"""
import torch

from . import robot, baseline_ik


def find_near_singular_configs(n, iters=300, lr=0.05, restarts_per_config=4, device=None):
    """
    For each of n target configs, run `restarts_per_config` independent
    gradient-descent minimizations of w(theta) from random starts and keep
    the lowest-w result. Returns theta (n, 7) and w (n,).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    limits = robot.JOINT_LIMITS.to(device)

    best_theta = torch.zeros(n, robot.N_JOINTS, device=device)
    best_w = torch.full((n,), float("inf"), device=device)

    for _ in range(restarts_per_config):
        theta = robot.sample_random_joints(n, limits, device).requires_grad_(True)
        opt = torch.optim.Adam([theta], lr=lr)
        for _ in range(iters):
            opt.zero_grad()
            w = robot.manipulability(theta)
            loss = w.sum()  # minimize manipulability (drive toward singularity)
            loss.backward()
            opt.step()
            with torch.no_grad():
                theta.data = robot.clamp_to_limits(theta.data, limits)

        with torch.no_grad():
            w_final = robot.manipulability(theta)
            improve = w_final < best_w
            best_w = torch.where(improve, w_final, best_w)
            best_theta = torch.where(improve.unsqueeze(-1), theta.detach(), best_theta)

    return best_theta, best_w


def compare_near_singular(model, n=30, local_max_step=0.3, device=None, dls_max_iters=200, seed=0):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    theta_seed, w_seed = find_near_singular_configs(n, device=device)
    print(f"constructed {n} near-singular seeds, mean w = {w_seed.mean().item():.5f}, "
          f"min w = {w_seed.min().item():.5f}, max w = {w_seed.max().item():.5f}")

    # Small local target motions from each near-singular seed -- the exact
    # scenario where classical J^-1-based methods are prone to violent
    # joint-velocity spikes.
    delta = (torch.rand(n, robot.N_JOINTS, device=device) * 2 - 1) * local_max_step
    theta_target = robot.clamp_to_limits(theta_seed + delta)
    target_pos, _ = robot.forward_kinematics(theta_target)

    model.eval()
    with torch.no_grad():
        theta_pinn = model.solve(target_pos, theta_seed)
        pos_pinn, _ = robot.forward_kinematics(theta_pinn)
        err_pinn_mm = torch.linalg.norm(pos_pinn - target_pos, dim=-1) * 1000
        w_pinn = robot.manipulability(theta_pinn)
        step_pinn = torch.linalg.norm(theta_pinn - theta_seed, dim=-1)

    theta_dls = baseline_ik.solve_dls(target_pos, theta_seed.clone(), max_iters=dls_max_iters, adaptive=True)
    pos_dls, _ = robot.forward_kinematics(theta_dls)
    err_dls_mm = torch.linalg.norm(pos_dls - target_pos, dim=-1) * 1000
    w_dls = robot.manipulability(theta_dls)
    step_dls = torch.linalg.norm(theta_dls - theta_seed, dim=-1)

    def summarize(name, err_mm, w, step):
        print(f"\n{name}:")
        print(f"  position error (mm): mean {err_mm.mean().item():7.2f}  median {err_mm.median().item():7.2f}  max {err_mm.max().item():7.2f}")
        print(f"  resulting manipulability w: mean {w.mean().item():.5f}  min {w.min().item():.5f}")
        print(f"  joint-space step norm (rad): mean {step.mean().item():.4f}  max {step.max().item():.4f}")
        n_fail = (err_mm > 5.0).sum().item()
        print(f"  samples with >5mm error: {n_fail}/{len(err_mm)}")

    summarize("PINN (seed-conditioned)", err_pinn_mm, w_pinn, step_pinn)
    summarize("Classical DLS (adaptive damping)", err_dls_mm, w_dls, step_dls)

    return {
        "pinn": {"err_mm": err_pinn_mm, "w": w_pinn, "step": step_pinn},
        "dls": {"err_mm": err_dls_mm, "w": w_dls, "step": step_dls},
        "seed_w": w_seed,
    }


def local_task_accuracy(model, n=3000, local_max_step=0.3, device=None, seed=1):
    """
    Accuracy on tasks drawn like the training data: random start pose, target =
    hand position after a random small joint move. Returns mean/median/p95 error (mm).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    seed_q = robot.sample_random_joints(n, device=device)
    target_q = robot.clamp_to_limits(seed_q + (torch.rand(n, robot.N_JOINTS, device=device) * 2 - 1) * local_max_step)
    target, _ = robot.forward_kinematics(target_q)
    model.eval()
    with torch.no_grad():
        pos, _ = robot.forward_kinematics(model.solve(target, seed_q))
        err = torch.linalg.norm(pos - target, dim=-1) * 1000
    out = {"mean": err.mean().item(), "median": err.median().item(), "p95": err.quantile(0.95).item()}
    print(f"\nAccuracy on {n} training-style tasks: mean {out['mean']:.2f} mm | "
          f"median {out['median']:.2f} mm | 95th percentile {out['p95']:.2f} mm")
    return out


if __name__ == "__main__":
    import argparse
    from .model import load_model

    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--n", type=int, default=30)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.checkpoint, device)
    compare_near_singular(model, n=args.n, device=device)
