"""
Physics-informed neural network inverse kinematics for a 7-DOF Franka Panda.

    py pinn_app.py train    --steps 20000 --hidden 512 --pos-loss l2 --out runs/mine.pt
    py pinn_app.py evaluate --checkpoint runs/final_model.pt
    py pinn_app.py solve    --target 0.45 0.10 0.60
    py pinn_app.py viz      --path circle                 # interactive MuJoCo window
    py pinn_app.py viz      --start singular --record singular.gif

All coordinates are metres in the robot's base frame (z up, shoulder at z = 0.333).
The network solves POSITION-only IK, one small step at a time from the current
pose, so `solve` walks the hand to a far-away target through short waypoints.
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from pinn_ik import baseline_ik, robot, viz_mujoco
from pinn_ik.model import load_model

MAX_REACH = 0.858  # measured: largest hand distance from the shoulder over 300k random poses


def cmd_train(a):
    from pinn_ik.train import train

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    train(steps=a.steps, batch_size=a.batch_size, lr=a.lr, lr_min=a.lr_min, hidden=a.hidden,
          n_layers=a.layers, position_mode=a.pos_loss, log_every=max(1, a.steps // 20),
          csv_path=a.out.replace(".pt", "_history.csv"), checkpoint_path=a.out)
    print(f"\nSaved {a.out}. Evaluate it with:  py pinn_app.py evaluate --checkpoint {a.out}")


def cmd_evaluate(a):
    from pinn_ik.evaluate import compare_near_singular, local_task_accuracy

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(a.checkpoint, device)
    local_task_accuracy(model, device=device)
    print()
    compare_near_singular(model, n=a.n, device=device)


def cmd_solve(a):
    model = load_model(a.checkpoint, "cpu")
    ready = torch.tensor([viz_mujoco.READY_POSE], dtype=torch.float32)
    if a.random_start:
        torch.manual_seed(a.seed)
        ready = robot.sample_random_joints(1)
    p0, _ = robot.forward_kinematics(ready)
    target = torch.tensor([a.target], dtype=torch.float32)

    shoulder = torch.tensor([[0.0, 0.0, robot._D[0].item()]])
    if (target - shoulder).norm() > MAX_REACH:
        print(f"WARNING: target is {float((target - shoulder).norm()):.3f} m from the shoulder, beyond the "
              f"arm's {MAX_REACH} m reach - expect a large error.")

    n_way = max(1, math.ceil(float((target - p0).norm()) / a.step))
    theta = ready
    with torch.no_grad():
        for k in range(1, n_way + 1):
            waypoint = p0 + (target - p0) * k / n_way
            theta = model.solve(waypoint, theta)
        pos, _ = robot.forward_kinematics(theta)
        w = robot.manipulability(theta)
    theta_dls = baseline_ik.solve_dls(target, ready, max_iters=200, adaptive=True)
    pos_d, _ = robot.forward_kinematics(theta_dls)
    w_d = robot.manipulability(theta_dls)

    deg = lambda t: [round(math.degrees(v), 1) for v in t[0].tolist()]
    print(f"start hand position : {[round(v, 3) for v in p0[0].tolist()]}")
    print(f"target              : {[round(v, 3) for v in target[0].tolist()]}   ({n_way} waypoint step(s))")
    print(f"\nPINN  joint angles (deg): {deg(theta)}")
    print(f"      reached {[round(v, 3) for v in pos[0].tolist()]}  error {float((pos - target).norm()) * 1000:.1f} mm  "
          f"manipulability {float(w):.4f}")
    print(f"DLS   joint angles (deg): {deg(theta_dls)}")
    print(f"      reached {[round(v, 3) for v in pos_d[0].tolist()]}  error {float((pos_d - target).norm()) * 1000:.1f} mm  "
          f"manipulability {float(w_d):.4f}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="train a new model")
    t.add_argument("--steps", type=int, default=20000)
    t.add_argument("--batch-size", type=int, default=2048)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--lr-min", type=float, default=1e-5)
    t.add_argument("--hidden", type=int, default=512)
    t.add_argument("--layers", type=int, default=4)
    t.add_argument("--pos-loss", choices=["sq", "l2"], default="l2")
    t.add_argument("--out", default="runs/my_model.pt")
    t.set_defaults(fn=cmd_train)

    e = sub.add_parser("evaluate", help="accuracy + near-singularity comparison against classical DLS")
    e.add_argument("--checkpoint", default="runs/final_model.pt")
    e.add_argument("--n", type=int, default=30)
    e.set_defaults(fn=cmd_evaluate)

    s = sub.add_parser("solve", help="solve IK for a target point you choose")
    s.add_argument("--checkpoint", default="runs/final_model.pt")
    s.add_argument("--target", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"))
    s.add_argument("--step", type=float, default=0.03, help="waypoint spacing in metres")
    s.add_argument("--random-start", action="store_true", help="start from a random pose instead of the ready pose")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_solve)

    v = sub.add_parser("viz", help="MuJoCo 3D tracking demo (PINN vs classical)")
    viz_mujoco.add_args(v)
    v.set_defaults(fn=viz_mujoco.run)

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
