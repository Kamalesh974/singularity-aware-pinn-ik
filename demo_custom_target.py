"""
Demo: ask the trained PINN IK model to reach a target point you choose.

Run directly (no -m needed):
    py demo_custom_target.py
    py demo_custom_target.py --dx 0.05 --dy 0.0 --dz 0.03
    py demo_custom_target.py --seed 7 --dx -0.05 --dy 0.05 --dz 0.0

The model needs two inputs: a starting pose (current joint angles) and a
target point. It predicts a small move from the starting pose toward the
target -- it was trained on small moves, not arbitrary long-range jumps, so
--dx/--dy/--dz should stay small (tens of centimeters at most) for accurate
results.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from pinn_ik import robot
from pinn_ik.model import load_model


def run_one(model, device, dx, dy, dz, label=""):
    # 1. Starting pose (7 joint angles) -- a random valid pose.
    theta_seed = robot.sample_random_joints(1, device=device)
    current_pos, _ = robot.forward_kinematics(theta_seed)

    # 2. Target point: current position plus the offset.
    offset = torch.tensor([[dx, dy, dz]], device=device)
    target = current_pos + offset

    # 3. Ask the model to solve it.
    with torch.no_grad():
        theta_pred = model.solve(target, theta_seed)
        pos_achieved, _ = robot.forward_kinematics(theta_pred)

    error_mm = (pos_achieved - target).norm().item() * 1000

    print(f"--- Example {label} ---")
    print(f"Starting joint angles : {[round(v, 3) for v in theta_seed[0].tolist()]}")
    print(f"Current hand position : {[round(v, 4) for v in current_pos[0].tolist()]}")
    print(f"Target point           : {[round(v, 4) for v in target[0].tolist()]}  (offset: {dx}, {dy}, {dz})")
    print(f"Predicted joint angles : {[round(v, 3) for v in theta_pred[0].tolist()]}")
    print(f"Position actually reached: {[round(v, 4) for v in pos_achieved[0].tolist()]}")
    print(f"Error: {error_mm:.2f} mm\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="runs/run4_singfix_model.pt")
    p.add_argument("--seed", type=int, default=None, help="random seed, for a repeatable demo")
    p.add_argument("--dx", type=float, default=0.05, help="target offset in x (meters)")
    p.add_argument("--dy", type=float, default=0.0, help="target offset in y (meters)")
    p.add_argument("--dz", type=float, default=0.03, help="target offset in z (meters)")
    p.add_argument("--count", type=int, default=1, help="run this many different examples in one go")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.seed is not None:
        torch.manual_seed(args.seed)

    model = load_model(args.checkpoint, device)

    for i in range(args.count):
        run_one(model, device, args.dx, args.dy, args.dz, label=f"{i + 1}/{args.count}")


if __name__ == "__main__":
    main()
