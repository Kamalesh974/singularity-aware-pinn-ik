"""
Real-time IK trajectory tracking in MuJoCo.

A red target sphere moves along a path. Every frame the solver(s) are asked
for joint angles that put the hand on the target, starting from the CURRENT
joint angles (this is how the seed-conditioned PINN is meant to be used):

  blue  arm : the trained PINN (one forward pass per frame)
  orange arm: classical damped-least-squares IK (a few iterations per frame)

Green sphere = actual hand position. Faint dots = the planned path; the
coloured trail = where the hand actually went.

    py -m pinn_ik.viz_mujoco                       # interactive window, both arms
    py -m pinn_ik.viz_mujoco --solver pinn --path figure8
    py -m pinn_ik.viz_mujoco --path reach          # stretch toward the singular, fully-extended pose
    py -m pinn_ik.viz_mujoco --record demo.gif     # no window, save a GIF
"""
import argparse
import math
import time
from collections import deque

import numpy as np
import torch

from . import baseline_ik, robot
from .model import load_model
from .mujoco_scene import ArmScene, add_sphere

READY_POSE = [0.0, -math.pi / 4, 0.0, -3 * math.pi / 4, 0.0, math.pi / 2, math.pi / 4]
SOLVER_RGBA = {"pinn": "0.20 0.45 0.90 1", "dls": "0.95 0.55 0.15 1"}
TRAIL_RGBA = {"pinn": [0.45, 0.75, 1.0, 1.0], "dls": [1.0, 0.75, 0.4, 1.0]}
LABEL = {"pinn": "PINN", "dls": "DLS "}
REACH_MAX = 0.82


def start_pose(kind):
    """Joint angles both solvers start from: a normal 'ready' pose, or a searched near-singular one."""
    if kind == "ready":
        return list(READY_POSE)
    from .evaluate import find_near_singular_configs

    print("Searching for a near-singular start pose (a few seconds)...")
    torch.manual_seed(3)
    th, w = find_near_singular_configs(1, iters=250, restarts_per_config=3, device="cpu")
    print(f"Start pose manipulability = {w.item():.5f}  (a typical random pose is about 0.066)")
    return th[0].tolist()


def make_path(kind, p0, radius, n):
    """Closed path (n,3) in the arm's base frame that starts and ends at p0."""
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    zero = np.zeros_like(t)
    if kind == "reach":
        # stretch the arm out to ~96% of its max reach (0.858 m) and back: heads toward the
        # fully-extended singular pose, the situation this project is about
        shoulder = np.array([0.0, 0.0, robot._D[0].item()])
        v = np.asarray(p0, dtype=np.float64) - shoulder
        r0 = np.linalg.norm(v)
        r = r0 + (REACH_MAX - r0) * (1.0 - np.cos(t)) / 2.0
        return shoulder + r[:, None] * (v / r0)
    if kind == "circle":
        off = np.stack([zero, radius * np.sin(t), radius * (1.0 - np.cos(t))], axis=1)
    elif kind == "figure8":
        off = np.stack([zero, radius * np.sin(t), 0.5 * radius * np.sin(2.0 * t)], axis=1)
    elif kind == "line":
        off = np.stack([radius * (1.0 - np.cos(t)), zero, zero], axis=1)
    else:
        raise ValueError(f"unknown path: {kind}")
    return np.asarray(p0, dtype=np.float64) + off


class Tracker:
    """Holds each solver's current joint angles and advances them one frame at a time."""

    def __init__(self, checkpoint, solvers, dls_iters=5, start=None):
        self.solvers = solvers
        self.dls_iters = dls_iters
        self.model = load_model(checkpoint, "cpu") if "pinn" in solvers else None
        self.theta = {s: torch.tensor([start or READY_POSE], dtype=torch.float32) for s in solvers}
        self.log = {s: {"err": [], "w": [], "ms": []} for s in solvers}

    def summary(self):
        lines = []
        for s in self.solvers:
            e, w, ms = (np.array(self.log[s][k]) for k in ("err", "w", "ms"))
            if len(e):
                lines.append(f"{LABEL[s]}: mean err {e.mean():6.2f} mm | max err {e.max():6.2f} mm | "
                             f"min manipulability {w.min():.4f} | mean {ms.mean():5.2f} ms/frame")
        return lines

    def step(self, target_xyz):
        tgt = torch.tensor(np.asarray(target_xyz), dtype=torch.float32).view(1, 3)
        out = {}
        for s in self.solvers:
            t0 = time.perf_counter()
            with torch.no_grad():
                if s == "pinn":
                    th = self.model.solve(tgt, self.theta[s])
                else:
                    th = baseline_ik.solve_dls(tgt, self.theta[s], max_iters=self.dls_iters, adaptive=True)
                pos, _ = robot.forward_kinematics(th)
                w = robot.manipulability(th)
            ms = (time.perf_counter() - t0) * 1000.0
            self.theta[s] = th
            self.log[s]["err"].append(float((pos - tgt).norm()) * 1000.0)
            self.log[s]["w"].append(float(w))
            self.log[s]["ms"].append(ms)
            out[s] = {"theta": th[0].numpy(), "pos": pos[0].numpy(),
                      "err_mm": float((pos - tgt).norm()) * 1000.0, "w": float(w), "ms": ms}
        return out


def draw_overlays(scn, scene, solvers, path, trails):
    """Path preview (faint dots) and the actual hand trail, added as decorative geoms."""
    for k, s in enumerate(solvers):
        for p in path[:: max(1, len(path) // 90)]:
            add_sphere(scn, scene.to_world(k, p), 0.004, [1, 1, 1, 0.35])
        col = TRAIL_RGBA[s]
        for p in trails[k]:
            add_sphere(scn, scene.to_world(k, p), 0.006, col)


def _apply(scene, solvers, res, target):
    for k, s in enumerate(solvers):
        scene.set_joints(k, res[s]["theta"])
        scene.set_target(k, target)
    scene.forward()


def _stats_line(t, solvers, res):
    parts = [f"{LABEL[s]} err {res[s]['err_mm']:6.1f} mm  w {res[s]['w']:.4f}  ({res[s]['ms']:5.2f} ms)" for s in solvers]
    return f"t={t:5.1f}s | " + " | ".join(parts)


def run(args):
    torch.set_num_threads(1)  # per-frame tensors are tiny; multithreading only adds overhead
    solvers ={"pinn": ["pinn"], "dls": ["dls"], "both": ["pinn", "dls"]}[args.solver]
    scene = ArmScene(n_arms=len(solvers), colors=[SOLVER_RGBA[s] for s in solvers])
    start = start_pose(args.start)
    tracker = Tracker(args.checkpoint, solvers, args.dls_iters, start=start)

    p0, _ = robot.forward_kinematics(torch.tensor([start], dtype=torch.float32))
    n = max(2, int(args.period * args.fps))
    radius = args.radius if args.radius is not None else (0.12 if args.start == "ready" else 0.05)
    path = make_path(args.path, p0[0].numpy(), radius, n)
    trails = [deque(maxlen=args.trail) for _ in solvers]
    dist = 2.6 if len(solvers) == 2 else 2.0

    print("Blue = PINN, orange = classical DLS, red sphere = target, green sphere = actual hand.")
    if args.record:
        _record(args, scene, tracker, solvers, path, trails, dist)
    else:
        _view(args, scene, tracker, solvers, path, trails, dist)


def _view(args, scene, tracker, solvers, path, trails, dist):
    import mujoco.viewer

    n, dt = len(path), 1.0 / args.fps
    with mujoco.viewer.launch_passive(scene.model, scene.data) as v:
        v.cam.lookat[:] = [0.3, 0.0, 0.6]
        v.cam.distance, v.cam.azimuth, v.cam.elevation = dist, 150, -18
        i = 0
        while v.is_running():
            t0 = time.perf_counter()
            target = path[i % n]
            res = tracker.step(target)
            _apply(scene, solvers, res, target)
            for k, s in enumerate(solvers):
                trails[k].append(res[s]["pos"])
            with v.lock():
                v.user_scn.ngeom = 0
                draw_overlays(v.user_scn, scene, solvers, path, trails)
            v.sync()
            if i % args.fps == 0:
                print(_stats_line(i * dt, solvers, res))
            i += 1
            time.sleep(max(0.0, dt - (time.perf_counter() - t0)))
    print("Summary:")
    for line in tracker.summary():
        print("  " + line)


def _record(args, scene, tracker, solvers, path, trails, dist):
    import mujoco
    from PIL import Image

    n, dt = len(path), 1.0 / args.fps
    renderer = mujoco.Renderer(scene.model, args.height, args.width)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.3, 0.0, 0.6]
    cam.distance, cam.azimuth, cam.elevation = dist, 150, -18
    frames = []
    for i in range(n * args.loops):
        target = path[i % n]
        res = tracker.step(target)
        _apply(scene, solvers, res, target)
        for k, s in enumerate(solvers):
            trails[k].append(res[s]["pos"])
        renderer.update_scene(scene.data, camera=cam)
        draw_overlays(renderer.scene, scene, solvers, path, trails)
        frames.append(Image.fromarray(renderer.render().copy()))
        if i % args.fps == 0:
            print(_stats_line(i * dt, solvers, res))
    frames[0].save(args.record, save_all=True, append_images=frames[1:],
                   duration=int(round(1000 / args.fps)), loop=0, optimize=True)
    print(f"saved {len(frames)} frames to {args.record}")
    print("Summary:")
    for line in tracker.summary():
        print("  " + line)


def add_args(p):
    p.add_argument("--checkpoint", default="runs/final_model.pt")
    p.add_argument("--solver", choices=["pinn", "dls", "both"], default="both")
    p.add_argument("--path", choices=["circle", "figure8", "line", "reach"], default="circle")
    p.add_argument("--start", choices=["ready", "singular"], default="ready",
                   help="'singular' starts both arms from a searched near-singular pose")
    p.add_argument("--radius", type=float, default=None,
                   help="path size in metres (default 0.12, or 0.05 for --start singular; ignored by --path reach)")
    p.add_argument("--period", type=float, default=8.0, help="seconds per loop of the path")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--dls-iters", type=int, default=5, help="DLS iterations per frame")
    p.add_argument("--trail", type=int, default=250, help="length of the hand trail (frames)")
    p.add_argument("--record", default=None, help="save a GIF here instead of opening a window")
    p.add_argument("--loops", type=int, default=1, help="path loops to record")
    p.add_argument("--width", type=int, default=800)
    p.add_argument("--height", type=int, default=450)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    add_args(ap)
    run(ap.parse_args())
