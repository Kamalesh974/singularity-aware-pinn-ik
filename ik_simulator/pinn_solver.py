"""
Inverse kinematics with the SAVED PINN, including multiple solutions for the redundant 7R arm.

How several solutions come out of one network
---------------------------------------------
The PINN is seed-conditioned: it turns a (target, current pose) pair into a joint step. A 7R arm has
a whole family of joint configurations for one hand position (it can move its elbow around while the
hand stays put), and which one the network lands on depends on the pose it starts from. So:

  1. start the PINN from many different poses (the current pose, the home pose, random poses);
  2. for each start, walk the hand to the target in short waypoints (the PINN is a local solver),
     every step being a PINN prediction, then apply extra refinement passes at the target;
  3. keep only candidates that are valid: inside the joint limits, finite, not on a singularity, and
     with the hand within the position tolerance of the target (checked with forward kinematics);
  4. keep at most MAX_SOLUTIONS that differ from each other by a minimum amount in joint space.

If fewer than MAX_SOLUTIONS distinct valid candidates exist, fewer are returned. Nothing is padded.
The optional classical DLS solver is only a secondary check and never supplies a solution.
"""
import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from pinn_ik import baseline_ik, robot
from pinn_ik import robot_config as cfg
from pinn_ik.model import load_model

from . import settings
from .kinematics import joint_limits


@dataclass
class IKSolution:
    index: int                 # 1-based, as shown in the GUI
    q: np.ndarray              # joint angles (rad)
    error_mm: float            # |forward kinematics(q) - target|
    manipulability: float
    distance_rad: float        # joint-space distance from the configuration the robot was in
    seed_label: str            # which starting pose the PINN was run from
    ee: np.ndarray             # hand position this solution actually reaches


@dataclass
class SolveResult:
    target: np.ndarray
    tolerance_mm: float
    num_candidates: int
    num_within_tolerance: int
    solutions: list
    message: str
    best_error_mm: float
    timings: dict = field(default_factory=dict)
    dls: dict = None


class PINNSolver:
    def __init__(self, checkpoint, device=settings.DEVICE):
        self.path = str(checkpoint)
        self.device = device
        self.model = load_model(self.path, device)
        for p in self.model.parameters():
            p.requires_grad_(False)

        trained_for = self.model.dh_fingerprint or cfg.PANDA_REFERENCE_FINGERPRINT
        self.robot_matches = trained_for == cfg.fingerprint()
        self.warning = "" if self.robot_matches else (
            "This PINN was trained for a different robot than the one in pinn_ik/robot_config.py "
            f"(trained for {trained_for}, current {cfg.fingerprint()}). Its predictions will not reach the "
            "target - retrain the network for the current robot.")
        self.single_pass_ms, self.single_fk_ms = self._benchmark()

    def describe(self):
        c = self.model.config
        n_params = sum(p.numel() for p in self.model.parameters())
        return {"file": self.path, "width": c.get("hidden", 256), "layers": c.get("n_layers", 4),
                "feedback": bool(c.get("feedback", False)), "passes": self.model.passes, "parameters": n_params}

    def _benchmark(self, reps=40):
        q = torch.tensor([cfg.HOME_POSITION], dtype=torch.float32)
        tgt, _ = robot.forward_kinematics(q)
        with torch.no_grad():
            for _ in range(5):
                self.model(tgt, q)
                robot.forward_kinematics(q)
            t = time.perf_counter()
            for _ in range(reps):
                self.model(tgt, q)
            pass_ms = (time.perf_counter() - t) / reps * 1000
            t = time.perf_counter()
            for _ in range(reps):
                robot.forward_kinematics(q)
            fk_ms = (time.perf_counter() - t) / reps * 1000
        return pass_ms, fk_ms

    def solve(self, target, q_current, num_candidates=settings.NUM_CANDIDATES, tol_mm=settings.POSITION_TOLERANCE_MM,
              max_solutions=settings.MAX_SOLUTIONS, min_manipulability=settings.MIN_MANIPULABILITY,
              seed=settings.RANDOM_SEED, waypoint_step=settings.WAYPOINT_STEP_M,
              waypoint_passes=settings.WAYPOINT_PASSES, final_passes=settings.FINAL_REFINE_PASSES, min_separation=settings.MIN_SOLUTION_SEPARATION_RAD,
              verify_with_dls=False):
        t_total = time.perf_counter()
        target = np.asarray(target, dtype=np.float32)
        lim = joint_limits()
        q_current = np.clip(np.asarray(q_current, dtype=float), lim[:, 0], lim[:, 1])

        rng = np.random.default_rng(seed)
        n = max(2, int(num_candidates))
        seeds = [q_current, np.array(cfg.HOME_POSITION, dtype=float)]
        seeds += list(rng.uniform(lim[:, 0], lim[:, 1], size=(n - 2, robot.N_JOINTS)))
        labels = ["current pose", "home pose"] + [f"random pose {k + 1}" for k in range(n - 2)]

        Q = torch.tensor(np.array(seeds), dtype=torch.float32, device=self.device)
        T = torch.tensor(target, device=self.device).expand(n, 3).contiguous()

        # ---- PINN prediction: walk the hand to the target in short waypoints, then refine ----
        t0 = time.perf_counter()
        with torch.no_grad():
            P0, _ = robot.forward_kinematics(Q)
            n_way = max(1, math.ceil(float((T - P0).norm(dim=-1).max()) / waypoint_step))
            for k in range(1, n_way + 1):
                Q = self.model.solve(P0 + (T - P0) * (k / n_way), Q, passes=waypoint_passes)
            if final_passes > 0:
                Q = self.model.solve(T, Q, passes=final_passes)
        pinn_ms = (time.perf_counter() - t0) * 1000

        # ---- forward-kinematics verification of every candidate ----
        t0 = time.perf_counter()
        with torch.no_grad():
            P, _ = robot.forward_kinematics(Q)
            manip = robot.manipulability(Q)
        fk_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        Qn, Pn = Q.numpy().astype(float), P.numpy().astype(float)
        err_mm = np.linalg.norm(Pn - target, axis=1) * 1000
        manip = manip.numpy()
        eps = 1e-6
        in_limits = np.all((Qn >= lim[:, 0] - eps) & (Qn <= lim[:, 1] + eps), axis=1)
        valid = np.isfinite(Qn).all(axis=1) & in_limits & (err_mm <= tol_mm) & (manip >= min_manipulability)

        # ---- choose up to max_solutions distinct valid ones ----
        picked = []
        idxs = np.where(valid)[0]
        if len(idxs):
            picked = [int(idxs[np.argmin(np.linalg.norm(Qn[idxs] - q_current, axis=1))])]
            remaining = [int(i) for i in idxs if i != picked[0]]
            while remaining and len(picked) < max_solutions:
                gap = [np.min(np.max(np.abs(Qn[picked] - Qn[i]), axis=1)) for i in remaining]
                j = int(np.argmax(gap))
                if gap[j] < min_separation:
                    break
                picked.append(remaining.pop(j))
        select_ms = (time.perf_counter() - t0) * 1000

        solutions = [IKSolution(k + 1, Qn[i], float(err_mm[i]), float(manip[i]), float(np.linalg.norm(Qn[i] - q_current)),
                                labels[i], Pn[i]) for k, i in enumerate(picked)]
        usable = np.isfinite(Qn).all(axis=1) & in_limits
        best = float(err_mm[usable].min()) if usable.any() else float("inf")

        if solutions:
            message = (f"Found {len(solutions)} valid IK solution{'s' if len(solutions) != 1 else ''} "
                       f"(from {n} PINN candidates; {int(valid.sum())} met the {tol_mm:g} mm tolerance).")
            if len(solutions) < max_solutions:
                message += (f" Fewer than {max_solutions} distinct valid solutions exist for this target within the "
                            "tolerance and limits, so only these are shown.")
        else:
            message = (f"No valid IK solution found. Best of {n} PINN candidates was {best:.1f} mm from the target "
                       f"(tolerance {tol_mm:g} mm). The target may sit near the edge of the workspace or next to a "
                       "singularity - try a larger tolerance or more candidates.")

        dls = None
        if verify_with_dls:
            t0 = time.perf_counter()
            with torch.no_grad():
                th = baseline_ik.solve_dls(torch.tensor(target).view(1, 3), torch.tensor(q_current, dtype=torch.float32).view(1, -1),
                                           max_iters=200, adaptive=True)
                p, _ = robot.forward_kinematics(th)
            dls = {"error_mm": float(np.linalg.norm(p[0].numpy() - target) * 1000),
                   "time_ms": (time.perf_counter() - t0) * 1000, "q": th[0].numpy().astype(float)}

        timings = {"pinn_ms": pinn_ms, "fk_ms": fk_ms, "select_ms": select_ms, "waypoints": n_way,
                   "total_ms": (time.perf_counter() - t_total) * 1000}
        return SolveResult(target.astype(float), tol_mm, n, int(valid.sum()), solutions, message, best, timings, dls)
