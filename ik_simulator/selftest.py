"""
Scripted end-to-end check that drives the REAL window: solve, animate, switch solutions, and try
unreachable targets, saving screenshots and printing what happened. Exit code 0 means every step passed.
"""
import time
import traceback
from pathlib import Path

import numpy as np
from PySide6.QtCore import QTimer

from .kinematics import ee_position, joint_limits


class SelfTest:
    def __init__(self, win, app, outdir):
        self.w, self.app = win, app
        self.out = Path(outdir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.failures = []

    def start(self):
        self.gen = self._script()
        QTimer.singleShot(800, self._next)

    # ---- tiny cooperative scheduler: the script yields either a delay (ms) or (condition, timeout_s) ----
    def _next(self):
        try:
            req = next(self.gen)
        except StopIteration:
            return self._finish()
        except Exception:
            traceback.print_exc()
            self.failures.append("exception in script")
            return self._finish()
        if isinstance(req, (int, float)):
            QTimer.singleShot(int(req), self._next)
        else:
            cond, timeout = req
            self._poll(cond, time.time() + timeout)

    def _poll(self, cond, deadline):
        if cond():
            self._next()
        elif time.time() > deadline:
            self.failures.append("timed out waiting for a condition")
            print("  TIMEOUT waiting for a condition")
            self._finish()
        else:
            QTimer.singleShot(100, lambda: self._poll(cond, deadline))

    def _check(self, ok, label):
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
        if not ok:
            self.failures.append(label)

    def _shot(self, name):
        try:
            self.w.plotter.screenshot(str(self.out / f"{name}_3d.png"))
            self.w.grab().save(str(self.out / f"{name}_window.png"))
        except Exception as exc:
            print(f"  (screenshot {name} failed: {exc})")

    def _set_target(self, xyz):
        for sp, v in zip((self.w.sp_x, self.w.sp_y, self.w.sp_z), xyz):
            sp.setValue(float(v))

    def _finish(self):
        print("\nSELFTEST", "PASSED" if not self.failures else f"FAILED: {self.failures}")
        self.w.close()
        self.app.exit(0 if not self.failures else 1)

    # ---- the script ----
    def _script(self):
        w = self.w
        print("SELFTEST: waiting for the PINN and workspace map...")
        yield (lambda: w.solver is not None and w.workspace.ready, 90)
        self._check(w.solver is not None, "PINN loaded")
        self._check(w.workspace.ready, "workspace map built")

        print("SELFTEST: reachable target, solve, animate")
        q_ref = np.array([0.35, -0.45, 0.25, -1.9, 0.15, 2.0, 0.6])
        tgt = ee_position(q_ref)
        self._set_target(tgt)
        yield 300
        w.solve()
        yield (lambda: w.result is not None and not w.is_busy(), 90)
        r = w.result
        n = len(r.solutions)
        print(f"  target {np.round(tgt, 3)} -> {n} solutions, errors mm: {[round(s.error_mm, 2) for s in r.solutions]}")
        print(f"  timings: PINN {r.timings['pinn_ms']:.0f} ms, FK {r.timings['fk_ms']:.1f} ms")
        self._check(1 <= n <= 5, "between 1 and 5 solutions (never more than 5)")
        self._check(all(s.error_mm <= r.tolerance_mm for s in r.solutions), "every solution within the position tolerance")
        lim = joint_limits()
        self._check(all(np.all(s.q >= lim[:, 0] - 1e-6) and np.all(s.q <= lim[:, 1] + 1e-6) for s in r.solutions),
                    "every solution inside the joint limits")
        yield (lambda: w.anim is not None, 5)
        yield 1300
        self._check(0 < w.progress.value() < 1000, "robot is mid-motion (progress between 0 and 100%)")
        self._shot("01_mid_motion")
        yield (lambda: w.anim is None, 20)
        final_err = np.linalg.norm(ee_position(w.q) - tgt) * 1000
        print(f"  final position error after animation: {final_err:.3f} mm | status: {w.live['state'].text()}")
        self._check(final_err <= w.sp_tol.value(), "robot finished within tolerance of the target")
        self._shot("02_reached")

        if n >= 2:
            print("SELFTEST: switch to solution 2")
            w._select_row(1)
            yield (lambda: w.anim is None, 20)
            self._check(np.allclose(w.q, r.solutions[1].q, atol=1e-6), "robot ended exactly on solution 2's joint angles")
            self._shot("03_solution2")

        print("SELFTEST: unreachable targets must give a message, not a crash")
        self._set_target((2.0, 2.0, 2.0))
        yield 200
        w.solve()
        yield 300
        txt = w.banner.text()
        print("  banner:", txt[:110].replace("\n", " "))
        self._check("nreachable" in txt and not w.is_busy(), "far target -> clear 'unreachable' message, app still running")
        self._shot("04_unreachable")
        self._set_target((0.4, 0.0, -0.3))
        yield 200
        w.solve()
        yield 300
        self._check("floor" in w.banner.text(), "below-floor target -> clear message")

        print("SELFTEST: reset")
        w.reset_robot()
        yield 300
        self._check(np.allclose(w.q, np.array([0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0, np.pi / 2, np.pi / 4])), "reset returns to the home pose")
