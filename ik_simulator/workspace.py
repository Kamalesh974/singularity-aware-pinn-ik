"""
Reachability check that does not depend on the PINN: sample many random joint configurations within
the joint limits, run forward kinematics, and record where the hand can be.

A target is called reachable only if the arm can actually put its hand there in the sample. This is
an empirical (sampled) workspace, so points right on its outer skin can be missed; the message says so.
"""
import math
from dataclasses import dataclass

import numpy as np
import torch

from pinn_ik import robot


@dataclass
class WorkspaceCheck:
    reachable: bool
    message: str
    radius: float
    max_radius: float


class WorkspaceModel:
    def __init__(self, n_samples=150_000, voxel=0.04, min_z=0.0, seed=0):
        self.n_samples, self.voxel, self.min_z, self.seed = n_samples, voxel, min_z, seed
        self.points = None
        self.max_radius = 0.0
        self._occupied = set()
        self._origin = None
        self._dims = None

    @property
    def ready(self):
        return self.points is not None

    def build(self):
        g = torch.Generator().manual_seed(self.seed)
        lim = robot.JOINT_LIMITS
        chunks = []
        with torch.no_grad():
            for _ in range(math.ceil(self.n_samples / 50_000)):
                u = torch.rand(50_000, robot.N_JOINTS, generator=g)
                q = lim[:, 0] + u * (lim[:, 1] - lim[:, 0])
                chunks.append(robot.forward_kinematics(q)[0].numpy())
        pts = np.concatenate(chunks)[: self.n_samples]
        pts = pts[pts[:, 2] >= self.min_z]
        self.points = pts
        self.max_radius = float(np.linalg.norm(pts, axis=1).max())
        self._origin = pts.min(axis=0) - 2 * self.voxel
        idx = np.floor((pts - self._origin) / self.voxel).astype(np.int64)
        self._dims = idx.max(axis=0) + 3
        self._occupied = set(self._key(i) for i in np.unique(idx, axis=0))

    def _key(self, ijk):
        return int(ijk[0]) + int(self._dims[0]) * (int(ijk[1]) + int(self._dims[1]) * int(ijk[2]))

    def _near_occupied(self, xyz):
        ijk = np.floor((np.asarray(xyz) - self._origin) / self.voxel).astype(np.int64)
        if np.any(ijk < 1) or np.any(ijk >= self._dims - 1):
            return False
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if self._key(ijk + (dx, dy, dz)) in self._occupied:
                        return True
        return False

    def check(self, xyz):
        xyz = np.asarray(xyz, dtype=float)
        r = float(np.linalg.norm(xyz))
        if not np.all(np.isfinite(xyz)):
            return WorkspaceCheck(False, "Target coordinates must be finite numbers.", r, self.max_radius)
        if xyz[2] < self.min_z:
            return WorkspaceCheck(False, f"Unreachable: the target is below the floor (z = {xyz[2]:.3f} m < {self.min_z:.3f} m).",
                                  r, self.max_radius)
        if not self.ready:
            return WorkspaceCheck(True, "Workspace map still loading - reachability not yet checked.", r, self.max_radius)
        if r > self.max_radius + 0.01:
            return WorkspaceCheck(False, f"Unreachable: the target is {r:.3f} m from the base, beyond the arm's maximum "
                                         f"reach of {self.max_radius:.3f} m.", r, self.max_radius)
        if not self._near_occupied(xyz):
            return WorkspaceCheck(False, "Unreachable: no joint configuration within the joint limits puts the hand at this "
                                         f"point (not found in {self.n_samples:,} sampled configurations). It is probably too "
                                         "close to the base, or in a region the joint limits rule out.", r, self.max_radius)
        return WorkspaceCheck(True, f"Inside the arm's reachable workspace ({r:.3f} m from base, max reach {self.max_radius:.3f} m).",
                              r, self.max_radius)

    def random_reachable_point(self, rng):
        """A hand position that some joint configuration really produced (so it is reachable)."""
        pts = self.points[self.points[:, 2] > 0.15]
        return pts[rng.integers(len(pts))].copy()
