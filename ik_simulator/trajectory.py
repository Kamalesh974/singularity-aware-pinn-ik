"""
Smooth joint-space trajectories. The robot never jumps: it starts at the current configuration,
ends exactly at the chosen IK solution, and velocity/acceleration are zero at both ends.

Profiles (all zero velocity at both ends):
    minimum-jerk (quintic):  s(t) = 10 t^3 - 15 t^4 + 6 t^5     (also zero acceleration at both ends)
    cubic:                   s(t) = 3 t^2 - 2 t^3
Because every joint uses the same scalar s(t), all joints start and stop together and the motion
stays inside the joint limits whenever both end points do.
"""
from dataclasses import dataclass

import numpy as np

PROFILES = {
    "Minimum-jerk (quintic)": (lambda t: 10 * t**3 - 15 * t**4 + 6 * t**5,
                               lambda t: 30 * t**2 - 60 * t**3 + 30 * t**4, 1.875),
    "Cubic": (lambda t: 3 * t**2 - 2 * t**3, lambda t: 6 * t - 6 * t**2, 1.5),
}


@dataclass
class Trajectory:
    q_start: np.ndarray
    q_goal: np.ndarray
    duration: float
    profile: str

    def _tau(self, t):
        return np.clip(np.asarray(t, dtype=float) / self.duration, 0.0, 1.0)

    def at(self, t):
        """Joint angles at time t (seconds); t may be a scalar or an array -> (N,) or (M, N)."""
        s = PROFILES[self.profile][0](self._tau(t))
        return self.q_start + np.multiply.outer(s, self.q_goal - self.q_start)

    def velocity(self, t):
        """Joint velocities (rad/s) at time t."""
        ds = PROFILES[self.profile][1](self._tau(t)) / self.duration
        return ds * (self.q_goal - self.q_start)

    @property
    def peak_speed(self):
        """Largest joint speed reached anywhere on the trajectory, rad/s."""
        return PROFILES[self.profile][2] * float(np.max(np.abs(self.q_goal - self.q_start))) / self.duration

    def sample(self, hz):
        """Evenly spaced samples: times (M,), joint angles (M, N). Includes both end points."""
        m = max(2, int(round(self.duration * hz)) + 1)
        t = np.linspace(0.0, self.duration, m)
        return t, self.at(t)


def plan(q_start, q_goal, duration=3.0, max_speed=2.0, profile="Minimum-jerk (quintic)"):
    """Build a trajectory; the duration is stretched if the requested one would exceed max_speed."""
    q0, q1 = np.asarray(q_start, dtype=float), np.asarray(q_goal, dtype=float)
    factor = PROFILES[profile][2]
    needed = factor * float(np.max(np.abs(q1 - q0))) / max_speed if max_speed > 0 else 0.0
    return Trajectory(q0, q1, max(duration, needed, 0.2), profile)
