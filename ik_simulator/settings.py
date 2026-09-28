"""
Application settings (NOT robot geometry -- that lives in pinn_ik/robot_config.py).
Tweak these freely; every value is also adjustable from the GUI where it makes sense.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ---- PINN ---------------------------------------------------------------------------------
DEFAULT_CHECKPOINT = PROJECT_ROOT / "runs" / "final_model.pt"
DEVICE = "cpu"                     # tiny batches: CPU is as fast as the GPU and avoids CUDA start-up

# ---- IK solution search ---------------------------------------------------------------------
NUM_CANDIDATES = 48                # PINN runs started from this many different poses
MAX_SOLUTIONS = 5                  # never show more than this many
POSITION_TOLERANCE_MM = 10.0       # a solution is valid only if its hand lands this close to the target
MIN_MANIPULABILITY = 1e-4          # reject configurations sitting on a singularity
MIN_SOLUTION_SEPARATION_RAD = 0.25 # two solutions must differ by at least this in some joint
WAYPOINT_STEP_M = 0.05             # the PINN is a local solver: it walks the hand to the target in steps of this size
WAYPOINT_PASSES = 2                # PINN passes at each intermediate waypoint
FINAL_REFINE_PASSES = 12           # extra PINN passes applied at the target itself
RANDOM_SEED = 0                    # makes the candidate poses (and so the solutions) repeatable

# ---- reachability -------------------------------------------------------------------------
WORKSPACE_SAMPLES = 150_000
WORKSPACE_VOXEL_M = 0.04
MIN_TARGET_Z = 0.0                 # the floor: targets below it are rejected

# ---- trajectory / animation ---------------------------------------------------------------
TRAJECTORY_DURATION_S = 3.0
MAX_JOINT_SPEED = 2.0              # rad/s; the duration is stretched if a move would exceed it
TRAJECTORY_PROFILE = "Minimum-jerk (quintic)"
ANIMATION_HZ = 60

# ---- look ---------------------------------------------------------------------------------
LINK_RADIUS = 0.038
JOINT_RADIUS = 0.058
JOINT_HEIGHT = 0.11
ROBOT_DEFAULT_COLOR = "#8fa8c8"
JOINT_COLOR = "#1f2430"
TARGET_COLOR = "#ff4d4f"
EE_COLOR = "#34d399"
PLANNED_PATH_COLOR = "#7aa2ff"
TRAIL_COLOR = "#ffb020"
SOLUTION_COLORS = ["#4c9aff", "#36b37e", "#ffab00", "#a78bfa", "#2dd4bf"]
