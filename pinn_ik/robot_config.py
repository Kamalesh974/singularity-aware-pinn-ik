"""
7R ROBOT PARAMETERS -- the ONE place to describe the robot.

Everything else reads from here: the PINN's forward-kinematics layer
(pinn_ik/robot.py), training, evaluation, the MuJoCo demo and the interactive
simulator (ik_simulator/). Edit the values below and the whole project follows.

IMPORTANT: a trained PINN is only valid for the robot it was trained on. If you
change DH_PARAMETERS / joint limits / tool offset, retrain the network
(py pinn_app.py train ...). Checkpoints record a fingerprint of these values and
the simulator warns when they no longer match.

Convention: MODIFIED (Craig) Denavit-Hartenberg. Row i describes joint i:

    T_i = Rot_x(alpha) * Trans_x(a) * Rot_z(theta_i + theta_offset) * Trans_z(d)

with a and alpha being the parameters of the link BEFORE joint i (a_{i-1}, alpha_{i-1}).
"""
import hashlib
import json
import math

pi = math.pi

# ==============================
# 7R ROBOT PARAMETERS
# ==============================

ROBOT_NAME = "Franka Emika Panda (7R)"

DH_PARAMETERS = [
    # a (m),   alpha (rad),  d (m),    theta_offset (rad)
    (0.0,      0.0,          0.333,    0.0),   # joint 1
    (0.0,     -pi / 2,       0.0,      0.0),   # joint 2
    (0.0,      pi / 2,       0.316,    0.0),   # joint 3
    (0.0825,   pi / 2,       0.0,      0.0),   # joint 4
    (-0.0825, -pi / 2,       0.384,    0.0),   # joint 5
    (0.0,      pi / 2,       0.0,      0.0),   # joint 6
    (0.088,    pi / 2,       0.0,      0.0),   # joint 7
]

# Fixed rigid offset from joint 7's frame to the end-effector point, along joint 7's z-axis (m).
TOOL_OFFSET_Z = 0.107

JOINT_NAMES = [
    "Shoulder yaw",
    "Shoulder pitch",
    "Upper-arm roll",
    "Elbow pitch",
    "Forearm roll",
    "Wrist pitch",
    "Wrist roll",
]

# (min, max) for each joint, radians.
JOINT_LIMITS_RAD = [
    (-2.8973, 2.8973),
    (-1.7628, 1.7628),
    (-2.8973, 2.8973),
    (-3.0718, -0.0698),
    (-2.8973, 2.8973),
    (-0.0175, 3.7525),
    (-2.8973, 2.8973),
]

# Pose the simulator starts in and returns to on "Reset" (radians, must respect the limits).
HOME_POSITION = [0.0, -pi / 4, 0.0, -3 * pi / 4, 0.0, pi / 2, pi / 4]

# ==============================
# END OF ROBOT PARAMETERS
# ==============================


N_JOINTS = len(DH_PARAMETERS)


def joint_offsets():
    """Constant angle added to each joint variable (the theta_offset column), radians."""
    return [row[3] for row in DH_PARAMETERS]


def link_offsets():
    """
    Position of each joint frame's origin relative to the previous frame, in the previous frame's
    coordinates: entry i is the static placement (a, -sin(alpha) d, cos(alpha) d) of joint i+1.
    Derived from the DH table, so it can never disagree with it.
    """
    return [(a, -math.sin(alpha) * d, math.cos(alpha) * d) for a, alpha, d, _ in DH_PARAMETERS]


def link_lengths():
    """Length (m) of each rigid segment: base->J1, J1->J2, ..., J6->J7, J7->tool (derived from DH)."""
    segs = [math.sqrt(x * x + y * y + z * z) for x, y, z in link_offsets()]
    return segs + [abs(TOOL_OFFSET_Z)]


def fingerprint():
    """Short hash of everything a trained network depends on (DH table, tool offset, limits)."""
    payload = json.dumps({
        "dh": [[round(v, 6) for v in row] for row in DH_PARAMETERS],
        "tool": round(TOOL_OFFSET_Z, 6),
        "limits": [[round(v, 6) for v in row] for row in JOINT_LIMITS_RAD],
    }, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


# Fingerprint of the original Panda table above. Checkpoints trained before fingerprints were
# recorded (all of runs/*.pt so far) were trained on exactly this robot. Do NOT edit this line
# when you change the robot -- that is the whole point of it.
PANDA_REFERENCE_FINGERPRINT = "ceccaa6096f2"


def _validate():
    assert len(JOINT_NAMES) == N_JOINTS, "JOINT_NAMES needs one entry per DH row"
    assert len(JOINT_LIMITS_RAD) == N_JOINTS, "JOINT_LIMITS_RAD needs one entry per DH row"
    assert len(HOME_POSITION) == N_JOINTS, "HOME_POSITION needs one entry per DH row"
    for (lo, hi), q, name in zip(JOINT_LIMITS_RAD, HOME_POSITION, JOINT_NAMES):
        assert lo < hi, f"{name}: lower limit must be below upper limit"
        assert lo <= q <= hi, f"HOME_POSITION violates the limits of joint '{name}'"


_validate()
