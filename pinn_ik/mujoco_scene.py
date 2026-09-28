"""
MuJoCo scene for the 7-DOF arm, generated from the SAME modified-DH table
used in robot.py, so the 3D model cannot drift from the kinematics the
network is trained on (checked numerically in tests/test_mujoco_scene.py).

Modified DH: T_i = Rx(alpha) Tx(a) Tz(d) Rz(theta).  In MJCF that is a static
placement (pos = [a, -sin(alpha) d, cos(alpha) d], rotation Rx(alpha)) followed
by a hinge about the body's own z-axis.

Physics is never stepped: poses are set through qpos + mj_forward, so this is
a pure kinematic visualisation (no gravity, contacts or dynamics).
"""
import math

import mujoco
import numpy as np

from . import robot

ARM_COLORS = {
    0: "0.20 0.45 0.90 1",   # blue   -> PINN
    1: "0.95 0.55 0.15 1",   # orange -> classical DLS
}


def _f(x):
    return f"{x:.9f}"


def _arm_body_xml(prefix, rgba):
    A, AL, D = robot._A.tolist(), robot._ALPHA.tolist(), robot._D.tolist()
    lim = robot.JOINT_LIMITS.tolist()
    n = robot.N_JOINTS
    place = [(A[i], -math.sin(AL[i]) * D[i], math.cos(AL[i]) * D[i]) for i in range(n)]

    def body(i, ind):
        pad = "  " * ind
        x, y, z = place[i]
        s = (f'{pad}<body name="{prefix}link{i + 1}" pos="{_f(x)} {_f(y)} {_f(z)}" '
             f'axisangle="1 0 0 {_f(AL[i])}">\n')
        s += (f'{pad}  <joint name="{prefix}j{i + 1}" type="hinge" axis="0 0 1" limited="true" '
              f'range="{_f(lim[i][0])} {_f(lim[i][1])}"/>\n')
        s += f'{pad}  <geom type="cylinder" size="0.055 0.05" rgba="0.15 0.16 0.2 1"/>\n'
        if i + 1 < n:
            cx, cy, cz = place[i + 1]
            if math.sqrt(cx * cx + cy * cy + cz * cz) > 1e-6:
                s += (f'{pad}  <geom type="capsule" size="0.038" '
                      f'fromto="0 0 0 {_f(cx)} {_f(cy)} {_f(cz)}" rgba="{rgba}"/>\n')
            s += body(i + 1, ind + 1)
        else:
            fl = robot._FLANGE_D
            s += f'{pad}  <geom type="capsule" size="0.03" fromto="0 0 0 0 0 {_f(fl)}" rgba="{rgba}"/>\n'
            s += f'{pad}  <geom type="sphere" size="0.022" pos="0 0 {_f(fl)}" rgba="0.2 0.9 0.3 1"/>\n'
            s += f'{pad}  <site name="{prefix}ee" pos="0 0 {_f(fl)}" size="0.01"/>\n'
        s += f"{pad}</body>\n"
        return s

    base_h = robot._D[0].item()
    s = (f'      <geom type="cylinder" size="0.09 0.02" pos="0 0 0.02" rgba="0.15 0.16 0.2 1"/>\n'
         f'      <geom type="capsule" size="0.05" fromto="0 0 0.02 0 0 {_f(base_h)}" rgba="{rgba}"/>\n')
    s += body(0, 3)
    return s


def build_mjcf(n_arms=1, spacing=1.1, colors=None):
    offsets = [np.array([0.0, (k - (n_arms - 1) / 2) * spacing, 0.0]) for k in range(n_arms)]
    arms, targets = "", ""
    for k in range(n_arms):
        p = f"a{k}"
        ox, oy, oz = offsets[k]
        arms += (f'    <body name="{p}root" pos="{_f(ox)} {_f(oy)} {_f(oz)}">\n'
                 f'{_arm_body_xml(p, (colors[k] if colors else ARM_COLORS.get(k, ARM_COLORS[1])))}'
                 f'    </body>\n')
        targets += (f'    <body name="{p}target" mocap="true" pos="{_f(ox)} {_f(oy)} 0.5">\n'
                    f'      <geom type="sphere" size="0.03" rgba="1 0.2 0.2 0.85"/>\n'
                    f'    </body>\n')
    xml = f"""
<mujoco model="panda_dh">
  <compiler angle="radian"/>
  <option gravity="0 0 0"/>
  <visual>
    <global offwidth="1280" offheight="720"/>
    <headlight ambient="0.45 0.45 0.45" diffuse="0.6 0.6 0.6" specular="0.1 0.1 0.1"/>
    <quality shadowsize="2048"/>
  </visual>
  <statistic center="0.3 0 0.5" extent="1.6"/>
  <default>
    <geom contype="0" conaffinity="0"/>
  </default>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.35 0.45 0.6" rgb2="0.02 0.03 0.05" width="512" height="3072"/>
    <texture type="2d" name="grid" builtin="checker" rgb1="0.22 0.26 0.3" rgb2="0.16 0.19 0.22"
             width="300" height="300" mark="edge" markrgb="0.32 0.37 0.42"/>
    <material name="grid" texture="grid" texrepeat="6 6" reflectance="0.12"/>
  </asset>
  <worldbody>
    <light pos="0.5 -1 2.5" dir="-0.1 0.3 -1" directional="true"/>
    <geom type="plane" size="2.5 2.5 0.05" material="grid"/>
{arms}{targets}  </worldbody>
</mujoco>
"""
    return xml, offsets


class ArmScene:
    """n_arms independent copies of the arm in one MuJoCo model."""

    def __init__(self, n_arms=1, spacing=1.1, colors=None):
        xml, self.offsets = build_mjcf(n_arms, spacing, colors)
        self.n_arms = n_arms
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self._ee = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, f"a{k}ee") for k in range(n_arms)]
        self._mocap = [self.model.body_mocapid[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"a{k}target")]
                       for k in range(n_arms)]

    def set_joints(self, arm, q):
        n = robot.N_JOINTS
        self.data.qpos[arm * n:(arm + 1) * n] = np.asarray(q, dtype=np.float64).reshape(-1)

    def set_target(self, arm, xyz_local):
        self.data.mocap_pos[self._mocap[arm]] = self.offsets[arm] + np.asarray(xyz_local, dtype=np.float64).reshape(3)

    def forward(self):
        mujoco.mj_forward(self.model, self.data)

    def ee_local(self, arm):
        """End-effector position in that arm's own base frame (comparable with robot.forward_kinematics)."""
        return self.data.site_xpos[self._ee[arm]] - self.offsets[arm]

    def to_world(self, arm, xyz_local):
        return self.offsets[arm] + np.asarray(xyz_local, dtype=np.float64).reshape(3)


def add_sphere(scn, pos, radius, rgba):
    """Append a decorative sphere to an mjvScene (viewer.user_scn or Renderer.scene)."""
    if scn.ngeom >= scn.maxgeom:
        return
    mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([radius, 0.0, 0.0]), np.asarray(pos, dtype=np.float64),
                        np.eye(3).flatten(), np.asarray(rgba, dtype=np.float32))
    scn.ngeom += 1
