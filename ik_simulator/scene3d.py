"""
3D view of the 7R arm (PyVista / VTK).

Link and joint meshes are built ONCE in their own local frames; every animation frame only updates a
4x4 matrix per actor from the numpy forward kinematics, so redrawing is cheap and smooth.
The geometry is derived from pinn_ik/robot_config.py, so a different DH table draws a different arm.
"""
import numpy as np
import pyvista as pv

from pinn_ik import robot_config as cfg

from . import settings
from .kinematics import frame_transforms

VIEWS = {
    "Isometric": ((1.75, -1.75, 1.3), (0.3, 0.0, 0.5), (0, 0, 1)),
    "Front": ((2.7, 0.0, 0.65), (0.3, 0.0, 0.5), (0, 0, 1)),
    "Side": ((0.3, -2.7, 0.65), (0.3, 0.0, 0.5), (0, 0, 1)),
    "Top": ((0.3, 0.0, 3.0), (0.3, 0.0, 0.0), (0, 1, 0)),
}


def _tube(p0, p1, radius):
    """Cylinder from p0 to p1 with a spherical cap at p1 (so consecutive links join smoothly)."""
    p0, p1 = np.asarray(p0, dtype=float), np.asarray(p1, dtype=float)
    v = p1 - p0
    length = float(np.linalg.norm(v))
    if length < 1e-9:
        return None
    cyl = pv.Cylinder(center=(p0 + p1) / 2, direction=v / length, radius=radius, height=length, resolution=32)
    cap = pv.Sphere(radius=radius, center=p1, theta_resolution=28, phi_resolution=28)
    return cyl + cap


class RobotScene:
    def __init__(self, plotter):
        self.p = plotter
        self.n = cfg.N_JOINTS
        self.link_actors, self.joint_actors, self.frame_actors = [], [], []
        self._planned = None
        self._trail_mesh = None
        self._trail_pts = None
        self._robot_color = settings.ROBOT_DEFAULT_COLOR
        self._build_environment()
        self._build_robot()
        self._build_markers()
        self.set_pose(cfg.HOME_POSITION)
        self.set_view("Isometric")

    # ---------------------------------------------------------------- environment
    def _build_environment(self):
        p = self.p
        p.set_background("#0d1118", top="#28334d")
        floor = pv.Plane(center=(0.35, 0, -0.003), direction=(0, 0, 1), i_size=2.8, j_size=2.8, i_resolution=1, j_resolution=1)
        p.add_mesh(floor, color="#141a26", opacity=0.92, name="floor", reset_camera=False)
        grid = pv.Plane(center=(0.35, 0, 0.0), direction=(0, 0, 1), i_size=2.8, j_size=2.8, i_resolution=28, j_resolution=28)
        p.add_mesh(grid, style="wireframe", color="#3a4258", line_width=1, opacity=0.7, name="grid", reset_camera=False)
        for label, direction, color in (("X", (1, 0, 0), "#ff5d5d"), ("Y", (0, 1, 0), "#5dff8a"), ("Z", (0, 0, 1), "#5da2ff")):
            arrow = pv.Arrow(start=(0, 0, 0), direction=direction, scale=0.45, tip_length=0.22, tip_radius=0.06, shaft_radius=0.022)
            p.add_mesh(arrow, color=color, name=f"axis_{label}", reset_camera=False)
            p.add_point_labels([np.array(direction) * 0.53], [label], font_size=16, text_color=color, shape=None,
                               show_points=False, always_visible=True, name=f"axis_label_{label}", reset_camera=False)
        p.add_axes(line_width=2, color="white")
        p.add_light(pv.Light(position=(2.5, -2.0, 3.5), focal_point=(0.3, 0, 0.5), intensity=0.55))

    # ---------------------------------------------------------------- robot
    def _add_robot_mesh(self, mesh, color, **kw):
        return self.p.add_mesh(mesh, color=color, smooth_shading=True, specular=0.5, specular_power=25,
                               ambient=0.25, reset_camera=False, **kw)

    def _build_robot(self):
        offsets = cfg.link_offsets()
        pedestal = pv.Cylinder(center=(0, 0, 0.02), direction=(0, 0, 1), radius=0.13, height=0.04, resolution=48)
        self._add_robot_mesh(pedestal, settings.JOINT_COLOR, name="pedestal")
        column = _tube((0, 0, 0), offsets[0], settings.LINK_RADIUS * 1.2)
        self.base_actor = self._add_robot_mesh(column, self._robot_color, name="base_column") if column is not None else None

        for i in range(self.n):
            child = offsets[i + 1] if i + 1 < self.n else (0.0, 0.0, cfg.TOOL_OFFSET_Z)
            tube = _tube((0, 0, 0), child, settings.LINK_RADIUS)
            self.link_actors.append(self._add_robot_mesh(tube, self._robot_color, name=f"link_{i}") if tube is not None else None)
            cyl = pv.Cylinder(center=(0, 0, 0), direction=(0, 0, 1), radius=settings.JOINT_RADIUS,
                              height=settings.JOINT_HEIGHT, resolution=40)
            self.joint_actors.append(self._add_robot_mesh(cyl, settings.JOINT_COLOR, name=f"joint_{i}"))

            frame = pv.PolyData(np.array([[0, 0, 0], [0.09, 0, 0], [0, 0.09, 0], [0, 0, 0.09]], dtype=float),
                                lines=np.array([2, 0, 1, 2, 0, 2, 2, 0, 3]))
            frame.cell_data["axis"] = np.array([0, 1, 2])
            actor = self.p.add_mesh(frame, scalars="axis", cmap=["#ff5d5d", "#5dff8a", "#5da2ff"], line_width=3,
                                    show_scalar_bar=False, reset_camera=False, name=f"frame_{i}", lighting=False)
            actor.visibility = False
            self.frame_actors.append(actor)

        ee = pv.Sphere(radius=0.024, theta_resolution=32, phi_resolution=32)
        self.ee_actor = self._add_robot_mesh(ee, settings.EE_COLOR, name="end_effector")

    # ---------------------------------------------------------------- markers
    def _build_markers(self):
        self.target_core = self.p.add_mesh(pv.Sphere(radius=0.03), color=settings.TARGET_COLOR, smooth_shading=True,
                                           name="target_core", reset_camera=False)
        self.target_halo = self.p.add_mesh(pv.Sphere(radius=0.055), color=settings.TARGET_COLOR, opacity=0.22,
                                           name="target_halo", reset_camera=False)
        self.start_marker = self.p.add_mesh(pv.Sphere(radius=0.017), color=settings.PLANNED_PATH_COLOR, smooth_shading=True,
                                            name="start_marker", reset_camera=False)
        for a in (self.target_core, self.target_halo, self.start_marker):
            a.visibility = False

    def set_target(self, xyz, reachable=True):
        xyz = np.asarray(xyz, dtype=float)
        color = settings.TARGET_COLOR if reachable else "#8a8f9c"
        for a in (self.target_core, self.target_halo):
            a.SetPosition(*xyz)
            a.prop.color = color
            a.visibility = True
        drop = pv.Line((xyz[0], xyz[1], 0.0), tuple(xyz))
        self.p.add_mesh(drop, color=color, opacity=0.55, line_width=1.5, name="target_drop", reset_camera=False)
        ring = pv.Disc(center=(xyz[0], xyz[1], 0.002), inner=0.035, outer=0.05, normal=(0, 0, 1), c_res=48)
        self.p.add_mesh(ring, color=color, opacity=0.7, name="target_ring", reset_camera=False)

    # ---------------------------------------------------------------- pose / colour / options
    def set_pose(self, q):
        """Move the arm to joint angles q; returns the end-effector position."""
        F = frame_transforms(np.asarray(q, dtype=float))
        for i in range(self.n):
            if self.link_actors[i] is not None:
                self.link_actors[i].user_matrix = F[i]
            self.joint_actors[i].user_matrix = F[i]
            if self.frame_actors[i].visibility:
                self.frame_actors[i].user_matrix = F[i]
        self.ee_actor.user_matrix = F[self.n]
        return F[self.n][:3, 3].copy()

    def set_robot_color(self, hex_color):
        self._robot_color = hex_color
        for a in self.link_actors + [self.base_actor]:
            if a is not None:
                a.prop.color = hex_color

    def show_joint_frames(self, visible, q=None):
        for a in self.frame_actors:
            a.visibility = visible
        if visible and q is not None:
            self.set_pose(q)

    def show_planned_path(self, visible):
        self._show_planned = visible
        actor = self.p.renderer.actors.get("planned_path")
        if actor is not None:
            actor.visibility = visible

    # ---------------------------------------------------------------- trajectory visuals
    def begin_motion(self, planned_pts, show_planned=True):
        """Draw the planned end-effector path and prepare an (initially empty) trail."""
        self._planned = np.asarray(planned_pts, dtype=float)
        n = len(self._planned)
        line = pv.lines_from_points(self._planned)
        actor = self.p.add_mesh(line, color=settings.PLANNED_PATH_COLOR, line_width=2.5, opacity=0.8,
                                name="planned_path", reset_camera=False)
        actor.visibility = show_planned
        self._trail_pts = np.repeat(self._planned[:1], n, axis=0)
        self._trail_mesh = pv.lines_from_points(self._trail_pts)
        self.p.add_mesh(self._trail_mesh, color=settings.TRAIL_COLOR, line_width=6, render_lines_as_tubes=True,
                        name="trail", reset_camera=False)
        self.start_marker.SetPosition(*self._planned[0])
        self.start_marker.visibility = True

    def update_trail(self, k):
        """Show the trail up to sample k of the planned path."""
        if self._trail_mesh is None:
            return
        k = int(np.clip(k, 0, len(self._planned) - 1))
        pts = self._trail_pts
        pts[: k + 1] = self._planned[: k + 1]
        pts[k + 1:] = self._planned[k]
        self._trail_mesh.points = pts

    def clear_motion(self):
        for name in ("planned_path", "trail"):
            self.p.remove_actor(name, render=False)
        self.start_marker.visibility = False
        self._trail_mesh = None

    # ---------------------------------------------------------------- camera
    def set_view(self, name):
        pos, focal, up = VIEWS[name]
        self.p.camera_position = [pos, focal, up]
        self.p.reset_camera_clipping_range()
