"""
Main window: input panel (left), 3D view (centre), solutions / joint angles / live state / research
metrics (right). Solving runs in a worker thread and the animation runs on a timer, so the window
stays responsive at all times.
"""
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import QSize, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QIcon, QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QFrame,
                               QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QListWidget, QListWidgetItem,
                               QMainWindow, QMessageBox, QProgressBar, QPushButton, QScrollArea, QSpinBox,
                               QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)
from pyvistaqt import QtInteractor

from pinn_ik import robot_config as cfg

from . import settings, trajectory
from .kinematics import ee_position, ee_positions, joint_limits
from .pinn_solver import PINNSolver
from .scene3d import VIEWS, RobotScene
from .workspace import WorkspaceModel

STYLE = """
QWidget { background: #11151d; color: #dfe6f1; font-family: 'Segoe UI'; font-size: 10pt; }
QMainWindow, QScrollArea, QScrollArea > QWidget > QWidget { background: #11151d; }
QGroupBox { border: 1px solid #262e3f; border-radius: 9px; margin-top: 14px; padding: 10px 8px 6px 8px; font-weight: 600; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; color: #8fb4ff; }
QLabel#dim { color: #8b97ad; }
QLabel#title { font-size: 15pt; font-weight: 700; color: #ffffff; }
QLabel#subtitle { color: #8b97ad; }
QLabel#value { font-family: Consolas, 'Courier New', monospace; font-weight: 600; color: #f2f6ff; }
QPushButton { background: #1d2535; border: 1px solid #2c3750; border-radius: 6px; padding: 7px 12px; }
QPushButton:hover { background: #27324a; }
QPushButton:pressed { background: #202a40; }
QPushButton:disabled { color: #566178; background: #161c28; }
QPushButton#primary { background: #2f6bff; border: none; color: white; font-weight: 700; padding: 11px; font-size: 11pt; }
QPushButton#primary:hover { background: #4a80ff; }
QPushButton#primary:disabled { background: #223055; color: #7c88a6; }
QDoubleSpinBox, QSpinBox, QComboBox { background: #0c1017; border: 1px solid #2c3750; border-radius: 5px; padding: 4px 6px; min-height: 20px; }
QComboBox QAbstractItemView { background: #0c1017; selection-background-color: #22315a; }
QListWidget { background: #0c1017; border: 1px solid #2c3750; border-radius: 7px; outline: none; }
QListWidget::item { padding: 4px 6px; border-bottom: 1px solid #1a2130; }
QListWidget::item:selected { background: #1f2f5c; color: #ffffff; border-left: 4px solid #4c9aff; }
QListWidget::item:hover { background: #172037; }
QTableWidget { background: #0c1017; border: 1px solid #2c3750; border-radius: 6px; gridline-color: #1c2333; }
QHeaderView::section { background: #171d29; color: #9fb2d0; border: none; padding: 4px; font-weight: 600; }
QProgressBar { background: #0c1017; border: 1px solid #2c3750; border-radius: 5px; text-align: center; min-height: 18px; }
QProgressBar::chunk { background: #2f6bff; border-radius: 4px; }
QCheckBox::indicator { width: 15px; height: 15px; }
QScrollBar:vertical { background: #11151d; width: 10px; }
QScrollBar::handle:vertical { background: #2c3750; border-radius: 5px; min-height: 24px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
"""

BANNER_STYLES = {
    "info": ("#12233f", "#2f6bff", "#cfe0ff"),
    "ok": ("#0f2a20", "#2ea36f", "#c6f5e0"),
    "warn": ("#33260c", "#d99a1a", "#ffe6b0"),
    "error": ("#331416", "#e5484d", "#ffd0d2"),
}


def _chip(color, size=16):
    pm = QPixmap(size, size)
    pm.fill(QColor(color))
    return QIcon(pm)


def _fmt_xyz(v):
    return f"{v[0]:+.3f}  {v[1]:+.3f}  {v[2]:+.3f}"


class SolveWorker(QThread):
    finished_ok = Signal(object)
    failed = Signal(str)

    def __init__(self, solver, kwargs):
        super().__init__()
        self.solver, self.kwargs = solver, kwargs

    def run(self):
        try:
            self.finished_ok.emit(self.solver.solve(**self.kwargs))
        except Exception as exc:  # never let a solver problem take the GUI down
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class WorkspaceWorker(QThread):
    finished_ok = Signal()

    def __init__(self, workspace):
        super().__init__()
        self.workspace = workspace

    def run(self):
        self.workspace.build()
        self.finished_ok.emit()


class MainWindow(QMainWindow):
    def __init__(self, checkpoint=None):
        super().__init__()
        self.setWindowTitle(f"7R Arm - PINN Inverse Kinematics Simulator  |  {cfg.ROBOT_NAME}")
        self.setStyleSheet(STYLE)

        self.q = np.array(cfg.HOME_POSITION, dtype=float)      # the robot's current joint angles
        self.solver = None
        self.workspace = WorkspaceModel(settings.WORKSPACE_SAMPLES, settings.WORKSPACE_VOXEL_M, settings.MIN_TARGET_Z)
        self.result = None
        self.selected = None
        self.anim = None                                        # active motion, or None
        self.last_start_q = self.q.copy()
        self._worker = None
        self._ws_worker = None
        self._traj_ms = None

        root = QWidget()
        self.setCentralWidget(root)
        lay = QHBoxLayout(root)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(10)
        lay.addWidget(self._wrap_scroll(self._build_left(), 318))
        lay.addWidget(self._build_center(), 1)
        lay.addWidget(self._wrap_scroll(self._build_right(), 392))

        scr = self.screen().availableGeometry()
        self.resize(min(1720, int(scr.width() * 0.96)), min(980, int(scr.height() * 0.94)))

        self.scene = RobotScene(self.plotter)
        self.plotter.enable_anti_aliasing("fxaa")
        self._on_target_changed()
        self._update_live(self.q)

        self.timer = QTimer(self)
        self.timer.setInterval(int(1000 / settings.ANIMATION_HZ))
        self.timer.timeout.connect(self._tick)

        self._banner("Loading the workspace map and the PINN...", "info")
        self._start_workspace_build()
        self._load_checkpoint(checkpoint or settings.DEFAULT_CHECKPOINT, initial=True)

    # =================================================================== layout
    @staticmethod
    def _wrap_scroll(widget, width):
        sa = QScrollArea()
        sa.setWidgetResizable(True)
        sa.setFrameShape(QFrame.NoFrame)
        sa.setFixedWidth(width)
        sa.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        sa.setWidget(widget)
        return sa

    def _spin(self, lo, hi, val, step=0.01, dec=3, suffix=" m"):
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setDecimals(dec)
        s.setSingleStep(step)
        s.setSuffix(suffix)
        s.setValue(val)
        s.setKeyboardTracking(False)
        return s

    def _build_left(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(4, 4, 8, 4)
        lay.setSpacing(6)

        title = QLabel("PINN IK Simulator")
        title.setObjectName("title")
        sub = QLabel(f"{cfg.ROBOT_NAME}  -  {cfg.N_JOINTS} revolute joints")
        sub.setObjectName("subtitle")
        lay.addWidget(title)
        lay.addWidget(sub)

        g = QGroupBox("Target end-effector position")
        f = QFormLayout(g)
        self.sp_x = self._spin(-1.5, 1.5, 0.45)
        self.sp_y = self._spin(-1.5, 1.5, 0.20)
        self.sp_z = self._spin(-0.5, 2.0, 0.55)
        for label, sp in (("X", self.sp_x), ("Y", self.sp_y), ("Z", self.sp_z)):
            f.addRow(label, sp)
            sp.valueChanged.connect(self._on_target_changed)
        self.lbl_reach = QLabel("")
        self.lbl_reach.setWordWrap(True)
        self.lbl_reach.setObjectName("dim")
        self.lbl_reach.setMinimumHeight(66)
        self.lbl_reach.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        f.addRow(self.lbl_reach)
        self.btn_solve = QPushButton("Solve IK with PINN")
        self.btn_solve.setObjectName("primary")
        self.btn_solve.clicked.connect(self.solve)
        f.addRow(self.btn_solve)
        row = QHBoxLayout()
        self.btn_random = QPushButton("Random reachable target")
        self.btn_random.clicked.connect(self.random_target)
        self.btn_reset = QPushButton("Reset robot")
        self.btn_reset.clicked.connect(self.reset_robot)
        row.addWidget(self.btn_random)
        row.addWidget(self.btn_reset)
        f.addRow(row)
        lay.addWidget(g)

        g = QGroupBox("PINN model")
        v = QVBoxLayout(g)
        self.lbl_model = QLabel("(none loaded)")
        self.lbl_model.setWordWrap(True)
        self.lbl_model_info = QLabel("")
        self.lbl_model_info.setObjectName("dim")
        self.lbl_model_info.setWordWrap(True)
        self.lbl_model_warn = QLabel("")
        self.lbl_model_warn.setWordWrap(True)
        self.lbl_model_warn.setStyleSheet("color:#ffb3b6;")
        self.lbl_model_warn.setVisible(False)
        btn = QPushButton("Load PINN checkpoint...")
        btn.clicked.connect(self.load_checkpoint_dialog)
        for x in (self.lbl_model, self.lbl_model_info, self.lbl_model_warn, btn):
            v.addWidget(x)
        lay.addWidget(g)

        g = QGroupBox("IK search")
        f = QFormLayout(g)
        self.sp_cand = QSpinBox()
        self.sp_cand.setRange(8, 256)
        self.sp_cand.setValue(settings.NUM_CANDIDATES)
        self.sp_tol = self._spin(0.5, 100.0, settings.POSITION_TOLERANCE_MM, 0.5, 1, " mm")
        self.sp_maxsol = QSpinBox()
        self.sp_maxsol.setRange(1, settings.MAX_SOLUTIONS)
        self.sp_maxsol.setValue(settings.MAX_SOLUTIONS)
        self.sp_seed = QSpinBox()
        self.sp_seed.setRange(0, 99999)
        self.sp_seed.setValue(settings.RANDOM_SEED)
        self.chk_dls = QCheckBox("Secondary check with classical DLS")
        f.addRow("PINN start poses", self.sp_cand)
        f.addRow("Position tolerance", self.sp_tol)
        f.addRow("Max solutions", self.sp_maxsol)
        f.addRow("Random seed", self.sp_seed)
        f.addRow(self.chk_dls)
        ik_group = g  # added below the motion controls: it is the more advanced of the two

        g = QGroupBox("Motion")
        f = QFormLayout(g)
        self.sp_dur = self._spin(0.5, 20.0, settings.TRAJECTORY_DURATION_S, 0.5, 1, " s")
        self.cmb_profile = QComboBox()
        self.cmb_profile.addItems(list(trajectory.PROFILES))
        self.cmb_profile.setCurrentText(settings.TRAJECTORY_PROFILE)
        self.chk_planned = QCheckBox("Show planned path")
        self.chk_planned.setChecked(True)
        self.chk_planned.toggled.connect(lambda on: (self.scene.show_planned_path(on), self.plotter.render()))
        self.chk_frames = QCheckBox("Show joint frames")
        self.chk_frames.toggled.connect(lambda on: (self.scene.show_joint_frames(on, self._display_q()), self.plotter.render()))
        f.addRow("Duration", self.sp_dur)
        f.addRow("Profile", self.cmb_profile)
        f.addRow(self.chk_planned)
        f.addRow(self.chk_frames)
        lay.addWidget(g)
        lay.addWidget(ik_group)
        lay.addStretch(1)
        return w

    def _build_center(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        bar = QHBoxLayout()
        for name in VIEWS:
            b = QPushButton(name)
            b.setFixedHeight(28)
            b.clicked.connect(lambda _=False, n=name: self._set_view(n))
            bar.addWidget(b)
        bar.addStretch(1)
        hint = QLabel("Rotate: left-drag    Pan: middle-drag / Shift+left-drag    Zoom: wheel / right-drag")
        hint.setObjectName("dim")
        bar.addWidget(hint)
        lay.addLayout(bar)

        self.plotter = QtInteractor(w)
        lay.addWidget(getattr(self.plotter, "interactor", self.plotter), 1)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setFormat("Trajectory progress  %p%")
        lay.addWidget(self.progress)
        lay.addWidget(self._build_readouts())

        self.banner = QLabel("")
        self.banner.setWordWrap(True)
        self.banner.setMinimumHeight(46)
        lay.addWidget(self.banner)
        return w

    def _build_right(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(8, 4, 4, 4)
        lay.setSpacing(6)

        g = QGroupBox("IK solutions (click one to move the robot)")
        v = QVBoxLayout(g)
        self.lbl_count = QLabel("No solve yet.")
        self.lbl_count.setWordWrap(True)
        self.lbl_count.setObjectName("dim")
        self.list = QListWidget()
        self.list.setFixedHeight(5 * 60 + 10)
        self.list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.list.itemClicked.connect(lambda item: self._select_row(self.list.row(item)))
        row = QHBoxLayout()
        self.btn_replay = QPushButton("Replay from start pose")
        self.btn_replay.clicked.connect(self.replay)
        self.btn_replay.setEnabled(False)
        row.addWidget(self.btn_replay)
        v.addWidget(self.lbl_count)
        v.addWidget(self.list)
        v.addLayout(row)
        lay.addWidget(g)

        g = QGroupBox("Joint angles")
        v = QVBoxLayout(g)
        self.tbl = QTableWidget(cfg.N_JOINTS, 4)
        self.tbl.setHorizontalHeaderLabels(["Joint", "Live", "IK target", "Limits"])
        self.tbl.verticalHeader().setVisible(False)
        self.tbl.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tbl.setSelectionMode(QAbstractItemView.NoSelection)
        self.tbl.setFocusPolicy(Qt.NoFocus)
        hh = self.tbl.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.Stretch)
        for c in (1, 2, 3):
            hh.setSectionResizeMode(c, QHeaderView.ResizeToContents)
        lim = np.degrees(joint_limits())
        for i in range(cfg.N_JOINTS):
            self.tbl.setItem(i, 0, QTableWidgetItem(f"θ{i + 1}  {cfg.JOINT_NAMES[i]}"))
            for c in (1, 2):
                it = QTableWidgetItem("-")
                it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.tbl.setItem(i, c, it)
            it = QTableWidgetItem(f"{lim[i, 0]:.0f}..{lim[i, 1]:.0f}")
            it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            it.setForeground(QColor("#8b97ad"))
            self.tbl.setItem(i, 3, it)
            self.tbl.setRowHeight(i, 26)
        self.tbl.horizontalHeader().setFixedHeight(28)
        self.tbl.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.tbl.setFixedHeight(28 + 26 * cfg.N_JOINTS + 6)
        v.addWidget(self.tbl)
        note = QLabel("Angles in degrees. Amber = within 3° of a joint limit.")
        note.setObjectName("dim")
        v.addWidget(note)
        lay.addWidget(g)

        g = QGroupBox("Notes")
        v = QVBoxLayout(g)
        note = QLabel("Each solution is checked for joint limits, position error and singularity. Only the hand POSITION is "
                      "solved (this PINN does not predict orientation), and link-to-link or floor collisions are not modelled.")
        note.setWordWrap(True)
        note.setObjectName("dim")
        v.addWidget(note)
        lay.addWidget(g)
        lay.addStretch(1)
        return w

    def _build_readouts(self):
        """Live state + research metrics, in a strip under the 3D view so they stay visible while the robot moves."""
        w = QWidget()
        strip = QHBoxLayout(w)
        strip.setContentsMargins(0, 0, 0, 0)
        strip.setSpacing(8)

        g = QGroupBox("Live state")
        grid = QGridLayout(g)
        grid.setVerticalSpacing(3)
        self.live = {}
        for r, (key, label) in enumerate([("cur", "Current XYZ (m)"), ("tgt", "Target XYZ (m)"), ("err", "Position error"),
                                          ("state", "Status")]):
            grid.addWidget(self._dim(label), r, 0)
            self.live[key] = self._value("-")
            grid.addWidget(self.live[key], r, 1)
        strip.addWidget(g, 5)

        g = QGroupBox("Research metrics")
        grid = QGridLayout(g)
        grid.setVerticalSpacing(3)
        self.m = {}
        rows = [("pinn", "PINN prediction time"), ("fk", "FK computation time"), ("traj", "Trajectory generation"),
                ("final", "Final position error"), ("cand", "PINN candidates"), ("peak", "Peak joint speed"),
                ("lat", "1 PINN pass / 1 FK"), ("dls", "DLS secondary check")]
        for i, (key, label) in enumerate(rows):
            r, c = i % 4, (i // 4) * 2
            grid.addWidget(self._dim(label), r, c)
            self.m[key] = self._value("-")
            grid.addWidget(self.m[key], r, c + 1)
        strip.addWidget(g, 7)
        return w

    def _dim(self, text):
        l = QLabel(text)
        l.setObjectName("dim")
        return l

    def _value(self, text):
        l = QLabel(text)
        l.setObjectName("value")
        l.setWordWrap(False)
        l.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        l.setTextInteractionFlags(Qt.TextSelectableByMouse)
        return l

    # =================================================================== small helpers
    def _banner(self, text, kind="info"):
        bg, border, fg = BANNER_STYLES[kind]
        self.banner.setStyleSheet(f"background:{bg}; border:1px solid {border}; border-radius:8px; color:{fg}; padding:8px 12px;")
        self.banner.setText(text)

    def _set_view(self, name):
        self.scene.set_view(name)
        self.plotter.render()

    def target(self):
        return np.array([self.sp_x.value(), self.sp_y.value(), self.sp_z.value()])

    def _display_q(self):
        return self.anim["q"] if self.anim else self.q

    # =================================================================== workspace / model loading
    def _start_workspace_build(self):
        self._ws_worker = WorkspaceWorker(self.workspace)
        self._ws_worker.finished_ok.connect(self._workspace_ready)
        self._ws_worker.start()

    def _workspace_ready(self):
        self._on_target_changed()
        if self.solver is not None and not self.is_busy():
            self._banner("Ready. Enter a target position and press 'Solve IK with PINN'.", "ok")

    def is_busy(self):
        return self._worker is not None and self._worker.isRunning()

    def load_checkpoint_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load PINN checkpoint", str(settings.PROJECT_ROOT / "runs"),
                                              "PyTorch checkpoint (*.pt)")
        if path:
            self._load_checkpoint(path)

    def _load_checkpoint(self, path, initial=False):
        path = Path(path)
        try:
            if not path.exists():
                raise FileNotFoundError(f"checkpoint not found: {path}")
            solver = PINNSolver(path)
        except Exception as exc:
            self.solver = None if initial else self.solver
            self.btn_solve.setEnabled(self.solver is not None)
            msg = f"Could not load the PINN checkpoint.\n{type(exc).__name__}: {exc}"
            self._banner(msg, "error")
            if not initial:
                QMessageBox.critical(self, "PINN checkpoint", msg)
            return
        self.solver = solver
        d = solver.describe()
        self.lbl_model.setText(f"<b>{path.name}</b>")
        self.lbl_model.setToolTip(str(path))
        self.lbl_model_info.setText(f"{d['layers']} layers x {d['width']} wide, {d['parameters']:,} parameters, "
                                    f"{'FK feedback inputs, ' if d['feedback'] else ''}{d['passes']} refinement passes per call")
        self.lbl_model_warn.setText(solver.warning)
        self.lbl_model_warn.setVisible(bool(solver.warning))
        self.m["lat"].setText(f"{solver.single_pass_ms:.1f} / {solver.single_fk_ms:.1f} ms")
        self.m["lat"].setToolTip("One PINN forward pass (batch of 1) / one forward-kinematics call, measured at load time.")
        self.btn_solve.setEnabled(True)
        if solver.warning:
            self._banner(solver.warning, "warn")
        elif self.workspace.ready:
            self._banner("Ready. Enter a target position and press 'Solve IK with PINN'.", "ok")

    # =================================================================== target handling
    def _on_target_changed(self):
        t = self.target()
        chk = self.workspace.check(t)
        self.lbl_reach.setText(chk.message)
        self.lbl_reach.setStyleSheet("color:#7be0b0;" if chk.reachable else "color:#ff9aa0;")
        self.scene.set_target(t, reachable=chk.reachable)
        self.live["tgt"].setText(_fmt_xyz(t))
        self._update_error_label(self._display_q())
        self.plotter.render()

    def random_target(self):
        if not self.workspace.ready:
            self._banner("The workspace map is still loading - try again in a moment.", "info")
            return
        p = self.workspace.random_reachable_point(np.random.default_rng())
        for sp, v in zip((self.sp_x, self.sp_y, self.sp_z), p):
            sp.blockSignals(True)
            sp.setValue(float(v))
            sp.blockSignals(False)
        self._on_target_changed()

    # =================================================================== solving
    def solve(self):
        if self.solver is None:
            self._banner("No PINN is loaded - use 'Load PINN checkpoint...'.", "error")
            return
        if self.is_busy():
            return
        t = self.target()
        chk = self.workspace.check(t)
        self.scene.set_target(t, reachable=chk.reachable)
        if not chk.reachable:
            self._clear_solutions()
            self._banner(chk.message + "  Choose a target inside the arm's workspace.", "error")
            self.live["state"].setText("Target unreachable")
            self.plotter.render()
            return
        self._banner("Solving: running the PINN from many starting poses...", "info")
        self.live["state"].setText("Solving...")
        self.btn_solve.setEnabled(False)
        self._worker = SolveWorker(self.solver, dict(
            target=t, q_current=self._display_q().copy(), num_candidates=self.sp_cand.value(), tol_mm=self.sp_tol.value(),
            max_solutions=self.sp_maxsol.value(), seed=self.sp_seed.value(), verify_with_dls=self.chk_dls.isChecked()))
        self._worker.finished_ok.connect(self._on_solved)
        self._worker.failed.connect(self._on_solve_failed)
        self._worker.start()

    def _on_solve_failed(self, msg):
        self.btn_solve.setEnabled(True)
        self.live["state"].setText("Error")
        self._banner(f"The solver hit an unexpected error and was stopped safely: {msg}", "error")

    def _clear_solutions(self):
        self.list.clear()
        self.result, self.selected = None, None
        self.btn_replay.setEnabled(False)
        self.lbl_count.setText("No valid solutions.")
        for r in range(cfg.N_JOINTS):
            self.tbl.item(r, 2).setText("-")

    def _on_solved(self, res):
        self.btn_solve.setEnabled(True)
        self.result = res
        tm = res.timings
        self.m["pinn"].setText(f"{tm['pinn_ms']:.0f} ms")
        self.m["pinn"].setToolTip(f"Whole PINN search: {res.num_candidates} start poses run in one batch, "
                                  f"{tm['waypoints']} waypoints each, plus final refinement passes.")
        self.m["fk"].setText(f"{tm['fk_ms']:.1f} ms")
        self.m["fk"].setToolTip(f"Forward kinematics used to verify all {res.num_candidates} candidates at once.")
        self.m["cand"].setText(f"{res.num_within_tolerance} of {res.num_candidates} in tol.")
        self.m["cand"].setToolTip(f"{res.num_within_tolerance} of the {res.num_candidates} PINN candidates landed within "
                                  f"{res.tolerance_mm:g} mm of the target (and passed the limit and singularity checks).")
        if res.dls is not None:
            self.m["dls"].setText(f"{res.dls['error_mm']:.2f} mm")
            self.m["dls"].setToolTip(f"Classical damped-least-squares, 200 iterations, {res.dls['time_ms']:.0f} ms. "
                                     "Reference only: it never supplies a solution.")
        else:
            self.m["dls"].setText("off")

        self.list.clear()
        for s in res.solutions:
            item = QListWidgetItem(_chip(settings.SOLUTION_COLORS[(s.index - 1) % len(settings.SOLUTION_COLORS)]),
                                   f"Solution {s.index}    error {s.error_mm:.2f} mm    w {s.manipulability:.3f}\n"
                                   f"travel {s.distance_rad:.2f} rad    from {s.seed_label}")
            item.setToolTip("error = distance between the reached hand position and the target\n"
                            "w = manipulability (higher is further from a singularity)\n"
                            "travel = joint-space distance from the robot's pose when you pressed Solve\n"
                            "from = the starting pose the PINN was run from to find this solution")
            item.setSizeHint(QSize(0, 60))
            self.list.addItem(item)
        n = len(res.solutions)
        self.lbl_count.setText(f"{n} valid solution{'s' if n != 1 else ''} found" +
                               ("" if n >= self.sp_maxsol.value() else f" (fewer than the {self.sp_maxsol.value()} requested)"))
        if n:
            self._banner(res.message, "ok" if not self.solver.warning else "warn")
            self._select_row(0)
        else:
            for r in range(cfg.N_JOINTS):
                self.tbl.item(r, 2).setText("-")
            self.btn_replay.setEnabled(False)
            self.live["state"].setText("No valid solution")
            self._banner(res.message, "warn")

    # =================================================================== selecting + moving
    def _select_row(self, row):
        if self.result is None or not (0 <= row < len(self.result.solutions)):
            return
        sol = self.result.solutions[row]
        self.selected = sol
        self.list.setCurrentRow(row)
        self.scene.set_robot_color(settings.SOLUTION_COLORS[(sol.index - 1) % len(settings.SOLUTION_COLORS)])
        qd = np.degrees(sol.q)
        for r in range(cfg.N_JOINTS):
            self.tbl.item(r, 2).setText(f"{qd[r]:+.1f}°")
        self.btn_replay.setEnabled(True)
        self._start_motion(sol.q)

    def replay(self):
        if self.selected is None:
            return
        self.q = self.last_start_q.copy()
        self.anim = None
        self._update_live(self.q)
        self.scene.set_pose(self.q)
        self._start_motion(self.selected.q)

    def _start_motion(self, q_goal):
        q_start = self._display_q().copy()
        self.last_start_q = q_start.copy()
        t0 = time.perf_counter()
        traj = trajectory.plan(q_start, q_goal, self.sp_dur.value(), settings.MAX_JOINT_SPEED, self.cmb_profile.currentText())
        times, Q = traj.sample(settings.ANIMATION_HZ)
        path = ee_positions(Q)
        self._traj_ms = (time.perf_counter() - t0) * 1000
        self.scene.begin_motion(path, show_planned=self.chk_planned.isChecked())
        self.m["traj"].setText(f"{self._traj_ms:.2f} ms")
        self.m["traj"].setToolTip(f"{len(Q)} joint-space samples over {traj.duration:.2f} s, plus the forward kinematics "
                                  "for the drawn end-effector path.")
        self.m["peak"].setText(f"{traj.peak_speed:.2f} rad/s")
        self.m["final"].setText("moving...")
        self.anim = {"traj": traj, "path": path, "t0": time.perf_counter(), "q": q_start.copy(), "goal": np.asarray(q_goal, float)}
        self.live["state"].setText("Moving")
        self.progress.setValue(0)
        if not self.timer.isActive():
            self.timer.start()

    def _tick(self):
        a = self.anim
        if a is None:
            self.timer.stop()
            return
        traj = a["traj"]
        elapsed = time.perf_counter() - a["t0"]
        q = traj.at(elapsed)
        a["q"] = q
        ee = self.scene.set_pose(q)
        self.scene.update_trail(int(round(min(elapsed, traj.duration) * settings.ANIMATION_HZ)))
        self._update_live(q, ee)
        self.progress.setValue(int(1000 * min(elapsed / traj.duration, 1.0)))
        self.plotter.render()
        if elapsed >= traj.duration:
            self._finish_motion()

    def _finish_motion(self):
        a = self.anim
        self.q = a["goal"].copy()
        self.anim = None
        self.timer.stop()
        ee = self.scene.set_pose(self.q)
        self.scene.update_trail(10**9)
        self._update_live(self.q, ee)
        err_mm = float(np.linalg.norm(ee - self.target()) * 1000)
        self.m["final"].setText(f"{err_mm:.3f} mm")
        self.live["state"].setText("Target reached" if err_mm <= self.sp_tol.value() else "Stopped (outside tolerance)")
        self.progress.setValue(1000)
        self.plotter.render()

    def reset_robot(self):
        self.anim = None
        self.timer.stop()
        self.q = np.array(cfg.HOME_POSITION, dtype=float)
        self.scene.clear_motion()
        self.scene.set_pose(self.q)
        self.scene.set_robot_color(settings.ROBOT_DEFAULT_COLOR)
        self.list.clearSelection()
        self.selected = None
        self.progress.setValue(0)
        self._update_live(self.q)
        self.live["state"].setText("Idle (home pose)")
        self.plotter.render()

    # =================================================================== live readouts
    def _update_error_label(self, q):
        err = float(np.linalg.norm(ee_position(q) - self.target()) * 1000)
        self.live["err"].setText(f"{err:.2f} mm")

    def _update_live(self, q, ee=None):
        ee = ee_position(q) if ee is None else ee
        self.live["cur"].setText(_fmt_xyz(ee))
        self.live["err"].setText(f"{np.linalg.norm(ee - self.target()) * 1000:.2f} mm")
        lim = np.degrees(joint_limits())
        qd = np.degrees(q)
        for r in range(cfg.N_JOINTS):
            it = self.tbl.item(r, 1)
            it.setText(f"{qd[r]:+.1f}°")
            near = qd[r] - lim[r, 0] < 3.0 or lim[r, 1] - qd[r] < 3.0
            it.setForeground(QColor("#ffc857" if near else "#dfe6f1"))
        if self.anim is None and self.live["state"].text() in ("-", ""):
            self.live["state"].setText("Idle (home pose)")

    # =================================================================== shutdown
    def closeEvent(self, event):
        self.timer.stop()
        for w in (self._worker, self._ws_worker):
            if w is not None and w.isRunning():
                w.wait(5000)
        try:
            self.plotter.close()
        except Exception:
            pass
        super().closeEvent(event)
