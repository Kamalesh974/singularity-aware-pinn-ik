# Singularity-Aware PINN Inverse Kinematics

A physics-informed neural network (PINN) that solves inverse kinematics for a 7-DOF redundant robot arm (Franka Emika Panda), trained self-supervised — no labeled IK dataset — through a differentiable forward-kinematics layer. It's benchmarked against a classical damped-least-squares (DLS) solver, and comes with an interactive 3D simulator built on the trained model.

## What this actually does, honestly

- **Solves position-only IK** for a 7R arm: given a target (x, y, z), predicts 7 joint angles. It does not predict orientation.
- **Trades off accuracy vs. safety**, and that trade-off is explicit, not hidden:
  - `runs/final_model.pt` (the default) — most accurate single-shot model: ~4mm mean position error, ~13mm 95th percentile, on tasks drawn from the training distribution.
  - `runs/expE_h512_l40_mine.pt` — the singularity-aware model: ~6.6mm mean error, but ~2.3x the classical baseline's manipulability near singular configurations (i.e. genuinely safer there), vs. the default model which is actually *less* safe than the classical baseline.
- **Does not beat a converged classical solver on raw accuracy.** DLS converges to ~0.1–0.5mm given enough iterations; a single-shot (or few-pass) PINN doesn't. The PINN's advantage shows up at small compute budgets (a few network passes vs. many DLS iterations) and in the singularity-avoidance behavior, not in final accuracy at convergence.
- These numbers, and the debugging history behind them (several real bugs found via testing — a Jacobian indexing bug, an unreachable-training-target bug, an inert-loss-term bug), are described in more detail under [Results](#results) and in the code comments where the fixes were made.

## How it works

```
Target (x, y, z) ──► PINN (seed-conditioned residual network) ──► Δθ
                              │                                    │
                     current joint angles θ ◄──────────────────────┘
                              │
                     θ_next = θ + Δθ   (repeated for several refinement passes)
                              │
                     Forward kinematics (independent check) ──► reached position, error
```

The network is **seed-conditioned**: it takes the current joint configuration as an input, not just the target, and predicts a small step. This is what lets it handle the arm's redundancy (7 joints, only 3 position constraints — many valid joint configurations reach the same point) without averaging conflicting solutions into garbage. Multiple valid IK solutions for one target are found by running the network from many different starting poses.

The training loss has three physics-informed terms, computed through a differentiable forward-kinematics layer:

```
loss = position_error(FK(θ), target) + λ_limits · joint_limit_violation(θ) + λ_singularity · singularity_penalty(θ)
```

No labeled dataset is used — every training example is generated on the fly (random joint configurations, small random target perturbations), and the position "label" is just whatever forward kinematics says the true answer is.

## Repository layout

```
pinn_ik/            Core library
  robot_config.py     Single source of truth for the robot: DH parameters, joint limits, tool offset
  robot.py            Differentiable forward kinematics, Jacobian, manipulability (PyTorch)
  model.py            The PINN architecture (seed-conditioned residual network, multi-pass refinement)
  losses.py           Position / joint-limit / singularity-avoidance loss terms
  train.py            Self-supervised training loop
  baseline_ik.py       Classical damped-least-squares IK solver (the comparison baseline)
  evaluate.py          Accuracy + near-singularity comparison against the baseline
  mujoco_scene.py      MuJoCo scene built from the same DH table (kinematics cross-check)
  viz_mujoco.py         Real-time MuJoCo tracking demo (PINN vs. DLS side by side)

ik_simulator/        Interactive 3D application (PyVista + PySide6)
  gui.py               Main window: XYZ input, 3D view, solutions, live state, research metrics
  pinn_solver.py       Multi-solution IK search (many start poses, filtered + verified candidates)
  scene3d.py           3D robot rendering
  trajectory.py        Smooth (minimum-jerk) joint-space trajectory planning
  workspace.py         Empirical reachability check (sampled robot workspace)
  selftest.py          Scripted end-to-end GUI test

pinn_app.py          CLI: train / evaluate / solve / viz
run_simulator.py     Launcher for the interactive simulator
tests/               Correctness tests (FK vs. autograd, FK vs. MuJoCo, simulator behavior)
runs/                Training run checkpoints, logs, and plots
slides/              Project review slide deck
```

## Setup

Requires Python 3.11+ (this project was built and tested on 3.11.8).

```bash
pip install -r requirements.txt
```

This installs PyTorch, NumPy, SciPy, MuJoCo, PyVista, PyVistaQt, and PySide6.

## Usage

**Interactive 3D simulator** — enter a target XYZ, get up to 5 verified IK solutions, watch the arm move there along a smooth trajectory:

```bash
python -m ik_simulator
# or, to use a specific checkpoint:
python -m ik_simulator --checkpoint runs/expE_h512_l40_mine.pt
```

**Command-line tools** (`pinn_app.py`):

```bash
python pinn_app.py train    --steps 20000 --hidden 512 --pos-loss l2 --out runs/mine.pt
python pinn_app.py evaluate --checkpoint runs/final_model.pt
python pinn_app.py solve    --target 0.45 0.10 0.60
python pinn_app.py viz      --path circle              # live MuJoCo window, PINN vs. DLS
python pinn_app.py viz      --start singular --record demo.gif
```

**Tests:**

```bash
python tests/test_kinematics.py       # FK/Jacobian correctness (cross-checked against autograd)
python tests/test_mujoco_scene.py     # MuJoCo model matches the PyTorch FK
python tests/test_ik_simulator.py     # simulator never fabricates solutions, detects robot mismatches
python -m ik_simulator --selftest some_folder   # scripted end-to-end GUI run, saves screenshots
```

## Results

Full experiment history — including three real bugs found during development and how they were diagnosed and fixed — is in the code comments (`pinn_ik/train.py`, `pinn_ik/model.py`) and `runs/*.csv`. Summary:

| Model | Mean position error | 95th percentile | Near-singularity manipulability |
|---|---|---|---|
| `final_model.pt` (default) | ~4.1 mm | ~13.2 mm | 0.0052 (less safe than DLS) |
| `expE_h512_l40_mine.pt` | ~6.6 mm | ~16.2 mm | **0.0217** (~2.3x DLS) |
| Classical DLS (reference) | ~0.1–0.5 mm | — | 0.0096 |

Loss curves and training-progress plots for each run are in `runs/*.png`.

**Known limitations:**
- The DH parameters are the commonly published values for the Panda, verified for internal self-consistency (PyTorch FK ≡ autograd ≡ MuJoCo ≡ MATLAB Robotics System Toolbox, to ~1e-9 m) but not independently checked against Franka Emika's official spec.
- The classical baseline uses simple adaptive-damping DLS, not a null-space-optimizing variant — a fairer safety comparison would add that.
- No collision checking (self-collision or environment) is modeled anywhere in this project.
- Orientation is not solved or represented.
