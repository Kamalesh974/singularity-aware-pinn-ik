"""
Small helper to turn a training-history CSV (written by train.train) into a
PNG so results can be inspected visually instead of only via log lines.
"""
import csv
import matplotlib.pyplot as plt


def plot_history(csv_path, out_path, title="Training history"):
    steps, pos_err, loss_sing, mean_w, lr = [], [], [], [], []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            steps.append(int(row["step"]))
            pos_err.append(float(row["pos_err_mm"]))
            loss_sing.append(float(row["loss_singularity"]))
            mean_w.append(float(row["mean_manipulability"]))
            lr.append(float(row["lr"]))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    fig.suptitle(title)

    axes[0].plot(steps, pos_err, color="#2563eb")
    axes[0].set_title("Position error (mm)")
    axes[0].set_xlabel("step")
    axes[0].grid(alpha=0.3)

    axes[1].plot(steps, loss_sing, color="#dc2626")
    axes[1].set_title("Singularity loss (weighted contribution excluded)")
    axes[1].set_xlabel("step")
    axes[1].grid(alpha=0.3)

    axes[2].plot(steps, mean_w, color="#16a34a")
    axes[2].set_title("Mean manipulability of predictions")
    axes[2].set_xlabel("step")
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"wrote {out_path}")


def plot_comparison(results, out_path, title="PINN vs classical DLS near singular configs"):
    """results: dict from evaluate.compare_near_singular"""
    import torch

    labels = ["pos err (mm)", "manipulability", "joint step (rad)"]
    pinn_vals = [
        results["pinn"]["err_mm"].mean().item(),
        results["pinn"]["w"].mean().item(),
        results["pinn"]["step"].mean().item(),
    ]
    dls_vals = [
        results["dls"]["err_mm"].mean().item(),
        results["dls"]["w"].mean().item(),
        results["dls"]["step"].mean().item(),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(title)
    for i, ax in enumerate(axes):
        ax.bar(["PINN", "DLS"], [pinn_vals[i], dls_vals[i]], color=["#2563eb", "#f59e0b"])
        ax.set_title(labels[i])
        ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"wrote {out_path}")


def plot_loss_components(runs, out_path, title="Training loss, by component"):
    """
    runs: list of (csv_path, label, colour). One curve per run in each panel:
    total loss, position error, singularity loss, joint-limit loss.
    """
    panels = [
        ("loss", "Total loss (log scale)", True),
        ("pos_err_mm", "Position error of the final answer (mm)", False),
        ("loss_singularity", "Singularity-avoidance loss (raw)", False),
        ("loss_joint_limits", "Joint-limit loss (raw, log scale)", True),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, (col, name, logy) in zip(axes.flat, panels):
        for path, label, colour in runs:
            steps, vals = [], []
            with open(path, newline="") as f:
                for row in csv.DictReader(f):
                    v = float(row[col])
                    if logy and v <= 0:
                        continue
                    steps.append(int(row["step"]))
                    vals.append(v)
            ax.plot(steps, vals, label=label, color=colour, lw=2)
        ax.set_title(name)
        ax.set_xlabel("training step")
        ax.grid(alpha=0.3)
        if logy:
            ax.set_yscale("log")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(title)
    fig.text(0.5, 0.005, "Total-loss values are not comparable across runs that use different position-loss definitions "
             "(squared vs plain distance) or number of passes; position error (top right) is.",
             ha="center", fontsize=8.5, color="#555")
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(out_path, dpi=130)
    print(f"wrote {out_path}")
