"""Top-down pictures of random room layouts (also used by eval.py to draw flown trajectories).

Walls and pillars are styled by their height interval; trajectories are colored by altitude.

Usage:
    python plot_layouts.py --level 4 --num 12        # sample layouts of a curriculum level -> logs/layouts_level4.png
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from config.task_cfg import get_cfgs  # noqa: E402
from task.rooms import LayoutBatch, LayoutGenerator, layout_capacity  # noqa: E402

OUTCOME_COLORS = {"success": "tab:green", "obstacle": "tab:red", "wall": "tab:purple", "out": "tab:brown",
                  "timeout": "tab:gray"}  # fmt: skip
WALL_STYLES = {  # by height interval: full height, standing on the floor (low), not reaching the floor (hanging / beam)
    "full": {"color": "0.2", "lw": 2.0},
    "low": {"color": "tab:olive", "lw": 2.4},
    "raised": {"color": "tab:blue", "lw": 1.6, "ls": (0, (2, 1.5))},
}
ALTITUDE = plt.Normalize(0.0, 4.0)


def draw_layout(ax, lay: LayoutBatch, radii: torch.Tensor, traj=None, outcome: str | None = None) -> None:
    """Draw one layout (a LayoutBatch indexed down to a single env) and optionally a flown trajectory (T, 3), colored by
    altitude."""
    lay = LayoutBatch(**{k: v.detach().cpu() for k, v in vars(lay).items()})
    top = float(lay.segment_z[..., 1].max())
    for x0, y0, x1, y1 in lay.rooms[: int(lay.num_rooms)].tolist():
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, color="tab:blue", alpha=0.06, lw=0))
    segs, seg_z = lay.segments[lay.segment_active], lay.segment_z[lay.segment_active]
    for kind in ("full", "low", "raised"):  # raised last: a window's lintel is drawn dashed over its sill
        for (x0, y0, x1, y1), (z_lo, z_hi) in zip(segs.tolist(), seg_z.tolist()):
            if kind == ("raised" if z_lo > 1e-3 else "full" if z_hi >= top - 1e-3 else "low"):
                ax.plot([x0, x1], [y0, y1], solid_capstyle="butt", **WALL_STYLES[kind])
    pillar_z = lay.pillar_z[lay.pillar_active].tolist()
    for (x, y), r, (z_lo, z_hi) in zip(lay.pillar_xy[lay.pillar_active].tolist(), radii.cpu()[lay.pillar_active].tolist(), pillar_z):
        if z_lo > 1e-3:  # hanging: outline only
            ax.add_patch(plt.Circle((x, y), r, fill=False, ec="tab:orange", lw=1.2, ls="--"))
        else:
            ax.add_patch(plt.Circle((x, y), r, color="tab:orange", alpha=1.0 if z_hi >= top - 1e-3 else 0.45))
    exits = set(lay.exit_wp[: int(lay.num_rooms)].tolist())
    num_wp = int(lay.exit_wp[int(lay.num_rooms) - 1]) + 1
    route = torch.cat([lay.start[None], lay.waypoints[:num_wp]])
    ax.plot(route[:, 0], route[:, 1], ls="--", lw=0.8, color="tab:blue", alpha=0.6)
    for i, (x, y, _) in enumerate(lay.waypoints[: num_wp - 1].tolist()):
        ax.plot(x, y, marker="s" if i in exits else ".", ms=5 if i in exits else 4, color="tab:blue")
    ax.plot(*lay.start[:2].tolist(), marker="^", ms=9, color="tab:blue")
    ax.plot(*lay.goal[:2].tolist(), marker="*", ms=13, color="tab:green")
    if traj is not None:
        traj = torch.as_tensor(traj, dtype=torch.float32)
        points = traj[:, None, :2]
        lines = LineCollection(torch.cat([points[:-1], points[1:]], dim=1).numpy(), cmap="viridis", norm=ALTITUDE, lw=1.6)
        lines.set_array(traj[:-1, 2].numpy())
        ax.add_collection(lines)
        ax.plot(*traj[-1, :2].tolist(), marker="X", ms=8, color=OUTCOME_COLORS.get(outcome, "k"))
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def save_grid(items, path: str, title: str | None = None) -> None:
    """items: list of kwargs for draw_layout, each with an optional 'label' for the panel title."""
    cols = min(4, len(items))
    rows = (len(items) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4.5 * rows + 0.6), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, item in zip(axes.flat, items):
        ax.axis("on")
        item = dict(item)
        if "label" in item:
            ax.set_title(item.pop("label"), fontsize=9)
        draw_layout(ax, **item)
    handles = [Line2D([], [], label=label, **WALL_STYLES[kind])
               for kind, label in (("full", "full-height wall"), ("low", "low wall / sill"),
                                   ("raised", "hanging wall / lintel / beam"))]  # fmt: skip
    handles += [Line2D([], [], ls="", marker="o", ms=8, color="tab:orange", label="pillar"),
                Line2D([], [], ls="", marker="o", ms=8, color="tab:orange", alpha=0.45, label="low pillar"),
                Line2D([], [], ls="", marker="o", ms=8, mfc="none", mec="tab:orange", label="hanging pillar")]  # fmt: skip
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=9, frameon=False)
    if any(item.get("traj") is not None for item in items):
        fig.colorbar(plt.cm.ScalarMappable(ALTITUDE, "viridis"), ax=axes, shrink=0.5, label="altitude (m)")
    if title:
        fig.suptitle(title)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, default=-1)
    parser.add_argument("--num", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg, _ = get_cfgs()
    levels = cfg["curriculum"]["levels"]
    level_idx = args.level % len(levels)
    cap = layout_capacity(cfg)
    radii = torch.linspace(*cfg["obstacles"]["radius_range"], cap["pillars"])
    gen = LayoutGenerator(cfg, radii.tolist(), seed=args.seed)
    batch = LayoutBatch.pack([gen.sample(levels[level_idx]) for _ in range(args.num)], cap, "cpu")
    os.makedirs("logs", exist_ok=True)
    out = args.out or os.path.join("logs", f"layouts_level{level_idx}.png")
    save_grid([{"lay": batch[i], "radii": radii} for i in range(args.num)], out, f"curriculum level {level_idx}")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
