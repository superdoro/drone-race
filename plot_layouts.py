"""Top-down pictures of random room layouts (also used by eval.py to draw flown trajectories).

Usage:
    python plot_layouts.py --level 4 --num 12        # sample layouts of a curriculum level -> logs/layouts_level4.png
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from config.task_cfg import get_cfgs  # noqa: E402
from task.rooms import LayoutBatch, LayoutGenerator, layout_capacity  # noqa: E402

OUTCOME_COLORS = {"success": "tab:green", "obstacle": "tab:red", "wall": "tab:purple", "out": "tab:brown",
                  "timeout": "tab:gray"}  # fmt: skip


def draw_layout(ax, lay: LayoutBatch, radii: torch.Tensor, traj=None, outcome: str | None = None) -> None:
    """Draw one layout (a LayoutBatch indexed down to a single env) and optionally a flown xy trajectory."""
    lay = LayoutBatch(**{k: v.detach().cpu() for k, v in vars(lay).items()})
    for x0, y0, x1, y1 in lay.rooms[: int(lay.num_rooms)].tolist():
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, color="tab:blue", alpha=0.06, lw=0))
    for x0, y0, x1, y1 in lay.segments[lay.segment_active].tolist():
        ax.plot([x0, x1], [y0, y1], color="0.2", lw=2.0, solid_capstyle="butt")
    for (x, y), r in zip(lay.pillar_xy[lay.pillar_active].tolist(), radii.cpu()[lay.pillar_active].tolist()):
        ax.add_patch(plt.Circle((x, y), r, color="tab:orange"))
    exits = set(lay.exit_wp[: int(lay.num_rooms)].tolist())
    num_wp = int(lay.exit_wp[int(lay.num_rooms) - 1]) + 1
    route = torch.cat([lay.start[None], lay.waypoints[:num_wp]])
    ax.plot(route[:, 0], route[:, 1], ls="--", lw=0.8, color="tab:blue", alpha=0.6)
    for i, (x, y) in enumerate(lay.waypoints[: num_wp - 1].tolist()):
        ax.plot(x, y, marker="s" if i in exits else ".", ms=5 if i in exits else 4, color="tab:blue")
    ax.plot(*lay.start.tolist(), marker="^", ms=9, color="tab:blue")
    ax.plot(*lay.goal.tolist(), marker="*", ms=13, color="tab:green")
    if traj is not None:
        ax.plot(traj[:, 0], traj[:, 1], lw=1.5, color=OUTCOME_COLORS.get(outcome, "k"))
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def save_grid(items, path: str, title: str | None = None) -> None:
    """items: list of kwargs for draw_layout, each with an optional 'label' for the panel title."""
    cols = min(4, len(items))
    rows = (len(items) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4.5 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, item in zip(axes.flat, items):
        ax.axis("on")
        item = dict(item)
        if "label" in item:
            ax.set_title(item.pop("label"), fontsize=9)
        draw_layout(ax, **item)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
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
