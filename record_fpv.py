"""Record complete episodes from a first-person view, with a map inset and flight data: one mp4 per episode plus one
with all of them.

The policy is heading-invariant (it does not turn its nose toward where it flies), so the camera looks along the
smoothed flight direction, pitch included, instead of along the body x axis, which would mostly show the side.

Usage:
    python record_fpv.py --run logs/rooms3d_ft_replay/<timestamp> --episodes 5 --max   # 6 rooms, everything maxed
    python record_fpv.py --run logs/rooms3d_ft_replay/<timestamp> --level 2
"""

import argparse
import copy
import json
import math
import os

import av
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from backends.base import make_backend  # noqa: E402
from config.drone_params import DRONES  # noqa: E402
from config.task_cfg import get_cfgs  # noqa: E402
from eval import latest_checkpoint, overlay  # noqa: E402
from plot_layouts import ALTITUDE, draw_layout  # noqa: E402
from task.goal_nav_task import GoalNavTask  # noqa: E402
from task.route import route_state  # noqa: E402

W, H, MAP = 1280, 720, 340  # frame size and map inset size (px)
OUTCOME_ZH = {"success": "到達終點", "obstacle": "撞到柱子", "wall": "撞牆", "out": "出界", "timeout": "逾時"}
OUTCOME_KEYS = (("success", "success_rate"), ("obstacle", "crash_obstacle"), ("wall", "crash_wall"),
                ("out", "out_of_bounds"), ("timeout", "timeout"))  # fmt: skip
FONT_PATHS = ("C:/Windows/Fonts/msjh.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")


def load_font(size):
    for path in FONT_PATHS:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


class VideoWriter:
    def __init__(self, path, fps):
        self.container = av.open(path, mode="w")
        self.stream = self.container.add_stream("libx264", rate=fps)
        self.stream.width, self.stream.height, self.stream.pix_fmt = W, H, "yuv420p"
        self.stream.options = {"crf": "20"}

    def write(self, rgb):
        for packet in self.stream.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(rgb), format="rgb24")):
            self.container.mux(packet)

    def close(self):
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()


class MapInset:
    """Top-down map of the episode's layout with the flown trail (colored by altitude) and the view direction."""

    def __init__(self, layout, radii):
        plt.rcParams["font.family"] = ["Microsoft JhengHei", "DejaVu Sans"]
        self.fig = plt.figure(figsize=(MAP / 100, MAP / 100), dpi=100)
        self.ax = self.fig.add_axes([0.02, 0.02, 0.96, 0.96])
        draw_layout(self.ax, layout, radii)
        self.trail = LineCollection([], cmap="viridis", norm=ALTITUDE, lw=1.8)
        self.ax.add_collection(self.trail)
        (self.view,) = self.ax.plot([], [], color="crimson", lw=1.6)
        (self.dot,) = self.ax.plot([], [], "o", ms=6, mfc="gold", mec="k", mew=0.8)

    def render(self, path, forward):
        p = np.asarray(path)
        self.trail.set_segments(np.stack([p[:-1, :2], p[1:, :2]], axis=1) if len(p) > 1 else np.zeros((0, 2, 2)))
        self.trail.set_array(p[:-1, 2] if len(p) > 1 else np.zeros(0))
        x, y = p[-1, :2]
        self.view.set_data([x, x + 1.2 * forward[0]], [y, y + 1.2 * forward[1]])
        self.dot.set_data([x], [y])
        self.fig.canvas.draw()
        return np.asarray(self.fig.canvas.buffer_rgba())[..., :3]

    def close(self):
        plt.close(self.fig)


def compose(view, inset, lines, banner, fonts):
    img = Image.fromarray(view)
    img.paste(Image.fromarray(inset), (W - MAP - 16, 16))
    draw = ImageDraw.Draw(img, "RGBA")
    draw.rectangle([W - MAP - 18, 14, W - 14, MAP + 18], outline=(255, 255, 255, 200), width=2)
    box_h = 18 + 34 * len(lines)
    draw.rectangle([16, H - box_h - 16, 760, H - 16], fill=(10, 14, 20, 170))
    for i, line in enumerate(lines):
        draw.text((30, H - box_h - 6 + 34 * i), line, font=fonts["hud"], fill=(240, 244, 248))
    if banner:
        text, color = banner
        w = draw.textlength(text, font=fonts["banner"])
        draw.rectangle([(W - w) / 2 - 28, H / 2 - 52, (W + w) / 2 + 28, H / 2 + 44], fill=(10, 14, 20, 190))
        draw.text(((W - w) / 2, H / 2 - 46), text, font=fonts["banner"], fill=color)
    return np.asarray(img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="training log directory")
    parser.add_argument("--ckpt", default=None, help="checkpoint file; defaults to the latest in --run")
    parser.add_argument("--level", type=int, default=4, help="curriculum level")
    parser.add_argument("--max", action="store_true", help="6 rooms, 2 partitions, 2 free walls, 5 pillars per room")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None, help="output directory (default: <run>/videos)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    with open(os.path.join(args.run, "cfgs.json")) as f:
        saved = json.load(f)
    env_cfg, train_cfg = overlay(get_cfgs()[0], saved["env_cfg"]), saved["train_cfg"]
    level = dict(env_cfg["curriculum"]["levels"][args.level])
    if args.max:
        level.update(rooms=(6, 6), partitions=(2, 2), free_walls=(2, 2), pillars=(5, 5))
    env_cfg["curriculum"]["levels"][args.level] = level
    env_cfg["curriculum"].update(initial_level=args.level, promote_success_rate=2.0, replay=0.0)
    env_cfg["rooms"]["pool_size"] = 64
    setting = "全部拉滿（6 間房、障礙全滿）" if args.max else f"第 {args.level} 級"

    backend = make_backend(saved["args"]["backend"], DRONES[env_cfg["drone"]], env_cfg["sim_dt"], device=args.device,
                           render_camera=True, seed=args.seed)  # fmt: skip
    env = GoalNavTask(env_cfg, backend, 1)
    runner = OnPolicyRunner(env, copy.deepcopy(train_cfg), log_dir=None, device=str(env.device))
    runner.load(args.ckpt or latest_checkpoint(args.run))
    policy = runner.get_inference_policy(device=str(env.device))

    out_dir = args.out or os.path.join(args.run, "videos")
    os.makedirs(out_dir, exist_ok=True)
    fonts = {"hud": load_font(24), "banner": load_font(48)}
    every = max(1, round(1.0 / (env.dt * args.fps)))
    smooth = 1.0 - math.exp(-env.dt / 0.25)  # view direction follows the flight direction with a 0.25 s lag
    all_writer = VideoWriter(os.path.join(out_dir, f"fpv_{args.episodes}_episodes.mp4"), args.fps)
    obs = env.get_observations()
    with torch.inference_mode():
        for ep in range(args.episodes):
            layout = env.layout[torch.tensor([0], device=env.device)]
            lay = type(layout)(**{k: v[0].cpu() for k, v in vars(layout).items()})
            inset = MapInset(lay, env.pillar_radii.cpu())
            writer = VideoWriter(os.path.join(out_dir, f"fpv_episode_{ep + 1}.mp4"), args.fps)
            state = backend.get_drone_state()
            door = route_state(state.pos, env.layout).door[0]
            forward = door - state.pos[0]
            forward[2] = 0.0
            forward = forward / forward.norm().clamp_min(1e-6)
            path, step, outcome = [state.pos[0].cpu().numpy()], 0, None
            while outcome is None:
                state = backend.get_drone_state()
                pos, vel = state.pos[0], state.lin_vel[0]
                if vel.norm() > 0.4:
                    forward = forward + smooth * (vel / vel.norm() - forward)
                    forward[2] = forward[2].clamp(-0.6, 0.6)
                    forward = forward / forward.norm()
                path.append(pos.cpu().numpy())
                route = route_state(state.pos, env.layout)
                lines = [
                    f"第 {ep + 1} / {args.episodes} 次 · {setting}",
                    f"t = {step * env.dt:5.1f} s    速度 {vel.norm():.1f} m/s    高度 {pos[2]:.2f} m    "
                    f"天花板 {route.ceiling[0]:.2f} m",
                    f"第 {int(route.room[0]) + 1} / {int(lay.num_rooms)} 間房",
                ]
                if step % every == 0:
                    horizontal = forward.clone()
                    horizontal[2] = 0.0
                    eye = pos + 0.08 * horizontal / horizontal.norm().clamp_min(1e-6) + torch.tensor([0.0, 0.0, 0.03], device=pos.device)
                    view = backend.render_view(eye, eye + forward)
                    frame = compose(view, inset.render(path, forward.cpu().numpy()), lines, None, fonts)
                    writer.write(frame)
                    all_writer.write(frame)
                obs, _, dones, extras = env.step(policy(obs))
                step += 1
                if dones[0]:
                    log = extras["log"]
                    outcome = next(name for name, key in OUTCOME_KEYS if log[f"Metrics/{key}"][0] > 0)
            # The env has already reset to the next layout, so the result is shown over the last rendered view.
            ok = outcome == "success"
            banner = (f"{OUTCOME_ZH[outcome]} · {step * env.dt:.1f} s", (120, 230, 140) if ok else (255, 120, 110))
            final = compose(view, inset.render(path, forward.cpu().numpy()), lines, banner, fonts)
            for _ in range(2 * args.fps):  # hold the result for two seconds
                writer.write(final)
                all_writer.write(final)
            writer.close()
            inset.close()
            print(f"episode {ep + 1}: {OUTCOME_ZH[outcome]} after {step * env.dt:.1f} s, {int(lay.num_rooms)} rooms")
    all_writer.close()
    print(f"Saved {args.episodes} episode videos and fpv_{args.episodes}_episodes.mp4 to {out_dir}")


if __name__ == "__main__":
    main()
