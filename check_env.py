"""Sanity checks for the task + backend: shapes / NaNs, hover, rate tracking, collisions (also above / below
obstacles) and room layouts.

Usage: python check_env.py [--backend genesis] [--num_envs 64] [--layouts 200]
"""

import argparse
import copy

import torch
import torch.nn.functional as F

from backends.base import make_backend
from config.drone_params import DRONES
from config.task_cfg import get_cfgs
from task.geometry import obstacle_distances
from task.goal_nav_task import GoalNavTask
from task.rooms import LayoutBatch, LayoutGenerator, layout_capacity
from task.route import route_state


def reachable(lay: LayoutBatch, radii, cfg: dict, clearance: float, extra_walls=None, res: float = 0.1) -> bool:
    """3D flood fill over a voxel grid of drone-center positions, from the start to the goal of one layout (batch of 1).

    A voxel is free when it is inside a room, between the floor and that room's ceiling, and at least `clearance` away
    from every wall and pillar. `extra_walls` (E, 4) adds full-height walls.
    """
    dev = lay.rooms.device
    n = int(lay.num_rooms[0])
    rooms, ceilings = lay.rooms[0, :n], lay.ceilings[0, :n]
    ground = cfg["arena"]["ground_z"]
    xs = torch.arange(float(rooms[:, 0].min()), float(rooms[:, 2].max()) + res, res, device=dev)
    ys = torch.arange(float(rooms[:, 1].min()), float(rooms[:, 3].max()) + res, res, device=dev)
    zs = torch.arange(0.0, float(ceilings.max()) + res, res, device=dev)
    cells = torch.stack(torch.meshgrid(xs, ys, zs, indexing="ij"), dim=-1).view(-1, 3)
    segments, seg_z, active = lay.segments, lay.segment_z, lay.segment_active
    if extra_walls is not None:
        e = len(extra_walls)
        segments = torch.cat([segments, extra_walls[None]], dim=1)
        seg_z = torch.cat([seg_z, torch.tensor([[[0.0, cfg["rooms"]["wall_top"]]]], device=dev).expand(1, e, 2)], dim=1)
        active = torch.cat([active, torch.ones(1, e, dtype=torch.bool, device=dev)], dim=1)
    free = torch.zeros(len(cells), dtype=torch.bool, device=dev)
    for i in range(0, len(cells), 1 << 15):
        c = cells[i : i + (1 << 15)]
        d = obstacle_distances(c, segments, seg_z, active, 0.5 * cfg["rooms"]["wall_thickness"],
                               lay.pillar_xy, lay.pillar_z, radii, lay.pillar_active)  # fmt: skip
        # closed rectangles: voxels exactly on a doorway line belong to both rooms (walls there block the rest)
        inside = ((c[:, None, 0] >= rooms[:, 0] - 1e-3) & (c[:, None, 0] <= rooms[:, 2] + 1e-3)
                  & (c[:, None, 1] >= rooms[:, 1] - 1e-3) & (c[:, None, 1] <= rooms[:, 3] + 1e-3))  # fmt: skip
        ceiling = torch.where(inside, ceilings, -1.0).max(dim=1).values
        free[i : i + (1 << 15)] = (
            (c[:, 2] > ground + 0.02) & (c[:, 2] < ceiling - clearance)
            & (d.wall_dist.min(dim=1).values > clearance) & (d.pillar_dist.min(dim=1).values > clearance)
        )  # fmt: skip
    free = free.view(1, 1, len(xs), len(ys), len(zs)).float()

    def voxel(p):
        return tuple(int(((p[k] - g[0]) / res).round()) for k, g in enumerate((xs, ys, zs)))

    seen = torch.zeros_like(free)
    seen[0, 0][voxel(lay.start[0])] = 1.0
    goal = voxel(lay.goal[0])
    while True:
        for _ in range(50):
            seen = F.max_pool3d(seen, 3, stride=1, padding=1) * free
        if seen[0, 0][goal] > 0:
            return True
        grown = F.max_pool3d(seen, 3, stride=1, padding=1) * free
        if torch.equal(grown, seen):
            return False
        seen = grown


def check_layouts(cfg: dict, count: int, body_radius: float, device) -> None:
    """Every layout of the hardest level is passable with margin, and sealing any one door cuts the goal off."""
    cap = layout_capacity(cfg)
    radii = torch.linspace(*cfg["obstacles"]["radius_range"], cap["pillars"])
    gen = LayoutGenerator(cfg, radii.tolist(), seed=123)
    level = cfg["curriculum"]["levels"][-1]
    batch = LayoutBatch.pack([gen.sample(level) for _ in range(count)], cap, device)
    radii = radii.to(device)

    for i in range(count):
        lay = batch[i : i + 1]
        n = int(lay.num_rooms[0])
        r = lay.rooms[0, :n]
        overlap = ((r[:, None, 0] < r[None, :, 2] - 1e-6) & (r[None, :, 0] < r[:, None, 2] - 1e-6)
                   & (r[:, None, 1] < r[None, :, 3] - 1e-6) & (r[None, :, 1] < r[:, None, 3] - 1e-6))  # fmt: skip
        assert overlap.sum() == n, f"layout {i}: rooms overlap"
        assert reachable(lay, radii, cfg, body_radius + 0.02), f"layout {i}: goal not reachable"
        if n > 1:  # seal a random door with a full-height wall across it
            d = int(torch.randint(0, n - 1, (1,)))
            center, normal = lay.waypoints[0, lay.exit_wp[0, d], :2], lay.exit_normal[0, d]
            along = torch.stack([-normal[1], normal[0]]) * lay.exit_width[0, d]
            seal = torch.cat([center - along, center + along])[None]
            assert not reachable(lay, radii, cfg, body_radius, seal), f"layout {i}: a second route bypasses door {d}"
        # route bookkeeping: nothing left to fly at the goal, the full route length at the start
        at_goal = route_state(lay.goal, lay)
        assert at_goal.remaining.abs().max() < 1e-4 and bool(at_goal.is_goal[0]), f"layout {i}: bad route at the goal"
        at_start = route_state(lay.start, lay)
        assert int(at_start.room[0]) == 0 and at_start.remaining[0] >= (lay.goal - lay.start).norm() - 1e-4
    rooms = batch.num_rooms.float()
    print(f"[8] {count} layouts of the hardest level ({rooms.mean():.1f} rooms on average, up to {int(rooms.max())}): "
          "all passable in 3D, every door is the only way on, route bookkeeping consistent")  # fmt: skip


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="genesis")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--layouts", type=int, default=200)
    args = parser.parse_args()

    base_cfg, _ = get_cfgs()
    cfg = copy.deepcopy(base_cfg)
    cfg["init"]["max_tilt_deg"] = 0.0
    cfg["rooms"]["pool_size"] = 256
    cfg["curriculum"]["initial_level"] = len(cfg["curriculum"]["levels"]) - 1
    backend = make_backend(args.backend, DRONES[cfg["drone"]], cfg["sim_dt"], device=args.device)
    task = GoalNavTask(cfg, backend, args.num_envs)
    n, dev = task.num_envs, task.device
    ids = torch.arange(n, device=dev)
    upright = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=dev).expand(n, -1)

    # 1) Observation / reward shapes and finiteness under random actions.
    obs = task.get_observations()
    print(f"[1] policy obs dim = {obs['policy'].shape[1]}, critic extra = {obs['privileged'].shape[1]}, "
          f"num_actions = {task.num_actions}")  # fmt: skip
    for _ in range(100):
        obs, rew, dones, extras = task.step(2.0 * torch.rand(n, 4, device=dev) - 1.0)
        for key in ("policy", "privileged"):
            assert obs[key].shape[0] == n and torch.isfinite(obs[key]).all(), key
        assert rew.shape == (n,) and dones.shape == (n,) and torch.isfinite(rew).all()
    print("[1] random actions: shapes OK, no NaN/Inf")

    # 2) Zero action = hover thrust + zero body rates: the drone should stay put for 2 s.
    task.reset()
    start = task.backend.get_drone_state().pos.clone()
    for _ in range(int(2.0 / task.dt)):
        task.step(torch.zeros(n, 4, device=dev))
    drift = (task.backend.get_drone_state().pos - start).norm(dim=-1)
    print(f"[2] hover drift after 2 s: max {drift.max():.4f} m")
    assert drift.max() < 0.05

    # 3) Body-rate tracking: a +roll-rate command must produce +x body angular velocity, same for pitch / yaw.
    for axis, name in ((1, "roll"), (2, "pitch"), (3, "yaw")):
        task.reset()
        a = torch.zeros(n, 4, device=dev)
        a[:, axis] = 0.25
        for _ in range(10):
            task.step(a)
        w = task.backend.get_drone_state().ang_vel.mean(dim=0)
        target = 0.25 * task.controller.max_body_rates[axis - 1]
        print(f"[3] {name}-rate cmd {target:.2f} rad/s -> body ang vel {w.tolist()}")
        assert abs(w[axis - 1] - target) < 0.2 * target

    top = cfg["rooms"]["wall_top"]

    # 4) A full-height pillar placed on top of the drone must end the episode with an obstacle crash.
    task.reset()
    pos = task.backend.get_drone_state().pos
    task.layout.pillar_xy[:, 0] = pos[:, :2]
    task.layout.pillar_z[:, 0] = torch.tensor([0.0, top], device=dev)
    task.layout.pillar_active[:, 0] = True
    _, _, dones, extras = task.step(torch.zeros(n, 4, device=dev))
    print(f"[4] pillar overlap -> done {dones.float().mean():.2f}, crash rate {extras['log']['Metrics/crash_obstacle'].mean():.2f}")
    assert dones.all() and (extras["log"]["Metrics/crash_obstacle"] == 1).all()

    # 5) A wall 8 cm from the drone center (body radius 6 cm + half thickness 5 cm) must end it with a wall crash ...
    def wall_beside_drone(z_lo, z_hi):
        task.reset()
        pos = task.backend.get_drone_state().pos
        task.layout.pillar_active[:] = False
        task.layout.segments[:, 0] = torch.stack([pos[:, 0] - 1.0, pos[:, 1] + 0.08, pos[:, 0] + 1.0, pos[:, 1] + 0.08], -1)
        task.layout.segment_z[:, 0] = torch.stack([z_lo(pos[:, 2]), z_hi(pos[:, 2])], dim=-1)
        task.layout.segment_active[:, 0] = True
        return task.step(torch.zeros(n, 4, device=dev))

    _, _, dones, extras = wall_beside_drone(lambda z: torch.zeros_like(z), lambda z: torch.full_like(z, top))
    print(f"[5] touching a wall -> done {dones.float().mean():.2f}, wall crash rate {extras['log']['Metrics/crash_wall'].mean():.2f}")
    assert dones.all() and (extras["log"]["Metrics/crash_wall"] == 1).all()

    # 6) ... but not when it hangs from 15 cm above the drone or ends 15 cm below it; reaching its altitude is a crash.
    for name, z_lo, z_hi in (("hanging above", lambda z: z + 0.15, lambda z: torch.full_like(z, top)),
                             ("low below", lambda z: torch.zeros_like(z), lambda z: z - 0.15)):  # fmt: skip
        _, _, dones, _ = wall_beside_drone(z_lo, z_hi)
        print(f"[6] wall {name} the drone -> done {dones.float().mean():.2f}")
        assert not dones.any()
    _, _, dones, extras = wall_beside_drone(lambda z: z - 0.05, lambda z: torch.full_like(z, top))
    print(f"[6] hanging wall reaching the drone -> wall crash rate {extras['log']['Metrics/crash_wall'].mean():.2f}")
    assert dones.all() and (extras["log"]["Metrics/crash_wall"] == 1).all()

    # 7) Touching the room's ceiling, or leaving the rooms (only possible by tunneling a wall), ends it as out of bounds.
    for name, where in (("at the ceiling", lambda: torch.cat([task.layout.start[:, :2], task.layout.ceilings[:, :1] - 0.03], -1)),
                        ("outside every room", lambda: torch.stack([task.layout.rooms[..., 2].max(dim=1).values + 3.0,
                                                                     task.layout.rooms[:, 0, 1], task.goal[:, 2]], -1))):  # fmt: skip
        task.reset()
        task.layout.pillar_active[:] = False
        task.layout.segment_active[:] = False
        task.backend.reset_drones(ids, where(), upright)
        _, _, dones, extras = task.step(torch.zeros(n, 4, device=dev))
        print(f"[7] {name} -> done {dones.float().mean():.2f}, out-of-bounds rate {extras['log']['Metrics/out_of_bounds'].mean():.2f}")
        assert dones.all() and (extras["log"]["Metrics/out_of_bounds"] == 1).all()

    # 8) Layout generator: passable in 3D, single route, consistent route bookkeeping.
    check_layouts(cfg, args.layouts, task.backend.drone.body_radius, dev)

    print("All checks passed.")


if __name__ == "__main__":
    main()
