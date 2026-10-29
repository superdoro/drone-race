"""Flight through a random chain of rooms full of walls, openings and pillars at any height to a goal, as an rsl-rl VecEnv.

Simulator access goes through SimBackend only.
"""

import math

import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from backends.base import DroneState, SimBackend
from control.ctbr import CTBRController
from task.geometry import ObstacleDistances, approach_danger, obstacle_distances
from task.observations import pillar_obs, privileged_obs, proprio_obs, route_obs, wall_obs
from task.rewards import compute_rewards
from task.rooms import LayoutBatch, LayoutPool, layout_capacity
from task.route import RouteState, route_state
from task.sampling import Curriculum, sample_attitude


class GoalNavTask(VecEnv):
    def __init__(self, cfg: dict, backend: SimBackend, num_envs: int):
        if cfg["obs"]["mode"] != "state":
            raise NotImplementedError(f"obs mode '{cfg['obs']['mode']}' is not implemented yet (only 'state').")
        self.cfg = cfg
        self.backend = backend
        self.num_envs = num_envs
        self.num_actions = 4
        self.dt = cfg["sim_dt"] * cfg["decimation"]
        self.max_episode_length = math.ceil(cfg["episode_length_s"] / self.dt)
        self.curriculum = Curriculum(cfg["curriculum"])

        # Fixed radius per pillar slot, shuffled so that every room / level sees a mix of sizes.
        cap = layout_capacity(cfg)
        num_slots = cap["pillars"]
        r_low, r_high = cfg["obstacles"]["radius_range"]
        radii = torch.linspace(r_low, r_high, num_slots)[torch.randperm(num_slots, generator=torch.Generator().manual_seed(0))]
        backend.build(num_envs, radii.tolist(), cap["segments"], cfg["rooms"]["wall_thickness"])
        self.device = backend.device
        self.pillar_radii = radii.to(self.device)
        self.wall_half_thickness = 0.5 * cfg["rooms"]["wall_thickness"]
        self.controller = CTBRController(backend.drone, **cfg["control"], device=self.device)
        self.pool = LayoutPool(cfg, radii.tolist(), self.device, seed=torch.initial_seed() % 2**31)

        n, dev = num_envs, self.device
        self.layout = LayoutBatch.empty(n, cap, dev)
        self.episode_length_buf = torch.zeros(n, dtype=torch.long, device=dev)
        self.time_limit = torch.zeros(n, dtype=torch.long, device=dev)  # per-episode step budget
        self.route_length = torch.zeros(n, device=dev)  # remaining route length at the start
        self.prev_remaining = torch.zeros(n, device=dev)
        self.actions = torch.zeros(n, self.num_actions, device=dev)
        self.last_actions = torch.zeros(n, self.num_actions, device=dev)

        self.reset_idx(torch.arange(n, device=dev))
        self._obs = self._compute_obs(*self._sense(self.backend.get_drone_state()))

    @property
    def goal(self) -> torch.Tensor:
        return self.layout.goal

    # ---------------------------------------------------------------- VecEnv API

    def get_observations(self) -> TensorDict:
        return self._obs

    def reset(self) -> tuple[TensorDict, dict]:
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self._obs = self._compute_obs(*self._sense(self.backend.get_drone_state()))
        return self._obs, {}

    def step(self, actions: torch.Tensor):
        self.actions = actions.clamp(-1.0, 1.0)
        collective, rates = self.controller.action_to_command(self.actions)
        for _ in range(self.cfg["decimation"]):
            ang_vel = self.backend.get_drone_state().ang_vel
            self.backend.apply_rotor_thrusts(self.controller.rotor_thrusts(collective, rates, ang_vel))
            self.backend.step()
        self.episode_length_buf += 1
        state, route, dists = self._sense(self.backend.get_drone_state())

        # Terminations
        rcfg, lay, z = self.cfg["rewards"], self.layout, state.pos[:, 2]
        body_radius = self.backend.drone.body_radius
        wall_clearances = dists.wall_dist - body_radius
        pillar_clearances = dists.pillar_dist - body_radius
        wall_clearance = wall_clearances.min(dim=1).values
        pillar_clearance = pillar_clearances.min(dim=1).values
        floor_clearance = z - self.cfg["arena"]["ground_z"]
        ceiling_clearance = route.ceiling - z - body_radius
        obstacle_clearance = torch.minimum(wall_clearance, pillar_clearance)
        min_clearance = torch.stack([obstacle_clearance, floor_clearance, ceiling_clearance]).min(dim=0).values
        # The route (and so the progress reward) already leads through doors and partition openings, but straight
        # through pillars and free-standing walls; warn about closing in on those before it is too late.
        free = lay.segment_free & lay.segment_active
        radius = rcfg["approach_radius"]
        obstacle_danger = torch.maximum(
            approach_danger(state.pos, state.lin_vel, dists.wall_point, wall_clearances, free, radius),
            approach_danger(state.pos, state.lin_vel, dists.pillar_point, pillar_clearances, lay.pillar_active, radius),
        )
        walls = lay.segment_active & ~lay.segment_free
        wall_danger = approach_danger(state.pos, state.lin_vel, dists.wall_point, wall_clearances, walls, radius)
        hit_obstacle = pillar_clearance < 0.0
        hit_wall = (wall_clearance < 0.0) & ~hit_obstacle
        out_of_bounds = ((floor_clearance < 0.0) | (ceiling_clearance < 0.0) | ~route.in_rooms) & ~hit_obstacle & ~hit_wall
        crashed = hit_obstacle | hit_wall | out_of_bounds
        reached = ((self.goal - state.pos).norm(dim=-1) < self.cfg["goal"]["reach_radius"]) & ~crashed
        time_outs = (self.episode_length_buf >= self.time_limit) & ~crashed & ~reached
        dones = crashed | reached | time_outs

        rewards, terms = compute_rewards(
            self.prev_remaining, route.remaining, state.lin_vel, state.ang_vel, actions, self.actions,
            self.last_actions, min_clearance, obstacle_clearance, obstacle_danger, wall_danger, crashed, reached, self.cfg,
        )  # fmt: skip
        self.prev_remaining = route.remaining
        self.last_actions = self.actions.clone()

        extras = {"time_outs": time_outs, "log": {f"Reward/{k}": v.mean() for k, v in terms.items()}}
        env_ids = dones.nonzero(as_tuple=False).flatten()
        if env_ids.numel() > 0:
            outcome = {"reached": reached, "hit_obstacle": hit_obstacle, "hit_wall": hit_wall,
                       "out_of_bounds": out_of_bounds, "time_outs": time_outs}  # fmt: skip
            self._log_finished_episodes(env_ids, outcome, route.remaining, extras["log"])
            self.reset_idx(env_ids)
            state, route, dists = self._sense(self.backend.get_drone_state())
        extras["log"]["Curriculum/level"] = float(self.curriculum.level_idx)

        self._obs = self._compute_obs(state, route, dists)
        return self._obs, rewards, dones.long(), extras

    # ---------------------------------------------------------------- internals

    def reset_idx(self, env_ids: torch.Tensor) -> None:
        n = env_ids.numel()
        cur = self.curriculum
        batch = self.pool.draw(n, cur.level, cur.levels[: cur.level_idx])
        self.layout.assign(env_ids, batch)
        route_length = route_state(batch.start, batch).remaining
        self.route_length[env_ids] = route_length
        self.prev_remaining[env_ids] = route_length
        tcfg = self.cfg["time_budget"]
        budget = torch.ceil((tcfg["base_s"] + route_length / tcfg["min_avg_speed"]) / self.dt).long()
        self.time_limit[env_ids] = budget.clamp_max(self.max_episode_length)
        self.last_actions[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0

        self.backend.reset_drones(env_ids, batch.start, sample_attitude(n, self.cfg, self.device))
        self.backend.set_obstacles(env_ids, batch.pillar_xy, batch.pillar_z, batch.pillar_active)
        self.backend.set_walls(env_ids, batch.segments, batch.segment_z, batch.segment_active)
        self.backend.set_goal_markers(env_ids, batch.goal)

    def _sense(self, state: DroneState) -> tuple[DroneState, RouteState, ObstacleDistances]:
        lay = self.layout
        dists = obstacle_distances(
            state.pos, lay.segments, lay.segment_z, lay.segment_active, self.wall_half_thickness,
            lay.pillar_xy, lay.pillar_z, self.pillar_radii, lay.pillar_active,
        )  # fmt: skip
        return state, route_state(state.pos, lay), dists

    def _compute_obs(self, state: DroneState, route: RouteState, dists: ObstacleDistances) -> TensorDict:
        lay = self.layout
        policy = torch.cat(
            [
                proprio_obs(state, route, self.last_actions, self.cfg),
                route_obs(state, route, self.cfg),
                pillar_obs(state, dists, lay.pillar_z, self.cfg),
                wall_obs(state, dists, lay.segments, lay.segment_z, self.cfg),
            ],
            dim=-1,
        )
        elapsed = self.episode_length_buf.float() / self.time_limit.float()
        privileged = privileged_obs(state, route, elapsed, self.cfg)
        return TensorDict({"policy": policy, "privileged": privileged}, batch_size=[self.num_envs])

    def _log_finished_episodes(self, env_ids, outcome: dict, remaining: torch.Tensor, log: dict):
        """Per-episode outcome metrics; rsl-rl averages the concatenated tensors over each iteration."""
        success = outcome["reached"][env_ids]
        log["Metrics/success_rate"] = success.float()
        log["Metrics/crash_obstacle"] = outcome["hit_obstacle"][env_ids].float()
        log["Metrics/crash_wall"] = outcome["hit_wall"][env_ids].float()
        log["Metrics/out_of_bounds"] = outcome["out_of_bounds"][env_ids].float()
        log["Metrics/timeout"] = outcome["time_outs"][env_ids].float()
        route_length = self.route_length[env_ids]
        log["Metrics/route_completion"] = torch.where(
            success, 1.0, (1.0 - remaining[env_ids] / route_length.clamp_min(1e-6)).clamp(0.0, 1.0)
        )
        log["Metrics/num_rooms"] = self.layout.num_rooms[env_ids].float()
        if success.any():
            ok = env_ids[success]
            flight_time = self.episode_length_buf[ok].float() * self.dt
            log["Metrics/time_to_goal_s"] = flight_time
            log["Metrics/avg_speed_mps"] = self.route_length[ok] / flight_time
        self.curriculum.update(success)
