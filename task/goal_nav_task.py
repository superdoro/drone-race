"""Point-to-point flight through cylinder clutter as an rsl-rl VecEnv. Simulator access goes through SimBackend only."""

import math

import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from backends.base import DroneState, SimBackend
from control.ctbr import CTBRController
from task.collision import FAR, obstacle_surface_distance
from task.observations import obstacle_obs, proprio_obs
from task.rewards import compute_rewards
from task.sampling import Curriculum, sample_num_obstacles, sample_obstacles, sample_start_goal


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

        # Fixed radius per obstacle slot, shuffled so that every curriculum level sees a mix of sizes.
        num_slots = self.curriculum.max_obstacles
        r_low, r_high = cfg["obstacles"]["radius_range"]
        radii = torch.linspace(r_low, r_high, num_slots)[torch.randperm(num_slots, generator=torch.Generator().manual_seed(0))]
        backend.build(num_envs, radii.tolist(), cfg["obstacles"]["height"])
        self.device = backend.device
        self.obstacle_radii = radii.to(self.device)
        self.controller = CTBRController(backend.drone, **cfg["control"], device=self.device)

        n, dev = num_envs, self.device
        self.episode_length_buf = torch.zeros(n, dtype=torch.long, device=dev)
        self.start = torch.zeros(n, 3, device=dev)
        self.goal = torch.zeros(n, 3, device=dev)
        self.start_dist = torch.zeros(n, device=dev)
        self.prev_dist = torch.zeros(n, device=dev)
        self.obs_xy = torch.zeros(n, num_slots, 2, device=dev)
        self.obs_active = torch.zeros(n, num_slots, dtype=torch.bool, device=dev)
        self.actions = torch.zeros(n, self.num_actions, device=dev)
        self.last_actions = torch.zeros(n, self.num_actions, device=dev)

        self.reset_idx(torch.arange(n, device=dev))
        self._obs = self._compute_obs(self.backend.get_drone_state())

    # ---------------------------------------------------------------- VecEnv API

    def get_observations(self) -> TensorDict:
        return self._obs

    def reset(self) -> tuple[TensorDict, dict]:
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self._obs = self._compute_obs(self.backend.get_drone_state())
        return self._obs, {}

    def step(self, actions: torch.Tensor):
        self.actions = actions.clamp(-1.0, 1.0)
        collective, rates = self.controller.action_to_command(self.actions)
        for _ in range(self.cfg["decimation"]):
            ang_vel = self.backend.get_drone_state().ang_vel
            self.backend.apply_rotor_thrusts(self.controller.rotor_thrusts(collective, rates, ang_vel))
            self.backend.step()
        self.episode_length_buf += 1
        state = self.backend.get_drone_state()

        # Terminations
        acfg = self.cfg["arena"]
        dist = (self.goal - state.pos).norm(dim=-1)
        if self.obs_xy.shape[1] > 0:
            surface = obstacle_surface_distance(state.pos, self.obs_xy, self.obstacle_radii, self.obs_active)
            min_clearance = surface.min(dim=1).values - self.backend.drone.body_radius
        else:
            min_clearance = torch.full_like(dist, FAR)
        center = 0.5 * (self.start[:, :2] + self.goal[:, :2])
        hit_obstacle = min_clearance < 0.0
        out_of_bounds = (
            (state.pos[:, 2] < acfg["ground_z"])
            | (state.pos[:, 2] > acfg["ceiling_z"])
            | ((state.pos[:, :2] - center).norm(dim=-1) > 0.5 * self.start_dist + acfg["horizontal_margin"])
        )
        crashed = hit_obstacle | out_of_bounds
        reached = (dist < self.cfg["goal"]["reach_radius"]) & ~crashed
        time_outs = (self.episode_length_buf >= self.max_episode_length) & ~crashed & ~reached
        dones = crashed | reached | time_outs

        rewards, terms = compute_rewards(
            self.prev_dist, dist, state.ang_vel, actions, self.actions, self.last_actions, min_clearance, crashed,
            reached, self.cfg,
        )
        self.prev_dist = dist
        self.last_actions = self.actions.clone()

        extras = {"time_outs": time_outs, "log": {f"Reward/{k}": v.mean() for k, v in terms.items()}}
        env_ids = dones.nonzero(as_tuple=False).flatten()
        if env_ids.numel() > 0:
            self._log_finished_episodes(env_ids, reached, hit_obstacle, out_of_bounds, time_outs, extras["log"])
            self.reset_idx(env_ids)
            state = self.backend.get_drone_state()
        extras["log"]["Curriculum/level"] = float(self.curriculum.level_idx)

        self._obs = self._compute_obs(state)
        return self._obs, rewards, dones.long(), extras

    # ---------------------------------------------------------------- internals

    def reset_idx(self, env_ids: torch.Tensor) -> None:
        level = self.curriculum.level
        start, goal, quat = sample_start_goal(env_ids.numel(), self.cfg, level, self.device)
        num_active = sample_num_obstacles(env_ids.numel(), level, self.device)
        xy, active = sample_obstacles(start, goal, self.obstacle_radii, num_active, self.cfg)

        self.start[env_ids] = start
        self.goal[env_ids] = goal
        self.start_dist[env_ids] = (goal - start).norm(dim=-1)
        self.prev_dist[env_ids] = self.start_dist[env_ids]
        self.obs_xy[env_ids] = xy
        self.obs_active[env_ids] = active
        self.last_actions[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0

        self.backend.reset_drones(env_ids, start, quat)
        if xy.shape[1] > 0:
            self.backend.set_obstacles(env_ids, xy, active)
        self.backend.set_goal_markers(env_ids, goal)

    def _compute_obs(self, state: DroneState) -> TensorDict:
        policy = torch.cat(
            [
                proprio_obs(state, self.goal, self.last_actions, self.cfg),
                obstacle_obs(state, self.obs_xy, self.obstacle_radii, self.obs_active, self.cfg),
            ],
            dim=-1,
        )
        return TensorDict({"policy": policy}, batch_size=[self.num_envs])

    def _log_finished_episodes(self, env_ids, reached, hit_obstacle, out_of_bounds, time_outs, log: dict) -> None:
        """Per-episode outcome metrics; rsl-rl averages the concatenated tensors over each iteration."""
        success = reached[env_ids]
        log["Metrics/success_rate"] = success.float()
        log["Metrics/crash_obstacle"] = hit_obstacle[env_ids].float()
        log["Metrics/out_of_bounds"] = (out_of_bounds & ~hit_obstacle)[env_ids].float()
        log["Metrics/timeout"] = time_outs[env_ids].float()
        if success.any():
            ok = env_ids[success]
            flight_time = self.episode_length_buf[ok].float() * self.dt
            log["Metrics/time_to_goal_s"] = flight_time
            log["Metrics/avg_speed_mps"] = self.start_dist[ok] / flight_time
        self.curriculum.update(success)
