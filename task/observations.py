"""Observation builders, split by source so that the obstacle part can later be replaced by a depth-image encoder."""

import torch

from backends.base import DroneState
from task.collision import nearest_obstacles
from task.corridor import NUM_WALLS, wall_distances, wall_normals
from utils.math_utils import quat_rotate_inverse


def proprio_obs(state: DroneState, goal: torch.Tensor, last_actions: torch.Tensor, cfg: dict) -> torch.Tensor:
    """Goal direction/distance, velocities and gravity, all in the body frame (heading invariant). Dim 17."""
    ocfg = cfg["obs"]
    rel_goal = quat_rotate_inverse(state.quat, goal - state.pos)
    dist = rel_goal.norm(dim=-1, keepdim=True)
    gravity = torch.tensor([0.0, 0.0, -1.0], device=goal.device).expand_as(rel_goal)
    return torch.cat(
        [
            rel_goal / dist.clamp_min(1e-6),
            dist.clamp_max(ocfg["goal_dist_clip"]) / ocfg["goal_dist_clip"],
            quat_rotate_inverse(state.quat, state.lin_vel) / ocfg["lin_vel_scale"],
            state.ang_vel / ocfg["ang_vel_scale"],
            quat_rotate_inverse(state.quat, gravity),
            last_actions,
        ],
        dim=-1,
    )


def obstacle_obs(state: DroneState, obs_xy, radii, active, cfg: dict) -> torch.Tensor:
    """Body-frame offsets to the k nearest sensed obstacle axes plus surface distances. Dim 4 * k.

    Obstacles beyond the sensing range (and padding slots) read as zero offset and normalized distance 1.
    """
    ocfg = cfg["obs"]
    k, sense_range = ocfg["num_sensed_obstacles"], ocfg["sense_range"]
    n = state.pos.shape[0]
    rel = torch.zeros(n, k, 3, device=state.pos.device)
    dist = torch.full((n, k), sense_range, device=state.pos.device)
    if obs_xy.shape[1] > 0:
        idx, surface = nearest_obstacles(state.pos, obs_xy, radii, active, k)
        kk = idx.shape[1]
        centers = torch.gather(obs_xy, 1, idx[..., None].expand(-1, -1, 2))
        rel[:, :kk, :2] = centers - state.pos[:, None, :2]
        rel[:, :kk] = quat_rotate_inverse(state.quat[:, None].expand(-1, kk, -1), rel[:, :kk])
        dist[:, :kk] = surface
    return _encode_sensed(rel, dist, sense_range)


def wall_obs(state: DroneState, origin, direction, length, cfg: dict) -> torch.Tensor:
    """Body-frame vector to the closest point of each corridor wall plus its distance. Dim 4 * 4.

    Same encoding as `obstacle_obs`: walls beyond the sensing range read as zero offset and distance 1.
    """
    dist = wall_distances(state.pos, origin, direction, length, cfg)
    normals = wall_normals(direction)
    rel = torch.zeros(*dist.shape, 3, device=dist.device)
    rel[..., :2] = normals * dist[..., None]
    rel = quat_rotate_inverse(state.quat[:, None].expand(-1, NUM_WALLS, -1), rel)
    return _encode_sensed(rel, dist, cfg["obs"]["sense_range"])


def _encode_sensed(rel: torch.Tensor, dist: torch.Tensor, sense_range: float) -> torch.Tensor:
    sensed = dist < sense_range
    rel = torch.where(sensed[..., None], rel / sense_range, 0.0)
    dist = torch.where(sensed, dist, sense_range) / sense_range
    return torch.cat([rel, dist[..., None]], dim=-1).flatten(1)
