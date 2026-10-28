"""Analytic drone-vs-obstacle geometry for vertical cylinders that span the whole flight altitude.

Done in torch instead of with the simulator's contact detection so that every backend terminates episodes
identically (Genesis drones do not collide at all).
"""

import torch

FAR = 1e3


def obstacle_surface_distance(drone_pos: torch.Tensor, obs_xy: torch.Tensor, radii: torch.Tensor, active: torch.Tensor):
    """Horizontal distance (N, K) from the drone center to each cylinder surface; inactive slots read FAR."""
    dist = (drone_pos[:, None, :2] - obs_xy).norm(dim=-1) - radii[None]
    return torch.where(active, dist, FAR)


def nearest_obstacles(drone_pos, obs_xy, radii, active, k: int):
    """Indices (N, k) and surface distances (N, k) of the k closest active obstacles, closest first."""
    surface = obstacle_surface_distance(drone_pos, obs_xy, radii, active)
    k = min(k, surface.shape[1])
    dist, idx = torch.topk(surface, k, dim=1, largest=False)
    return idx, dist
