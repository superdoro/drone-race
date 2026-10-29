"""Analytic geometry for walls (line segments with thickness) and pillars (vertical cylinders). Pure torch.

Walls and pillars span the whole flight altitude, so everything is 2D in xy. Done in torch instead of with the
simulator's contact detection so that every backend terminates episodes identically (Genesis drones do not collide).
"""

import torch

FAR = 1e3


def pillar_surface_distance(pos: torch.Tensor, xy: torch.Tensor, radii: torch.Tensor, active: torch.Tensor):
    """Horizontal distance (N, P) from the drone center to each pillar surface; inactive slots read FAR."""
    dist = (pos[:, None, :2] - xy).norm(dim=-1) - radii[None]
    return torch.where(active, dist, FAR)


def segment_closest_points(pos: torch.Tensor, segments: torch.Tensor, active: torch.Tensor):
    """Closest point (N, S, 2) on each wall center line and its horizontal distance (N, S); inactive slots read FAR."""
    a, ab = segments[..., :2], segments[..., 2:] - segments[..., :2]
    t = ((pos[:, None, :2] - a) * ab).sum(dim=-1) / ab.square().sum(dim=-1).clamp_min(1e-9)
    closest = a + t.clamp(0.0, 1.0)[..., None] * ab
    dist = (pos[:, None, :2] - closest).norm(dim=-1)
    return closest, torch.where(active, dist, FAR)


def clip_segments_to_disc(center: torch.Tensor, segments: torch.Tensor, radius: float):
    """End points (..., 2) of the part of each segment (..., 4) that lies within `radius` of `center` (..., 2).

    Only meaningful for segments that reach into the disc; callers mask the others out.
    """
    a, d = segments[..., :2], segments[..., 2:] - segments[..., :2]
    f = a - center
    qa = d.square().sum(dim=-1).clamp_min(1e-9)
    qb = 2.0 * (f * d).sum(dim=-1)
    qc = f.square().sum(dim=-1) - radius**2
    root = (qb.square() - 4.0 * qa * qc).clamp_min(0.0).sqrt()
    t0 = ((-qb - root) / (2.0 * qa)).clamp(0.0, 1.0)
    t1 = ((-qb + root) / (2.0 * qa)).clamp(0.0, 1.0)
    return a + t0[..., None] * d, a + t1[..., None] * d


def approach_danger(pos, vel, points, clearance, active, radius: float) -> torch.Tensor:
    """Worst closing speed (N,) toward obstacle points (N, M, 2), weighted by closeness: the velocity component toward
    each point, clipped at 0, times 1 - clearance / radius within `radius`. Flying past tangentially costs little."""
    rel = points - pos[:, None, :2]
    closing = (vel[:, None, :2] * rel).sum(dim=-1) / rel.norm(dim=-1).clamp_min(1e-6)
    near = (1.0 - clearance / radius).clamp(0.0, 1.0)
    return torch.where(active, closing.clamp_min(0.0) * near, 0.0).max(dim=1).values


def nearest(dist: torch.Tensor, k: int):
    """Indices and values (N, min(k, M)) of the k smallest entries of each row of `dist` (N, M), closest first."""
    values, idx = torch.topk(dist, min(k, dist.shape[1]), dim=1, largest=False)
    return idx, values


def locate(pos: torch.Tensor, rooms: torch.Tensor, num_rooms: torch.Tensor, eps: float = 1e-3):
    """Index (N,) of the room the drone is in (the later room inside a doorway) and whether it is in any room at all."""
    x, y = pos[:, None, 0], pos[:, None, 1]
    r_idx = torch.arange(rooms.shape[1], device=pos.device)
    inside = (x >= rooms[..., 0] - eps) & (x <= rooms[..., 2] + eps) & (y >= rooms[..., 1] - eps) & (y <= rooms[..., 3] + eps)
    inside &= r_idx[None] < num_rooms[:, None]
    room = torch.where(inside, r_idx[None], -1).max(dim=1).values
    return room.clamp_min(0), inside.any(dim=1)
