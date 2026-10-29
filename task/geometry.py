"""Analytic geometry for walls (line segments with thickness) and pillars (vertical cylinders). Pure torch.

Every obstacle is a vertical extrusion: a 2D shape over a height interval [z_lo, z_hi]. The distance from a point to such
a solid is sqrt(horizontal gap^2 + vertical gap^2), each gap clipped at 0 (the solid is the product of the 2D shape and
the interval). Done in torch instead of with the simulator's contact detection so that every backend terminates
episodes identically (Genesis drones do not collide).
"""

from dataclasses import dataclass

import torch

FAR = 1e3


@dataclass
class ObstacleDistances:
    """Per-step distances from each drone to every wall and pillar slot (inactive slots read FAR)."""

    wall_point: torch.Tensor  # (N, S, 3) closest point on each wall's center sheet
    wall_dist: torch.Tensor  # (N, S) distance to each wall's surface
    pillar_point: torch.Tensor  # (N, P, 3) closest point on each pillar's axis
    pillar_dist: torch.Tensor  # (N, P) distance to each pillar's surface


def obstacle_distances(pos, segments, segment_z, segment_active, half_thickness: float,
                       pillar_xy, pillar_z, pillar_radii, pillar_active) -> ObstacleDistances:  # fmt: skip
    closest, d_wall = segment_closest_points(pos, segments)
    wall_dist, wall_z = solid_distance(pos, d_wall - half_thickness, segment_z, segment_active)
    d_pillar = pillar_axis_distance(pos, pillar_xy)
    pillar_dist, pillar_z_closest = solid_distance(pos, d_pillar - pillar_radii[None], pillar_z, pillar_active)
    return ObstacleDistances(
        wall_point=torch.cat([closest, wall_z[..., None]], dim=-1),
        wall_dist=wall_dist,
        pillar_point=torch.cat([pillar_xy.expand(pos.shape[0], -1, -1), pillar_z_closest[..., None]], dim=-1),
        pillar_dist=pillar_dist,
    )


def pillar_axis_distance(pos: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
    """Horizontal distance (N, P) from the drone center to each pillar axis."""
    return (pos[:, None, :2] - xy).norm(dim=-1)


def segment_closest_points(pos: torch.Tensor, segments: torch.Tensor):
    """Closest point (N, S, 2) on each wall center line and its horizontal distance (N, S)."""
    a, ab = segments[..., :2], segments[..., 2:] - segments[..., :2]
    t = ((pos[:, None, :2] - a) * ab).sum(dim=-1) / ab.square().sum(dim=-1).clamp_min(1e-9)
    closest = a + t.clamp(0.0, 1.0)[..., None] * ab
    return closest, (pos[:, None, :2] - closest).norm(dim=-1)


def solid_distance(pos: torch.Tensor, horizontal_gap: torch.Tensor, z_range: torch.Tensor, active: torch.Tensor):
    """Distance (N, M) from the drone center to extruded solids, and the height (N, M) of the closest point on them.

    horizontal_gap (N, M): horizontal distance to the 2D shape's surface (negative inside); z_range (N, M, 2).
    Inactive slots read FAR.
    """
    z = pos[:, None, 2]
    z_closest = torch.minimum(torch.maximum(z, z_range[..., 0]), z_range[..., 1])
    dist = torch.sqrt(horizontal_gap.clamp_min(0.0).square() + (z - z_closest).square())
    return torch.where(active, dist, FAR), z_closest


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
    """Worst closing speed (N,) toward obstacle points (N, M, 3), weighted by closeness: the velocity component toward
    each point, clipped at 0, times 1 - clearance / radius within `radius`. Flying past tangentially costs little."""
    rel = points - pos[:, None]
    closing = (vel[:, None] * rel).sum(dim=-1) / rel.norm(dim=-1).clamp_min(1e-6)
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
