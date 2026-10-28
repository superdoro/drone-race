"""Closed rectangular corridor around the start->goal segment. Pure torch.

Corridor frame: origin at the start (xy), +x toward the goal, +y to the left. The inner wall faces are at
y = +/-half_width, x = -back_margin (behind the start) and x = length + front_margin (beyond the goal).
Walls are always ordered [left, right, back, front].
"""

import torch

NUM_WALLS = 4


def left_normal(direction: torch.Tensor) -> torch.Tensor:
    return torch.stack([-direction[..., 1], direction[..., 0]], dim=-1)


def to_corridor_frame(xy: torch.Tensor, origin: torch.Tensor, direction: torch.Tensor):
    """World xy (N, 2) -> corridor-frame (x, y), each (N,)."""
    rel = xy - origin
    return (rel * direction).sum(dim=-1), (rel * left_normal(direction)).sum(dim=-1)


def wall_distances(pos: torch.Tensor, origin, direction, length, cfg: dict) -> torch.Tensor:
    """Distance (N, 4) from the drone center to each inner wall face; negative once past the face."""
    ccfg = cfg["corridor"]
    x, y = to_corridor_frame(pos[:, :2], origin, direction)
    w = ccfg["half_width"]
    return torch.stack([w - y, w + y, x + ccfg["back_margin"], length + ccfg["front_margin"] - x], dim=-1)


def wall_normals(direction: torch.Tensor) -> torch.Tensor:
    """World-frame unit vectors (N, 4, 2) pointing from the corridor interior toward each wall."""
    left = left_normal(direction)
    return torch.stack([left, -left, -direction, direction], dim=1)


def wall_layout(origin, direction, length, cfg: dict):
    """World xy centers (N, 4, 2) of the wall boxes and the corridor yaw (N,) for the backend.

    Side walls have one fixed length for every env (long enough for the farthest goal) and are centered on the
    corridor, so for short corridors they simply extend past the end walls.
    """
    ccfg = cfg["corridor"]
    w, t = ccfg["half_width"], ccfg["wall_thickness"]
    back, front = ccfg["back_margin"], ccfg["front_margin"]
    side_x = 0.5 * (length + front - back)
    zeros = torch.zeros_like(length)
    local_x = torch.stack([side_x, side_x, zeros - back - 0.5 * t, length + front + 0.5 * t], dim=1)
    local_y = torch.stack([zeros + w + 0.5 * t, zeros - w - 0.5 * t, zeros, zeros], dim=1)
    centers = origin[:, None] + local_x[..., None] * direction[:, None] + local_y[..., None] * left_normal(direction)[:, None]
    return centers, torch.atan2(direction[:, 1], direction[:, 0])


def wall_box_sizes(cfg: dict, max_length: float) -> list[tuple[float, float, float]]:
    """Box sizes (along corridor x, along corridor y, height) in [left, right, back, front] order."""
    ccfg = cfg["corridor"]
    w, t, h = ccfg["half_width"], ccfg["wall_thickness"], ccfg["height"]
    side = (max_length + ccfg["back_margin"] + ccfg["front_margin"] + 2.0 * t, t, h)
    end = (t, 2.0 * (w + t), h)
    return [side, side, end, end]
