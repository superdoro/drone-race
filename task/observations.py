"""Observation builders, split by source so that the obstacle parts can later be replaced by a depth-image encoder."""

import torch

from backends.base import DroneState
from task.geometry import clip_segments_to_disc, nearest, pillar_surface_distance, segment_closest_points
from task.route import RouteState
from utils.math_utils import quat_rotate_inverse


def _to_body(quat: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate world vectors (N, ..., 3) into the body frame of each drone (quat (N, 4))."""
    return quat_rotate_inverse(quat.view(quat.shape[0], *([1] * (v.dim() - 2)), 4).expand(*v.shape[:-1], 4), v)


def _lift(xy: torch.Tensor) -> torch.Tensor:
    return torch.cat([xy, torch.zeros_like(xy[..., :1])], dim=-1)


def proprio_obs(state: DroneState, last_actions: torch.Tensor, cfg: dict) -> torch.Tensor:
    """Velocities and gravity in the body frame (heading invariant), last action and altitude. Dim 14."""
    ocfg = cfg["obs"]
    gravity = torch.tensor([0.0, 0.0, -1.0], device=state.pos.device).expand_as(state.pos)
    return torch.cat(
        [
            _to_body(state.quat, state.lin_vel) / ocfg["lin_vel_scale"],
            state.ang_vel / ocfg["ang_vel_scale"],
            _to_body(state.quat, gravity),
            last_actions,
            state.pos[:, 2:3] / cfg["arena"]["ceiling_z"],
        ],
        dim=-1,
    )


def route_obs(state: DroneState, route: RouteState, cfg: dict) -> torch.Tensor:
    """The exit of the current room and of the next one, in the body frame. Dim 14.

    Only doors are given; the gaps of the partitions inside a room have to be found from the wall observations.
    """
    clip = cfg["obs"]["goal_dist_clip"]

    def target(pos):
        rel = _to_body(state.quat, pos - state.pos)
        dist = rel.norm(dim=-1, keepdim=True)
        return rel / dist.clamp_min(1e-6), dist.clamp_max(clip) / clip

    door_dir, door_dist = target(route.door)
    next_dir, next_dist = target(route.next_door)
    has_next = route.has_next[:, None].float()
    return torch.cat(
        [
            door_dir,
            door_dist,
            _to_body(state.quat, _lift(route.door_normal)),
            route.door_width[:, None],
            route.is_goal[:, None].float(),
            next_dir * has_next,
            next_dist * has_next,
            has_next,
        ],
        dim=-1,
    )


def pillar_obs(state: DroneState, xy, radii, active, cfg: dict) -> torch.Tensor:
    """Body-frame offsets to the k nearest pillar axes plus surface distances. Dim 4 * k.

    Pillars beyond the sensing range (and padding slots) read as zero offset and normalized distance 1.
    """
    ocfg = cfg["obs"]
    k, sense_range = ocfg["num_sensed_pillars"], ocfg["sense_range"]
    n = state.pos.shape[0]
    rel = torch.zeros(n, k, 3, device=state.pos.device)
    dist = torch.full((n, k), sense_range, device=state.pos.device)
    if xy.shape[1] > 0:
        idx, surface = nearest(pillar_surface_distance(state.pos, xy, radii, active), k)
        kk = idx.shape[1]
        rel[:, :kk, :2] = torch.gather(xy, 1, idx[..., None].expand(-1, -1, 2)) - state.pos[:, None, :2]
        rel[:, :kk] = _to_body(state.quat, rel[:, :kk])
        dist[:, :kk] = surface
    return _encode_sensed(rel.flatten(2), dist, sense_range)


def wall_obs(state: DroneState, segments, active, cfg: dict) -> torch.Tensor:
    """The k nearest wall segments: body-frame vectors to the closest point and to both ends of the part within the
    sensing range, plus the distance. Dim 10 * k.

    The ends tell where a wall stops, i.e. where the gap of a partition or a door is. They are ordered so that the
    second is counter-clockwise of the first as seen from the drone, which removes the storage-order ambiguity.
    """
    ocfg = cfg["obs"]
    k, sense_range = ocfg["num_sensed_walls"], ocfg["sense_range"]
    n = state.pos.shape[0]
    closest, dist = segment_closest_points(state.pos, segments, active)
    idx, dist = nearest(dist, k)
    kk = idx.shape[1]
    seg = torch.gather(segments, 1, idx[..., None].expand(-1, -1, 4))
    end0, end1 = clip_segments_to_disc(state.pos[:, None, :2], seg, sense_range)
    rel_c = torch.gather(closest, 1, idx[..., None].expand(-1, -1, 2)) - state.pos[:, None, :2]
    rel_0, rel_1 = end0 - state.pos[:, None, :2], end1 - state.pos[:, None, :2]
    swap = (rel_0[..., 0] * rel_1[..., 1] - rel_0[..., 1] * rel_1[..., 0] < 0.0)[..., None]
    rel_0, rel_1 = torch.where(swap, rel_1, rel_0), torch.where(swap, rel_0, rel_1)

    rel = torch.zeros(n, k, 3, 3, device=state.pos.device)
    rel[:, :kk] = _to_body(state.quat, _lift(torch.stack([rel_c, rel_0, rel_1], dim=2)))
    full_dist = torch.full((n, k), sense_range, device=state.pos.device)
    full_dist[:, :kk] = dist
    return _encode_sensed(rel.flatten(2), full_dist, sense_range)


def privileged_obs(state: DroneState, route: RouteState, elapsed_frac: torch.Tensor, cfg: dict) -> torch.Tensor:
    """Critic-only: the next route waypoint (which may be a partition gap), remaining route length and time used. Dim 6."""
    ocfg = cfg["obs"]
    rel = _to_body(state.quat, route.waypoint - state.pos)
    dist = rel.norm(dim=-1, keepdim=True)
    return torch.cat(
        [
            rel / dist.clamp_min(1e-6),
            dist.clamp_max(ocfg["goal_dist_clip"]) / ocfg["goal_dist_clip"],
            route.remaining[:, None] / ocfg["route_len_scale"],
            elapsed_frac[:, None],
        ],
        dim=-1,
    )


def _encode_sensed(rel: torch.Tensor, dist: torch.Tensor, sense_range: float) -> torch.Tensor:
    """rel (N, k, d) and dist (N, k) -> (N, k * (d + 1)); unsensed entries read zero offset and distance 1."""
    sensed = dist < sense_range
    rel = torch.where(sensed[..., None], rel / sense_range, 0.0)
    dist = torch.where(sensed, dist, sense_range) / sense_range
    return torch.cat([rel, dist[..., None]], dim=-1).flatten(1)
