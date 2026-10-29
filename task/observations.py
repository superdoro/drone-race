"""Observation builders, split by source so that the obstacle parts can later be replaced by a depth-image encoder."""

import torch

from backends.base import DroneState
from task.geometry import ObstacleDistances, clip_segments_to_disc, nearest
from task.route import RouteState
from utils.math_utils import quat_rotate_inverse


def _to_body(quat: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate world vectors (N, ..., 3) into the body frame of each drone (quat (N, 4))."""
    return quat_rotate_inverse(quat.view(quat.shape[0], *([1] * (v.dim() - 2)), 4).expand(*v.shape[:-1], 4), v)


def _lift(xy: torch.Tensor, z: torch.Tensor | None = None) -> torch.Tensor:
    return torch.cat([xy, torch.zeros_like(xy[..., :1]) if z is None else z[..., None]], dim=-1)


def _gather(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """x (N, M, ...) indexed by idx (N, k) along dim 1."""
    return torch.gather(x, 1, idx.view(*idx.shape, *([1] * (x.dim() - 2))).expand(*idx.shape, *x.shape[2:]))


def proprio_obs(state: DroneState, route: RouteState, last_actions: torch.Tensor, cfg: dict) -> torch.Tensor:
    """Velocities and gravity in the body frame (heading invariant), last action, altitude and room ceiling. Dim 15."""
    ocfg, top = cfg["obs"], cfg["rooms"]["wall_top"]
    gravity = torch.tensor([0.0, 0.0, -1.0], device=state.pos.device).expand_as(state.pos)
    return torch.cat(
        [
            _to_body(state.quat, state.lin_vel) / ocfg["lin_vel_scale"],
            state.ang_vel / ocfg["ang_vel_scale"],
            _to_body(state.quat, gravity),
            last_actions,
            state.pos[:, 2:3] / top,
            (route.ceiling[:, None] - state.pos[:, 2:3]) / top,
        ],
        dim=-1,
    )


def route_obs(state: DroneState, route: RouteState, cfg: dict) -> torch.Tensor:
    """The exit of the current room (with the height of its opening) and of the next one, in the body frame. Dim 16.

    Only doors are given; the openings of the partitions inside a room have to be found from the wall observations.
    """
    clip, top = cfg["obs"]["goal_dist_clip"], cfg["rooms"]["wall_top"]

    def target(pos):
        rel = _to_body(state.quat, pos - state.pos)
        dist = rel.norm(dim=-1, keepdim=True)
        return rel / dist.clamp_min(1e-6), dist.clamp_max(clip) / clip

    door_dir, door_dist = target(route.door)
    next_dir, next_dist = target(route.next_door)
    has_next = route.has_next[:, None].float()
    door_z = torch.where(route.is_goal[:, None], 0.0, route.door_z - state.pos[:, 2:3]) / top
    return torch.cat(
        [
            door_dir,
            door_dist,
            _to_body(state.quat, _lift(route.door_normal)),
            route.door_width[:, None],
            door_z,
            route.is_goal[:, None].float(),
            next_dir * has_next,
            next_dist * has_next,
            has_next,
        ],
        dim=-1,
    )


def pillar_obs(state: DroneState, dists: ObstacleDistances, pillar_z: torch.Tensor, cfg: dict) -> torch.Tensor:
    """The k nearest pillars: body-frame offset to the closest point of the axis, the pillar's bottom and top relative to
    the drone, and the distance to its surface. Dim 6 * k.

    Pillars beyond the sensing range (and padding slots) read as zero offset and normalized distance 1.
    """
    ocfg = cfg["obs"]
    k, sense_range = ocfg["num_sensed_pillars"], ocfg["sense_range"]
    n = state.pos.shape[0]
    feats = torch.zeros(n, k, 5, device=state.pos.device)
    dist = torch.full((n, k), sense_range, device=state.pos.device)
    if dists.pillar_dist.shape[1] > 0:
        idx, d = nearest(dists.pillar_dist, k)
        kk = idx.shape[1]
        feats[:, :kk, :3] = _to_body(state.quat, _gather(dists.pillar_point, idx) - state.pos[:, None])
        feats[:, :kk, 3:] = _gather(pillar_z, idx) - state.pos[:, None, 2:3]
        dist[:, :kk] = d
    return _encode_sensed(feats, dist, sense_range)


def wall_obs(state: DroneState, dists: ObstacleDistances, segments, segment_z, cfg: dict) -> torch.Tensor:
    """The k nearest walls: body-frame vectors to the closest point and to both ends of the part within the sensing
    range (at the height of the closest point), the wall's bottom and top relative to the drone, and the distance.
    Dim 12 * k.

    The ends tell where a wall stops sideways and bottom / top where it stops vertically, i.e. where a gap, a window or
    the space above a low wall is. Ends are ordered so that the second is counter-clockwise of the first as seen from
    the drone, which removes the storage-order ambiguity.
    """
    ocfg = cfg["obs"]
    k, sense_range = ocfg["num_sensed_walls"], ocfg["sense_range"]
    n = state.pos.shape[0]
    idx, d = nearest(dists.wall_dist, k)
    kk = idx.shape[1]
    point = _gather(dists.wall_point, idx)
    end0, end1 = clip_segments_to_disc(state.pos[:, None, :2], _gather(segments, idx), sense_range)
    rel_c = point - state.pos[:, None]
    rel_0, rel_1 = _lift(end0 - state.pos[:, None, :2], rel_c[..., 2]), _lift(end1 - state.pos[:, None, :2], rel_c[..., 2])
    swap = (rel_0[..., 0] * rel_1[..., 1] - rel_0[..., 1] * rel_1[..., 0] < 0.0)[..., None]
    rel_0, rel_1 = torch.where(swap, rel_1, rel_0), torch.where(swap, rel_0, rel_1)

    feats = torch.zeros(n, k, 11, device=state.pos.device)
    feats[:, :kk, :9] = _to_body(state.quat, torch.stack([rel_c, rel_0, rel_1], dim=2)).flatten(2)
    feats[:, :kk, 9:] = _gather(segment_z, idx) - state.pos[:, None, 2:3]
    dist = torch.full((n, k), sense_range, device=state.pos.device)
    dist[:, :kk] = d
    return _encode_sensed(feats, dist, sense_range)


def privileged_obs(state: DroneState, route: RouteState, elapsed_frac: torch.Tensor, cfg: dict) -> torch.Tensor:
    """Critic-only: the next route waypoint (which may be a partition opening), remaining route length and time used.
    Dim 6."""
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


def _encode_sensed(feats: torch.Tensor, dist: torch.Tensor, sense_range: float) -> torch.Tensor:
    """feats (N, k, d) and dist (N, k) -> (N, k * (d + 1)); unsensed entries read zero features and distance 1."""
    sensed = dist < sense_range
    feats = torch.where(sensed[..., None], feats / sense_range, 0.0)
    dist = torch.where(sensed, dist, sense_range) / sense_range
    return torch.cat([feats, dist[..., None]], dim=-1).flatten(1)
