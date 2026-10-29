"""Where each drone is along its layout's single route: current room, next waypoint, remaining route length."""

from dataclasses import dataclass

import torch

from task.geometry import locate
from task.rooms import LayoutBatch


@dataclass
class RouteState:
    room: torch.Tensor  # (N,) index of the room the drone is in
    in_rooms: torch.Tensor  # (N,) False once the drone has left the chain of rooms (only possible by tunneling a wall)
    waypoint: torch.Tensor  # (N, 3) next route waypoint: first uncrossed partition gap of the room, else its exit
    remaining: torch.Tensor  # (N,) route length from the drone through the remaining waypoints to the goal
    door: torch.Tensor  # (N, 3) exit of the current room: the door to the next room, or the goal
    door_normal: torch.Tensor  # (N, 2) crossing direction of that door, zero for the goal
    door_width: torch.Tensor  # (N,) zero for the goal
    is_goal: torch.Tensor  # (N,) bool, the current room is the last one
    next_door: torch.Tensor  # (N, 3) exit of the following room (zero if there is none)
    has_next: torch.Tensor  # (N,) bool


def route_state(pos: torch.Tensor, layout: LayoutBatch, goal_z: torch.Tensor) -> RouteState:
    """Waypoints are 2D; they are lifted to the goal altitude so that the route also leads there.

    Partitions inside a room are parallel and can only be crossed through their gaps, so the number of partitions the
    drone is past (on the exit side of) is the number of route waypoints of that room it has already passed.
    """
    n = pos.shape[0]
    ar = torch.arange(n, device=pos.device)
    room, in_rooms = locate(pos, layout.rooms, layout.num_rooms)

    pt, normal = layout.partition_pt[ar, room], layout.partition_normal[ar, room]
    crossed = (((pos[:, None, :2] - pt) * normal).sum(dim=-1) > 0.0) & layout.partition_active[ar, room]
    wp_idx = layout.first_wp[ar, room] + crossed.sum(dim=1)

    def lift(xy):
        return torch.cat([xy, goal_z[:, None]], dim=-1)

    waypoint = lift(layout.waypoints[ar, wp_idx])
    exit_idx = layout.exit_wp[ar, room]
    has_next = room + 1 < layout.num_rooms
    next_room = (room + 1).clamp_max(layout.rooms.shape[1] - 1)
    next_door = lift(layout.waypoints[ar, layout.exit_wp[ar, next_room]]) * has_next[:, None]
    return RouteState(
        room=room,
        in_rooms=in_rooms,
        waypoint=waypoint,
        remaining=(waypoint - pos).norm(dim=-1) + layout.remaining[ar, wp_idx],
        door=lift(layout.waypoints[ar, exit_idx]),
        door_normal=layout.exit_normal[ar, room],
        door_width=layout.exit_width[ar, room],
        is_goal=~has_next,
        next_door=next_door,
        has_next=has_next,
    )
