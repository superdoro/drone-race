"""Random chains of rectangular rooms with walls, openings and pillars at any height. Generated in plain Python, served
from a GPU pool.

One layout:
  * A chain of axis-aligned rooms, each with its own ceiling height. Room r + 1 is attached to a side of room r other
    than the one the route enters through, and the two are joined by a single door at a random place in the shared wall;
    every other wall is closed, so there is exactly one route from the start (next to a wall of room 0) to the goal (next
    to a wall of the last room). A door can be a window: an opening that starts above the floor.
  * Inside a room: partitions across the whole room, passable only through their opening (a gap, a window, above a low
    wall or below a hanging wall; successive gaps tend to alternate sides, which forces a zig-zag), free-standing walls
    at any angle, and cylindrical pillars. Free walls and pillars can be full height, low or hanging, and a free wall
    can also be a beam in mid-air.
  * Every layout is passable by construction: partitions are parallel, every opening is at least the minimum door width
    wide and `min_passage` high, and every pillar / free wall keeps at least `pass_gap` horizontally from every other
    obstacle and wall, so whatever its height it can never seal a passage off.

Every obstacle is a vertical extrusion of a 2D shape (a wall center line of thickness `wall_thickness`, or a disc) over
a height interval [z_lo, z_hi]. All geometry is analytic (see task/geometry.py).

Route waypoints (3D): per room, the openings of its partitions in crossing order, then its exit (the door to the next
room, or the goal). The progress reward follows them, so it never asks the drone to push through a wall or a partition;
the policy itself only observes the doors, never the partition openings.
"""

import math
import random
from dataclasses import dataclass, fields

import numpy as np
import torch

W, E, S, N = range(4)
OUTWARD = ((-1.0, 0.0), (1.0, 0.0), (0.0, -1.0), (0.0, 1.0))
OPPOSITE = (E, W, N, S)


def _along_axis(side: int) -> int:
    """Axis (0 = x, 1 = y) that the wall on `side` runs along."""
    return 1 if side in (W, E) else 0


@dataclass(frozen=True)
class Rect:
    x0: float
    y0: float
    x1: float
    y1: float

    def side(self, s: int) -> tuple[float, float, float]:
        """(fixed coordinate, lo, hi) of the wall on side `s`; lo / hi run along the wall."""
        if s in (W, E):
            return (self.x0 if s == W else self.x1), self.y0, self.y1
        return (self.y0 if s == S else self.y1), self.x0, self.x1

    def point_on_side(self, s: int, u: float, inset: float = 0.0) -> tuple[float, float]:
        fixed = self.side(s)[0]
        x, y = (fixed, u) if s in (W, E) else (u, fixed)
        return x - inset * OUTWARD[s][0], y - inset * OUTWARD[s][1]

    def wall(self, s: int, a: float, b: float, z_lo: float, z_hi: float) -> tuple[float, ...]:
        """Wall segment along side `s` from a to b (along the wall) over [z_lo, z_hi]."""
        fixed = self.side(s)[0]
        return (fixed, a, fixed, b, z_lo, z_hi) if s in (W, E) else (a, fixed, b, fixed, z_lo, z_hi)

    def extent(self, axis: int) -> tuple[float, float]:
        return (self.x0, self.x1) if axis == 0 else (self.y0, self.y1)

    def contains(self, x: float, y: float, inset: float) -> bool:
        return self.x0 + inset <= x <= self.x1 - inset and self.y0 + inset <= y <= self.y1 - inset

    def overlaps(self, o: "Rect", eps: float = 1e-6) -> bool:
        """Interiors intersect (rooms sharing a wall only touch)."""
        return self.x0 < o.x1 - eps and o.x0 < self.x1 - eps and self.y0 < o.y1 - eps and o.y0 < self.y1 - eps


@dataclass(frozen=True)
class Port:
    """Where the route enters or leaves a room: a door center on the wall line, or the start / goal point."""

    x: float
    y: float
    side: int
    half_width: float  # half door width along the wall; 0 for the start / goal

    def extent(self, axis: int) -> tuple[float, float]:
        c = self.x if axis == 0 else self.y
        h = self.half_width if _along_axis(self.side) == axis else 0.0
        return c - h, c + h


@dataclass
class Layout:
    rooms: list[Rect]
    ceilings: list[float]
    segments: list[tuple[float, ...]]  # walls: (x0, y0, x1, y1, z_lo, z_hi)
    free_standing: list[bool]  # per segment: a free-standing wall, i.e. not a room wall or partition
    pillars: list[tuple[int, float, float, float, float]]  # (radius slot, x, y, z_lo, z_hi)
    partitions: list[list[tuple[float, float, float, float]]]  # per room: point on each partition, normal toward exit
    waypoints: list[tuple[float, float, float]]
    first_wp: list[int]  # per room: index of its first waypoint
    exit_wp: list[int]  # per room: index of its exit waypoint
    exit_normal: list[tuple[float, float]]  # per room: crossing direction of its exit door, (0, 0) for the goal
    exit_width: list[float]  # per room: exit door width, 0 for the goal
    exit_z: list[tuple[float, float]]  # per room: height interval of the exit door's opening, (0, 0) for the goal
    start: tuple[float, float, float]
    goal: tuple[float, float, float]


# ---------------------------------------------------------------- scalar geometry (horizontal)


def _point_seg_dist(px: float, py: float, seg) -> float:
    x0, y0, x1, y1 = seg[:4]
    dx, dy = x1 - x0, y1 - y0
    len2 = dx * dx + dy * dy
    t = 0.0 if len2 == 0.0 else min(1.0, max(0.0, ((px - x0) * dx + (py - y0) * dy) / len2))
    return math.hypot(px - x0 - t * dx, py - y0 - t * dy)


def _cross(ax, ay, bx, by, cx, cy) -> float:
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def _seg_seg_dist(a, b) -> float:
    a, b = a[:4], b[:4]
    d1, d2 = _cross(*a[:2], *a[2:], *b[:2]), _cross(*a[:2], *a[2:], *b[2:])
    d3, d4 = _cross(*b[:2], *b[2:], *a[:2]), _cross(*b[:2], *b[2:], *a[2:])
    if d1 * d2 < 0.0 and d3 * d4 < 0.0:
        return 0.0
    return min(
        _point_seg_dist(a[0], a[1], b), _point_seg_dist(a[2], a[3], b),
        _point_seg_dist(b[0], b[1], a), _point_seg_dist(b[2], b[3], a),
    )  # fmt: skip


def _subtract(lo: float, hi: float, holes: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Pieces of [lo, hi] not covered by `holes`."""
    pieces, cur = [], lo
    for a, b in sorted(holes):
        if a > cur:
            pieces.append((cur, min(a, hi)))
        cur = max(cur, b)
    if cur < hi:
        pieces.append((cur, hi))
    return [(a, b) for a, b in pieces if b - a > 1e-6]


# ---------------------------------------------------------------- generator


class LayoutGenerator:
    def __init__(self, cfg: dict, pillar_radii: list[float], seed: int = 0):
        self.c = cfg["rooms"]
        self.top = self.c["wall_top"]
        self.radii = pillar_radii
        self.rng = random.Random(seed)

    def sample(self, level: dict) -> Layout:
        n_rooms = self.rng.randint(*level["rooms"])
        # A self-avoiding chain can walk into a dead end; retry, and only then accept a shorter chain.
        for _ in range(20):
            rects, entry_sides, doors = self._chain(n_rooms, level["door_width"])
            if len(rects) == n_rooms:
                break
        return self._furnish(rects, entry_sides, doors, level)

    def _size(self) -> float:
        lo, hi = self.c["size_range"]
        g = self.c["grid"]
        return g * self.rng.randint(math.ceil(lo / g - 1e-9), math.floor(hi / g + 1e-9))

    def _chain(self, n_rooms: int, door_width):
        rects = [Rect(0.0, 0.0, self._size(), self._size())]
        entry_sides = [self.rng.randrange(4)]  # room 0: the wall next to the start
        doors = []  # (side of room r, door center along that wall, door width) between rooms r and r + 1
        for _ in range(n_rooms - 1):
            placed = self._attach(rects, entry_sides[-1], door_width)
            if placed is None:
                break
            rect, side, u, width = placed
            rects.append(rect)
            doors.append((side, u, width))
            entry_sides.append(OPPOSITE[side])
        return rects, entry_sides, doors

    def _attach(self, rects: list[Rect], entry_side: int, door_width):
        c, rng, g = self.c, self.rng, self.c["grid"]
        prev = rects[-1]
        sides = [s for s in range(4) if s != entry_side]
        rng.shuffle(sides)
        for side in sides:
            for _ in range(4):
                width = rng.uniform(*door_width)
                min_overlap = width + 2.0 * c["door_margin"]
                w, h = self._size(), self._size()
                fixed, lo, hi = prev.side(side)
                span = h if side in (W, E) else w  # new room's extent along the shared wall
                # new room runs along the wall from a = lo + k * g; both rooms must share at least `min_overlap`
                k_lo = math.ceil((min_overlap - span) / g - 1e-9)
                k_hi = math.floor((hi - lo - min_overlap) / g + 1e-9)
                if k_lo > k_hi:
                    continue
                a = lo + g * rng.randint(k_lo, k_hi)
                rect = {
                    W: Rect(fixed - w, a, fixed, a + h),
                    E: Rect(fixed, a, fixed + w, a + h),
                    S: Rect(a, fixed - h, a + w, fixed),
                    N: Rect(a, fixed, a + w, fixed + h),
                }[side]
                if any(rect.overlaps(o) for o in rects):
                    continue
                ov_lo, ov_hi = max(lo, a), min(hi, a + span)
                u = rng.uniform(ov_lo + c["door_margin"] + 0.5 * width, ov_hi - c["door_margin"] - 0.5 * width)
                return rect, side, u, width
        return None

    def _inner_port(self, rect: Rect, side: int, inset: float) -> Port:
        _, lo, hi = rect.side(side)
        x, y = rect.point_on_side(side, self.rng.uniform(lo + inset, hi - inset), inset)
        return Port(x, y, side, 0.0)

    def _window(self, ceiling: float) -> tuple[float, float]:
        """Height interval of a window-like opening under `ceiling`."""
        c = self.c
        height = self.rng.uniform(*c["window_height"])
        sill = self.rng.uniform(c["window_margin"], ceiling - c["window_margin"] - height)
        return sill, sill + height

    def _spawn_z(self, ceiling: float) -> float:
        lo, hi = self.c["spawn_z"]
        return self.rng.uniform(lo, min(hi, ceiling - self.c["min_passage"]))

    def _furnish(self, rects, entry_sides, doors, level) -> Layout:
        c, rng, top = self.c, self.rng, self.top
        n = len(rects)
        vertical = level["vertical"]
        ceilings = [rng.uniform(*level["ceiling"]) for _ in rects]
        start = self._inner_port(rects[0], entry_sides[0], c["start_inset"])
        goal = self._inner_port(rects[-1], rng.choice([s for s in range(4) if s != entry_sides[-1]]), c["goal_inset"])
        start_z, goal_z = self._spawn_z(ceilings[0]), self._spawn_z(ceilings[-1])
        ins, outs, door_z = [start], [], []
        for r, (side, u, width) in enumerate(doors):
            x, y = rects[r].point_on_side(side, u)
            outs.append(Port(x, y, side, 0.5 * width))
            ins.append(Port(x, y, OPPOSITE[side], 0.5 * width))
            ceiling = min(ceilings[r], ceilings[r + 1])
            door_z.append(self._window(ceiling) if rng.random() < vertical else (0.0, ceiling))
        outs.append(goal)

        segments = []
        for r, rect in enumerate(rects):
            holes = {s: [] for s in range(4)}
            if r > 0:  # the part of the entry wall shared with the previous room is emitted (with its door) by that room
                prev_side = doors[r - 1][0]
                _, p_lo, p_hi = rects[r - 1].side(prev_side)
                _, lo, hi = rect.side(OPPOSITE[prev_side])
                holes[OPPOSITE[prev_side]].append((max(lo, p_lo), min(hi, p_hi)))
            if r < n - 1:
                side, u, width = doors[r]
                a, b = u - 0.5 * width, u + 0.5 * width
                holes[side].append((a, b))
                z_lo, z_hi = door_z[r]
                if z_lo > 0.0:  # window: wall below and above the opening
                    segments += [rect.wall(side, a, b, 0.0, z_lo), rect.wall(side, a, b, z_hi, top)]
            for s in range(4):
                segments += [rect.wall(s, a, b, 0.0, top) for a, b in _subtract(*rect.side(s)[1:], holes[s])]

        free_standing = [False] * len(segments)
        pillars, partitions, first_wp, exit_wp = [], [], [], []
        wp_specs = []  # (x, y, z_lo, z_hi) of each waypoint's opening; heights are resolved below
        free_slots = list(range(len(self.radii)))
        rng.shuffle(free_slots)
        for r, rect in enumerate(rects):
            count = rng.randint(*level["partitions"])
            gaps, inner, openings = self._partitions(rect, ceilings[r], ins[r], outs[r], count, level)
            route = [(ins[r].x, ins[r].y)] + [(x, y) for x, y, _, _ in openings] + [(outs[r].x, outs[r].y)]
            num_partition_segs = len(inner)
            self._free_walls(rect, ceilings[r], route, inner, rng.randint(*level["free_walls"]), vertical)
            pillars += self._pillars(rect, ceilings[r], route, inner, rng.randint(*level["pillars"]), free_slots, vertical)
            segments += inner
            free_standing += [k >= num_partition_segs for k in range(len(inner))]
            partitions.append(gaps)
            first_wp.append(len(wp_specs))
            wp_specs += openings
            exit_wp.append(len(wp_specs))
            if r < n - 1:
                wp_specs.append((outs[r].x, outs[r].y, *door_z[r]))
            else:
                wp_specs.append((goal.x, goal.y, goal_z, goal_z))

        # Waypoint heights, from the goal backward: keep the next waypoint's height if it fits through this opening
        # (with a margin), otherwise the closest height that does.
        waypoints, z = [], goal_z
        for x, y, z_lo, z_hi in reversed(wp_specs):
            margin = min(c["waypoint_z_margin"], 0.5 * (z_hi - z_lo))
            z = min(max(z, z_lo + margin), z_hi - margin)
            waypoints.append((x, y, z))
        waypoints.reverse()

        return Layout(
            rooms=rects,
            ceilings=ceilings,
            segments=segments,
            free_standing=free_standing,
            pillars=pillars,
            partitions=partitions,
            waypoints=waypoints,
            first_wp=first_wp,
            exit_wp=exit_wp,
            exit_normal=[OUTWARD[side] for side, _, _ in doors] + [(0.0, 0.0)],
            exit_width=[width for _, _, width in doors] + [0.0],
            exit_z=door_z + [(0.0, 0.0)],
            start=(start.x, start.y, start_z),
            goal=(goal.x, goal.y, goal_z),
        )

    def _partitions(self, rect: Rect, ceiling: float, p_in: Port, p_out: Port, count: int, level: dict):
        """Parallel walls across the room between its entry and exit, each passable only through one opening.

        Returns, in crossing order, a point on each partition with the unit normal toward the exit, the wall segments,
        and the openings as (x, y, z_lo, z_hi) with (x, y) on the partition.
        """
        c, rng, top = self.c, self.rng, self.top
        if count == 0:
            return [], [], []
        clear, spacing = c["partition_clearance"], c["partition_spacing"]
        options = []  # partitions are perpendicular to an axis along which entry and exit are far enough apart
        for axis in (0, 1):
            in_lo, in_hi = p_in.extent(axis)
            out_lo, out_hi = p_out.extent(axis)
            if out_lo + out_hi > in_lo + in_hi:
                lo, hi, sign = in_hi + clear, out_lo - clear, 1.0
            else:
                lo, hi, sign = out_hi + clear, in_lo - clear, -1.0
            if hi >= lo:
                options.append((axis, lo, hi, sign))
        if not options:
            return [], [], []
        axis, lo, hi, sign = rng.choice(options)
        count = min(count, int((hi - lo) / spacing) + 1)
        slack = hi - lo - (count - 1) * spacing
        coords = [lo + o + k * spacing for k, o in enumerate(sorted(rng.uniform(0.0, slack) for _ in range(count)))]
        if sign < 0:
            coords.reverse()

        b0, b1 = rect.extent(1 - axis)
        mid = 0.5 * (b0 + b1)
        a_in, b_in = (p_in.x, p_in.y) if axis == 0 else (p_in.y, p_in.x)
        a_out, b_out = (p_out.x, p_out.y) if axis == 0 else (p_out.y, p_out.x)

        def point(coord, along):
            return (coord, along) if axis == 0 else (along, coord)

        def piece(coord, a, b, z_lo, z_hi):
            return (*point(coord, a), *point(coord, b), z_lo, z_hi)

        low_half = rng.random() < 0.5
        gaps, segs, openings = [], [], []
        for k, coord in enumerate(coords):
            kind = rng.choice(("window", "over", "under")) if rng.random() < level["vertical"] else "gap"
            if kind in ("gap", "window"):
                if k > 0 and rng.random() < c["zigzag_prob"]:
                    low_half = not low_half
                half = 0.5 * rng.uniform(*level["door_width"])
                lo_g, hi_g = (b0 + half, mid) if low_half else (mid, b1 - half)
                center = rng.uniform(min(lo_g, hi_g), max(lo_g, hi_g))
                # a wall stub shorter than door_margin would just be a sliver: make the gap flush with the wall instead
                if center - half - b0 < c["door_margin"]:
                    center = b0 + half
                elif b1 - center - half < c["door_margin"]:
                    center = b1 - half
                segs += [piece(coord, a, b, 0.0, top) for a, b in _subtract(b0, b1, [(center - half, center + half)])]
                z_lo, z_hi = 0.0, ceiling
                if kind == "window":
                    z_lo, z_hi = self._window(ceiling)
                    segs += [piece(coord, center - half, center + half, 0.0, z_lo),
                             piece(coord, center - half, center + half, z_hi, top)]  # fmt: skip
            else:
                # The opening spans the whole room width; put its waypoint where the straight entry-exit line crosses.
                f = (coord - a_in) / (a_out - a_in) if abs(a_out - a_in) > 1e-6 else 0.5
                margin = c["door_margin"]
                center = min(max(b_in + f * (b_out - b_in), b0 + margin), b1 - margin)
                if kind == "over":  # low wall: fly over it
                    height = min(rng.uniform(*c["low_height"]), ceiling - c["min_passage"])
                    segs.append(piece(coord, b0, b1, 0.0, height))
                    z_lo, z_hi = height, ceiling
                else:  # hanging wall: fly under it
                    bottom = rng.uniform(*c["hanging_bottom"])
                    segs.append(piece(coord, b0, b1, bottom, top))
                    z_lo, z_hi = 0.0, bottom
            x, y = point(coord, center)
            gaps.append((x, y, sign, 0.0) if axis == 0 else (x, y, 0.0, sign))
            openings.append((x, y, z_lo, z_hi))
        return gaps, segs, openings

    def _sample_point(self, rect: Rect, route: list[tuple[float, float]]) -> tuple[float, float]:
        """Mostly near the straight route through the room (so obstacles are in the way), otherwise anywhere in it."""
        rng = self.rng
        legs = [(a, b, math.dist(a, b)) for a, b in zip(route[:-1], route[1:])]
        total = sum(length for _, _, length in legs)
        if total > 0.0 and rng.random() < self.c["on_path_fraction"]:
            s = rng.uniform(0.0, total)
            for (ax, ay), (bx, by), length in legs:
                if s <= length:
                    break
                s -= length
            f = s / max(length, 1e-9)
            lateral = rng.gauss(0.0, self.c["on_path_sigma"]) / max(length, 1e-9)
            return ax + f * (bx - ax) - lateral * (by - ay), ay + f * (by - ay) + lateral * (bx - ax)
        return rng.uniform(rect.x0, rect.x1), rng.uniform(rect.y0, rect.y1)

    def _vertical_extent(self, ceiling: float, vertical: float, beam: bool) -> tuple[float, float]:
        """Height interval of a free wall / pillar: full height, or with probability `vertical` low, hanging or a beam."""
        c, rng = self.c, self.rng
        if rng.random() >= vertical:
            return 0.0, self.top
        kind = rng.choice(("low", "hanging", "beam") if beam else ("low", "hanging"))
        if kind == "low":
            return 0.0, min(rng.uniform(*c["low_height"]), ceiling - c["min_passage"])
        if kind == "hanging":
            return rng.uniform(*c["hanging_bottom"]), self.top
        thickness = rng.uniform(*c["beam_thickness"])
        z_lo = min(rng.uniform(*c["beam_z"]), ceiling - c["min_passage"] - thickness)
        return z_lo, z_lo + thickness

    def _free_walls(self, rect: Rect, ceiling: float, route, segs: list, count: int, vertical: float) -> None:
        """Append up to `count` free-standing wall segments to `segs` (which holds the room's partitions)."""
        c, rng = self.c, self.rng
        t, gap = c["wall_thickness"], c["pass_gap"]
        for _ in range(count):
            for _ in range(c["place_tries"]):
                cx, cy = self._sample_point(rect, route)
                half, angle = 0.5 * rng.uniform(*c["free_wall_length"]), rng.uniform(0.0, math.pi)
                dx, dy = half * math.cos(angle), half * math.sin(angle)
                seg = (cx - dx, cy - dy, cx + dx, cy + dy)
                if not (rect.contains(seg[0], seg[1], t + gap) and rect.contains(seg[2], seg[3], t + gap)):
                    continue
                if any(_seg_seg_dist(seg, o) < t + gap for o in segs):
                    continue
                if any(_point_seg_dist(px, py, seg) < c["port_clearance"] for px, py in route):
                    continue
                segs.append((*seg, *self._vertical_extent(ceiling, vertical, beam=True)))
                break

    def _pillars(self, rect: Rect, ceiling: float, route, segs, count: int, free_slots: list[int], vertical: float):
        c, rng = self.c, self.rng
        t, gap = c["wall_thickness"], c["pass_gap"]
        placed = []
        for _ in range(count):
            if not free_slots:
                break
            slot = free_slots.pop()
            r = self.radii[slot]
            for _ in range(c["place_tries"]):
                x, y = self._sample_point(rect, route)
                if not rect.contains(x, y, r + 0.5 * t + gap):
                    continue
                if any(_point_seg_dist(x, y, s) < r + 0.5 * t + gap for s in segs):
                    continue
                if any(math.hypot(x - px, y - py) < r + self.radii[ps] + gap for ps, px, py, _, _ in placed):
                    continue
                if any(math.hypot(x - px, y - py) < c["port_clearance"] + r for px, py in route):
                    continue
                placed.append((slot, x, y, *self._vertical_extent(ceiling, vertical, beam=False)))
                break
            else:
                free_slots.insert(0, slot)
        return placed


# ---------------------------------------------------------------- batched tensors


def layout_capacity(cfg: dict) -> dict[str, int]:
    """Padded tensor sizes that fit every layout any curriculum level can produce."""
    levels = cfg["curriculum"]["levels"]
    rooms = max(level["rooms"][1] for level in levels)
    parts = max(level["partitions"][1] for level in levels)
    return {
        "rooms": rooms,
        "partitions": parts,
        # per room: <= 6 pieces of its own walls (two doors split two sides) + 2 above / below a window exit,
        # <= 4 per partition (a window), the free walls
        "segments": rooms * (8 + 4 * parts + max(level["free_walls"][1] for level in levels)),
        "pillars": rooms * max(level["pillars"][1] for level in levels),
        "waypoints": rooms * (parts + 1),
    }


@dataclass
class LayoutBatch:
    segments: torch.Tensor  # (B, S, 4) x0, y0, x1, y1
    segment_z: torch.Tensor  # (B, S, 2) z_lo, z_hi
    segment_active: torch.Tensor  # (B, S) bool
    segment_free: torch.Tensor  # (B, S) bool, free-standing wall
    pillar_xy: torch.Tensor  # (B, P, 2), one row per radius slot
    pillar_z: torch.Tensor  # (B, P, 2) z_lo, z_hi
    pillar_active: torch.Tensor  # (B, P) bool
    rooms: torch.Tensor  # (B, R, 4) x0, y0, x1, y1
    ceilings: torch.Tensor  # (B, R)
    num_rooms: torch.Tensor  # (B,) long
    partition_pt: torch.Tensor  # (B, R, Q, 2) a point on each partition (its opening)
    partition_normal: torch.Tensor  # (B, R, Q, 2) toward the room exit
    partition_active: torch.Tensor  # (B, R, Q) bool
    waypoints: torch.Tensor  # (B, WP, 3)
    remaining: torch.Tensor  # (B, WP) route length from each waypoint to the goal
    first_wp: torch.Tensor  # (B, R) long
    exit_wp: torch.Tensor  # (B, R) long
    exit_normal: torch.Tensor  # (B, R, 2)
    exit_width: torch.Tensor  # (B, R)
    exit_z: torch.Tensor  # (B, R, 2) height interval of the exit door's opening
    start: torch.Tensor  # (B, 3)
    goal: torch.Tensor  # (B, 3)

    def __getitem__(self, idx) -> "LayoutBatch":
        return LayoutBatch(**{f.name: getattr(self, f.name)[idx] for f in fields(self)})

    def assign(self, env_ids: torch.Tensor, other: "LayoutBatch") -> None:
        for f in fields(self):
            getattr(self, f.name)[env_ids] = getattr(other, f.name)

    @staticmethod
    def pack(layouts: list[Layout], cap: dict[str, int], device) -> "LayoutBatch":
        b, r_cap, q_cap, wp_cap = len(layouts), cap["rooms"], cap["partitions"], cap["waypoints"]
        a = {
            "segments": np.zeros((b, cap["segments"], 4), np.float32),
            "segment_z": np.zeros((b, cap["segments"], 2), np.float32),
            "segment_active": np.zeros((b, cap["segments"]), bool),
            "segment_free": np.zeros((b, cap["segments"]), bool),
            "pillar_xy": np.zeros((b, cap["pillars"], 2), np.float32),
            "pillar_z": np.zeros((b, cap["pillars"], 2), np.float32),
            "pillar_active": np.zeros((b, cap["pillars"]), bool),
            "rooms": np.zeros((b, r_cap, 4), np.float32),
            "ceilings": np.zeros((b, r_cap), np.float32),
            "num_rooms": np.zeros(b, np.int64),
            "partition_pt": np.zeros((b, r_cap, q_cap, 2), np.float32),
            "partition_normal": np.zeros((b, r_cap, q_cap, 2), np.float32),
            "partition_active": np.zeros((b, r_cap, q_cap), bool),
            "waypoints": np.zeros((b, wp_cap, 3), np.float32),
            "remaining": np.zeros((b, wp_cap), np.float32),
            "first_wp": np.zeros((b, r_cap), np.int64),
            "exit_wp": np.zeros((b, r_cap), np.int64),
            "exit_normal": np.zeros((b, r_cap, 2), np.float32),
            "exit_width": np.zeros((b, r_cap), np.float32),
            "exit_z": np.zeros((b, r_cap, 2), np.float32),
            "start": np.zeros((b, 3), np.float32),
            "goal": np.zeros((b, 3), np.float32),
        }
        for i, lay in enumerate(layouts):
            n, ns, nw = len(lay.rooms), len(lay.segments), len(lay.waypoints)
            seg = np.asarray(lay.segments, np.float32)
            a["segments"][i, :ns], a["segment_z"][i, :ns] = seg[:, :4], seg[:, 4:]
            a["segment_active"][i, :ns] = True
            a["segment_free"][i, :ns] = lay.free_standing
            for slot, x, y, z_lo, z_hi in lay.pillars:
                a["pillar_xy"][i, slot], a["pillar_z"][i, slot] = (x, y), (z_lo, z_hi)
                a["pillar_active"][i, slot] = True
            a["rooms"][i, :n] = [(r.x0, r.y0, r.x1, r.y1) for r in lay.rooms]
            a["ceilings"][i, :n] = lay.ceilings
            a["num_rooms"][i] = n
            for r, gaps in enumerate(lay.partitions):
                for q, (gx, gy, nx, ny) in enumerate(gaps):
                    a["partition_pt"][i, r, q] = (gx, gy)
                    a["partition_normal"][i, r, q] = (nx, ny)
                    a["partition_active"][i, r, q] = True
            wp = np.asarray(lay.waypoints, np.float32)
            a["waypoints"][i, :nw] = wp
            a["remaining"][i, : nw - 1] = np.cumsum(np.linalg.norm(np.diff(wp, axis=0), axis=1)[::-1])[::-1]
            a["first_wp"][i, :n] = lay.first_wp
            a["exit_wp"][i, :n] = lay.exit_wp
            a["exit_normal"][i, :n] = lay.exit_normal
            a["exit_width"][i, :n] = lay.exit_width
            a["exit_z"][i, :n] = lay.exit_z
            a["start"][i], a["goal"][i] = lay.start, lay.goal
        return LayoutBatch(**{k: torch.from_numpy(v).to(device) for k, v in a.items()})

    @staticmethod
    def empty(batch: int, cap: dict[str, int], device) -> "LayoutBatch":
        template = LayoutBatch.pack([], cap, device)
        return LayoutBatch(**{
            f.name: getattr(template, f.name).new_zeros((batch, *getattr(template, f.name).shape[1:]))
            for f in fields(LayoutBatch)
        })  # fmt: skip


class LayoutPool:
    """Pre-generated layouts on the GPU. Resets draw from the pool, so generation never runs per reset.

    The pool is regenerated when the curriculum level changes and after `pool_reuse` draws per layout on average,
    so the policy keeps seeing fresh layouts. A `replay` share of each pool comes from a random easier level, so
    what only easy layouts teach (e.g. braking for a goal in an empty room) is not forgotten at a level without them.
    """

    def __init__(self, cfg: dict, pillar_radii: list[float], device, seed: int = 0):
        self.cap = layout_capacity(cfg)
        self.size = cfg["rooms"]["pool_size"]
        self.reuse = cfg["rooms"]["pool_reuse"]
        self.replay = cfg["curriculum"]["replay"]
        self.generator = LayoutGenerator(cfg, pillar_radii, seed)
        self.device = device
        self.level = None
        self.data = None
        self.draws = 0

    def fill(self, level: dict, easier: list[dict]) -> None:
        rng = self.generator.rng

        def pick():
            return rng.choice(easier) if easier and rng.random() < self.replay else level

        self.data = LayoutBatch.pack([self.generator.sample(pick()) for _ in range(self.size)], self.cap, self.device)
        self.level = level
        self.draws = 0

    def draw(self, n: int, level: dict, easier: list[dict]) -> LayoutBatch:
        if self.level is not level or self.draws >= self.size * self.reuse:
            self.fill(level, easier)
        self.draws += n
        return self.data[torch.randint(0, self.size, (n,), device=self.device)]
