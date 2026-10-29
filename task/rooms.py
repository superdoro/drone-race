"""Random chains of rectangular rooms with pillars and interior walls. Generated in plain Python, served from a GPU pool.

One layout:
  * A chain of axis-aligned rooms. Room r + 1 is attached to a side of room r other than the one the route enters
    through, and the two are joined by a single door at a random place in the shared wall; every other wall is closed,
    so there is exactly one route from the start (next to a wall of room 0) to the goal (next to a wall of the last room).
  * Inside a room: partitions (walls across the whole room with one gap; successive gaps tend to alternate sides, which
    forces a zig-zag), free-standing wall segments at any angle, and cylindrical pillars.
  * Every layout is passable by construction: partitions are parallel and each gap is at least the minimum door width,
    and every pillar / free wall keeps at least `pass_gap` from every other obstacle and wall, so none of them can seal a
    passage off.

Walls are line segments of thickness `wall_thickness`; all geometry is analytic (see task/geometry.py).

Route waypoints: per room, the gap centers of its partitions in crossing order, then its exit (the door to the next room,
or the goal). The progress reward follows them, so it never asks the drone to push through a wall; the policy itself
only observes the doors, never the gaps.
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
    segments: list[tuple[float, float, float, float]]  # walls: (x0, y0, x1, y1)
    free_standing: list[bool]  # per segment: a free-standing wall, i.e. not a room wall or partition
    pillars: list[tuple[int, float, float]]  # (radius slot, x, y)
    partitions: list[list[tuple[float, float, float, float]]]  # per room: gap center and unit normal toward the exit
    waypoints: list[tuple[float, float]]
    first_wp: list[int]  # per room: index of its first waypoint
    exit_wp: list[int]  # per room: index of its exit waypoint
    exit_normal: list[tuple[float, float]]  # per room: crossing direction of its exit door, (0, 0) for the goal
    exit_width: list[float]  # per room: exit door width, 0 for the goal
    start: tuple[float, float]
    goal: tuple[float, float]


# ---------------------------------------------------------------- scalar geometry


def _point_seg_dist(px: float, py: float, seg) -> float:
    x0, y0, x1, y1 = seg
    dx, dy = x1 - x0, y1 - y0
    len2 = dx * dx + dy * dy
    t = 0.0 if len2 == 0.0 else min(1.0, max(0.0, ((px - x0) * dx + (py - y0) * dy) / len2))
    return math.hypot(px - x0 - t * dx, py - y0 - t * dy)


def _cross(ax, ay, bx, by, cx, cy) -> float:
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def _seg_seg_dist(a, b) -> float:
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

    def _furnish(self, rects, entry_sides, doors, level) -> Layout:
        c, rng = self.c, self.rng
        n = len(rects)
        start = self._inner_port(rects[0], entry_sides[0], c["start_inset"])
        goal = self._inner_port(rects[-1], rng.choice([s for s in range(4) if s != entry_sides[-1]]), c["goal_inset"])
        ins, outs = [start], []
        for r, (side, u, width) in enumerate(doors):
            x, y = rects[r].point_on_side(side, u)
            outs.append(Port(x, y, side, 0.5 * width))
            ins.append(Port(x, y, OPPOSITE[side], 0.5 * width))
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
                holes[side].append((u - 0.5 * width, u + 0.5 * width))
            for s in range(4):
                fixed, lo, hi = rect.side(s)
                for a, b in _subtract(lo, hi, holes[s]):
                    segments.append((fixed, a, fixed, b) if s in (W, E) else (a, fixed, b, fixed))

        free_standing = [False] * len(segments)
        pillars, partitions, waypoints, first_wp, exit_wp = [], [], [], [], []
        free_slots = list(range(len(self.radii)))
        rng.shuffle(free_slots)
        for r, rect in enumerate(rects):
            gaps, inner = self._partitions(rect, ins[r], outs[r], rng.randint(*level["partitions"]), level["door_width"])
            route = [(ins[r].x, ins[r].y)] + [(gx, gy) for gx, gy, _, _ in gaps] + [(outs[r].x, outs[r].y)]
            num_partition_segs = len(inner)
            self._free_walls(rect, route, inner, rng.randint(*level["free_walls"]))
            pillars += self._pillars(rect, route, inner, rng.randint(*level["pillars"]), free_slots)
            segments += inner
            free_standing += [k >= num_partition_segs for k in range(len(inner))]
            partitions.append(gaps)
            first_wp.append(len(waypoints))
            waypoints += route[1:-1]
            exit_wp.append(len(waypoints))
            waypoints.append(route[-1])

        return Layout(
            rooms=rects,
            segments=segments,
            free_standing=free_standing,
            pillars=pillars,
            partitions=partitions,
            waypoints=waypoints,
            first_wp=first_wp,
            exit_wp=exit_wp,
            exit_normal=[OUTWARD[side] for side, _, _ in doors] + [(0.0, 0.0)],
            exit_width=[width for _, _, width in doors] + [0.0],
            start=(start.x, start.y),
            goal=(goal.x, goal.y),
        )

    def _partitions(self, rect: Rect, p_in: Port, p_out: Port, count: int, door_width):
        """Parallel walls across the room between its entry and exit, each with one gap.

        Returns the gaps in crossing order as (center x, center y, normal x, normal y), normal pointing toward the
        exit, and the wall segments.
        """
        c, rng = self.c, self.rng
        if count == 0:
            return [], []
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
            return [], []
        axis, lo, hi, sign = rng.choice(options)
        count = min(count, int((hi - lo) / spacing) + 1)
        slack = hi - lo - (count - 1) * spacing
        coords = [lo + o + k * spacing for k, o in enumerate(sorted(rng.uniform(0.0, slack) for _ in range(count)))]
        if sign < 0:
            coords.reverse()

        b0, b1 = rect.extent(1 - axis)
        mid = 0.5 * (b0 + b1)
        low_half = rng.random() < 0.5
        gaps, segs = [], []
        for k, coord in enumerate(coords):
            if k > 0 and rng.random() < c["zigzag_prob"]:
                low_half = not low_half
            half = 0.5 * rng.uniform(*door_width)
            lo_g, hi_g = (b0 + half, mid) if low_half else (mid, b1 - half)
            center = rng.uniform(min(lo_g, hi_g), max(lo_g, hi_g))
            # a wall stub shorter than door_margin would just be a sliver: make the gap flush with the room wall instead
            if center - half - b0 < c["door_margin"]:
                center = b0 + half
            elif b1 - center - half < c["door_margin"]:
                center = b1 - half
            for a, b in _subtract(b0, b1, [(center - half, center + half)]):
                segs.append((coord, a, coord, b) if axis == 0 else (a, coord, b, coord))
            gx, gy = (coord, center) if axis == 0 else (center, coord)
            gaps.append((gx, gy, sign, 0.0) if axis == 0 else (gx, gy, 0.0, sign))
        return gaps, segs

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

    def _free_walls(self, rect: Rect, route, segs: list, count: int) -> None:
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
                segs.append(seg)
                break

    def _pillars(self, rect: Rect, route, segs, count: int, free_slots: list[int]):
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
                if any(math.hypot(x - px, y - py) < r + self.radii[ps] + gap for ps, px, py in placed):
                    continue
                if any(math.hypot(x - px, y - py) < c["port_clearance"] + r for px, py in route):
                    continue
                placed.append((slot, x, y))
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
        # per room: <= 6 pieces of its own walls (two doors split two sides), 2 per partition, the free walls
        "segments": rooms * (6 + 2 * parts + max(level["free_walls"][1] for level in levels)),
        "pillars": rooms * max(level["pillars"][1] for level in levels),
        "waypoints": rooms * (parts + 1),
    }


@dataclass
class LayoutBatch:
    segments: torch.Tensor  # (B, S, 4) x0, y0, x1, y1
    segment_active: torch.Tensor  # (B, S) bool
    segment_free: torch.Tensor  # (B, S) bool, free-standing wall
    pillar_xy: torch.Tensor  # (B, P, 2), one row per radius slot
    pillar_active: torch.Tensor  # (B, P) bool
    rooms: torch.Tensor  # (B, R, 4) x0, y0, x1, y1
    num_rooms: torch.Tensor  # (B,) long
    partition_pt: torch.Tensor  # (B, R, Q, 2) gap centers
    partition_normal: torch.Tensor  # (B, R, Q, 2) toward the room exit
    partition_active: torch.Tensor  # (B, R, Q) bool
    waypoints: torch.Tensor  # (B, WP, 2)
    remaining: torch.Tensor  # (B, WP) route length from each waypoint to the goal
    first_wp: torch.Tensor  # (B, R) long
    exit_wp: torch.Tensor  # (B, R) long
    exit_normal: torch.Tensor  # (B, R, 2)
    exit_width: torch.Tensor  # (B, R)
    start: torch.Tensor  # (B, 2)
    goal: torch.Tensor  # (B, 2)

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
            "segment_active": np.zeros((b, cap["segments"]), bool),
            "segment_free": np.zeros((b, cap["segments"]), bool),
            "pillar_xy": np.zeros((b, cap["pillars"], 2), np.float32),
            "pillar_active": np.zeros((b, cap["pillars"]), bool),
            "rooms": np.zeros((b, r_cap, 4), np.float32),
            "num_rooms": np.zeros(b, np.int64),
            "partition_pt": np.zeros((b, r_cap, q_cap, 2), np.float32),
            "partition_normal": np.zeros((b, r_cap, q_cap, 2), np.float32),
            "partition_active": np.zeros((b, r_cap, q_cap), bool),
            "waypoints": np.zeros((b, wp_cap, 2), np.float32),
            "remaining": np.zeros((b, wp_cap), np.float32),
            "first_wp": np.zeros((b, r_cap), np.int64),
            "exit_wp": np.zeros((b, r_cap), np.int64),
            "exit_normal": np.zeros((b, r_cap, 2), np.float32),
            "exit_width": np.zeros((b, r_cap), np.float32),
            "start": np.zeros((b, 2), np.float32),
            "goal": np.zeros((b, 2), np.float32),
        }
        for i, lay in enumerate(layouts):
            n, ns, nw = len(lay.rooms), len(lay.segments), len(lay.waypoints)
            a["segments"][i, :ns] = lay.segments
            a["segment_active"][i, :ns] = True
            a["segment_free"][i, :ns] = lay.free_standing
            for slot, x, y in lay.pillars:
                a["pillar_xy"][i, slot] = (x, y)
                a["pillar_active"][i, slot] = True
            a["rooms"][i, :n] = [(r.x0, r.y0, r.x1, r.y1) for r in lay.rooms]
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
    so the policy keeps seeing fresh layouts.
    """

    def __init__(self, cfg: dict, pillar_radii: list[float], device, seed: int = 0):
        self.cap = layout_capacity(cfg)
        self.size = cfg["rooms"]["pool_size"]
        self.reuse = cfg["rooms"]["pool_reuse"]
        self.generator = LayoutGenerator(cfg, pillar_radii, seed)
        self.device = device
        self.level = None
        self.data = None
        self.draws = 0

    def fill(self, level: dict) -> None:
        self.data = LayoutBatch.pack([self.generator.sample(level) for _ in range(self.size)], self.cap, self.device)
        self.level = level
        self.draws = 0

    def draw(self, n: int, level: dict) -> LayoutBatch:
        if self.level is not level or self.draws >= self.size * self.reuse:
            self.fill(level)
        self.draws += n
        return self.data[torch.randint(0, self.size, (n,), device=self.device)]
