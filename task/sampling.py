"""Random initial attitudes and the success-driven curriculum. Pure torch."""

import math

import torch

from utils.math_utils import quat_from_euler


def uniform(n: int, low: float, high: float, device) -> torch.Tensor:
    return low + (high - low) * torch.rand(n, device=device)


def sample_attitude(n: int, cfg: dict, device) -> torch.Tensor:
    """Small random tilt and any heading."""
    max_tilt = math.radians(cfg["init"]["max_tilt_deg"])
    return quat_from_euler(
        uniform(n, -max_tilt, max_tilt, device),
        uniform(n, -max_tilt, max_tilt, device),
        uniform(n, -math.pi, math.pi, device),
    )


class Curriculum:
    """Global difficulty level, promoted when the success rate over a window of finished episodes is high enough."""

    def __init__(self, cfg: dict):
        self.levels = cfg["levels"]
        self.promote_rate = cfg["promote_success_rate"]
        self.window = cfg["window"]
        self.level_idx = cfg.get("initial_level", 0) % len(self.levels)
        self._successes = 0
        self._episodes = 0

    @property
    def level(self) -> dict:
        return self.levels[self.level_idx]

    def update(self, success: torch.Tensor) -> bool:
        """Feed the success flags of the episodes that just ended; return True when the level changed."""
        self._successes += int(success.sum())
        self._episodes += success.numel()
        if self._episodes < self.window:
            return False
        promote = self._successes / self._episodes >= self.promote_rate and self.level_idx < len(self.levels) - 1
        self._successes = self._episodes = 0
        if promote:
            self.level_idx += 1
        return promote
