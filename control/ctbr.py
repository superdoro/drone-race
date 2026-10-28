"""Collective thrust + body rates (CTBR) controller.

CTBR is the command interface accepted by Crazyflie, PX4 and Betaflight, so a policy trained against it can later be
deployed on hardware by forwarding the same command to the flight controller. The controller below plays the role of
the flight controller's inner rate loop and mixer, so it must run at every physics step.
"""

import torch

from config.drone_params import DroneParams


class CTBRController:
    def __init__(
        self,
        params: DroneParams,
        max_body_rates: tuple[float, float, float],
        rate_gain: tuple[float, float, float],
        device: torch.device | str,
    ):
        self.params = params
        self.max_body_rates = torch.tensor(max_body_rates, dtype=torch.float32, device=device)
        self.rate_gain = torch.tensor(rate_gain, dtype=torch.float32, device=device)
        self.inertia = torch.tensor(params.inertia, dtype=torch.float32, device=device)

        x = torch.tensor([xy[0] for xy in params.rotor_xy], dtype=torch.float64)
        y = torch.tensor([xy[1] for xy in params.rotor_xy], dtype=torch.float64)
        spin = torch.tensor(params.rotor_spin, dtype=torch.float64)
        # [collective, tau_x, tau_y, tau_z] = mixer @ rotor_thrusts, for vertical rotor forces applied at (x, y, 0).
        mixer = torch.stack([torch.ones_like(x), y, -x, (params.km / params.kf) * spin])
        self.inv_mixer_t = torch.linalg.inv(mixer).T.to(device=device, dtype=torch.float32)

    def action_to_command(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map normalized actions in [-1, 1] to (collective thrust in N, desired body rates in rad/s).

        Zero collective action is hover thrust and +1 is the maximum thrust, so an untrained policy starts near hover.
        """
        p = self.params
        collective = p.hover_thrust * (1.0 + actions[:, 0] * (p.thrust_to_weight - 1.0))
        collective = collective.clamp(0.0, p.thrust_to_weight * p.hover_thrust)
        return collective, actions[:, 1:4] * self.max_body_rates

    def rotor_thrusts(self, collective: torch.Tensor, rates_des: torch.Tensor, ang_vel_body: torch.Tensor):
        """Body-rate P loop with gyroscopic feed-forward, then mixing to per-rotor thrusts (N)."""
        jw = self.inertia * ang_vel_body
        torque = self.inertia * self.rate_gain * (rates_des - ang_vel_body) + torch.cross(ang_vel_body, jw, dim=-1)
        wrench = torch.cat([collective[:, None], torque], dim=-1)
        return (wrench @ self.inv_mixer_t).clamp(0.0, self.params.max_rotor_thrust)
