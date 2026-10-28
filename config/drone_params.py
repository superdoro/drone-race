"""Physical parameters of the quadrotor, shared by every simulator backend and the CTBR controller.

Keeping a single source of truth here is what keeps a backend swap (Genesis -> Isaac Lab) dynamically equivalent:
each backend must reproduce these numbers instead of reading its own from an asset file.
"""

from dataclasses import dataclass

GRAVITY = 9.81


@dataclass(frozen=True)
class DroneParams:
    name: str
    urdf: str  # path understood by the backend (Genesis resolves it inside its asset folder)
    mass: float  # kg
    inertia: tuple[float, float, float]  # diagonal inertia (kg m^2)
    kf: float  # rotor thrust = kf * rpm^2 (N)
    km: float  # rotor yaw torque = km * rpm^2 (N m)
    rotor_xy: tuple[tuple[float, float], ...]  # rotor positions in the body frame (m), x forward, y left
    rotor_spin: tuple[int, ...]  # sign of each rotor's yaw torque about body +z
    thrust_to_weight: float
    body_radius: float  # radius used by the analytic collision check (m)

    @property
    def num_rotors(self) -> int:
        return len(self.rotor_xy)

    @property
    def hover_thrust(self) -> float:
        return self.mass * GRAVITY

    @property
    def max_rotor_thrust(self) -> float:
        return self.thrust_to_weight * self.hover_thrust / self.num_rotors


# Crazyflie 2.x in X configuration, numbers from Genesis' urdf/drones/cf2x.urdf.
CF2X = DroneParams(
    name="cf2x",
    urdf="urdf/drones/cf2x.urdf",
    mass=0.027,
    inertia=(1.4e-5, 1.4e-5, 2.17e-5),
    kf=3.16e-10,
    km=7.94e-12,
    rotor_xy=((0.028, -0.028), (-0.028, -0.028), (-0.028, 0.028), (0.028, 0.028)),
    rotor_spin=(-1, 1, -1, 1),
    thrust_to_weight=2.25,
    body_radius=0.06,
)

DRONES = {p.name: p for p in (CF2X,)}
