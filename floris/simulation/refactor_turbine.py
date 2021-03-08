"""Draft for revamping of turbine map"""


import math
from copy import deepcopy
from typing import Any, Dict, List, Tuple, Union
from functools import partial
from itertools import product

import attr
import numpy as np
from scipy.stats import norm
from scipy.spatial import distance_matrix
from scipy.interpolate import interp1d

from floris.utilities import Vec3, FromDictMixin, cosd, wrap_180
from floris.logging_manager import LoggerBase


# from .refactor_flow_field import FlowField


def _circle_area(r):
    return np.pi * r ** 2


def check_between_0_1(instance, attribute, value):
    if not 0 <= value <= 1:
        raise ValueError(f"{attribute.name} must be between 0 and 1, inclusive.")


def interp_generator(x, y):
    return interp1d(x, y, fill_value="extrapolate")


def float_eq_int(name, value):
    if value % 1 != 0:
        raise ValueError(
            f"{name} must be an integer of float representation of an integer."
        )
    return int(value)


@attr.s(frozen=True, auto_attribs=True)
class PowerThrustTable:
    power: List[float]
    thrust: List[float]
    wind_speed: List[float]


@attr.s(auto_attribs=True)
class Turbine(FromDictMixin):
    description: str
    rotor_diameter: float
    hub_height: float
    blade_count: int
    pP: float
    pT: float
    generator_efficiency: float
    power_thrust_table: Dict[str, List[float]]
    yaw_angle: float
    tilt_angle: float
    TSR: float
    ngrid: int = attr.ib(default=5, converter=partial(float_eq_int, "ngrid"))
    rloc: float = attr.ib(default=0.5, validator=check_between_0_1)
    air_density: float = attr.ib(default=-1)
    use_turbulence_correction: bool = attr.ib(default=False)
    flow_field_point_indices: Any = attr.ib(default=None)
    use_points_on_perimeter: bool = attr.ib(default=False)
    fCpInterp: interp1d = attr.ib(init=False)
    fCtInterp: interp1d = attr.ib(init=False)
    grid_point_count: int = attr.ib(init=False)
    velocities: np.ndarray = attr.ib(init=False)
    coordinates: Vec3 = attr.ib(init=False)

    def __attrs_post_init__(self):
        """Creates the `power_thrust_table` to be `PowerThrustTable` object."""
        self.power_thrust_table = PowerThrustTable(**self.power_thrust_table)
        self._reinitialize()

    def _reinitialize(self):
        wind_speed = self.power_thrust_table.wind_speed
        self.fCpInterp = interp_generator(wind_speed, self.power_thrust_table.power)
        self.fCtInterp = interp_generator(wind_speed, self.power_thrust_table.thrust)
        self.grid_point_count = self.ngrid ** 2
        self.velocities = np.zeros(self.grid_point_count)
        self.rotor_radius = self.rotor_diameter / 2.0

        self.reset_velocities()

        self.grid = self._create_swept_area_grid()
        inner_power = np.array([self._power_inner_function(ws) for ws in wind_speed])
        self.pow_interp = interp_generator(wind_speed, inner_power)

    def reset_velocities(self):
        self.velocities = np.zeros(self.grid_point_count)

    def change_turbine_parameters(self, update_dict):
        for param, val in update_dict.items():
            self.logger.info(f"Setting {param} to {val}.")
            object.__setattr__(self, param, val)

    def _create_swept_area_grid(self):
        point = self.rloc * self.rotor_diameter / 2
        grid = np.linspace(-point, point, self.ngrid)
        grid = np.array(list(product(grid, grid)))
        if self.use_points_on_perimeter:
            swept_area = np.hypot(grid[:, 0], grid[:, 1]) <= self.rotor_radius
        else:
            swept_area = np.hypot(grid[:, 0], grid[:, 1]) < self.rotor_radius
        return grid[swept_area]

    def _fCp(self, at_wind_speed: float) -> float:
        wind_speed = self.power_thrust_table.wind_speed
        if at_wind_speed < min(wind_speed):
            return 0.0
        _cp = self.fCpInterp(at_wind_speed)
        _cp = _cp[0] if _cp.size > 1 else float(_cp)
        if _cp > 1:
            _cp = 1.0
        if _cp < 0:
            _cp = 0.0
        return _cp

    def _power_inner_function(self, yaw_effective_velocity):
        _cp = self._fCp(yaw_effective_velocity)
        val = (
            0.5
            * _circle_area(self.rotor_radius)
            * _cp
            * self.generator_efficiency
            * yaw_effective_velocity ** 3
        )
        return val

    def _fCt(self, at_wind_speed):
        wind_speed = self.power_thrust_table.wind_speed
        if at_wind_speed < min(wind_speed):
            return 0.99
        _ct = self.fCtInterp(at_wind_speed)
        _ct = _ct[0] if _ct.size > 1 else float(_ct)
        if _ct > 1.0:
            _ct = 0.9999
        if _ct <= 0.0:
            _ct = 0.0001
        return _ct

    def create_grid(self, coord: Vec3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.full(coord.x1, self.grid.shape[0]),
            self.grid[:, 0] + coord.x2,
            self.grid[:, 1] + self.hub_height,
        )

    def calculate_swept_area_velocities(
        self,
        local_wind_speed: float,
        coord: Vec3,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        additional_wind_speed: float = None,
    ) -> np.ndarray:
        if self.flow_field_point_indices is None:
            flow_grid_points = np.column_stack((x.flatten(), y.flatten(), z.flatten()))
            grid_array = np.column_stack(self.create_grid(coord))
            ii = np.argmin(distance_matrix(flow_grid_points, grid_array, axis=0))
        else:
            ii = self.flow_field_point_indices

        if additional_wind_speed is None:
            return np.array(local_wind_speed.flatten()[ii])

        return local_wind_speed.flatten()[ii], additional_wind_speed.flatten()[ii]

    def update_velocities(
        self,
        u_wake: np.ndarray,
        coord: Vec3,
        flow_field: "FlowField",
        rotated_x: np.ndarray,
        rotated_y: np.ndarray,
        rotated_z: np.ndarray,
    ) -> None:
        self.velocities = self.calculate_swept_area_velocities(
            flow_field.u_initial - u_wake, coord, rotated_x, rotated_y, rotated_z
        )

    def update_turbulence_intensity(self, ti: float) -> None:
        # Used for flow_field.py:560 in calculate to become list comprehension
        self.current_turbulence_intensity = ti
        self.reset_velocities()

    def update_air_density(self, air_density: float) -> None:
        self.air_density = air_density

    def TKE_to_TI(self, turbulence_kinetic_energy: np.ndarray) -> np.ndarray:
        return np.sqrt((2 / 3) * turbulence_kinetic_energy) / self.average_velocity

    def TI_to_TKE(self) -> np.ndarray:
        return (self.avererage_velocity * self.current_turbulence_intensity) ** 2 / (
            2 / 3
        )

    def u_prime(self) -> np.ndarray:
        return np.sqrt(2 * self.TI_to_TKE())

    def turbulence_parameter(self) -> float:
        if not self.use_turbulence_correction:
            return 1.0

        ws = self.power_thrust_table.wind_speed
        cp = self.power_thrust_table.power
        ws = ws[np.where(cp != 0)]
        cut_in_ws = ws[0]
        cut_out_ws = ws[len(ws) - 1]
        mu = self.average_velocity
        ws_check = cut_in_ws >= mu or cut_out_ws <= mu
        if ws_check or self.current_turbulence_intensity == 0 or math.isnan(mu):
            return 1.0
        sigma = self.current_turbulence_intensity * mu
        xp = np.linspace(mu - sigma, min(mu + sigma, cut_out_ws), 100)
        pdf = norm.pdf(xp, mu, sigma)
        npdf = pdf * (1 / np.sum(pdf))
        return npdf * self.pow_interp(xp) / self.pow_interp(mu)

    @property
    def yaw_effective_velocity(self):
        return self.average_velocity * cosd(self.yaw_angle) ** self.pP / 3.0

    @property
    def average_velocity(self):
        velocities = self.velocities[np.where(~np.isnan(self.velocities))]
        avg_vel = np.cbrt(np.mean(velocities ** 3))
        if np.isnan(avg_vel) or np.isninf(avg_vel):
            return 0
        return avg_vel

    @property
    def Cp(self):
        return self._fCp(self.yaw_effective_velocity)

    @property
    def Ct(self):
        return self._fCt(self.average_velocity) * cosd(self.yaw_angle)

    @property
    def power(self):
        return (
            self.air_density
            * self.pow_interp(self.yaw_effective_velocity)
            * self.turbulence_parameter
        )

    @property
    def aI(self):
        cos_yaw = cosd(self.yaw_angle)
        return 0.5 / cos_yaw * (1 - np.sqrt(1 - self.Ct * cos_yaw))


class TurbineMap(LoggerBase):
    def __init__(
        self, layout_x: List[float], layout_y: List[float], turbines: List[dict],
    ):
        if len(layout_x) != len(layout_y):
            err_msg = (
                f"The number of turbine x locations ({len(layout_x)}) and the number ",
                f"of turbine y locations ({len(layout_y)}) is not equal. Please check",
                "your layout.",
            )
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)

        self._turbine_map = {i: deepcopy(turb) for i, turb in enumerate(turbines)}
        self._set_locations(layout_x, layout_y)

    @property
    def coords(self) -> List[Vec3]:
        return [turb.coordinates for turb in self._turbine_map.values()]

    @property
    def coordinate_array(self) -> np.ndarray:
        return np.array([t.coordinates.elements for t in self._turbine_map.values()])

    @property
    def coordinate_array_prime(self) -> np.ndarray:
        return np.array(
            [t.coordinates.prime_elements for t in self._turbine_map.values()]
        )

    @property
    def turbines(self) -> List[Turbine]:
        return list(self._turbine_map.values())

    @property
    def rotor_diameters(self) -> np.ndarray:
        return np.array([t.rotor_diameter for t in self._turbine_map.values()])

    def _set_locations(self, layout_x: List[float], layout_y: List[float]) -> None:
        for turbine, x, y in zip(self._turbine_map.values(), layout_x, layout_y):
            turbine.coordinates = Vec3(x, y, turbine.hub_height)

    def rotate_coords(
        self, angles: List[float], center_of_rotation: Vec3
    ) -> "TurbineMap":
        [
            coord.rotate_on_x3(angle, center_of_rotation)
            for angle, coord in zip(angles, self.coords)
        ]
        layout = self.coordinate_array_prime
        return TurbineMap(layout[:, 0], layout[:, 1], list(self._turbine_map.values()))

    def sort_turbines(self, by="x") -> List[Tuple[Vec3, Turbine]]:
        by = by.strip().lower()
        if by == "x":
            order = self.coordinate_array[:, 0].argsort()
        elif by == "y":
            order = self.coordinate_array[:, 1].argsort()
        coords = np.array(self.coords)[order].tolist()
        turbs = np.array(self.turbines)[order].tolist()
        return list(zip(coords, turbs))

    def number_of_wakes_iec(self, wd: float, return_turbines=True):
        x1, x2, _ = self.coordinate_array.T
        diffs = np.array(
            [
                [np.delete(x1, i) - x1[i], np.delete(x2, i) - x2[i]]
                for i in range(len(x1))
            ]
        )
        dists = np.hypot(diffs[:, 0], diffs[:, 1]) / self.rotor_diameters
        angles = np.degrees(np.arctan2(diffs[:, 0], diffs[:, 1]))
        waked = dists <= 2.0
        waked = waked | (
            (dists <= 20.0)
            & (
                np.abs(wrap_180(wd - angles))
                <= 0.5 * (1.3 * np.degrees(np.arctan(2.5 / dists + 0.15)) + 10)
            )
        )
        return waked.sum(axis=0)

    def update_turbulence_intensities(self, turbulence_intensities: np.ndarray) -> None:
        # to be used for flow_field.py:560 to move some of the turbine_map work out of flow field
        [
            t.update_turbulence_intensity(ti)
            for t, ti in zip(self._turbine_map.values(), turbulence_intensities)
        ]

    def update_air_density(self, air_density: float) -> None:
        [
            turbine.update_air_density(air_density)
            for turbine in self._turbine_map.values()
        ]


# if __name__ == "__main__":
# lst = [0, 0.5, 1]
# turbine = dict(
#     description="turbine",
#     rotor_diameter=140,
#     hub_height=110,
#     blade_count=3,
#     pP=1.2,
#     pT=1.4,
#     generator_efficiency=0.9,
#     power_thrust_table=dict(power=lst, thrust=lst, wind_speed=lst),
#     yaw_angle=180,
#     tilt_angle=110,
#     TSR=1.5,
#     ngrid=12,
#     rloc=0.2,
# )

# turb = Turbine(**turbine)
# print(turb)
# print(turb.reset_velocities())
