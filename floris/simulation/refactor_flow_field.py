from typing import List, Tuple, Union, Optional

import attr
import numpy as np
import scipy as sp
from scipy.interpolate import griddata

from floris.utilities import Vec3, cosd, sind, tand

from .wake import Wake
from .wind_map import WindMap
from .refactor_turbine import Turbine, TurbineMap


class FlowField:
    wind_shear: Optional[float] = attr.ib(default=None)
    wind_veer: Optional[float] = attr.ib(default=None)
    air_density: Optional[float] = attr.ib(default=None)
    wake: Optional[Wake] = attr.ib(default=None)
    turbine_map: Optional[TurbineMap] = attr.ib(default=None)
    wind_map: Optional[WindMap] = attr.ib(default=None)
    with_resolution: Optional[Vec3] = attr.ib(default=None)
    bounds_to_set: Optional[List[float]] = attr.ib(default=None)
    specified_wind_height: Optional[float] = attr.ib(default=None)

    def reinitialize_flow_field(
        self,
        wind_shear: Optional[float],
        wind_veer: Optional[float],
        air_density: Optional[float],
        wake: Optional[Wake],
        turbine_map: Optional[TurbineMap],
        wind_map: Optional[WindMap],
        with_resolution: Optional[Vec3],
        bounds_to_set: Optional[List[float]],
        specified_wind_height: Optional[float],
    ) -> None:
        if turbine_map is not None:
            self.turbine_map = turbine_map
        if wind_map is not None:
            self.wind_map = wind_map
        if wind_shear is not None:
            self.wind_shear = wind_shear
        if wind_veer is not None:
            self.wind_veer = wind_veer
        if specified_wind_height is not None:
            self.specified_wind_height = specified_wind_height
        if air_density is not None:
            self.air_density = air_density
            self.turbine_map.update_air_density(air_density)
        if wake is not None:
            self.wake = wake
        if with_resolution is None:
            with_resolution = self.wake.velocity_model.model_grid_resolution

        self.max_diameter = self.turbine_map.rotor_diameters.max()
        self.set_bounds(bounds_to_set=bounds_to_set)
        self._compute_initialized_domain(with_resolution=with_resolution)
        self.turbine_map.update_turbulence_intensities(
            self.wind_map.turbine_turbulence_intensity
        )

    def set_bounds():
        return
