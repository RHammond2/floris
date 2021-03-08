from typing import List, Tuple, Union, Optional

import attr
import numpy as np
import scipy as sp
from scipy.interpolate import griddata

from floris.utilities import Vec3, FromDictMixin, cosd, sind, tand

from .wake import Wake
from .wind_map import WindMap
from .refactor_turbine import Turbine, TurbineMap


def _create_gridded_domain(
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    zmin: float,
    zmax: float,
    resolution: Vec3,
) -> np.ndarray:
    """
    Generate a structured grid for the entire flow field domain.
    resolution: Vec3

    PF: NOTE, PERHAPS A BETTER NAME IS SETUP_GRIDDED_DOMAIN
    """
    x = np.linspace(xmin, xmax, int(resolution.x1))
    y = np.linspace(ymin, ymax, int(resolution.x2))
    z = np.linspace(zmin, zmax, int(resolution.x3))
    return np.meshgrid(x, y, z, indexing="ij")


def _calculate_overlap_points(coord: Vec3, rx: np.ndarray, ry: np.ndarray) -> float:
    """Finds the index of the `coord` in the prime elements of the turbine map.

    Parameters
    ----------
    coord : Vec3
        The focal coordinate point.
    rx : np.ndarray
        The x1prime elements of the turbine coordinates.
    ry : np.ndarray
        The x2prime elements of the turbine coordinates.

    Returns
    -------
    float
        The index of the focal coordinate.
    """
    xloc, yloc = np.array(rx == coord.x1), np.array(ry == coord.x2)
    idx = int(np.where(np.logical_and(yloc, xloc))[0])
    return idx


class FlowField:
    def __init__(
        self,
        wind_shear: float,
        wind_veer: float,
        air_density: float,
        wake: Wake,
        turbine_map: TurbineMap,
        wind_map: WindMap,
        specified_wind_height: float,
        with_resolution: Optional[Vec3] = None,
        bounds_to_set: Optional[List[float]] = None,
    ) -> None:
        self.reinitialize_flow_field(
            wind_shear=wind_shear,
            wind_veer=wind_veer,
            air_density=air_density,
            wake=wake,
            turbine_map=turbine_map,
            wind_map=wind_map,
            with_resolution=wake.velocity_model.model_grid_resolution,
            specified_wind_height=specified_wind_height,
        )

    def _update_grid(
        self,
        x_grid_i: np.ndarray,
        y_grid_i: np.ndarray,
        wind_direction_i: np.ndarray,
        x1: np.ndarray,
        x2: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        xoffset = x_grid_i - x1
        yoffset = y_grid_i.T - x2
        wind_cos = cosd(-wind_direction_i)
        wind_sin = sind(-wind_direction_i)

        x_grid_i = xoffset * wind_cos - yoffset * wind_sin + x1
        y_grid_i = yoffset * wind_cos + xoffset * wind_sin + x2
        return x_grid_i, y_grid_i

    def _discretize_turbine_domain(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        xt = self.turbine_map.coordinate_array[:, 0]
        ngrid = self.turbine_map.turbines[0].ngrid
        grid_point_count = self.turbine_map.turbines[0].grid_point_count
        point_indices = grid_point_count * np.arange(grid_point_count)
        x_grid = np.zeros((len(xt), ngrid, ngrid))
        y_grid = np.zeros((len(xt), ngrid, ngrid))
        z_grid = np.zeros((len(xt), ngrid, ngrid))

        def inner(i, turbine: Turbine) -> None:
            x1, x2, x3 = turbine.coordinates.elements
            turbine.flow_field_point_indices = i * point_indices
            pt = turbine.rloc * turbine.rotor_radius
            yt = np.linspace(x2 - pt, x2 + pt, ngrid)
            zt = np.linspace(x3 - pt, x3 + pt, ngrid)
            x_grid[i] = xt[i]
            y_grid[i] = yt
            z_grid[i] = zt
            x_grid[i], y_grid[i] = self._update_grid(
                x_grid[i], y_grid[i], self.wind_map.turbine_wind_direction[i], x1, x2
            )

        [inner(i, t) for i, t in enumerate(self.turbine_map.turbines)]
        return x_grid, y_grid, z_grid

    def _compute_initialized_domain(
        self,
        with_resolution: Optional[Vec3] = None,
        points: Optional[np.ndarray] = None,
    ) -> None:
        if with_resolution is not None:
            self.x, self.y, self.z = self._create_gridded_domain(
                *self.domain_bounds, with_resolution
            )
        elif points is None:
            self.x, self.y, self.z = self._discretize_turbine_domain()
        else:
            elem_num_el = np.size(self.x[0])
            num_points_to_add = len(points[0])
            matrices_to_add = int(np.ceil(num_points_to_add / elem_num_el))
            buffer_amount = matrices_to_add * elem_num_el - num_points_to_add
            shape = tuple(np.array(self.x.shape) + (matrices_to_add, 0, 0))

            self.x = np.append(
                self.x, np.append(points[0, :], np.repeat(points[0, 0], buffer_amount)),
            )
            self.y = np.append(
                self.y, np.append(points[1, :], np.repeat(points[1, 0], buffer_amount)),
            )
            self.z = np.append(
                self.z, np.append(points[2, :], np.repeat(points[2, 0], buffer_amount)),
            )
            self.x = np.reshape(self.x, shape)
            self.y = np.reshape(self.y, shape)
            self.z = np.reshape(self.z, shape)

        # set grid point locations in wind_map
        self.wind_map.grid_layout = (self.x, self.y)

        # interpolate for initial values of flow field grid
        self.wind_map.calculate_turbulence_intensity(grid=True)
        self.wind_map.calculate_wind_direction(grid=True)
        self.wind_map.calculate_wind_speed(grid=True)

        self.u_initial = (
            self.wind_map.grid_wind_speed
            * (self.z / self.specified_wind_height) ** self.wind_shear
        )
        self.v_initial = np.zeros(np.shape(self.u_initial))
        self.w_initial = np.zeros(np.shape(self.u_initial))

        self.u = self.u_initial.copy()
        self.v = self.v_initial.copy()
        self.w = self.w_initial.copy()

    def _compute_turbine_velocity_deficit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        turbine: Turbine,
        coord: Vec3,
        deflection: np.ndarray,
        flow_field: "FlowField",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # velocity deficit calculation
        u_deficit, v_deficit, w_deficit = self.wake.velocity_function(
            x, y, z, turbine, coord, deflection, flow_field
        )

        # calculate spanwise and streamwise velocities if needed
        if hasattr(self.wake.velocity_model, "calculate_VW"):
            v_deficit, w_deficit = self.wake.velocity_model.calculate_VW(
                v_deficit, w_deficit, coord, turbine, flow_field, x, y, z
            )

        return u_deficit, v_deficit, w_deficit

    def _compute_turbine_wake_turbulence(
        self, ambient_TI: float, coord_ti: Vec3, turbine_coord: Vec3, turbine: Turbine
    ) -> float:
        return self.wake.turbulence_function(
            ambient_TI, coord_ti, turbine_coord, turbine
        )

    def _compute_turbine_wake_deflection(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        turbine: Turbine,
        coord: Vec3,
        flow_field: "FlowField",
    ) -> np.ndarray:
        return self.wake.deflection_function(x, y, z, turbine, coord, flow_field)

    def _rotated_grid(
        self, angle: float, center_of_rotation: Vec3
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x_center, y_center, _ = center_of_rotation.elements
        xoffset = self.x - x_center
        yoffset = self.y - y_center

        rotated_x = xoffset * cosd(angle) - yoffset * sind(angle) + x_center
        rotated_y = xoffset * sind(angle) + yoffset * cosd(angle) + y_center
        return rotated_x, rotated_y, self.z

    def _rotated_dir(
        self, angle: float, center_of_rotation: Vec3, rotated_map: TurbineMap
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.wake.velocity_model.model_string == "curl":
            coords = self.turbine_map.coordinate_array
            x = coords[:, 0]
            y = coords[:, 1]
            md = self.max_diameter
            xmin = np.min(x - 2 * md)
            xmax = np.max(x + 10 * md)
            ymin = np.min(y - 2 * md)
            ymax = np.max(y + 2 * md)
            zmin = 0.1
            zmax = 6 * self.specified_wind_height

            self._xmin, self._xmax = xmin, xmax
            self._ymin, self._ymax = ymin, ymax
            self._zmin, self._zmax = zmin, zmax

            self.x, self.y, self.z = self._discretize_gridded_domain(
                xmin,
                xmax,
                ymin,
                ymax,
                zmin,
                zmax,
                self.wake.velocity_model.model_grid_resolution,
            )
            rotated_x, rotated_y, rotated_z = self._rotated_grid(
                0.0, center_of_rotation
            )
            return rotated_x, rotated_y, rotated_z

        return self._rotated_grid(self.wind_map.grid_wind_direction, center_of_rotation)

    def _calculate_area_overlap(
        self,
        wake_velocities: np.ndarray,
        freestream_velocities: np.ndarray,
        grid_point_count: int,
    ) -> float:
        count = np.sum(freestream_velocities - wake_velocities <= 0.05)
        return (grid_point_count - count) / grid_point_count

    def set_bounds(self, bounds_to_set: Optional[List[float]] = None) -> None:
        if self.wake.velocity_model.model_string == "curl":
            coords = self.turbine_map.coordinate_array
            eps = 0.1
            self._xmin = coords[:, 0].min() - 2 * self.max_diameter
            self._xmax = coords[:, 0].max() + 10 * self.max_diameter
            self._ymin = coords[:, 1].min() - 2 * self.max_diameter
            self._ymax = coords[:, 1].max() + 2 * self.max_diameter
            self._zmin = 0 + eps
            self._zmax = 6 * self.specified_wind_height

        elif bounds_to_set is not None:
            # Set the boundaries with the provided inputs
            self._xmin = bounds_to_set[0]
            self._xmax = bounds_to_set[1]
            self._ymin = bounds_to_set[2]
            self._ymax = bounds_to_set[3]
            self._zmin = bounds_to_set[4]
            self._zmax = bounds_to_set[5]

        else:
            eps = 0.1
            D = self.max_diameter
            coords = self.turbine_map.coordinate_array
            # TODO: REVISIT POST WIND MAP VECTORIZATION and remove np.array(...)
            wd = (
                sp.stats.circmean(
                    np.array(self.wind_map.turbine_wind_direction) * np.pi / 180
                )
                * 180
                / np.pi
            )

            self._xmin = coords[:, 0].min() - (10 if 90 < wd < 270 else 2) * D
            self._xmax = coords[:, 0].max() + (10 if wd <= 90 or wd >= 270 else 2) * D
            self._ymin = coords[:, 1].min() - (5 if 5 <= wd <= 175 else 2) * D
            self._ymax = coords[:, 1].max() + (5 if 185 <= wd <= 355 else 2) * D
            self._zmin = 0 + eps
            self._zmax = 2 * self.specified_wind_height

    def reinitialize_flow_field(
        self,
        wind_shear: Optional[float] = None,
        wind_veer: Optional[float] = None,
        air_density: Optional[float] = None,
        wake: Optional[Wake] = None,
        turbine_map: Optional[TurbineMap] = None,
        wind_map: Optional[WindMap] = None,
        with_resolution: Optional[Vec3] = None,
        bounds_to_set: Optional[List[float]] = None,
        specified_wind_height: Optional[float] = None,
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
        self.set_bounds(bounds_to_set=bounds_to_set)  # Sets the domain bounds
        self._compute_initialized_domain(with_resolution=with_resolution)
        self.turbine_map.update_turbulence_intensities(
            self.wind_map.turbine_turbulence_intensity
        )

    def _rotate_grid(
        self,
        coord: Vec3,
        center_of_rotation: Vec3,
        rx: float,
        ry: float,
        initial_rotated_x: np.ndarray,
        initial_rotated_y: np.ndarray,
        rotate_once: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """[summary]

        Parameters
        ----------
        coord : Vec3
            The focal coordinate
        rx : float
            [description]
        ry : float
            [description]
        rotated_x : np.ndarray
            initial_rotated_x
        rotated_y : np.ndarray
            initial_rotated_y

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            [description]
        """
        idx = _calculate_overlap_points(coord, rx, ry)

        if rotate_once:
            return initial_rotated_x, initial_rotated_y

        x_center, y_center, _ = center_of_rotation.elements
        wd = (
            self.wind_map.turbine_wind_direction[idx]
            - self.wind_map.grid_wind_direction
        )

        # for straight wakes, change rx[idx] to initial_rotated_x
        xoffset = x_center - rx[idx]
        # for straight wakes, change ry[idx] to initial_rotated_y
        yoffset = y_center - ry[idx]

        y_grid_offset = xoffset * sind(wd) + yoffset * cosd(wd) - yoffset
        rotated_y = initial_rotated_y - y_grid_offset

        xoffset = x_center - initial_rotated_x
        yoffset = y_center - initial_rotated_y
        x_grid_offset = xoffset * cosd(wd) - yoffset * sind(wd) - xoffset
        rotated_x = initial_rotated_x - x_grid_offset
        return rotated_x, rotated_y

    def _calculate_wake_added_turbulence(
        self,
        coord_ti: Vec3,
        turbine_ti: Turbine,
        coord: Vec3,
        turbine: Turbine,
        rx: np.ndarray,
        ry: np.ndarray,
        rotated_x: np.ndarray,
        rotated_y: np.ndarray,
        rotated_z: np.ndarray,
        turb_u_wake: np.ndarray,
    ) -> None:
        idx = _calculate_overlap_points(coord_ti, rx, ry)

        # placeholder for TI/stability influence on how far
        # wakes (and wake added TI) propagate downstream
        downstream_influence_length = 15 * turbine.rotor_diameter

        overlap_check = coord_ti.x1 > coord.x1
        overlap_check &= abs(coord.x2 > coord_ti.x2)
        overlap_check &= coord_ti.x1 <= downstream_influence_length + coord.x1
        if not overlap_check:
            return

        (
            freestream_velocities,  # wake with all other wakes
            wake_velocities,
        ) = turbine_ti.calculate_swept_area_velocities(
            self.u_initial,
            coord_ti,
            rotated_x,
            rotated_y,
            rotated_z,
            additional_wind_speed=self.u_initial - turb_u_wake,
        )
        area_overlap = self._calculate_area_overlap(
            wake_velocities, freestream_velocities, turbine
        )
        if area_overlap <= 0:
            return

        ti_calculation = self._compute_turbine_wake_turbulence(
            self.wind_map.turbine_turbulence_intensity[idx], coord_ti, coord, turbine,
        )
        # multiply by area overlap
        ti_added = area_overlap * ti_calculation

        # TODO: need to revisit when we are returning fields of TI
        turbine_ti.current_turbulence_intensity = np.max(
            (
                np.sqrt(
                    ti_added ** 2 + self.wind_map.turbine_turbulence_intensity[idx] ** 2
                ),
                turbine_ti.current_turbulence_intensity,
            )
        )

        if self.track_n_upstream_wakes:
            # increment by one for each upstream wake
            self.wake_list[turbine_ti] += 1

    def calculate_wake(
        self,
        no_wake: Optional[bool] = False,
        points: Optional[np.ndarray] = None,
        track_n_upstream_wakes: Optional[bool] = False,
    ) -> None:
        if points is not None:
            # add points to flow field grid points
            self._compute_initialized_domain(points=points)

        self.track_n_upstream_wakes = track_n_upstream_wakes
        if self.track_n_upstream_wakes:
            self.wake_list = {
                turbine: 0 for turbine in self.turbine_map._turbine_map.values()
            }

        # reinitialize the turbines
        self.turbine_map.update_turbulence_intensities(
            self.wind_map.turbine_turbulence_intensity
        )

        # define the center of rotation with reference to 270 deg as center of flow field
        center_of_rotation = Vec3(
            (np.min(self.x) + np.max(self.x)) / 2,
            (np.min(self.y) + np.max(self.y)) / 2,
            0,
        )

        # Rotate the turbines such that they are now in the frame of reference
        # of the wind direction simplifying computing the wakes and wake overlap
        rotated_map = self.turbine_map.rotate_coords(
            self.wind_map.turbine_wind_direction, center_of_rotation
        )

        # rotate the discrete grid and turbine map
        initial_rotated_x, initial_rotated_y, rotated_z = self._rotated_dir(
            self.wind_map.grid_wind_direction, center_of_rotation, rotated_map
        )

        # sort the turbine map by x-coordinates
        sorted_map = self.turbine_map.sort_turbines(by="x")

        # calculate the wake velocity deficit adn deflection on the mesh
        u_wake = np.zeros(self.u.shape)

        # Empty the stored variables of v and w at start, these will be updated
        # and stored within the loop
        self.v = np.zeros(np.shape(self.u))
        self.w = np.zeros(np.shape(self.u))

        rx, ry, _ = self.turbine_map.coordinate_array_prime.T
        rotate_once = np.unique(self.wind_map.grid_wind_direction).size == 1

        for coord, turbine in sorted_map:
            rotated_x, rotated_y = self._rotate_grid(
                coord,
                center_of_rotation,
                rx,
                ry,
                initial_rotated_x,
                initial_rotated_y,
                rotate_once,
            )

            # update the turbine based on the velocity at its hub
            turbine.update_velocities(
                u_wake, coord, self, rotated_x, rotated_y, rotated_z
            )

            # get the wake deflection field
            deflection = self._compute_turbine_wake_deflection(
                rotated_x, rotated_y, rotated_z, turbine, coord, self
            )

            # get the velocity deficit accounting for the deflection
            (
                turb_u_wake,
                turb_v_wake,
                turb_w_wake,
            ) = self._compute_turbine_velocity_deficit(
                rotated_x, rotated_y, rotated_z, turbine, coord, deflection, self
            )

            if self.wake.turbulence_model.model_string in (
                "crespo_hernandez",
                "ishihara_qian",
            ):
                _args = (
                    coord,
                    turbine,
                    rx,
                    ry,
                    rotated_x,
                    rotated_y,
                    rotated_z,
                    turb_u_wake,
                )
                [
                    self._calculate_wake_added_turbulence(coord_ti, turbine_ti, *_args)
                    for coord_ti, turbine_ti in sorted_map
                ]
        # apply the velocity deficit field to the freestream
        if not no_wake:
            self.u = self.u_initial - u_wake
            # self.v = self.v_initial + v_wake
            # self.w = self.w_initial + w_wake

        # rotate the grid if it is curl
        if self.wake.velocity_model.model_string == "curl":
            self.x, self.y, self.z = self._rotated_grid(
                -1 * self.wind_map.grid_wind_direction, center_of_rotation
            )

    @property
    def specified_wind_height(self):
        return self._specified_wind_height

    @specified_wind_height.setter
    def specified_wind_height(self, value):
        if value == -1:
            self._specified_wind_height = self.turbine_map.turbines[0].hub_height
        else:
            self._specified_wind_height = value

    @property
    def domain_bounds(self):
        """
        The minimum and maximum values of the bounds of the flow field domain.

        Returns:
            float, float, float, float, float, float:
                minimum-x, maximum-x, minimum-y, maximum-y, minimum-z, maximum-z
        """
        return self._xmin, self._xmax, self._ymin, self._ymax, self._zmin, self._zmax
