# Copyright 2021 NREL
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# See https://floris.readthedocs.io for documentation


from __future__ import annotations

import copy
from typing import Any
from pathlib import Path
from itertools import repeat, product
from multiprocessing import cpu_count
from multiprocessing.pool import Pool

import attr
import numpy as np
import pandas as pd
import numpy.typing as npt
import matplotlib.pyplot as plt
from scipy.stats import norm
from numpy.lib.arraysetops import unique

from floris.utilities import Vec3, attrs_array_converter
from floris.simulation import Farm, Floris, FlowField, WakeModelManager
from floris.logging_manager import LoggerBase
from floris.tools.cut_plane import get_plane_from_flow_data
from floris.tools.flow_data import FlowData
from floris.simulation.turbine import Ct, power, axial_induction, average_velocity
from floris.tools.interface_utilities import get_params, set_params, show_params


# from .cut_plane import CutPlane, change_resolution, get_plane_from_flow_data
# from .visualization import visualize_cut_plane
# from .layout_functions import visualize_layout, build_turbine_loc


NDArrayFloat = npt.NDArray[np.float64]


DEFAULT_UNCERTAINTY = {"std_wd": 4.95, "std_yaw": 1.75, "pmf_res": 1.0, "pdf_cutoff": 0.995}


def global_calc_one_AEP_case(FlorisInterface, wd, ws, freq, yaw=None):
    return FlorisInterface._calc_one_AEP_case(wd, ws, freq, yaw)


def _generate_uncertainty_parameters(unc_options: dict, unc_pmfs: dict) -> dict:
    """Generates the uncertainty parameters for `FlorisInterface.get_farm_power` and
    `FlorisInterface.get_turbine_power` for more details.

    Args:
        unc_options (dict): See `FlorisInterface.get_farm_power` or `FlorisInterface.get_turbine_power`.
        unc_pmfs (dict): See `FlorisInterface.get_farm_power` or `FlorisInterface.get_turbine_power`.

    Returns:
        dict: [description]
    """
    if (unc_options is None) & (unc_pmfs is None):
        unc_options = DEFAULT_UNCERTAINTY

    if unc_pmfs is not None:
        return unc_pmfs

    wd_unc = np.zeros(1)
    wd_unc_pmf = np.ones(1)
    yaw_unc = np.zeros(1)
    yaw_unc_pmf = np.ones(1)

    # create normally distributed wd and yaw uncertaitny pmfs if appropriate
    if unc_options["std_wd"] > 0:
        wd_bnd = int(np.ceil(norm.ppf(unc_options["pdf_cutoff"], scale=unc_options["std_wd"]) / unc_options["pmf_res"]))
        bound = wd_bnd * unc_options["pmf_res"]
        wd_unc = np.linspace(-1 * bound, bound, 2 * wd_bnd + 1)
        wd_unc_pmf = norm.pdf(wd_unc, scale=unc_options["std_wd"])
        wd_unc_pmf /= np.sum(wd_unc_pmf)  # normalize so sum = 1.0

    if unc_options["std_yaw"] > 0:
        yaw_bnd = int(
            np.ceil(norm.ppf(unc_options["pdf_cutoff"], scale=unc_options["std_yaw"]) / unc_options["pmf_res"])
        )
        bound = yaw_bnd * unc_options["pmf_res"]
        yaw_unc = np.linspace(-1 * bound, bound, 2 * yaw_bnd + 1)
        yaw_unc_pmf = norm.pdf(yaw_unc, scale=unc_options["std_yaw"])
        yaw_unc_pmf /= np.sum(yaw_unc_pmf)  # normalize so sum = 1.0

    unc_pmfs = {
        "wd_unc": wd_unc,
        "wd_unc_pmf": wd_unc_pmf,
        "yaw_unc": yaw_unc,
        "yaw_unc_pmf": yaw_unc_pmf,
    }
    return unc_pmfs


def correct_for_all_combinations(
    wd: NDArrayFloat,
    ws: NDArrayFloat,
    freq: NDArrayFloat,
    yaw: NDArrayFloat | None = None,
) -> tuple[NDArrayFloat]:
    """Computes the probabilities for the complete windrose from the desired wind
    direction and wind speed combinations and their associated probabilities so that
    any undesired combinations are filled with a 0.0 probability.

    Args:
        wd (NDArrayFloat): List or array of wind direction values.
        ws (NDArrayFloat): List or array of wind speed values.
        freq (NDArrayFloat): Frequencies corresponding to wind
            speeds and directions in wind rose with dimensions
            (N wind directions x N wind speeds).
        yaw (NDArrayFloat | None): The corresponding yaw angles for each of the wind
            direction and wind speed combinations, or None. Defaults to None.

    Returns:
        NDArrayFloat, NDArrayFloat, NDArrayFloat: The unique wind directions, wind
            speeds, and the associated probability of their combination combinations in
            an array of shape (N wind directions x N wind speeds).
    """

    combos_to_compute = np.array(list(zip(wd, ws, freq)))

    unique_wd = wd.unique()
    unique_ws = ws.unique()
    all_combos = np.array(list(product(unique_wd, unique_ws)), dtype=float)
    all_combos = np.hstack((all_combos, np.zeros((all_combos.shape[0], 1), dtype=float)))
    expanded_yaw = np.array([None] * all_combos.shape[0]).reshape(unique_wd.size, unique_ws.size)

    ix_match = [np.where((all_combos[:, :2] == combo[:2]).all(1))[0][0] for combo in combos_to_compute]
    all_combos[ix_match, 2] = combos_to_compute[:, 2]
    if yaw is not None:
        expanded_yaw[ix_match] = yaw
    freq = all_combos.T[2].reshape((unique_wd.size, unique_ws.size))
    return unique_wd, unique_ws, freq


@attr.s(auto_attribs=True)
class FlorisInterface(LoggerBase):
    """
    FlorisInterface provides a high-level user interface to many of the
    underlying methods within the FLORIS framework. It is meant to act as a
    single entry-point for the majority of users, simplifying the calls to
    methods on objects within FLORIS.

    Args:
        configuration (:py:obj:`dict`): The Floris configuration dictarionary, JSON file,
            or YAML file. The configuration should have the following inputs specified.
                - **flow_field**: See `floris.simulation.flow_field.FlowField` for more details.
                - **farm**: See `floris.simulation.farm.Farm` for more details.
                - **turbine**: See `floris.simulation.turbine.Turbine` for more details.
                - **wake**: See `floris.simulation.wake.WakeManager` for more details.
                - **logging**: See `floris.simulation.floris.Floris` for more details.
    """

    configuration: dict | str | Path
    floris: Floris = attr.ib(init=False)
    unique_copy_id: int = attr.ib(init=False)

    def __attrs_post_init__(self) -> None:
        self.create_floris()
        self.unique_copy_id = 1

    def create_floris(self) -> None:
        if isinstance(self.configuration, (str, Path)):
            self.configuration = Path(self.configuration).resolve()
            if self.configuration.suffix in (".yml", ".yaml"):
                self.floris = Floris.from_yaml(self.configuration)
            elif self.configuration.suffix == ".json":
                self.floris = Floris.from_json(self.configuration)
            else:
                raise ValueError(
                    "The Floris `configuration` file inputs must be of type YAML", "(.yml or .yaml) or JSON (.json)!"
                )
        elif isinstance(self.configuration, dict):
            self.floris = Floris.from_dict(self.configuration)
        else:
            raise TypeError("The Floris `configuration` must of type 'dict', 'str', or 'Path'!")

    def calculate_wake(
        self,
        yaw_angles: NDArrayFloat | list[float] | None = None,
        no_wake: bool = False,
        points: NDArrayFloat | list[float] | None = None,
        track_n_upstream_wakes: bool = False,
    ) -> None:
        """
        Wrapper to the :py:meth:`~.Farm.set_yaw_angles` and
        :py:meth:`~.FlowField.calculate_wake` methods.

        Args:
            yaw_angles (NDArrayFloat | list[float] | None, optional): Turbine yaw angles.
                Defaults to None.
            no_wake: (bool, optional): When *True* updates the turbine
                quantities without calculating the wake or adding the
                wake to the flow field. Defaults to *False*.
            points: (NDArrayFloat | list[float] | None, optional): The x, y, and z
                coordinates at which the flow field velocity is to be recorded. Defaults
                to None.
            track_n_upstream_wakes (bool, optional): When *True*, will keep track of the
                number of upstream wakes a turbine is experiencing. Defaults to *False*.
        """
        if yaw_angles is not None:
            yaw_angles = np.array(yaw_angles)
            self.floris.farm.farm_controller.set_yaw_angles(yaw_angles)

        # TODO: These inputs need to be mapped
        # self.floris.flow_field.calculate_wake(
        #     no_wake=no_wake,
        #     points=points,
        #     track_n_upstream_wakes=track_n_upstream_wakes,
        # )
        self.floris.steady_state_atmospheric_condition()

    def reinitialize_flow_field(
        self,
        wind_speed: list[float] | NDArrayFloat | None = None,
        wind_direction: list[float] | NDArrayFloat | None = None,
        wind_rose_probability: list[float] | NDArrayFloat | None = None,
        wind_layout: list[float] | NDArrayFloat | None = None,
        wind_shear: float | None = None,
        wind_veer: float | None = None,
        specified_wind_height: float | None = None,
        turbulence_intensity=None,
        turbulence_kinetic_energy=None,
        air_density: float | None = None,
        wake: WakeModelManager = None,
        layout_array: list[list[float]] | NDArrayFloat | None = None,
        turbine_id: list[str] | None = None,
        wtg_id: list[str] | None = None,
        with_resolution: float | None = None,
    ):
        """
        Wrapper to :py:meth:`~.flow_field.reinitialize_flow_field`. All input
        values are used to update the :py:class:`~.flow_field.FlowField`
        instance.

        Args:
            wind_speed (list[float] | NDArrayFloat | None, optional): Background wind
                speed. Defaults to None.
            wind_direction (list[float] | NDArrayFloat | None, optional): Background
                wind direction. Defaults to None.
            wind_rose_probability (list[float] | NDArrayFloat | None, optional): The
                probability for each wind direction and wind speed combination, with
                shape (N wind directions, N wind_speeds)
            wind_layout (tuple, optional): Tuple of x- and y-locations of wind speed
                measurements. Defaults to None.
            wind_shear (float | None, optional): Shear exponent. Defaults to None.
            wind_veer (float | None, optional): Direction change over rotor. Defaults
                to None.
            specified_wind_height (float | None, optional): Specified wind height for
                shear. Defaults to None.
            turbulence_intensity (list, optional): Background turbulence
                intensity. Defaults to None.
            turbulence_kinetic_energy (list, optional): Background turbulence
                kinetic energy. Defaults to None.
            air_density (float | None, optional): Ambient air density. Defaults to None.
            wake (:py:class:`~.wake.WakeModelManager` | None, optional): A container
                class :py:class:`~.wake.WakeModelManager` with wake model information
                used to calculate the flow field. Defaults to None.
            layout_array (tuple[list[float]] | NDArrayFloat | None, optional): Array of
                x- and y-locations of wind turbines, with shape (2 x N turbines).
                Defaults to None.
            turbine_id (list[str]| None, optional): The turbine mapping for each of the
                turbines on the wind farm. This **must** be used if `layout_array` uses a
                different number of turbines than the original model and this is a
                multi turbine type farm.
            wtg_id (list[str]| None, optional): The description for each of the
                turbines on the wind farm. This should be used if there is a new number
                of turbines in `layout_array`.
            with_resolution (float, optional): Resolution of output flow_field. Defaults
                to None.
        """
        flow_field_dict = self.floris.flow_field._asdict()
        if wind_speed is not None:
            flow_field_dict["wind_speeds"] = wind_speed
        if wind_direction is not None:
            flow_field_dict["wind_directions"] = wind_direction
        if wind_rose_probability is not None:
            flow_field_dict["probability"] = wind_rose_probability
        else:
            flow_field_dict.pop("probability")
        if wind_shear is not None:
            flow_field_dict["wind_shear"] = wind_shear
        if wind_veer is not None:
            flow_field_dict["wind_veer"] = wind_veer
        if specified_wind_height is not None:
            flow_field_dict["reference_wind_height"] = specified_wind_height
        if air_density is not None:
            flow_field_dict["air_density"] = air_density
        if wake is not None:
            self.floris.wake = wake
        if wind_layout:
            pass  # TODO: will need for heterogeneous flow
        if turbulence_intensity is not None:
            pass  # TODO: this should be in the code, but maybe got skipped?
        if turbulence_kinetic_energy is not None:
            pass  # TODO: not needed until GCH
        if with_resolution is not None:
            # TODO: Update  whereever this goes
            # TODO: This is the grid_resolution, so the xxGrid will have to be updated, except that grid_resolution is is Vec3, or int
            # NOTE: probably not needed because this was for v2 Curl functionality, and we're adding a Curl solver
            pass

        self.floris.flow_field = FlowField.from_dict(flow_field_dict)

        reinitialize_farm = False
        x = self.floris.farm.layout_x
        if layout_array is not None:
            x, y = layout_array
            # Recreating a Farm object to avoid errors if new and original layouts are different sizes
            farm = self.floris.farm._asdict()
            farm["layout_x"] = x
            farm["layout_y"] = y
            farm["wind_directions"] = self.floris.flow_field.wind_directions
            farm["wind_speeds"] = self.floris.flow_field.wind_speeds
            reinitialize_farm = True

        turbine_id_check = len(x) != self.floris.farm.n_turbines
        turbine_id_check &= len(self.floris.farm.turbine_map) > 1

        # Check that turbine_id has bee provided if required
        if turbine_id is None and turbine_id_check:
            raise ValueError(
                "`turbine_id` must be provided if the layout is changing and there are " "multiple turbine types!"
            )

        # If no value is assigned, then create turbine_id
        if turbine_id is None and reinitialize_farm:
            farm["turbine_id"] = [[*self.floris.farm.turbine_map][0]] * len(farm["layout_x"])
            reinitialize_farm = True

        # Assign the new turbine_id if one is input
        if turbine_id is not None:
            if not farm:
                farm = self.floris.farm._asdict()
            farm["turbine_id"] = turbine_id
            reinitialize_farm = True

        if wtg_id is None and reinitialize_farm:
            if len(self.floris.farm.wtg_id) != len(farm["layout_x"]):
                farm.pop("wtg_id")
        if wtg_id is not None:
            if not farm:
                farm = self.floris.farm._asdict()
            farm["wtg_id"] = wtg_id
            reinitialize_farm = True

        if reinitialize_farm:
            self.floris.farm = Farm.from_dict(farm)

    def get_plane_of_points(
        self,
        x1_resolution=200,
        x2_resolution=200,
        normal_vector="z",
        x3_value=100,
        x1_bounds=None,
        x2_bounds=None,
    ):
        """
        Calculates velocity values through the
        :py:meth:`~.FlowField.calculate_wake` method at points in plane
        specified by inputs.

        Args:
            x1_resolution (float, optional): Output array resolution.
                Defaults to 200 points.
            x2_resolution (float, optional): Output array resolution.
                Defaults to 200 points.
            normal_vector (string, optional): Vector normal to plane.
                Defaults to z.
            x3_value (float, optional): Value of normal vector to slice through.
                Defaults to 100.
            x1_bounds (tuple, optional): Limits of output array (in m).
                Defaults to None.
            x2_bounds (tuple, optional): Limits of output array (in m).
                Defaults to None.

        Returns:
            :py:class:`pandas.DataFrame`: containing values of x1, x2, u, v, w
        """
        # Get a copy for the flow field so don't change underlying grid points
        flow_field = copy.deepcopy(self.floris.flow_field)

        if hasattr(self.floris.wake.velocity_model, "requires_resolution"):
            if self.floris.wake.velocity_model.requires_resolution:

                # If this is a gridded model, must extract from full flow field
                self.logger.info(
                    "Model identified as %s requires use of underlying grid points"
                    % self.floris.flow_field.wake.velocity_model.model_string
                )

                # Get the flow data and extract the plane using it
                flow_data = self.get_flow_data()
                return get_plane_from_flow_data(flow_data, normal_vector=normal_vector, x3_value=x3_value)

        coords = np.array([c.elements for c in self.floris.farm.coordinates])
        x, y, _ = coords.T
        max_diameter = max(self.floris.farm.rotor_diameter)
        hub_height = self.floris.farm.hub_height[0]

        # If x1 and x2 bounds are not provided, use rules of thumb
        if normal_vector == "z":  # Rules of thumb for horizontal plane
            if x1_bounds is None:
                x1_bounds = (min(x) - 2 * max_diameter, max(x) + 10 * max_diameter)
            if x2_bounds is None:
                x2_bounds = (min(y) - 2 * max_diameter, max(y) + 2 * max_diameter)
        if normal_vector == "x":  # Rules of thumb for cut plane plane
            if x1_bounds is None:
                x1_bounds = (min(y) - 2 * max_diameter, max(y) + 2 * max_diameter)
            if x2_bounds is None:
                x2_bounds = (10, hub_height * 2)
        if normal_vector == "y":  # Rules of thumb for cut plane plane
            if x1_bounds is None:
                x1_bounds = (min(x) - 2 * max_diameter, max(x) + 10 * max_diameter)
            if x2_bounds is None:
                x2_bounds = (10, hub_height * 2)

        # Set up the points to test
        x1_array = np.linspace(x1_bounds[0], x1_bounds[1], num=x1_resolution)
        x2_array = np.linspace(x2_bounds[0], x2_bounds[1], num=x2_resolution)

        # Grid the points and flatten
        x1_array, x2_array = np.meshgrid(x1_array, x2_array)
        x1_array = x1_array.flatten()
        x2_array = x2_array.flatten()
        x3_array = np.ones_like(x1_array) * x3_value

        # Create the points matrix
        if normal_vector == "z":
            points = np.row_stack((x1_array, x2_array, x3_array))
        if normal_vector == "x":
            points = np.row_stack((x3_array, x1_array, x2_array))
        if normal_vector == "y":
            points = np.row_stack((x1_array, x3_array, x2_array))

        # Recalculate wake with these points
        # TODO: Calculate wake inputs need to be mapped
        raise_error = True
        if raise_error:
            raise NotImplementedError("The specific points functionality is still undefined")
        flow_field.calculate_wake(points=points)

        # Get results vectors
        x_flat = flow_field.x.flatten()
        y_flat = flow_field.y.flatten()
        z_flat = flow_field.z.flatten()
        u_flat = flow_field.u.flatten()
        v_flat = flow_field.v.flatten()
        w_flat = flow_field.w.flatten()

        # Create a df of these
        if normal_vector == "z":
            df = pd.DataFrame(
                {
                    "x1": x_flat,
                    "x2": y_flat,
                    "x3": z_flat,
                    "u": u_flat,
                    "v": v_flat,
                    "w": w_flat,
                }
            )
        if normal_vector == "x":
            df = pd.DataFrame(
                {
                    "x1": y_flat,
                    "x2": z_flat,
                    "x3": x_flat,
                    "u": u_flat,
                    "v": v_flat,
                    "w": w_flat,
                }
            )
        if normal_vector == "y":
            df = pd.DataFrame(
                {
                    "x1": x_flat,
                    "x2": z_flat,
                    "x3": y_flat,
                    "u": u_flat,
                    "v": v_flat,
                    "w": w_flat,
                }
            )

        # Subset to plane
        df = df[df.x3 == x3_value]

        # Drop duplicates
        df = df.drop_duplicates()

        # Limit to requested points
        df = df[df.x1.isin(x1_array)]
        df = df[df.x2.isin(x2_array)]

        # Sort values of df to make sure plotting is acceptable
        df = df.sort_values(["x2", "x1"]).reset_index(drop=True)

        # Return the dataframe
        return df

    def get_set_of_points(self, x_points, y_points, z_points):
        """
        Calculates velocity values through the
        :py:meth:`~.FlowField.calculate_wake` method at points specified by
        inputs.

        Args:
            x_points (float): X-locations to get velocity values at.
            y_points (float): Y-locations to get velocity values at.
            z_points (float): Z-locations to get velocity values at.

        Returns:
            :py:class:`pandas.DataFrame`: containing values of x, y, z, u, v, w
        """
        # Get a copy for the flow field so don't change underlying grid points
        flow_field = copy.deepcopy(self.floris.flow_field)

        if hasattr(self.floris.wake.velocity_model, "requires_resolution"):
            if self.floris.velocity_model.requires_resolution:

                # If this is a gridded model, must extract from full flow field
                self.logger.info(
                    "Model identified as %s requires use of underlying grid print"
                    % self.floris.wake.velocity_model.model_string
                )
                self.logger.warning("FUNCTION NOT AVAILABLE CURRENTLY")

        # Set up points matrix
        points = np.row_stack((x_points, y_points, z_points))

        # TODO: Calculate wake inputs need to be mapped
        raise_error = True
        if raise_error:
            raise NotImplementedError("Additional point calculation is not yet supported!")
        # Recalculate wake with these points
        flow_field.calculate_wake(points=points)

        # Get results vectors
        x_flat = flow_field.x.flatten()
        y_flat = flow_field.y.flatten()
        z_flat = flow_field.z.flatten()
        u_flat = flow_field.u.flatten()
        v_flat = flow_field.v.flatten()
        w_flat = flow_field.w.flatten()

        df = pd.DataFrame(
            {
                "x": x_flat,
                "y": y_flat,
                "z": z_flat,
                "u": u_flat,
                "v": v_flat,
                "w": w_flat,
            }
        )

        # Subset to points requests
        df = df[df.x.isin(x_points)]
        df = df[df.y.isin(y_points)]
        df = df[df.z.isin(z_points)]

        # Drop duplicates
        df = df.drop_duplicates()

        # Return the dataframe
        return df

    def get_hor_plane(
        self,
        height=None,
        x_resolution=200,
        y_resolution=200,
        x_bounds=None,
        y_bounds=None,
    ):
        """
        Shortcut method to instantiate a :py:class:`~.tools.cut_plane.CutPlane`
        object containing the velocity field in a horizontal plane cut through
        the simulation domain at a specific height.

        Args:
            height (float): Height of cut plane. Defaults to Hub-height.
            x_resolution (float, optional): Output array resolution.
                Defaults to 200 points.
            y_resolution (float, optional): Output array resolution.
                Defaults to 200 points.
            x_bounds (tuple, optional): Limits of output array (in m).
                Defaults to None.
            y_bounds (tuple, optional): Limits of output array (in m).
                Defaults to None.

        Returns:
            :py:class:`~.tools.cut_plane.CutPlane`: containing values
            of x, y, u, v, w
        """
        # If height not provided, use the hub height
        if height is None:
            height = self.floris.farm.hub_height[0, 0, 0]  # TODO: needs multi-turbine support
            self.logger.info("Default to hub height = %.1f for horizontal plane." % height)

        # Get the points of data in a dataframe
        df = self.get_plane_of_points(
            x1_resolution=x_resolution,
            x2_resolution=y_resolution,
            normal_vector="z",
            x3_value=height,
            x1_bounds=x_bounds,
            x2_bounds=y_bounds,
        )

        # Compute and return the cutplane
        hor_plane = CutPlane(df)
        if self.floris.wake.velocity_model.model_grid_resolution is not None:
            hor_plane = change_resolution(
                hor_plane,
                resolution=(
                    self.floris.wake.velocity_model.model_grid_resolution.x1,
                    self.floris.wake.velocity_model.model_grid_resolution.x2,
                ),
            )
        return hor_plane

    def get_cross_plane(self, x_loc, y_resolution=200, z_resolution=200, y_bounds=None, z_bounds=None):
        """
        Shortcut method to instantiate a :py:class:`~.tools.cut_plane.CutPlane`
        object containing the velocity field in a vertical plane cut through
        the simulation domain perpendicular to the background flow at a
        specified downstream location.

        Args:
            x_loc (float): Downstream location of cut plane.
            y_resolution (float, optional): Output array resolution.
                Defaults to 200 points.
            z_resolution (float, optional): Output array resolution.
                Defaults to 200 points.
            y_bounds (tuple, optional): Limits of output array (in m).
                Defaults to None.
            z_bounds (tuple, optional): limits of output array (in m).
                Defaults to None.

        Returns:
            :py:class:`~.tools.cut_plane.CutPlane`: containing values
            of y, z, u, v, w
        """
        # Get the points of data in a dataframe
        df = self.get_plane_of_points(
            x1_resolution=y_resolution,
            x2_resolution=z_resolution,
            normal_vector="x",
            x3_value=x_loc,
            x1_bounds=y_bounds,
            x2_bounds=z_bounds,
        )

        # Compute and return the cutplane
        return CutPlane(df)

    def get_y_plane(self, y_loc, x_resolution=200, z_resolution=200, x_bounds=None, z_bounds=None):
        """
        Shortcut method to instantiate a :py:class:`~.tools.cut_plane.CutPlane`
        object containing the velocity field in a vertical plane cut through
        the simulation domain at parallel to the background flow at a specified
        spanwise location.

        Args:
            y_loc (float): Spanwise location of cut plane.
            x_resolution (float, optional): Output array resolution.
                Defaults to 200 points.
            z_resolution (float, optional): Output array resolution.
                Defaults to 200 points.
            x_bounds (tuple, optional): Limits of output array (in m).
                Defaults to None.
            z_bounds (tuple, optional): limits of output array (in m).
                Defaults to None.

        Returns:
            :py:class:`~.tools.cut_plane.CutPlane`: containing values
            of x, z, u, v, w
        """
        # Get the points of data in a dataframe
        df = self.get_plane_of_points(
            x1_resolution=x_resolution,
            x2_resolution=z_resolution,
            normal_vector="y",
            x3_value=y_loc,
            x1_bounds=x_bounds,
            x2_bounds=z_bounds,
        )

        # Compute and return the cutplane
        return CutPlane(df)

    def get_flow_data(self, resolution=None, grid_spacing=10, velocity_deficit=False):
        """
        Generate :py:class:`~.tools.flow_data.FlowData` object corresponding to
        active FLORIS instance.

        Velocity and wake models requiring calculation on a grid implement a
        discretized domain at resolution **grid_spacing**. This is distinct
        from the resolution of the returned flow field domain.

        Args:
            resolution (float, optional): Resolution of output data.
                Only used for wake models that require spatial
                resolution (e.g. curl). Defaults to None.
            grid_spacing (int, optional): Resolution of grid used for
                simulation. Model results may be sensitive to resolution.
                Defaults to 10.
            velocity_deficit (bool, optional): When *True*, normalizes velocity
                with respect to initial flow field velocity to show relative
                velocity deficit (%). Defaults to *False*.

        Returns:
            :py:class:`~.tools.flow_data.FlowData`: FlowData object
        """

        if resolution is None:
            if not self.floris.wake.velocity_model.requires_resolution:
                self.logger.info("Assuming grid with spacing %d" % grid_spacing)
                (
                    xmin,
                    xmax,
                    ymin,
                    ymax,
                    zmin,
                    zmax,
                ) = self.floris.flow_field.domain_bounds  # TODO: No grid attribute within FlowField
                resolution = Vec3(
                    1 + (xmax - xmin) / grid_spacing,
                    1 + (ymax - ymin) / grid_spacing,
                    1 + (zmax - zmin) / grid_spacing,
                )
            else:
                self.logger.info("Assuming model resolution")
                resolution = self.floris.wake.velocity_model.model_grid_resolution

        # Get a copy for the flow field so don't change underlying grid points
        flow_field = copy.deepcopy(self.floris.flow_field)

        if (
            flow_field.wake.velocity_model.requires_resolution
            and flow_field.wake.velocity_model.model_grid_resolution != resolution
        ):
            self.logger.warning(
                "WARNING: The current wake velocity model contains a "
                + "required grid resolution; the Resolution given to "
                + "FlorisInterface.get_flow_field is ignored."
            )
            resolution = flow_field.wake.velocity_model.model_grid_resolution
        flow_field.reinitialize_flow_field(with_resolution=resolution)  # TODO: Not implemented
        self.logger.info(resolution)
        # print(resolution)
        flow_field.steady_state_atmospheric_condition()

        order = "f"
        x = flow_field.x.flatten(order=order)
        y = flow_field.y.flatten(order=order)
        z = flow_field.z.flatten(order=order)

        u = flow_field.u.flatten(order=order)
        v = flow_field.v.flatten(order=order)
        w = flow_field.w.flatten(order=order)

        # find percent velocity deficit
        if velocity_deficit:
            u = abs(u - flow_field.u_initial.flatten(order=order)) / flow_field.u_initial.flatten(order=order) * 100
            v = abs(v - flow_field.v_initial.flatten(order=order)) / flow_field.v_initial.flatten(order=order) * 100
            w = abs(w - flow_field.w_initial.flatten(order=order)) / flow_field.w_initial.flatten(order=order) * 100

        # Determine spacing, dimensions and origin
        unique_x = np.sort(np.unique(x))
        unique_y = np.sort(np.unique(y))
        unique_z = np.sort(np.unique(z))
        spacing = Vec3(
            unique_x[1] - unique_x[0],
            unique_y[1] - unique_y[0],
            unique_z[1] - unique_z[0],
        )
        dimensions = Vec3(len(unique_x), len(unique_y), len(unique_z))
        origin = Vec3(0.0, 0.0, 0.0)
        return FlowData(x, y, z, u, v, w, spacing=spacing, dimensions=dimensions, origin=origin)

    def get_yaw_angles(self):
        """
        Reports yaw angles of wind turbines within the active
        :py:class:`~.turbine_map.TurbineMap` accessible as
        FlorisInterface.floris.tarm.turbine_map.turbines.yaw_angle.

        Returns:
            np.array: Wind turbine yaw angles.
        """
        return self.floris.farm.farm_controller.yaw_angles

    def _get_turbine_powers(self) -> NDArrayFloat:
        """Calculates the power at each turbine in the windfarm.

        Returns:
            NDArrayFloat: [description]
        """
        air_density = np.full(
            (
                self.floris.flow_field.n_wind_directions,
                self.floris.flow_field.n_wind_speeds,
                self.floris.farm.n_turbines,
            ),
            self.floris.flow_field.air_density,
        )
        farm_power = power(
            air_density=air_density,
            velocities=self.floris.flow_field.u,
            yaw_angle=self.floris.farm.farm_controller.yaw_angles,
            pP=self.floris.farm.pP,
            power_interp=self.floris.farm.power_interp,
        )
        return farm_power

    def get_farm_power(
        self,
        include_unc=False,
        unc_pmfs=None,
        unc_options=None,
        no_wake=False,
        use_turbulence_correction=False,
    ):
        """
        Report wind plant power from instance of floris. Optionally includes
        uncertainty in wind direction and yaw position when determining power.
        Uncertainty is included by computing the mean wind farm power for a
        distribution of wind direction and yaw position deviations from the
        original wind direction and yaw angles.

        Args:
            include_unc (bool): When *True*, uncertainty in wind direction
                and/or yaw position is included when determining wind farm
                power. Defaults to *False*.
            unc_pmfs (dictionary, optional): A dictionary containing optional
                probability mass functions describing the distribution of wind
                direction and yaw position deviations when wind direction and/or
                yaw position uncertainty is included in the power calculations.
                Contains the following key-value pairs:

                -   **wd_unc** (*np.array*): Wind direction deviations from the
                    original wind direction.
                -   **wd_unc_pmf** (*np.array*): Probability of each wind
                    direction deviation in **wd_unc** occuring.
                -   **yaw_unc** (*np.array*): Yaw angle deviations from the
                    original yaw angles.
                -   **yaw_unc_pmf** (*np.array*): Probability of each yaw angle
                    deviation in **yaw_unc** occuring.

                Defaults to None, in which case default PMFs are calculated
                using values provided in **unc_options**.
            unc_options (dictionary, optional): A dictionary containing values
                used to create normally-distributed, zero-mean probability mass
                functions describing the distribution of wind direction and yaw
                position deviations when wind direction and/or yaw position
                uncertainty is included. This argument is only used when
                **unc_pmfs** is None and contains the following key-value pairs:

                -   **std_wd** (*float*): A float containing the standard
                    deviation of the wind direction deviations from the
                    original wind direction.
                -   **std_yaw** (*float*): A float containing the standard
                    deviation of the yaw angle deviations from the original yaw
                    angles.
                -   **pmf_res** (*float*): A float containing the resolution in
                    degrees of the wind direction and yaw angle PMFs.
                -   **pdf_cutoff** (*float*): A float containing the cumulative
                    distribution function value at which the tails of the
                    PMFs are truncated.

                Defaults to None. Initializes to {'std_wd': 4.95, 'std_yaw':
                1.75, 'pmf_res': 1.0, 'pdf_cutoff': 0.995}.
            no_wake: (bool, optional): When *True* updates the turbine
                quantities without calculating the wake or adding the
                wake to the flow field. Defaults to *False*.
            use_turbulence_correction: (bool, optional): When *True* uses a
                turbulence parameter to adjust power output calculations.
                Defaults to *False*.

        Returns:
            float: Sum of wind turbine powers.
        """
        # TODO: Turbulence correction used in the power calculation, but may not be in
        # the model yet
        # TODO: Turbines need a switch for using turbulence correction
        # TODO: Uncomment out the following two lines once the above are resolved
        # for turbine in self.floris.farm.turbines:
        #     turbine.use_turbulence_correction = use_turbulence_correction

        if include_unc:
            unc_pmfs = _generate_uncertainty_parameters(unc_options, unc_pmfs)

            # TODO: The original form of this is:
            # self.floris.farm.wind_map.input_direction[0], but it's unclear why we're
            # capping at just the first wind direction. Should this behavior be kept?
            # I'm unsure as to how the first wind direction is the original, so it could
            # just be a naming thing that's throwing me off....
            wd_orig = self.floris.flow_field.wind_directions

            yaw_angles = self.get_yaw_angles()
            self.reinitialize_flow_field(wind_direction=wd_orig + unc_pmfs["wd_unc"])
            power_at_yaw = [
                self.get_farm_power_for_yaw_angle(yaw_angles + delta_yaw, no_wake=no_wake)
                for delta_yaw in unc_pmfs["yaw_unc"]
            ]
            mean_farm_power = unc_pmfs["wd_unc_pmf"] * unc_pmfs["yaw_unc_pmf"] * np.array(power_at_yaw)

            # reinitialize with original values
            self.reinitialize_flow_field(wind_direction=wd_orig)
            self.calculate_wake(yaw_angles=yaw_angles, no_wake=no_wake)
            return mean_farm_power

        return self._get_turbine_powers().sum()

    def get_turbine_layout(self, z=False):
        """
        Get turbine layout

        Args:
            z (bool): When *True*, return lists of x, y, and z coords,
            otherwise, return x and y only. Defaults to *False*.

        Returns:
            np.array: lists of x, y, and (optionally) z coordinates of
                      each turbine
        """
        xcoords, ycoords, zcoords = np.array([c.elements for c in self.floris.farm.coordinates]).T
        if z:
            return xcoords, ycoords, zcoords
        else:
            return xcoords, ycoords

    def get_turbine_power(
        self,
        include_unc=False,
        unc_pmfs=None,
        unc_options=None,
        no_wake=False,
        use_turbulence_correction=False,
    ):
        """
        Report power from each wind turbine.

        Args:
            include_unc (bool): If *True*, uncertainty in wind direction
                and/or yaw position is included when determining turbine
                powers. Defaults to *False*.
            unc_pmfs (dictionary, optional): A dictionary containing optional
                probability mass functions describing the distribution of wind
                direction and yaw position deviations when wind direction and/or
                yaw position uncertainty is included in the power calculations.
                Contains the following key-value pairs:

                -   **wd_unc** (*np.array*): Wind direction deviations from the
                    original wind direction.
                -   **wd_unc_pmf** (*np.array*): Probability of each wind
                    direction deviation in **wd_unc** occuring.
                -   **yaw_unc** (*np.array*): Yaw angle deviations from the
                    original yaw angles.
                -   **yaw_unc_pmf** (*np.array*): Probability of each yaw angle
                    deviation in **yaw_unc** occuring.

                Defaults to None, in which case default PMFs are calculated
                using values provided in **unc_options**.
            unc_options (dictionary, optional): A dictionary containing values
                used to create normally-distributed, zero-mean probability mass
                functions describing the distribution of wind direction and yaw
                position deviations when wind direction and/or yaw position
                uncertainty is included. This argument is only used when
                **unc_pmfs** is None and contains the following key-value pairs:

                -   **std_wd** (*float*): A float containing the standard
                    deviation of the wind direction deviations from the
                    original wind direction.
                -   **std_yaw** (*float*): A float containing the standard
                    deviation of the yaw angle deviations from the original yaw
                    angles.
                -   **pmf_res** (*float*): A float containing the resolution in
                    degrees of the wind direction and yaw angle PMFs.
                -   **pdf_cutoff** (*float*): A float containing the cumulative
                    distribution function value at which the tails of the
                    PMFs are truncated.

                Defaults to None. Initializes to {'std_wd': 4.95, 'std_yaw': 1.
                75, 'pmf_res': 1.0, 'pdf_cutoff': 0.995}.
            no_wake: (bool, optional): When *True* updates the turbine
                quantities without calculating the wake or adding the
                wake to the flow field. Defaults to *False*.
            use_turbulence_correction: (bool, optional): When *True* uses a
                turbulence parameter to adjust power output calculations.
                Defaults to *False*.

        Returns:
            np.array: Power produced by each wind turbine.
        """
        # TODO: Turbulence correction used in the power calculation, but may not be in
        # the model yet
        # TODO: Turbines need a switch for using turbulence correction
        # TODO: Uncomment out the following two lines once the above are resolved
        # for turbine in self.floris.farm.turbines:
        #     turbine.use_turbulence_correction = use_turbulence_correction

        if include_unc:
            unc_pmfs = _generate_uncertainty_parameters(unc_options, unc_pmfs)

            mean_farm_power = np.zeros(self.floris.farm.n_turbines)
            wd_orig = self.floris.flow_field.wind_directions  # TODO: same comment as in get_farm_power

            yaw_angles = self.get_yaw_angles()
            self.reinitialize_flow_field(wind_direction=wd_orig[0] + unc_pmfs["wd_unc"])
            for i, delta_yaw in enumerate(unc_pmfs["yaw_unc"]):
                self.calculate_wake(
                    yaw_angles=list(np.array(yaw_angles) + delta_yaw),
                    no_wake=no_wake,
                )
                mean_farm_power += unc_pmfs["wd_unc_pmf"] * unc_pmfs["yaw_unc_pmf"][i] * self._get_turbine_powers()

            # reinitialize with original values
            self.reinitialize_flow_field(wind_direction=wd_orig)
            self.calculate_wake(yaw_angles=yaw_angles, no_wake=no_wake)
            return mean_farm_power

        return self._get_turbine_powers()

    def get_power_curve(self, wind_speeds):
        """
        Return the power curve given a set of wind speeds

        Args:
            wind_speeds (np.array): array of wind speeds to get power curve
        """

        # TODO: Why is this done? Should we expand for evenutal multiple turbines types
        # or just allow a filter on the turbine index?
        # Temporarily set the farm to a single turbine
        saved_layout_x = self.layout_x
        saved_layout_y = self.layout_y

        self.reinitialize_flow_field(wind_speed=wind_speeds, layout_array=([0], [0]))
        self.calculate_wake()
        turbine_power = self._get_turbine_powers()

        # Set it back
        self.reinitialize_flow_field(layout_array=(saved_layout_x, saved_layout_y))

        return turbine_power

    def get_turbine_ct(self):
        """
        Reports thrust coefficient from each wind turbine.

        Returns:
            list: Thrust coefficient for each wind turbine.
        """
        turb_ct_array = Ct(
            velocities=self.floris.flow_field.u,
            yaw_angle=self.floris.farm.farm_controller.yaw_angles,
            fCt=self.floris.farm.fCt_interp,
        )
        return turb_ct_array

    def get_turbine_ti(self):
        """
        Reports turbulence intensity  from each wind turbine.

        Returns:
            list: Thrust ti for each wind turbine.
        """
        # TODO: Are we modeling this piece anymore?
        turb_ti_array = [
            turbine.current_turbulence_intensity for turbine in self.floris.flow_field.turbine_map.turbines
        ]
        return turb_ti_array

        # calculate the power under different yaw angles

    def get_farm_power_for_yaw_angle(
        self,
        yaw_angles,
        include_unc=False,
        unc_pmfs=None,
        unc_options=None,
        no_wake=False,
    ):
        """
        Assign yaw angles to turbines, calculate wake, and report farm power.

        Args:
            yaw_angles (np.array): Yaw to apply to each turbine.
            include_unc (bool, optional): When *True*, includes wind direction
                uncertainty in estimate of wind farm power. Defaults to *False*.
            unc_pmfs (dictionary, optional): A dictionary containing optional
                probability mass functions describing the distribution of wind
                direction and yaw position deviations when wind direction and/or
                yaw position uncertainty is included in the power calculations.
                Contains the following key-value pairs:

                -   **wd_unc** (*np.array*): Wind direction deviations from the
                    original wind direction.
                -   **wd_unc_pmf** (*np.array*): Probability of each wind
                    direction deviation in **wd_unc** occuring.
                -   **yaw_unc** (*np.array*): Yaw angle deviations from the
                    original yaw angles.
                -   **yaw_unc_pmf** (*np.array*): Probability of each yaw angle
                    deviation in **yaw_unc** occuring.

                Defaults to None, in which case default PMFs are calculated
                using values provided in **unc_options**.
            unc_options (dictionary, optional): A dictionary containing values
                used to create normally-distributed, zero-mean probability mass
                functions describing the distribution of wind direction and yaw
                position deviations when wind direction and/or yaw position
                uncertainty is included. This argument is only used when
                **unc_pmfs** is None and contains the following key-value pairs:

                -   **std_wd** (*float*): A float containing the standard
                    deviation of the wind direction deviations from the
                    original wind direction.
                -   **std_yaw** (*float*): A float containing the standard
                    deviation of the yaw angle deviations from the original yaw
                    angles.
                -   **pmf_res** (*float*): A float containing the resolution in
                    degrees of the wind direction and yaw angle PMFs.
                -   **pdf_cutoff** (*float*): A float containing the cumulative
                    distribution function value at which the tails of the
                    PMFs are truncated.

                Defaults to None. Initializes to {'std_wd': 4.95, 'std_yaw': 1.
                75, 'pmf_res': 1.0, 'pdf_cutoff': 0.995}.
            no_wake: (bool, optional): When *True* updates the turbine
                quantities without calculating the wake or adding the
                wake to the flow field. Defaults to *False*.

        Returns:
            float: Wind plant power. #TODO negative? in kW?
        """

        self.calculate_wake(yaw_angles=yaw_angles, no_wake=no_wake)

        return self.get_farm_power(include_unc=include_unc, unc_pmfs=unc_pmfs, unc_options=unc_options)

    def get_farm_AEP(
        self,
        wd: NDArrayFloat | list[float],
        ws: NDArrayFloat | list[float],
        freq: NDArrayFloat | list[list[float]],
        yaw: NDArrayFloat | list[float] | None = None,
        limit_ws: bool = False,
        ws_limit_tol: float = 0.001,
        ws_cutout: float = 30.0,
    ) -> float:
        """
        Estimate annual energy production (AEP) for distributions of wind speed, wind
        direction, wind rose probability, and yaw offset. This can be computed for
        pre-determined wind direction and wind speed combinations, as was the case in
        FLORIS v2, or additionally, the unique wind directions, wind speeds, and their
        probabilities can be input.

        Args:
            wd (NDArrayFloat | list[float]): List or array of wind direction values.
                Either a unique list of wind directions can be used or the wind
                directions corresponding to a pre-computed set of combinations
                should be used.
            ws (NDArrayFloat | list[float]): List or array of wind speed values.
                Either a unique list of wind speeds can be used or the wind speeds
                corresponding to a pre-computed set of combinations should be used.
            freq (NDArrayFloat | list[list[float]]): Frequencies corresponding to either
                the pre-computed combinations of wind directions and wind speeds or the
                full wind rose with dimensions (N wind directions x N wind speeds).
            yaw (NDArrayFloat | list[float] | None, optional): List or array of yaw
                values if wake is steering implemented that correspond with the number
                of wind directions. Defaults to None.
            limit_ws (bool, optional): When *True*, detect wind speed when power
                reaches it's maximum value for a given wind direction. For all
                higher wind speeds, use last calculated value when below cut
                out. Defaults to False.
            ws_limit_tol (float, optional): Tolerance fraction for determining
                wind speed where power stops changing. If limit_ws is *True*,
                assume power remains constant up to cut out for wind speeds
                above the point where power changes less than ws_limit_tol of
                the previous power. Defaults to 0.001.
            ws_cutout (float, optional): Cut out wind speed (m/s). If limit_ws
                is *True*, assume power is zero for wind speeds greater than or
                equal to ws_cutout. Defaults to 30.0

        Returns:
            float: AEP for wind farm.
        """

        # Convert the required inputs to arrays
        wd = np.array(wd)
        ws = np.array(ws)
        freq = np.array(freq)

        # Determine if the direction and speed inputs provided are a set of pre-determined
        # combinations, and compute the full combination set if so, where the value
        # in freq will be set to 0 if a combination was not in the original set to ensure
        # it's not counted.
        wd_unique = wd.unique()
        ws_unique = ws.unique()
        if np.array_equal(wd_unique, sorted(wd)) and np.array_equal(ws_unique, sorted(ws)):
            # Reshape the frequency input if required, and leave the unique inputs as-is
            if freq.shape != (wd_unique.size, ws_unique.size):
                freq = freq.reshape((wd_unique.size, ws_unique.size))
        else:
            # Compute all the combinations
            wd_unique, ws_unique, freq, yaw = correct_for_all_combinations(wd, ws, freq, yaw)

        # If the yaw input is still None, then create a None array as inputs
        if yaw is None:
            N = wd_unique.size * ws_unique.size
            yaw = np.array([None] * N).reshape(wd_unique.size, ws_unique.size)
        else:
            yaw = np.array(yaw)

        # filter out wind speeds beyond the cutoff, if necessary
        if limit_ws:
            ix_ws_filter = ws_unique >= ws_cutout
            ws_unique = ws_unique[ix_ws_filter]
            freq = freq[:, ix_ws_filter]
            yaw = yaw[:, ix_ws_filter]

        self.reinitialize_flow_field(wind_direction=wd_unique, wind_speed=ws_unique, wind_rose_probability=freq)
        self.calculate_wake(yaw_angles=yaw)
        farm_power = self.get_farm_power()  # TODO: Do we need to specify an axis since this is a sum?
        AEP = farm_power * freq * 8760
        return AEP.sum()

    def _calc_one_AEP_case(self, wd, ws, freq, yaw=None):
        self.reinitialize_flow_field(wind_direction=[wd], wind_speed=[ws])
        self.calculate_wake(yaw_angles=yaw)
        return self.get_farm_power() * freq * 8760

    def get_farm_AEP_parallel(
        self,
        wd: NDArrayFloat | list[float],
        ws: NDArrayFloat | list[float],
        freq: NDArrayFloat | list[list[float]],
        yaw: NDArrayFloat | list[float] | None = None,
        jobs=-1,
    ):
        """
        Estimate annual energy production (AEP) for distributions of wind
        speed, wind direction and yaw offset with parallel computations on
        a single comptuer.

        # TODO: Update the docstrings and allow for the use of precomputed combinations
        as well as unique inputs that need to be computed. Same for the other AEPs

        Args:
            wd (iterable): List or array of wind direction values.
            ws (iterable): List or array of wind speed values.
            freq (iterable): Frequencies corresponding to wind direction and wind speed
                combinations in the wind rose with, shape (N wind directions x N wind speeds).
            yaw (iterable, optional): List or array of yaw values if wake is steering
                implemented, with shape (N wind directions). Defaults to None.
            jobs (int, optional): The number of jobs (cores) to use in the parallel
                computations.

        Returns:
            float: AEP for wind farm.
        """
        if jobs < -1:
            raise ValueError("Input 'jobs' cannot be negative!")
        if jobs == -1:
            jobs = int(np.ceil(cpu_count() * 0.8))
        if jobs > 0:
            jobs = min(jobs, cpu_count())
        if jobs > len(wd):
            jobs = len(wd)

        if yaw is None:
            yaw = [None] * len(wd)

        wd = np.array(wd)
        ws = np.array(ws)
        freq = np.array(freq)

        # Make one large list of arguments, then flatten and resort the nested tuples
        # to the correct ordering of self, wd, ws, freq, yaw
        global_arguments = list(zip(repeat(self), zip(wd, yaw), ws, freq.flatten()))
        # OR is this supposed to be all wind speeds for each wind direction?:
        # global_arguments = list(zip(repeat(self), zip(wd, yaw), repeat(ws), freq))
        # global_arguments = [(s, n[0], wspd, f, n[1]) for s, n, wspd, f in global_arguments]
        global_arguments = [(s, n[0][0], n[1], f, n[0][1]) for s, n, f in global_arguments]

        num_cases = wd.size * ws.size
        chunksize = int(np.ceil(num_cases / jobs))

        with Pool(jobs) as pool:
            opt = pool.starmap(global_calc_one_AEP_case, global_arguments, chunksize=chunksize)
            # add AEP to overall AEP

        return 0.0 + np.sum(opt)

    def calculate_AEP_wind_limit(self, num_turbines, x_spacing, start_ws, threshold):
        orig_layout_x = self.layout_x
        orig_layout_y = self.layout_y
        D = self.floris.farm.turbines[0].rotor_diameter

        self.reinitialize_flow_field(
            layout_array=(
                [i * x_spacing * D for i in range(num_turbines)],
                [0.0] * num_turbines,
            ),
            wind_speed=start_ws,
        )
        self.calculate_wake()

        prev_power = 1.0
        cur_power = self.get_farm_power()
        ws = start_ws

        while np.abs(prev_power - cur_power) / prev_power > threshold:
            prev_power = cur_power
            ws += 0.2
            self.reinitialize_flow_field(wind_speed=ws)
            self.calculate_wake()
            cur_power = self.get_farm_power()
        ws += 1.0

        self.reinitialize_flow_field(layout_array=(orig_layout_x, orig_layout_y), wind_speed=ws)
        self.calculate_wake()
        self.max_power = self.get_farm_power()
        self.ws_limit = ws

    def copy_and_update_turbine_map(
        self, base_turbine_id: str, update_parameters: dict, new_id: str | None = None
    ) -> dict:
        """Creates a new copy of an existing turbine and updates the parameters based on
        user input. This function is a helper to make the v2 -> v3 transition easier.

        Args:
            base_turbine_id (str): The base turbine's ID in `floris.farm.turbine_id`.
            update_parameters (dict): A dictionary of the turbine parameters to update
                and their new valies.
            new_id (str, optional): The new `turbine_id`, if `None` a unique
                identifier will be appended to the end. Defaults to None.

        Returns:
            dict: A turbine mapping that can be passed directly to `change_turbine`.
        """
        if new_id is None:
            new_id = f"{base_turbine_id}_copy{self.unique_copy_id}"
            self.unique_copy_id += 1

        turbine = {new_id: self.floris.turbine[base_turbine_id]._asdict()}
        turbine[new_id].update(update_parameters)
        return turbine

    def change_turbine(
        self,
        turbine_indices: list[int],
        new_turbine_map: dict[str, dict[str, Any]],
        update_specified_wind_height: bool = False,
    ):
        """
        Change turbine properties for specified turbines.

        Args:
            turbine_indices (list[int]): List of turbine indices to change.
            new_turbine_map (dict[str, dict[str, Any]]): New dictionary of turbine
                parameters to create the new turbines for each of `turbine_indices`.
            update_specified_wind_height (bool, optional): When *True*, update specified
                wind height to match new hub_height. Defaults to *False*.
        """
        new_turbine = True
        new_turbine_id = [*new_turbine_map][0]
        if new_turbine_id in self.floris.farm.turbine_map:
            new_turbine = False
            self.logger.info(f"Turbines {turbine_indices} will be re-mapped to the definition for: {new_turbine_id}")

        self.floris.farm.turbine_id = [
            new_turbine_id if i in turbine_indices else t_id for i, t_id in enumerate(self.floris.farm.turbine_id)
        ]
        if new_turbine:
            self.logger.info(f"Turbines {turbine_indices} have been mapped to the new definition for: {new_turbine_id}")

        # Update the turbine mapping if a new turbine was provided, then regenerate the
        # farm arrays for the turbine farm
        if new_turbine:
            turbine_map = self.floris.farm._asdict()["turbine_map"]
            turbine_map.update(new_turbine_map)
            self.floris.farm.turbine_map = turbine_map
        self.floris.farm.generate_farm_points()

        new_hub_height = new_turbine_map[new_turbine_id]["hub_height"]
        changed_hub_height = new_hub_height != self.floris.flow_field.reference_wind_height

        # Alert user if changing hub-height and not specified wind height
        if changed_hub_height and not update_specified_wind_height:
            self.logger.info("Note, updating hub height but not updating " + "the specfied_wind_height")

        if changed_hub_height and update_specified_wind_height:
            self.logger.info(f"Note, specfied_wind_height changed to hub-height: {new_hub_height}")
            self.reinitialize_flow_field(specified_wind_height=new_hub_height)

        # Finish by re-initalizing the flow field
        self.reinitialize_flow_field()

    def set_use_points_on_perimeter(self, use_points_on_perimeter=False):
        """
        Set whether to use the points on the rotor diameter (perimeter) when
        calculating flow field and wake.

        Args:
            use_points_on_perimeter (bool): When *True*, use points at rotor
                perimeter in wake and flow calculations. Defaults to *False*.
        """
        for turbine in self.floris.farm.turbines:
            turbine.use_points_on_perimeter = use_points_on_perimeter
            turbine.initialize_turbine()

    def set_gch(self, enable=True):
        """
        Enable or disable Gauss-Curl Hybrid (GCH) functions
        :py:meth:`~.GaussianModel.calculate_VW`,
        :py:meth:`~.GaussianModel.yaw_added_recovery_correction`, and
        :py:attr:`~.VelocityDeflection.use_secondary_steering`.

        Args:
            enable (bool, optional): Flag whether or not to implement flow
                corrections from GCH model. Defaults to *True*.
        """
        self.set_gch_yaw_added_recovery(enable)
        self.set_gch_secondary_steering(enable)

    def set_gch_yaw_added_recovery(self, enable=True):
        """
        Enable or Disable yaw-added recovery (YAR) from the Gauss-Curl Hybrid
        (GCH) model and the control state of
        :py:meth:`~.GaussianModel.calculate_VW_velocities` and
        :py:meth:`~.GaussianModel.yaw_added_recovery_correction`.

        Args:
            enable (bool, optional): Flag whether or not to implement yaw-added
                recovery from GCH model. Defaults to *True*.
        """
        model_params = self.get_model_parameters()
        use_secondary_steering = model_params["Wake Deflection Parameters"]["use_secondary_steering"]

        if enable:
            model_params["Wake Velocity Parameters"]["use_yaw_added_recovery"] = True

            # If enabling be sure calc vw is on
            model_params["Wake Velocity Parameters"]["calculate_VW_velocities"] = True

        if not enable:
            model_params["Wake Velocity Parameters"]["use_yaw_added_recovery"] = False

            # If secondary steering is also off, disable calculate_VW_velocities
            if not use_secondary_steering:
                model_params["Wake Velocity Parameters"]["calculate_VW_velocities"] = False

        self.set_model_parameters(model_params)
        self.reinitialize_flow_field()

    def set_gch_secondary_steering(self, enable=True):
        """
        Enable or Disable secondary steering (SS) from the Gauss-Curl Hybrid
        (GCH) model and the control state of
        :py:meth:`~.GaussianModel.calculate_VW_velocities` and
        :py:attr:`~.VelocityDeflection.use_secondary_steering`.

        Args:
            enable (bool, optional): Flag whether or not to implement secondary
            steering from GCH model. Defaults to *True*.
        """
        model_params = self.get_model_parameters()
        use_yaw_added_recovery = model_params["Wake Velocity Parameters"]["use_yaw_added_recovery"]

        if enable:
            model_params["Wake Deflection Parameters"]["use_secondary_steering"] = True

            # If enabling be sure calc vw is on
            model_params["Wake Velocity Parameters"]["calculate_VW_velocities"] = True

        if not enable:
            model_params["Wake Deflection Parameters"]["use_secondary_steering"] = False

            # If yar is also off, disable calculate_VW_velocities
            if not use_yaw_added_recovery:
                model_params["Wake Velocity Parameters"]["calculate_VW_velocities"] = False

        self.set_model_parameters(model_params)
        self.reinitialize_flow_field()

    @property
    def layout_x(self):
        """
        Wind turbine coordinate information.

        Returns:
            np.array: Wind turbine x-coordinate.
        """
        return [c.x1 for c in self.farm.coordinates]

    @property
    def layout_y(self):
        """
        Wind turbine coordinate information.

        Returns:
            np.array: Wind turbine y-coordinate.
        """
        return [c.x2 for c in self.farm.coordinates]

    def TKE_to_TI(self, turbulence_kinetic_energy, wind_speed):
        """
        Converts a list of turbulence kinetic energy values to
        turbulence intensity.

        Args:
            turbulence_kinetic_energy (list): Values of turbulence kinetic
                energy in units of meters squared per second squared.
            wind_speed (list): Measurements of wind speed in meters per second.

        Returns:
            list: converted turbulence intensity values expressed as a decimal
            (e.g. 10%TI -> 0.10).
        """
        turbulence_intensity = [
            (np.sqrt((2 / 3) * turbulence_kinetic_energy[i])) / wind_speed[i]
            for i in range(len(turbulence_kinetic_energy))
        ]

        return turbulence_intensity

    def set_rotor_diameter(self, rotor_diameter):
        """
        This function has been replaced and no longer works correctly, assigning an error
        """
        raise Exception(
            "function set_rotor_diameter has been removed.  Please use the function change_turbine going forward.  See examples/change_turbine for syntax"
        )

    def show_model_parameters(
        self,
        params=None,
        verbose=False,
        wake_velocity_model=True,
        wake_deflection_model=True,
        turbulence_model=True,
    ):
        """
        Helper function to print the current wake model parameters and values.
        Shortcut to :py:meth:`~.tools.interface_utilities.show_params`.

        Args:
            params (list, optional): Specific model parameters to be returned,
                supplied as a list of strings. If None, then returns all
                parameters. Defaults to None.
            verbose (bool, optional): If set to *True*, will return the
                docstrings for each parameter. Defaults to *False*.
            wake_velocity_model (bool, optional): If set to *True*, will return
                parameters from the wake_velocity model. If set to *False*, will
                exclude parameters from the wake velocity model. Defaults to
                *True*.
            wake_deflection_model (bool, optional): If set to *True*, will
                return parameters from the wake deflection model. If set to
                *False*, will exclude parameters from the wake deflection
                model. Defaults to *True*.
            turbulence_model (bool, optional): If set to *True*, will return
                parameters from the wake turbulence model. If set to *False*,
                will exclude parameters from the wake turbulence model.
                Defaults to *True*.
        """
        show_params(
            self.floris.wake,
            params,
            verbose,
            wake_velocity_model,
            wake_deflection_model,
            turbulence_model,
        )

    def get_model_parameters(
        self,
        params=None,
        wake_velocity_model=True,
        wake_deflection_model=True,
        turbulence_model=True,
    ):
        """
        Helper function to return the current wake model parameters and values.
        Shortcut to :py:meth:`~.tools.interface_utilities.get_params`.

        Args:
            params (list, optional): Specific model parameters to be returned,
                supplied as a list of strings. If None, then returns all
                parameters. Defaults to None.
            wake_velocity_model (bool, optional): If set to *True*, will return
                parameters from the wake_velocity model. If set to *False*, will
                exclude parameters from the wake velocity model. Defaults to
                *True*.
            wake_deflection_model (bool, optional): If set to *True*, will
                return parameters from the wake deflection model. If set to
                *False*, will exclude parameters from the wake deflection
                model. Defaults to *True*.
            turbulence_model ([type], optional): If set to *True*, will return
                parameters from the wake turbulence model. If set to *False*,
                will exclude parameters from the wake turbulence model.
                Defaults to *True*.

        Returns:
            dict: Dictionary containing model parameters and their values.
        """
        model_params = get_params(
            self.floris.wake, params, wake_velocity_model, wake_deflection_model, turbulence_model
        )

        return model_params

    def set_model_parameters(self, params, verbose=True):
        """
        Helper function to set current wake model parameters.
        Shortcut to :py:meth:`~.tools.interface_utilities.set_params`.

        Args:
            params (dict): Specific model parameters to be set, supplied as a
                dictionary of key:value pairs.
            verbose (bool, optional): If set to *True*, will print information
                about each model parameter that is changed. Defaults to *True*.
        """
        self.floris.wake = set_params(self.floris.wake, params, verbose)

    def vis_layout(
        self,
        ax=None,
        show_wake_lines=False,
        limit_dist=None,
        turbine_face_north=False,
        one_index_turbine=False,
        black_and_white=False,
    ):
        """
        Visualize the layout of the wind farm in the floris instance.
        Shortcut to :py:meth:`~.tools.layout_functions.visualize_layout`.

        Args:
            ax (:py:class:`matplotlib.pyplot.axes`, optional):
                Figure axes. Defaults to None.
            show_wake_lines (bool, optional): Flag to control plotting of
                wake boundaries. Defaults to False.
            limit_dist (float, optional): Downstream limit to plot wakes.
                Defaults to None.
            turbine_face_north (bool, optional): Force orientation of wind
                turbines. Defaults to False.
            one_index_turbine (bool, optional): If *True*, 1st turbine is
                turbine 1.
        """
        for i, turbine in enumerate(self.floris.farm.turbines):
            D = turbine.rotor_diameter
            break
        layout_x, layout_y = self.get_turbine_layout()

        turbineLoc = build_turbine_loc(layout_x, layout_y)

        # Show visualize the turbine layout
        visualize_layout(
            turbineLoc,
            D,
            ax=ax,
            show_wake_lines=show_wake_lines,
            limit_dist=limit_dist,
            turbine_face_north=turbine_face_north,
            one_index_turbine=one_index_turbine,
            black_and_white=black_and_white,
        )

    def show_flow_field(self, ax=None):
        """
        Shortcut method to
        :py:meth:`~.tools.visualization.visualize_cut_plane`.

        Args:
            ax (:py:class:`matplotlib.pyplot.axes` optional):
                Figure axes. Defaults to None.
        """
        # Get horizontal plane at default height (hub-height)
        hor_plane = self.get_hor_plane()

        # Plot and show
        if ax is None:
            fig, ax = plt.subplots()
        visualize_cut_plane(hor_plane, ax=ax)
        plt.show()

    # TODO
    # Comment this out until sure we'll need it
    # def get_velocity_at_point(self, points, initial = False):
    #     """
    #     Get waked velocity at specified points in the flow field.

    #     Args:
    #         points (np.array): x, y and z coordinates of specified point(s)
    #             where flow_field velocity should be reported.
    #         initial(bool, optional): if set to True, the initial velocity of
    #             the flow field is returned instead of the waked velocity.
    #             Defaults to False.

    #     Returns:
    #         velocity (list): flow field velocity at specified grid point(s), in m/s.
    #     """
    #     xp, yp, zp = points[0], points[1], points[2]
    #     x, y, z = self.floris.flow_field.x, self.floris.flow_field.y, self.floris.flow_field.z
    #     velocity = self.floris.flow_field.u
    #     initial_velocity = self.floris.farm.wind_map.grid_wind_speed
    #     pVel = []
    #     for i in range(len(xp)):
    #         xloc, yloc, zloc =np.array(x == xp[i]),np.array(y == yp[i]),np.array(z == zp[i])
    #         loc = np.logical_and(np.logical_and(xloc, yloc) == True, zloc == True)
    #         if initial == True: pVel.append(np.mean(initial_velocity[loc]))
    #         else: pVel.append(np.mean(velocity[loc]))

    #     return pVel
