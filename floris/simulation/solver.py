# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

import copy
from abc import abstractmethod

import attrs
import numpy as np
from attrs import field, define

from floris.type_dec import NDArrayFloat
from floris.simulation import (
    Ct,
    Farm,
    FlowField,
    TurbineGrid,
    FlowFieldGrid,
    axial_induction,
)
from floris.simulation.wake import WakeModelManager
from floris.simulation.turbine import average_velocity
from floris.simulation.wake_deflection.gauss import (
    wake_added_yaw,
    yaw_added_turbulence_mixing,
    calculate_transverse_velocity,
)


def _expansion_mean(x: NDArrayFloat) -> NDArrayFloat:
    """Calculates the mean of the `x` over axis (3, 4) and returns the result as a new 5-dimensional
    object to align with expected internal structures.

    Args:
        x (NDArrayFloat): An array to calculate the mean.

    Returns:
        NDArrayFloat: The mean over axis 3 and 4 at turbine i and expandd to 5 dimensions.
    """
    return np.mean(x, axis=(3, 4))[:, :, :, None, None]


def _expansion_mean_i(x: NDArrayFloat, i: int) -> NDArrayFloat:
    """Calculates the mean of the `x` over axis (3, 4) at turbine index `i` and returns the result
    as a new 5-dimensional object to align with expected internal structures.

    Args:
        x (NDArrayFloat): An array to calculate the mean.
        i (int): The turbine index for where to compute the mean.

    Returns:
        NDArrayFloat: The mean over axis 3 and 4 at turbine i and expandd to 5 dimensions.
    """
    return _expansion_mean(x[:, :, i:i+1])


def calculate_area_overlap(wake_velocities, freestream_velocities, y_ngrid, z_ngrid):
    """
    compute wake overlap based on the number of points that are not freestream velocity, i.e. affected by the wake
    """
    # Count all of the rotor points with a negligible difference from freestream
    # count = np.sum(freestream_velocities - wake_velocities <= 0.05, axis=(3, 4))
    # return (y_ngrid * z_ngrid - count) / (y_ngrid * z_ngrid)
    # return 1 - count / (y_ngrid * z_ngrid)

    # Find the points on the rotor grids with a difference from freestream of greater
    # than some tolerance. These are all the points in the wake. The ratio of
    # these points to the total points is the portion of wake overlap.
    return np.sum(freestream_velocities - wake_velocities > 0.05, axis=(3, 4)) / (y_ngrid * z_ngrid)


@define(auto_attribs=True)
class Solver:
    farm: Farm = field(converter=copy.deepcopy, validator=attrs.validators.instance_of(Farm))
    flow_field: FlowField = field(converter=copy.deepcopy, validator=attrs.validators.instance_of(Farm))
    grid: TurbineGrid | FlowFieldGrid = field(
        converter=copy.deepcopy,
        validator=attrs.validators.instance_of((FlowFieldGrid, TurbineGrid))
    )
    model_manager: WakeModelManager = field(
        converter=copy.deepcopy, validator=attrs.validators.instance_of(WakeModelManager)
    )

    @abstractmethod
    def solve(
        self,
        *,
        full_flow: bool = False,
        farm: Farm = None,
        flow_field: FlowFieldGrid = None,
        grid: TurbineGrid | FlowFieldGrid = None
    ) -> None:
        # TODO: Update with the new logger functionality that is on its way
        pass

    def full_flow_solve(self):
        """Initializes all the additional attributes to compute the full flow field, then runs the
        sequential solver with a turbine grid, then again with the `self.grid` object.

        Raises:
            TypeError: Raised if initialized `grid` value is not a `FlowFieldGrid` object.
        """
        if not isinstance(self.grid, FlowFieldGrid):
            raise TypeError("Cannot run `full_flow_solve` with a `TurbineGrid` object, reinitialize with the input to `grid` being a  `FlowFieldGrid` object.")  # noqa: E501

        # TODO: Why are we making copies of the originals, and then never using the original,
        # can we just use the originally provided values and get on with the calculating?
        farm = copy.deepcopy(self.farm)
        flow_field = copy.deepcopy(self.flow_field)

        farm.construct_turbine_map()
        farm.construct_turbine_fCts()
        farm.construct_turbine_fCps()
        farm.construct_turbine_power_interps()
        farm.construct_hub_heights()
        farm.construct_rotor_diameters()
        farm.construct_turbine_TSRs()
        farm.construc_turbine_pPs()
        farm.construc_turbine_ref_density_cp_cts()
        farm.construct_coordinates()

        turbine_grid = TurbineGrid(
            turbine_coordinates=farm.coordinates,
            reference_turbine_diameter=farm.rotor_diameters,
            wind_directions=flow_field.wind_directions,
            wind_speeds=flow_field.wind_speeds,
            grid_resolution=3,
            time_series=flow_field.time_series,
        )
        farm.expand_farm_properties(
            flow_field.n_wind_directions, flow_field.n_wind_speeds, turbine_grid.sorted_coord_indices
        )
        flow_field.initialize_velocity_field(turbine_grid)
        farm.initialize(turbine_grid.sorted_indices)
        self.solve(full_flow=False, farm=farm, flow_field=flow_field, grid=turbine_grid)
        self.solve(full_flow=True, farm=farm, flow_field=flow_field, grid=turbine_grid)


@define(auto_attribs=True)
class SequentialSolver(Solver):

    def solve(
        self,
        *,
        full_flow: bool = False,
        farm: Farm,
        flow_field: FlowFieldGrid = None,
        grid: TurbineGrid | FlowFieldGrid = None
    ) -> None:
        """Runs the sequential sover methodology, or full flow sequential solver methodology for a
        wind farm.

        Args:
            full_flow (bool, optional): Runs the full flow solver when True, and the standard
                sequential solver, when False. Defaults to False.
            farm (Farm, optional): Allows for a non-initialized `farm` object to be used. It should
                be noted that this functionality is intended for use with `full_flow_solve`.
                Defaults to None.
            flow_field (FlowField, optional): Allows for a non-initialized `flow_field` object to be
                used. It should be noted that this functionality is intended for use with
                `full_flow_solve`.Defaults to None.
            grid (TurbineGrid | FlowFieldGrid, optional): Allows for a non-initialized `grid` object
                to be used. It should be noted that this functionality is intended for use with
                `full_flow_solve`, which computes over a `TurbineGrid`, then a `FlowFieldGrid`.
                Defaults to None.
        """
        if farm is None:
            farm = self.farm
        if flow_field is None:
            flow_field = self.flow_field
        if grid is None:
            grid = self.grid

        gch_gain = 2

        deflection_model_args = self.model_manager.deflection_model.prepare_function(grid, flow_field)
        deficit_model_args = self.model_manager.velocity_model.prepare_function(grid, flow_field)

        wake_field = np.zeros_like(flow_field.u_initial_sorted)
        v_wake = np.zeros_like(flow_field.v_initial_sorted)
        w_wake = np.zeros_like(flow_field.w_initial_sorted)

        if not full_flow:
            turbine_turbulence_intensity = np.full(
                (flow_field.n_wind_directions, flow_field.n_wind_speeds, farm.n_turbines, 1, 1),
                flow_field.turbulence_intensity
            )
            ambient_turbulence_intensity = flow_field.turbulence_intensity

        # Calculate the velocity deficit sequentially from upstream to downstream turbines
        for i in range(grid.n_turbines):
            # Get the current turbine quantities
            x_i = _expansion_mean_i(grid.x_sorted, i)
            y_i = _expansion_mean_i(grid.y_sorted, i)
            z_i = _expansion_mean_i(grid.z_sorted, i)
            u_i = flow_field.u_sorted[:, :, i:i+1]
            v_i = flow_field.v_sorted[:, :, i:i+1]

            # Since we are filtering for the ith turbine in the Ct function, get the first index here (0:1)
            ct_i = Ct(
                velocities=flow_field.u_sorted,
                yaw_angle=farm.yaw_angles_sorted,
                fCt=farm.turbine_fCts,
                turbine_type_map=farm.turbine_type_map_sorted,
                ix_filter=[i],
            )[:, :, 0:1, None, None]

            # Since we are filtering for the ith turbine in the axial induction function, get the first index here (0:1)
            axial_induction_i = axial_induction(
                velocities=flow_field.u_sorted,
                yaw_angle=farm.yaw_angles_sorted,
                fCt=farm.turbine_fCts,
                turbine_type_map=farm.turbine_type_map_sorted,
                ix_filter=[i],
            )[:, :, 0:1, None, None]
            turbulence_intensity_i = flow_field.turbulence_intensity_field[:, :, i:i+1]
            yaw_angle_i = farm.yaw_angles_sorted[:, :, i:i+1, None, None]
            hub_height_i = farm.hub_heights_sorted[:, :, i:i+1, None, None]
            rotor_diameter_i = farm.rotor_diameters_sorted[:, :, i:i+1, None, None]
            TSR_i = farm.TSRs_sorted[:, :, i:i+1, None, None]

            effective_yaw_i = np.zeros_like(yaw_angle_i)
            effective_yaw_i += yaw_angle_i

            if self.model_manager.enable_secondary_steering:
                effective_yaw_i += wake_added_yaw(
                    u_i,
                    v_i,
                    flow_field.u_initial_sorted,
                    grid.y_sorted[:, :, i:i+1] - y_i,
                    grid.z_sorted[:, :, i:i+1],
                    rotor_diameter_i,
                    hub_height_i,
                    ct_i,
                    TSR_i,
                    axial_induction_i
                )

            deflection_field = self.model_manager.deflection_model.function(
                x_i,
                y_i,
                effective_yaw_i,
                turbulence_intensity_i,
                ct_i,
                rotor_diameter_i,
                **deflection_model_args
            )

            if self.model_manager.enable_transverse_velocities:
                v_wake, w_wake = calculate_transverse_velocity(
                    u_i,
                    flow_field.u_initial_sorted,
                    flow_field.dudz_initial_sorted,
                    grid.x_sorted - x_i,
                    grid.y_sorted - y_i,
                    grid.z_sorted,
                    rotor_diameter_i,
                    hub_height_i,
                    yaw_angle_i,
                    ct_i,
                    TSR_i,
                    axial_induction_i
                )
            if not full_flow:
                if self.model_manager.enable_yaw_added_recovery:
                    I_mixing = yaw_added_turbulence_mixing(
                        u_i,
                        turbulence_intensity_i,
                        v_i,
                        flow_field.w_sorted[:, :, i:i+1],
                        v_wake[:, :, i:i+1],
                        w_wake[:, :, i:i+1],
                    )
                    turbine_turbulence_intensity[:, :, i:i+1] = turbulence_intensity_i + gch_gain * I_mixing

            # NOTE: exponential
            velocity_deficit = self.model_manager.velocity_model.function(
                x_i,
                y_i,
                z_i,
                axial_induction_i,
                deflection_field,
                yaw_angle_i,
                turbulence_intensity_i,
                ct_i,
                hub_height_i,
                rotor_diameter_i,
                **deficit_model_args
            )

            wake_field = self.model_manager.combination_model.function(
                wake_field,
                velocity_deficit * flow_field.u_initial_sorted
            )

            if not full_flow:
                wake_added_turbulence_intensity = self.model_manager.turbulence_model.function(
                    ambient_turbulence_intensity,
                    grid.x_sorted,
                    x_i,
                    rotor_diameter_i,
                    axial_induction_i
                )

                # Calculate wake overlap for wake-added turbulence (WAT)
                area_overlap = (
                    np.sum(velocity_deficit * flow_field.u_initial_sorted > 0.05, axis=(3, 4))
                    / (grid.grid_resolution * grid.grid_resolution)
                )[:, :, :, None, None]

                # Modify wake added turbulence by wake area overlap
                downstream_influence_length = 15 * rotor_diameter_i
                ti_added = (
                    area_overlap
                    * np.nan_to_num(wake_added_turbulence_intensity, posinf=0.0)
                    * np.array(grid.x_sorted > x_i)
                    * np.array(np.abs(y_i - grid.y_sorted) < 2 * rotor_diameter_i)
                    * np.array(grid.x_sorted <= downstream_influence_length + x_i)
                )

                # Combine turbine TIs with WAT
                turbine_turbulence_intensity = np.maximum(
                    np.sqrt(ti_added ** 2 + ambient_turbulence_intensity ** 2),
                    turbine_turbulence_intensity
                )

            flow_field.u_sorted = flow_field.u_initial_sorted - wake_field
            flow_field.v_sorted += v_wake
            flow_field.w_sorted += w_wake

        if not full_flow:
            flow_field.turbulence_intensity_field = _expansion_mean(turbine_turbulence_intensity)


@define(auto_attribs=True)
class CCSolver(Solver):

    def solve(
        self,
        *,
        full_flow: bool = False,
        farm: Farm = None,
        flow_field: FlowField = None,
        grid: TurbineGrid | FlowFieldGrid = None
    ) -> None:
        """Runs the CC sover methodology, or full flow CC solver methodology for a wind farm.

        Args:
            full_flow (bool, optional): Runs the full flow solver when True, and the standard CC
                solver, when False. Defaults to False.
            farm (Farm, optional): Allows for a non-initialized `farm` object to be used. It should
                be noted that this functionality is intended for use with `full_flow_solve`.
                Defaults to None.
            flow_field (FlowField, optional): Allows for a non-initialized `flow_field` object to be
                used. It should be noted that this functionality is intended for use with
                `full_flow_solve`.Defaults to None.
            grid (TurbineGrid | FlowFieldGrid, optional): Allows for a non-initialized `grid` object
                to be used. It should be noted that this functionality is intended for use with
                `full_flow_solve`, which computes over a `TurbineGrid`, then a `FlowFieldGrid`.
                Defaults to None.
        """
        if farm is None:
            farm = self.farm
        if flow_field is None:
            flow_field = self.flow_field
        if grid is None:
            grid = self.grid

        gch_gain = 1.0
        scale_factor = 2.0

        # <<interface>>
        deflection_model_args = self.model_manager.deflection_model.prepare_function(grid, flow_field)
        deficit_model_args = self.model_manager.velocity_model.prepare_function(grid, flow_field)

        # This is u_wake
        v_wake = np.zeros_like(flow_field.v_initial_sorted)
        w_wake = np.zeros_like(flow_field.w_initial_sorted)
        turb_u_wake = np.zeros_like(flow_field.u_initial_sorted)

        # Not needed for full flow solve, but isn't necessary to have in an if statement
        turbine_inflow_field = copy.deepcopy(flow_field.u_initial_sorted)
        turbine_turbulence_intensity = np.full(
            (flow_field.n_wind_directions, flow_field.n_wind_speeds, farm.n_turbines, 1, 1),
            flow_field.turbulence_intensity
        )
        ambient_turbulence_intensity = flow_field.turbulence_intensity

        Ctmp = np.zeros((farm.n_turbines,) + np.shape(flow_field.u_initial_sorted))

        # Calculate the velocity deficit sequentially from upstream to downstream turbines
        for i in range(grid.n_turbines):

            # Get the current turbine quantities
            x_i = _expansion_mean_i(grid.x_sorted, i)
            y_i = _expansion_mean_i(grid.y_sorted, i)
            z_i = _expansion_mean_i(grid.z_sorted, i)
            u_i = turbine_inflow_field[:, :, i:i+1]
            v_i = flow_field.v_sorted[:, :, i:i+1]

            if not full_flow:
                mask = (
                    (grid.x_sorted < x_i + 0.01)
                    * (grid.x_sorted > x_i - 0.01)
                    * (grid.y_sorted < y_i + 0.51 * 126.0)
                    * (grid.y_sorted > y_i - 0.51 * 126.0)
                )
                turbine_inflow_field *= ~mask + (flow_field.u_initial_sorted - turb_u_wake) * mask

            turbine_inflow_field = flow_field.u_sorted if full_flow else turbine_inflow_field

            turb_avg_vels = average_velocity(turbine_inflow_field)
            turb_Cts = Ct(
                turb_avg_vels,
                farm.yaw_angles_sorted,
                farm.turbine_fCts,
                turbine_type_map=farm.turbine_type_map_sorted,
            )[:, :, :, None, None]

            if not full_flow:
                turb_aIs = axial_induction(
                    turb_avg_vels,
                    farm.yaw_angles_sorted,
                    farm.turbine_fCts,
                    turbine_type_map=farm.turbine_type_map_sorted,
                    ix_filter=[i],
                )[:, :, :, None, None]

            axial_induction_i = axial_induction(
                velocities=flow_field.u_sorted,
                yaw_angle=farm.yaw_angles_sorted,
                fCt=farm.turbine_fCts,
                turbine_type_map=farm.turbine_type_map_sorted,
                ix_filter=[i],
            )[:, :, :, None, None]

            turbulence_intensity_i = turbine_turbulence_intensity[:, :, i:i+1]
            yaw_angle_i = farm.yaw_angles_sorted[:, :, i:i+1, None, None]
            hub_height_i = farm.hub_heights_sorted[:, :, i:i+1, None, None]
            rotor_diameter_i = farm.rotor_diameters_sorted[:, :, i:i+1, None, None]
            TSR_i = farm.TSRs_sorted[:, :, i:i+1, None, None]
            effective_yaw_i = yaw_angle_i.copy()

            if self.model_manager.enable_secondary_steering:
                effective_yaw_i += wake_added_yaw(
                    u_i,
                    v_i,
                    flow_field.u_initial_sorted,
                    grid.y_sorted[:, :, i:i+1] - y_i,
                    grid.z_sorted[:, :, i:i+1],
                    rotor_diameter_i,
                    hub_height_i,
                    turb_Cts[:, :, i:i+1],
                    TSR_i,
                    axial_induction_i,
                    scale=scale_factor,
                )

            # Model calculations
            # NOTE: exponential
            deflection_field = self.model_manager.deflection_model.function(
                x_i,
                y_i,
                effective_yaw_i,
                turbulence_intensity_i,
                turb_Cts[:, :, i:i+1],
                rotor_diameter_i,
                **deflection_model_args
            )

            if self.model_manager.enable_transverse_velocities:
                v_wake, w_wake = calculate_transverse_velocity(
                    u_i,
                    flow_field.u_initial_sorted,
                    flow_field.dudz_initial_sorted,
                    grid.x_sorted - x_i,
                    grid.y_sorted - y_i,
                    grid.z_sorted,
                    rotor_diameter_i,
                    hub_height_i,
                    yaw_angle_i,
                    turb_Cts[:, :, i:i+1],
                    TSR_i,
                    axial_induction_i,
                    scale=scale_factor,
                )

            if full_flow:
                turbine_turbulence_intensity = flow_field.turbulence_intensity_field
            else:
                if self.model_manager.enable_yaw_added_recovery:
                    I_mixing = yaw_added_turbulence_mixing(
                        u_i,
                        turbulence_intensity_i,
                        v_i,
                        flow_field.w_sorted[:, :, i:i+1],
                        v_wake[:, :, i:i+1],
                        w_wake[:, :, i:i+1],
                    )
                    turbine_turbulence_intensity[:, :, i:i+1] = turbulence_intensity_i + gch_gain * I_mixing

            # NOTE: exponential
            turb_u_wake, Ctmp = self.model_manager.velocity_model.function(
                i,
                x_i,
                y_i,
                z_i,
                u_i,
                deflection_field,
                yaw_angle_i,
                turbine_turbulence_intensity,
                turb_Cts,
                farm.rotor_diameters_sorted[:, :, :, None, None],
                turb_u_wake,
                Ctmp,
                **deficit_model_args
            )

            if not full_flow:
                wake_added_turbulence_intensity = self.model_manager.turbulence_model.function(
                    ambient_turbulence_intensity,
                    grid.x_sorted,
                    x_i,
                    rotor_diameter_i,
                    turb_aIs
                )

                # Calculate wake overlap for wake-added turbulence (WAT)
                area_overlap = (
                    1
                    - np.sum(turb_u_wake <= 0.05, axis=(3, 4))
                    / (grid.grid_resolution * grid.grid_resolution)
                )[:, :, :, None, None]

                # Modify wake added turbulence by wake area overlap
                downstream_influence_length = 15 * rotor_diameter_i
                ti_added = (
                    area_overlap
                    * np.nan_to_num(wake_added_turbulence_intensity, posinf=0.0)
                    * (grid.x_sorted > x_i)
                    * (np.abs(y_i - grid.y_sorted) < 2 * rotor_diameter_i)
                    * (grid.x_sorted <= downstream_influence_length + x_i)
                )

                # Combine turbine TIs with WAT
                turbine_turbulence_intensity = np.maximum(
                    np.sqrt(ti_added ** 2 + ambient_turbulence_intensity ** 2),
                    turbine_turbulence_intensity
                )

            flow_field.v_sorted += v_wake
            flow_field.w_sorted += w_wake

        flow_field.u_sorted = flow_field.u_initial_sorted - turb_u_wake if full_flow else turbine_inflow_field

        if not full_flow:
            flow_field.turbulence_intensity_field = _expansion_mean(turbine_turbulence_intensity)


@define(auto_attribs=True)
class TurbOParkSolver(Solver):

    def full_flow_solve(self):
        # TODO: Update with the new logger functionality that is on its way
        raise NotImplementedError("TurbOParkSolver has no full flow field solving capabilities yet.")

    def solve(
        self,
        *,
        full_flow: bool = False,
        farm: Farm = None,
        flow_field: FlowField = None,
        grid: TurbineGrid | FlowFieldGrid = None
    ) -> None:
        """Runs the TurbOPark sover methodology, or full flow TurbOPark solver methodology for a
        wind farm.

        Args:
            full_flow (bool, optional): Runs the full flow solver when True, and the standard
                TurbOPark solver, when False. Defaults to False.
            farm (Farm, optional): Allows for a non-initialized `farm` object to be used. It should
                be noted that this functionality is intended for use with `full_flow_solve`.
                Defaults to None.
            flow_field (FlowField, optional): Allows for a non-initialized `flow_field` object to be
                used. It should be noted that this functionality is intended for use with
                `full_flow_solve`.Defaults to None.
            grid (TurbineGrid | FlowFieldGrid, optional): Allows for a non-initialized `grid` object
                to be used. It should be noted that this functionality is intended for use with
                `full_flow_solve`, which computes over a `TurbineGrid`, then a `FlowFieldGrid`.
                Defaults to None.
        """
        # Algorithm
        # For each turbine, calculate its effect on every downstream turbine.
        # For the current turbine, we are calculating the deficit that it adds to downstream turbines.
        # Integrate this into the main data structure.
        # Move on to the next turbine.

        if farm is None:
            farm = self.farm
        if flow_field is None:
            flow_field = self.flow_field
        if grid is None:
            grid = self.grid

        gch_gain = 2

        # <<interface>>
        deflection_model_args = self.model_manager.deflection_model.prepare_function(grid, flow_field)
        deficit_model_args = self.model_manager.velocity_model.prepare_function(grid, flow_field)

        # This is u_wake
        wake_field = np.zeros_like(flow_field.u_initial_sorted)
        v_wake = np.zeros_like(flow_field.v_initial_sorted)
        w_wake = np.zeros_like(flow_field.w_initial_sorted)
        shape = (farm.n_turbines,) + np.shape(flow_field.u_initial_sorted)
        velocity_deficit = np.zeros(shape)
        deflection_field = np.zeros_like(flow_field.u_initial_sorted)

        turbine_turbulence_intensity = np.full(
            (flow_field.n_wind_directions, flow_field.n_wind_speeds, farm.n_turbines, 1, 1),
            flow_field.turbulence_intensity
        )
        ambient_turbulence_intensity = flow_field.turbulence_intensity

        # Calculate the velocity deficit sequentially from upstream to downstream turbines
        for i in range(grid.n_turbines):

            # Get the current turbine quantities
            x_i = _expansion_mean_i(grid.x_sorted, i)
            y_i = _expansion_mean_i(grid.y_sorted, i)
            z_i = _expansion_mean_i(grid.z_sorted, i)
            u_i = flow_field.u_sorted[:, :, i:i+1]
            v_i = flow_field.v_sorted[:, :, i:i+1]

            Cts = Ct(
                velocities=flow_field.u_sorted,
                yaw_angle=farm.yaw_angles_sorted,
                fCt=farm.turbine_fCts,
                turbine_type_map=farm.turbine_type_map_sorted,
            )

            # Since we are filtering for the ith turbine in the Ct function, get the first index here (0:1)
            ct_i = Ct(
                velocities=flow_field.u_sorted,
                yaw_angle=farm.yaw_angles_sorted,
                fCt=farm.turbine_fCts,
                turbine_type_map=farm.turbine_type_map_sorted,
                ix_filter=[i],
            )[:, :, 0:1, None, None]

            # Since we are filtering for the ith turbine in the axial induction function, get the first index here (0:1)
            axial_induction_i = axial_induction(
                velocities=flow_field.u_sorted,
                yaw_angle=farm.yaw_angles_sorted,
                fCt=farm.turbine_fCts,
                turbine_type_map=farm.turbine_type_map_sorted,
                ix_filter=[i],
            )[:, :, 0:1, None, None]

            turbulence_intensity_i = turbine_turbulence_intensity[:, :, i:i+1]
            yaw_angle_i = farm.yaw_angles_sorted[:, :, i:i+1, None, None]
            hub_height_i = farm.hub_heights_sorted[:, :, i:i+1, None, None]
            rotor_diameter_i = farm.rotor_diameters_sorted[:, :, i:i+1, None, None]
            TSR_i = farm.TSRs_sorted[:, :, i:i+1, None, None]

            effective_yaw_i = np.zeros_like(yaw_angle_i)
            effective_yaw_i += yaw_angle_i

            if self.model_manager.enable_secondary_steering:
                effective_yaw_i += wake_added_yaw(
                    u_i,
                    v_i,
                    flow_field.u_initial_sorted,
                    grid.y_sorted[:, :, i:i+1] - y_i,
                    grid.z_sorted[:, :, i:i+1],
                    rotor_diameter_i,
                    hub_height_i,
                    ct_i,
                    TSR_i,
                    axial_induction_i
                )

            # Model calculations
            # NOTE: exponential
            if not np.all(farm.yaw_angles_sorted):
                self.model_manager.deflection_model.logger.warning("WARNING: Deflection with the TurbOPark model has not been fully validated. This is an initial implementation, and we advise you use at your own risk and perform a thorough examination of the results.")  # noqa: #501
                for ii in range(i):
                    x_ii = _expansion_mean_i(grid.x_sorted, ii)
                    y_ii = _expansion_mean_i(grid.y_sorted, ii)

                    yaw_ii = farm.yaw_angles_sorted[:, :, ii:ii+1, None, None]
                    turbulence_intensity_ii = turbine_turbulence_intensity[:, :, ii:ii+1]
                    ct_ii = Ct(
                        velocities=flow_field.u_sorted,
                        yaw_angle=farm.yaw_angles_sorted,
                        fCt=farm.turbine_fCts,
                        turbine_type_map=farm.turbine_type_map_sorted,
                        ix_filter=[ii]
                    )[:, :, 0:1, None, None]
                    rotor_diameter_ii = farm.rotor_diameters_sorted[:, :, ii:ii+1, None, None]

                    deflection_field_ii = self.model_manager.deflection_model.function(
                        x_ii,
                        y_ii,
                        yaw_ii,
                        turbulence_intensity_ii,
                        ct_ii,
                        rotor_diameter_ii,
                        **deflection_model_args
                    )

                    deflection_field[:, :, ii:ii+1, :, :] = deflection_field_ii[:, :, i:i+1, :, :]

            if self.model_manager.enable_transverse_velocities:
                v_wake, w_wake = calculate_transverse_velocity(
                    u_i,
                    flow_field.u_initial_sorted,
                    flow_field.dudz_initial_sorted,
                    grid.x_sorted - x_i,
                    grid.y_sorted - y_i,
                    grid.z_sorted,
                    rotor_diameter_i,
                    hub_height_i,
                    yaw_angle_i,
                    ct_i,
                    TSR_i,
                    axial_induction_i
                )

            if self.model_manager.enable_yaw_added_recovery:
                I_mixing = yaw_added_turbulence_mixing(
                    u_i,
                    turbulence_intensity_i,
                    v_i,
                    flow_field.w_sorted[:, :, i:i+1],
                    v_wake[:, :, i:i+1],
                    w_wake[:, :, i:i+1],
                )
                turbine_turbulence_intensity[:, :, i:i+1] = turbulence_intensity_i + gch_gain * I_mixing

            # NOTE: exponential
            velocity_deficit = self.model_manager.velocity_model.function(
                x_i,
                y_i,
                z_i,
                turbine_turbulence_intensity,
                Cts[:, :, :, None, None],
                rotor_diameter_i,
                farm.rotor_diameters_sorted[:, :, :, None, None],
                i,
                deflection_field,
                **deficit_model_args
            )

            wake_field = self.model_manager.combination_model.function(
                wake_field,
                velocity_deficit * flow_field.u_initial_sorted
            )

            wake_added_turbulence_intensity = self.model_manager.turbulence_model.function(
                ambient_turbulence_intensity,
                grid.x_sorted,
                x_i,
                rotor_diameter_i,
                axial_induction_i
            )

            # TODO: leaving this in for GCH quantities; will need to find another way to compute area_overlap
            # as the current wake deficit is solved for only upstream turbines; could use WAT_upstream
            # Calculate wake overlap for wake-added turbulence (WAT)
            area_overlap = (
                np.sum(velocity_deficit * flow_field.u_initial_sorted > 0.05, axis=(3, 4))
                / (grid.grid_resolution ** 2)
            )[:, :, :, None, None]

            # Modify wake added turbulence by wake area overlap
            downstream_influence_length = 15 * rotor_diameter_i
            ti_added = (
                area_overlap
                * np.nan_to_num(wake_added_turbulence_intensity, posinf=0.0)
                * (grid.x_sorted > x_i)
                * (np.abs(y_i - grid.y_sorted) < 2 * rotor_diameter_i)
                * (grid.x_sorted <= downstream_influence_length + x_i)
            )

            # Combine turbine TIs with WAT
            turbine_turbulence_intensity = np.maximum(
                np.sqrt(ti_added ** 2 + ambient_turbulence_intensity ** 2),
                turbine_turbulence_intensity
            )

            flow_field.u_sorted = flow_field.u_initial_sorted - wake_field
            flow_field.v_sorted += v_wake
            flow_field.w_sorted += w_wake

        flow_field.turbulence_intensity_field = _expansion_mean(turbine_turbulence_intensity)

# flake8: noqa
def sequential_solver(farm: Farm, flow_field: FlowField, grid: TurbineGrid, model_manager: WakeModelManager) -> None:
    # Algorithm
    # For each turbine, calculate its effect on every downstream turbine.
    # For the current turbine, we are calculating the deficit that it adds to downstream turbines.
    # Integrate this into the main data structure.
    # Move on to the next turbine.

    # <<interface>>
    deflection_model_args = model_manager.deflection_model.prepare_function(grid, flow_field)
    deficit_model_args = model_manager.velocity_model.prepare_function(grid, flow_field)

    # This is u_wake
    wake_field = np.zeros_like(flow_field.u_initial_sorted)
    v_wake = np.zeros_like(flow_field.v_initial_sorted)
    w_wake = np.zeros_like(flow_field.w_initial_sorted)

    turbine_turbulence_intensity = flow_field.turbulence_intensity * np.ones((flow_field.n_wind_directions, flow_field.n_wind_speeds, farm.n_turbines, 1, 1))
    ambient_turbulence_intensity = flow_field.turbulence_intensity

    # Calculate the velocity deficit sequentially from upstream to downstream turbines
    for i in range(grid.n_turbines):

        # Get the current turbine quantities
        x_i = np.mean(grid.x_sorted[:, :, i:i+1], axis=(3, 4))
        x_i = x_i[:, :, :, None, None]
        y_i = np.mean(grid.y_sorted[:, :, i:i+1], axis=(3, 4))
        y_i = y_i[:, :, :, None, None]
        z_i = np.mean(grid.z_sorted[:, :, i:i+1], axis=(3, 4))
        z_i = z_i[:, :, :, None, None]

        u_i = flow_field.u_sorted[:, :, i:i+1]
        v_i = flow_field.v_sorted[:, :, i:i+1]

        ct_i = Ct(
            velocities=flow_field.u_sorted,
            yaw_angle=farm.yaw_angles_sorted,
            fCt=farm.turbine_fCts,
            turbine_type_map=farm.turbine_type_map_sorted,
            ix_filter=[i],
        )
        ct_i = ct_i[:, :, 0:1, None, None]  # Since we are filtering for the i'th turbine in the Ct function, get the first index here (0:1)
        axial_induction_i = axial_induction(
            velocities=flow_field.u_sorted,
            yaw_angle=farm.yaw_angles_sorted,
            fCt=farm.turbine_fCts,
            turbine_type_map=farm.turbine_type_map_sorted,
            ix_filter=[i],
        )
        axial_induction_i = axial_induction_i[:, :, 0:1, None, None]    # Since we are filtering for the i'th turbine in the axial induction function, get the first index here (0:1)
        turbulence_intensity_i = turbine_turbulence_intensity[:, :, i:i+1]
        yaw_angle_i = farm.yaw_angles_sorted[:, :, i:i+1, None, None]
        hub_height_i = farm.hub_heights_sorted[:, :, i:i+1, None, None]
        rotor_diameter_i = farm.rotor_diameters_sorted[:, :, i:i+1, None, None]
        TSR_i = farm.TSRs_sorted[:, :, i:i+1, None, None]

        effective_yaw_i = np.zeros_like(yaw_angle_i)
        effective_yaw_i += yaw_angle_i

        if model_manager.enable_secondary_steering:
            added_yaw = wake_added_yaw(
                u_i,
                v_i,
                flow_field.u_initial_sorted,
                grid.y_sorted[:, :, i:i+1] - y_i,
                grid.z_sorted[:, :, i:i+1],
                rotor_diameter_i,
                hub_height_i,
                ct_i,
                TSR_i,
                axial_induction_i
            )
            effective_yaw_i += added_yaw

        # Model calculations
        # NOTE: exponential
        deflection_field = model_manager.deflection_model.function(
            x_i,
            y_i,
            effective_yaw_i,
            turbulence_intensity_i,
            ct_i,
            rotor_diameter_i,
            **deflection_model_args
        )

        if model_manager.enable_transverse_velocities:
            v_wake, w_wake = calculate_transverse_velocity(
                u_i,
                flow_field.u_initial_sorted,
                flow_field.dudz_initial_sorted,
                grid.x_sorted - x_i,
                grid.y_sorted - y_i,
                grid.z_sorted,
                rotor_diameter_i,
                hub_height_i,
                yaw_angle_i,
                ct_i,
                TSR_i,
                axial_induction_i
            )

        if model_manager.enable_yaw_added_recovery:
            I_mixing = yaw_added_turbulence_mixing(
                u_i,
                turbulence_intensity_i,
                v_i,
                flow_field.w_sorted[:, :, i:i+1],
                v_wake[:, :, i:i+1],
                w_wake[:, :, i:i+1],
            )
            gch_gain = 2
            turbine_turbulence_intensity[:, :, i:i+1] = turbulence_intensity_i + gch_gain * I_mixing

        # NOTE: exponential
        velocity_deficit = model_manager.velocity_model.function(
            x_i,
            y_i,
            z_i,
            axial_induction_i,
            deflection_field,
            yaw_angle_i,
            turbulence_intensity_i,
            ct_i,
            hub_height_i,
            rotor_diameter_i,
            **deficit_model_args
        )

        wake_field = model_manager.combination_model.function(
            wake_field,
            velocity_deficit * flow_field.u_initial_sorted
        )

        wake_added_turbulence_intensity = model_manager.turbulence_model.function(
            ambient_turbulence_intensity,
            grid.x_sorted,
            x_i,
            rotor_diameter_i,
            axial_induction_i
        )

        # Calculate wake overlap for wake-added turbulence (WAT)
        area_overlap = np.sum(velocity_deficit * flow_field.u_initial_sorted > 0.05, axis=(3, 4)) / (grid.grid_resolution * grid.grid_resolution)
        area_overlap = area_overlap[:, :, :, None, None]

        # Modify wake added turbulence by wake area overlap
        downstream_influence_length = 15 * rotor_diameter_i
        ti_added = (
            area_overlap
            * np.nan_to_num(wake_added_turbulence_intensity, posinf=0.0)
            * np.array(grid.x_sorted > x_i)
            * np.array(np.abs(y_i - grid.y_sorted) < 2 * rotor_diameter_i)
            * np.array(grid.x_sorted <= downstream_influence_length + x_i)
        )

        # Combine turbine TIs with WAT
        turbine_turbulence_intensity = np.maximum( np.sqrt( ti_added ** 2 + ambient_turbulence_intensity ** 2 ) , turbine_turbulence_intensity )

        flow_field.u_sorted = flow_field.u_initial_sorted - wake_field
        flow_field.v_sorted += v_wake
        flow_field.w_sorted += w_wake

    flow_field.turbulence_intensity_field = np.mean(turbine_turbulence_intensity, axis=(3,4))
    flow_field.turbulence_intensity_field = flow_field.turbulence_intensity_field[:,:,:,None,None]


def full_flow_sequential_solver(farm: Farm, flow_field: FlowField, flow_field_grid: FlowFieldGrid, model_manager: WakeModelManager) -> None:

    # Get the flow quantities and turbine performance
    turbine_grid_farm = copy.deepcopy(farm)
    turbine_grid_flow_field = copy.deepcopy(flow_field)

    turbine_grid_farm.construct_turbine_map()
    turbine_grid_farm.construct_turbine_fCts()
    turbine_grid_farm.construct_turbine_fCps()
    turbine_grid_farm.construct_turbine_power_interps()
    turbine_grid_farm.construct_hub_heights()
    turbine_grid_farm.construct_rotor_diameters()
    turbine_grid_farm.construct_turbine_TSRs()
    turbine_grid_farm.construc_turbine_pPs()
    turbine_grid_farm.construc_turbine_ref_density_cp_cts()
    turbine_grid_farm.construct_coordinates()


    turbine_grid = TurbineGrid(
        turbine_coordinates=turbine_grid_farm.coordinates,
        reference_turbine_diameter=turbine_grid_farm.rotor_diameters,
        wind_directions=turbine_grid_flow_field.wind_directions,
        wind_speeds=turbine_grid_flow_field.wind_speeds,
        grid_resolution=3,
        time_series=turbine_grid_flow_field.time_series,
    )
    turbine_grid_farm.expand_farm_properties(
        turbine_grid_flow_field.n_wind_directions, turbine_grid_flow_field.n_wind_speeds, turbine_grid.sorted_coord_indices
    )
    turbine_grid_flow_field.initialize_velocity_field(turbine_grid)
    turbine_grid_farm.initialize(turbine_grid.sorted_indices)
    sequential_solver(turbine_grid_farm, turbine_grid_flow_field, turbine_grid, model_manager)

    ### Referring to the quantities from above, calculate the wake in the full grid

    # Use full flow_field here to use the full grid in the wake models
    deflection_model_args = model_manager.deflection_model.prepare_function(flow_field_grid, flow_field)
    deficit_model_args = model_manager.velocity_model.prepare_function(flow_field_grid, flow_field)

    wake_field = np.zeros_like(flow_field.u_initial_sorted)
    v_wake = np.zeros_like(flow_field.v_initial_sorted)
    w_wake = np.zeros_like(flow_field.w_initial_sorted)

    # Calculate the velocity deficit sequentially from upstream to downstream turbines
    for i in range(flow_field_grid.n_turbines):

        # Get the current turbine quantities
        x_i = np.mean(grid.x_sorted[:, :, i:i+1], axis=(3, 4))
        x_i = x_i[:, :, :, None, None]
        y_i = np.mean(grid.y_sorted[:, :, i:i+1], axis=(3, 4))
        y_i = y_i[:, :, :, None, None]
        z_i = np.mean(grid.z_sorted[:, :, i:i+1], axis=(3, 4))
        z_i = z_i[:, :, :, None, None]

        u_i = turbine_grid_flow_field.u_sorted[:, :, i:i+1]
        v_i = turbine_grid_flow_field.v_sorted[:, :, i:i+1]

        ct_i = Ct(
            velocities=turbine_grid_flow_field.u_sorted,
            yaw_angle=turbine_grid_farm.yaw_angles_sorted,
            fCt=turbine_grid_farm.turbine_fCts,
            turbine_type_map=turbine_grid_farm.turbine_type_map_sorted,
            ix_filter=[i],
        )
        ct_i = ct_i[:, :, 0:1, None, None]  # Since we are filtering for the i'th turbine in the Ct function, get the first index here (0:1)
        axial_induction_i = axial_induction(
            velocities=turbine_grid_flow_field.u_sorted,
            yaw_angle=turbine_grid_farm.yaw_angles_sorted,
            fCt=turbine_grid_farm.turbine_fCts,
            turbine_type_map=turbine_grid_farm.turbine_type_map_sorted,
            ix_filter=[i],
        )
        axial_induction_i = axial_induction_i[:, :, 0:1, None, None]    # Since we are filtering for the i'th turbine in the axial induction function, get the first index here (0:1)
        turbulence_intensity_i = turbine_grid_flow_field.turbulence_intensity_field[:, :, i:i+1]
        yaw_angle_i = turbine_grid_farm.yaw_angles_sorted[:, :, i:i+1, None, None]
        hub_height_i = turbine_grid_farm.hub_heights_sorted[:, :, i:i+1, None, None]
        rotor_diameter_i = turbine_grid_farm.rotor_diameters_sorted[:, :, i:i+1, None, None]
        TSR_i = turbine_grid_farm.TSRs_sorted[:, :, i:i+1, None, None]

        effective_yaw_i = np.zeros_like(yaw_angle_i)
        effective_yaw_i += yaw_angle_i

        if model_manager.enable_secondary_steering:
            added_yaw = wake_added_yaw(
                u_i,
                v_i,
                turbine_grid_flow_field.u_initial_sorted,
                turbine_grid.y_sorted[:, :, i:i+1] - y_i,
                turbine_grid.z_sorted[:, :, i:i+1],
                rotor_diameter_i,
                hub_height_i,
                ct_i,
                TSR_i,
                axial_induction_i
            )
            effective_yaw_i += added_yaw

        # Model calculations
        # NOTE: exponential
        deflection_field = model_manager.deflection_model.function(
            x_i,
            y_i,
            effective_yaw_i,
            turbulence_intensity_i,
            ct_i,
            rotor_diameter_i,
            **deflection_model_args
        )

        if model_manager.enable_transverse_velocities:
            v_wake, w_wake = calculate_transverse_velocity(
                u_i,
                flow_field.u_initial_sorted,
                flow_field.dudz_initial_sorted,
                flow_field_grid.x_sorted - x_i,
                flow_field_grid.y_sorted - y_i,
                flow_field_grid.z_sorted,
                rotor_diameter_i,
                hub_height_i,
                yaw_angle_i,
                ct_i,
                TSR_i,
                axial_induction_i
            )

        # NOTE: exponential
        velocity_deficit = model_manager.velocity_model.function(
            x_i,
            y_i,
            z_i,
            axial_induction_i,
            deflection_field,
            yaw_angle_i,
            turbulence_intensity_i,
            ct_i,
            hub_height_i,
            rotor_diameter_i,
            **deficit_model_args
        )

        wake_field = model_manager.combination_model.function(
            wake_field,
            velocity_deficit * flow_field.u_initial_sorted
        )

        flow_field.u_sorted = flow_field.u_initial_sorted - wake_field
        flow_field.v_sorted += v_wake
        flow_field.w_sorted += w_wake


def cc_solver(farm: Farm, flow_field: FlowField, grid: TurbineGrid, model_manager: WakeModelManager) -> None:

    # <<interface>>
    deflection_model_args = model_manager.deflection_model.prepare_function(grid, flow_field)
    deficit_model_args = model_manager.velocity_model.prepare_function(grid, flow_field)

    # This is u_wake
    v_wake = np.zeros_like(flow_field.v_initial_sorted)
    w_wake = np.zeros_like(flow_field.w_initial_sorted)
    turb_u_wake = np.zeros_like(flow_field.u_initial_sorted)
    turb_inflow_field = copy.deepcopy(flow_field.u_initial_sorted)

    turbine_turbulence_intensity = flow_field.turbulence_intensity * np.ones((flow_field.n_wind_directions, flow_field.n_wind_speeds, farm.n_turbines, 1, 1))
    ambient_turbulence_intensity = flow_field.turbulence_intensity

    shape = (farm.n_turbines,) + np.shape(flow_field.u_initial_sorted)
    Ctmp = np.zeros((shape))
    # Ctmp = np.zeros((len(x_coord), len(wd), len(ws), len(x_coord), y_ngrid, z_ngrid))

    sigma_i = np.zeros((shape))
    # sigma_i = np.zeros((len(x_coord), len(wd), len(ws), len(x_coord), y_ngrid, z_ngrid))

    # Calculate the velocity deficit sequentially from upstream to downstream turbines
    for i in range(grid.n_turbines):

        # Get the current turbine quantities
        x_i = np.mean(grid.x_sorted[:, :, i:i+1], axis=(3, 4))
        x_i = x_i[:, :, :, None, None]
        y_i = np.mean(grid.y_sorted[:, :, i:i+1], axis=(3, 4))
        y_i = y_i[:, :, :, None, None]
        z_i = np.mean(grid.z_sorted[:, :, i:i+1], axis=(3, 4))
        z_i = z_i[:, :, :, None, None]

        mask2 = np.array(grid.x_sorted < x_i + 0.01) * np.array(grid.x_sorted > x_i - 0.01) * np.array(grid.y_sorted < y_i + 0.51*126.0) * np.array(grid.y_sorted > y_i - 0.51*126.0)
        # mask2 = np.logical_and(np.logical_and(np.logical_and(grid.x_sorted < x_i + 0.01, grid.x_sorted > x_i - 0.01), grid.y_sorted < y_i + 0.51*126.0), grid.y_sorted > y_i - 0.51*126.0)
        turb_inflow_field = turb_inflow_field * ~mask2 + (flow_field.u_initial_sorted - turb_u_wake) * mask2

        turb_avg_vels = average_velocity(turb_inflow_field)
        turb_Cts = Ct(
            turb_avg_vels,
            farm.yaw_angles_sorted,
            farm.turbine_fCts,
            turbine_type_map=farm.turbine_type_map_sorted,
        )
        turb_Cts = turb_Cts[:, :, :, None, None]
        turb_aIs = axial_induction(
            turb_avg_vels,
            farm.yaw_angles_sorted,
            farm.turbine_fCts,
            turbine_type_map=farm.turbine_type_map_sorted,
            ix_filter=[i],
        )
        turb_aIs = turb_aIs[:, :, :, None, None]

        u_i = turb_inflow_field[:, :, i:i+1]
        v_i = flow_field.v_sorted[:, :, i:i+1]

        axial_induction_i = axial_induction(
            velocities=flow_field.u_sorted,
            yaw_angle=farm.yaw_angles_sorted,
            fCt=farm.turbine_fCts,
            turbine_type_map=farm.turbine_type_map_sorted,
            ix_filter=[i],
        )

        axial_induction_i = axial_induction_i[:, :, :, None, None]

        turbulence_intensity_i = turbine_turbulence_intensity[:, :, i:i+1]
        yaw_angle_i = farm.yaw_angles_sorted[:, :, i:i+1, None, None]
        hub_height_i = farm.hub_heights_sorted[:, :, i:i+1, None, None]
        rotor_diameter_i = farm.rotor_diameters_sorted[:, :, i:i+1, None, None]
        TSR_i = farm.TSRs_sorted[:, :, i:i+1, None, None]

        effective_yaw_i = np.zeros_like(yaw_angle_i)
        effective_yaw_i += yaw_angle_i

        if model_manager.enable_secondary_steering:
            added_yaw = wake_added_yaw(
                u_i,
                v_i,
                flow_field.u_initial_sorted,
                grid.y_sorted[:, :, i:i+1] - y_i,
                grid.z_sorted[:, :, i:i+1],
                rotor_diameter_i,
                hub_height_i,
                turb_Cts[:, :, i:i+1],
                TSR_i,
                axial_induction_i,
                scale=2.0,
            )
            effective_yaw_i += added_yaw

        # Model calculations
        # NOTE: exponential
        deflection_field = model_manager.deflection_model.function(
            x_i,
            y_i,
            effective_yaw_i,
            turbulence_intensity_i,
            turb_Cts[:, :, i:i+1],
            rotor_diameter_i,
            **deflection_model_args
        )

        if model_manager.enable_transverse_velocities:
            v_wake, w_wake = calculate_transverse_velocity(
                u_i,
                flow_field.u_initial_sorted,
                flow_field.dudz_initial_sorted,
                grid.x_sorted - x_i,
                grid.y_sorted - y_i,
                grid.z_sorted,
                rotor_diameter_i,
                hub_height_i,
                yaw_angle_i,
                turb_Cts[:, :, i:i+1],
                TSR_i,
                axial_induction_i,
                scale=2.0
            )

        if model_manager.enable_yaw_added_recovery:
            I_mixing = yaw_added_turbulence_mixing(
                u_i,
                turbulence_intensity_i,
                v_i,
                flow_field.w_sorted[:, :, i:i+1],
                v_wake[:, :, i:i+1],
                w_wake[:, :, i:i+1],
            )
            gch_gain = 1.0
            turbine_turbulence_intensity[:, :, i:i+1] = turbulence_intensity_i + gch_gain * I_mixing

        turb_u_wake, Ctmp = model_manager.velocity_model.function(
            i,
            x_i,
            y_i,
            z_i,
            u_i,
            deflection_field,
            yaw_angle_i,
            turbine_turbulence_intensity,
            turb_Cts,
            farm.rotor_diameters_sorted[:, :, :, None, None],
            turb_u_wake,
            Ctmp,
            **deficit_model_args
        )

        wake_added_turbulence_intensity = model_manager.turbulence_model.function(
            ambient_turbulence_intensity,
            grid.x_sorted,
            x_i,
            rotor_diameter_i,
            turb_aIs
        )

        # Calculate wake overlap for wake-added turbulence (WAT)
        area_overlap = 1 - np.sum(turb_u_wake <= 0.05, axis=(3, 4)) / (grid.grid_resolution * grid.grid_resolution)
        area_overlap = area_overlap[:, :, :, None, None]

        # Modify wake added turbulence by wake area overlap
        downstream_influence_length = 15 * rotor_diameter_i
        ti_added = (
            area_overlap
            * np.nan_to_num(wake_added_turbulence_intensity, posinf=0.0)
            * np.array(grid.x_sorted > x_i)
            * np.array(np.abs(y_i - grid.y_sorted) < 2 * rotor_diameter_i)
            * np.array(grid.x_sorted <= downstream_influence_length + x_i)
        )

        # Combine turbine TIs with WAT
        turbine_turbulence_intensity = np.maximum( np.sqrt( ti_added ** 2 + ambient_turbulence_intensity ** 2 ) , turbine_turbulence_intensity )

        flow_field.v_sorted += v_wake
        flow_field.w_sorted += w_wake
    flow_field.u_sorted = turb_inflow_field

    flow_field.turbulence_intensity_field = np.mean(turbine_turbulence_intensity, axis=(3,4))
    flow_field.turbulence_intensity_field = flow_field.turbulence_intensity_field[:,:,:,None,None]


def full_flow_cc_solver(farm: Farm, flow_field: FlowField, flow_field_grid: FlowFieldGrid, model_manager: WakeModelManager) -> None:
    # Get the flow quantities and turbine performance
    turbine_grid_farm = copy.deepcopy(farm)
    turbine_grid_flow_field = copy.deepcopy(flow_field)

    turbine_grid_farm.construct_turbine_map()
    turbine_grid_farm.construct_turbine_fCts()
    turbine_grid_farm.construct_turbine_fCps()
    turbine_grid_farm.construct_turbine_power_interps()
    turbine_grid_farm.construct_hub_heights()
    turbine_grid_farm.construct_rotor_diameters()
    turbine_grid_farm.construct_turbine_TSRs()
    turbine_grid_farm.construc_turbine_pPs()
    turbine_grid_farm.construc_turbine_ref_density_cp_cts()
    turbine_grid_farm.construct_coordinates()

    turbine_grid = TurbineGrid(
        turbine_coordinates=turbine_grid_farm.coordinates,
        reference_turbine_diameter=turbine_grid_farm.rotor_diameters,
        wind_directions=turbine_grid_flow_field.wind_directions,
        wind_speeds=turbine_grid_flow_field.wind_speeds,
        grid_resolution=3,
        time_series=turbine_grid_flow_field.time_series,
    )
    turbine_grid_farm.expand_farm_properties(
        turbine_grid_flow_field.n_wind_directions, turbine_grid_flow_field.n_wind_speeds, turbine_grid.sorted_coord_indices
    )
    turbine_grid_flow_field.initialize_velocity_field(turbine_grid)
    turbine_grid_farm.initialize(turbine_grid.sorted_indices)
    cc_solver(turbine_grid_farm, turbine_grid_flow_field, turbine_grid, model_manager)

    ### Referring to the quantities from above, calculate the wake in the full grid

    # Use full flow_field here to use the full grid in the wake models
    deflection_model_args = model_manager.deflection_model.prepare_function(flow_field_grid, flow_field)
    deficit_model_args = model_manager.velocity_model.prepare_function(flow_field_grid, flow_field)

    v_wake = np.zeros_like(flow_field.v_initial_sorted)
    w_wake = np.zeros_like(flow_field.w_initial_sorted)
    turb_u_wake = np.zeros_like(flow_field.u_initial_sorted)

    shape = (farm.n_turbines,) + np.shape(flow_field.u_initial_sorted)
    Ctmp = np.zeros((shape))

    # Calculate the velocity deficit sequentially from upstream to downstream turbines
    for i in range(flow_field_grid.n_turbines):

        # Get the current turbine quantities
        x_i = np.mean(grid.x_sorted[:, :, i:i+1], axis=(3, 4))
        x_i = x_i[:, :, :, None, None]
        y_i = np.mean(grid.y_sorted[:, :, i:i+1], axis=(3, 4))
        y_i = y_i[:, :, :, None, None]
        z_i = np.mean(grid.z_sorted[:, :, i:i+1], axis=(3, 4))
        z_i = z_i[:, :, :, None, None]

        u_i = turbine_grid_flow_field.u_sorted[:, :, i:i+1]
        v_i = turbine_grid_flow_field.v_sorted[:, :, i:i+1]

        turb_avg_vels = average_velocity(turbine_grid_flow_field.u_sorted)
        turb_Cts = Ct(
            velocities=turb_avg_vels,
            yaw_angle=turbine_grid_farm.yaw_angles_sorted,
            fCt=turbine_grid_farm.turbine_fCts,
            turbine_type_map=turbine_grid_farm.turbine_type_map_sorted,
        )
        turb_Cts = turb_Cts[:, :, :, None, None]

        axial_induction_i = axial_induction(
            velocities=turbine_grid_flow_field.u_sorted,
            yaw_angle=turbine_grid_farm.yaw_angles_sorted,
            fCt=turbine_grid_farm.turbine_fCts,
            turbine_type_map=turbine_grid_farm.turbine_type_map_sorted,
            ix_filter=[i],
        )
        axial_induction_i = axial_induction_i[:, :, :, None, None]

        turbulence_intensity_i = turbine_grid_flow_field.turbulence_intensity_field[:, :, i:i+1]
        yaw_angle_i = turbine_grid_farm.yaw_angles_sorted[:, :, i:i+1, None, None]
        hub_height_i = turbine_grid_farm.hub_heights_sorted[:, :, i:i+1, None, None]
        rotor_diameter_i = turbine_grid_farm.rotor_diameters_sorted[:, :, i:i+1, None, None]
        TSR_i = turbine_grid_farm.TSRs_sorted[:, :, i:i+1, None, None]

        effective_yaw_i = np.zeros_like(yaw_angle_i)
        effective_yaw_i += yaw_angle_i

        if model_manager.enable_secondary_steering:
            added_yaw = wake_added_yaw(
                u_i,
                v_i,
                turbine_grid_flow_field.u_initial_sorted,
                turbine_grid.y_sorted[:, :, i:i+1] - y_i,
                turbine_grid.z_sorted[:, :, i:i+1],
                rotor_diameter_i,
                hub_height_i,
                turb_Cts[:, :, i:i+1],
                TSR_i,
                axial_induction_i,
                scale=2.0
            )
            effective_yaw_i += added_yaw

        # Model calculations
        # NOTE: exponential
        deflection_field = model_manager.deflection_model.function(
            x_i,
            y_i,
            effective_yaw_i,
            turbulence_intensity_i,
            turb_Cts[:, :, i:i+1],
            rotor_diameter_i,
            **deflection_model_args
        )

        if model_manager.enable_transverse_velocities:
            v_wake, w_wake = calculate_transverse_velocity(
                u_i,
                flow_field.u_initial_sorted,
                flow_field.dudz_initial_sorted,
                flow_field_grid.x_sorted - x_i,
                flow_field_grid.y_sorted - y_i,
                flow_field_grid.z_sorted,
                rotor_diameter_i,
                hub_height_i,
                yaw_angle_i,
                turb_Cts[:, :, i:i+1],
                TSR_i,
                axial_induction_i,
                scale=2.0
            )

        # NOTE: exponential
        turb_u_wake, Ctmp = model_manager.velocity_model.function(
            i,
            x_i,
            y_i,
            z_i,
            u_i,
            deflection_field,
            yaw_angle_i,
            turbine_grid_flow_field.turbulence_intensity_field,
            turb_Cts,
            turbine_grid_farm.rotor_diameters_sorted[:, :, :, None, None],
            turb_u_wake,
            Ctmp,
            **deficit_model_args
        )

        flow_field.v_sorted += v_wake
        flow_field.w_sorted += w_wake
    flow_field.u_sorted = flow_field.u_initial_sorted - turb_u_wake

def turbopark_solver(farm: Farm, flow_field: FlowField, grid: TurbineGrid, model_manager: WakeModelManager) -> None:
    # Algorithm
    # For each turbine, calculate its effect on every downstream turbine.
    # For the current turbine, we are calculating the deficit that it adds to downstream turbines.
    # Integrate this into the main data structure.
    # Move on to the next turbine.

    # <<interface>>
    deflection_model_args = model_manager.deflection_model.prepare_function(grid, flow_field)
    deficit_model_args = model_manager.velocity_model.prepare_function(grid, flow_field)

    # This is u_wake
    wake_field = np.zeros_like(flow_field.u_initial_sorted)
    v_wake = np.zeros_like(flow_field.v_initial_sorted)
    w_wake = np.zeros_like(flow_field.w_initial_sorted)
    shape = (farm.n_turbines,) + np.shape(flow_field.u_initial_sorted)
    velocity_deficit = np.zeros(shape)
    deflection_field = np.zeros_like(flow_field.u_initial_sorted)

    turbine_turbulence_intensity = flow_field.turbulence_intensity * np.ones((flow_field.n_wind_directions, flow_field.n_wind_speeds, farm.n_turbines, 1, 1))
    ambient_turbulence_intensity = flow_field.turbulence_intensity

    # Calculate the velocity deficit sequentially from upstream to downstream turbines
    for i in range(grid.n_turbines):
        # Get the current turbine quantities
        x_i = np.mean(grid.x_sorted[:, :, i:i+1], axis=(3, 4))
        x_i = x_i[:, :, :, None, None]
        y_i = np.mean(grid.y_sorted[:, :, i:i+1], axis=(3, 4))
        y_i = y_i[:, :, :, None, None]
        z_i = np.mean(grid.z_sorted[:, :, i:i+1], axis=(3, 4))
        z_i = z_i[:, :, :, None, None]

        u_i = flow_field.u_sorted[:, :, i:i+1]
        v_i = flow_field.v_sorted[:, :, i:i+1]

        Cts = Ct(
            velocities=flow_field.u_sorted,
            yaw_angle=farm.yaw_angles_sorted,
            fCt=farm.turbine_fCts,
            turbine_type_map=farm.turbine_type_map_sorted,
        )

        ct_i = Ct(
            velocities=flow_field.u_sorted,
            yaw_angle=farm.yaw_angles_sorted,
            fCt=farm.turbine_fCts,
            turbine_type_map=farm.turbine_type_map_sorted,
            ix_filter=[i],
        )
        ct_i = ct_i[:, :, 0:1, None, None]  # Since we are filtering for the i'th turbine in the Ct function, get the first index here (0:1)
        axial_induction_i = axial_induction(
            velocities=flow_field.u_sorted,
            yaw_angle=farm.yaw_angles_sorted,
            fCt=farm.turbine_fCts,
            turbine_type_map=farm.turbine_type_map_sorted,
            ix_filter=[i],
        )
        axial_induction_i = axial_induction_i[:, :, 0:1, None, None]    # Since we are filtering for the i'th turbine in the axial induction function, get the first index here (0:1)
        turbulence_intensity_i = turbine_turbulence_intensity[:, :, i:i+1]
        yaw_angle_i = farm.yaw_angles_sorted[:, :, i:i+1, None, None]
        hub_height_i = farm.hub_heights_sorted[:, :, i:i+1, None, None]
        rotor_diameter_i = farm.rotor_diameters_sorted[:, :, i:i+1, None, None]
        TSR_i = farm.TSRs_sorted[:, :, i:i+1, None, None]

        effective_yaw_i = np.zeros_like(yaw_angle_i)
        effective_yaw_i += yaw_angle_i

        if model_manager.enable_secondary_steering:
            added_yaw = wake_added_yaw(
                u_i,
                v_i,
                flow_field.u_initial_sorted,
                grid.y_sorted[:, :, i:i+1] - y_i,
                grid.z_sorted[:, :, i:i+1],
                rotor_diameter_i,
                hub_height_i,
                ct_i,
                TSR_i,
                axial_induction_i
            )
            effective_yaw_i += added_yaw

        # Model calculations
        # NOTE: exponential
        if not np.all(farm.yaw_angles_sorted):
            model_manager.deflection_model.logger.warning("WARNING: Deflection with the TurbOPark model has not been fully validated. This is an initial implementation, and we advise you use at your own risk and perform a thorough examination of the results.")
            for ii in range(i):
                x_ii = np.mean(grid.x_sorted[:, :, ii:ii+1], axis=(3, 4))
                x_ii = x_ii[:, :, :, None, None]
                y_ii = np.mean(grid.y_sorted[:, :, ii:ii+1], axis=(3, 4))
                y_ii = y_ii[:, :, :, None, None]

                yaw_ii = farm.yaw_angles_sorted[:, :, ii:ii+1, None, None]
                turbulence_intensity_ii = turbine_turbulence_intensity[:, :, ii:ii+1]
                ct_ii = Ct(
                    velocities=flow_field.u_sorted,
                    yaw_angle=farm.yaw_angles_sorted,
                    fCt=farm.turbine_fCts,
                    turbine_type_map=farm.turbine_type_map_sorted,
                    ix_filter=[ii]
                )
                ct_ii = ct_ii[:, :, 0:1, None, None]
                rotor_diameter_ii = farm.rotor_diameters_sorted[:, :, ii:ii+1, None, None]

                deflection_field_ii = model_manager.deflection_model.function(
                    x_ii,
                    y_ii,
                    yaw_ii,
                    turbulence_intensity_ii,
                    ct_ii,
                    rotor_diameter_ii,
                    **deflection_model_args
                )

                deflection_field[:,:,ii:ii+1,:,:] = deflection_field_ii[:,:,i:i+1,:,:]

        if model_manager.enable_transverse_velocities:
            v_wake, w_wake = calculate_transverse_velocity(
                u_i,
                flow_field.u_initial_sorted,
                flow_field.dudz_initial_sorted,
                grid.x_sorted - x_i,
                grid.y_sorted - y_i,
                grid.z_sorted,
                rotor_diameter_i,
                hub_height_i,
                yaw_angle_i,
                ct_i,
                TSR_i,
                axial_induction_i
            )

        if model_manager.enable_yaw_added_recovery:
            I_mixing = yaw_added_turbulence_mixing(
                u_i,
                turbulence_intensity_i,
                v_i,
                flow_field.w_sorted[:, :, i:i+1],
                v_wake[:, :, i:i+1],
                w_wake[:, :, i:i+1],
            )
            gch_gain = 2
            turbine_turbulence_intensity[:, :, i:i+1] = turbulence_intensity_i + gch_gain * I_mixing

        # NOTE: exponential
        velocity_deficit = model_manager.velocity_model.function(
            x_i,
            y_i,
            z_i,
            turbine_turbulence_intensity,
            Cts[:, :, :, None, None],
            rotor_diameter_i,
            farm.rotor_diameters_sorted[:, :, :, None, None],
            i,
            deflection_field,
            **deficit_model_args
        )

        wake_field = model_manager.combination_model.function(
            wake_field,
            velocity_deficit * flow_field.u_initial_sorted
        )

        wake_added_turbulence_intensity = model_manager.turbulence_model.function(
            ambient_turbulence_intensity,
            grid.x_sorted,
            x_i,
            rotor_diameter_i,
            axial_induction_i
        )

        # TODO: leaving this in for GCH quantities; will need to find another way to compute area_overlap
        # as the current wake deficit is solved for only upstream turbines; could use WAT_upstream
        # Calculate wake overlap for wake-added turbulence (WAT)
        area_overlap = np.sum(velocity_deficit * flow_field.u_initial_sorted > 0.05, axis=(3, 4)) / (grid.grid_resolution * grid.grid_resolution)
        area_overlap = area_overlap[:, :, :, None, None]

        # Modify wake added turbulence by wake area overlap
        downstream_influence_length = 15 * rotor_diameter_i
        ti_added = (
            area_overlap
            * np.nan_to_num(wake_added_turbulence_intensity, posinf=0.0)
            * np.array(grid.x_sorted > x_i)
            * np.array(np.abs(y_i - grid.y_sorted) < 2 * rotor_diameter_i)
            * np.array(grid.x_sorted <= downstream_influence_length + x_i)
        )

        # Combine turbine TIs with WAT
        turbine_turbulence_intensity = np.maximum( np.sqrt( ti_added ** 2 + ambient_turbulence_intensity ** 2 ) , turbine_turbulence_intensity )

        flow_field.u_sorted = flow_field.u_initial_sorted - wake_field
        flow_field.v_sorted += v_wake
        flow_field.w_sorted += w_wake

    flow_field.turbulence_intensity_field = np.mean(turbine_turbulence_intensity, axis=(3,4))
    flow_field.turbulence_intensity_field = flow_field.turbulence_intensity_field[:,:,:,None,None]


def full_flow_turbopark_solver(farm: Farm, flow_field: FlowField, flow_field_grid: FlowFieldGrid, model_manager: WakeModelManager) -> None:
    raise NotImplementedError("Plotting for the TurbOPark model is not currently implemented.")

    # TODO: Below is a first attempt at plotting, and uses just the values on the rotor. The current TurbOPark model requires that
    # points to be calculated are only at turbine locations. Modification will be required to allow for full flow field calculations.

    # # Get the flow quantities and turbine performance
    # turbine_grid_farm = copy.deepcopy(farm)
    # turbine_grid_flow_field = copy.deepcopy(flow_field)

    # turbine_grid_farm.construct_turbine_map()
    # turbine_grid_farm.construct_turbine_fCts()
    # turbine_grid_farm.construct_turbine_fCps()
    # turbine_grid_farm.construct_turbine_power_interps()
    # turbine_grid_farm.construct_hub_heights()
    # turbine_grid_farm.construct_rotor_diameters()
    # turbine_grid_farm.construct_turbine_TSRs()
    # turbine_grid_farm.construc_turbine_pPs()
    # turbine_grid_farm.construct_coordinates()


    # turbine_grid = TurbineGrid(
    #     turbine_coordinates=turbine_grid_farm.coordinates,
    #     reference_turbine_diameter=turbine_grid_farm.rotor_diameters,
    #     wind_directions=turbine_grid_flow_field.wind_directions,
    #     wind_speeds=turbine_grid_flow_field.wind_speeds,
    #     grid_resolution=11,
    # )
    # turbine_grid_farm.expand_farm_properties(
    #     turbine_grid_flow_field.n_wind_directions, turbine_grid_flow_field.n_wind_speeds, turbine_grid.sorted_coord_indices
    # )
    # turbine_grid_flow_field.initialize_velocity_field(turbine_grid)
    # turbine_grid_farm.initialize(turbine_grid.sorted_indices)
    # turbopark_solver(turbine_grid_farm, turbine_grid_flow_field, turbine_grid, model_manager)



    # flow_field.u = copy.deepcopy(turbine_grid_flow_field.u)
    # flow_field.v = copy.deepcopy(turbine_grid_flow_field.v)
    # flow_field.w = copy.deepcopy(turbine_grid_flow_field.w)

    # flow_field_grid.x = copy.deepcopy(turbine_grid.x)
    # flow_field_grid.y = copy.deepcopy(turbine_grid.y)
    # flow_field_grid.z = copy.deepcopy(turbine_grid.z)
