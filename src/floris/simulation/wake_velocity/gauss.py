# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

from typing import Any, Dict

import attr
import numpy as np

from floris.utilities import cosd, sind, tand, float_attrib, model_attrib
from floris.simulation import Farm, Grid, BaseModel, FlowField


@attr.s(auto_attribs=True)
class GaussVelocityDeficit(BaseModel):
    """
    The new Gauss model blends the previously implemented Gussian model based
    on [1-5] with the super-Gaussian model of [6]. The blending is meant to
    provide consistency with previous results in the far wake while improving
    prediction of the near wake.

    Args:
        ka (float): Parameter used to determine the linear relationship between the
            turbulence intensity and the width of the Gaussian wake shape. Defaults to
            0.58.
        kb (float): Parameter used to determine the linear relationship between the
            turbulence intensity and the width of the Gaussian wake shape. Defaults to
            0.077.
        alpha (float): Parameter that determines the dependence of the downstream
            boundary between the near wake and far wake region on the turbulence
            intensity. Defaults to 0.38.
        beta (float): Parameter that determines the dependence of the downstream
            boundary between the near wake and far wake region on the turbine's
            induction factor. Defaults to 0.004.

    See :cite:`gvm-bastankhah2014new`, :cite:`gvm-abkar2015influence`,
    :cite:`gvm-bastankhah2016experimental`, :cite:`gvm-niayifar2016analytical`,
    :cite:`gvm-dilip2017wind`, :cite:`gvm-blondel2020alternative`, and
    :cite:`gvm-King2019Controls` for more information on Gaussian wake velocity
    deficit models.
    References:
        .. bibliography:: /source/zrefs.bib
            :style: unsrt
            :filter: docname in docnames
            :keyprefix: gvm-
    """

    alpha: float = float_attrib(default=0.58)
    beta: float = float_attrib(default=0.077)
    ka: float = float_attrib(default=0.38)
    kb: float = float_attrib(default=0.004)

    model_string = "gauss"

    def prepare_function(
        self,
        grid: Grid,
        farm: Farm,
        flow_field: FlowField
    ) -> Dict[str, Any]:

        kwargs = dict(
            x=grid.x,
            y=grid.y,
            z=grid.z,
            reference_hub_height=farm.reference_hub_height,
            reference_rotor_diameter=farm.reference_turbine_diameter,
            u_initial=flow_field.u_initial,
            wind_veer=flow_field.wind_veer,
        )
        return kwargs

    # @profile
    def function(
        self,
        x_i: np.ndarray,
        y_i: np.ndarray,
        z_i: np.ndarray,
        axial_induction_i: np.ndarray,
        deflection_field_i: np.ndarray,
        yaw_angle_i: np.ndarray,
        turbulence_intensity_i: np.ndarray,
        ct_i: np.ndarray,
        # enforces the use of the below as keyword arguments and adherence to the
        # unpacking of the results from prepare_function()
        *,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        reference_hub_height: float,
        reference_rotor_diameter: np.ndarray,
        u_initial: np.ndarray,
        wind_veer: float,
    ) -> None:

        # yaw_angle is all turbine yaw angles for each wind speed
        # Extract and broadcast only the current turbine yaw setting
        # for all wind speeds
        # TODO: Difference in yaw sign convention for v3
        yaw_angle = -1 * yaw_angle_i  # Opposite sign convention in this model

        # Initialize the velocity deficit
        uR = u_initial * ct_i / (2.0 * (1 - np.sqrt(1 - ct_i)))
        u0 = u_initial * np.sqrt(1 - ct_i)

        # Initial lateral bounds
        sigma_z0 = reference_rotor_diameter * 0.5 * np.sqrt(uR / (u_initial + u0))
        sigma_y0 = sigma_z0 * cosd(yaw_angle) * cosd(wind_veer)

        # Compute the bounds of the near and far wake regions and a mask

        # Start of the near wake
        xR = x_i

        # Start of the far wake
        x0 = np.ones_like(u_initial)
        x0 *= reference_rotor_diameter * cosd(yaw_angle) * (1 + np.sqrt(1 - ct_i) )
        x0 /= np.sqrt(2) * (4 * self.alpha * turbulence_intensity_i + 2 * self.beta * (1 - np.sqrt(1 - ct_i) ) )
        x0 += x_i

        # Masks
        near_wake_mask = np.array(x > xR) * np.array(x < x0)  # This mask defines the near wake; keeps the areas downstream of xR and upstream of x0
        far_wake_mask = np.array(x >= x0)

        # Compute the velocity deficit in the NEAR WAKE region
        # TODO: for the turbinegrid, do we need to do this near wake calculation at all?
        #       same question for any grid with a resolution larger than the near wake region

        # Calculate the wake expansion
        near_wake_ramp_up = (x - xR) / (x0 - xR)  # This is a linear ramp from 0 to 1 from the start of the near wake to the start of the far wake.
        near_wake_ramp_down = (x0 - x) / (x0 - xR)  # Another linear ramp, but positive upstream of the far wake and negative in the far wake; 0 at the start of the far wake
        # near_wake_ramp_down = -1 * (near_wake_ramp_up - 1)  # TODO: this is equivalent, right?

        sigma_y = near_wake_ramp_down * 0.501 * reference_rotor_diameter * np.sqrt(ct_i / 2.0) + near_wake_ramp_up * sigma_y0
        sigma_y = sigma_y * np.array(x >= xR) + np.ones_like(sigma_y) * np.array(x < xR) * 0.5 * reference_rotor_diameter

        sigma_z = near_wake_ramp_down * 0.501 * reference_rotor_diameter * np.sqrt(ct_i / 2.0) + near_wake_ramp_up * sigma_z0
        sigma_z = sigma_z * np.array(x >= xR) + np.ones_like(sigma_z) * np.array(x < xR) * 0.5 * reference_rotor_diameter

        r, C = rC(
            wind_veer,
            sigma_y,
            sigma_z,
            y,
            y_i,
            deflection_field_i,
            z,
            reference_hub_height,
            ct_i,
            yaw_angle,
            reference_rotor_diameter,
        )

        near_wake_deficit = gaussian_function(C, r, 1, np.sqrt(0.5))
        near_wake_deficit *= near_wake_mask

        # Compute the velocity deficit in the FAR WAKE region

        # Wake expansion in the lateral (y) and the vertical (z)
        ky = self.ka * turbulence_intensity_i + self.kb  # wake expansion parameters
        kz = self.ka * turbulence_intensity_i + self.kb  # wake expansion parameters
        sigma_y = (ky * (x - x0) + sigma_y0) * far_wake_mask + sigma_y0 * np.array(x < x0)
        sigma_z = (kz * (x - x0) + sigma_z0) * far_wake_mask + sigma_z0 * np.array(x < x0)

        r, C = rC(
            wind_veer,
            sigma_y,
            sigma_z,
            y,
            y_i,
            deflection_field_i,
            z,
            reference_hub_height,
            ct_i,
            yaw_angle,
            reference_rotor_diameter,
        )

        far_wake_deficit = gaussian_function(C, r, 1, np.sqrt(0.5))
        far_wake_deficit *= far_wake_mask

        # Combine the near and far wake regions
        velocity_deficit = np.sqrt(near_wake_deficit ** 2 + far_wake_deficit ** 2)

        return velocity_deficit


# @profile
def rC(wind_veer, sigma_y, sigma_z, y, y_i, delta, z, HH, Ct, yaw, D):
    a = cosd(wind_veer) ** 2 / (2 * sigma_y ** 2) + sind(wind_veer) ** 2 / (2 * sigma_z ** 2)
    b = -sind(2 * wind_veer) / (4 * sigma_y ** 2) + sind(2 * wind_veer) / (4 * sigma_z ** 2)
    c = sind(wind_veer) ** 2 / (2 * sigma_y ** 2) + cosd(wind_veer) ** 2 / (2 * sigma_z ** 2)
    r = a * (y - y_i - delta) ** 2 - 2 * b * (y - y_i - delta) * (z - HH) + c * (z - HH) ** 2
    C = 1 - np.sqrt(np.clip(1 - (Ct * cosd(yaw) / (8.0 * sigma_y * sigma_z / D ** 2)), 0.0, 1.0))
    return r, C


def mask_upstream_wake(mesh_y_rotated, x_coord_rotated, y_coord_rotated, turbine_yaw):
    yR = mesh_y_rotated - y_coord_rotated
    xR = yR * tand(turbine_yaw) + x_coord_rotated
    return xR, yR


def gaussian_function(C, r, n, sigma):
    return C * np.exp(-1 * r ** n / (2 * sigma ** 2))
