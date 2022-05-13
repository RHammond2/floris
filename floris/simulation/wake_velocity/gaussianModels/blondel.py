# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

import numpy as np
from scipy.special import gamma
from scipy.optimize import minimize_scalar

from floris.utilities import cosd, tand

from .gaussian_model_base import GaussianModel


class Blondel(GaussianModel):
    """
    Blondel is a direct implementation of the super-Gaussian model
    described in :cite:`bcv-blondel2020alternative` with GCH disabled by
    default. See :cite:`bcv-King2019Controls` for info on GCH. The current
    implementation is based on the calibration proposed in
    :cite:`bcv-Cathelain2020calibration`

    References:
        .. bibliography:: /source/zrefs.bib
            :style: unsrt
            :filter: docname in docnames
            :keyprefix: bcv-
    """

    default_parameters = {
        "a_s": 0.179367259,
        "b_s": 0.0118889215,
        "c_s1": 0.0563691592,
        "c_s2": 0.13290157,
        "b_f1": 1.59,
        "b_f2": -23.3114654,
        "b_f3": -2.15199155,
        "c_f": 2.98262872,
        "calculate_VW_velocities": False,
        "use_yaw_added_recovery": False,
        "eps_gain": 0.2,
    }

    def __init__(self, parameter_dictionary):
        """
        Stores model parameters for use by methods.

        Args:
            parameter_dictionary (dict): Model-specific parameters.
                Default values are used when a parameter is not included
                in `parameter_dictionary`. Possible key-value pairs include:

                    -   **a_s**: Parameter used to determine the linear
                        relationship between the turbulence intensity and the
                        width of the Gaussian wake shape.
                    -   **b_s**: Parameter used to determine the linear
                        relationship between the turbulence intensity and the
                        width of the Gaussian wake shape.
                    -   **c_s1**: Parameter used to determine the linear
                        relationship between the turbulence intensity and the
                        width of the Gaussian wake shape.
                    -   **c_s2**: Parameter used to determine the linear
                        relationship between the turbulence intensity and the
                        width of the Gaussian wake shape.
                    -   **b_f1**: Parameter used to determine super-Gaussian
                        order.
                    -   **b_f2**: Parameter used to determine super-Gaussian
                        order.
                    -   **b_f3**: Parameter used to determine super-Gaussian
                        order.
                    -   **c_f**: Parameter used to determine super-Gaussian
                        order.
                    -   **calculate_VW_velocities**: Flag to enable the
                        calculation of V- and W-component velocities using
                        methods developed in :cite:`bcv-King2019Controls`.
                    -   **use_yaw_added_recovery**: Flag to use yaw added
                        recovery on the wake velocity using methods developed
                        in :cite:`bcv-King2019Controls`.
                    -   **eps_gain**: Tuning value for calculating the V- and
                        W-component velocities using methods developed in
                        :cite:`bcv-King2019Controls`.
        """
        super().__init__(parameter_dictionary)

        self.model_string = "blondel"
        model_dictionary = self._get_model_dict(__class__.default_parameters)

        # wake expansion parameters
        # Table 4 of reference [3] in docstring
        self.a_s = model_dictionary["a_s"]
        self.b_s = model_dictionary["b_s"]
        self.c_s1 = model_dictionary["c_s1"]
        self.c_s2 = model_dictionary["c_s2"]

        # fitted parameters for super-Gaussian order n
        # Table 4 of reference [3] in docstring
        self.b_f1 = model_dictionary["b_f1"]
        self.b_f2 = model_dictionary["b_f2"]
        self.b_f3 = model_dictionary["b_f3"]
        self.c_f = model_dictionary["c_f"]

        self.model_grid_resolution = None

        # GCH Parameters
        self.calculate_VW_velocities = model_dictionary["calculate_VW_velocities"]
        self.use_yaw_added_recovery = model_dictionary["use_yaw_added_recovery"]
        self.eps_gain = model_dictionary["eps_gain"]

    def function(
        self,
        x_locations,
        y_locations,
        z_locations,
        turbine,
        turbine_coord,
        deflection_field,
        flow_field,
    ):
        """
        Using the Blondel super-Gaussian wake model, this method calculates and
        returns the wake velocity deficits, caused by the specified turbine,
        relative to the freestream velocities at the grid of points
        comprising the wind farm flow field.

        Args:
            x_locations (np.array): An array of floats that contains the
                streamwise direction grid coordinates of the flow field
                domain (m).
            y_locations (np.array): An array of floats that contains the grid
                coordinates of the flow field domain in the direction normal to
                x and parallel to the ground (m).
            z_locations (np.array): An array of floats that contains the grid
                coordinates of the flow field domain in the vertical
                direction (m).
            turbine (:py:obj:`floris.simulation.turbine`): Object that
                represents the turbine creating the wake.
            turbine_coord (:py:obj:`floris.utilities.Vec3`): Object containing
                the coordinate of the turbine creating the wake (m).
            deflection_field (np.array): An array of floats that contains the
                amount of wake deflection in meters in the y direction at each
                grid point of the flow field.
            flow_field (:py:class:`floris.simulation.flow_field`): Object
                containing the flow field information for the wind farm.

        Returns:
            np.array, np.array, np.array:
                Three arrays of floats that contain the wake velocity
                deficit in m/s created by the turbine relative to the freestream
                velocities for the U, V, and W components, aligned with the x, y,
                and z directions, respectively. The three arrays contain the
                velocity deficits at each grid point in the flow field.
        """
        # TODO: implement veer
        # Veer (degrees)
        # veer = flow_field.wind_veer

        # Turbulence intensity for wake width calculation
        TI = turbine.current_turbulence_intensity

        # Turbine parameters
        D = turbine.rotor_diameter
        HH = turbine.hub_height
        yaw = -1 * turbine.yaw_angle  # opposite sign convention in this model
        Ct = turbine.Ct
        U_local = flow_field.u_initial

        # Wake deflection
        delta = deflection_field

        # Calculate mask values to mask upstream wake
        yR = y_locations - turbine_coord.x2
        xR = yR * tand(yaw) + turbine_coord.x1

        # Compute scaled variables (Eq 1, pp 3 of ref. [1] in docstring)
        # Use absolute value to avoid overflows
        x_tilde = np.abs(x_locations - turbine_coord.x1) / D
        r_tilde = (
            np.sqrt(
                (y_locations - turbine_coord.x2 - delta) ** 2 + (z_locations - HH) ** 2,
                dtype=np.float128,
            )
            / D
        )

        # Calculate Beta (Eq 10, pp 5 of ref. [1] and table 4 of ref. [2] in docstring)
        beta = 0.5 * (1.0 + np.sqrt(1.0 - Ct)) / np.sqrt(1.0 - Ct)
        k = self.a_s * TI + self.b_s
        eps = (self.c_s1 * Ct + self.c_s2) * np.sqrt(beta)

        # Calculate sigma_tilde (Eq 9, pp 5 of ref. [1] and table 4 of ref. [2] in docstring)
        sigma_tilde = k * x_tilde + eps

        # Calculate super-Gaussian order using iterative method
        root_n = minimize_scalar(
            self.match_AD_theory,
            bounds=(0.0, 10.0),
            method="bounded",
            args=(Ct, eps, yaw, self.c_f),
        )
        a_f = root_n.x
        b_f = self.b_f1 * np.exp(self.b_f2 * TI) + self.b_f3
        n = a_f * np.exp(b_f * x_tilde) + self.c_f

        # Calculate max vel def (Eq 5, pp 4 of ref. [1] in docstring)
        a1 = 2 ** (2 / n - 1)
        a2 = 2 ** (4 / n - 2)
        C = a1 - np.sqrt(
            a2
            - (
                (n * Ct)
                * cosd(yaw)
                / (
                    16.0
                    * gamma(2 / n)
                    * np.sign(sigma_tilde)
                    * (np.abs(sigma_tilde) ** (4 / n))
                )
            )
        )

        # Compute wake velocity (Eq 1, pp 3 of ref. [1] in docstring)
        velDef1 = U_local * C * np.exp((-1 * r_tilde ** n) / (2 * sigma_tilde ** 2))
        velDef1[x_locations < xR] = 0

        return (
            np.sqrt(velDef1 ** 2),
            np.zeros(np.shape(velDef1)),
            np.zeros(np.shape(velDef1)),
        )

    def match_AD_theory(self, n_af, Ct, eps, yaw, n_min):
        """
        Calculate the difference in velocities at the rotor plane using the
        Actuator Disk theory and the super-Gaussian model.

        Args:
            n_af (float): Current value of the a_f parameter (.).
            Ct (float): Rotor thrust coefficient (.).
            eps (float): Initial wake expansion parameter (.)
            yaw (float): Rotor yaw angle (deg).
            n_min (float): Minimum value of the super-Gaussian order (.).

        Returns:
            float: difference betweend computed velocity deficit at the rotor
            disk and induction (AD theory)
        """

        # Calculate n using af and minimum value c_f  (Eq 13, pp 6 of ref. [1] in docstring)
        n_val = n_af + n_min
        induction = 0.5 * (1.0 - np.sqrt(1.0 - Ct))

        # Calculate velocity deficit at the disk (x_tilde = 0)
        u_disk = np.power(2.0, 2.0 / n_val - 1.0) - np.sqrt(
            np.power(2.0, 4.0 / n_val - 2.0)
            - n_val
            * Ct
            * cosd(yaw)
            / (16.0 * np.power(eps, 4.0 / n_val) * gamma(2.0 / n_val))
        )

        return abs(u_disk - induction)

    @property
    def a_s(self):
        """
        Constant coefficient used in calculation of wake expansion. See
        Eqn. 9 in [1] and Table 4 in [3].

        **Note:** This is a virtual property used to "get" or "set" a value.

        Args:
            value (float): Value to set.

        Returns:
            float: Value currently set.

        Raises:
            ValueError: Invalid value.
        """
        return self._a_s

    @a_s.setter
    def a_s(self, value):
        if type(value) is not float:
            err_msg = (
                "Invalid value type given for a_s: {}, " + "expected float."
            ).format(value)
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
        self._a_s = value
        if value != __class__.default_parameters["a_s"]:
            self.logger.info(
                (
                    "Current value of a_s, {0}, is not equal to tuned "
                    + "value of {1}."
                ).format(value, __class__.default_parameters["a_s"])
            )

    @property
    def b_s(self):
        """
        Constant coefficient used in calculation of wake expansion. See
        Eqn. 9 in [1] and Table 4 in [3].

        **Note:** This is a virtual property used to "get" or "set" a value.

        Args:
            value (float): Value to set.

        Returns:
            float: Value currently set.

        Raises:
            ValueError: Invalid value.
        """
        return self._b_s

    @b_s.setter
    def b_s(self, value):
        if type(value) is not float:
            err_msg = (
                "Invalid value type given for b_s: {}, " + "expected float."
            ).format(value)
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
        self._b_s = value
        if value != __class__.default_parameters["b_s"]:
            self.logger.info(
                (
                    "Current value of b_s, {0}, is not equal to tuned "
                    + "value of {1}."
                ).format(value, __class__.default_parameters["b_s"])
            )

    @property
    def c_s1(self):
        """
        Linear constant used in calculation of wake expansion. See
        Eqn. 9 in [1] and Table 4 in [3].

        **Note:** This is a virtual property used to "get" or "set" a value.

        Args:
            value (float): Value to set.

        Returns:
            float: Value currently set.

        Raises:
            ValueError: Invalid value.
        """
        return self._c_s1

    @c_s1.setter
    def c_s1(self, value):
        if type(value) is not float:
            err_msg = (
                "Invalid value type given for c_s1: {}, " + "expected float."
            ).format(value)
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
        self._c_s1 = value
        if value != __class__.default_parameters["c_s1"]:
            self.logger.info(
                (
                    "Current value of c_s1, {0}, is not equal to tuned "
                    + "value of {1}."
                ).format(value, __class__.default_parameters["c_s1"])
            )

    @property
    def c_s2(self):
        """
        Linear constant used in calculation of wake expansion. See
        Eqn. 9 in [1] and Table 4 in [3].

        **Note:** This is a virtual property used to "get" or "set" a value.

        Args:
            value (float): Value to set.

        Returns:
            float: Value currently set.

        Raises:
            ValueError: Invalid value.
        """
        return self._c_s2

    @c_s2.setter
    def c_s2(self, value):
        if type(value) is not float:
            err_msg = (
                "Invalid value type given for c_s2: {}, " + "expected float."
            ).format(value)
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
        self._c_s2 = value
        if value != __class__.default_parameters["c_s2"]:
            self.logger.info(
                (
                    "Current value of c_s2, {0}, is not equal to tuned "
                    + "value of {1}."
                ).format(value, __class__.default_parameters["c_s2"])
            )

    @property
    def b_f1(self):
        """
        Constant exponent coefficient used in calculation of the super-Gaussian
        order. See Eqn. 13 in [1] and Table 4 in [3].

        **Note:** This is a virtual property used to "get" or "set" a value.

        Args:
            value (float): Value to set.

        Returns:
            float: Value currently set.

        Raises:
            ValueError: Invalid value.
        """
        return self._b_f1

    @b_f1.setter
    def b_f1(self, value):
        if type(value) is not float:
            err_msg = (
                "Invalid value type given for b_f1: {}, " + "expected float."
            ).format(value)
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
        self._b_f1 = value
        if value != __class__.default_parameters["b_f1"]:
            self.logger.info(
                (
                    "Current value of b_f1, {0}, is not equal to tuned "
                    + "value of {1}."
                ).format(value, __class__.default_parameters["b_f1"])
            )

    @property
    def b_f2(self):
        """
        Constant exponent coefficient used in calculation of the super-Gaussian
        order. See Eqn. 13 in [1] and Table 4 in [3].

        **Note:** This is a virtual property used to "get" or "set" a value.

        Args:
            value (float): Value to set.

        Returns:
            float: Value currently set.

        Raises:
            ValueError: Invalid value.
        """
        return self._b_f2

    @b_f2.setter
    def b_f2(self, value):
        if type(value) is not float:
            err_msg = (
                "Invalid value type given for b_f2: {}, " + "expected float."
            ).format(value)
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
        self._b_f2 = value
        if value != __class__.default_parameters["b_f2"]:
            self.logger.info(
                (
                    "Current value of b_f2, {0}, is not equal to tuned "
                    + "value of {1}."
                ).format(value, __class__.default_parameters["b_f2"])
            )

    @property
    def b_f3(self):
        """
        Constant exponent coefficient used in calculation of the super-Gaussian
        order. See Eqn. 13 in [1] and Table 4 in [3].

        **Note:** This is a virtual property used to "get" or "set" a value.

        Args:
            value (float): Value to set.

        Returns:
            float: Value currently set.

        Raises:
            ValueError: Invalid value.
        """
        return self._b_f3

    @b_f3.setter
    def b_f3(self, value):
        if type(value) is not float:
            err_msg = (
                "Invalid value type given for b_f3: {}, " + "expected float."
            ).format(value)
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
        self._b_f3 = value
        if value != __class__.default_parameters["b_f3"]:
            self.logger.info(
                (
                    "Current value of b_f3, {0}, is not equal to tuned "
                    + "value of {1}."
                ).format(value, __class__.default_parameters["b_f3"])
            )

    @property
    def c_f(self):
        """
        Linear constant used in calculation of the super-Gaussian order. See
        Eqn. 13 in [1] and Table 4 in [3].

        **Note:** This is a virtual property used to "get" or "set" a value.

        Args:
            value (float): Value to set.

        Returns:
            float: Value currently set.

        Raises:
            ValueError: Invalid value.
        """
        return self._c_f

    @c_f.setter
    def c_f(self, value):
        if type(value) is not float:
            err_msg = (
                "Invalid value type given for c_f: {}, " + "expected float."
            ).format(value)
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
        self._c_f = value
        if value != __class__.default_parameters["c_f"]:
            self.logger.info(
                (
                    "Current value of c_f, {0}, is not equal to tuned "
                    + "value of {1}."
                ).format(value, __class__.default_parameters["c_f"])
            )
