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

from floris.simulation import Turbine, FlowField

from ...utilities import cosd, sind
from .base_velocity_deflection import VelocityDeflection


class Template(VelocityDeflection):
    """Another wake deflection model, derived from
    :cite: `jdm-citation`

    References:
        .. bibliography:: /source/zrefs.bib
            :style: unsrt
            :filter: docname in docnames
            :keyprefix" jdm

    Parameters
    ----------
    VelocityDeflection : `floris.VelocityDeflection` ???
        The base velocity deflection class.
    """

    def __init__(self, parameter_dictionary: dict):
        """
        Stores model parameters for use by methods.

        Args:
            parameter_dictionary (dict): Model-specific parameters.
                Default values are used when a parameter is not included
                in `parameter_dictionary`. Possible key-value pairs include:

                    -   **param1** (*float*): Parameter used to ....
        """
        super().__init__(parameter_dictionary)
        self.model_string = "template"
        model_dictionary = self._get_model_dict(__class__.default_parameters)
        self.param1 = float(model_dictionary["param1"])

    def function(
        self,
        x_locations: np.ndarray,
        y_locations: np.ndarray,
        z_locations: np.ndarray,
        turbine: Turbine,
        coords: np.ndarray,
        flow_field: FlowField,
    ):
        """
        Calcualtes the deflection field of the wake by.... This is coded as defined in [1].

        Args:
            x_locations (np.array): streamwise locations in wake
            y_locations (np.array): spanwise locations in wake
            z_locations (np.array): vertical locations in wake
            turbine (:py:class:`floris.simulation.turbine.Turbine`):
                Turbine object
                # NOTE: DO WE DEAL IN MULTI-TURBINE FARMS?
            coord
                (:py:meth:`floris.simulation.turbine_map.TurbineMap.coords`):
                Spatial coordinates of wind turbine.
            flow_field
                (:py:class:`floris.simulation.flow_field.FlowField`):
                Flow field object.

        Returns:
            deflection (np.array): Deflected wake centerline.
        """
        deflection = np.zeros(
            x_locations.shape
        )  # Should take the same shape as the locations input.
        return deflection
