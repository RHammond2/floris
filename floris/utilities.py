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

import os
import functools
from typing import Tuple

import yaml
import numpy as np
from attrs import field, define

from floris.type_dec import NDArrayFloat, floris_array_converter


def pshape(array: np.ndarray, label: str = ""):
    print(label, np.shape(array))


@define
class Vec3:
    """
    Contains 3-component vector information. All arithmetic operators are
    set so that Vec3 objects can operate on and with each other directly.

    Args:
        components (list(numeric, numeric, numeric), numeric): All three vector
            components.
        string_format (str, optional): Format to use in the
            overloaded __str__ function. Defaults to None.
    """
    components: NDArrayFloat = field(converter=floris_array_converter)
    # NOTE: this does not convert elements to float if they are given as int. Is this ok?

    @components.validator
    def _check_components(self, attribute, value) -> None:
        if np.ndim(value) > 1:
            raise ValueError(f"Vec3 must contain exactly 1 dimension, {np.ndim(value)} were given.")
        if np.size(value) != 3:
            raise ValueError(f"Vec3 must contain exactly 3 components, {np.size(value)} were given.")

    def __add__(self, arg):
        if type(arg) is Vec3:
            return Vec3(self.components + arg.components)
        elif type(arg) is int or type(arg) is float:
            return Vec3(self.components + arg)
        else:
            raise ValueError

    def __sub__(self, arg):
        if type(arg) is Vec3:
            return Vec3(self.components - arg.components)
        elif type(arg) is int or type(arg) is float:
            return Vec3(self.components - arg)
        else:
            raise ValueError

    def __mul__(self, arg):
        if type(arg) is Vec3:
            return Vec3(self.components * arg.components)
        elif type(arg) is int or type(arg) is float:
            return Vec3(self.components * arg)
        else:
            raise ValueError

    def __truediv__(self, arg):
        if type(arg) is Vec3:
            return Vec3(self.components / arg.components)
        elif type(arg) is int or type(arg) is float:
            return Vec3(self.components / arg)
        else:
            raise ValueError

    def __eq__(self, arg):
        return False not in np.isclose([self.x1, self.x2, self.x3], [arg.x1, arg.x2, arg.x3])

    def __hash__(self):
        return hash((self.x1, self.x2, self.x3))

    @property
    def x1(self):
        return self.components[0]

    @x1.setter
    def x1(self, value):
        self.components[0] = float(value)

    @property
    def x2(self):
        return self.components[1]

    @x2.setter
    def x2(self, value):
        self.components[1] = float(value)

    @property
    def x3(self):
        return self.components[2]

    @x3.setter
    def x3(self, value):
        self.components[2] = float(value)

    @property
    def elements(self) -> Tuple[float, float, float]:
        # TODO: replace references to elements with components
        # and remove this @property
        return self.components


def cosd(angle):
    """
    Cosine of an angle with the angle given in degrees.

    Args:
        angle (float): Angle in degrees.

    Returns:
        float
    """
    return np.cos(np.radians(angle))


def sind(angle):
    """
    Sine of an angle with the angle given in degrees.

    Args:
        angle (float): Angle in degrees.

    Returns:
        float
    """
    return np.sin(np.radians(angle))


def tand(angle):
    """
    Tangent of an angle with the angle given in degrees.

    Args:
        angle (float): Angle in degrees.

    Returns:
        float
    """
    return np.tan(np.radians(angle))


def wrap_180(x):
    """
    Shift the given values to within the range (-180, 180].

    Args:
        x (numeric or np.array): Scalar value or np.array of values to shift.

    Returns:
        np.array: Shifted values.
    """
    x = np.where(x <= -180.0, x + 360.0, x)
    x = np.where(x > 180.0, x - 360.0, x)
    return x


def wrap_360(x):
    """
    Shift the given values to within the range (0, 360].

    Args:
        x (numeric or np.array): Scalar value or np.array of values to shift.

    Returns:
        np.array: Shifted values.
    """
    x = np.where(x < 0.0, x + 360.0, x)
    x = np.where(x >= 360.0, x - 360.0, x)
    return x


def wind_delta(wind_directions):
    """
    This is included as a function in order to facilitate testing.
    """
    return ((wind_directions - 270) % 360 + 360) % 360


def rotate_coordinates_rel_west(wind_directions, coordinates):
    # Calculate the difference in given wind direction from 270 / West
    wind_deviation_from_west = wind_delta(wind_directions)
    wind_deviation_from_west = np.reshape(wind_deviation_from_west, (len(wind_directions), 1, 1))

    # Construct the arrays storing the turbine locations
    x_coordinates, y_coordinates, z_coordinates = coordinates.T

    # Find center of rotation - this is the center of box bounding all of the turbines
    x_center_of_rotation = (np.min(x_coordinates) + np.max(x_coordinates)) / 2
    y_center_of_rotation = (np.min(y_coordinates) + np.max(y_coordinates)) / 2

    # Rotate turbine coordinates about the center
    x_coord_offset = x_coordinates - x_center_of_rotation
    y_coord_offset = y_coordinates - y_center_of_rotation
    x_coord_rotated = (
        x_coord_offset * cosd(wind_deviation_from_west)
        - y_coord_offset * sind(wind_deviation_from_west)
        + x_center_of_rotation
    )
    y_coord_rotated = (
        x_coord_offset * sind(wind_deviation_from_west)
        + y_coord_offset * cosd(wind_deviation_from_west)
        + y_center_of_rotation
    )
    z_coord_rotated = np.ones_like(wind_deviation_from_west) * z_coordinates
    return x_coord_rotated, y_coord_rotated, z_coord_rotated


class Loader(yaml.SafeLoader):

    def __init__(self, stream):

        self._root = os.path.split(stream.name)[0]

        super().__init__(stream)

    def include(self, node):

        filename = os.path.join(self._root, self.construct_scalar(node))

        with open(filename, 'r') as f:
            return yaml.load(f, self.__class__)


Loader.add_constructor('!include', Loader.include)

def load_yaml(filename, loader=Loader):
    if isinstance(filename, dict):
        return filename  # filename already yaml dict
    with open(filename) as fid:
        return yaml.load(fid, loader)

from multiprocessing import Pool


def split_calculate_join(func):
    """Wrapper/decorator to split the 0th dimension of a NumPy array once the
    array has at least 2 million cells, and adheres to a 5 dimensional
    WS x WD x Turb x grid x grid layout.
    """
    @functools.wraps(func)
    def wrapper_split_and_calculate(*args):
        # Get the indices for the 5-D arrays containing at least 2M cells
        split = []
        for i, arg in enumerate(args):
            if not isinstance(arg, np.ndarray):
                continue
            if arg.ndim < 5:
                continue
            if arg.size > 2e6:
                split.append(i)

        # If not splitting, return the processed function
        if not split:
            return func(*args)

        # Create a list of arguments to be run with the arrays split up
        # according to the maximum array size, limited by the 0th dimension
        ix = split[0]
        arr = args[ix]
        n_split = min(int(np.ceil(arr.size / 2e6)), arr.shape[0])
        split_args = {i: np.array_split(args[i], n_split) for i in split}
        new_args = [args for _ in range(len(split))]
        for i, val in split_args.items():
            for j, v in enumerate(val):
                new_args[j][i] = v

        # Get the results of the function and concatenate back together
        # results = [func(*a) for a in new_args]

        # Optional multiprocessing workflow?
        # TODO: This will likely need to have an environment variable set for
        # the number of nodes to use
        # from multiprocessing import Pool
        with Pool(4) as p:
            results = list(p.starmap(func, new_args))
        results = np.concatenate(results)

    return wrapper_split_and_calculate
