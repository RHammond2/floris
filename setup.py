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


from pathlib import Path

from setuptools import Extension, setup, find_packages


try:
    from Cython.Build import cythonize

    ext_modules = cythonize([
        # Simualtion Core
        Extension("floris.simulation.base", ["floris/simulation/base.py"]),
        Extension("floris.simulation.farm", ["floris/simulation/farm.py"]),
        Extension("floris.simulation.grid", ["floris/simulation/grid.py"]),
        Extension("floris.simulation.wake", ["floris/simulation/wake.py"]),
        Extension("floris.simulation.floris", ["floris/simulation/floris.py"]),
        Extension("floris.simulation.solver", ["floris/simulation/solver.py"]),
        Extension("floris.simulation.turbine", ["floris/simulation/turbine.py"]),
        Extension("floris.simulation.flow_field", ["floris/simulation/flow_field.py"]),

        # Velocity Models
        Extension("floris.simulation.wake_velocity.gauss", ["floris/simulation/wake_velocity/gauss.py"]),
        Extension("floris.simulation.wake_velocity.turbopark", ["floris/simulation/wake_velocity/turbopark.py"]),
        Extension("floris.simulation.wake_velocity.cumulative_gauss_curl", ["floris/simulation/wake_velocity/cumulative_gauss_curl.py"]),

        # Deflection Models
        Extension("floris.simulation.wake_deflection.curl", ["floris/simulation/wake_deflection/curl.py"]),
        Extension("floris.simulation.wake_deflection.gauss", ["floris/simulation/wake_deflection/gauss.py"]),
        Extension("floris.simulation.wake_deflection.jimenez", ["floris/simulation/wake_deflection/jimenez.py"]),

        # Combination Models
        Extension("floris.simulation.wake_combination.fls", ["floris/simulation/wake_combination/fls.py"]),
        Extension("floris.simulation.wake_combination.max", ["floris/simulation/wake_combination/max.py"]),
        Extension("floris.simulation.wake_combination.sosfs", ["floris/simulation/wake_combination/sosfs.py"]),

        # Turbulence Models
        Extension("floris.simulation.wake_turbulence.crespo_hernandez", ["floris/simulation/wake_turbulence/crespo_hernandez.py"]),

        # Tools
        Extension("floris.tools.rews", ["floris/tools/rews.py"]),
        Extension("floris.tools.plotting", ["floris/tools/plotting.py"]),
        Extension("floris.tools.cut_plane", ["floris/tools/cut_plane.py"]),
        Extension("floris.tools.flow_data", ["floris/tools/flow_data.py"]),
        Extension("floris.tools.wind_rose", ["floris/tools/wind_rose.py"]),
        Extension("floris.tools.power_rose", ["floris/tools/power_rose.py"]),
        Extension("floris.tools.visualization", ["floris/tools/visualization.py"]),
        Extension("floris.tools.sowfa_utilities", ["floris/tools/sowfa_utilities.py"]),
        Extension("floris.tools.floris_interface", ["floris/tools/floris_interface.py"]),
        Extension("floris.tools.layout_functions", ["floris/tools/layout_functions.py"]),
        Extension("floris.tools.cc_blade_utilities", ["floris/tools/cc_blade_utilities.py"]),
        Extension("floris.tools.interface_utilities", ["floris/tools/interface_utilities.py"]),
        Extension("floris.tools.uncertainty_interface", ["floris/tools/uncertainty_interface.py"]),
        Extension("floris.tools.floris_interface_legacy_reader", ["floris/tools/floris_interface_legacy_reader.py"]),
    ])
except ImportError:
    ext_modules = None


# Package meta-data.
NAME = "FLORIS"
DESCRIPTION = "A controls-oriented engineering wake model."
URL = "https://github.com/NREL/FLORIS"
EMAIL = "rafael.mudafort@nrel.gov"
AUTHOR = "NREL National Wind Technology Center"
REQUIRES_PYTHON = ">=3.8.0"

# What packages are required for this module to be executed?
REQUIRED = [
    # simulation
    "attrs",
    "pyyaml",
    "numexpr",
    "numpy>=1.20",
    "scipy>=1.1",
    "cython",

    # tools
    "matplotlib>=3",
    "pandas",
    "shapely",

    # utilities
    "coloredlogs>=10.0",
]

# What packages are optional?
EXTRAS = {
    "docs": {"readthedocs-sphinx-ext", "Sphinx", "sphinxcontrib-napoleon"},
    "develop": {"pytest", "coverage[toml]", "pre-commit", "black", "isort"},
}

ROOT = Path(__file__).parent
with open(ROOT / "floris" / "version.py") as version_file:
    VERSION = version_file.read().strip()

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=DESCRIPTION,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    package_data={'floris': ['turbine_library/*.yaml']},
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license="Apache-2.0",
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy"
    ],
    ext_modules=ext_modules,
)
