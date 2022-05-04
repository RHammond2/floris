"""Modified test script 5a_aep_3x3_simple"""

import itertools
import numpy as np
import floris.tools as wfct


# PARAMETERS
turn_off_gch = True
N_row = 3

# Initialize the FLORIS interface fi
fi = wfct.floris_interface.FlorisInterface("../../example_input.json")

if turn_off_gch:
    fi.set_gch(False)

# Set to a 5 turbine case
D = fi.floris.farm.turbines[0].rotor_diameter
spc = 5
layout_x = []
layout_y = []
for i in range(N_row):
    for k in range(N_row):
        layout_x.append(i * spc * D)
        layout_y.append(k * spc * D)
N_turb = len(layout_x)

fi.reinitialize_flow_field(
    layout_array=(layout_x, layout_y), wind_direction=[270.0], wind_speed=[8.0]
)
fi.calculate_wake()

# Set up the wind rose assuming every wind speed and direction equaly likely
ws_list = np.arange(3, 26, 1)
wd_list = np.arange(0, 360, 5)
combined = np.array(list(itertools.product(ws_list, wd_list)))
ws_list = combined[:, 0]
wd_list = combined[:, 1]
num_cases = len(ws_list)

# Use simple weibull
wind_rose = wfct.wind_rose.WindRose()
freq = wind_rose.weibull(ws_list)
freq = freq / np.sum(freq)
# freq = np.ones_like(ws_list) / num_cases

power_result = fi.get_farm_AEP(wd_list, ws_list, freq, limit_ws=True)
