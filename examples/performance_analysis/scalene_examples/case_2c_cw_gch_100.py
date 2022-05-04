"""Modified test script 2c_cw_gch_100"""

import numpy as np
import floris.tools as wfct


# PARAMETERS
num_turbine = 100
turn_off_gch = False

# Initialize the FLORIS interface fi
fi = wfct.floris_interface.FlorisInterface("../../example_input.json")

if turn_off_gch:
    fi.set_gch(False)

D = 126.0
fi.reinitialize_flow_field(
    layout_array=[
        [D * 6 * i for i in range(num_turbine)],
        [0 for i in range(num_turbine)],
    ]
)

# Calculate wake
fi.calculate_wake()

# Collect the turbine powers
fi.calculate_wake()
turbine_powers = np.array(fi.get_turbine_power())
