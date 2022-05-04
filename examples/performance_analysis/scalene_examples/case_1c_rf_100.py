"""Modified test script 1c_rf_100"""

import numpy as np
import floris.tools as wfct


# PARAMETERS
num_turbine = 100

# Initialize the FLORIS interface fi
fi = wfct.floris_interface.FlorisInterface("../../example_input.json")

D = 126.0
fi.reinitialize_flow_field(
    layout_array=[
        [D * 6 * i for i in range(num_turbine)],
        [0 for i in range(num_turbine)],
    ]
)

# Calculate wake
fi.calculate_wake()

# Now check the timing
fi.reinitialize_flow_field(wind_speed=8.0 + i / N, wind_direction=270.0 + i / N)

# Collect the turbine powers
fi.calculate_wake()
turbine_powers = np.array(fi.get_turbine_power())
