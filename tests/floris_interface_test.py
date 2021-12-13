from pathlib import Path

from floris.tools.floris_interface import FlorisInterface


TEST_DATA = Path(__file__).resolve().parent / "data"
YAML_INPUT = TEST_DATA / "input_full_v3.yaml"
JSON_INPUT = TEST_DATA / "input_full_v3.json"


def test_read_json():
    fi = FlorisInterface(configuration=JSON_INPUT)
    assert isinstance(fi, FlorisInterface)


def test_read_yaml():
    fi = FlorisInterface(configuration=YAML_INPUT)
    assert isinstance(fi, FlorisInterface)


def test_calculate_wake():
    pass


def test_reinitialize_flow_field():
    pass


def test_get_plane_of_points():
    pass


def test_get_set_of_points():
    pass


def test_get_hor_plane():
    pass


def test_get_cross_plane():
    pass


def test_get_y_plane():
    pass


def test_get_flow_data():
    pass


def test_get_yaw_angles():
    pass


def test_get_farm_power():
    pass


def test_get_turbine_layout():
    pass


def test_get_power_curve():
    pass


def test_get_turbine_ct():
    pass


def test_get_turbine_ti():
    pass


def test_get_farm_power_for_yaw_angle():
    pass


def test_get_farm_AEP():
    pass


def test_calc_one_AEP_case():
    pass


def test_get_farm_AEP_parallel():
    pass


def test_calc_AEP_wind_limit():
    pass


def test_calc_change_turbine():
    pass


def test_set_use_points_on_perimeter():
    pass


def test_set_gch():
    pass


def test_set_gch_yaw_added_recovery():
    pass


def test_set_gch_secondary_steering():
    pass


def test_layout_x():  # TODO
    pass


def test_layout_y():  # TODO
    pass


def test_TKE_to_TI():
    pass


def test_set_rotor_diameter():  # TODO
    pass


def test_show_model_parameters():  # TODO
    pass


def test_get_model_parameters():
    # TODO: Add in the turbulence model component once implemented!
    fi = FlorisInterface(configuration=JSON_INPUT)

    # Test to check that all parameters are returned
    correct_parameters = {
        "wake_deflection_parameters": {"ad": 0.0, "bd": 0.0, "kd": 0.05},
        "wake_velocity_parameters": {"we": 0.05},
    }
    parameters = fi.get_model_parameters(turbulence_model=False)
    assert parameters == correct_parameters

    # Test to check that only deflection parameters are returned
    correct_parameters = {"wake_deflection_parameters": {"ad": 0.0, "bd": 0.0, "kd": 0.05}}
    parameters = fi.get_model_parameters(turbulence_model=False, wake_velocity_model=False)
    assert parameters == correct_parameters

    # Test to check that only velocity models are returned
    correct_parameters = {"wake_velocity_parameters": {"we": 0.05}}
    parameters = fi.get_model_parameters(turbulence_model=False, wake_deflection_model=False)
    assert parameters == correct_parameters

    # Check that oly "ad" and "kd" values are returned
    correct_parameters = {"wake_deflection_parameters": {"ad": 0.0, "kd": 0.05}, "wake_velocity_parameters": {}}
    parameters = fi.get_model_parameters(turbulence_model=False, params=["ad", "kd"])
    assert parameters == correct_parameters


def test_set_model_parameters():
    # TODO: Add in the turbulence model component once implemented!
    fi = FlorisInterface(configuration=JSON_INPUT)

    correct_parameters = {
        "wake_deflection_parameters": {"ad": 0.1, "bd": 0.0, "kd": 0.05},
        "wake_velocity_parameters": {"we": 0.22},
    }
    update_parameters = {
        "Wake Deflection Parameters": {"ad": 0.1},  # Check v2 naming convention
        "wake velocity_Parameters": {"we": 0.22},  # Check mixed naming convention
    }
    fi.set_model_parameters(params=update_parameters)
    parameters = fi.get_model_parameters(turbulence_model=False)
    assert parameters == correct_parameters


def test_vis_layout():
    pass


def test_show_flow_field():
    pass
