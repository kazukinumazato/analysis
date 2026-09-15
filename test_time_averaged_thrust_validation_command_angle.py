import importlib.util
import sys
import types
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("time_averaged_thrust_validation_command_angle.py")

rosbag_stub = types.ModuleType("rosbag")
rosbag_stub.ROSBagException = RuntimeError
sys.modules.setdefault("rosbag", rosbag_stub)

spec = importlib.util.spec_from_file_location("time_averaged_thrust_validation_command_angle", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_filter_summaries_by_pwm_and_speed_range():
    summaries = [
        {"pwm": 0.70, "speed_deg_s": 0.0, "relative_error_percent": 1.0},
        {"pwm": 0.70, "speed_deg_s": 12.0, "relative_error_percent": 2.0},
        {"pwm": 0.70, "speed_deg_s": 30.0, "relative_error_percent": 3.0},
        {"pwm": 0.75, "speed_deg_s": 10.0, "relative_error_percent": 4.0},
        {"pwm": 0.75, "speed_deg_s": 40.0, "relative_error_percent": 5.0},
    ]

    filtered = module.filter_summaries(
        summaries,
        pwm_values=[0.70],
        speed_min=5.0,
        speed_max=30.0,
    )

    assert filtered == [
        {"pwm": 0.70, "speed_deg_s": 12.0, "relative_error_percent": 2.0},
        {"pwm": 0.70, "speed_deg_s": 30.0, "relative_error_percent": 3.0},
    ]
