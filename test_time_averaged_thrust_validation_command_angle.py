import importlib.util
import math
import sys
import types
from pathlib import Path

import numpy as np

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


def make_cycle(speed_deg_s, mean_force, sensitivity):
    return module.CycleResult(
        pwm=0.8,
        speed=speed_deg_s,
        cycle_index=0,
        start=0.0,
        end=0.1,
        servo_angle=90.0,
        theta=0.0,
        mean_force=np.asarray(mean_force, dtype=float),
        linearized_force_rate_sensitivity=np.asarray(sensitivity, dtype=float),
        max_force_magnitude=2.0,
        force_sample_count=3,
    )


def test_build_summary_adds_equation_9_first_order_prediction():
    cycles = [
        make_cycle(0.0, [0.0, 0.0, 2.0], [0.0, -0.1, 0.0]),
        make_cycle(60.0, [0.0, -0.2, 2.0], [99.0, 99.0, 99.0]),
    ]

    summaries = module.build_summary(cycles)
    moving = next(row for row in summaries if row["speed_deg_s"] == 60.0)

    # For f=[0, 0, 2] N and T_f=0.1 s, Eq. (9) gives the
    # sensitivity ||mean(t * (a x f))|| = 0.1 N s.  At 60 deg/s,
    # Eq. (10)'s theoretical left-hand side is omega * 0.1 / 2.
    expected_percent = 100.0 * math.radians(60.0) * 0.1 / 2.0
    assert math.isclose(
        moving["theoretical_relative_error_percent"],
        expected_percent,
        rel_tol=1.0e-12,
    )
