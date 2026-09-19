import importlib.util
import math
import sys
import types
from pathlib import Path

import numpy as np

MODULE_PATH = Path(__file__).with_name(
    "time_averaged_thrust_validation_command_angle_yz_plane.py"
)

rosbag_stub = types.ModuleType("rosbag")
rosbag_stub.ROSBagException = RuntimeError
sys.modules.setdefault("rosbag", rosbag_stub)

spec = importlib.util.spec_from_file_location("time_averaged_thrust_validation_command_angle", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_filter_summaries_by_pwm_and_speed_range():
    summaries = [
        {"pwm": 0.70, "speed_deg_s": 0.0, "relative_yz_error_percent": 1.0},
        {"pwm": 0.70, "speed_deg_s": 12.0, "relative_yz_error_percent": 2.0},
        {"pwm": 0.70, "speed_deg_s": 30.0, "relative_yz_error_percent": 3.0},
        {"pwm": 0.75, "speed_deg_s": 10.0, "relative_yz_error_percent": 4.0},
        {"pwm": 0.75, "speed_deg_s": 40.0, "relative_yz_error_percent": 5.0},
    ]

    filtered = module.filter_summaries(
        summaries,
        pwm_values=[0.70],
        speed_min=5.0,
        speed_max=30.0,
    )

    assert filtered == [
        {"pwm": 0.70, "speed_deg_s": 12.0, "relative_yz_error_percent": 2.0},
        {"pwm": 0.70, "speed_deg_s": 30.0, "relative_yz_error_percent": 3.0},
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


def test_build_summary_adds_equation_9_yz_plane_prediction():
    cycles = [
        make_cycle(0.0, [0.0, 0.0, 2.0], [0.0, -0.1, 0.0]),
        make_cycle(60.0, [0.0, -0.2, 2.0], [99.0, 99.0, 99.0]),
    ]

    summaries = module.build_summary(cycles)
    moving = next(row for row in summaries if row["speed_deg_s"] == 60.0)

    # For f=[0, 0, 2] N and T_f=0.1 s, Eq. (9) gives the
    # y-z-plane sensitivity norm 0.1 N s. At 60 deg/s, the y-z-plane
    # version of Eq. (10)'s left-hand side is omega * 0.1 / 2.
    expected_percent = 100.0 * math.radians(60.0) * 0.1 / 2.0
    assert math.isclose(
        moving["theoretical_relative_yz_error_percent"],
        expected_percent,
        rel_tol=1.0e-12,
    )


def test_build_summary_ignores_x_error():
    cycles = [
        make_cycle(0.0, [3.0, 4.0, 2.0], [0.0, 0.0, 0.0]),
        make_cycle(10.0, [30.0, 4.2, 2.1], [0.0, 0.0, 0.0]),
    ]

    summaries = module.build_summary(cycles)
    moving = next(row for row in summaries if row["speed_deg_s"] == 10.0)

    assert math.isclose(moving["absolute_yz_error_N"], math.sqrt(0.05))
    assert math.isclose(moving["relative_yz_error_percent"], 5.0)


def test_default_flapping_periods_match_measured_frequencies(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["program", "input.bag"])

    arguments = module.parse_arguments()

    assert arguments.flapping_periods == {
        0.70: 0.07100,
        0.75: 0.05690,
        0.80: 0.04998,
    }
    assert arguments.motor_off_offset_correction is True
    assert math.isclose(arguments.motor_off_offset_trim, 0.15)


def test_subtract_interpolated_motor_off_offset():
    force_times = np.arange(13, dtype=float)
    forces = np.zeros((13, 3), dtype=float)
    forces[1:4] = [1.0, 2.0, 3.0]
    forces[9:12] = [3.0, 4.0, 5.0]
    forces[6] = [12.0, 13.0, 14.0]
    intervals = [module.MotorInterval(start=4.0, end=8.0, pwm=0.8)]

    corrected, offset_times, offsets = (
        module.subtract_interpolated_motor_off_offset(
            force_times, forces, intervals, edge_trim=0.0
        )
    )

    assert np.allclose(offset_times, [2.0, 10.0])
    assert np.allclose(offsets, [[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])
    assert np.allclose(corrected[6], [10.0, 10.0, 10.0])
