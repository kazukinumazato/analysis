#!/usr/bin/env python3
"""Validate the force-vector model using commanded servo angles."""

import argparse
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt

try:
    import rosbag
except ImportError as exc:
    raise SystemExit(
        "rosbag could not be imported. Source the ROS1 workspace before running."
    ) from exc


@dataclass
class MotorInterval:
    start: float
    end: float
    pwm: float
    speed: float = math.nan
    estimated_speed: float = math.nan


@dataclass
class CycleResult:
    pwm: float
    speed: float
    cycle_index: int
    start: float
    end: float
    servo_angle: float
    theta: float
    mean_force: np.ndarray
    force_sample_count: int


def comma_separated_floats(text: str) -> List[float]:
    try:
        values = [float(item.strip()) for item in text.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated numbers") from exc
    if not values:
        raise argparse.ArgumentTypeError("at least one value is required")
    return values


def pwm_flapping_periods(text: str) -> Dict[float, float]:
    periods: Dict[float, float] = {}
    try:
        for item in text.split(","):
            pwm_text, period_text = item.strip().split(":", 1)
            pwm = round(float(pwm_text), 2)
            period = float(period_text)
            if period <= 0.0:
                raise ValueError
            if pwm in periods:
                raise argparse.ArgumentTypeError(
                    "duplicate PWM in --flapping-periods: {:.2f}".format(pwm)
                )
            periods[pwm] = period
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError(
            "expected PWM:period pairs, for example 0.70:0.0808,0.75:0.0654"
        ) from exc
    if not periods:
        raise argparse.ArgumentTypeError("at least one PWM:period pair is required")
    return periods


def servo_angle_from_message(msg, servo_index: int) -> Optional[float]:
    indices = list(msg.index)
    angles = list(msg.angles)
    if len(indices) != len(angles):
        return None
    for index, angle in zip(indices, angles):
        if int(index) == servo_index:
            return float(angle)
    return None


def motor_pwm_from_message(msg, motor_index: int) -> Optional[float]:
    pwms = list(msg.pwms)
    indices = list(msg.motor_index)

    # An empty PWM command exits PWM-test mode and therefore stops this test.
    if not pwms:
        return 0.0

    if indices:
        if len(indices) != len(pwms):
            return None
        for index, pwm in zip(indices, pwms):
            if int(index) == motor_index:
                return float(pwm)
        return None

    # A command without indices applies pwms[0] to every motor in spinal.
    return float(pwms[0])


def read_bag(
    bag_path: Path,
    wrench_topic: str,
    pwm_topic: str,
    servo_topic: str,
    motor_index: int,
    servo_index: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Tuple[float, float]], float]:
    force_times: List[float] = []
    forces: List[Tuple[float, float, float]] = []
    servo_times: List[float] = []
    servo_angles: List[float] = []
    motor_events: List[Tuple[float, float]] = []

    with rosbag.Bag(str(bag_path), "r") as bag:
        bag_end = float(bag.get_end_time())
        for topic, msg, bag_time in bag.read_messages(
            topics=[wrench_topic, pwm_topic, servo_topic]
        ):
            # PWM and servo messages have no header. Bag time is therefore used
            # consistently for all three topics.
            timestamp = float(bag_time.to_sec())
            if topic == wrench_topic:
                force_times.append(timestamp)
                forces.append(
                    (
                        float(msg.wrench.force.x),
                        float(msg.wrench.force.y),
                        float(msg.wrench.force.z),
                    )
                )
            elif topic == servo_topic:
                angle = servo_angle_from_message(msg, servo_index)
                if angle is not None:
                    servo_times.append(timestamp)
                    servo_angles.append(angle)
            elif topic == pwm_topic:
                pwm = motor_pwm_from_message(msg, motor_index)
                if pwm is not None:
                    motor_events.append((timestamp, pwm))

    if not force_times:
        raise ValueError("no force samples found on {}".format(wrench_topic))
    if not servo_times:
        raise ValueError(
            "no commands for servo index {} found on {}".format(
                servo_index, servo_topic
            )
        )
    if not motor_events:
        raise ValueError(
            "no commands for motor index {} found on {}".format(
                motor_index, pwm_topic
            )
        )

    force_order = np.argsort(force_times)
    servo_order = np.argsort(servo_times)
    return (
        np.asarray(force_times, dtype=float)[force_order],
        np.asarray(forces, dtype=float)[force_order],
        np.asarray(servo_times, dtype=float)[servo_order],
        np.asarray(servo_angles, dtype=float)[servo_order],
        sorted(motor_events),
        bag_end,
    )


def find_motor_intervals(
    events: Sequence[Tuple[float, float]],
    active_threshold: float,
    bag_end: float,
) -> List[MotorInterval]:
    intervals: List[MotorInterval] = []
    active_start: Optional[float] = None
    active_pwm: Optional[float] = None

    for timestamp, pwm in events:
        if pwm > active_threshold:
            if active_start is None:
                active_start = timestamp
                active_pwm = pwm
            elif not math.isclose(pwm, active_pwm, abs_tol=1.0e-4):
                intervals.append(MotorInterval(active_start, timestamp, active_pwm))
                active_start = timestamp
                active_pwm = pwm
        elif active_start is not None:
            intervals.append(MotorInterval(active_start, timestamp, active_pwm))
            active_start = None
            active_pwm = None

    if active_start is not None:
        intervals.append(MotorInterval(active_start, bag_end, active_pwm))
    return intervals


def lowpass_forces(
    force_times: np.ndarray,
    forces: np.ndarray,
    cutoff_frequency: float,
) -> Tuple[np.ndarray, float]:
    time_steps = np.diff(force_times)
    time_steps = time_steps[time_steps > 0.0]
    if not len(time_steps):
        raise ValueError("could not estimate the force sampling frequency")

    sampling_frequency = 1.0 / float(np.median(time_steps))
    nyquist_frequency = 0.5 * sampling_frequency
    if cutoff_frequency <= 0.0:
        raise ValueError("--cutoff-frequency must be positive")
    if cutoff_frequency >= nyquist_frequency:
        raise ValueError(
            "--cutoff-frequency ({:.3f} Hz) must be below the estimated "
            "Nyquist frequency ({:.3f} Hz)".format(
                cutoff_frequency, nyquist_frequency
            )
        )

    coefficients = butter(
        4,
        cutoff_frequency / nyquist_frequency,
        btype="low",
        output="sos",
    )
    return sosfiltfilt(coefficients, forces, axis=0), sampling_frequency


def estimate_servo_speed(
    interval: MotorInterval,
    servo_times: np.ndarray,
    servo_angles: np.ndarray,
) -> float:
    mask = (servo_times >= interval.start) & (servo_times <= interval.end)
    times = servo_times[mask]
    angles = servo_angles[mask]
    if len(times) < 2 or np.ptp(angles) < 1.0:
        return 0.0
    relative_times = times - times[0]
    return abs(float(np.polyfit(relative_times, angles, 1)[0]))


def label_intervals(
    intervals: List[MotorInterval],
    speed_sequence: Sequence[float],
    servo_times: np.ndarray,
    servo_angles: np.ndarray,
) -> None:
    occurrence: Dict[float, int] = defaultdict(int)
    for interval in intervals:
        pwm_key = round(interval.pwm, 2)
        sequence_index = occurrence[pwm_key]
        if sequence_index >= len(speed_sequence):
            raise ValueError(
                "PWM {:.2f} has more motor intervals than "
                "--speed-sequence".format(pwm_key)
            )
        interval.pwm = pwm_key
        interval.speed = float(speed_sequence[sequence_index])
        interval.estimated_speed = estimate_servo_speed(
            interval, servo_times, servo_angles
        )
        occurrence[pwm_key] += 1


def servo_rotation_in_sensor(
    servo_angle_deg: float,
    servo_zero_angle_deg: float,
    base_z_rotation_deg: float,
) -> np.ndarray:
    theta = math.radians(servo_angle_deg - servo_zero_angle_deg)
    base_yaw = math.radians(base_z_rotation_deg)

    cos_yaw, sin_yaw = math.cos(base_yaw), math.sin(base_yaw)
    cos_theta, sin_theta = math.cos(theta), math.sin(theta)
    rotation_z = np.array(
        [
            [cos_yaw, -sin_yaw, 0.0],
            [sin_yaw, cos_yaw, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rotation_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cos_theta, -sin_theta],
            [0.0, sin_theta, cos_theta],
        ]
    )
    return rotation_z.dot(rotation_x)


def force_samples_for_cycle(
    start: float,
    end: float,
    force_times: np.ndarray,
    forces: np.ndarray,
    min_force_samples: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
    if start < force_times[0] or end > force_times[-1]:
        return None, None, 0

    first = int(np.searchsorted(force_times, start, side="right"))
    last = int(np.searchsorted(force_times, end, side="left"))
    interior_times = force_times[first:last]
    interior_forces = forces[first:last]
    if len(interior_times) < min_force_samples:
        return None, None, len(interior_times)

    start_force = np.array(
        [np.interp(start, force_times, forces[:, axis]) for axis in range(3)]
    )
    end_force = np.array(
        [np.interp(end, force_times, forces[:, axis]) for axis in range(3)]
    )
    integration_times = np.concatenate(([start], interior_times, [end]))
    integration_forces = np.vstack((start_force, interior_forces, end_force))
    return integration_times, integration_forces, len(interior_times)


def command_angle_at(
    timestamp: float,
    servo_times: np.ndarray,
    servo_angles: np.ndarray,
) -> Optional[float]:
    index = int(np.searchsorted(servo_times, timestamp, side="right") - 1)
    if index < 0:
        return None
    return float(servo_angles[index])


def compute_cycle_results(
    interval: MotorInterval,
    force_times: np.ndarray,
    forces: np.ndarray,
    servo_times: np.ndarray,
    servo_angles: np.ndarray,
    flapping_period: float,
    settle_time: float,
    end_trim: float,
    servo_zero_angle: float,
    base_z_rotation: float,
    min_force_samples: int,
) -> List[CycleResult]:
    first_cycle = int(math.ceil(settle_time / flapping_period - 1.0e-9))
    usable_duration = interval.end - interval.start - end_trim
    cycle_limit = int(math.floor(usable_duration / flapping_period + 1.0e-9))
    results: List[CycleResult] = []

    for cycle_index in range(first_cycle, cycle_limit):
        start = interval.start + cycle_index * flapping_period
        end = start + flapping_period
        integration_times, integration_forces, sample_count = force_samples_for_cycle(
            start,
            end,
            force_times,
            forces,
            min_force_samples,
        )
        if integration_times is None or integration_forces is None:
            continue

        # The fixed condition remains analyzable if recording began just after
        # its 90-degree command. Moving conditions use the latest command at t0.
        if math.isclose(interval.speed, 0.0, abs_tol=1.0e-12):
            servo_angle = servo_zero_angle
        else:
            servo_angle = command_angle_at(start, servo_times, servo_angles)
            if servo_angle is None:
                continue
        theta = servo_angle - servo_zero_angle
        rotation = servo_rotation_in_sensor(
            servo_angle, servo_zero_angle, base_z_rotation
        )
        force_in_cycle_start_frame = integration_forces.dot(rotation)
        mean_force = np.trapz(
            force_in_cycle_start_frame, integration_times, axis=0
        ) / (end - start)
        if np.all(np.isfinite(mean_force)):
            results.append(
                CycleResult(
                    pwm=interval.pwm,
                    speed=interval.speed,
                    cycle_index=cycle_index,
                    start=start,
                    end=end,
                    servo_angle=servo_angle,
                    theta=theta,
                    mean_force=np.asarray(mean_force, dtype=float),
                    force_sample_count=sample_count,
                )
            )
    return results


def build_summary(cycles: Sequence[CycleResult]) -> List[dict]:
    grouped: Dict[Tuple[float, float], List[np.ndarray]] = defaultdict(list)
    for cycle in cycles:
        grouped[(cycle.pwm, cycle.speed)].append(cycle.mean_force)

    summaries: List[dict] = []
    pwm_values = sorted({key[0] for key in grouped})
    for pwm in pwm_values:
        fixed_cycle_vectors = grouped.get((pwm, 0.0), [])
        if not fixed_cycle_vectors:
            print(
                "Warning: skipping PWM {:.2f}: no usable fixed-angle cycles".format(
                    pwm
                ),
                file=sys.stderr,
            )
            continue
        fixed_mean = np.mean(np.vstack(fixed_cycle_vectors), axis=0)
        fixed_norm = float(np.linalg.norm(fixed_mean))
        if math.isclose(fixed_norm, 0.0, abs_tol=1.0e-12):
            raise ValueError(
                "fixed-angle mean force vector is zero for PWM {:.2f}".format(pwm)
            )

        speeds = sorted(speed for key_pwm, speed in grouped if key_pwm == pwm)
        for speed in speeds:
            cycle_vectors = np.vstack(grouped[(pwm, speed)])
            mean_force = np.mean(cycle_vectors, axis=0)
            if (
                not math.isclose(speed, 0.0, abs_tol=1.0e-12)
                and float(mean_force.dot(fixed_mean)) <= 0.0
            ):
                print(
                    "Warning: skipping PWM {:.2f}, {:.1f} deg/s: mean force "
                    "points opposite to the fixed-angle mean (invalid or "
                    "failed-actuator data)".format(pwm, speed),
                    file=sys.stderr,
                )
                continue
            error_vector = mean_force - fixed_mean
            absolute_error = float(np.linalg.norm(error_vector))
            relative_error = 100.0 * absolute_error / fixed_norm
            summaries.append(
                {
                    "pwm": pwm,
                    "speed_deg_s": speed,
                    "cycle_count": len(cycle_vectors),
                    "mean_force_N": mean_force,
                    "fixed_mean_force_N": fixed_mean,
                    "absolute_error_N": absolute_error,
                    "relative_error_percent": relative_error,
                }
            )
    return summaries


def plot_error(summaries: Sequence[dict]) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    pwm_values = sorted({row["pwm"] for row in summaries})
    for pwm in pwm_values:
        rows = sorted(
            (row for row in summaries if row["pwm"] == pwm),
            key=lambda row: row["speed_deg_s"],
        )
        axis.plot(
            [row["speed_deg_s"] for row in rows],
            [row["relative_error_percent"] for row in rows],
            marker="o",
            linewidth=1.8,
            markersize=5.5,
            label="PWM {:.2f}".format(pwm),
        )

    axis.set_xlabel("Servo angular velocity [deg/s]")
    axis.set_ylabel("Relative error norm of mean cycle-averaged force [%]")
    axis.set_title("Validation of the time-averaged thrust model")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    plt.show()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare fixed- and time-varying-servo cycle-averaged force vectors for "
            "each PWM value in a dc_motor_servo_test rosbag."
        )
    )
    parser.add_argument("bag", type=Path, help="input ROS1 bag")
    parser.add_argument("--wrench-topic", default="/cfs/data")
    parser.add_argument("--pwm-topic", default="/pwm_test")
    parser.add_argument("--servo-topic", default="/extra_servo_cmd")
    parser.add_argument("--motor-index", type=int, default=3)
    parser.add_argument("--servo-index", type=int, default=7)
    parser.add_argument(
        "--flapping-periods",
        type=pwm_flapping_periods,
        default=pwm_flapping_periods(
            "0.70:0.08085,0.75:0.06538,0.80:0.05532"
        ),
        metavar="PWM:T_F,...",
        help=(
            "flapping period for each PWM in seconds "
            "(default: 0.70:0.08085,0.75:0.06538,0.80:0.05532)"
        ),
    )
    parser.add_argument(
        "--cutoff-frequency",
        type=float,
        default=30.0,
        help="cutoff frequency of the fourth-order zero-phase force LPF in Hz",
    )
    parser.add_argument(
        "--speed-sequence",
        type=comma_separated_floats,
        default=comma_separated_floats(
            "0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60"
        ),
        metavar="V0,V1,...",
        help=(
            "test order within each PWM group in deg/s "
            "(default: 0,2,...,60)"
        ),
    )
    parser.add_argument(
        "--active-pwm-threshold",
        type=float,
        default=0.55,
        help="commands above this value define a motor-running interval",
    )
    parser.add_argument(
        "--servo-zero-angle",
        type=float,
        default=90.0,
        help="physical servo angle corresponding to theta=0 deg",
    )
    parser.add_argument(
        "--base-z-rotation",
        type=float,
        default=-45.0,
        help="sensor-to-servo base rotation about sensor z in degrees",
    )
    parser.add_argument(
        "--settle-time",
        type=float,
        default=0.0,
        help="optional time omitted at the start of each motor interval",
    )
    parser.add_argument(
        "--end-trim",
        type=float,
        default=0.0,
        help="optional time omitted at the end of each motor interval",
    )
    parser.add_argument(
        "--min-force-samples",
        type=int,
        default=2,
        help="minimum interior force samples required per flapping cycle",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    if args.settle_time < 0.0 or args.end_trim < 0.0:
        raise ValueError("--settle-time and --end-trim must be nonnegative")
    if args.min_force_samples < 1:
        raise ValueError("--min-force-samples must be at least one")
    if not args.bag.is_file():
        raise ValueError("bag does not exist: {}".format(args.bag))

    (
        force_times,
        raw_forces,
        servo_times,
        servo_angles,
        motor_events,
        bag_end,
    ) = read_bag(
        args.bag,
        args.wrench_topic,
        args.pwm_topic,
        args.servo_topic,
        args.motor_index,
        args.servo_index,
    )
    forces, force_sampling_frequency = lowpass_forces(
        force_times, raw_forces, args.cutoff_frequency
    )
    print(
        "Force LPF: fourth-order Butterworth, cutoff {:.3f} Hz, "
        "estimated sampling frequency {:.3f} Hz".format(
            args.cutoff_frequency, force_sampling_frequency
        )
    )
    print("Servo angle: latest /extra_servo_cmd angle at each cycle start")
    intervals = find_motor_intervals(
        motor_events, args.active_pwm_threshold, bag_end
    )
    if not intervals:
        raise ValueError("no motor-running intervals were found")
    label_intervals(
        intervals,
        args.speed_sequence,
        servo_times,
        servo_angles,
    )
    missing_periods = sorted(
        {interval.pwm for interval in intervals}
        - set(args.flapping_periods)
    )
    if missing_periods:
        raise ValueError(
            "--flapping-periods has no value for PWM {}".format(
                ", ".join("{:.2f}".format(pwm) for pwm in missing_periods)
            )
        )

    print(
        "Flapping periods: {}".format(
            ", ".join(
                "PWM {:.2f}: {:.5f} s".format(pwm, period)
                for pwm, period in sorted(args.flapping_periods.items())
            )
        )
    )

    all_cycles: List[CycleResult] = []
    print("Detected motor intervals:")
    for interval in intervals:
        print(
            "  PWM {:.2f}, {:>4.1f} deg/s, duration {:>6.3f} s, "
            "servo-command estimate {:>5.2f} deg/s".format(
                interval.pwm,
                interval.speed,
                interval.end - interval.start,
                interval.estimated_speed,
            )
        )
        interval_cycles = compute_cycle_results(
            interval,
            force_times,
            forces,
            servo_times,
            servo_angles,
            args.flapping_periods[interval.pwm],
            args.settle_time,
            args.end_trim,
            args.servo_zero_angle,
            args.base_z_rotation,
            args.min_force_samples,
        )
        all_cycles.extend(interval_cycles)

    summaries = build_summary(all_cycles)
    if not summaries:
        raise ValueError(
            "no PWM group contains both a usable fixed-angle baseline and "
            "force data"
        )

    print("\nSummary:")
    for row in summaries:
        mean_force = row["mean_force_N"]
        print(
            "  PWM {pwm:.2f}, {speed_deg_s:>4.1f} deg/s: "
            "mean=({: .5f}, {: .5f}, {: .5f}) N, "
            "|delta mean|={absolute_error_N:.5f} N, "
            "error={relative_error_percent:6.2f}%, cycles={cycle_count}".format(
                mean_force[0], mean_force[1], mean_force[2], **row
            )
        )
    plot_error(summaries)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, rosbag.ROSBagException) as error:
        print("Error: {}".format(error), file=sys.stderr)
        sys.exit(1)
