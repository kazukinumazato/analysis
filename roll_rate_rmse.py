#!/usr/bin/env python3
"""Plot tracking RMSE against the commanded roll rate for ROS1 bags.

The default input set is the five ``2026-09-14`` roll-sweep bags in
``../crobat_flight``.  Two analysis windows are shown separately: the first
non-zero ``/crobat/desire_coordinate`` command through +0.5 rad, and the
``/crobat/final_target_baselink_rpy`` -0.5 rad command through the first
``/crobat/desire_coordinate`` arrival at -0.5 rad.  Optional start/end offsets
are added to both detected windows.

Position targets are read from ``/crobat/debug/pose/pid``.  Roll and pitch
targets are read directly from ``/crobat/desire_coordinate``.  The yaw command
from that topic is added to the PID yaw target because yaw has a world-frame
offset (the desire-coordinate yaw command is zero in these experiments).

Servo angles 4--7 are read from ``/crobat/extra_servo_cmd`` (degrees),
resampled, and differentiated.  A low-pass filter is applied to the resulting
angular velocity.  The reported statistics are ``mean(abs(angular velocity))``
and ``max(abs(angular velocity))`` in rad/s.
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import rosbag
from scipy.signal import butter, sosfiltfilt


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BAG_DIR = SCRIPT_DIR.parent / "crobat_flight"
DEFAULT_PATTERN = "2026-09-14-15-*roll_0p5to-0p5_*radpers.bag"
RATE_RE = re.compile(r"_(?P<rate>[0-9]+(?:p[0-9]+)?)radpers\.bag$")


@dataclass
class Topics:
    desire: str = "/crobat/desire_coordinate"
    final_target: str = "/crobat/final_target_baselink_rpy"
    pid: str = "/crobat/debug/pose/pid"
    position: str = "/crobat/uav/cog/odom"
    attitude: str = "/crobat/uav/baselink/odom"
    servo: str = "/crobat/extra_servo_cmd"


@dataclass
class BagSeries:
    desire: np.ndarray
    final_target: np.ndarray
    pid: np.ndarray
    position: np.ndarray
    attitude: np.ndarray
    servo: np.ndarray


@dataclass
class Result:
    rate: float
    bag: Path
    start_time: float
    end_time: float
    position_samples: int
    attitude_samples: int
    x_rmse: float
    y_rmse: float
    z_rmse: float
    position_norm_rmse: float
    roll_rmse: float
    roll_rmse_lpf: float
    pitch_rmse: float
    yaw_rmse: float
    target_rise_rate: float
    target_descent_rate: float
    servo_speed_means: Tuple[float, float, float, float]
    servo_speed_mean: float
    servo_speed_maxima: Tuple[float, float, float, float]
    servo_speed_max_mean: float


def quaternion_to_rpy(x: float, y: float, z: float, w: float) -> Tuple[float, float, float]:
    """Convert a ROS quaternion to intrinsic roll, pitch and yaw [rad]."""
    sin_roll = 2.0 * (w * x + y * z)
    cos_roll = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sin_roll, cos_roll)

    sin_pitch = 2.0 * (w * y - z * x)
    pitch = math.asin(max(-1.0, min(1.0, sin_pitch)))

    sin_yaw = 2.0 * (w * z + x * y)
    cos_yaw = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(sin_yaw, cos_yaw)
    return roll, pitch, yaw


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    """Wrap angular values to [-pi, pi)."""
    return (np.asarray(angle, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi


def sorted_array(rows: Iterable[Sequence[float]], width: int) -> np.ndarray:
    """Make a time-sorted 2-D float array, including the empty-array case."""
    values = np.asarray(list(rows), dtype=float)
    if values.size == 0:
        return np.empty((0, width), dtype=float)
    values = values.reshape((-1, width))
    return values[np.argsort(values[:, 0], kind="stable")]


def zoh(
    source_time: np.ndarray,
    source_value: np.ndarray,
    query_time: np.ndarray,
    hold_after_end: bool = False,
) -> np.ndarray:
    """Sample by zero-order hold; optionally keep the final value afterward."""
    source_time = np.asarray(source_time, dtype=float)
    source_value = np.asarray(source_value, dtype=float)
    query_time = np.asarray(query_time, dtype=float)

    valid = np.isfinite(source_time) & np.all(np.isfinite(source_value), axis=1)
    source_time = source_time[valid]
    source_value = source_value[valid]
    output = np.full((len(query_time), source_value.shape[1]), np.nan, dtype=float)
    if len(source_time) == 0:
        return output

    order = np.argsort(source_time, kind="stable")
    source_time = source_time[order]
    source_value = source_value[order]
    index = np.searchsorted(source_time, query_time, side="right") - 1
    in_range = np.isfinite(query_time) & (index >= 0)
    if not hold_after_end:
        in_range &= query_time <= source_time[-1]
    output[in_range] = source_value[index[in_range]]
    return output


def rmse(error: np.ndarray) -> float:
    error = np.asarray(error, dtype=float)
    if error.size == 0 or not np.all(np.isfinite(error)):
        return float("nan")
    return float(np.sqrt(np.mean(np.square(error))))


def servo_speed_statistics(
    servo: np.ndarray,
    start_time: float,
    end_time: float,
    cutoff_hz: float,
    filter_order: int = 4,
) -> Tuple[np.ndarray, float, np.ndarray, float, float]:
    """Return mean/maximum absolute speed after filtering angular velocity.

    ``extra_servo_cmd.angles`` is in degrees.  Its publication timing is
    bursty, so angles are first resampled with zero-order hold onto a uniform
    timebase using the average publication rate.  The angle is differentiated,
    then a zero-phase Butterworth LPF is applied to angular velocity.
    """
    time = np.asarray(servo[:, 0], dtype=float)
    angle_rad = np.deg2rad(np.asarray(servo[:, 1:5], dtype=float))
    valid = np.isfinite(time) & np.all(np.isfinite(angle_rad), axis=1)
    time = time[valid]
    angle_rad = angle_rad[valid]
    if len(time) < 3 * (filter_order + 1):
        raise ValueError("too few servo samples for low-pass filtering")
    if cutoff_hz <= 0.0:
        raise ValueError("--servo-cutoff-hz must be positive")

    order = np.argsort(time, kind="stable")
    time = time[order]
    angle_rad = angle_rad[order]

    # Keep the final command if two messages happen to have the same bag time.
    _, reverse_index = np.unique(time[::-1], return_index=True)
    keep = np.sort(len(time) - 1 - reverse_index)
    time = time[keep]
    angle_rad = angle_rad[keep]
    duration = float(time[-1] - time[0])
    if duration <= 0.0:
        raise ValueError("servo timestamps do not span a positive duration")

    sampling_hz = float((len(time) - 1) / duration)
    nyquist_hz = 0.5 * sampling_hz
    if cutoff_hz >= nyquist_hz:
        raise ValueError(
            f"--servo-cutoff-hz ({cutoff_hz:g} Hz) must be below "
            f"the servo Nyquist frequency ({nyquist_hz:.3f} Hz)"
        )

    sample_count = max(2, int(math.floor(duration * sampling_hz)) + 1)
    uniform_time = np.linspace(time[0], time[-1], sample_count)
    command_index = np.searchsorted(time, uniform_time, side="right") - 1
    uniform_angle = angle_rad[command_index]

    sos = butter(
        filter_order,
        cutoff_hz / nyquist_hz,
        btype="lowpass",
        output="sos",
    )
    angular_velocity = np.gradient(uniform_angle, uniform_time, axis=0)
    filtered_velocity = sosfiltfilt(sos, angular_velocity, axis=0)
    window = (uniform_time >= start_time) & (uniform_time <= end_time)
    if np.count_nonzero(window) < 2:
        raise ValueError("too few filtered servo samples in analysis window")

    absolute_speed = np.abs(filtered_velocity[window])
    speed_means = np.mean(absolute_speed, axis=0)
    speed_maxima = np.max(absolute_speed, axis=0)
    return (
        speed_means,
        float(np.mean(speed_means)),
        speed_maxima,
        float(np.mean(speed_maxima)),
        sampling_hz,
    )


def lowpass_irregular_angle(
    time: np.ndarray,
    angle: np.ndarray,
    cutoff_hz: float,
    filter_order: int,
) -> np.ndarray:
    """Resample an angle uniformly, apply zero-phase LPF, and map it back."""
    time = np.asarray(time, dtype=float)
    angle = np.asarray(angle, dtype=float)
    output = np.full_like(angle, np.nan)
    valid = np.isfinite(time) & np.isfinite(angle)
    if np.count_nonzero(valid) < 3 * (filter_order + 1):
        raise ValueError("too few roll samples for low-pass filtering")
    if cutoff_hz <= 0.0:
        raise ValueError("--roll-cutoff-hz must be positive")
    if filter_order <= 0:
        raise ValueError("--roll-lpf-order must be positive")

    source_time = time[valid]
    source_angle = np.unwrap(angle[valid])
    order = np.argsort(source_time, kind="stable")
    source_time = source_time[order]
    source_angle = source_angle[order]

    # Keep the final value at an exactly duplicated timestamp.
    _, reverse_index = np.unique(source_time[::-1], return_index=True)
    keep = np.sort(len(source_time) - 1 - reverse_index)
    source_time = source_time[keep]
    source_angle = source_angle[keep]
    duration = float(source_time[-1] - source_time[0])
    if duration <= 0.0:
        raise ValueError("roll timestamps do not span a positive duration")

    sampling_hz = float((len(source_time) - 1) / duration)
    nyquist_hz = 0.5 * sampling_hz
    if cutoff_hz >= nyquist_hz:
        raise ValueError(
            f"--roll-cutoff-hz ({cutoff_hz:g} Hz) must be below "
            f"the roll Nyquist frequency ({nyquist_hz:.3f} Hz)"
        )

    sample_count = max(2, int(math.floor(duration * sampling_hz)) + 1)
    uniform_time = np.linspace(source_time[0], source_time[-1], sample_count)
    uniform_angle = np.interp(uniform_time, source_time, source_angle)
    sos = butter(
        filter_order,
        cutoff_hz / nyquist_hz,
        btype="lowpass",
        output="sos",
    )
    filtered_uniform = sosfiltfilt(sos, uniform_angle)
    output[valid] = np.interp(time[valid], uniform_time, filtered_uniform)
    return wrap_angle(output)


def rate_from_filename(path: Path) -> float:
    match = RATE_RE.search(path.name)
    if match is None:
        raise ValueError(f"Could not read roll rate from filename: {path.name}")
    return float(match.group("rate").replace("p", "."))


def discover_bags(bag_dir: Path, pattern: str) -> List[Path]:
    paths = sorted(bag_dir.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No bags matched {bag_dir / pattern}")
    return paths


def read_series(path: Path, topics: Topics, servo_indices: Sequence[int]) -> BagSeries:
    desire_rows: List[Tuple[float, float, float, float]] = []
    final_target_rows: List[Tuple[float, float, float, float]] = []
    pid_rows: List[Tuple[float, float, float, float, float, float, float]] = []
    position_rows: List[Tuple[float, float, float, float]] = []
    attitude_rows: List[Tuple[float, float, float, float]] = []
    servo_rows: List[Tuple[float, float, float, float, float]] = []
    servo_state: Dict[int, float] = {}

    requested_topics = [
        topics.desire,
        topics.final_target,
        topics.pid,
        topics.position,
        topics.attitude,
        topics.servo,
    ]
    with rosbag.Bag(str(path), "r") as bag:
        for topic, msg, stamp in bag.read_messages(topics=requested_topics):
            time = stamp.to_sec()
            if topic == topics.desire:
                desire_rows.append((time, float(msg.roll), float(msg.pitch), float(msg.yaw)))
            elif topic == topics.final_target:
                final_target_rows.append(
                    (
                        time,
                        float(msg.vector.x),
                        float(msg.vector.y),
                        float(msg.vector.z),
                    )
                )
            elif topic == topics.pid:
                pid_rows.append(
                    (
                        time,
                        float(msg.x.target_p),
                        float(msg.y.target_p),
                        float(msg.z.target_p),
                        float(msg.roll.target_p),
                        float(msg.pitch.target_p),
                        float(msg.yaw.target_p),
                    )
                )
            elif topic == topics.position:
                value = msg.pose.pose.position
                position_rows.append((time, float(value.x), float(value.y), float(value.z)))
            elif topic == topics.attitude:
                value = msg.pose.pose.orientation
                roll, pitch, yaw = quaternion_to_rpy(value.x, value.y, value.z, value.w)
                attitude_rows.append((time, roll, pitch, yaw))
            elif topic == topics.servo:
                servo_state.update(
                    (int(index), float(angle))
                    for index, angle in zip(msg.index, msg.angles)
                )
                if all(index in servo_state for index in servo_indices):
                    servo_rows.append(
                        (time,) + tuple(servo_state[index] for index in servo_indices)
                    )

    series = BagSeries(
        desire=sorted_array(desire_rows, 4),
        final_target=sorted_array(final_target_rows, 4),
        pid=sorted_array(pid_rows, 7),
        position=sorted_array(position_rows, 4),
        attitude=sorted_array(attitude_rows, 4),
        servo=sorted_array(servo_rows, 5),
    )
    counts: Dict[str, int] = {
        topics.desire: len(series.desire),
        topics.final_target: len(series.final_target),
        topics.pid: len(series.pid),
        topics.position: len(series.position),
        topics.attitude: len(series.attitude),
        topics.servo: len(series.servo),
    }
    missing = [name for name, count in counts.items() if count == 0]
    if missing:
        raise ValueError(f"{path.name}: missing data on {', '.join(missing)}")
    return series


def find_analysis_window(
    desire: np.ndarray,
    final_target: np.ndarray,
    motion: str,
    target_angle: float,
    tolerance: float,
    start_offset_sec: float,
    end_offset_sec: float,
) -> Tuple[float, float]:
    desire_time = desire[:, 0]
    desire_roll = desire[:, 1]
    if motion == "rising":
        moving = np.flatnonzero(np.abs(desire_roll) > tolerance)
        if len(moving) == 0:
            raise ValueError("desire-coordinate roll never moves away from 0 rad")
        start_index = int(moving[0])
        detected_start = float(desire_time[start_index])

        positive = np.flatnonzero(
            (np.arange(len(desire_roll)) >= start_index)
            & (desire_roll >= target_angle - tolerance)
        )
        if len(positive) == 0:
            raise ValueError(
                f"desire-coordinate roll does not reach +{target_angle:g} rad"
            )
        detected_end = float(desire_time[int(positive[0])])
    elif motion == "falling":
        final_time = final_target[:, 0]
        final_roll = final_target[:, 1]
        positive = np.flatnonzero(final_roll >= target_angle - tolerance)
        if len(positive) == 0:
            raise ValueError(
                f"final target does not command +{target_angle:g} rad"
            )
        positive_index = int(positive[0])

        negative_target = np.flatnonzero(
            (np.arange(len(final_roll)) > positive_index)
            & (final_roll <= -target_angle + tolerance)
        )
        if len(negative_target) == 0:
            raise ValueError(
                f"final target does not command -{target_angle:g} rad after "
                f"+{target_angle:g} rad"
            )
        detected_start = float(final_time[int(negative_target[0])])

        negative = np.flatnonzero(
            (desire_time >= detected_start)
            & (desire_roll <= -target_angle + tolerance)
        )
        if len(negative) == 0:
            raise ValueError(
                f"desire-coordinate roll does not reach -{target_angle:g} rad "
                "after the final-target command"
            )
        detected_end = float(desire_time[int(negative[0])])
    else:
        raise ValueError(f"unknown roll motion: {motion}")

    if not np.isfinite(start_offset_sec) or not np.isfinite(end_offset_sec):
        raise ValueError("analysis-window offsets must be finite")
    start_time = detected_start + start_offset_sec
    end_time = detected_end + end_offset_sec
    if end_time <= start_time:
        raise ValueError(
            "analysis window is empty or reversed after applying offsets: "
            f"start={start_time:.6f}, end={end_time:.6f}"
        )
    return start_time, end_time


def estimate_target_descent_rate(
    desire: np.ndarray,
    final_target: np.ndarray,
    target_angle: float,
    tolerance: float,
) -> float:
    """Estimate the recorded +target -> -target slope magnitude [rad/s]."""
    start_time, end_time = find_analysis_window(
        desire,
        final_target,
        "falling",
        target_angle,
        tolerance,
        0.0,
        0.0,
    )
    in_window = (desire[:, 0] >= start_time) & (desire[:, 0] <= end_time)
    command_time = np.concatenate(([start_time], desire[in_window, 0]))
    command_roll = np.concatenate(([target_angle], desire[in_window, 1]))
    if len(command_time) < 2 or command_time[-1] <= command_time[0]:
        return float("nan")
    slope = np.polyfit(command_time - start_time, command_roll, 1)[0]
    return float(abs(slope))


def estimate_target_rise_rate(
    desire: np.ndarray,
    final_target: np.ndarray,
    target_angle: float,
    tolerance: float,
) -> float:
    """Estimate the recorded 0 -> +target slope magnitude [rad/s]."""
    start_time, end_time = find_analysis_window(
        desire,
        final_target,
        "rising",
        target_angle,
        tolerance,
        0.0,
        0.0,
    )
    in_window = (desire[:, 0] >= start_time) & (desire[:, 0] <= end_time)
    command_time = desire[in_window, 0]
    command_roll = desire[in_window, 1]
    if len(command_time) < 2 or command_time[-1] <= command_time[0]:
        return float("nan")
    slope = np.polyfit(command_time - command_time[0], command_roll, 1)[0]
    return float(abs(slope))


def analyze_series(
    path: Path,
    series: BagSeries,
    motion: str,
    servo_indices: Sequence[int],
    servo_cutoff_hz: float,
    roll_cutoff_hz: float,
    roll_lpf_order: int,
    target_angle: float,
    tolerance: float,
    start_offset_sec: float,
    end_offset_sec: float,
) -> Result:
    start_time, end_time = find_analysis_window(
        series.desire,
        series.final_target,
        motion,
        target_angle,
        tolerance,
        start_offset_sec,
        end_offset_sec,
    )

    position_mask = (
        (series.position[:, 0] >= start_time)
        & (series.position[:, 0] <= end_time)
    )
    position_actual = series.position[position_mask]
    position_target = zoh(series.pid[:, 0], series.pid[:, 1:4], position_actual[:, 0])
    position_valid = np.all(np.isfinite(position_actual[:, 1:4]), axis=1) & np.all(
        np.isfinite(position_target), axis=1
    )
    position_error = position_actual[position_valid, 1:4] - position_target[position_valid]
    if len(position_error) == 0:
        raise ValueError(f"{path.name}: no synchronized position samples in analysis window")

    attitude_mask = (
        (series.attitude[:, 0] >= start_time)
        & (series.attitude[:, 0] <= end_time)
    )
    attitude_actual = series.attitude[attitude_mask]
    roll_actual_lpf_all = lowpass_irregular_angle(
        series.attitude[:, 0],
        series.attitude[:, 1],
        roll_cutoff_hz,
        roll_lpf_order,
    )
    roll_actual_lpf = roll_actual_lpf_all[attitude_mask]
    desire_time = series.desire[:, 0]
    desire_value = series.desire[:, 1:4]
    if motion == "rising" and start_time < desire_time[0]:
        desire_time = np.concatenate(([start_time], desire_time))
        desire_value = np.vstack((np.zeros((1, 3)), desire_value))
    desire_target = zoh(
        desire_time,
        desire_value,
        attitude_actual[:, 0],
        hold_after_end=True,
    )
    pid_yaw_target = zoh(series.pid[:, 0], series.pid[:, 6:7], attitude_actual[:, 0])
    attitude_target = desire_target.copy()
    attitude_target[:, 2] += pid_yaw_target[:, 0]
    attitude_target = wrap_angle(attitude_target)
    attitude_valid = (
        np.all(np.isfinite(attitude_actual[:, 1:4]), axis=1)
        & np.all(np.isfinite(attitude_target), axis=1)
        & np.isfinite(roll_actual_lpf)
    )
    attitude_error = wrap_angle(
        attitude_actual[attitude_valid, 1:4] - attitude_target[attitude_valid]
    )
    roll_error_lpf = wrap_angle(
        roll_actual_lpf[attitude_valid] - attitude_target[attitude_valid, 0]
    )
    if len(attitude_error) == 0:
        raise ValueError(f"{path.name}: no synchronized attitude samples in analysis window")

    (
        servo_speed_means,
        servo_speed_mean,
        servo_speed_maxima,
        servo_speed_max_mean,
        _,
    ) = servo_speed_statistics(
        series.servo,
        start_time,
        end_time,
        servo_cutoff_hz,
    )
    position_norm = np.linalg.norm(position_error, axis=1)
    target_rise_rate = (
        estimate_target_rise_rate(
            series.desire,
            series.final_target,
            target_angle,
            tolerance,
        )
        if motion == "rising"
        else float("nan")
    )
    target_descent_rate = (
        estimate_target_descent_rate(
            series.desire,
            series.final_target,
            target_angle,
            tolerance,
        )
        if motion == "falling"
        else float("nan")
    )
    return Result(
        rate=rate_from_filename(path),
        bag=path,
        start_time=start_time,
        end_time=end_time,
        position_samples=len(position_error),
        attitude_samples=len(attitude_error),
        x_rmse=rmse(position_error[:, 0]),
        y_rmse=rmse(position_error[:, 1]),
        z_rmse=rmse(position_error[:, 2]),
        position_norm_rmse=rmse(position_norm),
        roll_rmse=rmse(attitude_error[:, 0]),
        roll_rmse_lpf=rmse(roll_error_lpf),
        pitch_rmse=rmse(attitude_error[:, 1]),
        yaw_rmse=rmse(attitude_error[:, 2]),
        target_rise_rate=target_rise_rate,
        target_descent_rate=target_descent_rate,
        servo_speed_means=tuple(float(value) for value in servo_speed_means),
        servo_speed_mean=servo_speed_mean,
        servo_speed_maxima=tuple(float(value) for value in servo_speed_maxima),
        servo_speed_max_mean=servo_speed_max_mean,
    )


def plot_results(
    results: Sequence[Result],
    servo_indices: Sequence[int],
    servo_cutoff_hz: float,
    window_title: str,
    target_rate_mode: str = "",
    show_legend: bool = True,
    line_width: float = 2.0,
    marker_size: float = 6.0,
) -> None:
    rate = np.asarray([result.rate for result in results])
    show_target_rate = target_rate_mode in ("rising", "falling")
    row_count = 5 if show_target_rate else 4
    figure_height = 15.5 if show_target_rate else 13.0
    fig, axes = plt.subplots(
        row_count,
        1,
        figsize=(7.2, figure_height),
        sharex=True,
    )

    axes[0].plot(
        rate,
        [r.x_rmse for r in results],
        "o-",
        linewidth=line_width,
        markersize=marker_size,
        label="x",
    )
    axes[0].plot(
        rate,
        [r.y_rmse for r in results],
        "s-",
        linewidth=line_width,
        markersize=marker_size,
        label="y",
    )
    axes[0].plot(
        rate,
        [r.z_rmse for r in results],
        "^-",
        linewidth=line_width,
        markersize=marker_size,
        label="z",
    )
    axes[0].plot(
        rate,
        [r.position_norm_rmse for r in results],
        "D-",
        linewidth=line_width,
        markersize=marker_size,
        label=r"$\|e_{pos}\|$",
    )
    axes[0].set_ylabel("RMSE [m]")
    if show_legend:
        axes[0].legend(ncol=4)

    axes[1].plot(
        rate,
        [r.roll_rmse for r in results],
        "o-",
        linewidth=line_width,
        markersize=marker_size,
        label="roll",
    )
    axes[1].plot(
        rate,
        [r.pitch_rmse for r in results],
        "s-",
        linewidth=line_width,
        markersize=marker_size,
        label="pitch",
    )
    axes[1].plot(
        rate,
        [r.yaw_rmse for r in results],
        "^-",
        linewidth=line_width,
        markersize=marker_size,
        label="yaw",
    )
    axes[1].set_ylabel("RMSE [rad]")
    if show_legend:
        axes[1].legend(ncol=3)

    markers = ["o", "s", "^", "v"]
    for column, (index, marker) in enumerate(zip(servo_indices, markers)):
        axes[2].plot(
            rate,
            [result.servo_speed_means[column] for result in results],
            marker + "-",
            linewidth=line_width,
            markersize=marker_size,
            label=f"servo {index}",
        )
    axes[2].plot(
        rate,
        [result.servo_speed_mean for result in results],
        "D-",
        linewidth=line_width,
        markersize=marker_size,
        label="4-servo mean",
    )
    axes[2].set_ylabel(r"Mean $|\dot{\theta}|$ [rad/s]")
    if show_legend:
        axes[2].legend(ncol=3)
    axes[2].set_title(f"Servo angular-velocity LPF cutoff: {servo_cutoff_hz:g} Hz")

    for column, (index, marker) in enumerate(zip(servo_indices, markers)):
        axes[3].plot(
            rate,
            [result.servo_speed_maxima[column] for result in results],
            marker + "-",
            linewidth=line_width,
            markersize=marker_size,
            label=f"servo {index}",
        )
    axes[3].plot(
        rate,
        [result.servo_speed_max_mean for result in results],
        "D-",
        linewidth=line_width,
        markersize=marker_size,
        label="4-servo mean",
    )
    axes[3].set_ylabel(r"Max $|\dot{\theta}|$ [rad/s]")
    if show_legend:
        axes[3].legend(ncol=3)

    if show_target_rate:
        if target_rate_mode == "rising":
            measured_rate = np.asarray(
                [result.target_rise_rate for result in results],
                dtype=float,
            )
            rate_label = "recorded target rise rate"
        else:
            measured_rate = np.asarray(
                [result.target_descent_rate for result in results],
                dtype=float,
            )
            rate_label = "recorded target descent rate"
        axes[4].plot(
            rate,
            measured_rate,
            "o-",
            linewidth=line_width,
            markersize=marker_size,
            label=rate_label,
        )
        axes[4].plot(
            rate,
            rate,
            "--",
            linewidth=line_width,
            color="black",
            label="nominal (y=x)",
        )
        axes[4].set_ylabel(r"Target $|\dot{roll}|$ [rad/s]")
        if show_legend:
            axes[4].legend(ncol=2)
        bottom_axis = axes[4]
    else:
        bottom_axis = axes[3]

    bottom_axis.set_xlabel("Nominal roll command rate [rad/s]")
    bottom_axis.set_xticks(rate)

    fig.suptitle(window_title)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))


def plot_roll_lpf_comparison(
    results: Sequence[Result],
    cutoff_hz: float,
    filter_order: int,
    window_title: str,
    show_legend: bool = True,
    line_width: float = 2.0,
    marker_size: float = 6.0,
) -> None:
    rate = np.asarray([result.rate for result in results])
    fig, axis = plt.subplots(figsize=(7.2, 5.0))
    axis.plot(
        rate,
        [result.roll_rmse_lpf for result in results],
        "o-",
        linewidth=line_width,
        markersize=marker_size,
        label="LPF roll RMSE",
    )
    axis.set_xlabel("Nominal roll command rate [rad/s]")
    axis.set_ylabel("Roll RMSE [rad]")
    axis.set_xticks(rate)
    if show_legend:
        axis.legend()
    axis.set_title(
        f"{window_title}\n"
        f"Actual-roll LPF: {cutoff_hz:g} Hz, order {filter_order}"
    )
    fig.tight_layout()


def plot_roll_rmse_vs_mean_servo_speed(
    results: Sequence[Result],
    cutoff_hz: float,
    filter_order: int,
    window_title: str,
    show_legend: bool = True,
    line_width: float = 2.0,
    marker_size: float = 6.0,
) -> None:
    """Plot roll tracking error against measured mean servo angular speed."""
    ordered_results = sorted(results, key=lambda result: result.servo_speed_mean)
    mean_speed = np.asarray(
        [result.servo_speed_mean for result in ordered_results],
        dtype=float,
    )

    fig, axis = plt.subplots(figsize=(7.2, 5.0))
    axis.plot(
        mean_speed,
        [result.roll_rmse for result in ordered_results],
        "o--",
        linewidth=line_width,
        markersize=marker_size,
        label="raw roll RMSE",
    )
    axis.plot(
        mean_speed,
        [result.roll_rmse_lpf for result in ordered_results],
        "s-",
        linewidth=line_width,
        markersize=marker_size,
        label="LPF roll RMSE",
    )
    axis.set_xlabel(r"4-servo mean $|\dot{\theta}|$ [rad/s]")
    axis.set_ylabel("Roll RMSE [rad]")
    if show_legend:
        axis.legend()
    axis.set_title(
        f"{window_title}\n"
        f"Actual-roll LPF: {cutoff_hz:g} Hz, order {filter_order}"
    )
    fig.tight_layout()


def print_results(
    results: Sequence[Result],
    servo_indices: Sequence[int],
    servo_cutoff_hz: float,
) -> None:
    header = (
        "rate  duration  n_pos  n_att      x        y        z      pos_norm"
        "    roll  roll_lpf    pitch      yaw"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        print(
            f"{result.rate:4.1f}  {result.end_time - result.start_time:8.3f}"
            f"  {result.position_samples:5d}  {result.attitude_samples:5d}"
            f"  {result.x_rmse:7.4f}  {result.y_rmse:7.4f}"
            f"  {result.z_rmse:7.4f}  {result.position_norm_rmse:8.4f}"
            f"  {result.roll_rmse:7.4f}  {result.roll_rmse_lpf:8.4f}"
            f"  {result.pitch_rmse:8.4f}"
            f"  {result.yaw_rmse:7.4f}"
        )
    print("position: m, attitude: rad")
    print()
    labels = "  ".join(f"servo_{index:02d}" for index in servo_indices)
    print(
        f"Mean absolute servo angular speed [rad/s], "
        f"velocity LPF={servo_cutoff_hz:g} Hz"
    )
    print(f"rate  {labels}  4-servo_mean")
    for result in results:
        values = "  ".join(f"{value:8.4f}" for value in result.servo_speed_means)
        print(f"{result.rate:4.1f}  {values}  {result.servo_speed_mean:12.4f}")
    print()
    print(
        f"Maximum absolute servo angular speed [rad/s], "
        f"velocity LPF={servo_cutoff_hz:g} Hz"
    )
    print(f"rate  {labels}  4-servo_mean")
    for result in results:
        values = "  ".join(f"{value:8.4f}" for value in result.servo_speed_maxima)
        print(f"{result.rate:4.1f}  {values}  {result.servo_speed_max_mean:12.4f}")

    if any(np.isfinite(result.target_rise_rate) for result in results):
        print()
        print("Recorded target rise rate [rad/s]")
        print("nominal  recorded")
        for result in results:
            print(f"{result.rate:7.1f}  {result.target_rise_rate:8.4f}")

    if any(np.isfinite(result.target_descent_rate) for result in results):
        print()
        print("Recorded target descent rate [rad/s]")
        print("nominal  recorded")
        for result in results:
            print(f"{result.rate:7.1f}  {result.target_descent_rate:8.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate x/y/z/RPY tracking RMSE during roll sweeps and plot it against rad/s."
    )
    parser.add_argument(
        "bags",
        nargs="*",
        type=Path,
        help="input bag paths; if omitted, the five 2026-09-14 bags are discovered automatically",
    )
    parser.add_argument("--bag-dir", type=Path, default=DEFAULT_BAG_DIR)
    parser.add_argument("--pattern", default=DEFAULT_PATTERN)
    parser.add_argument("--target-angle", type=float, default=0.5, help="roll limit [rad]")
    parser.add_argument(
        "--angle-tolerance",
        type=float,
        default=0.001,
        help="tolerance used to detect +/- target angle [rad]",
    )
    parser.add_argument(
        "--start-offset-sec",
        type=float,
        default=0.0,
        help="seconds added to the detected start time; negative moves it earlier (default: 0)",
    )
    parser.add_argument(
        "--end-offset-sec",
        type=float,
        default=0.0,
        help="seconds added to the detected end time; negative moves it earlier (default: 0)",
    )
    parser.add_argument("--desire-topic", default=Topics.desire)
    parser.add_argument("--final-target-topic", default=Topics.final_target)
    parser.add_argument("--pid-topic", default=Topics.pid)
    parser.add_argument("--position-topic", default=Topics.position)
    parser.add_argument("--attitude-topic", default=Topics.attitude)
    parser.add_argument("--servo-topic", default=Topics.servo)
    parser.add_argument(
        "--servo-indices",
        nargs=4,
        type=int,
        default=(4, 5, 6, 7),
        metavar=("S1", "S2", "S3", "S4"),
        help="four indices in extra_servo_cmd (default: 4 5 6 7)",
    )
    parser.add_argument(
        "--servo-cutoff-hz",
        type=float,
        default=5.0,
        help="cutoff frequency of the fourth-order zero-phase servo-velocity LPF (default: 5 Hz)",
    )
    parser.add_argument(
        "--roll-cutoff-hz",
        type=float,
        default=5.0,
        help="cutoff frequency of the zero-phase actual-roll LPF (default: 5 Hz)",
    )
    parser.add_argument(
        "--roll-lpf-order",
        type=int,
        default=4,
        help="Butterworth order of the actual-roll LPF (default: 4)",
    )
    parser.add_argument(
        "--line-width",
        type=float,
        default=2.0,
        help="line width used in all plots (default: 2.0)",
    )
    parser.add_argument(
        "--marker-size",
        type=float,
        default=6.0,
        help="marker size used in all plots (default: 6.0)",
    )
    legend_group = parser.add_mutually_exclusive_group()
    legend_group.add_argument(
        "--legend",
        dest="show_legend",
        action="store_true",
        default=True,
        help="show plot legends (default)",
    )
    legend_group.add_argument(
        "--no-legend",
        dest="show_legend",
        action="store_false",
        help="hide plot legends",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = list(args.bags) if args.bags else discover_bags(args.bag_dir, args.pattern)
    paths = [path.expanduser().resolve() for path in paths]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Bag file not found: " + ", ".join(missing))

    topics = Topics(
        desire=args.desire_topic,
        final_target=args.final_target_topic,
        pid=args.pid_topic,
        position=args.position_topic,
        attitude=args.attitude_topic,
        servo=args.servo_topic,
    )
    if len(set(args.servo_indices)) != 4:
        raise ValueError("--servo-indices must contain four different indices")
    if not np.isfinite(args.line_width) or args.line_width <= 0.0:
        raise ValueError("--line-width must be a positive finite number")
    if not np.isfinite(args.marker_size) or args.marker_size <= 0.0:
        raise ValueError("--marker-size must be a positive finite number")
    series_by_path = [
        (path, read_series(path, topics, args.servo_indices))
        for path in paths
    ]

    rising_results = [
        analyze_series(
            path,
            series,
            "rising",
            args.servo_indices,
            args.servo_cutoff_hz,
            args.roll_cutoff_hz,
            args.roll_lpf_order,
            args.target_angle,
            args.angle_tolerance,
            args.start_offset_sec,
            args.end_offset_sec,
        )
        for path, series in series_by_path
    ]
    falling_results = [
        analyze_series(
            path,
            series,
            "falling",
            args.servo_indices,
            args.servo_cutoff_hz,
            args.roll_cutoff_hz,
            args.roll_lpf_order,
            args.target_angle,
            args.angle_tolerance,
            args.start_offset_sec,
            args.end_offset_sec,
        )
        for path, series in series_by_path
    ]
    rising_results.sort(key=lambda result: result.rate)
    falling_results.sort(key=lambda result: result.rate)
    rates = [result.rate for result in rising_results]
    if len(rates) != len(set(rates)):
        raise ValueError(f"Duplicate roll rates parsed from bag names: {rates}")

    print(
        f"analysis offsets: start={args.start_offset_sec:+.3f} s, "
        f"end={args.end_offset_sec:+.3f} s"
    )
    print(
        f"actual-roll LPF: cutoff={args.roll_cutoff_hz:g} Hz, "
        f"order={args.roll_lpf_order}"
    )
    print("\n=== Rising interval: 0 -> +target ===")
    print_results(rising_results, args.servo_indices, args.servo_cutoff_hz)
    print("\n=== Falling interval: +target -> -target ===")
    print_results(falling_results, args.servo_indices, args.servo_cutoff_hz)

    plot_results(
        rising_results,
        args.servo_indices,
        args.servo_cutoff_hz,
        "Roll command: 0 to +0.5 rad",
        target_rate_mode="rising",
        show_legend=args.show_legend,
        line_width=args.line_width,
        marker_size=args.marker_size,
    )
    plot_results(
        falling_results,
        args.servo_indices,
        args.servo_cutoff_hz,
        "Roll command: +0.5 to -0.5 rad",
        target_rate_mode="falling",
        show_legend=args.show_legend,
        line_width=args.line_width,
        marker_size=args.marker_size,
    )
    plot_roll_lpf_comparison(
        rising_results,
        args.roll_cutoff_hz,
        args.roll_lpf_order,
        "Roll command: 0 to +0.5 rad",
        show_legend=args.show_legend,
        line_width=args.line_width,
        marker_size=args.marker_size,
    )
    plot_roll_lpf_comparison(
        falling_results,
        args.roll_cutoff_hz,
        args.roll_lpf_order,
        "Roll command: +0.5 to -0.5 rad",
        show_legend=args.show_legend,
        line_width=args.line_width,
        marker_size=args.marker_size,
    )
    plot_roll_rmse_vs_mean_servo_speed(
        rising_results,
        args.roll_cutoff_hz,
        args.roll_lpf_order,
        "Roll command: 0 to +0.5 rad",
        show_legend=args.show_legend,
        line_width=args.line_width,
        marker_size=args.marker_size,
    )
    plot_roll_rmse_vs_mean_servo_speed(
        falling_results,
        args.roll_cutoff_hz,
        args.roll_lpf_order,
        "Roll command: +0.5 to -0.5 rad",
        show_legend=args.show_legend,
        line_width=args.line_width,
        marker_size=args.marker_size,
    )
    plt.show()


if __name__ == "__main__":
    main()
