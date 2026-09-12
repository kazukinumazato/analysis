#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rosbag
import numpy as np
import argparse
import sys
import math


def compute_stats(values):
    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        return {
            "count": 0,
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "median": np.nan,
            "rms": np.nan,
            "p95": np.nan,
        }

    return {
        "count": len(values),
        "mean": np.mean(values),
        "std": np.std(values),
        "min": np.min(values),
        "max": np.max(values),
        "median": np.median(values),
        "rms": np.sqrt(np.mean(values ** 2)),
        "p95": np.percentile(values, 95),
    }


def quaternion_to_roll(x, y, z, w):
    """
    quaternion -> roll [rad]
    ROS geometry_msgs/Quaternion: x, y, z, w
    """
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    return math.atan2(sinr_cosp, cosr_cosp)


def angle_diff_rad(a, b):
    """
    wrapped angle difference a - b in [-pi, pi]
    """
    d = a - b
    return math.atan2(math.sin(d), math.cos(d))


def print_stats_block(title, stats):
    print(title)
    print("count     : {}".format(stats["count"]))
    print("mean      : {:.6f}".format(stats["mean"]))
    print("std       : {:.6f}".format(stats["std"]))
    print("min       : {:.6f}".format(stats["min"]))
    print("max       : {:.6f}".format(stats["max"]))
    print("median    : {:.6f}".format(stats["median"]))
    print("rms       : {:.6f}".format(stats["rms"]))
    print("p95       : {:.6f}".format(stats["p95"]))
    print("")


def print_candidate_line(name, value, force_ratio, T_f, epsilon=None):
    eps_req = (T_f * value / force_ratio) if force_ratio > 0.0 else np.nan
    print("[{}]".format(name))
    print("theta_dot_candidate : {:.6f} rad/s".format(value))
    print("required epsilon    : {:.6f}".format(eps_req))
    if epsilon is not None:
        print("satisfy cond.       : {}".format("YES" if eps_req <= epsilon else "NO"))
    print("")


def main():
    parser = argparse.ArgumentParser(
        description="Estimate representative theta_dot_max from cycle-wise peak roll angular velocity."
    )
    parser.add_argument("bag_path", help="Path to input rosbag")
    parser.add_argument(
        "--topic",
        default="/crobat/mocap_node/mocap/servo/pose",
        help="PoseStamped topic name"
    )
    parser.add_argument(
        "--flap_freq",
        type=float,
        default=20.0,
        help="Flapping frequency [Hz] (default: 20.0)"
    )
    parser.add_argument(
        "--force_ratio",
        type=float,
        required=True,
        help="||f_i0|| / F_max (e.g. 0.45 ~ 0.65)"
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=None,
        help="Optional prescribed relative accuracy epsilon"
    )
    parser.add_argument(
        "--max_abs_vel",
        type=float,
        default=None,
        help="Optional outlier threshold for absolute angular velocity [rad/s]"
    )

    args = parser.parse_args()

    bag_path = args.bag_path
    topic_name = args.topic
    flap_freq = args.flap_freq
    force_ratio = args.force_ratio
    epsilon = args.epsilon
    max_abs_vel = args.max_abs_vel

    if flap_freq <= 0.0:
        print("Error: flap_freq must be positive.")
        sys.exit(1)

    if force_ratio <= 0.0:
        print("Error: force_ratio must be positive.")
        sys.exit(1)

    if epsilon is not None and epsilon <= 0.0:
        print("Error: epsilon must be positive.")
        sys.exit(1)

    T_f = 1.0 / flap_freq
    vel_threshold = None
    if epsilon is not None:
        vel_threshold = (epsilon / T_f) * force_ratio

    prev_time = None
    prev_roll = None
    t0 = None

    abs_vel_history = []

    cycle_abs_vels = []
    current_cycle_index = None
    current_cycle_values = []

    try:
        bag = rosbag.Bag(bag_path, "r")
    except Exception as e:
        print("Failed to open bag: {}".format(e))
        sys.exit(1)

    msg_count = 0
    vel_count = 0

    with bag:
        for _, msg, t in bag.read_messages(topics=[topic_name]):
            msg_count += 1
            current_time = t.to_sec()

            try:
                q = msg.pose.orientation
                roll = quaternion_to_roll(q.x, q.y, q.z, q.w)
            except Exception:
                continue

            if t0 is None:
                t0 = current_time

            if prev_time is None:
                prev_time = current_time
                prev_roll = roll
                continue

            dt = current_time - prev_time
            if dt <= 0.0:
                prev_time = current_time
                prev_roll = roll
                continue

            droll = angle_diff_rad(roll, prev_roll)
            abs_vel = abs(droll / dt)

            if max_abs_vel is not None and abs_vel > max_abs_vel:
                prev_time = current_time
                prev_roll = roll
                continue

            abs_vel_history.append(abs_vel)

            elapsed = current_time - t0
            cycle_index = int(elapsed / T_f)

            if current_cycle_index is None:
                current_cycle_index = cycle_index

            if cycle_index != current_cycle_index:
                if len(current_cycle_values) > 0:
                    cycle_abs_vels.append(current_cycle_values)
                current_cycle_values = []
                current_cycle_index = cycle_index

            current_cycle_values.append(abs_vel)

            vel_count += 1
            prev_time = current_time
            prev_roll = roll

    if len(current_cycle_values) > 0:
        cycle_abs_vels.append(current_cycle_values)

    abs_vel_history = np.asarray(abs_vel_history, dtype=float)

    overall_stats = compute_stats(abs_vel_history)

    cycle_max_list = []
    cycle_mean_list = []
    cycle_rms_list = []

    for vals in cycle_abs_vels:
        vals = np.asarray(vals, dtype=float)
        if len(vals) == 0:
            continue
        cycle_max_list.append(np.max(vals))
        cycle_mean_list.append(np.mean(vals))
        cycle_rms_list.append(np.sqrt(np.mean(vals ** 2)))

    cycle_max_list = np.asarray(cycle_max_list, dtype=float)
    cycle_mean_list = np.asarray(cycle_mean_list, dtype=float)
    cycle_rms_list = np.asarray(cycle_rms_list, dtype=float)

    cycle_max_stats = compute_stats(cycle_max_list)
    cycle_mean_stats = compute_stats(cycle_mean_list)
    cycle_rms_stats = compute_stats(cycle_rms_list)

    # representative theta_dot_max candidates estimated from cycle-wise peaks
    theta_dot_global_max = overall_stats["max"]
    theta_dot_cyclemax_mean = cycle_max_stats["mean"]
    theta_dot_cyclemax_median = cycle_max_stats["median"]
    theta_dot_cyclemax_p95 = cycle_max_stats["p95"]

    print("")
    print("======================================================")
    print("Representative theta_dot_max estimation from cycle peaks")
    print("======================================================")
    print("bag file           : {}".format(bag_path))
    print("topic              : {}".format(topic_name))
    print("msg count          : {}".format(msg_count))
    print("vel samples        : {}".format(vel_count))
    print("flap frequency     : {:.3f} Hz".format(flap_freq))
    print("flap period        : {:.6f} s".format(T_f))
    print("cycle count        : {}".format(len(cycle_abs_vels)))
    print("force ratio        : {:.6f}".format(force_ratio))
    if epsilon is not None:
        print("epsilon            : {:.6f}".format(epsilon))
        print("vel threshold      : {:.6f} rad/s".format(vel_threshold))
    if max_abs_vel is not None:
        print("outlier threshold  : {:.6f} rad/s".format(max_abs_vel))
    print("unit               : rad/s")
    print("======================================================")
    print("")

    print_stats_block("---- absolute roll angular velocity (all samples) ----", overall_stats)
    print_stats_block("---- per-cycle max(|roll angular velocity|) ----", cycle_max_stats)
    print_stats_block("---- per-cycle mean(|roll angular velocity|) ----", cycle_mean_stats)
    print_stats_block("---- per-cycle rms(|roll angular velocity|) ----", cycle_rms_stats)

    print("---- representative theta_dot_max candidates ----")
    print_candidate_line(
        "global max of all samples (most conservative reference)",
        theta_dot_global_max, force_ratio, T_f, epsilon
    )
    print_candidate_line(
        "mean of cycle-wise maxima",
        theta_dot_cyclemax_mean, force_ratio, T_f, epsilon
    )
    print_candidate_line(
        "median of cycle-wise maxima (robust representative peak)",
        theta_dot_cyclemax_median, force_ratio, T_f, epsilon
    )
    print_candidate_line(
        "p95 of cycle-wise maxima (moderately conservative peak)",
        theta_dot_cyclemax_p95, force_ratio, T_f, epsilon
    )

    if epsilon is not None:
        print("---- interpretation guide ----")
        print("Use 'global max'  : if you want a strict worst-case reference.")
        print("Use 'cycle-max median' : if you want a robust representative peak under noisy differentiation.")
        print("Use 'cycle-max p95'    : if you want a conservative but not spike-dominated estimate.")
        print("")


if __name__ == "__main__":
    main()
