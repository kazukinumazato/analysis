#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rosbag
import numpy as np
import csv
import argparse
import sys
import math


def compute_stats(values):
    if len(values) == 0:
        return {
            "count": 0,
            "mean": np.nan,
            "mean_abs": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "median": np.nan,
            "rms": np.nan,
        }

    return {
        "count": len(values),
        "mean": np.mean(values),
        "mean_abs": np.mean(np.abs(values)),
        "std": np.std(values),
        "min": np.min(values),
        "max": np.max(values),
        "median": np.median(values),
        "rms": np.sqrt(np.mean(values ** 2)),
    }


def quaternion_to_roll(x, y, z, w):
    """
    quaternion -> roll [rad]
    ROS geometry_msgs/Quaternion: x, y, z, w
    """
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    return roll


def angle_diff_rad(a, b):
    """
    return wrapped angle difference a - b in [-pi, pi]
    """
    d = a - b
    return math.atan2(math.sin(d), math.cos(d))


def main():
    parser = argparse.ArgumentParser(
        description="Compute roll angular velocity (rad/s) from PoseStamped in a ROS1 bag."
    )
    parser.add_argument("bag_path", help="Path to input rosbag")
    parser.add_argument(
        "--topic",
        default="/crobat/mocap_node/mocap/servo/pose",
        help="PoseStamped topic name"
    )
    parser.add_argument(
        "--csv",
        default="",
        help="Optional CSV output path"
    )
    parser.add_argument(
        "--max_abs_vel",
        type=float,
        default=None,
        help="Outlier threshold in rad/s"
    )

    args = parser.parse_args()

    bag_path = args.bag_path
    topic_name = args.topic
    csv_path = args.csv
    max_abs_vel = args.max_abs_vel

    prev_time = None
    prev_roll = None

    vel_history = []
    rows = []

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

            if prev_time is None:
                prev_time = current_time
                prev_roll = roll
                continue

            dt = current_time - prev_time

            if dt <= 0.0:
                prev_time = current_time
                prev_roll = roll
                continue

            # roll差分を -pi ~ pi に折りたたんで不連続を回避
            droll = angle_diff_rad(roll, prev_roll)

            # rad/s
            vel = droll / dt

            if max_abs_vel is not None:
                if abs(vel) > max_abs_vel:
                    prev_time = current_time
                    prev_roll = roll
                    continue

            vel_history.append(vel)

            rows.append([
                current_time,
                dt,
                roll,
                droll,
                vel
            ])

            vel_count += 1
            prev_time = current_time
            prev_roll = roll

    vel_history = np.array(vel_history, dtype=float)

    print("")
    print("========================================")
    print("Roll angular velocity statistics")
    print("========================================")
    print("bag file     : {}".format(bag_path))
    print("topic        : {}".format(topic_name))
    print("msg count    : {}".format(msg_count))
    print("vel samples  : {}".format(vel_count))
    if max_abs_vel is not None:
        print("outlier thr  : {} rad/s".format(max_abs_vel))
    print("unit         : rad/s")
    print("========================================")
    print("")

    stats = compute_stats(vel_history)

    print("---- roll angular velocity ----")
    print("count     : {}".format(stats["count"]))
    print("mean      : {:.6f}".format(stats["mean"]))
    print("mean_abs  : {:.6f}".format(stats["mean_abs"]))
    print("std       : {:.6f}".format(stats["std"]))
    print("min       : {:.6f}".format(stats["min"]))
    print("max       : {:.6f}".format(stats["max"]))
    print("median    : {:.6f}".format(stats["median"]))
    print("rms       : {:.6f}".format(stats["rms"]))
    print("")

    if csv_path:
        try:
            with open(csv_path, "w") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "time",
                    "dt",
                    "roll_rad",
                    "droll_rad",
                    "roll_vel_rad_s"
                ])
                writer.writerows(rows)

            print("Saved CSV: {}".format(csv_path))
        except Exception as e:
            print("Failed to save CSV: {}".format(e))


if __name__ == "__main__":
    main()
