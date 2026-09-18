import argparse
import rosbag
import matplotlib.pyplot as plt
import math
import numpy as np
from tf.transformations import euler_from_quaternion
from scipy.signal import butter, filtfilt


parser = argparse.ArgumentParser()
parser.add_argument("bag_path", help="解析するrosbagファイルのパス")
parser.add_argument(
    "--trim-start",
    type=float,
    default=15.0,
    help="この相対時刻より前をプロットから除外する [s] (default: 15.0)",
)
args = parser.parse_args()

bag_path = args.bag_path

pid_topic = "/crobat/debug/pose/pid"
odom_topic = "/crobat/uav/baselink/odom"
desire_topic = "/crobat/desire_coordinate"

t_pid = []
x_target = []
y_target = []
z_target = []
roll_target_pid = []
pitch_target_pid = []
yaw_target_pid = []

t_odom = []
x_actual = []
y_actual = []
z_actual = []
roll_actual = []
pitch_actual = []
yaw_actual = []

t_roll_target = []
roll_target = []
t_pitch_target = []
pitch_target = []
t_yaw_target = []
yaw_target = []


def butter_lowpass_filter(data, time, cutoff_hz=0.5, order=4):
    if len(data) < 2:
        return data

    data = np.asarray(data)
    time = np.asarray(time)

    dt = np.diff(time)
    dt_mean = np.mean(dt)
    fs = 1.0 / dt_mean
    nyq = 0.5 * fs

    if cutoff_hz >= nyq:
        raise ValueError(
            f"cutoff_hz={cutoff_hz} Hz is too high. Nyquist frequency is {nyq:.3f} Hz"
        )

    wn = cutoff_hz / nyq
    b, a = butter(order, wn, btype="low")
    return filtfilt(b, a, data)


def interpolate_linear(time, data, query_time):
    time = np.asarray(time, dtype=float)
    data = np.asarray(data, dtype=float)
    query_time = np.asarray(query_time, dtype=float)
    valid = np.isfinite(time) & np.isfinite(data)
    if np.count_nonzero(valid) < 2:
        return np.full(query_time.shape, np.nan, dtype=float)

    time = time[valid]
    data = data[valid]
    order = np.argsort(time, kind="stable")
    return np.interp(query_time, time[order], data[order])


def interpolate_previous(time, data, query_time):
    time = np.asarray(time, dtype=float)
    data = np.asarray(data, dtype=float)
    query_time = np.asarray(query_time, dtype=float)
    valid = np.isfinite(time) & np.isfinite(data)
    if np.count_nonzero(valid) == 0:
        return np.full(query_time.shape, np.nan, dtype=float)

    time = time[valid]
    data = data[valid]
    order = np.argsort(time, kind="stable")
    time = time[order]
    data = data[order]
    indices = np.searchsorted(time, query_time, side="right") - 1
    # Before the first command, use the first available target, as in the plot.
    indices = np.clip(indices, 0, len(data) - 1)
    return data[indices]


def combine_attitude_target(
    pid_time,
    pid_target,
    desire_time,
    desire_target,
    query_time,
):
    query_time = np.asarray(query_time, dtype=float)
    combined = np.zeros(query_time.shape, dtype=float)
    has_target = False

    if len(pid_target) > 0:
        combined += interpolate_previous(pid_time, pid_target, query_time)
        has_target = True
    if len(desire_target) > 0:
        combined += interpolate_previous(desire_time, desire_target, query_time)
        has_target = True

    if not has_target:
        return np.full(query_time.shape, np.nan, dtype=float)
    return np.arctan2(np.sin(combined), np.cos(combined))


def calculate_rmse(target, actual, angular=False):
    target = np.asarray(target, dtype=float)
    actual = np.asarray(actual, dtype=float)
    valid = np.isfinite(target) & np.isfinite(actual)
    if np.count_nonzero(valid) == 0:
        return float("nan")

    error = actual[valid] - target[valid]
    if angular:
        error = np.arctan2(np.sin(error), np.cos(error))
    return float(np.sqrt(np.mean(error ** 2)))


with rosbag.Bag(bag_path, "r") as bag:
    t0 = None

    for topic, msg, t in bag.read_messages(topics=[pid_topic, odom_topic, desire_topic]):
        ts = t.to_sec()

        if t0 is None:
            t0 = ts

        ts_rel = ts - t0

        if topic == pid_topic:
            z_val = msg.z.target_p
            if not math.isnan(z_val) and abs(z_val) > 1e-6:
                t_pid.append(ts_rel)
                x_target.append(msg.x.target_p)
                y_target.append(msg.y.target_p)
                z_target.append(z_val)
                roll_target_pid.append(msg.roll.target_p)
                pitch_target_pid.append(msg.pitch.target_p)
                yaw_target_pid.append(msg.yaw.target_p)

        elif topic == odom_topic:
            t_odom.append(ts_rel)
            x_actual.append(msg.pose.pose.position.x)
            y_actual.append(msg.pose.pose.position.y)
            z_actual.append(msg.pose.pose.position.z)

            q = msg.pose.pose.orientation
            quat = [q.x, q.y, q.z, q.w]
            roll, pitch, yaw = euler_from_quaternion(quat)
            roll_actual.append(roll)
            pitch_actual.append(pitch)
            yaw_actual.append(yaw)

        elif topic == desire_topic:
            if not math.isnan(msg.roll):
                t_roll_target.append(ts_rel)
                roll_target.append(msg.roll)
                # degree の場合は次を使う
                # roll_target.append(math.radians(msg.roll))
            if not math.isnan(msg.pitch):
                t_pitch_target.append(ts_rel)
                pitch_target.append(msg.pitch)
            if not math.isnan(msg.yaw):
                t_yaw_target.append(ts_rel)
                yaw_target.append(msg.yaw)


attitude_target_sources = {}
for axis_name, desire_data, pid_data in (
    ("roll", roll_target, roll_target_pid),
    ("pitch", pitch_target, pitch_target_pid),
    ("yaw", yaw_target, yaw_target_pid),
):
    if len(pid_data) > 0 and len(desire_data) > 0:
        attitude_target_sources[axis_name] = f"{pid_topic} + {desire_topic}"
    elif len(pid_data) > 0:
        attitude_target_sources[axis_name] = pid_topic
    elif len(desire_data) > 0:
        attitude_target_sources[axis_name] = desire_topic
    else:
        attitude_target_sources[axis_name] = "unavailable"


# 姿勢実測値に、rollグラフと共通のローパスフィルタを適用
roll_actual_lp = butter_lowpass_filter(
    data=roll_actual,
    time=t_odom,
    cutoff_hz=2.0,
    order=4
)
pitch_actual_lp = butter_lowpass_filter(
    data=pitch_actual,
    time=t_odom,
    cutoff_hz=2.0,
    order=4
)
# Filter yaw continuously across the -pi/pi boundary, then wrap it back.
yaw_actual_lp = butter_lowpass_filter(
    data=np.unwrap(yaw_actual),
    time=t_odom,
    cutoff_hz=2.0,
    order=4
)
yaw_actual_lp = np.arctan2(np.sin(yaw_actual_lp), np.cos(yaw_actual_lp))

# z target が出ている時間範囲だけ抽出
t_odom_filtered = []
x_actual_filtered = []
y_actual_filtered = []
z_actual_filtered = []
roll_actual_filtered = []
pitch_actual_filtered = []
yaw_actual_filtered = []

t_pid_filtered = []
x_target_filtered = []
y_target_filtered = []
z_target_filtered = []

if len(t_pid) > 0:
    t_min = max(min(t_pid), args.trim_start)
    t_max = max(t_pid)

    if t_min > t_max:
        raise ValueError(
            f"trim start ({args.trim_start:.3f} s) is later than "
            f"the available data end ({t_max:.3f} s)"
        )

    print(f"plot time range: {t_min:.3f}–{t_max:.3f} s")

    for t, x, y, z in zip(t_pid, x_target, y_target, z_target):
        if t_min <= t <= t_max:
            t_pid_filtered.append(t)
            x_target_filtered.append(x)
            y_target_filtered.append(y)
            z_target_filtered.append(z)

    for t, x, y, z, r, p, yw in zip(
        t_odom,
        x_actual,
        y_actual,
        z_actual,
        roll_actual_lp,
        pitch_actual_lp,
        yaw_actual_lp,
    ):
        if t_min <= t <= t_max:
            t_odom_filtered.append(t)
            x_actual_filtered.append(x)
            y_actual_filtered.append(y)
            z_actual_filtered.append(z)
            roll_actual_filtered.append(r)
            pitch_actual_filtered.append(p)
            yaw_actual_filtered.append(yw)
else:
    print("Warning: target z が一度も検出されていません")
    t_min = None
    t_max = None


# プロット区間内の actual 時刻に target を同期して6軸RMSEを計算
if t_min is not None:
    if len(t_odom_filtered) == 0:
        raise ValueError("プロット区間内に odometry データがありません")

    rmse_time = np.asarray(t_odom_filtered, dtype=float)

    x_target_on_odom = interpolate_linear(
        t_pid_filtered, x_target_filtered, rmse_time
    )
    y_target_on_odom = interpolate_linear(
        t_pid_filtered, y_target_filtered, rmse_time
    )
    z_target_on_odom = interpolate_linear(
        t_pid_filtered, z_target_filtered, rmse_time
    )

    roll_target_on_odom = combine_attitude_target(
        t_pid,
        roll_target_pid,
        t_roll_target,
        roll_target,
        rmse_time,
    )
    pitch_target_on_odom = combine_attitude_target(
        t_pid,
        pitch_target_pid,
        t_pitch_target,
        pitch_target,
        rmse_time,
    )
    yaw_target_on_odom = combine_attitude_target(
        t_pid,
        yaw_target_pid,
        t_yaw_target,
        yaw_target,
        rmse_time,
    )

    x_rmse = calculate_rmse(x_target_on_odom, x_actual_filtered)
    y_rmse = calculate_rmse(y_target_on_odom, y_actual_filtered)
    z_rmse = calculate_rmse(z_target_on_odom, z_actual_filtered)
    roll_rmse = calculate_rmse(
        roll_target_on_odom, roll_actual_filtered, angular=True
    )
    pitch_rmse = calculate_rmse(
        pitch_target_on_odom, pitch_actual_filtered, angular=True
    )
    yaw_rmse = calculate_rmse(
        yaw_target_on_odom, yaw_actual_filtered, angular=True
    )

    print("=== Tracking RMSE in plotted time range ===")
    print("attitude actual: fourth-order Butterworth LPF, cutoff=2.0 Hz")
    print(
        "attitude target sources: "
        + ", ".join(
            f"{axis}={source}"
            for axis, source in attitude_target_sources.items()
        )
    )
    print(f"x RMSE    : {x_rmse:.6f} m")
    print(f"y RMSE    : {y_rmse:.6f} m")
    print(f"z RMSE    : {z_rmse:.6f} m")
    print(f"roll RMSE : {roll_rmse:.6f} rad")
    print(f"pitch RMSE: {pitch_rmse:.6f} rad")
    print(f"yaw RMSE  : {yaw_rmse:.6f} rad")


# RMSEと同じ合成targetをroll target表示にも使用
t_roll_target_plot = []
roll_target_plot = []

if t_min is not None and (
    len(roll_target_pid) > 0 or len(roll_target) > 0
):
    target_times_in_window = [
        time
        for time in list(t_pid) + list(t_roll_target)
        if t_min <= time <= t_max
    ]
    t_roll_target_plot = np.unique(
        np.asarray([t_min] + target_times_in_window + [t_max], dtype=float)
    )
    roll_target_plot = combine_attitude_target(
        t_pid,
        roll_target_pid,
        t_roll_target,
        roll_target,
        t_roll_target_plot,
    )


# プロット
fig, ax1 = plt.subplots(figsize=(12, 6))
ax2 = ax1.twinx()

# z: 青
ax1.plot(t_pid_filtered, z_target_filtered, linestyle="--", color="blue", lw=5)
ax1.plot(t_odom_filtered, z_actual_filtered, linestyle="-", color="blue", lw=5)

# x: 緑
ax1.plot(t_pid_filtered, x_target_filtered, linestyle="--", color="green", lw=5)
ax1.plot(t_odom_filtered, x_actual_filtered, linestyle="-", color="green", lw=5)

# roll: 赤
if len(t_roll_target_plot) > 0:
    ax2.plot(t_roll_target_plot, roll_target_plot, linestyle="--", color="red", lw=5)
ax2.plot(t_odom_filtered, roll_actual_filtered, linestyle="-", color="red", lw=5)

if t_min is not None:
    ax1.set_xlim(t_min, t_max)

ax1.set_xlabel("Time [s]")
ax1.set_ylabel("Position [m]")
ax2.set_ylabel("Roll [rad]")

ax1.tick_params(axis="y")
ax2.tick_params(axis="y")

ax1.grid(False)
plt.title("X, Z and Roll")
plt.tight_layout()
plt.show()
