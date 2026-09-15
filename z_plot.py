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
z_target = []

t_odom = []
x_actual = []
z_actual = []
roll_actual = []

t_roll_target = []
roll_target = []


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
                z_target.append(z_val)

        elif topic == odom_topic:
            t_odom.append(ts_rel)
            x_actual.append(msg.pose.pose.position.x)
            z_actual.append(msg.pose.pose.position.z)

            q = msg.pose.pose.orientation
            quat = [q.x, q.y, q.z, q.w]
            roll, pitch, yaw = euler_from_quaternion(quat)
            roll_actual.append(roll)

        elif topic == desire_topic:
            if not math.isnan(msg.roll):
                t_roll_target.append(ts_rel)
                roll_target.append(msg.roll)
                # degree の場合は次を使う
                # roll_target.append(math.radians(msg.roll))


# roll 実測値にローパスフィルタ
roll_actual_lp = butter_lowpass_filter(
    data=roll_actual,
    time=t_odom,
    cutoff_hz=2.0,
    order=4
)

# z target が出ている時間範囲だけ抽出
t_odom_filtered = []
x_actual_filtered = []
z_actual_filtered = []
roll_actual_filtered = []

t_pid_filtered = []
x_target_filtered = []
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

    for t, x, z in zip(t_pid, x_target, z_target):
        if t_min <= t <= t_max:
            t_pid_filtered.append(t)
            x_target_filtered.append(x)
            z_target_filtered.append(z)

    for t, x, z, r in zip(t_odom, x_actual, z_actual, roll_actual_lp):
        if t_min <= t <= t_max:
            t_odom_filtered.append(t)
            x_actual_filtered.append(x)
            z_actual_filtered.append(z)
            roll_actual_filtered.append(r)
else:
    print("Warning: target z が一度も検出されていません")
    t_min = None
    t_max = None


# roll target を表示用に整形
# 欲しい補完は「最初の前」と「最後の後」だけ
t_roll_target_plot = []
roll_target_plot = []

if len(t_roll_target) > 0 and t_min is not None:
    # まず表示区間内の元データをそのまま使う
    for t, r in zip(t_roll_target, roll_target):
        if t_min <= t <= t_max:
            t_roll_target_plot.append(t)
            roll_target_plot.append(r)

    # 表示区間に1点も無い場合でも、前後の値で埋めるための処理
    if len(t_roll_target_plot) == 0:
        # t_min以前で最後に出た値
        prev_idx = None
        for i, t in enumerate(t_roll_target):
            if t <= t_min:
                prev_idx = i
            else:
                break

        # t_max以降で最初に出る値
        next_idx = None
        for i, t in enumerate(t_roll_target):
            if t >= t_min:
                next_idx = i
                break

        if prev_idx is not None:
            fill_value = roll_target[prev_idx]
        elif next_idx is not None:
            fill_value = roll_target[next_idx]
        else:
            fill_value = None

        if fill_value is not None:
            t_roll_target_plot = [t_min, t_max]
            roll_target_plot = [fill_value, fill_value]

    else:
        # 最初の値が出る前を、その最初の値で埋める
        if t_roll_target_plot[0] > t_min:
            t_roll_target_plot.insert(0, t_min)
            roll_target_plot.insert(0, roll_target_plot[0])

        # 最後の値が出た後を、その最後の値で埋める
        if t_roll_target_plot[-1] < t_max:
            t_roll_target_plot.append(t_max)
            roll_target_plot.append(roll_target_plot[-1])


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
