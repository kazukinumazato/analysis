import rosbag
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, welch

# ====== 設定 ======
BAG_PATH = "../crobat_flight/2025-12-31-23-12-10-tilt-successfull.bag"
TOPIC = "/crobat20/mocap/pose"     # rosbag info で確認した名前に変更
AXIS = "roll"             # "roll" / "pitch" / "yaw"
BAND = (10.0, 20.0)       # 羽ばたき周波数帯 [Hz]

# ====== quaternion → roll pitch yaw ======
def quat_to_rpy(q):
    x, y, z, w = q.x, q.y, q.z, q.w

    roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    pitch = np.arcsin(np.clip(2*(w*y - z*x), -1, 1))
    yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return roll, pitch, yaw

# ====== rosbag 読み取り ======
t_list, sig_list = [], []

with rosbag.Bag(BAG_PATH, "r") as bag:
    for topic, msg, t in bag.read_messages(topics=[TOPIC]):
        pose = msg.pose
        r, p, y = quat_to_rpy(pose.orientation)

        val = {"roll": r, "pitch": p, "yaw": y}[AXIS]

        t_list.append(t.to_sec())
        sig_list.append(val)

if len(t_list) == 0:
    raise RuntimeError("No mocap data read. Check TOPIC name.")

t = np.asarray(t_list)
x = np.unwrap(np.asarray(sig_list))

# 時刻を 0 始まり
t = t - t[0]

# ====== 等間隔化 ======
dt = np.median(np.diff(t))
fs = 1.0 / dt

t_u = np.arange(0, t[-1], dt)
interp = interp1d(t, x, kind="linear", fill_value="extrapolate")
xu = interp(t_u)

# DC 成分除去
xu = xu - np.mean(xu)

print(f"Sampling rate ≈ {fs:.2f} Hz")
# ====== バンドパスフィルタ ======
def bandpass(sig, fs, f1, f2, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [f1/nyq, f2/nyq], btype="band")
    return filtfilt(b, a, sig)

x_band = bandpass(xu, fs, BAND[0], BAND[1])

# ====== 振幅評価 ======
rms_band = np.sqrt(np.mean(x_band**2))
peak_band = np.max(np.abs(x_band))

print("\n=== 10–20 Hz 成分の評価 ===")
print(f"RMS amplitude : {rms_band:.6f} rad")
print(f"Peak amplitude: {peak_band:.6f} rad")
print(f"±0.1 rad 比率: {peak_band / 0.1 * 100:.2f} %")
# ====== スペクトル（Welch） ======
f, Pxx = welch(xu, fs=fs, nperseg=int(2*fs))

rms_all = np.sqrt(np.mean(xu**2))

print("\n=== RMS比（エネルギ比） ===")
print(f"RMS(all)     : {rms_all:.6f} rad")
print(f"RMS(10-20Hz) : {rms_band:.6f} rad")
print(f"ratio        : {rms_band/rms_all*100:.2f} %")
bands = [(0.1, 5), (5, 10), (10, 15), (15, 20), (20, 30)]
for f1, f2 in bands:
    xb = bandpass(xu, fs, f1, f2)
    rms = np.sqrt(np.mean(xb**2))
    peak = np.max(np.abs(xb))
    print(f"{f1:>5.1f}-{f2:<5.1f} Hz : RMS={rms:.6f} rad, Peak={peak:.6f} rad")


plt.figure(figsize=(6,4))
plt.semilogy(f, Pxx)
plt.axvspan(BAND[0], BAND[1], color="r", alpha=0.2, label="15–20 Hz")
plt.xlim(0, 50)
plt.xlabel("Frequency [Hz]")
plt.ylabel("PSD [rad^2/Hz]")
plt.title(f"{AXIS} PSD (mocap)")
plt.grid(True, which="both")
plt.legend()
plt.tight_layout()
plt.show()
plt.figure(figsize=(7,4))
plt.plot(t_u, xu, label="Original")
plt.plot(t_u, x_band, label="15–20 Hz only", linewidth=2)
plt.xlabel("Time [s]")
plt.ylabel("Angle [rad]")
plt.title(f"{AXIS} time signal")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
