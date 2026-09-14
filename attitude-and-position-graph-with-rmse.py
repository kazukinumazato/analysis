#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import matplotlib.pyplot as plt
import rosbag
from scipy.signal import butter, filtfilt

try:
    from tf.transformations import euler_from_quaternion
except Exception:
    raise ImportError("tf.transformations が見つかりません。ROS環境で実行してください。")


# ---------------- color style ----------------

PALETTE = {
    "blue": "#1F77B4",
    "pale_blue": "#17BECF",
    "lavender": "#9467BD",
    "peach": "#D62728",
    "gray": "#7F7F7F",
    "dark": "#000000",
}

TARGET_COLOR = PALETTE["peach"]
ACTUAL_COLOR = PALETTE["blue"]
TREND_COLOR = PALETTE["dark"]
VIB_COLOR = PALETTE["lavender"]
SPECTRUM_COLOR = PALETTE["peach"]
EDGE_COLOR = PALETTE["dark"]


def _apply_paper_style(ax=None):
    """
    論文図版風の淡色スタイルを適用
    罫線なし
    """
    if ax is None:
        ax = plt.gca()

    ax.grid(False)

    ax.spines["top"].set_color(EDGE_COLOR)
    ax.spines["right"].set_color(EDGE_COLOR)
    ax.spines["left"].set_color(EDGE_COLOR)
    ax.spines["bottom"].set_color(EDGE_COLOR)

    ax.tick_params(colors=EDGE_COLOR)
    ax.xaxis.label.set_color(EDGE_COLOR)
    ax.yaxis.label.set_color(EDGE_COLOR)
    ax.title.set_color(EDGE_COLOR)


# ---------------- utilities ----------------

def _rmse(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if np.count_nonzero(m) == 0:
        return float("nan")
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2)))


def _interp1(t_src, y_src, t_dst):
    """線形補間（範囲外はnan）"""
    t_src = np.asarray(t_src, float)
    y_src = np.asarray(y_src, float)
    t_dst = np.asarray(t_dst, float)

    m = np.isfinite(t_src) & np.isfinite(y_src)
    t_src, y_src = t_src[m], y_src[m]
    if len(t_src) < 2:
        return np.full_like(t_dst, np.nan)

    idx = np.argsort(t_src)
    t_src, y_src = t_src[idx], y_src[idx]

    y = np.interp(t_dst, t_src, y_src)
    y[(t_dst < t_src[0]) | (t_dst > t_src[-1])] = np.nan
    return y


def _interp_previous(t_src, y_src, t_dst):
    """直前の指令値を次の指令時刻まで保持する（範囲外はnan）。"""
    t_src = np.asarray(t_src, float)
    y_src = np.asarray(y_src, float)
    t_dst = np.asarray(t_dst, float)

    m = np.isfinite(t_src) & np.isfinite(y_src)
    t_src, y_src = t_src[m], y_src[m]
    if len(t_src) == 0:
        return np.full_like(t_dst, np.nan)

    order = np.argsort(t_src, kind="stable")
    t_src, y_src = t_src[order], y_src[order]

    src_index = np.searchsorted(t_src, t_dst, side="right") - 1
    in_range = (
        np.isfinite(t_dst)
        & (t_dst >= t_src[0])
        & (t_dst <= t_src[-1])
    )

    y = np.full_like(t_dst, np.nan)
    y[in_range] = y_src[src_index[in_range]]
    return y


def _estimate_fs(t):
    t = np.asarray(t, float)
    t = t[np.isfinite(t)]
    if len(t) < 3:
        return float("nan")

    dt = np.diff(t)
    dt = dt[(dt > 0) & np.isfinite(dt)]
    if len(dt) == 0:
        return float("nan")

    return float(1.0 / np.mean(dt))


def _moving_average(x, win):
    x = np.asarray(x, float)

    if win <= 1:
        return x.copy()

    k = np.ones(win, dtype=float) / float(win)
    valid = np.isfinite(x).astype(float)

    num = np.convolve(np.nan_to_num(x, nan=0.0), k, mode="same")
    den = np.convolve(valid, k, mode="same")

    y = np.full_like(x, np.nan)
    m = den > 0
    y[m] = num[m] / den[m]

    return y


def _uniform_resample(t, x, fs):
    """不等間隔(t,x)を fs で等間隔化して返す（DC除去）"""
    t = np.asarray(t, float)
    x = np.asarray(x, float)

    m = np.isfinite(t) & np.isfinite(x)
    t, x = t[m], x[m]

    if len(t) < 8 or (not np.isfinite(fs)) or fs <= 0:
        return None, None, None

    idx = np.argsort(t)
    t, x = t[idx], x[idx]

    tu = np.arange(t[0], t[-1], 1.0 / fs)
    xu = np.interp(tu, t, x)
    xu = xu - np.mean(xu)

    return tu, xu, fs


def _fft_spectra(x_u, fs):
    """
    hann窓 + 片側スペクトル
    戻り値: freqs, mag, power
    """
    x_u = np.asarray(x_u, float)
    n = len(x_u)

    if n < 32:
        return None, None, None

    w = np.hanning(n)
    xw = x_u * w

    X = np.fft.rfft(xw)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    mag = np.abs(X)
    power = mag ** 2

    return freqs, mag, power


def _dominant_peak(freqs, mag, fmin=0.1, fmax=None):
    """指定帯域で |X| 最大の周波数を返す"""
    freqs = np.asarray(freqs, float)
    mag = np.asarray(mag, float)

    if len(freqs) == 0:
        return float("nan"), float("nan")

    if fmax is None:
        fmax = float(freqs[-1])

    band = (freqs >= fmin) & (freqs <= fmax) & np.isfinite(mag)

    if np.count_nonzero(band) == 0:
        return float("nan"), float("nan")

    k = int(np.argmax(mag[band]))
    fpk = float(freqs[band][k])
    mpk = float(mag[band][k])

    return fpk, mpk


def _band_contribution(freqs, power, f_lo, f_hi, f_total_lo=0.1, f_total_hi=60.0):
    """
    P_band / P_total と band/全体ピーク周波数
    """
    freqs = np.asarray(freqs, float)
    power = np.asarray(power, float)

    m_total = (freqs >= f_total_lo) & (freqs <= f_total_hi) & np.isfinite(power)
    m_band = (freqs >= f_lo) & (freqs <= f_hi) & np.isfinite(power)

    if np.count_nonzero(m_total) == 0 or np.count_nonzero(m_band) == 0:
        return {
            "power_ratio": float("nan"),
            "P_band": float("nan"),
            "P_total": float("nan"),
            "peak_freq_band": float("nan"),
            "peak_freq_all": float("nan"),
        }

    P_total = float(np.sum(power[m_total]))
    P_band = float(np.sum(power[m_band]))
    power_ratio = P_band / P_total if P_total > 0 else float("nan")

    peak_freq_all = float(freqs[m_total][np.argmax(power[m_total])])
    peak_freq_band = float(freqs[m_band][np.argmax(power[m_band])])

    return {
        "power_ratio": power_ratio,
        "P_band": P_band,
        "P_total": P_total,
        "peak_freq_band": peak_freq_band,
        "peak_freq_all": peak_freq_all,
    }


def _butter_lowpass_filter(t, x, cutoff_hz, order=4):
    """
    不等間隔時系列を平均サンプリング周波数で Butterworth LPF
    nan を含む場合は有限区間のみ処理
    """
    t = np.asarray(t, float)
    x = np.asarray(x, float)

    m = np.isfinite(t) & np.isfinite(x)
    y = np.full_like(x, np.nan)

    if np.count_nonzero(m) < max(8, order * 3):
        return x.copy()

    t_valid = t[m]
    x_valid = x[m]

    fs = _estimate_fs(t_valid)
    if not np.isfinite(fs) or fs <= 0:
        return x.copy()

    nyq = 0.5 * fs
    if cutoff_hz <= 0 or cutoff_hz >= nyq:
        return x.copy()

    wn = cutoff_hz / nyq
    b, a = butter(order, wn, btype="low")
    y_valid = filtfilt(b, a, x_valid)

    y[m] = y_valid

    return y


def _analyze_vibration_fft(name, t, vibration, fs, fft_fmax, flap_flo, flap_fhi):
    """振動のFFTと指定帯域の寄与率を計算して表示する。"""
    t_u, vibration_u, fs_u = _uniform_resample(t, vibration, fs)

    freqs = mag = power = None
    dominant_frequency = float("nan")
    band = None

    if t_u is None:
        print(f"{name} FFT: resampling failed (too few samples or fs invalid).")
        return freqs, power, dominant_frequency, band

    freqs, mag, power = _fft_spectra(vibration_u, fs_u)
    if freqs is None:
        print(f"{name} FFT: failed (n too small).")
        return freqs, power, dominant_frequency, band

    dominant_frequency, _ = _dominant_peak(
        freqs,
        mag,
        fmin=0.1,
        fmax=fft_fmax,
    )
    print(f"=== {name.capitalize()} FFT peak ===")
    print(
        f"dominant f (|X| peak, 0.1–{fft_fmax:.1f}Hz): "
        f"{dominant_frequency:.3f} Hz"
    )

    band = _band_contribution(
        freqs,
        power,
        f_lo=flap_flo,
        f_hi=flap_fhi,
        f_total_lo=0.1,
        f_total_hi=fft_fmax,
    )
    print(f"=== {name.capitalize()} flapping-band contribution (power) ===")
    print(f"band                    : {flap_flo:.1f}–{flap_fhi:.1f} Hz")
    print(f"power ratio (band/total): {band['power_ratio'] * 100.0:.2f} %")
    print(f"peak freq (all)         : {band['peak_freq_all']:.3f} Hz")
    print(f"peak freq (band)        : {band['peak_freq_band']:.3f} Hz")

    return freqs, power, dominant_frequency, band


# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bag", help="rosbag file path")

    # actual topics
    ap.add_argument(
        "--pos-actual-topic",
        default="/crobat/uav/cog/odom",
        help="nav_msgs/Odometry; use msg.pose.pose.position.x",
    )
    ap.add_argument(
        "--att-actual-topic",
        default="/crobat/uav/baselink/odom",
        help="nav_msgs/Odometry; use msg.pose.pose.orientation (quat->roll/pitch)",
    )

    # target topics
    ap.add_argument(
        "--pos-target-topic",
        default="/crobat/debug/pose/pid",
        help="position target topic; use msg.x.target_p (float)",
    )
    ap.add_argument(
        "--att-target-topic",
        default="/crobat/desire_coordinate",
        help="spinal/DesireCoord; use roll and pitch in rad",
    )

    ap.add_argument("--trend-window-sec", type=float, default=0.5)
    ap.add_argument("--fft-fmax", type=float, default=60.0)
    ap.add_argument("--flap-flo", type=float, default=12.0)
    ap.add_argument("--flap-fhi", type=float, default=20.0)

    # roll LPF
    ap.add_argument(
        "--roll-lpf-cutoff",
        type=float,
        default=30.0,
        help="Butterworth low-pass cutoff frequency for actual roll [Hz]",
    )
    ap.add_argument(
        "--roll-lpf-order",
        type=int,
        default=4,
        help="Butterworth low-pass filter order",
    )

    # pitch LPF
    ap.add_argument(
        "--pitch-lpf-cutoff",
        type=float,
        default=30.0,
        help="Butterworth low-pass cutoff frequency for actual pitch [Hz]",
    )
    ap.add_argument(
        "--pitch-lpf-order",
        type=int,
        default=4,
        help="Butterworth low-pass filter order for actual pitch",
    )

    # plot margin
    ap.add_argument(
        "--plot-margin-sec",
        type=float,
        default=3.0,
        help="seconds to extend actual plot before/after target window",
    )

    # spectrum broken y-axis
    ap.add_argument(
        "--spectrum-break-lower",
        type=float,
        default=40,
        help="lower value of omitted y-axis range in spectrum plot",
    )
    ap.add_argument(
        "--spectrum-break-upper",
        type=float,
        default=60,
        help="upper value of omitted y-axis range in spectrum plot",
    )
    ap.add_argument(
        "--spectrum-auto-break-percentile",
        type=float,
        default=95.0,
        help="used only when break lower/upper are not specified",
    )

    args = ap.parse_args()

    # ---------- storage ----------

    # position
    t_px, x_tgt = [], []
    t_pa, x_act = [], []

    # attitude (roll and pitch share the same timestamps)
    t_at, roll_tgt, pitch_tgt = [], [], []
    t_aa, roll_act, pitch_act = [], [], []

    topics = [
        args.pos_target_topic,
        args.pos_actual_topic,
        args.att_target_topic,
        args.att_actual_topic,
    ]
    topics = list(dict.fromkeys(topics))

    with rosbag.Bag(args.bag) as bag:
        for topic, msg, t in bag.read_messages(topics=topics):
            ts = t.to_sec()

            # position target: msg.x.target_p (float)
            if topic == args.pos_target_topic:
                try:
                    t_px.append(ts)
                    x_tgt.append(float(msg.x.target_p))
                except Exception:
                    pass

            # position actual (nav_msgs/Odometry): msg.pose.pose.position.x
            if topic == args.pos_actual_topic:
                try:
                    t_pa.append(ts)
                    x_act.append(float(msg.pose.pose.position.x))
                except Exception:
                    pass

            # attitude target
            if topic == args.att_target_topic:
                try:
                    roll = float(msg.roll)
                    pitch = float(msg.pitch)
                    t_at.append(ts)
                    roll_tgt.append(roll)
                    pitch_tgt.append(pitch)
                except Exception:
                    pass

            # attitude actual: quaternion -> roll and pitch
            if topic == args.att_actual_topic:
                try:
                    q = msg.pose.pose.orientation
                    r, p, _ = euler_from_quaternion((q.x, q.y, q.z, q.w))
                    t_aa.append(ts)
                    roll_act.append(float(r))
                    pitch_act.append(float(p))
                except Exception:
                    pass

    # ---------- numpy ----------

    t_px = np.asarray(t_px, float)
    x_tgt = np.asarray(x_tgt, float)

    t_pa = np.asarray(t_pa, float)
    x_act = np.asarray(x_act, float)

    t_at = np.asarray(t_at, float)
    roll_tgt = np.asarray(roll_tgt, float)
    pitch_tgt = np.asarray(pitch_tgt, float)

    t_aa = np.asarray(t_aa, float)
    roll_act = np.asarray(roll_act, float)
    pitch_act = np.asarray(pitch_act, float)

    print(
        "counts:",
        "pos_tgt", len(t_px),
        "pos_act", len(t_pa),
        "att_tgt", len(t_at),
        "att_act", len(t_aa),
    )
    print(
        "topics:",
        "pos_act", args.pos_actual_topic,
        "att_act", args.att_actual_topic,
        "att_tgt", args.att_target_topic,
    )

    if len(t_px) < 2:
        raise ValueError("Position target topic has too few samples (need >=2).")

    if len(t_aa) < 2:
        raise ValueError("Attitude actual topic has too few samples (need >=2).")

    # ---------- actual attitude low-pass ----------

    fs_att = _estimate_fs(t_aa)

    print(f"attitude actual fs_est: {fs_att:.2f} Hz")
    print(
        f"roll LPF           : Butterworth low-pass, "
        f"order={args.roll_lpf_order}, cutoff={args.roll_lpf_cutoff:.2f} Hz"
    )
    print(
        f"pitch LPF          : Butterworth low-pass, "
        f"order={args.pitch_lpf_order}, cutoff={args.pitch_lpf_cutoff:.2f} Hz"
    )

    roll_act_lpf = _butter_lowpass_filter(
        t_aa,
        roll_act,
        cutoff_hz=args.roll_lpf_cutoff,
        order=args.roll_lpf_order,
    )
    pitch_act_lpf = _butter_lowpass_filter(
        t_aa,
        pitch_act,
        cutoff_hz=args.pitch_lpf_cutoff,
        order=args.pitch_lpf_order,
    )

    # ---------- position RMSE (target timebase) ----------

    x_act_i = _interp1(t_pa, x_act, t_px)
    rmse_x = _rmse(x_tgt, x_act_i)

    # ---------- attitude target fallback ----------

    has_desired = len(t_at) >= 1

    if not has_desired:
        print("WARNING: desire_coordinate not found (or empty).")
        print("         Attitude target is assumed to be 0 rad for entire duration.")

    # ---------- attitude window ----------

    if has_desired:
        att_t0 = float(np.nanmin(t_at))
        att_t1 = float(np.nanmax(t_at))
    else:
        att_t0 = float(np.nanmin(t_aa))
        att_t1 = float(np.nanmax(t_aa))

    # RMSE用: target区間そのもの
    m_act_rmse = (t_aa >= att_t0) & (t_aa <= att_t1)
    t_act_rmse = t_aa[m_act_rmse]
    roll_act_rmse = roll_act_lpf[m_act_rmse]
    pitch_act_rmse = pitch_act_lpf[m_act_rmse]

    if len(t_act_rmse) < 2:
        raise ValueError("Attitude actual has too few samples in the RMSE window (need >=2).")

    if has_desired:
        roll_tgt_on_act_rmse = _interp_previous(t_at, roll_tgt, t_act_rmse)
        pitch_tgt_on_act_rmse = _interp_previous(t_at, pitch_tgt, t_act_rmse)
    else:
        roll_tgt_on_act_rmse = np.zeros_like(t_act_rmse, dtype=float)
        pitch_tgt_on_act_rmse = np.zeros_like(t_act_rmse, dtype=float)

    rmse_roll = _rmse(roll_tgt_on_act_rmse, roll_act_rmse)
    rmse_pitch = _rmse(pitch_tgt_on_act_rmse, pitch_act_rmse)

    # プロット用: 前後 margin 秒拡張
    plot_t0 = att_t0 - args.plot_margin_sec
    plot_t1 = att_t1 + args.plot_margin_sec

    m_act_plot = (t_aa >= plot_t0) & (t_aa <= plot_t1)
    t_act = t_aa[m_act_plot]
    roll_act_w = roll_act_lpf[m_act_plot]
    pitch_act_w = pitch_act_lpf[m_act_plot]

    if len(t_act) < 2:
        raise ValueError("Attitude actual has too few samples in the plot window (need >=2).")

    if has_desired:
        roll_tgt_on_act = _interp_previous(t_at, roll_tgt, t_act)
        pitch_tgt_on_act = _interp_previous(t_at, pitch_tgt, t_act)
    else:
        roll_tgt_on_act = np.zeros_like(t_act, dtype=float)
        pitch_tgt_on_act = np.zeros_like(t_act, dtype=float)

    print("=== Tracking RMSE ===")
    print(f"x RMSE    : {rmse_x:.6f}")
    print(f"roll RMSE : {rmse_roll:.6f} rad  (target held; actual=LPF applied)")
    print(f"pitch RMSE: {rmse_pitch:.6f} rad  (target held; actual=LPF applied)")

    # ---------- vibration analysis (continuous actual in plot window) ----------

    fs = _estimate_fs(t_act)
    win = int(max(1, round(fs * args.trend_window_sec))) if np.isfinite(fs) else 1

    roll_trend = _moving_average(roll_act_w, win)
    pitch_trend = _moving_average(pitch_act_w, win)
    roll_vib = roll_act_w - roll_trend
    pitch_vib = pitch_act_w - pitch_trend

    roll_vib_rms = (
        float(np.sqrt(np.nanmean(roll_vib * roll_vib)))
        if np.any(np.isfinite(roll_vib))
        else float("nan")
    )
    roll_vib_p2p = (
        float(np.nanmax(roll_vib) - np.nanmin(roll_vib))
        if np.any(np.isfinite(roll_vib))
        else float("nan")
    )
    roll_vib_p95 = (
        float(np.nanpercentile(roll_vib, 97.5) - np.nanpercentile(roll_vib, 2.5))
        if np.any(np.isfinite(roll_vib))
        else float("nan")
    )
    pitch_vib_rms = (
        float(np.sqrt(np.nanmean(pitch_vib * pitch_vib)))
        if np.any(np.isfinite(pitch_vib))
        else float("nan")
    )
    pitch_vib_p2p = (
        float(np.nanmax(pitch_vib) - np.nanmin(pitch_vib))
        if np.any(np.isfinite(pitch_vib))
        else float("nan")
    )
    pitch_vib_p95 = (
        float(
            np.nanpercentile(pitch_vib, 97.5)
            - np.nanpercentile(pitch_vib, 2.5)
        )
        if np.any(np.isfinite(pitch_vib))
        else float("nan")
    )

    print("=== Roll vibration basic ===")
    print(f"fs_est       : {fs:.2f} Hz")
    print(f"trend window : {args.trend_window_sec:.3f} s (~{win} samples)")
    print(f"vib RMS      : {roll_vib_rms:.6f} rad")
    print(f"vib p2p      : {roll_vib_p2p:.6f} rad")
    print(f"vib 95% width: {roll_vib_p95:.6f} rad")
    print("=== Pitch vibration basic ===")
    print(f"fs_est       : {fs:.2f} Hz")
    print(f"trend window : {args.trend_window_sec:.3f} s (~{win} samples)")
    print(f"vib RMS      : {pitch_vib_rms:.6f} rad")
    print(f"vib p2p      : {pitch_vib_p2p:.6f} rad")
    print(f"vib 95% width: {pitch_vib_p95:.6f} rad")

    # ---------- FFT peak & band contribution ----------

    roll_freqs, roll_power, roll_dom_f, roll_band = _analyze_vibration_fft(
        "roll",
        t_act,
        roll_vib,
        fs,
        args.fft_fmax,
        args.flap_flo,
        args.flap_fhi,
    )
    pitch_freqs, pitch_power, pitch_dom_f, pitch_band = _analyze_vibration_fft(
        "pitch",
        t_act,
        pitch_vib,
        fs,
        args.fft_fmax,
        args.flap_flo,
        args.flap_fhi,
    )

    # ---------- plots ----------

    # position: target vs actual
    plt.figure()

    plt.plot(
        t_px - t_px[0],
        x_tgt,
        color=TARGET_COLOR,
        linewidth=2.2,
        linestyle="--",
        label="target",
    )
    plt.plot(
        t_px - t_px[0],
        x_act_i,
        color=ACTUAL_COLOR,
        linewidth=2.2,
        label="actual",
    )

    plt.xlabel("time [s]")
    plt.ylabel("x")
    plt.title("x: target vs actual")
    plt.legend(frameon=False)
    _apply_paper_style()

    # roll: target vs actual (with pre/post margin)
    plt.figure()

    plt.plot(
        t_act - plot_t0,
        roll_tgt_on_act,
        color=TARGET_COLOR,
        linewidth=4.0,
        linestyle="--",
        drawstyle="steps-post",
        label="target",
    )
    plt.plot(
        t_act - plot_t0,
        roll_act_w,
        color=ACTUAL_COLOR,
        linewidth=3.0,
        label="actual",
    )

    plt.xlabel("time [s]")
    plt.ylabel("roll [rad]")

    if has_desired:
        plt.title(f"roll: target vs actual, margin={args.plot_margin_sec:.1f}s")
    else:
        plt.title("roll: target 0 rad fallback vs actual")

    plt.legend(frameon=False)
    _apply_paper_style()

    # roll: actual and trend
    plt.figure()

    plt.plot(
        t_act - plot_t0,
        roll_act_w,
        color=ACTUAL_COLOR,
        linewidth=1.8,
        label="actual LPF",
    )
    plt.plot(
        t_act - plot_t0,
        roll_trend,
        color=TREND_COLOR,
        linewidth=2.2,
        label="trend",
    )

    plt.xlabel("time [s]")
    plt.ylabel("roll [rad]")
    plt.title("roll: actual LPF and trend")
    plt.legend(frameon=False)
    _apply_paper_style()

    # roll: vibration
    plt.figure()

    plt.plot(
        t_act - plot_t0,
        roll_vib,
        color=VIB_COLOR,
        linewidth=1.8,
    )

    plt.xlabel("time [s]")
    plt.ylabel("roll vibration [rad]")
    plt.title(
        f"roll vibration: RMS={roll_vib_rms:.3f} rad, "
        f"f*={roll_dom_f:.2f} Hz"
    )
    _apply_paper_style()

    # pitch: target vs actual (with pre/post margin)
    plt.figure()

    plt.plot(
        t_act - plot_t0,
        pitch_tgt_on_act,
        color=TARGET_COLOR,
        linewidth=4.0,
        linestyle="--",
        drawstyle="steps-post",
        label="target",
    )
    plt.plot(
        t_act - plot_t0,
        pitch_act_w,
        color=ACTUAL_COLOR,
        linewidth=3.0,
        label="actual",
    )

    plt.xlabel("time [s]")
    plt.ylabel("pitch [rad]")

    if has_desired:
        plt.title(f"pitch: target vs actual, margin={args.plot_margin_sec:.1f}s")
    else:
        plt.title("pitch: target 0 rad fallback vs actual")

    plt.legend(frameon=False)
    _apply_paper_style()

    # pitch: actual and trend
    plt.figure()

    plt.plot(
        t_act - plot_t0,
        pitch_act_w,
        color=ACTUAL_COLOR,
        linewidth=1.8,
        label="actual LPF",
    )
    plt.plot(
        t_act - plot_t0,
        pitch_trend,
        color=TREND_COLOR,
        linewidth=2.2,
        label="trend",
    )

    plt.xlabel("time [s]")
    plt.ylabel("pitch [rad]")
    plt.title("pitch: actual LPF and trend")
    plt.legend(frameon=False)
    _apply_paper_style()

    # pitch: vibration
    plt.figure()

    plt.plot(
        t_act - plot_t0,
        pitch_vib,
        color=VIB_COLOR,
        linewidth=1.8,
    )

    plt.xlabel("time [s]")
    plt.ylabel("pitch vibration [rad]")
    plt.title(
        f"pitch vibration: RMS={pitch_vib_rms:.3f} rad, "
        f"f*={pitch_dom_f:.2f} Hz"
    )
    _apply_paper_style()

    # spectrum with directly specified broken y-axis
    if roll_freqs is not None and roll_power is not None:
        m = (roll_freqs >= 0.0) & (roll_freqs <= args.fft_fmax)

        f_plot = roll_freqs[m]
        p_plot = roll_power[m]

        finite_p = p_plot[np.isfinite(p_plot)]

        if len(finite_p) > 0:
            p_max = float(np.nanmax(finite_p))

            title = "roll vibration power spectrum"

            if roll_band is not None and np.isfinite(roll_band["power_ratio"]):
                title += (
                    f"  ({args.flap_flo:.0f}–{args.flap_fhi:.0f}Hz="
                    f"{roll_band['power_ratio'] * 100.0:.1f}%)"
                )

            # 軸ブレーク範囲を直接指定
            # lower: 下段グラフの上端
            # upper: 上段グラフの下端
            if args.spectrum_break_lower is not None and args.spectrum_break_upper is not None:
                p_low_max = float(args.spectrum_break_lower)
                p_high_min = float(args.spectrum_break_upper)
            else:
                # 直接指定しない場合は自動設定
                p_low_max = float(
                    np.nanpercentile(
                        finite_p,
                        args.spectrum_auto_break_percentile,
                    )
                )
                p_high_min = float(p_low_max * 1.2)

            # 指定値の妥当性チェック
            use_broken_axis = True

            if p_low_max <= 0:
                print("WARNING: spectrum-break-lower must be > 0. Normal spectrum plot is used.")
                use_broken_axis = False

            if p_high_min <= p_low_max:
                print("WARNING: spectrum-break-upper must be larger than spectrum-break-lower. Normal spectrum plot is used.")
                use_broken_axis = False

            if p_max <= p_high_min:
                print("WARNING: spectrum-break-upper is larger than the maximum spectrum power. Normal spectrum plot is used.")
                use_broken_axis = False

            if not use_broken_axis:
                plt.figure()

                plt.plot(
                    f_plot,
                    p_plot,
                    color=SPECTRUM_COLOR,
                    linewidth=1.6,
                )

                plt.xlabel("frequency [Hz]")
                plt.ylabel("power (a.u.)")
                plt.title(title)
                _apply_paper_style()

            else:
                fig, (ax_high, ax_low) = plt.subplots(
                    2,
                    1,
                    sharex=True,
                    figsize=(8, 6),
                    gridspec_kw={
                        "height_ratios": [1, 1],
                        "hspace": 0.3,
                    },
                )

                ax_high.plot(
                    f_plot,
                    p_plot,
                    color=SPECTRUM_COLOR,
                    linewidth=1.6,
                )
                ax_low.plot(
                    f_plot,
                    p_plot,
                    color=SPECTRUM_COLOR,
                    linewidth=1.6,
                )

                # 下段: 小さい成分を見る範囲
                ax_low.set_ylim(0, p_low_max)

                # 上段: 大きいピークを見る範囲
                ax_high.set_ylim(p_high_min, p_max * 1.05)

                # 上下の境界線を非表示
                ax_high.spines["bottom"].set_visible(False)
                ax_low.spines["top"].set_visible(False)

                # 上段のx軸ラベルを消す
                ax_high.tick_params(labelbottom=False)

                ax_low.set_xlabel("frequency [Hz]")
                ax_low.set_ylabel("")
                ax_high.set_ylabel("")

                ax_high.tick_params(labelleft=False)
                ax_low.tick_params(labelleft=False)

                ax_high.set_title(title)

                _apply_paper_style(ax_high)
                _apply_paper_style(ax_low)

                # _apply_paper_style() 後に、ブレーク用に必要な非表示設定を再適用
                ax_high.spines["bottom"].set_visible(False)
                ax_low.spines["top"].set_visible(False)
                ax_high.tick_params(labelbottom=False)
                ax_high.tick_params(labelleft=False)
                ax_low.tick_params(labelleft=False)

                # 軸ブレークを示す斜線
                d = 0.020

                kwargs = dict(
                    transform=ax_high.transAxes,
                    color=EDGE_COLOR,
                    clip_on=False,
                    linewidth=1.0,
                )
                ax_high.plot((-d, +d), (-d, +d), **kwargs)
                ax_high.plot((1 - d, 1 + d), (-d, +d), **kwargs)

                kwargs.update(transform=ax_low.transAxes)
                ax_low.plot((-d, +d), (1 - d, 1 + d), **kwargs)
                ax_low.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

                print("=== Spectrum broken y-axis ===")
                print(f"break lower / lower axis ymax : {p_low_max:.6e}")
                print(f"break upper / upper axis ymin : {p_high_min:.6e}")
                print(f"spectrum max                  : {p_max:.6e}")
                print("DEBUG spectrum break lower:", args.spectrum_break_lower)
                print("DEBUG spectrum break upper:", args.spectrum_break_upper)
                print("DEBUG p_low_max:", p_low_max)
                print("DEBUG p_high_min:", p_high_min)
                print("DEBUG p_max:", p_max)

    # pitch spectrum with the same broken y-axis settings as roll
    if pitch_freqs is not None and pitch_power is not None:
        m = (pitch_freqs >= 0.0) & (pitch_freqs <= args.fft_fmax)

        f_plot = pitch_freqs[m]
        p_plot = pitch_power[m]

        finite_p = p_plot[np.isfinite(p_plot)]

        if len(finite_p) > 0:
            p_max = float(np.nanmax(finite_p))

            title = "pitch vibration power spectrum"

            if pitch_band is not None and np.isfinite(pitch_band["power_ratio"]):
                title += (
                    f"  ({args.flap_flo:.0f}–{args.flap_fhi:.0f}Hz="
                    f"{pitch_band['power_ratio'] * 100.0:.1f}%)"
                )

            if args.spectrum_break_lower is not None and args.spectrum_break_upper is not None:
                p_low_max = float(args.spectrum_break_lower)
                p_high_min = float(args.spectrum_break_upper)
            else:
                p_low_max = float(
                    np.nanpercentile(
                        finite_p,
                        args.spectrum_auto_break_percentile,
                    )
                )
                p_high_min = float(p_low_max * 1.2)

            use_broken_axis = True

            if p_low_max <= 0:
                print("WARNING: spectrum-break-lower must be > 0. Normal pitch spectrum plot is used.")
                use_broken_axis = False

            if p_high_min <= p_low_max:
                print("WARNING: spectrum-break-upper must be larger than spectrum-break-lower. Normal pitch spectrum plot is used.")
                use_broken_axis = False

            if p_max <= p_high_min:
                print("WARNING: spectrum-break-upper is larger than the maximum pitch spectrum power. Normal spectrum plot is used.")
                use_broken_axis = False

            if not use_broken_axis:
                plt.figure()

                plt.plot(
                    f_plot,
                    p_plot,
                    color=SPECTRUM_COLOR,
                    linewidth=1.6,
                )

                plt.xlabel("frequency [Hz]")
                plt.ylabel("power (a.u.)")
                plt.title(title)
                _apply_paper_style()

            else:
                fig, (ax_high, ax_low) = plt.subplots(
                    2,
                    1,
                    sharex=True,
                    figsize=(8, 6),
                    gridspec_kw={
                        "height_ratios": [1, 1],
                        "hspace": 0.3,
                    },
                )

                ax_high.plot(
                    f_plot,
                    p_plot,
                    color=SPECTRUM_COLOR,
                    linewidth=1.6,
                )
                ax_low.plot(
                    f_plot,
                    p_plot,
                    color=SPECTRUM_COLOR,
                    linewidth=1.6,
                )

                ax_low.set_ylim(0, p_low_max)
                ax_high.set_ylim(p_high_min, p_max * 1.05)

                ax_high.spines["bottom"].set_visible(False)
                ax_low.spines["top"].set_visible(False)
                ax_high.tick_params(labelbottom=False)

                ax_low.set_xlabel("frequency [Hz]")
                ax_low.set_ylabel("")
                ax_high.set_ylabel("")

                ax_high.tick_params(labelleft=False)
                ax_low.tick_params(labelleft=False)

                ax_high.set_title(title)

                _apply_paper_style(ax_high)
                _apply_paper_style(ax_low)

                ax_high.spines["bottom"].set_visible(False)
                ax_low.spines["top"].set_visible(False)
                ax_high.tick_params(labelbottom=False)
                ax_high.tick_params(labelleft=False)
                ax_low.tick_params(labelleft=False)

                d = 0.020

                kwargs = dict(
                    transform=ax_high.transAxes,
                    color=EDGE_COLOR,
                    clip_on=False,
                    linewidth=1.0,
                )
                ax_high.plot((-d, +d), (-d, +d), **kwargs)
                ax_high.plot((1 - d, 1 + d), (-d, +d), **kwargs)

                kwargs.update(transform=ax_low.transAxes)
                ax_low.plot((-d, +d), (1 - d, 1 + d), **kwargs)
                ax_low.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

                print("=== Pitch spectrum broken y-axis ===")
                print(f"break lower / lower axis ymax : {p_low_max:.6e}")
                print(f"break upper / upper axis ymin : {p_high_min:.6e}")
                print(f"spectrum max                  : {p_max:.6e}")

    plt.show()


if __name__ == "__main__":
    main()
