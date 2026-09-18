#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import matplotlib.pyplot as plt
import rosbag

try:
    from tf.transformations import euler_from_quaternion
except Exception:
    raise ImportError("tf.transformations が見つかりません。ROS環境で実行してください。")

try:
    from scipy.signal import butter, filtfilt
except Exception:
    raise ImportError("SciPy (scipy.signal) が見つかりません。方法A（Butterworth + filtfilt）には SciPy が必要です。")


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


def _zoh_hold(t_src, y_src, t_dst):
    """Zero-Order Hold（最後の値を保持）"""
    t_src = np.asarray(t_src, float)
    y_src = np.asarray(y_src, float)
    t_dst = np.asarray(t_dst, float)

    m = np.isfinite(t_src) & np.isfinite(y_src)
    t_src, y_src = t_src[m], y_src[m]
    if len(t_src) == 0:
        return np.full_like(t_dst, np.nan)

    idx = np.argsort(t_src)
    t_src, y_src = t_src[idx], y_src[idx]

    j = np.searchsorted(t_src, t_dst, side="right") - 1
    y = np.full_like(t_dst, np.nan)
    ok = j >= 0
    y[ok] = y_src[j[ok]]
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


def _lowpass_filtfilt(x, fs, fc_hz, order=4):
    """
    Butterworth Low-pass + filtfilt（ゼロ位相）
    - x に NaN がある場合、有限値の連続区間ごとにフィルタリング
    - fs, fc が不正/無効なら元データ返却
    """
    x = np.asarray(x, float)
    if (not np.isfinite(fs)) or fs <= 0 or (not np.isfinite(fc_hz)) or fc_hz <= 0:
        return x.copy()

    nyq = 0.5 * fs
    wn = fc_hz / nyq
    if wn >= 1.0:
        return x.copy()

    y = x.copy()
    m = np.isfinite(x)
    if np.count_nonzero(m) < max(16, 3 * order):
        return x.copy()

    idx = np.where(m)[0]
    cuts = np.where(np.diff(idx) > 1)[0]
    segments = np.split(idx, cuts + 1)

    b, a = butter(order, wn, btype="low")

    for seg in segments:
        if len(seg) < max(16, 3 * order):
            continue
        xs = x[seg]
        try:
            y[seg] = filtfilt(b, a, xs, method="pad")
        except Exception:
            y[seg] = xs

    return y


def _find_start_time_by_target_change(
    t_nav,
    x_tgt,
    y_tgt,
    threshold_time=1771146720.0,
    eps=1e-9,
):
    """
    t_nav >= threshold_time の範囲で、target_pos_x または target_pos_y が
    「初めて変化」した時刻を返す。変化判定は各軸の abs(diff) > eps。
    見つからない場合は threshold_time 以降で最初に存在するサンプル時刻（なければ NaN）。
    """
    t_nav = np.asarray(t_nav, float)
    x_tgt = np.asarray(x_tgt, float)
    y_tgt = np.asarray(y_tgt, float)

    m = np.isfinite(t_nav) & np.isfinite(x_tgt) & np.isfinite(y_tgt)
    t_nav, x_tgt, y_tgt = t_nav[m], x_tgt[m], y_tgt[m]
    if len(t_nav) < 2:
        return float("nan")

    idx = np.argsort(t_nav)
    t_nav, x_tgt, y_tgt = t_nav[idx], x_tgt[idx], y_tgt[idx]

    k0 = int(np.searchsorted(t_nav, threshold_time, side="left"))
    if k0 >= len(t_nav):
        return float("nan")

    for i in range(max(k0 + 1, 1), len(t_nav)):
        x_changed = abs(x_tgt[i] - x_tgt[i - 1]) > eps
        y_changed = abs(y_tgt[i] - y_tgt[i - 1]) > eps
        if x_changed or y_changed:
            return float(t_nav[i])

    return float(t_nav[k0])


# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bag", help="rosbag file path")

    # target topics
    ap.add_argument("--nav-target-topic", default="/crobat/uav/nav",
                    help="target topic for x/y pos and attitude; fields: target_pos_x/y, "
                         "target_roll/pitch/yaw")

    ap.add_argument("--z-target-topic", default="/crobat/debug/pose/pid",
                    help="target topic for z position; field: msg.z.target_p")

    # actual topics
    ap.add_argument("--pos-actual-topic", default="/crobat/uav/cog/odom",
                    help="nav_msgs/Odometry; use msg.pose.pose.position.{x,y,z}")
    ap.add_argument("--att-actual-topic", default="/crobat/uav/baselink/odom",
                    help="nav_msgs/Odometry; use msg.pose.pose.orientation (quat->rpy)")

    # unit for nav target attitude
    ap.add_argument("--nav-att-unit", choices=["rad", "deg"], default="rad",
                    help="unit of target_roll/pitch/yaw in /crobat/uav/nav (default: rad)")

    # start condition: time threshold & first change detection
    ap.add_argument("--start-threshold-time", type=float, default=1771146720.0,
                    help="start condition threshold time [sec] (default: 1771146720)")
    ap.add_argument("--target-change-eps", type=float, default=1e-9,
                    help="abs(diff) > eps regarded as change for target_pos_x/y")

    # lowpass for plots (error only)
    ap.add_argument("--plot-lp-fc-pos", type=float, default=5.0,
                    help="lowpass cutoff [Hz] for position error plot (0 to disable)")
    ap.add_argument("--plot-lp-fc-att", type=float, default=8.0,
                    help="lowpass cutoff [Hz] for attitude error plot (0 to disable)")
    ap.add_argument("--plot-lp-order", type=int, default=4,
                    help="lowpass filter order (Butterworth)")
    ap.add_argument("--att-line-width", type=float, default=1.8,
                    help="line width for roll/pitch/yaw target and actual plots (default: 1.8)")

    args = ap.parse_args()
    if not np.isfinite(args.att_line_width) or args.att_line_width <= 0.0:
        ap.error("--att-line-width must be a positive finite number")

    # ---------- storage ----------
    # nav target: x/y position + attitude
    t_nav = []
    x_tgt, y_tgt = [], []
    roll_tgt, pitch_tgt, yaw_tgt = [], [], []

    # z target from /crobat/debug/pose/pid
    t_zpid, z_tgt_pid = [], []

    # actual position
    t_pa, x_act, y_act, z_act = [], [], [], []

    # actual attitude
    t_aa, roll_act_deg, pitch_act_deg, yaw_act_deg = [], [], [], []

    topics = [
        args.nav_target_topic,
        args.z_target_topic,
        args.pos_actual_topic,
        args.att_actual_topic,
    ]
    topics = list(dict.fromkeys(topics))

    with rosbag.Bag(args.bag) as bag:
        for topic, msg, t in bag.read_messages(topics=topics):
            ts = t.to_sec()

            # nav target: x/y + attitude
            if topic == args.nav_target_topic:
                try:
                    t_nav.append(ts)
                    x_tgt.append(float(msg.target_pos_x))
                    y_tgt.append(float(msg.target_pos_y))

                    rr = float(msg.target_roll)
                    pp = float(msg.target_pitch)
                    yy = float(msg.target_yaw)

                    if args.nav_att_unit == "deg":
                        rr = np.deg2rad(rr)
                        pp = np.deg2rad(pp)
                        yy = np.deg2rad(yy)

                    # 内部では rad のまま扱う
                    roll_tgt.append(rr)
                    pitch_tgt.append(pp)
                    yaw_tgt.append(yy)
                except Exception:
                    pass

            # z target from /crobat/debug/pose/pid
            if topic == args.z_target_topic:
                try:
                    t_zpid.append(ts)
                    z_tgt_pid.append(float(msg.z.target_p))
                except Exception:
                    pass

            # actual position
            if topic == args.pos_actual_topic:
                try:
                    p = msg.pose.pose.position
                    t_pa.append(ts)
                    x_act.append(float(p.x))
                    y_act.append(float(p.y))
                    z_act.append(float(p.z))
                except Exception:
                    pass

            # actual attitude
            if topic == args.att_actual_topic:
                try:
                    q = msg.pose.pose.orientation
                    r, p, y = euler_from_quaternion((q.x, q.y, q.z, q.w))
                    t_aa.append(ts)
                    roll_act_deg.append(float(r))
                    pitch_act_deg.append(float(p))
                    yaw_act_deg.append(float(y))
                except Exception:
                    pass

    # ---------- numpy ----------
    t_nav = np.asarray(t_nav, float)
    x_tgt = np.asarray(x_tgt, float)
    y_tgt = np.asarray(y_tgt, float)

    roll_tgt = np.asarray(roll_tgt, float)
    pitch_tgt = np.asarray(pitch_tgt, float)
    yaw_tgt = np.asarray(yaw_tgt, float)

    t_zpid = np.asarray(t_zpid, float)
    z_tgt_pid = np.asarray(z_tgt_pid, float)

    t_pa = np.asarray(t_pa, float)
    x_act = np.asarray(x_act, float)
    y_act = np.asarray(y_act, float)
    z_act = np.asarray(z_act, float)

    t_aa = np.asarray(t_aa, float)
    roll_act_deg = np.asarray(roll_act_deg, float)
    pitch_act_deg = np.asarray(pitch_act_deg, float)
    yaw_act_deg = np.asarray(yaw_act_deg, float)

    print("counts:",
          "nav_tgt", len(t_nav),
          "z_tgt_pid", len(t_zpid),
          "pos_act", len(t_pa),
          "att_act", len(t_aa))

    print("topics:",
          "nav_tgt", args.nav_target_topic,
          "z_tgt_pid", args.z_target_topic,
          "pos_act", args.pos_actual_topic,
          "att_act", args.att_actual_topic)

    print("z target source:", args.z_target_topic, "field: z.target_p")
    print("nav_att_unit:", args.nav_att_unit)

    if len(t_nav) < 2:
        raise ValueError("Nav target topic has too few samples (need >=2).")
    if len(t_zpid) < 1:
        raise ValueError("Z target topic has too few samples. /crobat/debug/pose/pid の z.target_p を確認してください。")
    if len(t_pa) < 2:
        raise ValueError("Position actual topic has too few samples (need >=2).")
    if len(t_aa) < 2:
        raise ValueError("Attitude actual topic has too few samples (need >=2).")

    # ---------- determine start time by target_pos_x/y change ----------
    start_time = _find_start_time_by_target_change(
        t_nav, x_tgt, y_tgt,
        threshold_time=args.start_threshold_time,
        eps=args.target_change_eps,
    )
    if not np.isfinite(start_time):
        raise ValueError("開始時刻を決定できませんでした（/crobat/uav/nav の時刻範囲を確認してください）。")

    # ---------- sort nav arrays first, then slice by start_time ----------
    m_nav = np.isfinite(t_nav)
    t_nav0 = t_nav[m_nav]
    x_tgt0 = x_tgt[m_nav]
    y_tgt0 = y_tgt[m_nav]
    roll_tgt0 = roll_tgt[m_nav]
    pitch_tgt0 = pitch_tgt[m_nav]
    yaw_tgt0 = yaw_tgt[m_nav]

    idx = np.argsort(t_nav0)
    t_nav0 = t_nav0[idx]
    x_tgt0 = x_tgt0[idx]
    y_tgt0 = y_tgt0[idx]
    roll_tgt0 = roll_tgt0[idx]
    pitch_tgt0 = pitch_tgt0[idx]
    yaw_tgt0 = yaw_tgt0[idx]

    m_win = t_nav0 >= start_time
    if np.count_nonzero(m_win) < 2:
        raise ValueError("開始時刻以降の nav ターゲットが少なすぎます（need >=2）。")

    t_nav_w = t_nav0[m_win]
    x_tgt_w = x_tgt0[m_win]
    y_tgt_w = y_tgt0[m_win]
    roll_tgt_w = roll_tgt0[m_win]
    pitch_tgt_w = pitch_tgt0[m_win]
    yaw_tgt_w = yaw_tgt0[m_win]

    end_time = float(t_nav_w[-1])

    # ---------- z target from /crobat/debug/pose/pid ----------
    # z.target_p は nav timebase に Zero-Order Hold で合わせる
    z_tgt_w = _zoh_hold(t_zpid, z_tgt_pid, t_nav_w)

    if np.count_nonzero(np.isfinite(z_tgt_w)) < 2:
        raise ValueError(
            "開始時刻以降の z target が少なすぎます。"
            "/crobat/debug/pose/pid の z.target_p の時刻範囲を確認してください。"
        )

    print("analysis window:",
          f"start_time={start_time:.6f}",
          f"end_time={end_time:.6f}",
          f"(start triggered by first change of target_pos_x/y after {args.start_threshold_time:.0f}s)")

    # ---------- position RMSE (nav target timebase in window) ----------
    x_act_i = _interp1(t_pa, x_act, t_nav_w)
    y_act_i = _interp1(t_pa, y_act, t_nav_w)
    z_act_i = _interp1(t_pa, z_act, t_nav_w)

    rmse_x = _rmse(x_tgt_w, x_act_i)
    rmse_y = _rmse(y_tgt_w, y_act_i)
    rmse_z = _rmse(z_tgt_w, z_act_i)

    # plot uses error (actual - target) on nav timebase in window
    ex = x_act_i - x_tgt_w
    ey = y_act_i - y_tgt_w
    ez = z_act_i - z_tgt_w

    # ---------- attitude window: [start_time, end_time] ----------
    m_act = (t_aa >= start_time) & (t_aa <= end_time)
    t_act = t_aa[m_act]
    roll_act_w = roll_act_deg[m_act]
    pitch_act_w = pitch_act_deg[m_act]
    yaw_act_w = yaw_act_deg[m_act]

    if len(t_act) < 2:
        raise ValueError("Attitude actual has too few samples in the chosen window (need >=2).")

    # target held during gaps, sampled on actual timebase (use windowed nav target)
    roll_tgt_on_act = _zoh_hold(t_nav_w, roll_tgt_w, t_act)
    pitch_tgt_on_act = _zoh_hold(t_nav_w, pitch_tgt_w, t_act)
    yaw_tgt_on_act = _zoh_hold(t_nav_w, yaw_tgt_w, t_act)

    # ---------- attitude RMSE ----------
    rmse_roll = _rmse(roll_tgt_on_act, roll_act_w)
    rmse_pitch = _rmse(pitch_tgt_on_act, pitch_act_w)
    rmse_yaw = _rmse(yaw_tgt_on_act, yaw_act_w)

    print("=== Tracking RMSE (windowed) ===")
    print(f"x RMSE    : {rmse_x:.6f}")
    print(f"y RMSE    : {rmse_y:.6f}")
    print(f"z RMSE    : {rmse_z:.6f}  (target from {args.z_target_topic}: z.target_p)")
    print(f"roll RMSE : {rmse_roll:.6f} rad  (target ZOH-hold from /crobat/uav/nav)")
    print(f"pitch RMSE: {rmse_pitch:.6f} rad  (target ZOH-hold from /crobat/uav/nav)")
    print(f"yaw RMSE  : {rmse_yaw:.6f} rad  (target ZOH-hold from /crobat/uav/nav)")

    # ---------- errors for plot ----------
    er = roll_act_w - roll_tgt_on_act
    ep = pitch_act_w - pitch_tgt_on_act
    eyaw = yaw_act_w - yaw_tgt_on_act

    # ---------- lowpass for plot series only ----------
    fs_pos = _estimate_fs(t_nav_w)
    fs_att = _estimate_fs(t_act)

    if args.plot_lp_fc_pos > 0:
        ex_plot = _lowpass_filtfilt(ex, fs_pos, args.plot_lp_fc_pos, order=args.plot_lp_order)
        ey_plot = _lowpass_filtfilt(ey, fs_pos, args.plot_lp_fc_pos, order=args.plot_lp_order)
        ez_plot = _lowpass_filtfilt(ez, fs_pos, args.plot_lp_fc_pos, order=args.plot_lp_order)
    else:
        ex_plot, ey_plot, ez_plot = ex, ey, ez

    if args.plot_lp_fc_att > 0:
        er_plot = _lowpass_filtfilt(er, fs_att, args.plot_lp_fc_att, order=args.plot_lp_order)
        ep_plot = _lowpass_filtfilt(ep, fs_att, args.plot_lp_fc_att, order=args.plot_lp_order)
        eyaw_plot = _lowpass_filtfilt(eyaw, fs_att, args.plot_lp_fc_att, order=args.plot_lp_order)
    else:
        er_plot, ep_plot, eyaw_plot = er, ep, eyaw

    # ---------- lowpass for attitude target vs actual plot ----------
    if args.plot_lp_fc_att > 0:
        roll_act_plot  = _lowpass_filtfilt(roll_act_w,  fs_att, args.plot_lp_fc_att, order=args.plot_lp_order)
        pitch_act_plot = _lowpass_filtfilt(pitch_act_w, fs_att, args.plot_lp_fc_att, order=args.plot_lp_order)
        yaw_act_plot   = _lowpass_filtfilt(yaw_act_w,   fs_att, args.plot_lp_fc_att, order=args.plot_lp_order)

        roll_tgt_plot  = _lowpass_filtfilt(roll_tgt_on_act,  fs_att, args.plot_lp_fc_att, order=args.plot_lp_order)
        pitch_tgt_plot = _lowpass_filtfilt(pitch_tgt_on_act, fs_att, args.plot_lp_fc_att, order=args.plot_lp_order)
        yaw_tgt_plot   = _lowpass_filtfilt(yaw_tgt_on_act,   fs_att, args.plot_lp_fc_att, order=args.plot_lp_order)
    else:
        roll_act_plot,  pitch_act_plot,  yaw_act_plot  = roll_act_w,  pitch_act_w,  yaw_act_w
        roll_tgt_plot,  pitch_tgt_plot,  yaw_tgt_plot  = roll_tgt_on_act, pitch_tgt_on_act, yaw_tgt_on_act

    # ---------- plots (no legend, no title, no grid, no savefig) ----------

    # 1) position errors (xyz) in one figure
    plt.figure()
    tt = t_nav_w - t_nav_w[0]
    plt.plot(tt, ex_plot, linewidth=1.8)
    plt.plot(tt, ey_plot, linewidth=1.8)
    plt.plot(tt, ez_plot, linewidth=1.8)
    plt.xlabel("time [s]")
    plt.ylabel("position error [m]")

    # 2) attitude errors (rpy) in one figure
    plt.figure()
    ta = t_act - start_time
    plt.plot(ta, er_plot, linewidth=1.8)
    plt.plot(ta, ep_plot, linewidth=1.8)
    plt.plot(ta, eyaw_plot, linewidth=1.8)
    plt.xlabel("time [s]")
    plt.ylabel("attitude error [rad]")

    # ---------- attitude target vs actual (roll) ----------
    plt.figure()
    ta = t_act - start_time
    plt.plot(ta, roll_tgt_plot, linewidth=args.att_line_width, linestyle="--")
    plt.plot(ta, roll_act_plot, linewidth=args.att_line_width, linestyle="-")
    plt.xlabel("time [s]")
    plt.ylabel("roll [rad]")

    # ---------- attitude target vs actual (pitch) ----------
    plt.figure()
    plt.plot(ta, pitch_tgt_plot, linewidth=args.att_line_width, linestyle="--")
    plt.plot(ta, pitch_act_plot, linewidth=args.att_line_width, linestyle="-")
    plt.xlabel("time [s]")
    plt.ylabel("pitch [rad]")

    # ---------- attitude target vs actual (yaw) ----------
    plt.figure()
    plt.plot(ta, yaw_tgt_plot, linewidth=args.att_line_width, linestyle="--")
    plt.plot(ta, yaw_act_plot, linewidth=args.att_line_width, linestyle="-")
    plt.xlabel("time [s]")
    plt.ylabel("yaw [rad]")

    # ---------- attitude all-in-one (roll, pitch, yaw target & actual) ----------
    plt.figure()
    ta = t_act - start_time

    # roll
    plt.plot(ta, roll_tgt_plot, linewidth=args.att_line_width, linestyle="--")
    plt.plot(ta, roll_act_plot, linewidth=args.att_line_width, linestyle="-")

    # pitch
    plt.plot(ta, pitch_tgt_plot, linewidth=args.att_line_width, linestyle="--")
    plt.plot(ta, pitch_act_plot, linewidth=args.att_line_width, linestyle="-")

    # yaw
    plt.plot(ta, yaw_tgt_plot, linewidth=args.att_line_width, linestyle="--")
    plt.plot(ta, yaw_act_plot, linewidth=args.att_line_width, linestyle="-")

    plt.xlabel("time [s]")
    plt.ylabel("attitude [rad]")

    # 3) XY trajectory: target vs actual (same timebase = t_nav_w)
    #    actual is interpolated onto nav timebase; then NaNを除去してプロット
    plt.figure()
    m_xy = np.isfinite(x_tgt_w) & np.isfinite(y_tgt_w) & np.isfinite(x_act_i) & np.isfinite(y_act_i)
    plt.plot(x_tgt_w[m_xy], y_tgt_w[m_xy], linewidth=6.0, linestyle='--')
    plt.plot(x_act_i[m_xy], y_act_i[m_xy], linewidth=6.0)
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.axis("equal")

    plt.show()


if __name__ == "__main__":
    main()
