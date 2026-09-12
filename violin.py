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
    raise ImportError("SciPy (scipy.signal) が見つかりません。SciPy が必要です。")


# ---------------- color style ----------------

PALETTE = {
    "blue": "#92B1D9",
    "pale_blue": "#C1D8E9",
    "lavender": "#DBDDEF",
    "peach": "#F6C8B6",
    "gray": "#D4D4D4",
    "dark": "#4A4A4A",
}

POS_COLORS = [
    PALETTE["blue"],
    PALETTE["pale_blue"],
    PALETTE["lavender"],
]

ATT_COLORS = [
    PALETTE["blue"],
    PALETTE["lavender"],
    PALETTE["peach"],
]

TRAJ_COLOR = PALETTE["blue"]
EDGE_COLOR = PALETTE["dark"]


def _apply_paper_style():
    """
    論文図版風の淡色スタイルを適用
    """
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(EDGE_COLOR)
    ax.spines["bottom"].set_color(EDGE_COLOR)
    ax.tick_params(colors=EDGE_COLOR)
    ax.xaxis.label.set_color(EDGE_COLOR)
    ax.yaxis.label.set_color(EDGE_COLOR)


# ---------------- utilities ----------------

def _zoh_hold(t_src, y_src, t_dst):
    """Zero-Order Hold（最後の値を保持）"""
    t_src = np.asarray(t_src, float)
    y_src = np.asarray(y_src)
    t_dst = np.asarray(t_dst, float)

    m = np.isfinite(t_src) & np.isfinite(y_src)
    t_src, y_src = t_src[m], y_src[m]

    if len(t_src) == 0:
        return np.full_like(t_dst, np.nan, dtype=float)

    idx = np.argsort(t_src)
    t_src, y_src = t_src[idx], y_src[idx]

    j = np.searchsorted(t_src, t_dst, side="right") - 1

    y = np.full_like(t_dst, np.nan, dtype=float)
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

    if (not np.isfinite(fs)) or fs <= 0:
        return x.copy()

    if (not np.isfinite(fc_hz)) or fc_hz <= 0:
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


def _sort_by_time(t, *arrays):
    """時刻昇順に並べ替える"""
    t = np.asarray(t)
    idx = np.argsort(t)

    out = [t[idx]]
    for a in arrays:
        out.append(np.asarray(a)[idx])

    return out


def _finite_mask(*arrays):
    m = np.ones(len(arrays[0]), dtype=bool)

    for a in arrays:
        a = np.asarray(a)
        m &= np.isfinite(a)

    return m


def _colored_violinplot(data_list, labels, ylabel, colors, alpha=0.82):
    """
    1つのグラフ内に複数のバイオリンプロットを並べて描く
    """
    plt.figure()

    parts = plt.violinplot(
        data_list,
        showmeans=True,
        showmedians=True,
    )

    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor(EDGE_COLOR)
        body.set_alpha(alpha)
        body.set_linewidth(0.8)

    if "cmeans" in parts:
        parts["cmeans"].set_edgecolor(EDGE_COLOR)
        parts["cmeans"].set_linewidth(1.0)

    if "cmedians" in parts:
        parts["cmedians"].set_edgecolor(PALETTE["peach"])
        parts["cmedians"].set_linewidth(1.4)

    if "cbars" in parts:
        parts["cbars"].set_edgecolor(EDGE_COLOR)
        parts["cbars"].set_linewidth(0.9)

    if "cmins" in parts:
        parts["cmins"].set_edgecolor(EDGE_COLOR)
        parts["cmins"].set_linewidth(0.9)

    if "cmaxes" in parts:
        parts["cmaxes"].set_edgecolor(EDGE_COLOR)
        parts["cmaxes"].set_linewidth(0.9)

    plt.xticks(range(1, len(labels) + 1), labels)
    plt.ylabel(ylabel)

    _apply_paper_style()


def _safe_get_target_p(msg, name):
    """
    msg.<name>.target_p を安全に取得する。

    例:
        msg.x.target_p
        msg.y.target_p
        msg.z.target_p
        msg.roll.target_p
        msg.pitch.target_p
        msg.yaw.target_p
    """
    try:
        return float(getattr(getattr(msg, name), "target_p"))
    except Exception:
        return np.nan


# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bag", help="rosbag file path")

    # actual topics
    ap.add_argument(
        "--pos-actual-topic",
        default="/crobat/uav/cog/odom",
        help="nav_msgs/Odometry; use msg.pose.pose.position.{x,y,z}",
    )
    ap.add_argument(
        "--att-actual-topic",
        default="/crobat/uav/baselink/odom",
        help="nav_msgs/Odometry; use msg.pose.pose.orientation (quat->rpy)",
    )

    # target topic
    ap.add_argument(
        "--target-topic",
        default="/crobat/debug/pose/pid",
        help="target topic; use msg.x.target_p (float)",
    )

    ap.add_argument(
        "--flight-state-topic",
        default="/crobat/flight_state",
        help="std_msgs/Int* などを想定。msg.data を使用",
    )

    # filter condition
    ap.add_argument(
        "--flight-state-value",
        type=int,
        default=5,
        help="この値のときだけデータを抽出する（default: 5）",
    )

    # lowpass for plots
    ap.add_argument(
        "--plot-lp-fc-pos",
        type=float,
        default=5.0,
        help="lowpass cutoff [Hz] for position plot (0 to disable)",
    )
    ap.add_argument(
        "--plot-lp-fc-att",
        type=float,
        default=8.0,
        help="lowpass cutoff [Hz] for attitude plot (0 to disable)",
    )
    ap.add_argument(
        "--plot-lp-order",
        type=int,
        default=4,
        help="lowpass filter order (Butterworth)",
    )

    ap.add_argument(
        "--error-sign",
        choices=["actual-minus-target", "target-minus-actual"],
        default="actual-minus-target",
        help="error の符号。default: actual-minus-target",
    )

    ap.add_argument(
        "--attitude-in-deg",
        action="store_true",
        help="姿勢を rad から deg に変換して描画・error 計算する",
    )

    args = ap.parse_args()

    # ---------- storage ----------

    # actual position
    t_pa, x_act, y_act, z_act = [], [], [], []

    # actual attitude
    t_aa, roll_act, pitch_act, yaw_act = [], [], [], []

    # target position / attitude
    t_tg = []
    x_tgt, y_tgt, z_tgt = [], [], []
    roll_tgt, pitch_tgt, yaw_tgt = [], [], []

    # flight state
    t_fs, fs_val = [], []

    topics = [
        args.pos_actual_topic,
        args.att_actual_topic,
        args.target_topic,
        args.flight_state_topic,
    ]
    topics = list(dict.fromkeys(topics))

    with rosbag.Bag(args.bag) as bag:
        for topic, msg, t in bag.read_messages(topics=topics):
            ts = t.to_sec()

            # flight state
            if topic == args.flight_state_topic:
                try:
                    t_fs.append(ts)
                    fs_val.append(int(msg.data))
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
                    roll_act.append(float(r))
                    pitch_act.append(float(p))
                    yaw_act.append(float(y))
                except Exception:
                    pass

            # target pose
            if topic == args.target_topic:
                try:
                    t_tg.append(ts)

                    # position target
                    x_tgt.append(_safe_get_target_p(msg, "x"))
                    y_tgt.append(_safe_get_target_p(msg, "y"))
                    z_tgt.append(_safe_get_target_p(msg, "z"))

                    # attitude target
                    # msg.roll.target_p などが存在しない場合は NaN になります
                    roll_tgt.append(_safe_get_target_p(msg, "roll"))
                    pitch_tgt.append(_safe_get_target_p(msg, "pitch"))
                    yaw_tgt.append(_safe_get_target_p(msg, "yaw"))

                except Exception:
                    pass

    # ---------- numpy ----------

    t_pa = np.asarray(t_pa, float)
    x_act = np.asarray(x_act, float)
    y_act = np.asarray(y_act, float)
    z_act = np.asarray(z_act, float)

    t_aa = np.asarray(t_aa, float)
    roll_act = np.asarray(roll_act, float)
    pitch_act = np.asarray(pitch_act, float)
    yaw_act = np.asarray(yaw_act, float)

    t_tg = np.asarray(t_tg, float)
    x_tgt = np.asarray(x_tgt, float)
    y_tgt = np.asarray(y_tgt, float)
    z_tgt = np.asarray(z_tgt, float)
    roll_tgt = np.asarray(roll_tgt, float)
    pitch_tgt = np.asarray(pitch_tgt, float)
    yaw_tgt = np.asarray(yaw_tgt, float)

    t_fs = np.asarray(t_fs, float)
    fs_val = np.asarray(fs_val, float)

    print(
        "counts before filter:",
        "pos_act", len(t_pa),
        "att_act", len(t_aa),
        "target", len(t_tg),
        "flight_state", len(t_fs),
    )
    print(
        "topics:",
        "pos_act", args.pos_actual_topic,
        "att_act", args.att_actual_topic,
        "target", args.target_topic,
        "flight_state", args.flight_state_topic,
    )
    print("flight_state filter value:", args.flight_state_value)
    print("error sign:", args.error_sign)

    if len(t_pa) < 2:
        raise ValueError("Position actual topic has too few samples (need >=2).")

    if len(t_aa) < 2:
        raise ValueError("Attitude actual topic has too few samples (need >=2).")

    if len(t_tg) < 1:
        raise ValueError("Target topic has no valid samples.")

    if len(t_fs) < 1:
        raise ValueError("Flight state topic has no valid samples.")

    # ---------- sort actual ----------

    m_pa = _finite_mask(t_pa, x_act, y_act, z_act)
    t_pa, x_act, y_act, z_act = (
        t_pa[m_pa],
        x_act[m_pa],
        y_act[m_pa],
        z_act[m_pa],
    )
    t_pa, x_act, y_act, z_act = _sort_by_time(t_pa, x_act, y_act, z_act)

    m_aa = _finite_mask(t_aa, roll_act, pitch_act, yaw_act)
    t_aa, roll_act, pitch_act, yaw_act = (
        t_aa[m_aa],
        roll_act[m_aa],
        pitch_act[m_aa],
        yaw_act[m_aa],
    )
    t_aa, roll_act, pitch_act, yaw_act = _sort_by_time(
        t_aa,
        roll_act,
        pitch_act,
        yaw_act,
    )

    # ---------- sort target ----------

    m_tg_pos = _finite_mask(t_tg, x_tgt, y_tgt, z_tgt)
    t_tg_pos = t_tg[m_tg_pos]
    x_tgt_pos = x_tgt[m_tg_pos]
    y_tgt_pos = y_tgt[m_tg_pos]
    z_tgt_pos = z_tgt[m_tg_pos]

    if len(t_tg_pos) < 1:
        raise ValueError(
            "Target topic に position target が見つかりません。"
            " msg.x.target_p, msg.y.target_p, msg.z.target_p を確認してください。"
        )

    t_tg_pos, x_tgt_pos, y_tgt_pos, z_tgt_pos = _sort_by_time(
        t_tg_pos,
        x_tgt_pos,
        y_tgt_pos,
        z_tgt_pos,
    )

    m_tg_att = _finite_mask(t_tg, roll_tgt, pitch_tgt, yaw_tgt)
    t_tg_att = t_tg[m_tg_att]
    roll_tgt_att = roll_tgt[m_tg_att]
    pitch_tgt_att = pitch_tgt[m_tg_att]
    yaw_tgt_att = yaw_tgt[m_tg_att]

    has_att_target = len(t_tg_att) >= 1

    if has_att_target:
        t_tg_att, roll_tgt_att, pitch_tgt_att, yaw_tgt_att = _sort_by_time(
            t_tg_att,
            roll_tgt_att,
            pitch_tgt_att,
            yaw_tgt_att,
        )
    else:
        print(
            "warning: attitude target が見つかりません。"
            " attitude error violin は描画しません。"
        )

    # ---------- sort flight_state ----------

    m_fs = _finite_mask(t_fs, fs_val)
    t_fs, fs_val = t_fs[m_fs], fs_val[m_fs]
    t_fs, fs_val = _sort_by_time(t_fs, fs_val)

    # ---------- filter by flight_state ----------

    fs_on_pa = _zoh_hold(t_fs, fs_val, t_pa)
    mask_pa = fs_on_pa == args.flight_state_value

    fs_on_aa = _zoh_hold(t_fs, fs_val, t_aa)
    mask_aa = fs_on_aa == args.flight_state_value

    t_pa = t_pa[mask_pa]
    x_act = x_act[mask_pa]
    y_act = y_act[mask_pa]
    z_act = z_act[mask_pa]

    t_aa = t_aa[mask_aa]
    roll_act = roll_act[mask_aa]
    pitch_act = pitch_act[mask_aa]
    yaw_act = yaw_act[mask_aa]

    print(
        "counts after flight_state filter:",
        "pos_act", len(t_pa),
        "att_act", len(t_aa),
    )

    if len(t_pa) < 2:
        raise ValueError(
            f"flight_state == {args.flight_state_value} の position データが少なすぎます。"
        )

    if len(t_aa) < 2:
        raise ValueError(
            f"flight_state == {args.flight_state_value} の attitude データが少なすぎます。"
        )

    # ---------- convert attitude unit ----------

    if args.attitude_in_deg:
        roll_act_unit = np.rad2deg(roll_act)
        pitch_act_unit = np.rad2deg(pitch_act)
        yaw_act_unit = np.rad2deg(yaw_act)

        if has_att_target:
            roll_tgt_att = np.rad2deg(roll_tgt_att)
            pitch_tgt_att = np.rad2deg(pitch_tgt_att)
            yaw_tgt_att = np.rad2deg(yaw_tgt_att)

        att_ylabel = "attitude [deg]"
        att_err_ylabel = "attitude error [deg]"
    else:
        roll_act_unit = roll_act
        pitch_act_unit = pitch_act
        yaw_act_unit = yaw_act

        att_ylabel = "attitude [rad]"
        att_err_ylabel = "attitude error [rad]"

    # ---------- lowpass for plots ----------

    fs_pos = _estimate_fs(t_pa)
    fs_att = _estimate_fs(t_aa)

    if args.plot_lp_fc_pos > 0:
        x_plot = _lowpass_filtfilt(
            x_act,
            fs_pos,
            args.plot_lp_fc_pos,
            order=args.plot_lp_order,
        )
        y_plot = _lowpass_filtfilt(
            y_act,
            fs_pos,
            args.plot_lp_fc_pos,
            order=args.plot_lp_order,
        )
        z_plot = _lowpass_filtfilt(
            z_act,
            fs_pos,
            args.plot_lp_fc_pos,
            order=args.plot_lp_order,
        )
    else:
        x_plot, y_plot, z_plot = x_act, y_act, z_act

    if args.plot_lp_fc_att > 0:
        roll_plot = _lowpass_filtfilt(
            roll_act_unit,
            fs_att,
            args.plot_lp_fc_att,
            order=args.plot_lp_order,
        )
        pitch_plot = _lowpass_filtfilt(
            pitch_act_unit,
            fs_att,
            args.plot_lp_fc_att,
            order=args.plot_lp_order,
        )
        yaw_plot = _lowpass_filtfilt(
            yaw_act_unit,
            fs_att,
            args.plot_lp_fc_att,
            order=args.plot_lp_order,
        )
    else:
        roll_plot, pitch_plot, yaw_plot = (
            roll_act_unit,
            pitch_act_unit,
            yaw_act_unit,
        )

    # ---------- target hold on actual timestamps ----------

    x_tgt_on_pa = _zoh_hold(t_tg_pos, x_tgt_pos, t_pa)
    y_tgt_on_pa = _zoh_hold(t_tg_pos, y_tgt_pos, t_pa)
    z_tgt_on_pa = _zoh_hold(t_tg_pos, z_tgt_pos, t_pa)

    if args.error_sign == "actual-minus-target":
        x_err = x_plot - x_tgt_on_pa
        y_err = y_plot - y_tgt_on_pa
        z_err = z_plot - z_tgt_on_pa
    else:
        x_err = x_tgt_on_pa - x_plot
        y_err = y_tgt_on_pa - y_plot
        z_err = z_tgt_on_pa - z_plot

    if has_att_target:
        roll_tgt_on_aa = _zoh_hold(t_tg_att, roll_tgt_att, t_aa)
        pitch_tgt_on_aa = _zoh_hold(t_tg_att, pitch_tgt_att, t_aa)
        yaw_tgt_on_aa = _zoh_hold(t_tg_att, yaw_tgt_att, t_aa)

        if args.error_sign == "actual-minus-target":
            roll_err = roll_plot - roll_tgt_on_aa
            pitch_err = pitch_plot - pitch_tgt_on_aa
            yaw_err = yaw_plot - yaw_tgt_on_aa
        else:
            roll_err = roll_tgt_on_aa - roll_plot
            pitch_err = pitch_tgt_on_aa - pitch_plot
            yaw_err = yaw_tgt_on_aa - yaw_plot
    else:
        roll_err = np.asarray([], float)
        pitch_err = np.asarray([], float)
        yaw_err = np.asarray([], float)

    # ---------- plots ----------

    # 1) actual position x,y,z time series
    plt.figure()
    tp = t_pa - t_pa[0]

    plt.plot(
        tp,
        x_plot,
        linewidth=1.8,
        color=POS_COLORS[0],
        label="x actual",
    )
    plt.plot(
        tp,
        y_plot,
        linewidth=1.8,
        color=POS_COLORS[1],
        label="y actual",
    )
    plt.plot(
        tp,
        z_plot,
        linewidth=1.8,
        color=POS_COLORS[2],
        label="z actual",
    )

    plt.xlabel("time [s]")
    plt.ylabel("position [m]")
    plt.legend(frameon=False)
    _apply_paper_style()

    # 2) actual attitude roll,pitch,yaw time series
    plt.figure()
    ta = t_aa - t_aa[0]

    plt.plot(
        ta,
        roll_plot,
        linewidth=1.8,
        color=ATT_COLORS[0],
        label="roll actual",
    )
    plt.plot(
        ta,
        pitch_plot,
        linewidth=1.8,
        color=ATT_COLORS[1],
        label="pitch actual",
    )
    plt.plot(
        ta,
        yaw_plot,
        linewidth=1.8,
        color=ATT_COLORS[2],
        label="yaw actual",
    )

    plt.xlabel("time [s]")
    plt.ylabel(att_ylabel)
    plt.legend(frameon=False)
    _apply_paper_style()

    # 3) actual XY trajectory
    plt.figure()
    m_xy = np.isfinite(x_plot) & np.isfinite(y_plot)

    plt.plot(
        x_plot[m_xy],
        y_plot[m_xy],
        linewidth=1.8,
        color=TRAJ_COLOR,
    )

    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.axis("equal")
    _apply_paper_style()

    # 4) position error time series
    plt.figure()

    plt.plot(
        tp,
        x_err,
        linewidth=1.8,
        color=POS_COLORS[0],
        label="x error",
    )
    plt.plot(
        tp,
        y_err,
        linewidth=1.8,
        color=POS_COLORS[1],
        label="y error",
    )
    plt.plot(
        tp,
        z_err,
        linewidth=1.8,
        color=POS_COLORS[2],
        label="z error",
    )

    plt.axhline(0.0, color=EDGE_COLOR, linewidth=0.8, alpha=0.6)
    plt.xlabel("time [s]")
    plt.ylabel("position error [m]")
    plt.legend(frameon=False)
    _apply_paper_style()

    # 5) attitude error time series
    if has_att_target:
        plt.figure()

        plt.plot(
            ta,
            roll_err,
            linewidth=1.8,
            color=ATT_COLORS[0],
            label="roll error",
        )
        plt.plot(
            ta,
            pitch_err,
            linewidth=1.8,
            color=ATT_COLORS[1],
            label="pitch error",
        )
        plt.plot(
            ta,
            yaw_err,
            linewidth=1.8,
            color=ATT_COLORS[2],
            label="yaw error",
        )

        plt.axhline(0.0, color=EDGE_COLOR, linewidth=0.8, alpha=0.6)
        plt.xlabel("time [s]")
        plt.ylabel(att_err_ylabel)
        plt.legend(frameon=False)
        _apply_paper_style()

    # 6) violin: position error x,y,z in one figure
    _colored_violinplot(
        [
            x_err[np.isfinite(x_err)],
            y_err[np.isfinite(y_err)],
            z_err[np.isfinite(z_err)],
        ],
        ["x error", "y error", "z error"],
        "position error [m]",
        ATT_COLORS,
    )

    # 7) violin: attitude error roll,pitch,yaw in one figure
    if has_att_target:
        _colored_violinplot(
            [
                roll_err[np.isfinite(roll_err)],
                pitch_err[np.isfinite(pitch_err)],
                yaw_err[np.isfinite(yaw_err)],
            ],
            ["roll error", "pitch error", "yaw error"],
            att_err_ylabel,
            ATT_COLORS,
        )

    plt.show()


if __name__ == "__main__":
    main()
