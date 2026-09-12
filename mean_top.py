#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt

try:
    import rosbag
except ImportError:
    raise SystemExit("ROS1 環境で実行してください（rosbag が必要です）。")

@dataclass
class Sample:
    t: float
    v: float

def time_of(msg, bag_t) -> float:
    if hasattr(msg, "header") and hasattr(msg.header, "stamp"):
        try:
            return float(msg.header.stamp.to_sec())
        except Exception:
            pass
    return float(bag_t.to_sec())

def read_pwm0_and_fz(bag_path: str, wrench_topic: str, pwm_topic: str) -> Tuple[List[Sample], List[Sample]]:
    pwm_samples: List[Sample] = []
    fz_samples: List[Sample] = []
    with rosbag.Bag(bag_path, "r") as bag:
        for topic, msg, t in bag.read_messages(topics=[wrench_topic, pwm_topic]):
            ts = time_of(msg, t)
            if topic == pwm_topic:
                if hasattr(msg, "pwms") and msg.pwms is not None and len(msg.pwms) >= 1:
                    pwm_samples.append(Sample(ts, float(msg.pwms[0])))
            else:
                try:
                    fz_samples.append(Sample(ts, float(msg.wrench.force.z)))
                except Exception:
                    continue

    pwm_samples.sort(key=lambda s: s.t)
    fz_samples.sort(key=lambda s: s.t)
    return pwm_samples, fz_samples

def assign_pwm_to_force(pwm_samples: List[Sample], fz_samples: List[Sample]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(pwm_samples) == 0:
        raise ValueError("PWMサンプルが空です。")
    if len(fz_samples) == 0:
        raise ValueError("Wrenchサンプルが空です。")

    t_pwm = np.array([s.t for s in pwm_samples], dtype=float)
    pwm = np.array([s.v for s in pwm_samples], dtype=float)

    t_fz = np.array([s.t for s in fz_samples], dtype=float)
    fz = np.array([s.v for s in fz_samples], dtype=float) / 4.0

    idx = np.searchsorted(t_pwm, t_fz, side="right") - 1
    valid = idx >= 0
    t_fz = t_fz[valid]
    fz = fz[valid]
    idx = idx[valid]
    pwm_at_fz = pwm[idx]
    return t_fz, fz, pwm_at_fz

def round_pwm_values(pwm: np.ndarray, step: float) -> np.ndarray:
    if step <= 0:
        raise ValueError("--pwm_bin は正である必要があります。")
    return np.round(pwm / step) * step

def compute_pwm_stats(
    pwm_vals: np.ndarray,
    fz_vals: np.ndarray,
    top_ratio: float
) -> List[Dict[str, float]]:
    if not (0 < top_ratio <= 1.0):
        raise ValueError("--top_ratio は 0 < top_ratio <= 1.0 である必要があります。")

    results = []
    unique_pwms = np.unique(pwm_vals)

    for p in unique_pwms:
        mask = pwm_vals == p
        f = fz_vals[mask]
        if f.size == 0:
            continue

        f_sorted = np.sort(f)
        n_top = max(1, int(np.ceil(f.size * top_ratio)))
        top_f = f_sorted[-n_top:]

        results.append({
            "pwm": float(p),
            "count": int(f.size),
            "mean": float(np.mean(f)),
            "max": float(np.max(f)),
            "mean_of_top": float(np.mean(top_f)),   # ← 平均最大値
            "std": float(np.std(f, ddof=0)),
        })

    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True)
    ap.add_argument("--wrench_topic", default="/cfs/data")
    ap.add_argument("--pwm_topic", default="/pwm_test")

    # 時間窓
    ap.add_argument("--t_start", type=float, default=1763653162.81, help="絶対開始時刻（基準）")
    ap.add_argument("--offset", type=float, default=1.0, help="t_startから何秒後に窓を開始するか")
    ap.add_argument("--duration", type=float, default=1.0, help="窓の長さ[sec]")

    # PWM集約用
    ap.add_argument("--pwm_bin", type=float, default=0.01, help="PWMをこの刻みで丸めてグループ化")
    ap.add_argument("--top_ratio", type=float, default=0.1, help="上位何割を平均最大値に使うか")
    ap.add_argument("--min_count", type=int, default=1, help="この数未満のサンプルしかないPWMは除外")

    args = ap.parse_args()

    if args.duration <= 0:
        raise ValueError("--duration は正である必要があります。")
    if args.min_count <= 0:
        raise ValueError("--min_count は正である必要があります。")

    pwm_samples, fz_samples = read_pwm0_and_fz(args.bag, args.wrench_topic, args.pwm_topic)
    t_fz, fz, pwm_at_fz = assign_pwm_to_force(pwm_samples, fz_samples)

    t0 = args.t_start + args.offset
    t1 = t0 + args.duration

    # 時間窓で抽出
    mask = (t_fz >= t0) & (t_fz <= t1)
    if not np.any(mask):
        raise RuntimeError(f"指定区間 [{t0:.6f}, {t1:.6f}] にサンプルがありません。")

    t_sel = t_fz[mask]
    fz_sel = fz[mask]
    pwm_sel = pwm_at_fz[mask]

    # PWMを丸めてグループ化
    pwm_grouped = round_pwm_values(pwm_sel, args.pwm_bin)

    stats = compute_pwm_stats(pwm_grouped, fz_sel, args.top_ratio)
    stats = [s for s in stats if s["count"] >= args.min_count]

    if len(stats) == 0:
        raise RuntimeError("条件を満たすPWMグループがありません。")

    stats.sort(key=lambda x: x["pwm"])

    print("=== Window ===")
    print(f"t_start   = {args.t_start:.6f}")
    print(f"offset    = {args.offset:.6f}")
    print(f"duration  = {args.duration:.6f}")
    print(f"window    = [{t0:.6f}, {t1:.6f}]")
    print(f"pwm_bin   = {args.pwm_bin}")
    print(f"top_ratio = {args.top_ratio}")
    print(f"min_count = {args.min_count}")

    print("\n=== Per-PWM stats ===")
    for s in stats:
        print(
            f"PWM={s['pwm']:.4f}, "
            f"count={s['count']}, "
            f"mean={s['mean']:.6f}, "
            f"max={s['max']:.6f}, "
            f"mean_of_top={s['mean_of_top']:.6f}, "
            f"std={s['std']:.6f}"
        )

    # プロット用
    pwms = np.array([s["pwm"] for s in stats], dtype=float)
    means = np.array([s["mean"] for s in stats], dtype=float)
    maxs = np.array([s["max"] for s in stats], dtype=float)
    mean_of_tops = np.array([s["mean_of_top"] for s in stats], dtype=float)

    x = np.arange(len(pwms))
    w = 0.25

    plt.figure(figsize=(10, 5))
    plt.bar(x - w, means, width=w, label="mean")
    plt.bar(x, mean_of_tops, width=w, label="mean_of_top")
    plt.bar(x + w, maxs, width=w, label="max")
    plt.xticks(x, [f"{p:.2f}" for p in pwms], rotation=45)
    plt.xlabel("PWM")
    plt.ylabel("Force Z")
    plt.title("Mean / Mean-of-top / Max for each PWM")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
