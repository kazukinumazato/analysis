#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from dataclasses import dataclass
from typing import List, Tuple

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
    # header.stampがあればそれを採用、無ければbag時刻
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
                # wrench_topic
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
    fz = np.array([s.v for s in fz_samples], dtype=float) / 4

    idx = np.searchsorted(t_pwm, t_fz, side="right") - 1
    valid = idx >= 0
    t_fz = t_fz[valid]
    fz = fz[valid]
    idx = idx[valid]
    pwm_at_fz = pwm[idx]
    return t_fz, fz, pwm_at_fz

def bin_means(t: np.ndarray, fz: np.ndarray, t0: float, t1: float, n_bins: int) -> Tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(t0, t1, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    means = np.full(n_bins, np.nan, dtype=float)

    for k in range(n_bins):
        a, b = edges[k], edges[k + 1]
        # 最終ビンのみ右端含む
        mask = (t >= a) & (t < b) if k < n_bins - 1 else (t >= a) & (t <= b)
        if np.any(mask):
            means[k] = float(np.mean(fz[mask]))
    return centers, means

def compute_stats(x: np.ndarray) -> dict:
    xv = x[~np.isnan(x)]
    if xv.size == 0:
        raise ValueError("ビン平均が全てNaNです（窓内にサンプル無し）。")
    out = {
        "count_valid": int(xv.size),
        "mean": float(np.mean(xv)),
        "median": float(np.median(xv)),
        "min": float(np.min(xv)),
        "max": float(np.max(xv)),
        "var_pop": float(np.var(xv, ddof=0)),
        "std_pop": float(np.std(xv, ddof=0)),
    }
    if xv.size >= 2:
        out["var_sample"] = float(np.var(xv, ddof=1))
        out["std_sample"] = float(np.std(xv, ddof=1))
    else:
        out["var_sample"] = float("nan")
        out["std_sample"] = float("nan")
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True)
    ap.add_argument("--wrench_topic", default="/cfs/data")
    ap.add_argument("--pwm_topic", default="/pwm_test")
    ap.add_argument("--target_pwm", type=float, default=0.79)
    ap.add_argument("--tol", type=float, default=0.02)

    # ★ ここで区間を自由化
    ap.add_argument("--t_start", type=float, default=1763653154.32, help="絶対開始時刻（基準）")
    ap.add_argument("--offset", type=float, default=1.0, help="t_startから何秒後に窓を開始するか")
    ap.add_argument("--duration", type=float, default=0.5, help="窓の長さ[sec]")
    ap.add_argument("--n_bins", type=int, default=10, help="窓を何分割するか")
    args = ap.parse_args()

    if args.duration <= 0:
        raise ValueError("--duration は正である必要があります。")
    if args.n_bins <= 0:
        raise ValueError("--n_bins は正である必要があります。")

    pwm_samples, fz_samples = read_pwm0_and_fz(args.bag, args.wrench_topic, args.pwm_topic)
    t_fz, fz, pwm_at_fz = assign_pwm_to_force(pwm_samples, fz_samples)

    t0 = args.t_start + args.offset
    t1 = t0 + args.duration

    # 解析対象：時間窓 & PWM条件
    mask = (
        (t_fz >= t0) & (t_fz <= t1) &
        (np.abs(pwm_at_fz - args.target_pwm) <= args.tol)
    )
    if not np.any(mask):
        raise RuntimeError(
            f"指定区間 [{t0:.6f}, {t1:.6f}] に PWM≈{args.target_pwm}±{args.tol} のWrenchサンプルがありません。\n"
            "対処: --tol を増やす / --offset, --duration を変える / その区間で本当にPWMが0.8か確認"
        )

    t_sel = t_fz[mask]
    fz_sel = fz[mask]

    centers, means = bin_means(t_sel, fz_sel, t0, t1, n_bins=args.n_bins)
    stats = compute_stats(means)

    # 標準出力に列挙
    print("=== Window ===")
    print(f"t_start   = {args.t_start:.6f}")
    print(f"offset    = {args.offset:.6f}")
    print(f"duration  = {args.duration:.6f}")
    print(f"window    = [{t0:.6f}, {t1:.6f}]")
    print(f"PWM cond  = {args.target_pwm} ± {args.tol}")
    print(f"n_bins    = {args.n_bins}")

    print("\n=== Bin means (Fz) ===")
    for i, (tc, m) in enumerate(zip(centers, means), start=1):
        print(f"bin{i:02d}: center_time={tc:.6f}, mean_fz={m}")

    print("\n=== Stats over bin means (ignore NaN) ===")
    for k, v in stats.items():
        print(f"{k}: {v}")

    # 棒グラフ（保存なし・凡例なし）
    width = (t1 - t0) / args.n_bins * 0.9
    plt.figure()
    plt.bar(centers, means, width=width)
    plt.xlabel("Time [s] (absolute, bin centers)")
    plt.ylabel("Mean Force Z (wrench.force.z)")
    plt.title(f"Bin-mean Fz in window (PWM≈{args.target_pwm})")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
