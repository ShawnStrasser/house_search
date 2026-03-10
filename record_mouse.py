#!/usr/bin/env python3
"""
Mouse movement recorder — statistical profiler.

Records your real movements, then EXTRACTS statistical parameters
from them. The raw movements are discarded; only the fingerprint is saved.

Output: mouse_profile.json  (statistical parameters, NOT raw coordinates)

Usage:
    python record_mouse.py

Follow the prompts. Two phases:
  1. APPROACH  — move naturally from one side of the screen to a target
  2. HOLD      — rest hand on mouse, hold left button for ~8 seconds

The solver will use these statistics to SYNTHESIZE new movements that
match your style, never replaying your actual data.
"""

import json
import time
import math
import random
import sys
import numpy as np
from pathlib import Path

import win32api
import win32con

PROFILE_PATH = r"C:\Users\shawn\House\mouse_profile.json"
SAMPLE_HZ    = 125


# ---------------------------------------------------------------------------
# Recording helpers
# ---------------------------------------------------------------------------

def _countdown(seconds: int, msg: str):
    print(f"\n{msg}")
    for i in range(seconds, 0, -1):
        print(f"  {i}...", flush=True)
        time.sleep(1)
    print("  NOW!\n")


def _poll(duration_s: float, label: str) -> list[tuple[int, int, float]]:
    """
    Poll cursor at SAMPLE_HZ.
    Returns list of (abs_x, abs_y, timestamp_s).
    """
    print(f"  Recording '{label}' for {duration_s}s...")
    samples = []
    interval = 1.0 / SAMPLE_HZ
    t_start = time.perf_counter()
    t_end   = t_start + duration_s
    t_prev  = t_start

    while True:
        now = time.perf_counter()
        if now >= t_end:
            break
        sleep_for = interval - (now - t_prev)
        if sleep_for > 0:
            time.sleep(sleep_for)
        now = time.perf_counter()
        x, y = win32api.GetCursorPos()
        samples.append((x, y, now - t_start))
        t_prev = now

    print(f"  Captured {len(samples)} samples.")
    return samples


# ---------------------------------------------------------------------------
# Statistical extraction — Approach
# ---------------------------------------------------------------------------

def extract_approach_stats(samples: list[tuple[int, int, float]]) -> dict:
    """
    Decompose approach movement into statistical parameters.

    Parameters extracted:
      - speed_profile_shape : alpha, beta of a Beta distribution fit to the
                              normalised velocity curve (captures acceleration
                              and deceleration character)
      - curvature_std       : std of perpendicular deviation from straight line
                              (how curved/wiggly the path is)
      - correction_rate     : fraction of frames with a direction reversal
                              (captures micro-adjustment frequency)
      - overshoot_fraction  : how far past the target the hand typically goes
                              before correcting, as fraction of total distance
      - total_duration_s    : typical movement duration (used to scale replay)
      - speed_mean_frac     : mean normalised speed (0-1), backup for synthesis
      - speed_std_frac      : std of normalised speed
    """
    if len(samples) < 10:
        return {}

    xs = np.array([s[0] for s in samples], dtype=float)
    ys = np.array([s[1] for s in samples], dtype=float)
    ts = np.array([s[2] for s in samples], dtype=float)

    # Filter out segments with no movement (sitting still at start/end)
    dists = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2)
    moving = np.where(dists > 0.5)[0]
    if len(moving) < 5:
        return {}
    start_i, end_i = moving[0], moving[-1] + 1
    xs, ys, ts = xs[start_i:end_i+1], ys[start_i:end_i+1], ts[start_i:end_i+1]

    dts   = np.diff(ts)
    dts   = np.where(dts < 1e-6, 1e-6, dts)  # avoid division by zero
    dx    = np.diff(xs)
    dy    = np.diff(ys)
    speed = np.sqrt(dx**2 + dy**2) / dts

    # Normalise speed to [0, 1]
    max_speed = speed.max()
    if max_speed < 1e-6:
        return {}
    speed_norm = speed / max_speed

    # Fit Beta distribution to normalised speed profile
    # Beta(alpha, beta) captures the ramp-up/ramp-down shape
    mean_s = float(speed_norm.mean())
    var_s  = float(speed_norm.var())
    var_s  = max(var_s, 1e-6)
    # Method of moments
    common = (mean_s * (1 - mean_s) / var_s) - 1
    alpha  = max(0.5, mean_s * common)
    beta_p = max(0.5, (1 - mean_s) * common)

    # Curvature: perpendicular deviation from straight line start->end
    p0 = np.array([xs[0], ys[0]])
    p1 = np.array([xs[-1], ys[-1]])
    seg = p1 - p0
    seg_len = np.linalg.norm(seg)
    if seg_len > 1:
        perp = np.column_stack([-(ys - xs[0] * 0 - ys[0]), xs - xs[0]])  # placeholder
        # Proper perpendicular distance: |cross(seg_unit, point-p0)|
        seg_unit = seg / seg_len
        vecs = np.column_stack([xs - p0[0], ys - p0[1]])
        cross = vecs[:, 0] * seg_unit[1] - vecs[:, 1] * seg_unit[0]
        curvature_std = float(np.std(cross))
    else:
        curvature_std = 0.0

    # Direction reversals
    angles = np.arctan2(dy, dx)
    angle_diff = np.abs(np.diff(angles))
    # Wrap to [-pi, pi]
    angle_diff = np.where(angle_diff > np.pi, 2*np.pi - angle_diff, angle_diff)
    reversal_threshold = np.pi * 0.4   # >72 degrees = reversal
    correction_rate = float((angle_diff > reversal_threshold).mean())

    # Overshoot: does the cursor pass beyond the endpoint?
    # Project all points onto the line; overshoot = max projection > seg_len
    if seg_len > 1:
        projections = vecs[:, 0] * seg_unit[0] + vecs[:, 1] * seg_unit[1]
        max_proj = float(projections.max())
        overshoot_fraction = max(0.0, (max_proj - seg_len) / seg_len)
    else:
        overshoot_fraction = 0.0

    total_duration = float(ts[-1] - ts[0])

    return {
        "speed_alpha":        round(alpha, 4),
        "speed_beta":         round(beta_p, 4),
        "curvature_std_px":   round(curvature_std, 2),
        "correction_rate":    round(correction_rate, 4),
        "overshoot_fraction": round(overshoot_fraction, 4),
        "total_duration_s":   round(total_duration, 4),
        "speed_mean_frac":    round(mean_s, 4),
        "speed_std_frac":     round(float(speed_norm.std()), 4),
    }


# ---------------------------------------------------------------------------
# Statistical extraction — Hold tremor
# ---------------------------------------------------------------------------

def extract_hold_stats(samples: list[tuple[int, int, float]]) -> dict:
    """
    Decompose hold tremor into statistical parameters.

    Parameters extracted:
      - ou_theta_x/y     : Ornstein-Uhlenbeck mean-reversion speed per axis
                           (how quickly drift corrects itself)
      - ou_sigma_x/y     : OU noise magnitude per axis
      - tremor_freq_hz   : dominant physiological tremor frequency from FFT
      - tremor_amp_px    : amplitude of dominant frequency component
      - adjustment_prob  : probability per frame of a sudden micro-adjustment
      - adjustment_std_x/y : magnitude of micro-adjustments
      - drift_period_s   : period of slow fatigue drift (if present)
      - drift_amp_x/y    : amplitude of slow fatigue drift
    """
    if len(samples) < 50:
        return {}

    xs = np.array([s[0] for s in samples], dtype=float)
    ys = np.array([s[1] for s in samples], dtype=float)
    ts = np.array([s[2] for s in samples], dtype=float)

    # Remove overall linear drift trend (hand slowly moving across screen)
    xs -= np.polyval(np.polyfit(ts, xs, 1), ts)
    ys -= np.polyval(np.polyfit(ts, ys, 1), ys)

    dx = np.diff(xs)
    dy = np.diff(ys)
    dt_arr = np.diff(ts)
    dt_arr = np.where(dt_arr < 1e-6, 1e-6, dt_arr)

    # -- OU parameter estimation via method of moments on autocorrelation --
    # For OU: corr(dx_t, dx_{t-1}) = exp(-theta * dt)
    # => theta = -log(corr) / dt
    def _ou_params(d_arr, dt_mean):
        if len(d_arr) < 4:
            return 8.0, 0.3
        ac = float(np.corrcoef(d_arr[:-1], d_arr[1:])[0, 1])
        ac = max(-0.9999, min(0.9999, ac))
        theta = -math.log(abs(ac) + 1e-9) / max(dt_mean, 1e-4)
        theta = max(0.5, min(200.0, theta))
        sigma = float(np.std(d_arr)) * math.sqrt(2 * theta)
        sigma = max(0.01, sigma)
        return round(theta, 3), round(sigma, 4)

    dt_mean = float(dt_arr.mean())
    theta_x, sigma_x = _ou_params(dx, dt_mean)
    theta_y, sigma_y = _ou_params(dy, dt_mean)

    # -- FFT: find dominant tremor frequency --
    # Use the displacement signal (not deltas) on X axis
    n = len(xs)
    fft_vals = np.abs(np.fft.rfft(xs - xs.mean()))
    freqs    = np.fft.rfftfreq(n, d=dt_mean)
    # Ignore DC and very low frequencies (<1 Hz) and very high (>20 Hz — noise)
    mask = (freqs >= 1.0) & (freqs <= 20.0)
    if mask.any():
        idx = np.argmax(fft_vals[mask])
        tremor_freq = float(freqs[mask][idx])
        tremor_amp  = float(fft_vals[mask][idx]) * 2 / n  # normalise
    else:
        tremor_freq = 10.0
        tremor_amp  = 0.3

    # -- Micro-adjustment detection --
    # A micro-adjustment is a sudden jump: |delta| > 3 * median(|delta|)
    d_mag = np.sqrt(dx**2 + dy**2)
    median_mag = float(np.median(d_mag))
    threshold  = max(2.0, median_mag * 4)
    adj_mask   = d_mag > threshold
    adj_prob   = float(adj_mask.mean())
    if adj_mask.any():
        adj_std_x = float(np.std(np.abs(dx[adj_mask])))
        adj_std_y = float(np.std(np.abs(dy[adj_mask])))
    else:
        adj_std_x, adj_std_y = 1.0, 0.8

    # -- Slow fatigue drift: fit low-frequency sinusoid --
    # Use FFT on long-period band (0.05 – 1 Hz)
    low_mask = (freqs >= 0.05) & (freqs < 1.0)
    if low_mask.any():
        low_idx     = np.argmax(fft_vals[low_mask])
        drift_freq  = float(freqs[low_mask][low_idx])
        drift_period = 1.0 / max(drift_freq, 1e-3)
        drift_amp_x  = float(fft_vals[low_mask][low_idx]) * 2 / n
        # Y axis
        fft_y   = np.abs(np.fft.rfft(ys - ys.mean()))
        freqs_y = np.fft.rfftfreq(len(ys), d=dt_mean)
        lm_y    = (freqs_y >= 0.05) & (freqs_y < 1.0)
        drift_amp_y = float(fft_y[lm_y][np.argmax(fft_y[lm_y])]) * 2 / len(ys) if lm_y.any() else drift_amp_x
    else:
        drift_period = 28.0
        drift_amp_x  = 0.4
        drift_amp_y  = 0.35

    return {
        "ou_theta_x":      theta_x,
        "ou_theta_y":      theta_y,
        "ou_sigma_x":      sigma_x,
        "ou_sigma_y":      sigma_y,
        "tremor_freq_hz":  round(tremor_freq, 2),
        "tremor_amp_px":   round(tremor_amp, 4),
        "adjustment_prob": round(adj_prob, 5),
        "adjustment_std_x": round(adj_std_x, 3),
        "adjustment_std_y": round(adj_std_y, 3),
        "drift_period_s":  round(drift_period, 2),
        "drift_amp_x":     round(drift_amp_x, 4),
        "drift_amp_y":     round(drift_amp_y, 4),
    }


# ---------------------------------------------------------------------------
# Recording phases
# ---------------------------------------------------------------------------

def record_approach() -> dict:
    print("=" * 55)
    print("  PHASE 1: APPROACH MOVEMENT")
    print("=" * 55)
    print()
    print("  Move your mouse naturally from wherever it is to")
    print("  a target across the screen (like clicking a button).")
    print("  Move at your normal pace. 3 seconds of recording.")
    _countdown(3, "Get ready...")
    samples = _poll(3.0, "approach")
    stats   = extract_approach_stats(samples)
    if not stats:
        print("  WARNING: Not enough movement detected. Try again.")
        return {}
    print(f"  Speed shape: alpha={stats['speed_alpha']} beta={stats['speed_beta']}")
    print(f"  Curvature std: {stats['curvature_std_px']:.1f}px  |  "
          f"Correction rate: {stats['correction_rate']:.3f}  |  "
          f"Overshoot: {stats['overshoot_fraction']:.3f}")
    return stats


def record_hold() -> dict:
    print()
    print("=" * 55)
    print("  PHASE 2: HOLD TREMOR")
    print("=" * 55)
    print()
    print("  Rest your hand on the mouse. Press and HOLD the left")
    print("  button for about 8 seconds. Keep cursor roughly still.")
    print("  This captures your natural hand tremor signature.")
    _countdown(3, "Get ready to hold...")
    samples = _poll(8.0, "hold")
    stats   = extract_hold_stats(samples)
    if not stats:
        print("  WARNING: Could not extract tremor stats. Try again.")
        return {}
    print(f"  Tremor: {stats['tremor_freq_hz']:.1f} Hz  amp={stats['tremor_amp_px']:.3f}px")
    print(f"  OU theta x={stats['ou_theta_x']:.1f} y={stats['ou_theta_y']:.1f}  "
          f"sigma x={stats['ou_sigma_x']:.3f} y={stats['ou_sigma_y']:.3f}")
    print(f"  Micro-adj prob={stats['adjustment_prob']:.4f}  "
          f"Drift period={stats['drift_period_s']:.1f}s")
    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print()
    print("  Mouse Statistical Profiler")
    print("  Output ->", PROFILE_PATH)
    print()
    print("  Extracts your movement fingerprint (NOT your actual path).")
    print()

    approach_stats = record_approach()
    hold_stats     = record_hold()

    if not approach_stats or not hold_stats:
        print("\nERROR: Profile incomplete — please re-run and move/hold as instructed.")
        sys.exit(1)

    profile = {
        "approach": approach_stats,
        "hold":     hold_stats,
    }

    with open(PROFILE_PATH, "w") as f:
        json.dump(profile, f, indent=2)

    print(f"\nSaved profile: {PROFILE_PATH}")
    print("Run solve_captcha.py — it will synthesize movements from your fingerprint.")


if __name__ == "__main__":
    main()
