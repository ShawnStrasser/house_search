#!/usr/bin/env python3
"""
Captcha solver for the Zillow "Press & Hold" challenge.

Usage:
    python solve_captcha.py            # solve (no recording)
    python solve_captcha.py --record   # solve and save screen recording to recordings/

For best results, first run:
    python record_mouse.py

The solver synthesizes entirely new movements that match your statistical
fingerprint — velocity profile shape, tremor frequency, OU correlation,
micro-adjustment rate, etc. Your actual recorded path is never stored or used.

Requires:
    pip install mss opencv-python pywin32 pillow numpy
"""

import sys
import time
import random
import math
import json
import threading
from datetime import datetime
from pathlib import Path

import mss
import cv2
import numpy as np
import win32api
import win32con

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CAPTCHA_DIR = Path(__file__).resolve().parent
BUTTON_TEMPLATE = str(CAPTCHA_DIR / "captcha_button.png")
FULL_TEMPLATE = str(CAPTCHA_DIR / "captcha.png")
PROFILE_PATH = str(CAPTCHA_DIR / "mouse_profile.json")

MATCH_THRESHOLD  = 0.70
GONE_THRESHOLD   = 0.55
MAX_HOLD_SECONDS = 20.0
POLL_INTERVAL    = 0.25   # seconds between captcha-gone checks

# Move mouse >this many pixels from the hold point to abort (emergency escape)
INTERRUPT_DISTANCE_PX = 80

RECORDINGS_DIR = str(CAPTCHA_DIR / "recordings")
RECORD_FPS       = 15   # frames per second — balance file size vs. smoothness
RECORD_VIDEO     = False  # set True or use --record flag to save screen recording

# How different the captcha region must look (vs. baseline) to count as "gone".
# Measured as mean absolute pixel difference across the header region (0-255).
# The fill animation changes the button only; the header stays constant until
# the whole overlay disappears, at which point the region changes dramatically.
REGION_CHANGE_THRESHOLD = 30   # pixels of mean absolute difference

# Button "solved" (dots visible) detection.
# Looks at the CENTER of the button only (x 35-65%, y 78-93% of overlay).
# When the dots appear, that region is fully blue with small lighter circles.
#
# From pixel analysis of real screenshots:
#   unsolved  (white button):   mean=237  std=43  -> mean too high
#   mid-fill  (half blue):      mean=172  std=64  -> mean too high
#   solved s1 (dots, bright):   mean=118  std=27  -> MATCH
#   solved s2 (dots, medium):   mean=117  std=20  -> MATCH
#   solved s3 (dots, faint):    mean=112  std=6   -> MATCH
#
# Detection rule: mean brightness 95-135 (fully blue zone) AND std > 4.0
# The mean check confirms the button is fully blue; std > 4 confirms there's
# some local contrast (the dots), even when they're very faint.
BUTTON_BLUE_FRACTION_MIN = 0.80  # stage 1: button must be >= this fraction filled
DOT_MEAN_MIN = 95                # stage 2: center strip mean brightness floor
DOT_MEAN_MAX = 135               # stage 2: center strip mean brightness ceiling
DOT_STD_MIN  = 4.0               # stage 2: minimum std deviation (dot contrast)

# Fallback profile when no recording exists — reasonable defaults
DEFAULT_APPROACH = {
    "speed_alpha":        2.5,
    "speed_beta":         2.0,
    "curvature_std_px":   18.0,
    "correction_rate":    0.04,
    "overshoot_fraction": 0.03,
    "total_duration_s":   0.75,
}
DEFAULT_HOLD = {
    "ou_theta_x":       9.0,
    "ou_theta_y":       9.0,
    "ou_sigma_x":       0.28,
    "ou_sigma_y":       0.22,
    "tremor_freq_hz":   10.5,
    "tremor_amp_px":    0.15,
    "adjustment_prob":  0.008,
    "adjustment_std_x": 0.9,
    "adjustment_std_y": 0.7,
    "drift_period_s":   28.0,
    "drift_amp_x":      0.35,
    "drift_amp_y":      0.28,
}


# ---------------------------------------------------------------------------
# Screen recorder
# ---------------------------------------------------------------------------

class ScreenRecorder:
    """
    Records the primary monitor in a background thread.
    Saves one MP4 per run to RECORDINGS_DIR/<timestamp>.mp4.

    Usage:
        rec = ScreenRecorder()
        rec.start()
        ...do work...
        rec.stop()          # blocks until file is written
        print(rec.path)
    """

    def __init__(self, fps: int = RECORD_FPS):
        self.fps     = fps
        self.frames  = []
        self._stop   = threading.Event()
        self._thread = threading.Thread(target=self._capture, daemon=True)

        ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir   = Path(RECORDINGS_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.path = str(out_dir / f"{ts}.mp4")

        # Grab monitor dimensions up front
        with mss.mss() as sct:
            mon        = sct.monitors[1]
            self.width  = mon["width"]
            self.height = mon["height"]
            self._mon   = dict(mon)

    def start(self):
        self._thread.start()
        print(f"  [rec] recording started -> {self.path}")

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=10)
        self._write()

    def _capture(self):
        interval = 1.0 / self.fps
        with mss.mss() as sct:
            while not self._stop.is_set():
                t0  = time.perf_counter()
                raw = sct.grab(self._mon)
                # Convert BGRA -> BGR (what VideoWriter expects)
                img = np.frombuffer(raw.bgra, dtype=np.uint8).reshape(
                    raw.height, raw.width, 4)[:, :, :3]
                self.frames.append(img)
                elapsed = time.perf_counter() - t0
                sleep_for = interval - elapsed
                if sleep_for > 0:
                    time.sleep(sleep_for)

    def _write(self):
        if not self.frames:
            print("  [rec] no frames captured — skipping save")
            return
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out    = cv2.VideoWriter(self.path, fourcc, self.fps,
                                 (self.width, self.height))
        for frame in self.frames:
            # Resize if mss returned a slightly different size on this system
            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                frame = cv2.resize(frame, (self.width, self.height))
            out.write(frame)
        out.release()
        duration = len(self.frames) / self.fps
        size_mb  = Path(self.path).stat().st_size / 1_048_576
        print(f"  [rec] saved {len(self.frames)} frames "
              f"({duration:.1f}s, {size_mb:.1f} MB) -> {self.path}")


# ---------------------------------------------------------------------------
# Load profile
# ---------------------------------------------------------------------------

def load_profile() -> tuple[dict, dict]:
    """Return (approach_params, hold_params). Falls back to defaults."""
    if Path(PROFILE_PATH).exists():
        try:
            with open(PROFILE_PATH) as f:
                p = json.load(f)
            ap = p.get("approach", {})
            hp = p.get("hold", {})
            if ap and hp:
                print(f"  Using recorded profile: {PROFILE_PATH}")
                return ap, hp
        except Exception as e:
            print(f"  Profile load error ({e}) — using defaults.")
    print("  No profile found — using default parameters.")
    print("  Run record_mouse.py for personalised movement.")
    return DEFAULT_APPROACH.copy(), DEFAULT_HOLD.copy()


# ---------------------------------------------------------------------------
# Screen capture + template matching
# ---------------------------------------------------------------------------

def grab_screen() -> np.ndarray:
    with mss.mss() as sct:
        mon = sct.monitors[1]
        raw = sct.grab(mon)
    img = np.frombuffer(raw.bgra, dtype=np.uint8).reshape(raw.height, raw.width, 4)
    return img[:, :, :3]


def find_template(screen_bgr: np.ndarray, template_path: str, threshold: float):
    template = cv2.imread(template_path)
    if template is None:
        raise FileNotFoundError(f"Template not found: {template_path}")
    sg = cv2.cvtColor(screen_bgr, cv2.COLOR_BGR2GRAY)
    tg = cv2.cvtColor(template,   cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(sg, tg, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    if max_val >= threshold:
        h, w = tg.shape
        return max_loc[0] + w // 2, max_loc[1] + h // 2, max_val
    return None


def locate_captcha_button(threshold: float = MATCH_THRESHOLD):
    """
    Returns (center_x, center_y, region_bbox) where region_bbox is
    (left, top, right, bottom) of the full captcha overlay in screen coords.
    Returns None if not found.
    """
    screen = grab_screen()
    r = find_template(screen, BUTTON_TEMPLATE, threshold)
    if r:
        print(f"  [match] button template  conf={r[2]:.3f}  pos=({r[0]},{r[1]})")
        cx, cy = r[0], r[1]
    else:
        r = find_template(screen, FULL_TEMPLATE, threshold * 0.9)
        if r:
            print(f"  [match] full template    conf={r[2]:.3f}  pos=({r[0]},{r[1]})")
            cx, cy = r[0], r[1]
        else:
            return None

    # Estimate the full overlay bounding box.
    # captcha.png is 387x261; the button is near the bottom (y≈215 in template).
    # We use the full template size to infer overlay bounds around the match.
    tmpl = cv2.imread(FULL_TEMPLATE)
    th, tw = tmpl.shape[:2]
    # Button is at ~55% down the full template → top of overlay is above button
    top  = cy - int(th * 0.85)
    left = cx - tw // 2
    bbox = (left, top, left + tw, top + th)
    return cx, cy, bbox


def snapshot_region(bbox: tuple[int, int, int, int]) -> np.ndarray:
    """
    Capture the captcha overlay region and return the TOP HALF only
    (the "Before we continue..." header — stays constant during fill animation,
    disappears completely when solved).
    """
    left, top, right, bottom = bbox
    mid_y = (top + bottom) // 2
    with mss.mss() as sct:
        region = {"left": left, "top": top, "width": right - left, "height": mid_y - top}
        raw = sct.grab(region)
    img = np.frombuffer(raw.bgra, dtype=np.uint8).reshape(raw.height, raw.width, 4)
    return img[:, :, :3]


def captcha_region_gone(baseline: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
    """
    Returns True when the captcha overlay has disappeared.

    Strategy: compare the header region (top half of the overlay) to the
    baseline snapshot taken before the hold started. During the fill animation,
    only the button changes — the header stays white. When the whole overlay
    goes away, the header region becomes completely different pixels.

    Uses mean absolute difference; a large difference means the overlay is gone.
    """
    current = snapshot_region(bbox)
    # Resize current to match baseline in case of minor capture size differences
    if current.shape != baseline.shape:
        current = cv2.resize(current, (baseline.shape[1], baseline.shape[0]))
    diff = np.mean(np.abs(current.astype(float) - baseline.astype(float)))
    return diff > REGION_CHANGE_THRESHOLD


def snapshot_dot_region(bbox: tuple[int, int, int, int]) -> np.ndarray:
    """
    Capture only the center of the button where the dots appear:
    x 35-65%, y 78-93% of the overlay bounding box.
    Tightly excludes the border, fill animation edges, and the header.
    """
    left, top, right, bottom = bbox
    h = bottom - top
    w = right  - left
    with mss.mss() as sct:
        region = {
            "left":   left + int(w * 0.35),
            "top":    top  + int(h * 0.78),
            "width":  int(w * 0.30),
            "height": int(h * 0.15),
        }
        raw = sct.grab(region)
    img = np.frombuffer(raw.bgra, dtype=np.uint8).reshape(raw.height, raw.width, 4)
    return img[:, :, :3]   # BGR


def button_shows_dots(bbox: tuple[int, int, int, int]) -> bool:
    """
    Returns True when the button is fully filled AND the dots are visible.

    Two-stage gate (must both be true):

    Stage 1 — full-button blue coverage check (whole button, inner 70% width):
      Grab the full button row and measure what fraction of pixels are
      dark-blue-dominant (B-R > 50 in BGR).
        unsolved  (white button):   ~0.06  -> FAIL stage 1
        mid-fill  (half blue):      ~0.47  -> FAIL stage 1
        solved    (fully blue):     ~0.98  -> PASS stage 1
      Threshold: >= 0.80

    Stage 2 — dot presence check (center strip x35-65%, y78-93%):
      Only reached once stage 1 passes. Checks mean brightness and std of
      the tight center crop where the dots live.
        fully blue, no dots yet:  mean ~112, std ~1-2  -> FAIL stage 2
        fully blue, dots present: mean ~112-118, std > 4 -> PASS stage 2

    The two-stage approach prevents the fill animation from triggering early:
    the center strip enters the 95-135 mean range while still filling, but
    it won't pass stage 1 (full coverage) until the fill is complete.
    """
    left, top, right, bottom = bbox
    h = bottom - top
    w = right - left

    # Stage 1: full button coverage — grab the inner 70% of the button row
    btn_top = top + int(h * 0.72)
    btn_bot = top + int(h * 0.97)
    pad_x   = int(w * 0.15)
    with mss.mss() as sct:
        raw = sct.grab({
            "left":   left + pad_x,
            "top":    btn_top,
            "width":  w - 2 * pad_x,
            "height": btn_bot - btn_top,
        })
    btn = np.frombuffer(raw.bgra, dtype=np.uint8).reshape(raw.height, raw.width, 4)[:, :, :3]
    B = btn[:, :, 0].astype(np.int32)
    R = btn[:, :, 2].astype(np.int32)
    blue_coverage = float(((B - R) > 50).mean())
    if blue_coverage < BUTTON_BLUE_FRACTION_MIN:
        return False

    # Stage 2: dot presence — tight center strip only
    region = snapshot_dot_region(bbox)
    gray = region.astype(np.float32).mean(axis=2)
    mean = float(gray.mean())
    std  = float(gray.std())
    return DOT_MEAN_MIN < mean < DOT_MEAN_MAX and std > DOT_STD_MIN


# ---------------------------------------------------------------------------
# Approach movement synthesis
# ---------------------------------------------------------------------------

def _screen_bounds() -> tuple[int, int, int, int]:
    with mss.mss() as sct:
        m = sct.monitors[1]
    return m["left"], m["top"], m["left"] + m["width"] - 1, m["top"] + m["height"] - 1


def _clamp(x: int, y: int) -> tuple[int, int]:
    l, t, r, b = _screen_bounds()
    return max(l, min(r, x)), max(t, min(b, y))


def _beta_speed_curve(n_steps: int, alpha: float, beta: float) -> np.ndarray:
    """
    Sample n_steps values from Beta(alpha, beta) to form the speed profile.
    Normalised so values sum to 1 (each value = fraction of total distance
    covered in that step).
    """
    # Sample from beta distribution
    raw = np.random.beta(alpha, beta, size=n_steps)
    # Smooth slightly so adjacent steps aren't wildly different
    kernel = np.array([0.15, 0.7, 0.15])
    smoothed = np.convolve(raw, kernel, mode='same')
    smoothed = np.clip(smoothed, 1e-6, None)
    return smoothed / smoothed.sum()


def synthesize_approach(x0: int, y0: int, x1: int, y1: int, ap: dict):
    """
    Synthesize a human-like approach path from statistical parameters.

    Stages:
      1. Generate a speed profile matching the Beta shape
      2. Add a curved deviation (matching curvature_std) via a random
         perpendicular offset that fades at start and end
      3. Add micro-corrections at the rate correction_rate
      4. Add a tiny overshoot at the end matching overshoot_fraction
    """
    dist = math.hypot(x1 - x0, y1 - y0)
    if dist < 2:
        win32api.SetCursorPos((x1, y1))
        return

    # Duration: scale by distance, bounded to a human range
    # Typical: ~700px in 0.6-1.0s; we use total_duration_s as reference
    ref_dur  = ap.get("total_duration_s", 0.75)
    duration = ref_dur * random.uniform(0.85, 1.15)
    # Clamp to [0.3s, 1.5s]
    duration = max(0.3, min(1.5, duration))

    hz       = 125
    n_steps  = max(15, int(duration * hz))
    alpha    = max(0.3, ap["speed_alpha"]  * random.uniform(0.85, 1.15))
    beta_p   = max(0.3, ap["speed_beta"]   * random.uniform(0.85, 1.15))

    speed_weights = _beta_speed_curve(n_steps, alpha, beta_p)

    # Direction vector from start to (slightly-overshoot) end
    overshoot = ap.get("overshoot_fraction", 0.03) * random.uniform(0.5, 1.5)
    end_x = x1 + int((x1 - x0) * overshoot)
    end_y = y1 + int((y1 - y0) * overshoot)

    # Path points: linear from start to end, distance proportional to speed
    cumulative = np.cumsum(speed_weights)  # 0..1
    path_x = x0 + (end_x - x0) * cumulative
    path_y = y0 + (end_y - y0) * cumulative

    # Add curved lateral deviation
    # The deviation is a sine arch perpendicular to the direction of travel
    curvature_std = ap.get("curvature_std_px", 18.0)
    if curvature_std > 0.5 and dist > 10:
        # Perpendicular unit vector
        dx_total = x1 - x0
        dy_total = y1 - y0
        perp_x = -dy_total / dist
        perp_y =  dx_total / dist
        # Sine arch — peaks at midpoint, zero at ends
        t       = np.linspace(0, 1, n_steps)
        arch    = np.sin(np.pi * t)
        # Random signed amplitude from your curvature distribution
        amp     = random.gauss(0, curvature_std) * random.choice([-1, 1])
        path_x += arch * amp * perp_x
        path_y += arch * amp * perp_y

    # Add micro-corrections (brief direction reversals)
    correction_rate = ap.get("correction_rate", 0.04)
    for i in range(1, n_steps - 1):
        if random.random() < correction_rate:
            # Small kick opposite to current direction of travel
            dx = path_x[i] - path_x[i-1]
            dy = path_y[i] - path_y[i-1]
            mag = math.hypot(dx, dy)
            if mag > 0.1:
                kick = random.uniform(0.5, 2.5)
                path_x[i] += -dx / mag * kick
                path_y[i] += -dy / mag * kick

    # Ensure we land exactly on target at the end (after overshoot correction)
    # Add a smooth return from overshoot in the last 12% of steps
    tail_start = int(n_steps * 0.88)
    for i in range(tail_start, n_steps):
        t_tail = (i - tail_start) / max(1, n_steps - tail_start)
        path_x[i] = path_x[i] + (x1 - path_x[i]) * t_tail
        path_y[i] = path_y[i] + (y1 - path_y[i]) * t_tail

    # Move
    step_time = duration / n_steps
    win32api.SetCursorPos((x0, y0))
    prev = (x0, y0)

    for i in range(n_steps):
        nx, ny = _clamp(int(round(path_x[i])), int(round(path_y[i])))
        if (nx, ny) != prev:
            win32api.SetCursorPos((nx, ny))
            prev = (nx, ny)
        # Per-step timing: slightly random around the nominal step time
        time.sleep(max(0.002, step_time * random.uniform(0.92, 1.08)))

    win32api.SetCursorPos((x1, y1))


# ---------------------------------------------------------------------------
# Hold tremor synthesis
# ---------------------------------------------------------------------------

class TremorSynthesizer:
    """
    Synthesizes hold tremor from statistical parameters.

    Combines:
      - Ornstein-Uhlenbeck correlated noise (per axis, with your theta/sigma)
      - Dominant physiological tremor frequency (sinusoidal, from your FFT peak)
      - Random micro-adjustments (at your observed rate and magnitude)
      - Slow fatigue drift (low-frequency sinusoid from your drift params)

    All parameters are slightly jittered each instantiation so no two
    hold sequences are identical even with the same profile.
    """

    def __init__(self, hp: dict, anchor_x: int, anchor_y: int):
        self.ax = float(anchor_x)
        self.ay = float(anchor_y)

        # OU state (per axis)
        self.ox = 0.0
        self.oy = 0.0

        # Slightly jitter all parameters so each hold is unique
        j = lambda v, pct=0.12: v * random.uniform(1 - pct, 1 + pct)

        self.theta_x = j(hp["ou_theta_x"])
        self.theta_y = j(hp["ou_theta_y"])
        self.sigma_x = j(hp["ou_sigma_x"])
        self.sigma_y = j(hp["ou_sigma_y"])

        self.tremor_freq = j(hp["tremor_freq_hz"], 0.08)
        self.tremor_amp  = j(hp["tremor_amp_px"],  0.20)
        self.tremor_phase_x = random.uniform(0, 2 * math.pi)
        self.tremor_phase_y = random.uniform(0, 2 * math.pi)

        self.adj_prob    = hp["adjustment_prob"] * random.uniform(0.7, 1.4)
        self.adj_std_x   = j(hp["adjustment_std_x"])
        self.adj_std_y   = j(hp["adjustment_std_y"])

        self.drift_period = j(hp["drift_period_s"], 0.25)
        self.drift_amp_x  = j(hp["drift_amp_x"], 0.20)
        self.drift_amp_y  = j(hp["drift_amp_y"], 0.20)
        self.drift_phase_x = random.uniform(0, 2 * math.pi)
        self.drift_phase_y = random.uniform(0, 2 * math.pi)

        # Nominal tick at 125 Hz with per-step jitter
        self.dt = 1.0 / 125.0

    def next(self, elapsed: float) -> tuple[int, int, float]:
        """Return (screen_x, screen_y, sleep_seconds) for this frame."""
        dt = self.dt

        # 1. Ornstein-Uhlenbeck step (correlated drift)
        self.ox += (-self.theta_x * self.ox * dt
                    + self.sigma_x * random.gauss(0, 1) * math.sqrt(dt))
        self.oy += (-self.theta_y * self.oy * dt
                    + self.sigma_y * random.gauss(0, 1) * math.sqrt(dt))

        # 2. Physiological tremor sinusoid
        w = 2 * math.pi * self.tremor_freq
        tx = self.tremor_amp * math.sin(w * elapsed + self.tremor_phase_x)
        ty = self.tremor_amp * math.cos(w * elapsed + self.tremor_phase_y)

        # 3. Slow fatigue drift
        wd = 2 * math.pi / self.drift_period
        fx = self.drift_amp_x * math.sin(wd * elapsed + self.drift_phase_x)
        fy = self.drift_amp_y * math.cos(wd * elapsed + self.drift_phase_y)

        # 4. Occasional micro-adjustment
        adj_x = adj_y = 0.0
        if random.random() < self.adj_prob:
            adj_x = random.gauss(0, self.adj_std_x)
            adj_y = random.gauss(0, self.adj_std_y)

        total_x = self.ax + self.ox + tx + fx + adj_x
        total_y = self.ay + self.oy + ty + fy + adj_y

        nx, ny = _clamp(int(round(total_x)), int(round(total_y)))
        sleep_s = dt * random.uniform(0.93, 1.07)
        return nx, ny, sleep_s


# ---------------------------------------------------------------------------
# Press-and-hold
# ---------------------------------------------------------------------------

def press_hold_until_done(cx: int, cy: int, hp: dict,
                          baseline: np.ndarray,
                          bbox: tuple[int, int, int, int],
                          max_seconds: float = MAX_HOLD_SECONDS) -> str:
    """
    Press and hold with tremor synthesis.

    Release conditions (checked every POLL_INTERVAL seconds):
      1. DOTS APPEAR  — button turns fully blue with gray dots = Kasada accepted
                        the hold. Release immediately, overlay still visible.
      2. OVERLAY GONE — fallback: whole overlay disappeared without dots phase
      3. USER INTERRUPT — mouse moved >INTERRUPT_DISTANCE_PX away
      4. TIMEOUT      — max_seconds exceeded

    Returns one of: "dots", "gone", "interrupted", "timeout"
    """
    tremor = TremorSynthesizer(hp, cx, cy)

    print(f"  [hold] pressing at ({cx},{cy}) — max {max_seconds:.1f}s")
    print(f"  [hold] release triggers: dots appear | overlay gone | mouse escape | timeout")
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)

    elapsed   = 0.0
    next_poll = POLL_INTERVAL
    result    = "timeout"

    try:
        while elapsed < max_seconds:
            t0 = time.perf_counter()
            nx, ny, sleep_s = tremor.next(elapsed)
            win32api.SetCursorPos((nx, ny))
            time.sleep(sleep_s)
            elapsed += time.perf_counter() - t0

            if elapsed >= next_poll:
                raw_x, raw_y = win32api.GetCursorPos()
                dist = math.hypot(raw_x - cx, raw_y - cy)
                if dist > INTERRUPT_DISTANCE_PX:
                    print(f"  [hold] user interrupt — mouse {dist:.0f}px away")
                    result = "interrupted"
                    break

                if button_shows_dots(bbox):
                    print(f"  [hold] dots detected at {elapsed:.2f}s — releasing click")
                    result = "dots"
                    break

                if captcha_region_gone(baseline, bbox):
                    print(f"  [hold] overlay gone at {elapsed:.2f}s — releasing click")
                    result = "gone"
                    break

                next_poll += POLL_INTERVAL

    finally:
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)

    if result == "timeout":
        print(f"  [hold] max hold time ({max_seconds:.1f}s) reached")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def solve(record: bool = RECORD_VIDEO):
    print("=" * 55)
    print("  Captcha Solver -- Press & Hold")
    print("=" * 55)

    ap, hp = load_profile()

    rec = ScreenRecorder() if record else None
    if rec:
        rec.start()

    try:
        _solve(ap, hp)
    finally:
        if rec:
            rec.stop()


def _solve(ap: dict, hp: dict):
    print("\n[1] Scanning for captcha button...")
    found = locate_captcha_button()
    if found is None:
        print("  NOT FOUND: Is the captcha visible on screen?")
        sys.exit(1)
    target_x, target_y, bbox = found
    print(f"  Found at ({target_x},{target_y})  overlay bbox={bbox}")

    # Snapshot the header region NOW as our "captcha present" baseline.
    # This is done before any movement so the button is still white/unaffected.
    baseline = snapshot_region(bbox)
    print(f"  Baseline snapshot: {baseline.shape[1]}x{baseline.shape[0]}px header region")

    delay = random.uniform(0.5, 1.6)
    print(f"\n[2] Reacting in {delay:.2f}s...")
    time.sleep(delay)

    start_x, start_y = win32api.GetCursorPos()
    print(f"\n[3] Moving ({start_x},{start_y}) -> ({target_x},{target_y})")
    synthesize_approach(start_x, start_y, target_x, target_y, ap)

    time.sleep(random.uniform(0.06, 0.22))

    print("\n[4] Holding...")
    hold_result = press_hold_until_done(target_x, target_y, hp,
                                        baseline=baseline, bbox=bbox,
                                        max_seconds=MAX_HOLD_SECONDS)

    print()

    if hold_result == "interrupted":
        print("ABORTED: User interrupted — mouse moved away.")
        sys.exit(3)

    if hold_result in ("dots", "gone"):
        solved = True
    else:
        # Timeout — do one final check
        time.sleep(0.5)
        solved = captcha_region_gone(baseline, bbox)

    if not solved:
        print("FAILED: Not solved. Try running record_mouse.py first.")
        # Still wander back before exiting so mouse isn't left on captcha
        synthesize_approach(target_x, target_y, start_x, start_y, ap)
        sys.exit(2)

    # Dots were detected (click released) but overlay may still be animating out.
    # Wait for it to fully disappear before wandering away — max 5s.
    if hold_result == "dots":
        print("[5] Waiting for overlay to clear...")
        t_wait = 0.0
        while t_wait < 5.0:
            if captcha_region_gone(baseline, bbox):
                print(f"  Overlay cleared after {t_wait:.1f}s")
                break
            time.sleep(0.15)
            t_wait += 0.15
        else:
            print("  Overlay did not clear in 5s — continuing anyway")

    # Natural wander back to where the mouse started
    cur_x, cur_y = win32api.GetCursorPos()
    print(f"\n[6] Wandering back to start ({start_x},{start_y})...")
    # Slightly longer, more relaxed movement (task done, no urgency)
    relaxed_ap = dict(ap)
    relaxed_ap["total_duration_s"] = ap.get("total_duration_s", 0.75) * random.uniform(1.1, 1.5)
    time.sleep(random.uniform(0.3, 0.9))   # brief pause before moving away
    synthesize_approach(cur_x, cur_y, start_x, start_y, relaxed_ap)

    print("\nSOLVED: Done.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Solve Zillow Press & Hold captcha")
    p.add_argument("--record", action="store_true", help="Save screen recording to recordings/")
    args = p.parse_args()
    solve(record=args.record)
