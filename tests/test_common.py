#!/usr/bin/env python3
"""
Tests for scripts/common.py — the pure parts (no ROS, no bags, no Open3D).

Run:  cd tests && python3 -m pytest test_common.py -v
      (or plain `python3 test_common.py` for a quick pass/fail)

These started life as the ad-hoc equivalence checks that validated the 2026-07-16
refactor (extracting common.py without changing any product). They are kept because
two of the things they pin down are INVISIBLE in normal operation:

  - read_nav's sort/dedup is dead code on the Andratx bags, which are already clean
    (20 Hz, dt = 0.05 s exactly). If it broke, no product would change and nobody
    would notice — until a bag with reordered nav silently produced wrong poses.
  - enhance_data's two nodata semantics differ in ~24% of pixels. Collapsing them
    into one (the "obvious" cleanup) would quietly change the waterfall.
"""

import os
import sys

import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))

from common import read_nav, nav_interpolators, enhance_data, slant_to_ground, is_port_topic


# ── fakes: a nav message and a bag, so the tests need neither ROS nor a bagfile ──

class _FakeNavMsg:
    def __init__(self, t, north, east=0.0, depth=0.0, roll=0.0, pitch=0.0, yaw=0.0, alt=5.0):
        self.header = type('h', (), {'stamp': type('s', (), {'to_sec': lambda _s, t=t: t})()})()
        self.position = type('p', (), {'north': north, 'east': east, 'depth': depth})()
        self.orientation = type('o', (), {'roll': roll, 'pitch': pitch, 'yaw': yaw})()
        self.altitude = alt


class _FakeBag:
    def __init__(self, msgs):
        self.msgs = msgs

    def read_messages(self, topics=None):
        return [(None, m, None) for m in self.msgs]


# ── read_nav ────────────────────────────────────────────────────────────────────

def test_read_nav_sorts_and_deduplicates():
    """Out-of-order + duplicated timestamps -> strictly increasing, values still paired.

    The pairing is the subtle half: sorting ts but not the value arrays would leave
    every pose attached to the wrong timestamp.
    """
    raw = [(3.0, 30.0), (1.0, 10.0), (2.0, 20.0), (2.0, 99.0), (5.0, 50.0), (4.0, 40.0)]
    nav = read_nav(_FakeBag([_FakeNavMsg(t, v) for t, v in raw]), '/nav')

    assert list(nav['ts']) == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert list(nav['north']) == [10.0, 20.0, 30.0, 40.0, 50.0]
    assert np.all(np.diff(nav['ts']) > 0), "interp1d needs a strictly increasing x"


def test_interp1d_alone_tolerates_unsorted_input():
    """Pins down what is NOT the reason for the sort, because the obvious answer is wrong.

    interp1d defaults to assume_sorted=False and sorts x internally, so unordered
    timestamps alone do NOT corrupt the interpolation. If a future reader deletes the
    sort reasoning as cargo cult, the three tests below are the ones that matter.
    """
    from scipy.interpolate import interp1d

    raw = [(3.0, 30.0), (1.0, 10.0), (2.0, 20.0), (5.0, 50.0), (4.0, 40.0)]
    unguarded = interp1d(np.array([t for t, _ in raw]), np.array([v for _, v in raw]),
                         bounds_error=False, fill_value=np.nan)
    assert abs(float(unguarded(2.5)) - 25.0) < 1e-9


def test_duplicate_timestamps_would_break_interp1d():
    """Reason 1 for the dedup: two y for one x -> silent garbage (measured: 64.5 vs 25.0)."""
    from scipy.interpolate import interp1d

    raw = [(3.0, 30.0), (1.0, 10.0), (2.0, 20.0), (2.0, 99.0), (5.0, 50.0), (4.0, 40.0)]
    unguarded = interp1d(np.array([t for t, _ in raw]), np.array([v for _, v in raw]),
                         bounds_error=False, fill_value=np.nan)
    assert abs(float(unguarded(2.5)) - 25.0) > 1.0

    nav = read_nav(_FakeBag([_FakeNavMsg(t, v) for t, v in raw]), '/nav')
    assert abs(float(nav_interpolators(nav)['n'](2.5)) - 25.0) < 1e-9


def test_unsorted_yaw_would_break_unwrap():
    """Reason 2 for the sort: np.unwrap is order-dependent and invents heading jumps."""
    yaw = np.radians(np.array([0.0, 350.0, 10.0, 340.0, 20.0]))
    ts = np.array([3.0, 1.0, 2.0, 5.0, 4.0])

    assert not np.allclose(np.unwrap(yaw), np.unwrap(yaw[np.argsort(ts)]))

    nav = read_nav(_FakeBag([_FakeNavMsg(t, 0.0, yaw=y) for t, y in zip(ts, yaw)]), '/nav')
    assert np.array_equal(np.unwrap(nav['yaw']), np.unwrap(yaw[np.argsort(ts)]))


def test_unsorted_timestamps_would_break_yaw_rate():
    """Reason 3 for the sort: np.gradient(ts) goes negative -> the attitude gate misfires."""
    ts = np.array([3.0, 1.0, 2.0, 5.0, 4.0])
    assert (np.gradient(ts) < 0).any(), "unsorted ts yields negative dt"

    nav = read_nav(_FakeBag([_FakeNavMsg(t, 0.0) for t in ts]), '/nav')
    assert np.all(np.gradient(nav['ts']) > 0)


def test_read_nav_empty_raises():
    try:
        read_nav(_FakeBag([]), '/nav')
    except RuntimeError:
        return
    assert False, "an empty nav topic must raise, not yield empty interpolators"


# ── nav_interpolators ───────────────────────────────────────────────────────────

def test_yaw_smoothing_is_opt_in():
    """The multibeam must NOT get the sidescan's yaw smoothing (it would move the cloud)."""
    ts = np.arange(0.0, 10.0, 0.05)
    yaw = np.where((np.arange(len(ts)) % 20) == 0, 0.4, 0.0)     # spiky heading
    msgs = [_FakeNavMsg(t, 0.0, yaw=y) for t, y in zip(ts, yaw)]
    nav = read_nav(_FakeBag(msgs), '/nav')

    raw = nav_interpolators(nav)['y'](ts)
    smooth = nav_interpolators(nav, smooth_yaw_sigma=2)['y'](ts)

    assert np.array_equal(raw, np.unwrap(yaw)), "default must leave yaw untouched"
    assert not np.allclose(raw, smooth), "sigma=2 must actually smooth"


def test_yaw_rate_is_opt_in_and_in_degrees():
    ts = np.arange(0.0, 5.0, 0.05)
    yaw = np.radians(10.0) * ts                                  # 10 deg/s, constant
    nav = read_nav(_FakeBag([_FakeNavMsg(t, 0.0, yaw=y) for t, y in zip(ts, yaw)]), '/nav')

    assert 'yr' not in nav_interpolators(nav)
    yr = nav_interpolators(nav, with_yaw_rate=True)['yr'](2.5)
    assert abs(float(yr) - 10.0) < 1e-6


def test_out_of_range_is_nan_not_extrapolation():
    """Callers gate on NaN; silent extrapolation would place pings with invented poses."""
    nav = read_nav(_FakeBag([_FakeNavMsg(t, t) for t in (1.0, 2.0, 3.0)]), '/nav')
    f = nav_interpolators(nav)
    assert np.isnan(float(f['n'](0.0))) and np.isnan(float(f['n'](9.0)))


# ── enhance_data ────────────────────────────────────────────────────────────────

def _reference_mosaic_enhance(img_input):
    """The pre-refactor mosaic version, verbatim. The unified one must still match it."""
    valid = img_input > 0
    vmin, vmax = np.percentile(img_input[valid], (2, 98))
    vmax = max(vmax, vmin + 1e-6)
    img = np.clip((img_input - vmin) * 255.0 / (vmax - vmin), 0, 255).astype(np.uint8)
    img = cv2.medianBlur(img, 5)
    img = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(img)
    return cv2.filter2D(img, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))


def _reference_waterfall_enhance(img_gray):
    """The pre-refactor waterfall version, verbatim (no nodata concept)."""
    img = img_gray.copy()
    p2, p98 = np.percentile(img, (2, 98))
    denom = p98 - p2 if (p98 - p2) > 0 else 1
    img = np.clip((img - p2) * 255.0 / denom, 0, 255).astype(np.uint8)
    img = cv2.medianBlur(img, 5)
    img = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(img)
    return cv2.filter2D(img, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))


def test_enhance_data_masked_matches_the_mosaic_reference():
    rng = np.random.default_rng(0)
    img = rng.gamma(2.0, 30.0, (200, 300)).astype(np.float32)
    img[rng.random(img.shape) < 0.4] = 0.0                       # empty cells
    assert np.array_equal(enhance_data(img, nodata_mask=~(img > 0)), _reference_mosaic_enhance(img))


def test_enhance_data_unmasked_matches_the_waterfall_reference():
    rng = np.random.default_rng(1)
    img = rng.gamma(1.0, 3.0, (400, 900))
    img[rng.random(img.shape) < 0.475] = 0                       # raw SSS echo: ~47% zeros
    img = np.clip(img, 0, 255).astype(np.uint8)
    assert np.array_equal(enhance_data(img), _reference_waterfall_enhance(img))


def test_the_two_nodata_semantics_are_not_interchangeable():
    """Why the parameter exists. Merging the two would silently change the waterfall."""
    rng = np.random.default_rng(2)
    img = rng.gamma(1.0, 3.0, (400, 900))
    img[rng.random(img.shape) < 0.475] = 0
    img = np.clip(img, 0, 255).astype(np.uint8)

    differing = (enhance_data(img) != enhance_data(img, nodata_mask=~(img > 0))).mean()
    assert differing > 0.1, f"expected a large difference, got {differing:.1%}"


def test_enhance_data_all_nodata_returns_zeros():
    img = np.zeros((20, 20), dtype=np.float32)
    assert not enhance_data(img, nodata_mask=np.ones_like(img, dtype=bool)).any()


# ── sidescan geometry ───────────────────────────────────────────────────────────

def test_slant_to_ground_assumes_sample_zero_is_nadir():
    """Sample 0 -> ground 0 / theta 0. This is the contract the port reversal serves."""
    ground, theta = slant_to_ground(1000, sonar_range=50.0, altitude=5.0)

    assert ground[0] == 0.0 and theta[0] == 0.0
    assert np.all(np.diff(ground) >= 0), "ground range must grow away from nadir"
    assert theta[-1] > 80.0, "the far end must be a grazing incidence"


def test_slant_to_ground_blind_zone_is_the_leading_samples():
    """Everything closer than the altitude is water column: ground stays 0 at the start."""
    ground, _ = slant_to_ground(2000, sonar_range=50.0, altitude=10.0)
    # slant < altitude -> clamped to 0. 10 m of 50 over 2000 samples = the first 400.
    assert ground[399] == 0.0 and ground[420] > 0.0


def test_slant_to_ground_matches_pythagoras():
    ground, _ = slant_to_ground(2000, sonar_range=50.0, altitude=5.0)
    i = 1000
    slant = i * (50.0 / 2000)
    assert abs(ground[i] - np.sqrt(slant ** 2 - 5.0 ** 2)) < 1e-9


def test_is_port_topic():
    """Port is the mirrored channel (raw order far->nadir); everything else is starboard."""
    assert is_port_topic('/sparus2/marinesonic_scoutmkii_sidescan/raw_data/port')
    assert not is_port_topic('/sparus2/marinesonic_scoutmkii_sidescan/raw_data/starboard')


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception as e:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
