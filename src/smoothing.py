"""
OneEuro filter for smooth, latency-aware landmark jitter reduction.

Reference: Géry Casiez, Nicolas Roussel, Daniel Vogel (2012).
  "1€ Filter: A Simple Speed-Based Low-Pass Filter for Noisy Input in
   Interactive Systems." CHI 2012.

Key idea: slow movements get more smoothing (low cutoff), fast movements
get less smoothing (high cutoff) to minimise latency.
"""
from __future__ import annotations

import math
from src.landmarks import LM, Point3D


# ---------------------------------------------------------------------------
# Primitive building blocks
# ---------------------------------------------------------------------------

def _alpha(cutoff: float, freq: float) -> float:
    """Compute the exponential smoothing coefficient for a given cutoff frequency."""
    te  = 1.0 / freq
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / te)


class _EWMAFilter:
    """Simple exponentially-weighted moving average (1-pole low-pass)."""

    __slots__ = ("_y",)

    def __init__(self) -> None:
        self._y: float | None = None

    def filter(self, x: float, alpha: float) -> float:
        if self._y is None:
            self._y = x
        else:
            self._y = alpha * x + (1.0 - alpha) * self._y
        return self._y

    def reset(self) -> None:
        self._y = None

    @property
    def value(self) -> float | None:
        return self._y


class OneEuroFilter:
    """
    1€ filter for a single scalar time-series.

    Parameters
    ----------
    min_cutoff : float
        Minimum smoothing cutoff frequency (Hz).  Lower = smoother when
        still, but more lag.  Default 1.0 Hz works well for body landmarks.
    beta : float
        Speed coefficient.  Higher = less smoothing during fast movement.
        Default 0.007; increase to ~0.1 for very fast motions (sports).
    d_cutoff : float
        Cutoff for the derivative (speed estimate).  Rarely needs tuning.
    """

    __slots__ = ("min_cutoff", "beta", "d_cutoff", "_x", "_dx", "_last_t")

    def __init__(self, min_cutoff: float = 1.0,
                 beta: float = 0.007,
                 d_cutoff: float = 1.0) -> None:
        self.min_cutoff = min_cutoff
        self.beta       = beta
        self.d_cutoff   = d_cutoff
        self._x   = _EWMAFilter()
        self._dx  = _EWMAFilter()
        self._last_t: float | None = None

    def __call__(self, x: float, t: float) -> float:
        """Filter a new sample `x` observed at time `t` (seconds)."""
        if self._last_t is None:
            freq = 30.0
        else:
            dt   = t - self._last_t
            freq = 1.0 / dt if dt > 1e-6 else 30.0
        self._last_t = t

        # Estimate the derivative (speed) with a low-pass filter
        prev = self._x.value if self._x.value is not None else x
        raw_dx = (x - prev) * freq
        edx = self._dx.filter(raw_dx, _alpha(self.d_cutoff, freq))

        # Adapt cutoff based on signal speed
        cutoff = self.min_cutoff + self.beta * abs(edx)
        return self._x.filter(x, _alpha(cutoff, freq))

    def reset(self) -> None:
        self._x.reset()
        self._dx.reset()
        self._last_t = None


# ---------------------------------------------------------------------------
# Convenience wrapper for a full set of landmarks
# ---------------------------------------------------------------------------

class LandmarkSmoother:
    """
    Applies a OneEuroFilter to every (x, y, z) dimension of every landmark.

    Usage::

        smoother = LandmarkSmoother()
        smooth_wld = smoother.smooth(world_landmarks_list, timestamp_seconds)
    """

    def __init__(self, min_cutoff: float = 1.0,
                 beta: float = 0.007) -> None:
        # Lazy-initialised: one filter per (landmark_index, axis)
        self._filters: dict[tuple[int, int], OneEuroFilter] = {}
        self._min_cutoff = min_cutoff
        self._beta       = beta

    def _get(self, lm_idx: int, axis: int) -> OneEuroFilter:
        key = (lm_idx, axis)
        if key not in self._filters:
            self._filters[key] = OneEuroFilter(self._min_cutoff, self._beta)
        return self._filters[key]

    def smooth(self, world_lm: list[Point3D], t: float) -> list[Point3D]:
        """
        Return a new list of Point3D with all coordinates smoothed.

        Parameters
        ----------
        world_lm : list[Point3D]
            Raw world landmarks from MediaPipe for the current frame.
        t : float
            Current timestamp in seconds.
        """
        out: list[Point3D] = []
        for idx, pt in enumerate(world_lm):
            sx = self._get(idx, 0)(pt.x, t)
            sy = self._get(idx, 1)(pt.y, t)
            sz = self._get(idx, 2)(pt.z, t)
            out.append(Point3D(sx, sy, sz))
        return out

    def reset(self) -> None:
        """Reset all internal filter states (call when switching exercises or persons)."""
        for f in self._filters.values():
            f.reset()
