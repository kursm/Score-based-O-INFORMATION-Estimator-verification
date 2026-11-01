"""Exact continuous piecewise-linear densities; all calculations use float64."""
from dataclasses import dataclass
from math import comb
import numpy as np


@dataclass
class LinearDensity:
    heights: np.ndarray

    def __post_init__(self):
        self.heights = np.asarray(self.heights, dtype=np.float64).copy()
        h = self.heights
        if h.ndim != 1 or len(h) < 3 or not np.isfinite(h).all():
            raise ValueError("Expected k+2 finite heights, k>=1")
        if np.any(h < 0) or h[0] != 0 or h[-1] != 0:
            raise ValueError("Heights must be nonnegative with zero endpoints")
        self.areas = (h[:-1] + h[1:]) / 2
        self.z = float(self.areas.sum())
        if self.z <= 0:
            raise ValueError("Density has zero mass")
        self.k = len(h) - 2
        self.knots = np.arange(len(h), dtype=np.float64)
        self.mean = self.moment(1)
        self.second_moment = self.moment(2)
        self.variance = self.second_moment - self.mean**2
        if self.variance <= 0:
            raise ValueError("Density has nonpositive variance")
        self.std = np.sqrt(self.variance)

    @classmethod
    def random(cls, k, rng):
        if not isinstance(k, (int, np.integer)) or isinstance(k, bool) or k < 1:
            raise ValueError("k must be an integer >=1")
        return cls(np.r_[0., rng.uniform(size=k), 0.])

    def moment(self, p):
        j = np.arange(len(self.heights)-1, dtype=np.float64)
        a, slope = self.heights[:-1], np.diff(self.heights)
        return float(sum(comb(p, q) * np.sum(j**(p-q) *
                         (a/(q+1) + slope/(q+2))) for q in range(p+1)) / self.z)

    def pdf(self, x, standardized=False):
        x = np.asarray(x, dtype=np.float64)
        raw = x*self.std+self.mean if standardized else x
        safe = np.clip(raw, 0, self.k+1)
        j = np.minimum(np.floor(safe).astype(np.int64), self.k)
        u = safe-j
        out = (self.heights[j] + (self.heights[j+1]-self.heights[j])*u)/self.z
        out = np.where((raw >= 0) & (raw <= self.k+1), out, 0.)
        return out*self.std if standardized else out

    def cdf(self, x, standardized=False):
        x = np.asarray(x, dtype=np.float64)
        raw = x*self.std+self.mean if standardized else x
        safe = np.clip(raw, 0, self.k+1)
        j = np.minimum(np.floor(safe).astype(np.int64), self.k)
        u = safe-j
        cum = np.r_[0., np.cumsum(self.areas)]
        out = (cum[j] + self.heights[j]*u +
               .5*(self.heights[j+1]-self.heights[j])*u*u)/self.z
        return np.where(raw <= 0, 0., np.where(raw >= self.k+1, 1., out))

    def sample(self, size, rng, standardized=False):
        # Restrict selection to positive mass segments, even when heights contain zeros.
        positive = np.flatnonzero(self.areas > 0)
        cumulative = np.cumsum(self.areas[positive]/self.z)
        cumulative[-1] = 1.
        j = positive[np.searchsorted(cumulative, rng.random(size), side="right")]
        a, b = self.heights[j], self.heights[j+1]
        choose_left = rng.random(size) < a/(a+b)
        r = rng.random(size)
        # r/(1+sqrt(1-r)) is the stable form of 1-sqrt(1-r).
        u = np.where(choose_left, r/(1+np.sqrt(1-r)), np.sqrt(r))
        raw = j+u
        return (raw-self.mean)/self.std if standardized else raw

    def metadata(self):
        return dict(k=self.k, knots=self.knots.tolist(), heights=self.heights.tolist(),
                    normalizer=self.z, mean=self.mean, second_moment=self.second_moment,
                    variance=self.variance, std=float(self.std),
                    local_maxima=int(np.sum((self.heights[1:-1] > self.heights[:-2]) &
                                            (self.heights[1:-1] > self.heights[2:]))),
                    standardized_knot_spacing=float(1/self.std))
