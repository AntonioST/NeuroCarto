from __future__ import annotations

import sys

import numpy as np

from chmap.probe import ProbeDesp, M, E
from chmap.probe_npx import NpxProbeDesp
from chmap.probe_npx.npx import ChannelMap
from chmap.views.data import DataReader

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

__all__ = ['ElectrodeDensityData']


class ElectrodeDensityData(DataReader):
    def __init__(self):
        self.probe: ProbeDesp[M, E] | None = None
        self.chmap: M | None = None
        self._data: list[tuple[list[float], list[float]]] | None = None

    @property
    def name(self) -> str:
        return 'Electrode Density Curve'

    @classmethod
    def match_file(cls, filename: str) -> bool:
        return filename == '!density'

    def on_probe_update(self, probe: ProbeDesp[M, E], chmap: M | None, e: list[E] | None):
        self.probe = probe
        self.chmap = chmap

        if isinstance(probe, NpxProbeDesp):
            self._data = electrode_density_npx(probe, chmap)

    def data(self) -> list[tuple[list[float], list[float]]] | None:
        if self.probe is None:
            return None

        return self._data


class ElectrodeDensity:
    def __init__(self, electrode: int, channel: int):
        self.electrode = electrode
        self.channel = channel

    def __add__(self, other: ElectrodeDensity) -> ElectrodeDensity:
        return ElectrodeDensity(self.electrode + other.electrode, self.channel + other.channel)

    def __iadd__(self, other: ElectrodeDensity) -> Self:
        self.electrode += other.electrode
        self.channel += other.channel
        return self

    def __float__(self) -> float:
        return self.channel / self.electrode

    def __str__(self):
        return f'{self.channel}/{self.electrode}'


def electrode_density_npx(probe: NpxProbeDesp, chmap: ChannelMap) -> list[tuple[list[float], list[float]]]:
    from scipy.stats import norm

    kind = chmap.probe_type

    channels = probe.all_channels(chmap)
    electrodes = set([it.electrode for it in channels])
    C = kind.n_col_shank
    R = kind.n_row_shank

    def find(s: int, c: int, r: int) -> ElectrodeDensity:
        if 0 <= r < R and 0 <= c < C:
            if (s, c, r) in electrodes:
                return ElectrodeDensity(1, 1)
            else:
                return ElectrodeDensity(1, 0)
        return ElectrodeDensity(0, 0)

    def density(s: int, c: int, r: int) -> float:
        d = ElectrodeDensity(1, 1)
        d += find(s, c - 1, r - 1)
        d += find(s, c, r - 1)
        d += find(s, c + 1, r - 1)
        d += find(s, c - 1, r)
        d += find(s, c + 1, r)
        d += find(s, c - 1, r + 1)
        d += find(s, c, r + 1)
        d += find(s, c + 1, r + 1)
        return float(d)

    y = np.arange(0, kind.n_row_shank * kind.r_space, dtype=float)
    f = kind.s_space

    ret = []

    for shank in range(kind.n_shank):
        x = np.zeros_like(y)
        for ch in channels:
            if ch.electrode[0] == shank:
                x += norm.pdf(y, ch.y, 30) * density(*ch.electrode) * f * 4

        x += (kind.n_col_shank - 1) * kind.c_space + shank * kind.s_space
        ret.append((list(x), list(y)))

    return ret
