from __future__ import annotations

import sys

import numpy as np
from numpy.typing import NDArray

from chmap.config import ChannelMapEditorConfig
from chmap.probe import ProbeDesp
from chmap.probe_npx import NpxProbeDesp
from chmap.probe_npx.npx import ChannelMap
from chmap.util.bokeh_app import run_later
from chmap.views.data import Data1DView

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

__all__ = ['ElectrodeDensityDataView']


class ElectrodeDensityDataView(Data1DView):
    """Show electrode (selected) density curve beside the shank."""

    def __init__(self, config: ChannelMapEditorConfig):
        super().__init__(config, logger='chmap.view.density')

        self._data = None

    @property
    def name(self) -> str:
        return 'Electrode Density Curve'

    @property
    def description(self) -> str | None:
        return 'show electrode density curve along the shanks'

    def on_probe_update(self, probe: ProbeDesp, chmap, e):
        if chmap is None:
            self._data = None
        elif isinstance(probe, NpxProbeDesp):
            # self.logger.debug('on_probe_update()')

            try:
                self._data = self.arr_to_dict(electrode_density_npx(probe, chmap))
            except RuntimeError as ex:
                self.logger.warning(repr(ex), exc_info=ex)
                self._data = None

        run_later(self.update)

    def data(self):
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


def electrode_density_npx(probe: NpxProbeDesp, chmap: ChannelMap) -> NDArray[np.float_]:
    """

    :param probe:
    :param chmap:
    :return: density curve array. Array[float, S, Y, (x, y)].
    """
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
        ret.append(np.vstack([x, y]).T)

    return np.array(ret)
