from __future__ import annotations

import sys
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from chmap.probe_npx import NpxProbeDesp, NpxElectrodeDesp
from chmap.probe_npx.npx import ChannelMap
from chmap.probe_npx.select import ElectrodeSelector, load_select

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

__all__ = [
    'ElectrodeEfficientStat',
    'npx_electrode_density',
    'npx_channel_efficient',
    'npx_electrode_probability'
]


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


def npx_electrode_density(probe: NpxProbeDesp, chmap: ChannelMap) -> NDArray[np.float_]:
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


class ElectrodeEfficientStat(NamedTuple):
    total_channel: int
    used_channel: int
    used_channel_on_shanks: list[int]

    request_electrodes: float
    channel_efficiency: float
    remain_channel: int  # number of electrode selected in remainder policy
    remain_electrode: int  # number of electrode set in remainder policy


def npx_channel_efficient(chmap: ChannelMap, e: list[NpxElectrodeDesp]) -> ElectrodeEfficientStat:
    used_channel = len(chmap)
    used_channel_on_shanks = [
        len([it for it in chmap.electrodes if it.shank == s])
        for s in range(chmap.probe_type.n_shank)
    ]

    p, c = _npx_request_electrodes(e)
    cp = 0 if p == 0 else c / p
    re, rc = _get_electrode(e, [NpxProbeDesp.POLICY_REMAINDER, NpxProbeDesp.POLICY_UNSET])

    return ElectrodeEfficientStat(
        chmap.probe_type.n_channels,
        used_channel,
        used_channel_on_shanks,
        request_electrodes=p,
        channel_efficiency=cp,
        remain_electrode=re,
        remain_channel=rc
    )


def _npx_request_electrodes(e: list[NpxElectrodeDesp]) -> tuple[float, int]:
    p0, s0 = _get_electrode(e, [NpxProbeDesp.POLICY_SET, NpxProbeDesp.POLICY_D1])
    p2, s2 = _get_electrode(e, [NpxProbeDesp.POLICY_D2])
    p4, s4 = _get_electrode(e, [NpxProbeDesp.POLICY_D4])
    p = p0 + p2 / 2 + p4 / 4
    s = s0 + s2 + s4
    return p, s


def _get_electrode(e: list[NpxElectrodeDesp], policies: list[int]) -> tuple[int, int]:
    e1 = [it for it in e if it.policy in policies]
    e2 = [it for it in e1 if it.state == NpxProbeDesp.STATE_USED]
    return len(e1), len(e2)


def npx_electrode_probability(probe: NpxProbeDesp, chmap: ChannelMap, e: list[NpxElectrodeDesp],
                              selector: str | ElectrodeSelector = 'default',
                              sample_times: int = 1000) -> NDArray[np.float_]:
    """

    :param probe:
    :param chmap:
    :param e:
    :param selector:
    :param sample_times:
    :return: probability matrix Array[prob:float, S, C, R]
    """
    if isinstance(selector, str):
        selector = load_select(selector)

    pt = chmap.probe_type
    mat = np.zeros((pt.n_shank, pt.n_col_shank, pt.n_row_shank))

    for _ in range(sample_times):
        chmap = selector(probe, chmap, e)
        for t in chmap.electrodes:
            mat[t.shank, t.column, t.row] += 1

    return mat / sample_times
