from __future__ import annotations

import functools
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
    'ElectrodeEfficiencyStat',
    'npx_electrode_density',
    'npx_channel_efficiency',
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


class ElectrodeEfficiencyStat(NamedTuple):
    total_channel: int
    used_channel: int
    used_channel_on_shanks: list[int]

    request_electrodes: float
    selected_electrodes: int
    area_efficiency: float
    channel_efficiency: float
    remain_channel: int  # number of electrode selected in remainder policy
    remain_electrode: int  # number of electrode set in remainder policy


def npx_channel_efficiency(chmap: ChannelMap, e: list[NpxElectrodeDesp]) -> ElectrodeEfficiencyStat:
    used_channel = len(chmap)
    used_channel_on_shanks = [
        len([it for it in chmap.electrodes if it.shank == s])
        for s in range(chmap.probe_type.n_shank)
    ]

    p, c = _npx_request_electrodes(e)
    ae = 0 if p == 0 else c / p
    ce = 0 if ae == 0 else min(ae, 1 / ae)
    re, rc = _get_electrode(e, [NpxProbeDesp.POLICY_LOW, NpxProbeDesp.POLICY_UNSET])

    return ElectrodeEfficiencyStat(
        chmap.probe_type.n_channels,
        used_channel,
        used_channel_on_shanks,
        request_electrodes=p,
        selected_electrodes=c,
        area_efficiency=ae,
        channel_efficiency=ce,
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


class ElectrodeProbability(NamedTuple):
    sample_times: int
    summation: NDArray[np.int_]  # summation matrix Array[count:int, S, C, R]
    complete: int
    channel_efficiency: float

    @property
    def probability(self) -> NDArray[np.float_]:
        """probability matrix Array[prob:float, S, C, R]"""
        return self.summation.astype(float) / self.sample_times

    @property
    def complete_rate(self) -> float:
        return self.complete / self.sample_times

    def __add__(self, other: ElectrodeProbability) -> ElectrodeProbability:
        return ElectrodeProbability(
            self.sample_times + other.sample_times,
            self.summation + other.summation,
            self.complete + other.complete,
            max(self.channel_efficiency, other.channel_efficiency)
        )


def npx_electrode_probability(probe: NpxProbeDesp, chmap: ChannelMap, e: list[NpxElectrodeDesp],
                              selector: str | ElectrodeSelector = 'default',
                              sample_times: int = 1000,
                              n_worker: int = 1) -> ElectrodeProbability:
    """

    :param probe:
    :param chmap:
    :param e:
    :param selector:
    :param sample_times:
    :param n_worker:
    :return: ElectrodeProbability
    """
    if isinstance(selector, str):
        selector = load_select(selector)

    if n_worker == 1:
        return _npx_electrode_probability_0(probe, chmap, e, selector, sample_times)
    else:
        return _npx_electrode_probability_n(probe, chmap, e, selector, sample_times, n_worker=n_worker)


def _npx_electrode_probability_0(probe: NpxProbeDesp, chmap: ChannelMap, e: list[NpxElectrodeDesp],
                                 selector: ElectrodeSelector,
                                 sample_times: int) -> ElectrodeProbability:
    pt = chmap.probe_type
    mat = np.zeros((pt.n_shank, pt.n_col_shank, pt.n_row_shank))
    complete = 0
    channel_efficiency = 0.0

    for _ in range(sample_times):
        chmap = selector(probe, chmap, e)

        for t in chmap.electrodes:
            mat[t.shank, t.column, t.row] += 1

        if probe.is_valid(chmap):
            complete += 1

        channel_efficiency = max(channel_efficiency, npx_channel_efficiency(chmap, e).channel_efficiency)

    return ElectrodeProbability(sample_times, mat, complete, channel_efficiency)


def _npx_electrode_probability_n(probe: NpxProbeDesp, chmap: ChannelMap, e: list[NpxElectrodeDesp],
                                 selector: ElectrodeSelector,
                                 sample_times: int,
                                 n_worker: int) -> ElectrodeProbability:
    if n_worker <= 1:
        raise ValueError()

    sample_times_per_worker = sample_times // n_worker
    sample_times_list = [sample_times_per_worker] * n_worker
    sample_times_list[-1] += sample_times - sum(sample_times_list)
    assert sum(sample_times_list) == sample_times

    import multiprocessing
    with multiprocessing.Pool(n_worker) as pool:
        jobs = []
        for _sample_times in sample_times_list:
            jobs.append(pool.apply_async(_npx_electrode_probability_0, (probe, chmap, e, selector, _sample_times)))
        pool.close()
        pool.join()

    return functools.reduce(ElectrodeProbability.__add__, [it.get() for it in jobs])
