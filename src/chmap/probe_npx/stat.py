from __future__ import annotations

import sys
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from chmap.probe_npx import NpxProbeDesp, NpxElectrodeDesp
from chmap.probe_npx.npx import ChannelMap, Electrode
from chmap.probe_npx.select import ElectrodeSelector, load_select

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

__all__ = [
    'ElectrodeEfficiencyStat',
    'npx_electrode_density',
    'npx_channel_efficiency',
    'ElectrodeProbability',
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


def npx_electrode_density(chmap: ChannelMap) -> NDArray[np.float_]:
    """

    :param chmap:
    :return: density curve array. Array[float, S, (v, y), Y].
    """
    from scipy.ndimage import maximum_filter

    kind = chmap.probe_type
    electrodes = set([(it.shank, it.column, it.row) for it in chmap.electrodes])
    C = kind.n_col_shank
    R = kind.n_row_shank

    def find(s: int, c: int, r: int) -> ElectrodeDensity:
        if 0 <= r < R and 0 <= c < C:
            if (s, c, r) in electrodes:
                return ElectrodeDensity(1, 1)
            else:
                return ElectrodeDensity(1, 0)
        return ElectrodeDensity(0, 0)

    def density(ch: Electrode) -> float:
        s = ch.shank
        c = ch.column
        r = ch.row

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

    y = np.arange(0, kind.n_row_shank, dtype=float)

    ret = []

    for shank in range(kind.n_shank):
        x = np.zeros_like(y)
        for ch in chmap.electrodes[shank, :, :]:
            x[ch.row] = max(density(ch), x[ch.row])

        x = maximum_filter(x, size=3, mode='nearest')
        ret.append(np.vstack([x, y * kind.r_space]))

    return np.array(ret)


class ElectrodeEfficiencyStat(NamedTuple):
    total_channel: int
    """total channel number"""
    used_channel: int
    """number of used channel"""
    used_channel_on_shanks: list[int]
    """number of used channel on each shank"""

    request_electrodes: float
    """number of request electrodes for some categories"""

    selected_electrodes: int
    """number of selected electrodes in some categories"""

    area_efficiency: float
    """area efficiency"""

    channel_efficiency: float
    """channel efficiency"""

    remain_channel: int
    """number of channel in low-priority category"""

    remain_electrode: int
    """number of electrode set in low-priority category"""


def npx_channel_efficiency(chmap: ChannelMap, e: list[NpxElectrodeDesp]) -> ElectrodeEfficiencyStat:
    """
    Calculate the channel efficiency for a blueprint *e* and its outcomes *chmap*.

    :param chmap: channelmap outcomes from blueprint *e*
    :param e: blueprint
    :return:
    """
    used_channel = len(chmap)
    used_channel_on_shanks = [
        len([it for it in chmap.electrodes if it.shank == s])
        for s in range(chmap.probe_type.n_shank)
    ]

    selected = set([(it.shank, it.column, it.row) for it in chmap.electrodes])
    p, c = _npx_request_electrodes(selected, e)
    ae = 0 if p == 0 else c / p
    ce = 0 if ae == 0 else min(ae, 1 / ae)
    re, rc = _get_electrode(selected, e, [NpxProbeDesp.CATE_LOW, NpxProbeDesp.CATE_UNSET])

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


def _npx_request_electrodes(selected: set, e: list[NpxElectrodeDesp]) -> tuple[float, int]:
    p0, s0 = _get_electrode(selected, e, [NpxProbeDesp.CATE_SET, NpxProbeDesp.CATE_FULL])
    p2, s2 = _get_electrode(selected, e, [NpxProbeDesp.CATE_HALF])
    p4, s4 = _get_electrode(selected, e, [NpxProbeDesp.CATE_QUARTER])
    # pf, sf = _get_electrode(selected, e, [NpxProbeDesp.CATE_FORBIDDEN])
    p = p0 + p2 / 2 + p4 / 4
    s = s0 + s2 + s4  # - sf
    return p, s


def _get_electrode(selected: set, e: list[NpxElectrodeDesp], categories: list[int]) -> tuple[int, int]:
    e1 = [it for it in e if it.category in categories]
    e2 = [it for it in e1 if it.electrode in selected]
    return len(e1), len(e2)


class ElectrodeProbability(NamedTuple):
    sample_times: int
    """number of sample times"""
    summation: NDArray[np.int_]
    """summation matrix Array[count:int, S, C, R]"""
    complete: int
    """number of sample that get a complete result"""
    channel_efficiency_: NDArray[np.float_]
    """collected channel_efficiency array."""

    @property
    def probability(self) -> NDArray[np.float_]:
        """probability matrix Array[prob:float, S, C, R]"""
        return self.summation.astype(float) / self.sample_times

    @property
    def complete_rate(self) -> float:
        return self.complete / self.sample_times

    @property
    def channel_efficiency(self) -> float:
        """max channel efficiency"""
        return np.max(self.channel_efficiency_)

    @property
    def channel_efficiency_mean(self) -> float:
        """mean channel efficiency"""
        return np.mean(self.channel_efficiency_)

    @property
    def channel_efficiency_var(self) -> float:
        """channel efficiency variance"""
        return np.var(self.channel_efficiency_)

    def __add__(self, other: ElectrodeProbability) -> Self:
        return ElectrodeProbability(
            self.sample_times + other.sample_times,
            self.summation + other.summation,
            self.complete + other.complete,
            np.concatenate([self.channel_efficiency_, other.channel_efficiency_]),
        )

    @classmethod
    def _reduce_add(cls, result: list[ElectrodeProbability]) -> Self:
        return ElectrodeProbability(
            sum([it.sample_times for it in result]),
            np.sum([it.summation for it in result], axis=0),
            sum([it.complete for it in result]),
            np.concatenate([it.channel_efficiency_ for it in result]),
        )


def npx_electrode_probability(probe: NpxProbeDesp, chmap: ChannelMap, blueprint: list[NpxElectrodeDesp],
                              selector: str | ElectrodeSelector = 'default',
                              sample_times: int = 1000,
                              n_worker: int = 1) -> ElectrodeProbability:
    """
    Sample *sample_times* channelmap outcomes for a given *blueprint*.

    :param probe:
    :param chmap: channelmap instance, use as a reference.
    :param blueprint: a given blueprint.
    :param selector: use which electrode selecting method.
    :param sample_times:
    :param n_worker: number of process.
    :return: ElectrodeProbability
    """
    if isinstance(selector, str):
        selector = load_select(selector)

    if n_worker == 1:
        return _npx_electrode_probability_0(probe, chmap, blueprint, selector, sample_times)
    else:
        return _npx_electrode_probability_n(probe, chmap, blueprint, selector, sample_times, n_worker=n_worker)


def _npx_electrode_probability_0(probe: NpxProbeDesp, chmap: ChannelMap, blueprint: list[NpxElectrodeDesp],
                                 selector: ElectrodeSelector,
                                 sample_times: int) -> ElectrodeProbability:
    pt = chmap.probe_type
    mat = np.zeros((pt.n_shank, pt.n_col_shank, pt.n_row_shank))
    complete = 0
    channel_efficiency = []

    for _ in range(sample_times):
        chmap = selector(probe, chmap, blueprint)

        for t in chmap.electrodes:
            mat[t.shank, t.column, t.row] += 1

        if probe.is_valid(chmap):
            complete += 1

        channel_efficiency.append(npx_channel_efficiency(chmap, blueprint).channel_efficiency)

    return ElectrodeProbability(sample_times, mat, complete, np.array(channel_efficiency))


def _npx_electrode_probability_n(probe: NpxProbeDesp, chmap: ChannelMap, blueprint: list[NpxElectrodeDesp],
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
            jobs.append(pool.apply_async(_npx_electrode_probability_0, (probe, chmap, blueprint, selector, _sample_times)))
        pool.close()
        pool.join()

    return ElectrodeProbability._reduce_add([it.get() for it in jobs])
