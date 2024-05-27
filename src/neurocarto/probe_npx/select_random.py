"""
Neuropixels another electrode selection method.
It uses random selection and ignore all density rule for control reference purpose.
"""
from __future__ import annotations

import random

from .desp import NpxProbeDesp, NpxElectrodeDesp, K
from .npx import ChannelMap

__all__ = ['electrode_select']


def electrode_select(desp: NpxProbeDesp, chmap: ChannelMap, blueprint: list[NpxElectrodeDesp],
                     ignore_preselected=False, ignore_exclude=False, **kwargs) -> ChannelMap:
    ret = desp.new_channelmap(chmap)
    cand: dict[K, NpxElectrodeDesp] = {it.electrode: it for it in desp.all_electrodes(ret)}
    for e in blueprint:
        cand[e.electrode].category = e.category

    # add pre-selected
    if not ignore_preselected:
        for e in blueprint:
            if e.category == NpxProbeDesp.CATE_SET:
                _add(desp, ret, cand, e)

    # remove excluded electrodes from the candidate set
    if not ignore_exclude:
        for e in blueprint:
            if e.category == NpxProbeDesp.CATE_EXCLUDED:
                try:
                    del cand[e.electrode]
                except KeyError:
                    pass

    return select_loop(desp, ret, cand, **kwargs)


def select_loop(desp: NpxProbeDesp, chmap: ChannelMap, cand: dict[K, NpxElectrodeDesp],
                ignore_exclude=False, **kwargs) -> ChannelMap:
    while len(cand):
        e = pick_electrode(cand)
        if e.category == NpxProbeDesp.CATE_EXCLUDED and ignore_exclude:
            pass
        else:
            _add(desp, chmap, cand, e)

    return chmap


def pick_electrode(cand: dict[K, NpxElectrodeDesp]) -> NpxElectrodeDesp | None:
    return random.choice(list(cand.values()))


def _add(desp: NpxProbeDesp, chmap: ChannelMap, cand: dict[K, NpxElectrodeDesp], e: NpxElectrodeDesp):
    try:
        desp.add_electrode(chmap, e)
    except BaseException:
        return

    _del(cand, e)

    for k in list(cand):
        # if not desp.probe_rule(chmap, e, cand[k]):
        if e.channel == cand[k].channel:
            del cand[k]


def _del(cand: dict[K, NpxElectrodeDesp], e: NpxElectrodeDesp | None):
    if e is not None:
        try:
            del cand[e.electrode]
        except KeyError:
            pass
