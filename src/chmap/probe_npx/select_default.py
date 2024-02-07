import random

from .desp import NpxProbeDesp, NpxElectrodeDesp, K
from .npx import ChannelMap, ProbeType

__all__ = ['electrode_select']


def electrode_select(desp: NpxProbeDesp, chmap: ChannelMap, blueprint: list[NpxElectrodeDesp], **kwargs) -> ChannelMap:
    ret = desp.new_channelmap(chmap)

    cand: dict[K, NpxElectrodeDesp] = {it.electrode: it for it in desp.all_electrodes(ret)}
    for e in blueprint:
        cand[e.electrode].policy = e.policy

    for e in blueprint:
        # add pre-selected
        if e.policy == NpxProbeDesp.POLICY_SET:
            _add(desp, ret, cand, e)

        # remove forbidden electrodes from the candidate set
        elif e.policy == NpxProbeDesp.POLICY_FORBIDDEN:
            try:
                del cand[e.electrode]
            except KeyError:
                pass

    return select_loop(desp, ret, cand, **kwargs)


def select_loop(desp: NpxProbeDesp, chmap: ChannelMap, cand: dict[K, NpxElectrodeDesp],
                **kwargs) -> ChannelMap:
    while len(cand):
        p, e = pick_electrode(cand)
        if p == NpxProbeDesp.POLICY_FORBIDDEN:
            break
        elif e is not None:
            update(desp, chmap, cand, e, p)
        else:
            break

    return chmap


def pick_electrode(cand: dict[K, NpxElectrodeDesp]) -> tuple[int, NpxElectrodeDesp | None]:
    if len(cand) == 0:
        return NpxProbeDesp.POLICY_FORBIDDEN, None

    if len(ret := [e for e in cand.values() if e.policy == NpxProbeDesp.POLICY_D1]) > 0:
        return NpxProbeDesp.POLICY_D1, random.choice(ret)

    if len(ret := [e for e in cand.values() if e.policy == NpxProbeDesp.POLICY_D2]) > 0:
        return NpxProbeDesp.POLICY_D2, random.choice(ret)

    if len(ret := [e for e in cand.values() if e.policy == NpxProbeDesp.POLICY_D4]) > 0:
        return NpxProbeDesp.POLICY_D4, random.choice(ret)

    if len(ret := [e for e in cand.values() if e.policy == NpxProbeDesp.POLICY_LOW]) > 0:
        return NpxProbeDesp.POLICY_LOW, random.choice(ret)

    if len(ret := [e for e in cand.values() if e.policy == NpxProbeDesp.POLICY_UNSET]) > 0:
        return NpxProbeDesp.POLICY_UNSET, random.choice(ret)

    return NpxProbeDesp.POLICY_FORBIDDEN, None


def update(desp: NpxProbeDesp, chmap: ChannelMap, cand: dict[K, NpxElectrodeDesp], e: NpxElectrodeDesp, policy: int):
    match policy:
        case NpxProbeDesp.POLICY_D1:
            return update_d1(desp, chmap, cand, e)
        case NpxProbeDesp.POLICY_D2:
            return update_d2(desp, chmap, cand, e)
        case NpxProbeDesp.POLICY_D4:
            return update_d4(desp, chmap, cand, e)
        case NpxProbeDesp.POLICY_LOW | NpxProbeDesp.POLICY_UNSET:
            return _add(desp, chmap, cand, e)
        case _:
            raise ValueError()


def update_d1(desp: NpxProbeDesp, chmap: ChannelMap, cand: dict[K, NpxElectrodeDesp], e: NpxElectrodeDesp):
    _add(desp, chmap, cand, e)

    if (t := _get(chmap, cand, e, 1, 0)) is not None:
        _add(desp, chmap, cand, t)

    if (t := _get(chmap, cand, e, 0, 1)) is not None:
        update_d1(desp, chmap, cand, t)
    if (t := _get(chmap, cand, e, 0, -1)) is not None:
        update_d1(desp, chmap, cand, t)


def update_d2(desp: NpxProbeDesp, chmap: ChannelMap, cand: dict[K, NpxElectrodeDesp], e: NpxElectrodeDesp):
    _add(desp, chmap, cand, e)
    _del(cand, _get(chmap, cand, e, 1, 0))

    if (t := _get(chmap, cand, e, 1, 1)) is not None:
        update_d2(desp, chmap, cand, t)
    if (t := _get(chmap, cand, e, 1, -1)) is not None:
        update_d2(desp, chmap, cand, t)


def update_d4(desp: NpxProbeDesp, chmap: ChannelMap, cand: dict[K, NpxElectrodeDesp], e: NpxElectrodeDesp):
    _add(desp, chmap, cand, e)
    _del(cand, _get(chmap, cand, e, 1, 0))
    _del(cand, _get(chmap, cand, e, 0, 1))
    _del(cand, _get(chmap, cand, e, 1, 1))
    _del(cand, _get(chmap, cand, e, 0, -1))
    _del(cand, _get(chmap, cand, e, 1, -1))

    if (t := _get(chmap, cand, e, 1, 2)) is not None:
        update_d4(desp, chmap, cand, t)
    if (t := _get(chmap, cand, e, 1, -2)) is not None:
        update_d4(desp, chmap, cand, t)


def _add(desp: NpxProbeDesp, chmap: ChannelMap, cand: dict[K, NpxElectrodeDesp], e: NpxElectrodeDesp):
    try:
        desp.add_electrode(chmap, e)
    except BaseException:
        return

    _del(cand, e)

    for k in list(cand):
        if not desp.probe_rule(chmap, e, cand[k]):
            del cand[k]


def _del(cand: dict[K, NpxElectrodeDesp], e: NpxElectrodeDesp | None):
    if e is not None:
        try:
            del cand[e.electrode]
        except KeyError:
            pass


def _get(chmap: ChannelMap, cand: dict[K, NpxElectrodeDesp], e: NpxElectrodeDesp, c: int, r: int) -> NpxElectrodeDesp | None:
    ret = cand.get(_move(chmap.probe_type, e, c, r), None)
    return ret if ret is not None and ret.policy == e.policy else None


def _move(probe_type: ProbeType, e: NpxElectrodeDesp, c: int, r: int) -> K:
    eh, ec, er = e.electrode
    nc = probe_type.n_col_shank
    return eh, (ec + c) % nc, (er + r)
