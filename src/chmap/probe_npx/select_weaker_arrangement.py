import random
from collections.abc import Iterator

from .desp import NpxProbeDesp, NpxElectrodeDesp, K
from .npx import ChannelMap, ProbeType

__all__ = ['electrode_select']


def electrode_select(desp: NpxProbeDesp, chmap: ChannelMap, s: list[NpxElectrodeDesp], *,
                     limit: int = 2000,
                     **kwargs) -> ChannelMap:
    ret = desp.new_channelmap(chmap)

    cand: dict[K, NpxElectrodeDesp] = {it.electrode: it for it in desp.all_electrodes(ret)}
    for e in s:
        cand[e.electrode].policy = e.policy

    for e in cand.values():
        e.prob = policy_mapping_priority(e.policy)

    for e in s:
        # add pre-selected
        if e.policy == NpxProbeDesp.POLICY_SET:
            _add(desp, ret, cand, e)

    count = 0
    while len(cand) and count < limit:
        e = pick_electrode(cand)
        if e is not None:
            print('pick', e, e.channel, e.prob)
            update(desp, chmap, cand, e)
            count += 1
        else:
            break

    return build_channelmap(desp, chmap, cand)


def policy_mapping_priority(p: int) -> float:
    match p:
        case NpxProbeDesp.POLICY_SET:
            return 1.0
        case NpxProbeDesp.POLICY_D1:
            return 0.9
        case NpxProbeDesp.POLICY_D2:
            return 0.8  # 0.4, 0.2
        case NpxProbeDesp.POLICY_D4:
            return 0.7  # 0.35, 0.125
        case NpxProbeDesp.POLICY_REMAINDER:
            return 0.6
        case NpxProbeDesp.POLICY_FORBIDDEN:
            return 0
        # case NpxProbeDesp.POLICY_UNSET:
        case _:
            return 0.5


def build_channelmap(desp: NpxProbeDesp, chmap: ChannelMap, cand: dict[K, NpxElectrodeDesp]) -> ChannelMap:
    ret = desp.new_channelmap(chmap)

    for e in cand.values():
        if e.prob == 1:
            desp.add_electrode(ret, e, overwrite=True)

    print(ret)
    return ret


def pick_electrode(cand: dict[K, NpxElectrodeDesp]) -> NpxElectrodeDesp | None:
    if len(cand) == 0:
        raise RuntimeError()

    hp = max([it.prob for it in cand.values() if it.prob < 1])

    if hp == 0:
        return None

    print(f'{hp=}')
    sub = [it for it in cand.values() if it.prob >= hp]
    assert len(sub) > 0
    return random.choice(sub)


def update(desp: NpxProbeDesp, chmap: ChannelMap, cand: dict[K, NpxElectrodeDesp], e: NpxElectrodeDesp):
    _add(desp, chmap, cand, e)

    for t in surr(chmap.probe_type, cand, e):
        if t is not None and t.prob < 1:
            t.prob /= 2.0


def surr(probe_type: ProbeType, cand: dict[K, NpxElectrodeDesp], e: NpxElectrodeDesp) -> Iterator[NpxElectrodeDesp]:
    match e.policy:
        case NpxProbeDesp.POLICY_D2:
            yield _get(probe_type, cand, e, 1, 0)
            yield _get(probe_type, cand, e, 0, 1)
            yield _get(probe_type, cand, e, 0, -1)
        case NpxProbeDesp.POLICY_D4:
            yield _get(probe_type, cand, e, 1, 0)
            yield _get(probe_type, cand, e, 0, 1)
            yield _get(probe_type, cand, e, 1, 1)
            yield _get(probe_type, cand, e, 0, -1)
            yield _get(probe_type, cand, e, 1, -1)
        case _:
            return


def _add(desp: NpxProbeDesp, chmap: ChannelMap, cand: dict[K, NpxElectrodeDesp], e: NpxElectrodeDesp):
    e.prob = 1.0

    for k in cand.values():
        if not desp.probe_rule(chmap, e, k):
            k.prob = 0


def _get(probe_type: ProbeType, cand: dict[K, NpxElectrodeDesp], e: NpxElectrodeDesp, c: int, r: int) -> NpxElectrodeDesp | None:
    ret = cand.get(_move(probe_type, e, c, r), None)
    return ret if ret is not None and ret.policy == e.policy else None


def _move(probe_type: ProbeType, e: NpxElectrodeDesp, c: int, r: int) -> K:
    eh, ec, er = e.electrode
    nc = probe_type.n_col_shank
    return eh, (ec + c) % nc, (er + r)


if __name__ == '__main__':
    debug()
