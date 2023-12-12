from pathlib import Path
from typing import Any

from probeinterface import Probe

from .npx import *

__all__ = [
    'parse_imro',
    'string_imro',
    'load_meta',
    'load_imro',
    'save_imro',
    'from_probe',
    'to_probe'
]


def parse_imro(source: str) -> ChannelMap:
    if not source.startswith('(') or not source.endswith(')'):
        raise RuntimeError('not imro format')

    from .npx import e2cr, e2c21, e2c24

    type_code = -1
    ref = 0
    electrodes = []

    i = 0  # left '('
    j = source.index(')')  # right ')'
    k = 0  # count of '(...)'

    while 0 <= i < j:
        part = source[i + 1:j]

        if k == 0:  # first ()
            type_code, n = tuple(map(int, part.split(',')))
            if type_code not in PROBE_TYPE:
                raise RuntimeError(f"unsupported probe type : {type_code}")
            k += 1

        elif type_code == 0:  # NP1
            ch, bank, ref, a, l, f = tuple(map(int, part.split(' ')))
            e = Electrode(0, *e2cr(PROBE_TYPE_NP1, ch))
            e.ap_band_gain = a
            e.lf_band_gain = l
            e.ap_hp_filter = f != 0
            electrodes.append(e)

        elif type_code == 21:  # NP 2.0, single multiplexed shank
            ch, bank, ref, ed = tuple(map(int, part.split(' ')))
            assert e2c21(ed) == (ch, bank)
            electrodes.append(Electrode(0, *e2cr(PROBE_TYPE_NP21, ed)))

        elif type_code == 24:  # NP 2.0, 4-shank
            ch, s, bank, ref, ed = tuple(map(int, part.split(' ')))
            assert e2c24(s, ed) == (ch, bank)
            electrodes.append(Electrode(s, *e2cr(PROBE_TYPE_NP24, ed)))

        else:
            raise RuntimeError(f'unsupported imro type : {type_code}')

        i = j + 1
        if i < len(source):
            if source[i] != '(':
                raise RuntimeError()

            j = source.index(')', i)
        else:
            j = -1

    ret = ChannelMap(type_code, electrodes)
    ret.reference = ref
    return ret


def string_imro(chmap: ChannelMap) -> str:
    if len(chmap) != chmap.n_channels:
        raise RuntimeError()

    # header
    ret = [f'({chmap.probe_type.code},{chmap.n_channels})']

    # channels
    match chmap.probe_type.code:
        case 0:
            for ch, e in enumerate(chmap.electrodes):
                ret.append(f'({ch} 0 {chmap.reference} {e.ap_band_gain} {e.lf_band_gain} {1 if e.ap_hp_filter else 0})')

        case 21:
            from .npx import cr2e, e2c21
            ref = chmap.reference
            for e in chmap.electrodes:
                electrode = cr2e(PROBE_TYPE_NP21, e)
                channel, bank = e2c21(electrode)
                ret.append(f'({channel} {bank} {ref} {electrode})')

        case 24:
            from .npx import cr2e, e2c24
            ref = chmap.reference
            for e in chmap.electrodes:
                electrode = cr2e(PROBE_TYPE_NP24, e)
                bank, channel = e2c24(e.shank, electrode)
                ret.append(f'({channel} {e.shank} {bank} {ref} {electrode})')
        case _:
            raise RuntimeError(f'unknown imro type : {chmap.probe_type}')

    return ''.join(ret)


def load_meta(path: str | Path) -> ChannelMap:
    path = Path(path)
    if path.suffix != '.meta':
        raise RuntimeError()
    return ChannelMap.parse(_load_meta(path)['~imroTbl'])


def load_imro(path: str | Path) -> ChannelMap:
    path = Path(path)
    if path.suffix != '.imro':
        raise RuntimeError()

    return ChannelMap.parse(path.read_text().split('\n')[0])


def _load_meta(path: Path) -> dict[str, Any]:
    ret = {}
    with path.open() as f:
        for line in f:
            k, _, v = line.rstrip().partition('=')
            ret[k] = v
    return ret


def save_imro(chmap: ChannelMap, path: str | Path):
    path = Path(path)
    if path.suffix != '.imro':
        raise RuntimeError()

    with path.open('w') as f:
        print(str(chmap), file=f)


def from_probe(probe: Probe) -> ChannelMap:
    if probe.manufacturer != 'IMEC':
        raise RuntimeError('not a Neuropixels probe')
    if 'Neuropixels' not in probe.model_name:
        raise RuntimeError('not a Neuropixels probe')

    probe_type_raw = probe.annotations['probe_type']
    probe_type = PROBE_TYPE[probe_type_raw]
    s = probe.shank_ids.astype(int)
    cr = probe.contact_positions.astype(int)
    cr[:, 0] %= probe_type.s_space
    cr[:, 0] //= probe_type.c_space
    cr[:, 1] //= probe_type.r_space

    ret = ChannelMap(probe_type)
    for ss, (cc, rr) in zip(s, cr):
        ret.add_electrode((ss, cc, rr))
    return ret


def to_probe(chmap: ChannelMap) -> Probe:
    from probeinterface.io import _read_imro_string
    return _read_imro_string(repr(chmap))
