from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, TYPE_CHECKING

from .meta import NpxMeta
from .npx import *

if TYPE_CHECKING:
    import pandas as pd  # type: ignore[import]
    import polars as pl  # type: ignore[import]
    from probeinterface import Probe  # type: ignore[import]

__all__ = [
    'parse_imro',
    'string_imro',
    'load_meta',
    'load_imro',
    'save_imro',
    'from_probe',
    'to_probe',
    'to_pandas',
    'to_polars'
]


# ===================== #
# imro table expression #
# ===================== #

def parse_imro(source: str) -> ChannelMap:
    source = source.strip()
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
            try:
                ProbeType[type_code]
            except KeyError as e:
                raise RuntimeError(f"unsupported probe type : {type_code}") from e

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
            for ch, e in enumerate(chmap.channels):
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
                channel, bank = e2c24(e.shank, electrode)
                ret.append(f'({channel} {e.shank} {bank} {ref} {electrode})')
        case _:
            raise RuntimeError(f'unknown imro type : {chmap.probe_type}')

    return ''.join(ret)


# ======================= #
# SpikeGLX imro/meta file #
# ======================= #

def load_meta(path: str | Path) -> ChannelMap:
    path = Path(path)
    if path.suffix != '.meta':
        raise IOError()

    data = _load_meta(path)

    meta = NpxMeta(serial_number=data['imDatPrb_sn'], imro_table=data['~imroTbl'])
    meta.update(data)

    ret = ChannelMap.parse(meta['imro_table'])
    ret.meta = meta
    return ret


def load_imro(path: str | Path) -> ChannelMap:
    path = Path(path)
    if path.suffix != '.imro':
        raise IOError(f'unknown file format {path.suffix}')

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
        raise IOError()

    imro = string_imro(chmap)
    with path.open('w') as f:
        print(imro, file=f)


# ============== #
# probeinterface #
# ============== #

def from_probe(probe: Probe) -> ChannelMap:
    if probe.manufacturer != 'IMEC':
        raise RuntimeError('not a Neuropixels probe')
    if 'Neuropixels' not in probe.model_name:
        raise RuntimeError('not a Neuropixels probe')

    probe_type_raw = probe.annotations['probe_type']
    probe_type = ProbeType[probe_type_raw]
    s = probe.shank_ids.astype(int)
    cr = probe.contact_positions.astype(int)
    cr[:, 0] %= probe_type.s_space
    cr[:, 0] //= probe_type.c_space
    cr[:, 1] //= probe_type.r_space

    meta = NpxMeta(serial_number=probe.serial_number)
    meta.update(probe.annotations)

    ret = ChannelMap(probe_type)
    ret.meta = meta

    for ss, (cc, rr) in zip(s, cr):
        ret.add_electrode((ss, cc, rr))
    return ret


def to_probe(chmap: ChannelMap) -> Probe:
    from probeinterface.io import _read_imro_string  # type: ignore[import]
    return _read_imro_string(string_imro(chmap))


# ======================= #
# pandas/polars dataframe #
# ======================= #

def to_pandas(chmap: ChannelMap) -> pd.DataFrame:
    """
    To a pandas dataframe.

    Use ``-1`` to fill empty channels.

    ::

                 shank  column  row  in_used    x     y
        channel
        0           -1      -1   -1    False   -1    -1
        1            1       1  144     True  282  2160
        ...        ...     ...  ...      ...  ...   ...

    :param chmap:
    :return: a pandas dataframe
    """
    import pandas as pd

    probe_type = chmap.probe_type
    h = list(range(probe_type.n_channels))
    s = [it.shank if it is not None else -1 for it in chmap.channels]
    c = [it.column if it is not None else -1 for it in chmap.channels]
    r = [it.row if it is not None else -1 for it in chmap.channels]
    u = [it.in_used if it is not None else False for it in chmap.channels]

    ret = pd.DataFrame(data=dict(shank=s, column=c, row=r, in_used=u), index=pd.Index(h, name='channel'))
    ret['x'] = ret['shank'] * probe_type.s_space + ret['column'] * probe_type.c_space
    ret['y'] = ret['row'] * probe_type.r_space
    ret.loc[ret['shank'] == -1, ['x', 'y']] = -1

    return ret


def to_polars(chmap: ChannelMap) -> pl.DataFrame:
    """
    To a polars dataframe.

    Use ``null`` to fill empty channels.

    ::

        ┌─────────┬───────┬────────┬──────┬─────────┬──────┬──────┐
        │ channel ┆ shank ┆ column ┆ row  ┆ in_used ┆ x    ┆ y    │
        │ i64     ┆ i64?  ┆ i64?   ┆ i64? ┆ bool    ┆ i64? ┆ i64? │
        ╞═════════╪═══════╪════════╪══════╪═════════╪══════╪══════╡
        │ 0       ┆ null  ┆ null   ┆ null ┆ false   ┆ null ┆ null │
        │ 1       ┆ 1     ┆ 1      ┆ 144  ┆ true    ┆ 282  ┆ 2160 │
        └─────────┴───────┴────────┴──────┴─────────┴──────┴──────┘


    :param chmap:
    :return: a polars dataframe
    """
    import polars as pl

    probe_type = chmap.probe_type
    h = list(range(probe_type.n_channels))
    s = [it.shank if it is not None else None for it in chmap.channels]
    c = [it.column if it is not None else None for it in chmap.channels]
    r = [it.row if it is not None else None for it in chmap.channels]
    u = [it.in_used if it is not None else False for it in chmap.channels]

    return pl.DataFrame(data=dict(channel=h, shank=s, column=c, row=r, in_used=u)).with_columns(
        x=pl.col('shank') * probe_type.s_space + pl.col('column') * probe_type.c_space,
        y=pl.col('row') * probe_type.r_space
    )


if __name__ == '__main__':
    parse_imro(Path(sys.argv[1]).read_text())
