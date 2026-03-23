from __future__ import annotations

import sys
import textwrap
from pathlib import Path
from typing import Any, TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

from neurocarto.util.utils import doc_link
from .meta import NpxMeta
from .npx import *

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl
    from probeinterface import Probe
    from .npx import ImroEC_NP1110

__all__ = [
    'parse_imro',
    'string_imro',
    'load_meta',
    'load_imro',
    'save_imro',
    'from_probe',
    'to_numpy',
    'to_probe',
    'to_pandas',
    'to_polars'
]


# ===================== #
# imro table expression #
# ===================== #

@doc_link(DOC=textwrap.dedent(ChannelMap.parse.__doc__))
def parse_imro(source: str) -> ChannelMap:
    """
    {DOC}
    :see: {ChannelMap#parse()}
    """
    # https://github.com/billkarsh/SpikeGLX/blob/bc2c10e99e68dcc9ec6b9a9c75272a74c7e53034/Src-imro/IMROTbl_T0base.cpp#L133
    # bool IMROTbl_T0base::fromString( QString *msg, const QString &s )
    # https://github.com/billkarsh/SpikeGLX/blob/bc2c10e99e68dcc9ec6b9a9c75272a74c7e53034/Src-imro/IMROTbl_T0base.cpp#L34
    # bool IMRODesc_T0base::fromString( QString *msg, const QString &s )
    source = source.strip()
    if not source.startswith('(') or not source.endswith(')'):
        raise RuntimeError('not imro format')

    type_code = -1
    io: ImroIO = None
    electrodes = []

    i = 0  # left '('
    j = source.index(')')  # right ')'
    k = 0  # count of '(...)'


    while 0 <= i < j:
        part = source[i + 1:j]

        if k == 0:  # first ()
            type_code, *args = tuple(map(int, part.split(',')))
            try:
                io = ImroIO(ProbeType[type_code])
            except KeyError as e:
                raise RuntimeError(f"unsupported probe type : {type_code}") from e

            io.parse_header(*args)
            k += 1
        else:
            electrodes.append(io.parse_electrode(*tuple(map(int, part.split(' ')))))

        i = j + 1
        if i < len(source):
            if source[i] != '(':
                raise RuntimeError()

            j = source.index(')', i)
        else:
            j = -1

    ret = ChannelMap(type_code, electrodes)
    ret.reference = io.reference
    return ret


@doc_link(DOC=textwrap.dedent(ChannelMap.to_imro.__doc__))
def string_imro(chmap: ChannelMap) -> str:
    """
    {DOC}
    :see: {ChannelMap#to_imro()}
    """
    if len(chmap) != chmap.n_channels:
        raise RuntimeError(f'channel number in chmap is not {chmap.n_channels}, but {len(chmap)}')

    io = ImroIO(chmap.probe_type)

    # header
    header = ','.join(map(str, io.string_header(chmap)))
    ret = [f'({header})']

    # channels
    for c, e in enumerate(chmap.electrodes):
        contents = ' '.join(map(str, io.string_electrode(chmap, c, e)))
        ret.append(f'({contents})')

    return ''.join(ret)


class ImroIO(object):
    def __new__(cls, probe_type: ProbeType):
        match probe_type.code:
            case 0:
                ret = ImroIO_NP1
            case 21 | 2003:
                ret = ImroIO_NP21
            case 24 | 2013:
                ret = ImroIO_NP24
            case 1110:
                ret = ImroIO_NP1110
            case 2020:
                ret = ImroIO_NP2020
            case 3010:
                ret = ImroIO_NP3010
            case 3020:
                ret = ImroIO_NP3020
            case _:
                ret = ImroIO_NP1

        return object.__new__(ret)

    def __init__(self, probe_type: ProbeType):
        self.probe_type: ProbeType = probe_type
        self.reference: int = 0

        from .npx import ImroEC
        self.to = ImroEC(probe_type)

    def parse_header(self, *args: int):
        pass

    def parse_electrode(self, *args: int) -> Electrode:
        raise NotImplementedError()

    def string_header(self, chmap: ChannelMap) -> tuple[int, ...]:
        return chmap.probe_type.code, chmap.n_channels

    def string_electrode(self, chmap: ChannelMap, ch: int, e: Electrode) -> tuple[int, ...]:
        raise NotImplementedError()


class ImroIO_NP1(ImroIO):

    def parse_electrode(self, *args: int) -> Electrode:
        # https://github.com/billkarsh/SpikeGLX/blob/bc2c10e99e68dcc9ec6b9a9c75272a74c7e53034/Src-imro/IMROTbl_T0base.cpp#L34
        ch, bank, ref, a, l, f = args
        e = self.to.c2e(ch, bank)
        e = Electrode(0, *self.to.e2cr(e))
        e.ap_band_gain = a
        e.lf_band_gain = l
        e.ap_hp_filter = f != 0
        self.reference = ref
        return e

    def string_electrode(self, chmap: ChannelMap, ch: int, e: Electrode) -> tuple[int, ...]:
        # https://github.com/billkarsh/SpikeGLX/blob/bc2c10e99e68dcc9ec6b9a9c75272a74c7e53034/Src-imro/IMROTbl_T0base.cpp#L20
        from .npx import cr2e
        c, bank = self.to.e2c(cr2e(self.probe_type, e))
        assert c == ch
        return ch, bank, chmap.reference, e.ap_band_gain, e.lf_band_gain, 1 if e.ap_hp_filter else 0


class ImroIO_NP21(ImroIO):

    def parse_electrode(self, *args: int) -> Electrode:
        # https://github.com/billkarsh/SpikeGLX/blob/bc2c10e99e68dcc9ec6b9a9c75272a74c7e53034/Src-imro/IMROTbl_T21base.cpp#L80
        ch, bank, ref, ed = args
        # https://github.com/billkarsh/SpikeGLX/blob/bc2c10e99e68dcc9ec6b9a9c75272a74c7e53034/Src-imro/IMROTbl_T21base.cpp#L21
        # mbank is multibank field, we take only lowest connected bank
        match (bank & -bank):
            case 1:
                bank = 0
            case 2:
                bank = 1
            case 4:
                bank = 2
            case _:
                bank = 3
        assert self.to.e2c(ed) == (ch, bank), f'{ed=},{ch=},{bank=},e2c={self.to.e2c(ed)}'
        self.reference = ref
        return Electrode(0, *self.to.e2cr(ed))

    def string_electrode(self, chmap: ChannelMap, ch: int, e: Electrode) -> tuple[int, ...]:
        # https://github.com/billkarsh/SpikeGLX/blob/bc2c10e99e68dcc9ec6b9a9c75272a74c7e53034/Src-imro/IMROTbl_T21base.cpp#L68
        from .npx import cr2e
        electrode = cr2e(self.probe_type, e)
        channel, bank = self.to.e2c(electrode)
        bank = 1 << bank
        return ch, bank, chmap.reference, electrode


class ImroIO_NP24(ImroIO):

    def parse_electrode(self, *args: int) -> Electrode:
        # https://github.com/billkarsh/SpikeGLX/blob/bc2c10e99e68dcc9ec6b9a9c75272a74c7e53034/Src-imro/IMROTbl_T24base.cpp#L50
        ch, s, bank, ref, ed = args
        assert self.to.e2c(ed, s) == (ch, bank), f'{ed=},{s=},{ch=},{bank=},e2c={self.to.e2c(ed, s)}'
        self.reference = ref
        return Electrode(s, *self.to.e2cr(ed))

    def string_electrode(self, chmap: ChannelMap, ch: int, e: Electrode) -> tuple[int, ...]:
        # https://github.com/billkarsh/SpikeGLX/blob/bc2c10e99e68dcc9ec6b9a9c75272a74c7e53034/Src-imro/IMROTbl_T24base.cpp#L37
        from .npx import cr2e
        electrode = cr2e(self.probe_type, e)
        channel, bank = self.to.e2c(electrode, e.shank)
        return ch, e.shank, bank, chmap.reference, electrode


class ImroIO_NP1110(ImroIO):
    INNER = 0
    OUTER = 1
    ALL = 2

    col_mode: Literal[0, 1, 2] = 2
    """
    0 : INNER mode upper cols are odd  and lower cols are even.
    1 : OUTER mode upper cols are even and lower cols are odd.
    2 : ALL mode bankA and bankB are the same.
    """

    ap: int = 500
    lf: int = 250
    af: int = 1
    to: ImroEC_NP1110

    def parse_header(self, *args: int):
        col_mode, ref_id, ap, lf, af = args

        if col_mode not in (0, 1, 2):
            raise ValueError(f'illegal col mode value : {col_mode}')

        self.col_mode = col_mode
        self.reference = ref_id
        self.ap = ap
        self.lf = lf
        self.af = af

    def _bank(self, ch: int, bank_a: int, bank_b: int):
        # https://github.com/billkarsh/SpikeGLX/blob/bc2c10e99e68dcc9ec6b9a9c75272a74c7e53034/Src-imro/IMROTbl_T1110.cpp#L201
        match self.col_mode:
            case self.ALL:
                if bank_a != bank_b:
                    raise ValueError('In All col mode must have same bankA and bankB value')

                return bank_a
            case self.OUTER:
                if not self._is_bank_crossed(bank_a, bank_b):
                    raise ValueError('In OUTER mode, only either bankA or bankB is col-crossed exactly')

                if self.to.col(ch, bank_a) in (0, 2, 5, 7):
                    return bank_a
                else:
                    return bank_b

            case self.INNER:
                if not self._is_bank_crossed(bank_a, bank_b):
                    raise ValueError('In INNER mode, only either bankA or bankB is col-crossed exactly')

                if self.to.col(ch, bank_a) in (1, 3, 4, 6):
                    return bank_a
                else:
                    return bank_b

            case _:
                raise ValueError('illegal mode')

    @classmethod
    def _is_bank_crossed(cls, bank_a: int, bank_b: int) -> bool:
        # https://github.com/billkarsh/SpikeGLX/blob/bc2c10e99e68dcc9ec6b9a9c75272a74c7e53034/Src-imro/IMROTbl_T1110.cpp#L170
        a = (bank_a / 4) % 2
        b = (bank_b / 4) % 2
        return a != b

    def parse_electrode(self, *args: int) -> Electrode:
        # https://github.com/billkarsh/SpikeGLX/blob/bc2c10e99e68dcc9ec6b9a9c75272a74c7e53034/Src-imro/IMROTbl_T1110.cpp#L42
        group, bank_a, bank_b = args
        bank = self._bank(group, bank_a, bank_b)
        col = self.to.col(group, bank)
        row = self.to.row(group, bank)
        e = Electrode(0, col, row)
        e.bank_a = bank_a
        e.bank_b = bank_b
        e.ap_band_gain = self.ap
        e.lf_band_gain = self.lf
        e.ap_hp_filter = self.af != 0
        return e

    def string_header(self, chmap: ChannelMap) -> tuple[int, ...]:
        # https://github.com/billkarsh/SpikeGLX/blob/bc2c10e99e68dcc9ec6b9a9c75272a74c7e53034/Src-imro/IMROTbl_T1110.cpp#L19
        e = chmap.channels[0]
        return chmap.probe_type.code, self.col_mode, chmap.reference, e.ap_band_gain, e.lf_band_gain, 1 if e.ap_hp_filter else 0

    def string_electrode(self, chmap: ChannelMap, ch: int, e: Electrode) -> tuple[int, ...]:
        # https://github.com/billkarsh/SpikeGLX/blob/bc2c10e99e68dcc9ec6b9a9c75272a74c7e53034/Src-imro/IMROTbl_T1110.cpp#L32
        return ch, e.bank_a, e.bank_b


class ImroIO_NP2020(ImroIO):

    def parse_electrode(self, *args: int) -> Electrode:
        # https://github.com/billkarsh/SpikeGLX/blob/bc2c10e99e68dcc9ec6b9a9c75272a74c7e53034/Src-imro/IMROTbl_T2020.cppL#35
        ch, s, bank, ref, ed = args
        assert self.to.e2c(ed, s) == (ch, bank), f'{ed=},{s=},{ch=},{bank=},e2c={self.to.e2c(ed, s)}'
        self.reference = ref
        return Electrode(s, *self.to.e2cr(ed))

    def string_electrode(self, chmap: ChannelMap, ch: int, e: Electrode) -> tuple[int, ...]:
        # https://github.com/billkarsh/SpikeGLX/blob/bc2c10e99e68dcc9ec6b9a9c75272a74c7e53034/Src-imro/IMROTbl_T2020.cpp#L22
        from .npx import cr2e
        electrode = cr2e(self.probe_type, e)
        channel, bank = self.to.e2c(electrode, e.shank)
        return ch, e.shank, bank, chmap.reference, electrode % 384


class ImroIO_NP3010(ImroIO):

    def parse_electrode(self, *args: int) -> Electrode:
        # https://github.com/billkarsh/SpikeGLX/blob/bc2c10e99e68dcc9ec6b9a9c75272a74c7e53034/Src-imro/IMROTbl_T3010base.cpp#L32
        ch, bank, ref, ed = args
        assert self.to.e2c(ed) == (ch, bank), f'{ed=},{ch=},{bank=},e2c={self.to.e2c(ed)}'
        self.reference = ref
        return Electrode(0, *self.to.e2cr(ed))

    def string_electrode(self, chmap: ChannelMap, ch: int, e: Electrode) -> tuple[int, ...]:
        # https://github.com/billkarsh/SpikeGLX/blob/bc2c10e99e68dcc9ec6b9a9c75272a74c7e53034/Src-imro/IMROTbl_T3010base.cpp#L20
        from .npx import cr2e
        electrode = cr2e(self.probe_type, e)
        channel, bank = self.to.e2c(electrode)
        return ch, bank, chmap.reference, electrode


class ImroIO_NP3020(ImroIO):
    def parse_electrode(self, *args: int) -> Electrode:
        # https://github.com/billkarsh/SpikeGLX/blob/bc2c10e99e68dcc9ec6b9a9c75272a74c7e53034/Src-imro/IMROTbl_T3020base.cpp#L50
        ch, s, bank, ref, ed = args
        assert self.to.e2c(ed, s) == (ch, bank), f'{ed=},{s=},{ch=},{bank=},e2c={self.to.e2c(ed, s)}'
        self.reference = ref
        return Electrode(s, *self.to.e2cr(ed))

    def string_electrode(self, chmap: ChannelMap, ch: int, e: Electrode) -> tuple[int, ...]:
        # https://github.com/billkarsh/SpikeGLX/blob/bc2c10e99e68dcc9ec6b9a9c75272a74c7e53034/Src-imro/IMROTbl_T3020base.cpp#L37
        from .npx import cr2e
        electrode = cr2e(self.probe_type, e)
        channel, bank = self.to.e2c(e.shank, electrode)
        return ch, e.shank, bank, chmap.reference, electrode

# ======================= #
# SpikeGLX imro/meta file #
# ======================= #

@doc_link(DOC=textwrap.dedent(ChannelMap.from_meta.__doc__))
def load_meta(path: str | Path) -> ChannelMap:
    """
    {DOC}
    :see: {ChannelMap#from_meta()}
    """
    path = Path(path)
    if path.suffix != '.meta':
        raise IOError()

    data = _load_meta(path)

    meta = NpxMeta(serial_number=data['imDatPrb_sn'], imro_table=data['~imroTbl'])
    meta.update(data)

    ret = ChannelMap.parse(meta['imro_table'])
    ret.meta = meta
    return ret


@doc_link(DOC=textwrap.dedent(ChannelMap.from_imro.__doc__))
def load_imro(path: str | Path) -> ChannelMap:
    """
    {DOC}
    :see: {ChannelMap#from_imro()}
    """
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


@doc_link(DOC=textwrap.dedent(ChannelMap.save_imro.__doc__))
def save_imro(chmap: ChannelMap, path: str | Path):
    """
    {DOC}
    :see: {ChannelMap#save_imro()}
    """
    path = Path(path)
    if path.suffix != '.imro':
        raise IOError()

    imro = string_imro(chmap)
    with path.open('w') as f:
        print(imro, file=f)


# ============== #
# probeinterface #
# ============== #

@doc_link(DOC=textwrap.dedent(ChannelMap.from_probe.__doc__))
def from_probe(probe: Probe) -> ChannelMap:
    """
    {DOC}
    :see: {ChannelMap#from_probe()}
    """
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


@doc_link(DOC=textwrap.dedent(ChannelMap.to_probe.__doc__))
def to_probe(chmap: ChannelMap) -> Probe:
    """
    {DOC}
    :see: {ChannelMap#to_probe()}
    """
    from probeinterface.io import _read_imro_string
    return _read_imro_string(string_imro(chmap))


# ================ #
# numpy electrodes #
# ================ #

@doc_link(DOC=textwrap.dedent(ChannelMap.to_numpy.__doc__))
def to_numpy(chmap: ChannelMap, unit: Literal['cr', 'xy', 'sxy'] = 'cr') -> NDArray[np.int_]:
    """
    {DOC}
    :see: {ChannelMap#to_numpy()}
    """

    match unit:
        case 'cr':
            def mapper(e: Electrode):
                return e.shank, e.column, e.row
        case 'xy':
            from .npx import e2p

            def mapper(e: Electrode):
                x, y = e2p(chmap.probe_type, e)
                return int(x), int(y)
        case 'sxy':
            def mapper(e: Electrode):
                x, y = e2p(chmap.probe_type, e)
                return e.shank, int(x), int(y)
        case _:
            raise ValueError(f'unsupported unit: {unit}')

    return np.array([mapper(e) for e in chmap.electrodes])

# ======================= #
# pandas/polars dataframe #
# ======================= #


@doc_link(DOC=textwrap.dedent(ChannelMap.to_pandas.__doc__))
def to_pandas(chmap: ChannelMap) -> pd.DataFrame:
    """
    {DOC}
    :see: {ChannelMap#to_pandas()}
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


@doc_link(DOC=textwrap.dedent(ChannelMap.to_polars.__doc__))
def to_polars(chmap: ChannelMap) -> pl.DataFrame:
    """
    {DOC}
    :see: {ChannelMap#to_polars()}
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
