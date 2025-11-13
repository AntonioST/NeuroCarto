from __future__ import annotations

import math
import sys
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sized, Sequence
from pathlib import Path
from typing import Any, NamedTuple, Final, Literal, overload, cast, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from neurocarto.util.utils import all_int, as_set, align_arr, doc_link
from .meta import NpxMeta

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl
    from probeinterface import Probe
    from .plot import ELECTRODE_UNIT

__all__ = [
    'ProbeType',
    'PROBE_TYPE',
    'PROBE_TYPE_NP1',
    'PROBE_TYPE_NP21',
    'PROBE_TYPE_NP24',
    'Electrode',
    'ReferenceInfo',
    'ChannelMap',
    'channel_coordinate',
    'electrode_coordinate',
    'ChannelHasUsedError',
]


class ProbeType(NamedTuple):
    """Probe profile.

    References:

    * `open-ephys-plugins <https://github.com/open-ephys-plugins/neuropixels-pxi/blob/master/Source/Probes/Geometry.cpp#L27>`_
    * `SpikeGLX <https://github.com/billkarsh/SpikeGLX/blob/bc2c10e99e68dcc9ec6b9a9c75272a74c7e53034/Src-imro/IMROTbl.h#L105>`_

    """
    code: int
    n_shank: int  # number of shank
    n_col_shank: int  # number of columns per shank (_ncolhwr)
    n_electrode_shank: int  # number of electrode per shank.
    n_channels: int  # number of total channels.
    c_space: float  # electrodes column space (_xpitch), um
    r_space: float  # electrodes row space (_zpitch), um
    s_space: int  # shank space (_shankpitch), um
    n_reference: int  # number of references

    @property
    def n_row_shank(self) -> int:
        return self.n_electrode_shank // self.n_col_shank

    @property
    def n_bank(self) -> int:
        """number of total banks"""
        return int(math.ceil(self.n_electrode_shank / self.n_channels))

    @classmethod
    def __class_getitem__(cls, item: int | str) -> ProbeType:
        return PROBE_TYPE[item]


# https://github.com/billkarsh/SpikeGLX/blob/bc2c10e99e68dcc9ec6b9a9c75272a74c7e53034/Src-imro/IMROTbl.cpp#L115

# https://github.com/billkarsh/SpikeGLX/blob/bc2c10e99e68dcc9ec6b9a9c75272a74c7e53034/Src-imro/IMROTbl_T0.h#L12
PROBE_TYPE_NP1 = ProbeType(0, 1, 2, 960, 384, 32, 20, 0, 5)
# https://github.com/billkarsh/SpikeGLX/blob/bc2c10e99e68dcc9ec6b9a9c75272a74c7e53034/Src-imro/IMROTbl_T21.h#L12
PROBE_TYPE_NP21 = ProbeType(21, 1, 2, 1280, 384, 32, 15, 0, 6)
# https://github.com/billkarsh/SpikeGLX/blob/bc2c10e99e68dcc9ec6b9a9c75272a74c7e53034/Src-imro/IMROTbl_T24.h#L12
PROBE_TYPE_NP24 = ProbeType(24, 4, 2, 1280, 384, 32, 15, 250, 21)

# PROBE_TYPE_NP1 based
# https://github.com/billkarsh/SpikeGLX/blob/bc2c10e99e68dcc9ec6b9a9c75272a74c7e53034/Src-imro/IMROTbl_T1020.h#L12
PROBE_TYPE_NP1020 = ProbeType(1020, 1, 2, 2496, 384, 87, 20, 0, 9)
PROBE_TYPE_NP1022 = ProbeType(1020, 1, 2, 2496, 384, 103, 20, 0, 9)
# https://github.com/billkarsh/SpikeGLX/blob/bc2c10e99e68dcc9ec6b9a9c75272a74c7e53034/Src-imro/IMROTbl_T1030.h#L12
PROBE_TYPE_NP1030 = ProbeType(1030, 1, 2, 4416, 384, 87, 20, 0, 14)
PROBE_TYPE_NP1032 = ProbeType(1030, 1, 2, 4416, 384, 103, 20, 0, 14)
# https://github.com/billkarsh/SpikeGLX/blob/bc2c10e99e68dcc9ec6b9a9c75272a74c7e53034/Src-imro/IMROTbl_T1100.h#L12
PROBE_TYPE_NP1100 = ProbeType(1100, 1, 8, 384, 384, 6, 6, 0, 2)
# https://github.com/billkarsh/SpikeGLX/blob/bc2c10e99e68dcc9ec6b9a9c75272a74c7e53034/Src-imro/IMROTbl_T1110.h#L61
PROBE_TYPE_NP1110 = ProbeType(1110, 1, 8, 6144, 384, 6, 6, 0, 2)
# https://github.com/billkarsh/SpikeGLX/blob/bc2c10e99e68dcc9ec6b9a9c75272a74c7e53034/Src-imro/IMROTbl_T1120.h#L12
PROBE_TYPE_NP1120 = ProbeType(1120, 1, 2, 384, 384, 4.5, 4.5, 0, 2)
# https://github.com/billkarsh/SpikeGLX/blob/bc2c10e99e68dcc9ec6b9a9c75272a74c7e53034/Src-imro/IMROTbl_T1121.h#L12
PROBE_TYPE_NP1121 = ProbeType(1121, 1, 1, 384, 384, 3, 3, 0, 2)
# https://github.com/billkarsh/SpikeGLX/blob/bc2c10e99e68dcc9ec6b9a9c75272a74c7e53034/Src-imro/IMROTbl_T1122.h#L12
PROBE_TYPE_NP1122 = ProbeType(1122, 1, 16, 384, 384, 3, 3, 0, 2)
# https://github.com/billkarsh/SpikeGLX/blob/bc2c10e99e68dcc9ec6b9a9c75272a74c7e53034/Src-imro/IMROTbl_T1123.h#L12
PROBE_TYPE_NP1123 = ProbeType(1123, 1, 12, 384, 384, 4.5, 4.5, 0, 2)
# https://github.com/billkarsh/SpikeGLX/blob/bc2c10e99e68dcc9ec6b9a9c75272a74c7e53034/Src-imro/IMROTbl_T1200.h#L12
PROBE_TYPE_NP1200 = ProbeType(1200, 1, 2, 128, 128, 32, 31, 0, 2)
# https://github.com/billkarsh/SpikeGLX/blob/bc2c10e99e68dcc9ec6b9a9c75272a74c7e53034/Src-imro/IMROTbl_T1300.h#L12
PROBE_TYPE_NP1300 = ProbeType(1300, 1, 2, 960, 384, 48, 20, 0, 5)

# https://github.com/billkarsh/SpikeGLX/blob/bc2c10e99e68dcc9ec6b9a9c75272a74c7e53034/Src-imro/IMROTbl_T2003.h#L12
PROBE_TYPE_NP2003 = ProbeType(2003, 1, 2, 1280, 384, 32, 15, 0, 3)
# https://github.com/billkarsh/SpikeGLX/blob/bc2c10e99e68dcc9ec6b9a9c75272a74c7e53034/Src-imro/IMROTbl_T2013.h#L12
PROBE_TYPE_NP2013 = ProbeType(2013, 4, 2, 1280, 384, 32, 15, 250, 6)

# NP2020 2.0 quad base (Ph 2C)
# https://github.com/billkarsh/SpikeGLX/blob/bc2c10e99e68dcc9ec6b9a9c75272a74c7e53034/Src-imro/IMROTbl_T2020.h#L43
PROBE_TYPE_NP2020 = ProbeType(2020, 4, 2, 1280, 1536, 32, 15, 250, 3)

# PROBE_TYPE_NP24 based

# NXT multishank
PROBE_TYPE_NP3000 = ProbeType(1200, 1, 2, 128, 128, 15, 15, 0, 2)
# https://github.com/billkarsh/SpikeGLX/blob/bc2c10e99e68dcc9ec6b9a9c75272a74c7e53034/Src-imro/IMROTbl_T3010base.h#L47
# https://github.com/billkarsh/SpikeGLX/blob/bc2c10e99e68dcc9ec6b9a9c75272a74c7e53034/Src-imro/IMROTbl_T3010.h#L12
PROBE_TYPE_NP3010 = ProbeType(3010, 1, 2, 1280, 912, 32, 15, 0, 3)
# https://github.com/billkarsh/SpikeGLX/blob/bc2c10e99e68dcc9ec6b9a9c75272a74c7e53034/Src-imro/IMROTbl_T3020base.h#L47
# https://github.com/billkarsh/SpikeGLX/blob/bc2c10e99e68dcc9ec6b9a9c75272a74c7e53034/Src-imro/IMROTbl_T3020.h#L12
PROBE_TYPE_NP3020 = ProbeType(3020, 4, 2, 1280, 912, 32, 15, 250, 7)


# https://github.com/billkarsh/SpikeGLX/blob/bc2c10e99e68dcc9ec6b9a9c75272a74c7e53034/Src-imro/IMROTbl.cpp#L1112
# bool IMROTbl::pnToType( int &type, const QString &pn )
class _PROBE_TYPE(defaultdict[int | str, ProbeType]):
    def __init__(self):
        super().__init__()
        self.update({
            0: PROBE_TYPE_NP1,
            1000: PROBE_TYPE_NP1,
            1020: PROBE_TYPE_NP1020,
            1022: PROBE_TYPE_NP1022,
            1030: PROBE_TYPE_NP1030,
            1032: PROBE_TYPE_NP1032,
            1100: PROBE_TYPE_NP1100,
            1110: PROBE_TYPE_NP1110,
            1120: PROBE_TYPE_NP1120,
            1121: PROBE_TYPE_NP1121,
            1122: PROBE_TYPE_NP1122,
            1123: PROBE_TYPE_NP1123,
            1200: PROBE_TYPE_NP1200,
            1300: PROBE_TYPE_NP1300,
            21: PROBE_TYPE_NP21,
            2000: PROBE_TYPE_NP21,
            2003: PROBE_TYPE_NP2003,
            24: PROBE_TYPE_NP24,
            2010: PROBE_TYPE_NP24,
            2013: PROBE_TYPE_NP2013,
            2020: PROBE_TYPE_NP2020,
            3000: PROBE_TYPE_NP3000,
            3010: PROBE_TYPE_NP3010,
            3020: PROBE_TYPE_NP3020,
        })

    def __missing__(self, key):
        origin_key = key

        if isinstance(key, int):
            key = f'NP{key}'

        # old probe code
        if key.startswith('PRB_1_4') or key.startswith('PRB_1_2'):
            # PRB_1_4_0480_1 (Silicon cap)
            # PRB_1_4_0480_1_C (Metal cap)
            # PRB_1_2_0480_2
            return PROBE_TYPE_NP1
        if key.startswith('PRB2_1'):
            # PRB2_1_2_0640_0 (NP 2.0 SS scrambled el 1280)
            return PROBE_TYPE_NP21
        if key.startswith('PRB2_4'):
            # PRB2_4_2_0640_0 (NP 2.0 MS el 1280)
            return PROBE_TYPE_NP24

        # new probe code
        match key:
            case 'NP1000' | 'NP1001' | 'NP1010' | 'NP1011' | 'NP1012' | 'NP1013' | 'NP1014' | 'NP1015' | 'NP1016' | 'NP1017':
                return PROBE_TYPE_NP1
            case 'NP1020' | 'NP1021':  # NHP phase 2
                return PROBE_TYPE_NP1020
            case 'NP1022':  # NHP phase 2
                return PROBE_TYPE_NP1022
            case 'NP1030' | 'NP1031':  # NHP phase 2
                return PROBE_TYPE_NP1030
            case 'NP1032' | 'NP1033':  # NHP phase 2
                return PROBE_TYPE_NP1032
            case 'NP1100':  # UHD phase 1
                return PROBE_TYPE_NP1100
            case 'NP1110':  # UHD phase 2
                return PROBE_TYPE_NP1110
            case 'NP1120':  # UHD phase 3
                return PROBE_TYPE_NP1120
            case 'NP1121':  # UHD phase 3
                return PROBE_TYPE_NP1121
            case 'NP1122':  # UHD phase 3
                return PROBE_TYPE_NP1122
            case 'NP1123':  # UHD phase 3
                return PROBE_TYPE_NP1123
            case 'NP1200' | 'NP1210' | 'NP1221':  # NHP 128
                return PROBE_TYPE_NP1200
            case 'NP1300':  # Opto
                return PROBE_TYPE_NP1300
            case 'NP2000':  # NP 2.0 SS scrambled el 1280
                return PROBE_TYPE_NP21
            case 'NP2003' | 'NP2004' | 'NP2005' | 'NP2006':  # Neuropixels 2.0 single shank probe
                return PROBE_TYPE_NP2003
            case 'NP2010':  # NP 2.0 MS el 1280
                return PROBE_TYPE_NP24
            case 'NP2013' | 'NP2014':  # Neuropixels 2.0 multishank probe
                return PROBE_TYPE_NP2013
            case 'NP2020' | 'NP2021':  # Neuropixels 2.0 quad base
                return PROBE_TYPE_NP2020
            case 'NP3000':  # Passive NXT probe
                return PROBE_TYPE_NP1200
            case 'NP3010' | 'NP3011':  # NXT single shank
                return PROBE_TYPE_NP3010
            case 'NP3020' | 'NP3021' | 'NP3022':  # NXT multishank
                return PROBE_TYPE_NP3020

        raise KeyError(origin_key)


PROBE_TYPE = _PROBE_TYPE()


class Electrode:
    # positions
    shank: Final[int]
    column: Final[int]
    row: Final[int]

    # properties
    in_used: bool

    # for NP1
    ap_band_gain: int
    lf_band_gain: int
    ap_hp_filter: bool

    # for NP1110
    bank_a: int
    bank_b: int

    __slots__ = 'shank', 'column', 'row', 'in_used', 'ap_band_gain', 'lf_band_gain', 'ap_hp_filter', 'bank_a', 'bank_b'
    __match_args__ = ('shank', 'column', 'row')

    def __init__(self, shank: int, column: int, row: int, in_used: bool | int = True):
        self.shank = int(shank)
        self.column = int(column)
        self.row = int(row)

        if isinstance(in_used, int):
            in_used = in_used != 0

        self.in_used = in_used

    def copy(self, other: Electrode):
        """
        Copy electrode properties

        :param other: copy reference.
        """
        self.in_used = other.in_used

        try:
            self.ap_band_gain = other.ap_band_gain
            self.lf_band_gain = other.lf_band_gain
            self.ap_hp_filter = other.ap_hp_filter
        except AttributeError:
            pass

        try:
            self.bank_a = other.bank_a
            self.bank_b = other.bank_b
        except AttributeError:
            pass

    def __str__(self) -> str:
        return f'Electrode[{self.shank},{self.column},{self.row}]'

    def __repr__(self) -> str:
        return f'Electrode[{self.shank},{self.column},{self.row}]'

    def __eq__(self, other: Any) -> bool:
        try:
            return self.shank == other.shank and \
                self.column == other.column and \
                self.row == other.row
        except AttributeError:
            return False

    def __hash__(self) -> int:
        return hash((self.shank, self.column, self.row))

    def __lt__(self, other: Electrode) -> bool:
        return (self.shank, self.row, self.column) < (other.shank, other.row, other.column)


class ReferenceInfo(NamedTuple):
    code: int
    type: Literal['ext', 'tip', 'bank', 'ground', 'unknown']
    shank: int  # 0 if reference_type is 'ext'
    channel: int  # 0 if reference_type is not 'bank'

    @classmethod
    def max_reference_value(cls, probe_type: ProbeType) -> int:
        """get max possible value for probe type."""
        return probe_type.n_reference

    @classmethod
    def of(cls, probe_type: ProbeType, reference: int) -> Self:
        """
        get information of reference value.

        :param probe_type:
        :param reference:
        :return:
        """
        # https://github.com/billkarsh/SpikeGLX/blob/bc2c10e99e68dcc9ec6b9a9c75272a74c7e53034/Src-imro/IMROTbl.h#L191
        # int IMROTbl_T0base::refTypeAndFields( int &shank, int &bank, int ch )
        # XXX SpikeGLX number of references for some probe types do not match to the description

        if not (0 <= reference < probe_type.n_reference):
            raise ValueError(f'reference id out of boundary for probe type {probe_type.code}: {reference}')

        if reference == 0:
            return ReferenceInfo(0, 'ext', 0, 0)

        # https://github.com/billkarsh/SpikeGLX/blob/bc2c10e99e68dcc9ec6b9a9c75272a74c7e53034/Src-imro/IMROTbl_T2003.cpp#L12
        # https://github.com/billkarsh/SpikeGLX/blob/bc2c10e99e68dcc9ec6b9a9c75272a74c7e53034/Src-imro/IMROTbl_T2013.cpp#L12
        # https://github.com/billkarsh/SpikeGLX/blob/bc2c10e99e68dcc9ec6b9a9c75272a74c7e53034/Src-imro/IMROTbl_T2020.cpp#L279
        # https://github.com/billkarsh/SpikeGLX/blob/bc2c10e99e68dcc9ec6b9a9c75272a74c7e53034/Src-imro/IMROTbl_T3010base.cpp#L247
        # https://github.com/billkarsh/SpikeGLX/blob/bc2c10e99e68dcc9ec6b9a9c75272a74c7e53034/Src-imro/IMROTbl_T3020base.cpp#L369
        if probe_type.code in (2003, 2013, 2020, 3010, 3020):
            if reference == 1:
                return ReferenceInfo(reference, 'ground', 0, 0)
            if reference - 2 < probe_type.n_shank:
                return ReferenceInfo(reference, 'tip', reference - 2, 0)

            raise RuntimeError('')

        # https://github.com/billkarsh/SpikeGLX/blob/bc2c10e99e68dcc9ec6b9a9c75272a74c7e53034/Src-imro/IMROTbl_T21.cpp#L16
        elif probe_type.code == 21:
            n_shank = probe_type.n_shank

            if reference - 1 < n_shank:
                return ReferenceInfo(reference, 'tip', reference - 1, 0)

            references = (127, 507, 887, 1251)
            return ReferenceInfo(reference, 'bank', 0, references[reference - 2])

        # https://github.com/billkarsh/SpikeGLX/blob/bc2c10e99e68dcc9ec6b9a9c75272a74c7e53034/Src-imro/IMROTbl_T24.cpp#L19
        elif probe_type.code == 24:
            n_shank = probe_type.n_shank

            if reference - 1 < n_shank:
                return ReferenceInfo(reference, 'tip', reference - 1, 0)

            references = (127, 511, 895, 1279)
            x = reference - n_shank - 1
            s, i = divmod(x, len(references))
            return ReferenceInfo(reference, 'bank', s, references[i])

        # https://github.com/billkarsh/SpikeGLX/blob/bc2c10e99e68dcc9ec6b9a9c75272a74c7e53034/Src-imro/IMROTbl_T0base.cpp#L239
        else:
            assert probe_type.n_shank == 1

            if reference == 1:
                return ReferenceInfo(reference, 'tip', reference - 1, 0)

            return ReferenceInfo(reference, 'bank', reference - 2, -1)



E = int | tuple[int, int] | tuple[int, int, int] | Electrode
"""single electrode types"""

A = list[int] | NDArray[np.int_]
"""Array-like"""

Es = list[int] | NDArray[np.int_] | list[Electrode]
"""electrode set"""


def _channelmap_len_(e: list[Electrode | None]) -> int:
    """number of channels (C) in e"""
    ret = 0
    for it in e:
        if it is not None:
            ret += 1
    return ret


class ChannelMap:
    """Neuropixels channelmap"""

    __match_args__ = 'probe_type',

    def __init__(self, probe_type: int | str | ProbeType | ChannelMap,
                 electrodes: Sequence[E | None] = None, *,
                 meta: NpxMeta = None):
        """

        :param probe_type: probe type code, ProbeType or a ChannelMap for coping.
        :param electrodes: pre-add electrodes
        """
        if isinstance(probe_type, ChannelMap):
            chmap = cast(ChannelMap, probe_type)
            probe_type = chmap.probe_type
            if electrodes is None:
                electrodes = chmap._electrodes
            if meta is None:
                meta = chmap.meta

        if isinstance(probe_type, (int, str)):
            probe_type = PROBE_TYPE[probe_type]
        elif not isinstance(probe_type, ProbeType):
            raise TypeError(f'{type(probe_type)=}')

        self.probe_type: Final[ProbeType] = probe_type
        self._electrodes: Final[list[Electrode | None]] = [None] * probe_type.n_channels
        self._reference = 0
        self.meta: NpxMeta | None = meta

        # pre compute channels for all electrodes
        ns = probe_type.n_shank
        nc = probe_type.n_col_shank
        nr = probe_type.n_row_shank
        s, c, r = np.mgrid[0:ns, 0:nc, 0:nr]
        channels, _ = e2cb(probe_type, (s.ravel(), c.ravel(), r.ravel()))
        self._channels: Final[NDArray[np.int_]] = channels.reshape((ns, nc, nr))

        if electrodes is not None:
            for e in electrodes:
                if e is not None:
                    t = self.add_electrode(e)
                    if isinstance(e, Electrode):
                        t.copy(e)

    # ========= #
    # load/save #
    # ========= #

    @classmethod
    def parse(cls, source: str) -> Self:
        """
        Parse imro table.

        :param source: an imro table.
        :return:
        """
        from .io import parse_imro
        return parse_imro(source)

    @classmethod
    def from_meta(cls, path: str | Path) -> Self:
        """
        Read imro table from SpikeGLX meta file.

        :param path: file path
        :return:
        """
        from .io import load_meta
        return load_meta(path)

    @classmethod
    def from_imro(cls, path: str | Path) -> Self:
        """
        Read imro file.

        :param path: file path
        :return:
        """
        from .io import load_imro
        return load_imro(path)

    @classmethod
    def from_probe(cls, probe: Probe) -> Self:
        """
        From probeinterface.Probe

        .. note::

            The package ``probeinterface`` is optional dependency.

        :param probe:
        :return:
        """
        from .io import from_probe
        return from_probe(probe)

    def to_imro(self) -> str:
        """format as imro table"""
        from .io import string_imro
        return string_imro(self)

    def save_imro(self, path: str | Path):
        """save into imro file"""
        from .io import save_imro
        save_imro(self, path)

    def to_probe(self) -> Probe:
        """
        to probeinterface.Probe

        .. note::

            The package ``probeinterface`` is optional dependency.

        :return:
        """
        from .io import to_probe
        return to_probe(self)

    def to_numpy(self, unit: Literal['cr', 'xy', 'sxy'] = 'cr') -> NDArray[np.int_]:
        """
        To a numpy array. Empty channels are skipped.

        The layout of returned array is depending on the *unit*

        * ``cr``: column and row, return ``Array[int, N, (S, C, R)]``
        * ``xy``: x and y position, return ``Array[um:int, N, (X, Y)]``
        * ``sxy``: x and y position with shank, return ``Array[int, N, (S, X, Y)]``

        :param unit: electrode ordering kind
        :return:
        """
        from .io import to_numpy
        return to_numpy(self, unit)

    def to_pandas(self) -> pd.DataFrame:
        """
        To a pandas dataframe.

        Use ``-1`` to fill empty channels.

        ::

                     shank  column  row  in_used    x     y
            channel
            0           -1      -1   -1    False   -1    -1
            1            1       1  144     True  282  2160
            ...        ...     ...  ...      ...  ...   ...

        .. note::

            The package ``pandas`` is optional dependency.

        :return: a pandas dataframe
        """
        from .io import to_pandas
        return to_pandas(self)

    def to_polars(self) -> pl.DataFrame:
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

        .. note::

            The package ``polars`` is optional dependency.

        :return: a polars dataframe
        """
        from .io import to_polars
        return to_polars(self)

    def __hash__(self):
        ret = 3 + 5 * self.probe_type.code
        ret = 7 * ret + 11 * self._reference
        for e in self._electrodes:
            ret = 7 * ret + 11 * (0 if e is None else hash(e))
        return ret

    def __eq__(self, other):
        try:
            if self.probe_type.code != other.probe_type.code:
                return False
            if self._reference != other._reference:
                return False
            if len(self._electrodes) != len(other._electrodes):
                return False
            for e, t in zip(self._electrodes, other._electrodes):
                if e != t:
                    return False
            return True
        except AttributeError:
            return False

    def __str__(self) -> str:
        return f'ChannelMap[{self.n_shank},{self.n_col_shank},{self.n_row_shank},{len(self._electrodes)},{len(self)}]'

    __repr__ = __str__

    # ================= #
    # Basic information #
    # ================= #

    def __len__(self) -> int:
        """number of channels"""
        return _channelmap_len_(self._electrodes)

    def __contains__(self, item: E | None) -> bool:
        """Is the electrode *item* in the map?"""
        if item is None:
            for e in self._electrodes:
                if e is None:
                    return True
            else:
                return False

        return self.get_electrode(item) is not None

    @property
    def n_shank(self) -> int:
        """number of shanks"""
        return self.probe_type.n_shank

    @property
    def n_col_shank(self) -> int:
        """number of columns per shank"""
        return self.probe_type.n_col_shank

    @property
    def n_row_shank(self) -> int:
        """number of rows per shank"""
        return self.probe_type.n_row_shank

    @property
    def n_electrode_shank(self) -> int:
        """number of electrodes per shank"""
        return self.probe_type.n_electrode_shank

    @property
    def n_channels(self) -> int:
        """number of total channels"""
        return self.probe_type.n_channels

    @property
    def n_electrode_block(self) -> int:
        """number of electrode blocks"""
        return self.probe_type.n_electrode_block

    @property
    @doc_link()
    def reference(self) -> int:
        """reference type. see {ReferenceInfo} for more information."""
        return self._reference

    @reference.setter
    def reference(self, value: int):
        if not (0 <= value < self.probe_type.n_reference):
            raise ValueError(f'illegal reference value : {value}')

        self._reference = value

    @property
    def reference_info(self) -> ReferenceInfo:
        """ reference information. """
        return ReferenceInfo.of(self.probe_type, self._reference)

    # =================== #
    # channel information #
    # =================== #

    @property
    def channel_shank(self) -> NDArray[np.float64]:
        """

        :return: Array[shank:int|NaN, C]
        """
        return np.array([it.shank if it is not None else np.nan for it in self.channels], dtype=float)

    @property
    def channel_pos_x(self) -> NDArray[np.float64]:
        """

        :return: Array[um:float, C]
        """
        return channel_coordinate(self, electrode_unit='xy')[:, 0]

    @property
    def channel_pos_y(self) -> NDArray[np.float64]:
        """

        :return: Array[um:float, C]
        """
        return channel_coordinate(self, electrode_unit='xy')[:, 1]

    @property
    def channel_pos(self) -> NDArray[np.float64]:
        """

        :return: Array[um:float, C, 2]
        """
        return channel_coordinate(self, electrode_unit='xy')

    def get_channel(self, channel: int) -> Electrode | None:
        """
        Get electrode via channel ID

        :param channel: channel ID.
        :return: found electrode
        """
        return self._electrodes[channel]

    @property
    def channels(self) -> Channels:
        return Channels(self.probe_type, self._electrodes)

    @doc_link()
    def get_electrode(self, electrode: E) -> Electrode | None:
        """
        Get electrode via electrode ID, or position.

        :param electrode: electrode ID, tuple of (shank, electrode), tuple of (shank, column, row) or an {Electrode}.
        :return: found electrodes.
        """
        match electrode:
            case int(electrode):
                shank = 0
                column, row = e2cr(self.probe_type, electrode)
            case (int(shank), int(column), int(row)):
                pass
            case Electrode(shank=shank, column=column, row=row):
                pass
            case electrode if all_int(electrode):
                shank = 0
                column, row = e2cr(self.probe_type, electrode)
            case (shank, electrode) if all_int(shank, electrode):
                column, row = e2cr(self.probe_type, electrode)
            case (shank, column, row) if all_int(shank, column, row):
                pass
            case _:
                raise TypeError()

        for e in self._electrodes:
            if e is not None and e.shank == shank and e.column == column and e.row == row:
                return e
        return None

    @property
    def electrodes(self) -> Electrodes:
        return Electrodes(self.probe_type, self._electrodes)

    def disconnect_channels(self) -> list[int]:
        """
        list of channel that it is not in used, or it is disconnected.

        ``None`` channels are not included in the return list.

        :return: list of channel numbers.
        """
        return [i for i, e in enumerate(self._electrodes) if e is not None and not e.in_used]

    # ==================== #
    # add/delete electrode #
    # ==================== #

    @doc_link()
    def add_electrode(self, electrode: E,
                      in_used: bool = True,
                      exist_ok: bool = False) -> Electrode:
        """
        Add an electrode into this channelmap.

        :param electrode: electrode ID, tuple of (shank, electrode), tuple of (shank, column, row), or an {Electrode}
        :param in_used: Is it used?
        :param exist_ok: if not exist_ok, an error will raise if electrode has existed.
        :return: a correspond electrode.
        :raise ValueError: electrode position out of range
        :raise ChannelHasUsedError: channel has been used by other electrode in this channelmap.
        """
        match electrode:
            case electrode if all_int(electrode):
                shank = 0
                column, row = e2cr(self.probe_type, int(electrode))
            case (shank, electrode) if all_int(shank, electrode):
                shank = int(shank)
                column, row = e2cr(self.probe_type, int(electrode))
            case (shank, column, row) if all_int(shank, column, row):
                shank = int(shank)
                column = int(column)
                row = int(row)
            case Electrode(shank=shank, column=column, row=row):
                pass
            case _:
                raise TypeError(repr(electrode))

        probe_type = self.probe_type

        if not (0 <= shank < probe_type.n_shank):
            raise ValueError(f'shank value out of range : {shank}')

        if not (0 <= column < probe_type.n_col_shank):
            raise ValueError(f'column value out of range : {column}')

        if not (0 <= row < probe_type.n_row_shank):
            raise ValueError(f'row value out of range : {row}')

        c = int(self._channels[shank, column, row])
        if (t := self._electrodes[c]) is not None:
            if t.shank == shank and t.column == column and t.row == row:
                if exist_ok:
                    return t
                else:
                    raise ChannelHasUsedError(t)
            else:
                raise ChannelHasUsedError(t)
        else:
            self._electrodes[c] = e = Electrode(shank, column, row, in_used)
            return e

    @doc_link()
    def del_electrode(self, electrode: E) -> Electrode | None:
        """
        Remove an electrode from this channelmap.

        :param electrode: electrode ID, tuple of (shank, electrode), tuple of (shank, column, row), or an {Electrode}
        :return: removed electrodes
        """
        match electrode:
            case electrode if all_int(electrode):
                shank = 0
                column, row = e2cr(self.probe_type, electrode)
            case (shank, electrode) if all_int(shank, electrode):
                column, row = e2cr(self.probe_type, electrode)
            case (shank, column, row) if all_int(shank, column, row):
                pass
            case Electrode(shank=shank, column=column, row=row):
                pass
            case _:
                raise TypeError(repr(electrode))

        for c, e in enumerate(self._electrodes):
            if e is not None and e.shank == shank and e.column == column and e.row == row:
                self._electrodes[c] = None
                return e
        return None


class ChannelHasUsedError(RuntimeError):
    """Error of a map contains two electrodes with same channel."""

    def __init__(self, electrode: Electrode):
        """

        :param electrode: channel is occupied by this electrode.
        """
        super().__init__(str(electrode))
        self.electrode: Final = electrode


class Channels(Sized, Iterable[Electrode | None]):
    """Dict-like accessor for navigating channels via channel ID."""

    __slots__ = '_probe_type', '_electrodes'

    def __init__(self, probe_type: ProbeType, electrode: list[Electrode | None]):
        self._probe_type: Final = probe_type
        self._electrodes: Final = electrode

    def __len__(self):
        """number of total channels"""
        return self._probe_type.n_channels

    def __contains__(self, item: int | None) -> bool:
        """
        Is the channel *item* in the map?

        :param item: channel ID, or ``None``.
        :return: ``True`` if channel is set (when *item* is ``int``), or
            ``True`` if there are at least one channel is unbound (when *item* is ``None``).
        """
        if item is None:
            for e in self._electrodes:
                if e is None:
                    return True
            else:
                return False

        return self._electrodes[item] is not None

    def __getitem__(self, item):
        """
        Get electrode via channel ID.

        :param item: channel id, id-slicing, or id-array
        :return: electrode/s
        """
        if all_int(item):
            return self._electrodes[int(item)]
        elif isinstance(item, slice):
            return [self._electrodes[it] for it in range(len(self._electrodes))[item]]
        else:
            return [self._electrodes[int(it)] for it in np.arange(len(self._electrodes))[item]]

    def __setitem__(self, item, value: Electrode):
        """
        Copy electrode properties.

        :param item: channel id, id-slicing, or id-array
        :param value: copy reference
        """
        if all_int(item):
            if (e := self._electrodes[int(item)]) is not None:
                e.copy(value)
        elif isinstance(item, slice):
            for it in range(len(self._electrodes))[item]:
                if (e := self._electrodes[it]) is not None:
                    e.copy(value)
        else:
            for it in np.arange(len(self._electrodes))[item]:
                if (e := self._electrodes[int(it)]) is not None:
                    e.copy(value)

    def __delitem__(self, item):
        """
        Remove electrodes via channel IDs.

        :param item: channel id, id-slicing, or id-array
        """
        if all_int(item):
            self._electrodes[int(item)] = None
        elif isinstance(item, slice):
            for it in range(len(self._electrodes))[item]:
                self._electrodes[it] = None
        else:
            for it in np.arange(len(self._electrodes))[item]:
                self._electrodes[int(it)] = None

    def __iter__(self) -> Iterator[Electrode | None]:
        """iterating over channels, index imply its channel ID."""
        for e in self._electrodes:
            yield e


class Electrodes(Sized, Iterable[Electrode]):
    """Dict-like accessor for navigating channels via electrode position (shank, column, row)"""

    __slots__ = '_probe_type', '_electrodes'

    def __init__(self, probe_type: ProbeType, electrode: list[Electrode | None]):
        self._probe_type: Final = probe_type
        self._electrodes: Final = electrode

    def __len__(self):
        """number of channels (C)"""
        return _channelmap_len_(self._electrodes)

    def __contains__(self, item) -> bool:
        """
        Is the electrode at position *item* existed?

        :param item: tuple of (shank:I, column:I, row:I), where type I = `None | int | slice | tuple (union) | Iterable[int]`
        :return: True when any electrodes found.
        """
        match self[item]:
            case None | []:
                return False
            case _:
                return True

    def __getitem__(self, item):
        """
        Get electrode via electrode positions.

        :param item: tuple of (shank:I, column:I, row:I), where type I = `None | int | slice | tuple (union) | Iterable[int]`
        :return: found electrode/s
        """
        shank, cols, rows = item
        match item:
            case (None, None, None):
                return [e for e in self._electrodes if e is not None]
            case (shank, column, row) if all_int(shank, column, row):
                for e in self._electrodes:
                    if e is not None and e.shank == shank and e.column == column and e.row == row:
                        return e
                return None
            case (_, _, _):
                shank = as_set(shank, self._probe_type.n_shank)
                cols = as_set(cols, self._probe_type.n_col_shank)
                rows = as_set(rows, self._probe_type.n_row_shank)
                ret = []
                for e in self._electrodes:
                    if e is not None and e.shank in shank and e.column in cols and e.row in rows:
                        ret.append(e)
                return ret
            case _:
                raise TypeError(repr(item))

    def __setitem__(self, item, value: Electrode):
        """
        Copy electrode properties.

        :param item: tuple of (shank:I, column:I, row:I), where type I = `None | int | slice | tuple (union) | Iterable[int]`
        :param value: copy reference
        """
        shank, cols, rows = item
        match item:
            case (None, None, None):
                for e in self._electrodes:
                    if e is not None:
                        e.copy(value)
                return [e for e in self._electrodes if e is not None]
            case (shank, column, row) if all_int(shank, column, row):
                for e in self._electrodes:
                    if e is not None and e.shank == shank and e.column == column and e.row == row:
                        e.copy(value)
                return None
            case (_, _, _):
                shank = as_set(shank, self._probe_type.n_shank)
                cols = as_set(cols, self._probe_type.n_col_shank)
                rows = as_set(rows, self._probe_type.n_row_shank)
                for e in self._electrodes:
                    if e is not None and e.shank in shank and e.column in cols and e.row in rows:
                        e.copy(value)
            case _:
                raise TypeError(repr(item))

    def __delitem__(self, item):
        """
        Remove electrodes via electrode position.

        :param item: tuple of (shank:I, column:I, row:I), where type I = `None | int | slice | tuple (union) | Iterable[int]`
        """
        shank, cols, rows = item
        match item:
            case (None, None, None):
                for c in range(len(self._electrodes)):
                    self._electrodes[c] = None
            case (shank, column, row) if all_int(shank, column, row):
                for c in range(len(self._electrodes)):
                    e = self._electrodes[c]
                    if e is not None and e.shank == shank and e.column == column and e.row == row:
                        self._electrodes[c] = e
                        break
            case (_, _, _):
                shank = as_set(shank, self._probe_type.n_shank)
                cols = as_set(cols, self._probe_type.n_col_shank)
                rows = as_set(rows, self._probe_type.n_row_shank)
                for c in range(len(self._electrodes)):
                    e = self._electrodes[c]
                    if e is not None and e.shank in shank and e.column in cols and e.row in rows:
                        self._electrodes[c] = None
            case _:
                raise TypeError(repr(item))

    def __iter__(self) -> Iterator[Electrode]:
        """iterating over channels"""
        for e in self._electrodes:
            if e is not None:
                yield e


def channel_coordinate(shank_map: ChannelMap,
                       electrode_unit: ELECTRODE_UNIT = 'cr',
                       include_unused=False) -> NDArray[np.float64]:
    """
    Get coordinate of all channels.

    :param shank_map:
    :param electrode_unit: 'xy'=(X,Y), 'cr'=(S,C,R)
    :param include_unused: including disconnected channels
    :return: Array[um:float, E, (S, C, R)|(X, Y)]. NaN if electrode is missing.
    """
    if electrode_unit not in ('cr', 'xy'):
        raise ValueError(f'unsupported electrode unit : {electrode_unit}')

    probe_type = shank_map.probe_type

    s = []
    r = []
    c = []

    for e in shank_map.channels:
        if e is not None and (include_unused or e.in_used):
            s.append(e.shank)
            c.append(e.column)
            r.append(e.row)
        else:
            s.append(np.nan)
            c.append(np.nan)
            r.append(np.nan)

    if electrode_unit == 'cr':
        return np.column_stack([s, c, r])
    else:
        s = np.array(s)
        x = np.array(c) * probe_type.c_space + s * probe_type.s_space
        y = np.array(r) * probe_type.r_space

        return np.column_stack([x, y])


def electrode_coordinate(probe_type: int | str | ChannelMap | ProbeType,
                         electrode_unit: ELECTRODE_UNIT = 'cr') -> NDArray[np.int_]:
    """
     Get coordinate of all electrodes.

    :param probe_type:
    :param electrode_unit: 'xy'=(X,Y), 'cr'=(S,C,R)
    :return: Array[um:int, E, (S, C, R)|(X, Y)]
    """
    if electrode_unit not in ('cr', 'xy'):
        raise ValueError(f'unsupported electrode unit : {electrode_unit}')

    match probe_type:
        case ChannelMap(probe_type=probe_type):
            pass
        case ProbeType():
            pass
        case str() | int():
            probe_type = PROBE_TYPE[probe_type]
        case _:
            raise TypeError(repr(probe_type))

    y = np.arange(probe_type.n_row_shank)
    x = np.arange(probe_type.n_col_shank)
    s = np.arange(probe_type.n_shank)

    if electrode_unit == 'xy':
        y *= probe_type.r_space
        x *= probe_type.c_space
        s *= probe_type.s_space

    j, i = np.mgrid[0:len(y), 0:len(x)]
    e = np.column_stack([
        x[i].ravel(), y[j].ravel()
    ])

    if electrode_unit == 'cr':
        return np.vstack([
            np.hstack([
                np.full((e.shape[0], 1), ss), e
            ])
            for ss in s
        ])
    else:
        return np.vstack([
            np.column_stack([
                e[:, 0] + ss, e[:, 1]
            ])
            for ss in s
        ])




@overload
def e2p(probe_type: ProbeType, e: E) -> tuple[float, float]:
    pass


@overload
def e2p(probe_type: ProbeType, e: Es) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    pass


def e2p(probe_type: ProbeType, e):
    match e:
        case e if all_int(e):
            s = 0
            c, r = e2cr(probe_type, e)
        case (s, e) if all_int(s, e):
            s = int(s)
            c, r = e2cr(probe_type, e)
        case (s, c, r) if all_int(s, c, r):
            s = int(s)
            c = int(c)
            r = int(r)
        case Electrode(shank=s, column=c, row=r):
            pass
        case [Electrode(), *_]:
            s = np.array([it.shank for it in e])
            c = np.array([it.column for it in e])
            r = np.array([it.row for it in e])
        case e if isinstance(e, np.ndarray):
            if e.ndim != 1:
                raise ValueError()
            s, e = align_arr(0, e)
            c, r = e2cr(probe_type, e)
        case (s, c, r):
            s, c, r = align_arr(s, c, r)
        case [*_]:
            s, e = align_arr(0, np.array(e))
            c, r = e2cr(probe_type, e)
        case _:
            raise TypeError(repr(e))

    x = probe_type.s_space * s + probe_type.c_space * c
    y = probe_type.r_space * r
    return x, y


@overload
def e2cr(probe_type: ProbeType, e: E) -> tuple[int, int]:
    pass


@overload
def e2cr(probe_type: ProbeType, e: Es) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
    pass


def e2cr(probe_type: ProbeType, e):
    """

    :param probe_type:
    :param e: electrode number
    :return: (column, row)
    """
    match e:
        case e if all_int(e):
            e = int(e)
        case (s, e) if all_int(s, e):
            e = int(e)
        case (s, c, r) if all_int(s, c, r):
            return int(c), int(r)
        case Electrode(column=c, row=r):
            return c, r
        case [Electrode(), *_]:
            c = np.array([it.column for it in e])
            r = np.array([it.row for it in e])
            return c, r
        case e if isinstance(e, np.ndarray):
            if e.ndim != 1:
                raise ValueError()
        case [*_]:
            e = np.array(e)
        case _:
            raise TypeError(repr(e))

    return ImroEC(probe_type).e2cr(e)


@overload
def cr2e(probe_type: ProbeType, p: E) -> int:
    pass


@overload
def cr2e(probe_type: ProbeType, p: Es | tuple[int | A, int | A]) -> NDArray[np.int_]:
    pass


def cr2e(probe_type: ProbeType, p):
    """

    :param probe_type:
    :param p:
    :return:
    """
    match p:
        case int(e):
            return e
        case (int(c), int(r)):
            s = None
        case (int(s), int(c), int(r)):
            pass
        case e if all_int(e):
            return int(e)
        case (c, r) if all_int(c, r):
            c = int(c)
            r = int(r)
            s = 0
        case (s, c, r) if all_int(s, c, r):
            c = int(c)
            r = int(r)
            s = int(s)
        case Electrode(shank=s, column=c, row=r):
            pass
        case (c, r):
            c, r = align_arr(c, r)
            s = None
        case [Electrode(), *_]:
            c = np.array([it.column for it in p])
            r = np.array([it.row for it in p])
            s = np.array([it.shank for it in p])
        case _ if isinstance(p, np.ndarray):
            match p.shape:
                case (_, ):
                    return p
                case (_, 2):
                    c = p[:, 0]
                    r = p[:, 1]
                    s = None
                case _:
                    raise ValueError()
        case [*_]:
            return np.array(p)
        case _:
            raise TypeError(repr(p))

    return ImroEC(probe_type).cr2e(c, r, s)


@overload
def e2c(probe_type: ProbeType, electrode: E) -> int:
    pass


@overload
def e2c(probe_type: ProbeType, electrode: Es | tuple[int | A, A] | tuple[int | A, A, A]) -> NDArray[np.int_]:
    pass


def e2c(probe_type: ProbeType, electrode):
    return e2cb(probe_type, electrode)[0]


@overload
def e2cb(probe_type: ProbeType, electrode: E) -> tuple[int, int]:
    pass


@overload
def e2cb(probe_type: ProbeType, electrode: Es | tuple[int | A, A] | tuple[int | A, A, A]) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
    pass


def e2cb(probe_type: ProbeType, electrode):
    match electrode:
        case int(electrode):
            shank = None
        case (int(shank), int(electrode)):
            pass
        case (int(shank), int(column), int(row)):
            electrode = cr2e(probe_type, (column, row))
        case electrode if all_int(electrode):
            shank = None
            electrode = int(electrode)
        case (shank, electrode) if all_int(shank, electrode):
            shank = int(shank)
            electrode = int(electrode)
        case (shank, column, row) if all_int(shank, column, row):
            shank = int(shank)
            electrode = cr2e(probe_type, (column, row))
        case Electrode(shank=shank, column=column, row=row):
            electrode = cr2e(probe_type, (column, row))
        case [Electrode(), *_]:
            shank = np.array([it.shank for it in electrode])
            electrode = np.array([it.electrode for it in electrode])
        case (shank, electrode):
            shank, electrode = align_arr(shank, electrode)
        case (shank, column, row):
            electrode = cr2e(probe_type, (column, row))
            shank, electrode = align_arr(shank, electrode)
        case _ if isinstance(electrode, np.ndarray):
            shank = np.zeros_like(electrode, dtype=int)
        case [*_]:
            shank, electrode = align_arr(0, np.array(electrode))
        case _:
            raise TypeError()

    return ImroEC(probe_type).e2c(electrode, shank)


@overload
def c2e(probe_type: ProbeType, channel: int, bank: int = None, shank: int = None) -> int:
    pass


@overload
def c2e(probe_type: ProbeType, channel: A, bank: int | A = None, shank: int | A = None) -> NDArray[np.int_]:
    pass


def c2e(probe_type: ProbeType, channel, bank=None, shank=None):
    match (bank, channel):
        case (None, channel) if all_int(channel):
            n = probe_type.n_channels
            bank, channel = divmod(int(channel), n)
        case (None, channel):
            n = probe_type.n_channels
            bank, channel = divmod(np.asarray(channel), n)
        case (bank, channel) if all_int(bank, channel):
            bank = int(bank)
            channel = int(channel)
        case (bank, channel):
            bank, channel = align_arr(bank, channel)
        case _:
            raise TypeError()

    if all_int(bank, channel) and shank is None:
        bank = int(bank)
        channel = int(channel)
    elif all_int(bank, channel, shank):
        bank = int(bank)
        channel = int(channel)
        shank = int(shank)
    elif shank is None:
        bank, channel = align_arr(bank, channel)
    else:
        bank, channel, shank = align_arr(bank, channel, shank)

    return ImroEC(probe_type).c2e(channel, bank, shank)


class ImroEC:
    def __new__(cls, probe_type: ProbeType):
        match probe_type.code:
            case 0:
                ret = ImroEC_NP1
            case 21 | 2003:
                ret = ImroEC_NP21
            case 24 | 2013:
                ret = ImroEC_NP24
            case 1110:
                ret = ImroEC_NP1110
            case 2020:
                ret = ImroEC_NP2020
            case 3010:
                ret = ImroEC_NP3010
            case 3020:
                ret = ImroEC_NP3020
            case _:
                ret = ImroEC_NP1

        return object.__new__(ret)

    def __init__(self, probe_type: ProbeType):
        self.probe_type: ProbeType = probe_type

    @overload
    def e2cr(self, e: int) -> tuple[int, int]:
        pass

    @overload
    def e2cr(self, e: NDArray[np.int_]) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
        pass

    def e2cr(self, e):
        """

        :param e: electrode
        :return: tuple of (col, row)
        """
        # https://github.com/billkarsh/SpikeGLX/blob/bc2c10e99e68dcc9ec6b9a9c75272a74c7e53034/Src-imro/IMROTbl_T0base.cpp#L199
        r, c = divmod(e, self.probe_type.n_col_shank)
        return c, r

    @overload
    def cr2e(self, c: int, r: int, s: int = None) -> int:
        pass

    @overload
    def cr2e(self, c: NDArray[np.int_], r: NDArray[np.int_], s: NDArray[np.int_] = None) -> NDArray[np.int_]:
        pass

    def cr2e(self, c, r, s=None):
        """

        :param c: column
        :param r: row
        :param s: shank
        :return: electrode
        """
        return r * self.probe_type.n_col_shank + c

    @overload
    def e2c(self, e: int, s: int = None) -> tuple[int, int]:
        pass

    @overload
    def e2c(self, e: NDArray[np.int_], s: NDArray[np.int_] = None) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
        pass

    def e2c(self, e, s=None):
        """

        :param e: electrode
        :param s: shank
        :return: tuple of (channel, bank)
        """
        raise NotImplementedError()

    @overload
    def c2e(self, c: int, b: int, s: int = None) -> int:
        pass

    @overload
    def c2e(self, c: NDArray[np.int_], b: NDArray[np.int_], s: NDArray[np.int_] = None) -> NDArray[np.int_]:
        pass

    def c2e(self, c, b, s=None):
        """

        :param c: channel
        :param b: bank
        :param s: shank
        :return: electrode
        """
        raise NotImplementedError()


class ImroEC_NP1(ImroEC):
    def c2e(self, c, b, s=None):
        # https://github.com/billkarsh/SpikeGLX/blob/bc2c10e99e68dcc9ec6b9a9c75272a74c7e53034/Src-imro/IMROTbl_T0base.cpp#L12
        return b * 384 + c % 384

    def e2c(self, e, s=None):
        bank, channel = divmod(e, 384)
        return channel, bank


class ImroEC_NP21(ImroEC):
    BF = np.array([1, 7, 5, 3], dtype=int)
    BA = np.array([0, 4, 8, 12], dtype=int)

    def c2e(self, c, b, s=None):
        # https://github.com/billkarsh/SpikeGLX/blob/bc2c10e99e68dcc9ec6b9a9c75272a74c7e53034/Src-imro/IMROTbl_T21base.cpp#L34
        block, index = divmod(c, 32)
        row, col = divmod(index, 2)
        rows = np.arange(0, 16)

        if isinstance(b, int):
            mat = (rows * self.BF[b] + col * self.BA[b]) % 16
            row = np.nonzero(mat == row)[0][0]
        else:
            mat = (np.multiply.outer(rows, self.BF[b]) + col * self.BA[b]) % 16  # (16, E)
            row, i = np.nonzero(mat == row)
            row = row[np.argsort(i)]

        return b * 384 + block * 32 + row * 2 + col

    def e2c(self, e, s=None):
        bank, e = divmod(e, 384)
        block, index = divmod(e, 32)
        row, col = divmod(index, 2)
        channel = 2 * ((row * self.BF[bank] + col * self.BA[bank]) % 16) + 32 * block + col
        return channel, bank


class ImroEC_NP24(ImroEC):
    ELECTRODE_MAP_24 = np.array([
        [0, 2, 4, 6, 5, 7, 1, 3],  # shank-0
        [1, 3, 5, 7, 4, 6, 0, 2],  # shank-1
        [4, 6, 0, 2, 1, 3, 5, 7],  # shank-2
        [5, 7, 1, 3, 0, 2, 4, 6],  # shank-3
    ], dtype=int)

    def c2e(self, c, b, s=None):
        # https://github.com/billkarsh/SpikeGLX/blob/bc2c10e99e68dcc9ec6b9a9c75272a74c7e53034/Src-imro/IMROTbl_T24base.cpp#L26
        if s is None:
            if isinstance(c, int):
                s = 0
            else:
                s = np.zeros_like(c)

        block, index = divmod(c, 48)

        if isinstance(s, int):
            block = np.nonzero(self.ELECTRODE_MAP_24[s] == block)[0][0]
        else:
            block, i = np.nonzero(self.ELECTRODE_MAP_24[s].T == block)
            block = block[np.argsort(i)]

        return 384 * b + 48 * block + index

    def e2c(self, e, s=None):
        if s is None:
            if isinstance(e, int):
                s = 0
            else:
                s = np.zeros_like(e)

        bank, e = divmod(e, 384)
        block, index = divmod(e, 48)
        block = self.ELECTRODE_MAP_24[s, block]
        return 48 * block + index, bank


class ImroEC_NP1110(ImroEC):
    ELECTRODE_MAP_1110 = np.array([
        [0, 3, 1, 2],
        [1, 2, 0, 3]
    ], dtype=int)

    @classmethod
    def group(cls, c):
        # https://github.com/billkarsh/SpikeGLX/blob/bc2c10e99e68dcc9ec6b9a9c75272a74c7e53034/Src-imro/IMROTbl_T1110.cpp#L298
        c = c % 384
        return 2 * (c // 32) + (c % 2)

    @classmethod
    def row(cls, c, b, g=None):
        # https://github.com/billkarsh/SpikeGLX/blob/bc2c10e99e68dcc9ec6b9a9c75272a74c7e53034/Src-imro/IMROTbl_T1110.cpp#L318
        if g is None:
            g = cls.group(c)

        group_row = g // 4
        in_group_row = ((c % 64) % 32) // 4
        if isinstance(c, int):
            bank_row = 8 * group_row + (in_group_row if c % 2 == 0 else 7 - in_group_row)
        else:
            bank_row = 8 * group_row + np.where(c % 2 == 0, in_group_row, 7 - in_group_row)

        return 48 * b + bank_row

    @classmethod
    def col(cls, c, b, g=None):
        # https://github.com/billkarsh/SpikeGLX/blob/bc2c10e99e68dcc9ec6b9a9c75272a74c7e53034/Src-imro/IMROTbl_T1110.cpp#L306
        if g is None:
            g = cls.group(c)

        group_col = cls.ELECTRODE_MAP_1110[b % 2, g % 4]
        crossed = (b // 4) % 2
        in_group_col = (((c % 64) % 32) // 2) % 2
        in_group_col = in_group_col ^ crossed
        if isinstance(c, int):
            col = 2 * group_col + (in_group_col if c % 2 == 0 else 1 - in_group_col)
        else:
            col = 2 * group_col + np.where(c % 2 == 0, in_group_col, 1 - in_group_col)

        return col

    def c2e(self, c, b, s=None):
        # https://github.com/billkarsh/SpikeGLX/blob/bc2c10e99e68dcc9ec6b9a9c75272a74c7e53034/Src-imro/IMROTbl_T1110.cpp#L349
        g = self.group(c)
        r = self.row(c, b, g)
        i = self.col(c, b, g)
        return 8 * r + i

    _C2E_CACHE = None

    @classmethod
    def c2e_electrode_array(cls) -> NDArray[np.int_]:
        if (e := cls._C2E_CACHE) is not None:
            return e

        t = PROBE_TYPE_NP1110
        e = []
        assert t.n_electrode_shank // t.n_channels == 16
        bank = 16  # t.n_electrode_shank // t.n_channels

        b = np.zeros((t.n_channels,), dtype=np.int_)
        c = np.arange(t.n_channels)
        g = cls.group(c)
        for _ in range(bank):
            r = cls.row(c, b, g)
            i = cls.col(c, b, g)
            e.append(8 * r + i)
            b += 1

        e = cls._C2E_CACHE = np.concatenate(e)
        return e

    def e2cr(self, e):
        if isinstance(e, int):
            i = np.nonzero(self.c2e_electrode_array() == e)[0][0]
        else:
            ii = np.nonzero(self.c2e_electrode_array() == e[:, np.newaxis])
            i = np.full_like(e, -1)
            i[ii[0]] = ii[1]
            if np.any(i < 0):
                raise ValueError('electrode out of bound')

        r, c = divmod(i, 8)
        return c, r

    def e2c(self, e, s=None):
        if isinstance(e, int):
            i = np.nonzero(self.c2e_electrode_array() == e)[0][0]
        else:
            ii = np.nonzero(self.c2e_electrode_array() == e[:, np.newaxis])
            i = np.full_like(e, -1)
            i[ii[0]] = ii[1]
            if np.any(i < 0):
                raise ValueError('electrode out of bound')

        b, c = divmod(i, 384)
        return c, b


class ImroEC_NP2020(ImroEC):
    def e2c(self, e, s=None):
        if s is None:
            if isinstance(e, int):
                s = 0
            else:
                s = np.zeros_like(e)

        bank, _ = divmod(e, 384)
        return s * 384 + e, bank

    def c2e(self, c, b, s=None):
        return c % 384 + b * 384


class ImroEC_NP3010(ImroEC):
    def c2e(self, c, b, s=None):
        # https://github.com/billkarsh/SpikeGLX/blob/bc2c10e99e68dcc9ec6b9a9c75272a74c7e53034/Src-imro/IMROTbl_T3010base.cpp#L12
        return b * 912 + c

    def e2c(self, e, s=None):
        b, c = divmod(e, 912)
        return c, b


class ImroEC_NP3020(ImroEC):
    ELECTRODE_MAP_3020 = np.array([  # 4 shanks X 32 blocks, 99 = forbidden
        [0, 16, 1, 17, 2, 18, 3, 99, 4, 99, 5, 99, 6, 99, 7, 99, 8, 99, 9, 99, 10, 99, 11, 99, 12, 99, 13, 99, 14, 99, 15, 99],  # shank-0
        [16, 0, 17, 1, 18, 2, 99, 3, 99, 4, 99, 5, 99, 6, 99, 7, 99, 8, 99, 9, 99, 10, 99, 11, 99, 12, 99, 13, 99, 14, 99, 15],  # shank-1
        [8, 99, 9, 99, 10, 99, 11, 99, 12, 99, 13, 99, 14, 99, 15, 99, 0, 16, 1, 17, 2, 18, 3, 99, 4, 99, 5, 99, 6, 99, 7, 99],  # shank-2
        [99, 8, 99, 9, 99, 10, 99, 11, 99, 12, 99, 13, 99, 14, 99, 15, 16, 0, 17, 1, 18, 2, 99, 3, 99, 4, 99, 5, 99, 6, 99, 7],  # shank-3
    ], dtype=int)

    def c2e(self, c, b, s=None):
        # https://github.com/billkarsh/SpikeGLX/blob/bc2c10e99e68dcc9ec6b9a9c75272a74c7e53034/Src-imro/IMROTbl_T3020base.cpp#L26
        if s is None:
            if isinstance(c, int):
                s = 0
            else:
                s = np.zeros_like(c)

        block, index = divmod(c, 48)

        block = self.ELECTRODE_MAP_3020[s, block]
        if np.any(block == 99):
            raise ValueError('illegal channel')

        return 912 * b + 48 * block + index

    def e2c(self, e, s=None):
        if s is None:
            if isinstance(e, int):
                s = 0
            else:
                s = np.zeros_like(e)

        bank, e = divmod(e, 912)
        block, index = divmod(e, 48)

        if isinstance(s, int):
            block = np.nonzero(self.ELECTRODE_MAP_3020[s] == block)[0][0]
        else:
            block, i = np.nonzero(self.ELECTRODE_MAP_3020[s].T == block)
            block = block[np.argsort(i)]

        return block * 48 + index, bank
