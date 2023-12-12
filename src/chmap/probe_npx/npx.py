from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import NamedTuple, Final, Literal, overload, Sized, cast, Any

import numpy as np
from numpy.typing import NDArray

__all__ = [
    'ProbeType',
    'PROBE_TYPE',
    'PROBE_TYPE_NP1',
    'PROBE_TYPE_NP21',
    'PROBE_TYPE_NP24',
    'Electrode',
    'ChannelMap',
    'channel_coordinate',
    'electrode_coordinate',
]


class ProbeType(NamedTuple):
    """Probe profile.

    References
    ----------

    https://github.com/open-ephys-plugins/neuropixels-pxi/blob/master/Source/Probes/Geometry.cpp#L27
    https://github.com/jenniferColonell/SGLXMetaToCoords/blob/140452d43a55ea7c7904f09e03858bfe0d499df3/SGLXMetaToCoords.py#L79

    """
    code: int
    n_shank: int  # number of shank
    n_col_shank: int  # number of columns per shank
    n_row_shank: int  # number of rows per shank
    n_electrode_shank: int  # number of electrode per shank. It is equals to `n_col_shank * n_row_shank`.
    n_channels: int  # number of total channels.
    n_electrode_block: int  # number of electrode per block.
    c_space: int  # electrodes column space, um
    r_space: int  # electrodes row space, um
    s_space: int  # shank space, um
    reference: tuple[int, ...]


PROBE_TYPE_NP1 = ProbeType(0, 1, 2, 480, 960, 384, 32, 32, 20, 0, (192, 576, 960))
# PROBE_TYPE_NHP1 = ProbeProfile(1, 2, 64, 128, 128, 32)
# PROBE_TYPE_NHP2_10 = ProbeProfile(1, 2, 480, 960, 384, 32)
# PROBE_TYPE_NHP2_25 = ProbeProfile(1, 2, 1248, 2496, 384, 32)
# PROBE_TYPE_NHP2_45 = ProbeProfile(1, 2, 2208, 4416, 384, 32)
PROBE_TYPE_NP21 = ProbeType(21, 1, 2, 640, 1280, 384, 48, 32, 15, 0, (127, 507, 887, 1251))
PROBE_TYPE_NP24 = ProbeType(24, 4, 2, 640, 1280, 384, 48, 32, 15, 250, (127, 511, 895, 1279))
# PROBE_TYPE_UHD = ProbeProfile(1, 8, 48, 384, 384, 48)


PROBE_TYPE = {
    # probe_type: (n_shank, n_col_shank, n_row_shank)
    0: PROBE_TYPE_NP1,
    # 'NP1': PROBE_TYPE_NP1,
    # 'PRB_1_4_0480_1': PROBE_TYPE_NP1,
    # 'PRB_1_4_0480_1_C': PROBE_TYPE_NP1,
    # 'PRB_1_2_0480_2': PROBE_TYPE_NP1,
    # 'NHP1': PROBE_TYPE_NHP1,
    # 'NP1200': PROBE_TYPE_NHP1,
    # 'NP1210': PROBE_TYPE_NHP1,
    # 'NHP10': PROBE_TYPE_NHP2_10,
    # 'NP1010': PROBE_TYPE_NHP2_10,
    # 'NHP25': PROBE_TYPE_NHP2_25,
    # 'NP1020': PROBE_TYPE_NHP2_25,
    # 'NP1021': PROBE_TYPE_NHP2_25,
    # 'NHP45': PROBE_TYPE_NHP2_45,
    # 'NP1030': PROBE_TYPE_NHP2_45,
    # 'NP1031': PROBE_TYPE_NHP2_45,
    # 'UHD1': PROBE_TYPE_UHD,
    # 'NP1100': PROBE_TYPE_UHD,
    # 'UHD2': PROBE_TYPE_UHD,
    # 'NP1110': PROBE_TYPE_UHD,
    #
    21: PROBE_TYPE_NP21,
    '21': PROBE_TYPE_NP21,
    'NP2_1': PROBE_TYPE_NP21,
    'PRB2_1_2_0640_0': PROBE_TYPE_NP21,
    'PRB2_1_4_0480_1': PROBE_TYPE_NP21,
    'NP2000': PROBE_TYPE_NP21,
    'NP2003': PROBE_TYPE_NP21,
    'NP2004': PROBE_TYPE_NP21,
    #
    24: PROBE_TYPE_NP24,
    '24': PROBE_TYPE_NP24,
    'NP2_4': PROBE_TYPE_NP24,
    'PRB2_4_2_0640_0': PROBE_TYPE_NP24,
    'NP2010': PROBE_TYPE_NP24,
    'NP2013': PROBE_TYPE_NP24,
    'NP2014': PROBE_TYPE_NP24,
}


class Electrode:
    shank: Final[int]
    column: Final[int]
    row: Final[int]

    in_used: bool
    ap_band_gain: int
    lf_band_gain: int
    ap_hp_filter: bool

    __match_args__ = ('shank', 'column', 'row')

    def __init__(self, shank: int, column: int, row: int, in_used: bool | int = True):
        self.shank = int(shank)
        self.column = int(column)
        self.row = int(row)

        if isinstance(in_used, int):
            in_used = in_used != 0

        self.in_used = in_used

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
    type: Literal['ext', 'tip', 'on-shank']
    shank: int  # 0 if reference_type is 'ext'
    channel: int  # 0 if reference_type is 'ext' or 'tip'

    @classmethod
    def max_reference_value(cls, probe_type: ProbeType) -> int:
        """get max possible value for probe type."""
        n_shank = probe_type.n_shank
        refs = probe_type.reference
        return 1 + n_shank + n_shank * len(refs)

    @classmethod
    def of(cls, probe_type: ProbeType, reference: int) -> ReferenceInfo:
        """
        get information of reference value.

        :param probe_type:
        :param reference:
        :return:
        """
        if reference == 0:
            return ReferenceInfo('ext', 0, 0)

        n_shank = probe_type.n_shank
        ref_shank = probe_type.reference

        if reference < n_shank + 1:
            return ReferenceInfo('tip', reference - n_shank, 0)

        x = reference - n_shank - 1
        s = x // len(ref_shank)
        c = ref_shank[x % len(ref_shank)]
        return ReferenceInfo('on-shank', s, c)


E = int | tuple[int, int] | tuple[int, int, int] | Electrode
A = list[int] | NDArray[np.int_]
Es = list[int] | NDArray[np.int_] | list[Electrode]


class ChannelMap:
    __match_args__ = 'probe_type',

    def __init__(self, probe_type: int | str | ProbeType,
                 electrodes: list[Electrode] | ChannelMap = None):
        if electrodes is None:
            electrodes = []
        elif isinstance(electrodes, ChannelMap):
            electrodes = cast(list[Electrode], electrodes._electrodes)

        if isinstance(probe_type, (int, str)):
            probe_type = PROBE_TYPE[probe_type]

        self.probe_type: Final = probe_type
        self._electrodes: Final[list[Electrode | None]] = [None] * probe_type.n_channels
        self._reference = 0

        for e in electrodes:
            c, _ = e2cb(probe_type, e)
            if self._electrodes[c] is not None:
                raise ChannelHasUsedError(e)
            self._electrodes[c] = Electrode(e.shank, e.column, e.row)

    # ========= #
    # load/save #
    # ========= #

    @classmethod
    def parse(cls, source: str) -> ChannelMap:
        from .io import parse_imro
        return parse_imro(source)

    @classmethod
    def from_meta(cls, path: str | Path) -> ChannelMap:
        from .io import load_meta
        return load_meta(path)

    @classmethod
    def from_imro(cls, path: str | Path) -> ChannelMap:
        from .io import load_imro
        return load_imro(path)

    @classmethod
    def from_probe(cls, probe) -> ChannelMap:
        from .io import from_probe
        return from_probe(probe)

    def save_imro(self, path: str | Path):
        from .io import save_imro
        save_imro(self, path)

    def to_probe(self):
        from .io import to_probe
        return to_probe(self)

    def __str__(self) -> str:
        from .io import string_imro
        return string_imro(self)

    def __repr__(self) -> str:
        return f'ChannelMap[{self.n_shank},{self.n_col_shank},{self.n_row_shank},{len(self._electrodes)}]'

    # ================= #
    # Basic information #
    # ================= #

    def __len__(self) -> int:
        ret = 0
        for e in self._electrodes:
            if e is not None:
                ret += 1
        return ret

    @property
    def n_shank(self) -> int:
        return self.probe_type.n_shank

    @property
    def n_col_shank(self) -> int:
        return self.probe_type.n_col_shank

    @property
    def n_row_shank(self) -> int:
        return self.probe_type.n_row_shank

    @property
    def n_electrode_shank(self) -> int:
        return self.probe_type.n_electrode_shank

    @property
    def n_channels(self) -> int:
        return self.probe_type.n_channels

    @property
    def n_electrode_block(self) -> int:
        return self.probe_type.n_electrode_block

    @property
    def reference(self) -> int:
        """reference type. see `reference_info` for more information."""
        return self._reference

    @reference.setter
    def reference(self, value: int):
        if not (0 <= value < ReferenceInfo.max_reference_value(self.probe_type)):
            raise ValueError(f'illegal reference value : {value}')

        self._reference = value

    @property
    def reference_info(self) -> ReferenceInfo:
        """
        reference information.
        """
        return ReferenceInfo.of(self.probe_type, self._reference)

    # =================== #
    # channel information #
    # =================== #

    @property
    def channel_shank(self) -> NDArray[np.int_]:
        return np.array([it.shank for it in self.electrodes])

    @property
    def channel_pos_x(self) -> NDArray[np.int_]:
        x, y = channel_coordinate(self)
        return x

    @property
    def channel_pos_y(self) -> NDArray[np.int_]:
        x, y = channel_coordinate(self)
        return y

    def get_channel(self, channel: int) -> Electrode | None:
        return self._electrodes[channel]

    @property
    def channels(self) -> Channels:
        return Channels(self._electrodes)

    def get_electrode(self, electrode: E) -> Electrode | None:
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
                raise TypeError()

        for e in self._electrodes:
            if e is not None and e.shank == shank and e.column == column and e.row == row:
                return e
        return None

    @property
    def electrodes(self) -> Electrodes:
        return Electrodes(self.probe_type, self._electrodes)

    # ==================== #
    # add/delete electrode #
    # ==================== #

    def add_electrode(self, electrode: E,
                      in_used: bool = True,
                      exist_ok: bool = False) -> Electrode | None:
        """

        :param electrode: electrode ID, tuple of (shank, electrode) or tuple of (shank, column, row)
        :param in_used: Is it used.
        :param exist_ok:
            `exist_ok == False` will raise an error if electrode has existed.
            `exist_ok == None` will return None if electrode has existed.
        :return: return an electrode which has been created.
        :raise ValueError: one of shank, column and row out of range
        :raise ChannelHasUsedError: duplicated channels using in electrodes
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

        e = self.get_electrode((shank, column, row))

        if e is None:
            e = Electrode(shank, column, row, in_used)
            c, _ = e2cb(self.probe_type, e)
            if self._electrodes[c] is not None:
                raise ChannelHasUsedError(e)
            self._electrodes[c] = e
            return e
        elif exist_ok is None:
            return None
        elif exist_ok:
            return e
        else:
            raise ChannelHasUsedError(e)

    def del_electrode(self, electrode: E) -> Electrode | None:
        """

        :param electrode:
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
    """Error of a map contains two electrodes with same channel.

    This limitation came from 4-shank Neuropixels version 2.
    """

    def __init__(self, electrode: Electrode):
        super().__init__(str(electrode))
        self.electrode: Final = electrode


class Channels(Sized, Iterable[Electrode]):
    def __init__(self, electrode: list[Electrode | None]):
        self._electrodes: Final = electrode

    def __len__(self):
        ret = 0
        for e in self._electrodes:
            if e is not None:
                ret += 1
        return ret

    def __getitem__(self, item):
        if all_int(item):
            return self._electrodes[int(item)]
        elif isinstance(item, slice):
            return [self._electrodes[it] for it in range(len(self._electrodes))[item]]
        else:
            return [self._electrodes[int(it)] for it in np.arange(len(self._electrodes))[item]]

    def __delitem__(self, item):
        if all_int(item):
            self._electrodes[int(item)] = None
        elif isinstance(item, slice):
            for it in range(len(self._electrodes))[item]:
                self._electrodes[it] = None
        else:
            for it in np.arange(len(self._electrodes))[item]:
                self._electrodes[int(it)] = None

    def __iter__(self) -> Iterator[Electrode | None]:
        for e in self._electrodes:
            yield e


class Electrodes(Sized, Iterable[Electrode]):
    def __init__(self, probe_type: ProbeType, electrode: list[Electrode | None]):
        self._probe_type: Final = probe_type
        self._electrodes: Final = electrode

    def __len__(self):
        ret = 0
        for e in self._electrodes:
            if e is not None:
                ret += 1
        return ret

    def __getitem__(self, item):
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

    def __delitem__(self, item):
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
        for e in self._electrodes:
            if e is not None:
                yield e


def channel_coordinate(shank_map: ChannelMap,
                       s_step: int = None) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
    """
    get coordinate of channels.

    :param shank_map:
    :param s_step: overwrite (horizontal) distance between shanks.
    :return: tuple of (x, y) in um.
    """
    probe_type = shank_map.probe_type
    h_step = probe_type.c_space
    v_step = probe_type.r_space
    s_step = probe_type.s_space if s_step is None else s_step

    t = len(shank_map.electrodes)
    x = np.zeros((t,), dtype=int)
    y = np.zeros((t,), dtype=int)

    for i, e in enumerate(shank_map.electrodes):
        x[i] = s_step * e.shank + h_step * e.column
        y[i] = v_step * e.row

    return x, y


def electrode_coordinate(probe_type: int | str | ChannelMap | ProbeType) -> NDArray[np.int_]:
    match probe_type:
        case ChannelMap(probe_type=probe_type):
            pass
        case ProbeType():
            pass
        case str() | int():
            probe_type = PROBE_TYPE[probe_type]
        case _:
            raise TypeError(repr(probe_type))

    y = probe_type.r_space * np.arange(probe_type.n_row_shank)
    x = probe_type.c_space * np.arange(probe_type.n_col_shank)
    s = probe_type.s_space * np.arange(probe_type.n_shank)
    x = np.add.outer(x, s).ravel()
    i, j = np.mgrid[0:len(x), 0:len(y)]
    return np.vstack([
        x[i].ravel(), y[j].ravel()
    ]).T


ELECTRODE_MAP_21 = (np.array([1, 7, 5, 3], dtype=int),
                    np.array([0, 4, 8, 12], dtype=int))
ELECTRODE_MAP_24 = np.array([
    [0, 2, 4, 6, 5, 7, 1, 3],  # shank-0
    [1, 3, 5, 7, 4, 6, 0, 2],  # shank-1
    [4, 6, 0, 2, 1, 3, 5, 7],  # shank-2
    [5, 7, 1, 3, 0, 2, 4, 6],  # shank-3
], dtype=int)


@overload
def e2p(probe_type: ProbeType, e: E) -> tuple[float, float]:
    pass


@overload
def e2p(probe_type: ProbeType, e: Es) -> tuple[NDArray[np.float_], NDArray[np.float_]]:
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
        case [*_]:
            s, e = align_arr(0, np.array(e))
            c, r = e2cr(probe_type, e)
        case e if isinstance(e, np.ndarray):
            if e.ndim != 1:
                raise ValueError()
            s, e = align_arr(0, e)
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
        case [*_]:
            e = np.array(e)
        case e if isinstance(e, np.ndarray):
            if e.ndim != 1:
                raise ValueError()
        case _:
            raise TypeError(repr(e))

    n = probe_type.n_col_shank
    r = e // n
    c = e % n

    return c, r


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
        case e if all_int(e):
            return int(e)
        case (c, r) if all_int(c, r):
            c = int(c)
            r = int(r)
        case (s, c, r) if all_int(s, c, r):
            c = int(c)
            r = int(r)
        case Electrode(column=c, row=r):
            pass
        case (c, r):
            c, r = align_arr(c, r)
        case [Electrode(), *_]:
            c = np.array([it.column for it in p])
            r = np.array([it.row for it in p])
        case [*_]:
            return np.array(p)
        case _ if isinstance(p, np.ndarray):
            match p.shape:
                case (_, ):
                    return p
                case (_, 2):
                    c = p[:, 0]
                    r = p[:, 1]
                case _:
                    raise ValueError()
        case _:
            raise TypeError(repr(p))

    n = probe_type.n_col_shank
    return r * n + c


@overload
def e2c(probe_type: ProbeType, electrode: E) -> int:
    pass


@overload
def e2c(probe_type: ProbeType, electrode: Es | tuple[int | A, A] | tuple[int | A, A, A]) -> NDArray[np.int_]:
    pass


def e2c(probe_type: ProbeType, electrode):
    c, b = e2cb(probe_type, electrode)
    n = probe_type.n_channels
    return c + b * n


@overload
def e2cb(probe_type: ProbeType, electrode: E) -> tuple[int, int]:
    pass


@overload
def e2cb(probe_type: ProbeType, electrode: Es | tuple[int | A, A] | tuple[int | A, A, A]) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
    pass


def e2cb(probe_type: ProbeType, electrode):
    match electrode:
        case electrode if all_int(electrode):
            shank = 0
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
        case [*_]:
            shank, electrode = align_arr(0, np.array(electrode))
        case _ if isinstance(electrode, np.ndarray):
            shank = np.zeros_like(electrode, dtype=int)
        case (shank, electrode):
            shank, electrode = align_arr(shank, electrode)
        case (shank, column, row):
            electrode = cr2e(probe_type, (column, row))
            shank, electrode = align_arr(shank, electrode)
        case _:
            raise TypeError()

    match probe_type.code:
        case 0:
            return e2c0(electrode)
        case 21:
            return e2c21(electrode)
        case 24:
            return e2c24(shank, electrode)
        case _:
            raise RuntimeError()


@overload
def e2c0(electrode: int) -> tuple[int, int]:
    pass


@overload
def e2c0(electrode: NDArray[np.int_]) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
    pass


def e2c0(electrode):
    bank, channel = divmod(electrode, 384)
    return channel, bank


@overload
def e2c21(electrode: int) -> tuple[int, int]:
    pass


@overload
def e2c21(electrode: NDArray[np.int_]) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
    pass


def e2c21(electrode):
    bf, ba = ELECTRODE_MAP_21
    bank, electrode = divmod(electrode, 384)
    block, index = divmod(electrode, 32)
    row, col = divmod(index, 2)
    channel = 2 * ((row * bf[bank] + col * ba[bank]) % 16) + 32 * block + col
    return channel, bank


@overload
def e2c24(shank: int, electrode: int) -> tuple[int, int]:
    pass


@overload
def e2c24(shank: NDArray[np.int_], electrode: NDArray[np.int_]) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
    pass


def e2c24(shank, electrode):
    bank, electrode = divmod(electrode, 384)
    block, index = divmod(electrode, 48)
    block = ELECTRODE_MAP_24[shank, block]
    return 48 * block + index, bank


@overload
def c2e(probe_type: ProbeType, channel: int, bank: int = None, shank: int = None) -> int:
    pass


@overload
def c2e(probe_type: ProbeType, channel: A, bank: int | A = None, shank: int | A = None) -> NDArray[np.int_]:
    pass


def c2e(probe_type: ProbeType, channel, bank=None, shank=None):
    match (bank, channel):
        case (None, channel) if all_int(channel):
            channel = int(channel)
            n = probe_type.n_channels
            bank = channel // n
            channel = channel % n
        case (None, channel):
            n = probe_type.n_channels
            channel = np.asarray(channel)
            bank = channel // n
            channel = channel % n
        case (bank, channel) if all_int(bank, channel):
            bank = int(bank)
            channel = int(channel)
        case (bank, channel):
            bank, channel = align_arr(bank, channel)
        case _:
            raise TypeError()

    match probe_type.code:
        case 0:
            return c2e0(bank, channel)
        case 21:
            return c2e21(bank, channel)
        case 24:
            match (shank, channel):
                case (None, channel) if all_int(channel):
                    shank = 0
                case (None, channel):
                    shank = np.zeros_like(channel)
                case (shank, channel) if all_int(shank, channel):
                    shank = int(shank)
                    channel = int(channel)
                case (shank, channel):
                    shank, bank, channel = align_arr(shank, bank, channel)
                case _:
                    raise TypeError()

            return c2e24(shank, bank, channel)
        case _:
            raise RuntimeError()


@overload
def c2e0(bank: int, channel: int) -> int:
    pass


@overload
def c2e0(bank: NDArray[np.int_], channel: NDArray[np.int_]) -> NDArray[np.int_]:
    pass


def c2e0(bank, channel):
    return bank * 384 + channel


@overload
def c2e21(bank: int, channel: int) -> int:
    pass


@overload
def c2e21(bank: NDArray[np.int_], channel: NDArray[np.int_]) -> NDArray[np.int_]:
    pass


def c2e21(bank, channel):
    bf, ba = ELECTRODE_MAP_21

    if all_int(bank, channel):
        bank = int(bank)
        channel = int(channel)
    else:
        bank, channel = align_arr(bank, channel)

    block, index = divmod(channel, 32)
    row, col = divmod(index, 2)
    rows = np.arange(0, 16)

    if isinstance(bank, int):
        mat = (rows * bf[bank] + col * ba[bank]) % 16
        row = np.nonzero(mat == row)[0][0]
    else:
        mat = (np.multiply.outer(rows, bf[bank]) + col * ba[bank]) % 16  # (16, E)
        row, i = np.nonzero(mat == row)
        row = row[np.argsort(i)]

    return bank * 384 + block * 32 + row * 2 + col


@overload
def c2e24(shank: int, bank: int, channel: int) -> int:
    pass


@overload
def c2e24(shank: NDArray[np.int_], bank: NDArray[np.int_], channel: NDArray[np.int_]) -> NDArray[np.int_]:
    pass


def c2e24(shank, bank, channel):
    if all_int(shank, bank, channel):
        shank = int(shank)
        bank = int(bank)
        channel = int(channel)
    else:
        shank, bank, channel = align_arr(shank, bank, channel)

    index = channel % 48
    block = channel // 48

    if isinstance(shank, int):
        block = np.nonzero(ELECTRODE_MAP_24[shank] == block)[0][0]
    else:
        block, i = np.nonzero(ELECTRODE_MAP_24[shank].T == block)
        block = block[np.argsort(i)]

    return 384 * bank + 48 * block + index


def all_int(*x) -> bool:
    for xx in x:
        if not isinstance(xx, (int, np.integer)):
            return False
    return True


def align_arr(*x: int | NDArray[np.int_]) -> list[NDArray[np.int_]]:
    if len(x) < 2:
        raise RuntimeError()

    ret = [np.asarray(it) for it in x]
    sz = max([len(it) for it in ret])
    ret = [np.full((sz,), it) if it.ndim == 0 else it for it in ret]
    if not all([it.ndim == 1 and len(it) == sz for it in ret]):
        raise RuntimeError()

    return ret


def as_set(x, n: int) -> set[int]:
    if x is None:
        return set(range(n))
    if all_int(x):
        return {int(x)}
    elif isinstance(x, slice):
        return set(range(n)[x])
    elif isinstance(x, tuple):
        ret = set()
        for xx in x:
            ret.update(as_set(xx, n))
        return ret
    else:
        return set(map(int, x))
