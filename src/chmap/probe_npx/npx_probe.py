from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import NamedTuple, Final, Literal, overload, Sized, cast, Any

import numpy as np
from numpy.typing import NDArray

__all__ = [
    'ProbeType',
    'PROBE_TYPE_NP1',
    'PROBE_TYPE_NP21',
    'PROBE_TYPE_NP24',
    'Electrodes',
    'ChannelMap',
    'channel_coordinate',
    'electrode_coordinate',
    'parse_imro',
    'string_imro',
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
    'NP2_1': PROBE_TYPE_NP21,
    'PRB2_1_2_0640_0': PROBE_TYPE_NP21,
    'PRB2_1_4_0480_1': PROBE_TYPE_NP21,
    'NP2000': PROBE_TYPE_NP21,
    'NP2003': PROBE_TYPE_NP21,
    'NP2004': PROBE_TYPE_NP21,
    #
    24: PROBE_TYPE_NP24,
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
        self.shank = shank
        self.column = column
        self.row = row

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

    @classmethod
    def parse(cls, source: str) -> ChannelMap:
        return parse_imro(source)

    def __len__(self) -> int:
        ret = 0
        for e in self._electrodes:
            if e is not None:
                ret += 1
        return ret

    def __str__(self) -> str:
        return string_imro(self)

    def __repr__(self) -> str:
        return f'ChannelMap[{self.n_shank},{self.n_col_shank},{self.n_row_shank},{len(self._electrodes)}]'

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
            case int():
                shank = 0
                column, row = e2cr(self.probe_type, electrode)
            case (int(shank), int(electrode)):
                column, row = e2cr(self.probe_type, electrode)
            case (int(shank), int(column), int(row)) | Electrode(shank=shank, column=column, row=row):
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
            case int():
                shank = 0
                column, row = e2cr(self.probe_type, electrode)
            case (int(shank), int(electrode)):
                column, row = e2cr(self.probe_type, electrode)
            case (int(shank), int(column), int(row)) | Electrode(shank=shank, column=column, row=row):
                pass
            case _:
                raise TypeError()

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
            case int():
                shank = 0
                column, row = e2cr(self.probe_type, electrode)
            case (int(shank), int(electrode)):
                column, row = e2cr(self.probe_type, electrode)
            case (int(shank), int(column), int(row)) | Electrode(shank=shank, column=column, row=row):
                pass
            case _:
                raise TypeError()

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
        if isinstance(item, int) or isinstance(item, np.integer):
            return self._electrodes[int(item)]
        elif isinstance(item, slice):
            return [self._electrodes[it] for it in range(len(self._electrodes))[item]]
        else:
            return [self._electrodes[int(it)] for it in np.arange(len(self._electrodes))[item]]

    def __delitem__(self, item):
        if isinstance(item, int) or isinstance(item, np.integer):
            self._electrodes[int(item)] = None
        elif isinstance(item, slice):
            for it in range(len(self._electrodes))[item]:
                self._electrodes[it] = None
        else:
            for it in np.arange(len(self._electrodes))[item]:
                self._electrodes[int(it)] = None

    def __iter__(self) -> Iterator[Electrode]:
        for e in self._electrodes:
            if e is not None:
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

    @classmethod
    def _set(cls, x, n: int) -> set[int]:
        if x is None:
            return set(range(n))
        if isinstance(x, int) or isinstance(x, np.integer):
            return {int(x)}
        elif isinstance(x, slice):
            return set(range(n)[x])
        elif isinstance(x, tuple):
            ret = set()
            for xx in x:
                ret.update(cls._set(xx, n))
            return ret
        else:
            return set(map(int, x))

    def __getitem__(self, item):
        shank, cols, rows = item
        match item:
            case (None, None, None):
                return [e for e in self._electrodes if e is not None]
            case (int(shank), int(column), int(row)):
                for e in self._electrodes:
                    if e is not None and e.shank == shank and e.column == column and e.row == row:
                        return e
                return None
            case (_, _, _):
                shank = self._set(shank, self._probe_type.n_shank)
                cols = self._set(cols, self._probe_type.n_col_shank)
                rows = self._set(rows, self._probe_type.n_row_shank)
                ret = []
                for e in self._electrodes:
                    if e is not None and e.shank in shank and e.column in cols and e.row in rows:
                        ret.append(e)
                return ret
            case _:
                raise TypeError()

    def __delitem__(self, item):
        shank, cols, rows = item
        match item:
            case (None, None, None):
                for c in range(len(self._electrodes)):
                    self._electrodes[c] = None
            case (int(shank), int(column), int(row)):
                for c in range(len(self._electrodes)):
                    e = self._electrodes[c]
                    if e is not None and e.shank == shank and e.column == column and e.row == row:
                        self._electrodes[c] = e
                        break
            case (_, _, _):
                shank = self._set(shank, self._probe_type.n_shank)
                cols = self._set(cols, self._probe_type.n_col_shank)
                rows = self._set(rows, self._probe_type.n_row_shank)
                for c in range(len(self._electrodes)):
                    e = self._electrodes[c]
                    if e is not None and e.shank in shank and e.column in cols and e.row in rows:
                        self._electrodes[c] = None
            case _:
                raise TypeError()

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


def electrode_coordinate(probe_type: ChannelMap | ProbeType) -> NDArray[np.int_]:
    if isinstance(probe_type, ChannelMap):
        probe_type = probe_type.probe_type

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
        case int(e):
            pass
        case (int(), int(e)):
            pass
        case (int(), int(c), int(r)):
            return c, r
        case Electrode(column=c, row=r):
            return c, r
        case [int(), *_]:
            e = np.array(e)
        case [Electrode(), *_]:
            c = np.array([it.column for it in e])
            r = np.array([it.row for it in e])
            return c, r

        case _ if isinstance(e, np.ndarray):
            if e.ndim != 1:
                raise ValueError()
        case _:
            raise TypeError()

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
        case int(e):
            return e
        case (int(c), int(r)):
            pass
        case (int(), int(c), int(r)):
            pass
        case Electrode(column=c, row=r):
            pass
        case (int(c), r):
            r = np.asarray(r)
            c = np.full_like(r, c)
        case (c, int(r)):
            c = np.asarray(c)
            r = np.full_like(c, r)
        case (c, r):
            r = np.asarray(r)
            c = np.asarray(c)
        case [int(), *_]:
            return np.array(p)
        case [Electrode(), *_]:
            c = np.array([it.column for it in p])
            r = np.array([it.row for it in p])
        case _ if isinstance(p, np.ndarray):
            match p.shape:
                case (int(), ):
                    return p
                case (int(), 2):
                    c = p[:, 0]
                    r = p[:, 1]
                case _:
                    raise ValueError()
        case _:
            raise TypeError()

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
        case int():
            shank = 0
        case (int(shank), int(electrode)):
            pass
        case (int(shank), int(column), int(row)):
            electrode = cr2e(probe_type, (column, row))
        case Electrode(shank=shank, column=column, row=row):
            electrode = cr2e(probe_type, (column, row))
        case [int(), *_]:
            electrode = np.array(electrode)
            shank = np.zeros_like(electrode)
        case [Electrode(), *_]:
            shank = np.array([it.shank for it in electrode])
            electrode = np.array([it.electrode for it in electrode])
        case _ if isinstance(electrode, np.ndarray):
            shank = np.zeros_like(electrode, dtype=int)
        case (int(shank), electrode):
            electrode = np.array(electrode)
            shank = np.full_like(electrode, shank)
        case (shank, electrode):
            electrode = np.array(electrode)
            shank = np.array(shank)
            _ = electrode + shank
        case (int(shank), column, row):
            electrode = cr2e(probe_type, (column, row))
            shank = np.full_like(electrode, shank)
        case (shank, column, row):
            electrode = cr2e(probe_type, (column, row))
            shank = np.array(shank)
            _ = electrode + shank
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
        case (None, None, int(channel)):
            n = probe_type.n_channels
            bank = channel // n
            channel = channel % n
        case (None, channel):
            n = probe_type.n_channels
            channel = np.asarray(channel)
            bank = channel // n
            channel = channel % n
        case (int(bank), int(channel)):
            pass
        case (int(bank), _):
            channel = np.asarray(channel)
            bank = np.full_like(channel, bank)
        case (_, int()):
            raise TypeError()
        case (_, _):
            bank = np.asarray(bank)
            channel = np.asarray(channel)
            _ = bank + channel
        case _:
            raise TypeError()

    match probe_type.code:
        case 0:
            return c2e0(bank, channel)
        case 21:
            return c2e21(bank, channel)
        case 24:
            match (shank, channel):
                case (None, int()):
                    shank = 0
                case (None, _):
                    shank = np.zeros_like(channel)
                case (int(), int()):
                    pass
                case (int(), _):
                    shank = np.full_like(channel, shank)
                case (_, int()):
                    shank = np.asarray(shank)
                    bank = np.full_like(shank, bank)
                    channel = np.full_like(shank, channel)
                case (_, _):
                    shank = np.asarray(shank)
                    _ = channel + shank
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

    block, index = divmod(channel, 32)
    row, col = divmod(index, 2)
    rows = np.arange(0, 16)

    if isinstance(bank, int) or isinstance(bank, np.integer):
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
    index = channel % 48
    block = channel // 48

    if isinstance(shank, int) or isinstance(shank, np.integer):
        block = np.nonzero(ELECTRODE_MAP_24[shank] == block)[0][0]
    else:
        block, i = np.nonzero(ELECTRODE_MAP_24[shank].T == block)
        block = block[np.argsort(i)]

    return 384 * bank + 48 * block + index


def parse_imro(source: str) -> ChannelMap:
    if not source.startswith('(') or not source.endswith(')'):
        raise RuntimeError('not imro format')

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
    ret = [f'({chmap.probe_type},{chmap.n_channels})']

    # channels
    match chmap.probe_type.code:
        case 0:
            for ch, e in enumerate(chmap.electrodes):
                ret.append(f'({ch} 0 {chmap.reference} {e.ap_band_gain} {e.lf_band_gain} {1 if e.ap_hp_filter else 0})')

        case 21:
            ref = chmap.reference
            for e in chmap.electrodes:
                electrode = cr2e(PROBE_TYPE_NP21, e)
                channel, bank = e2c21(electrode)
                ret.append(f'({channel} {bank} {ref} {electrode})')

        case 24:
            ref = chmap.reference
            for e in chmap.electrodes:
                electrode = cr2e(PROBE_TYPE_NP24, e)
                bank, channel = e2c24(e.shank, electrode)
                ret.append(f'({channel} {e.shank} {bank} {ref} {electrode})')
        case _:
            raise RuntimeError(f'unknown imro type : {chmap.probe_type}')

    return ''.join(ret)
