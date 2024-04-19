from __future__ import annotations

import math
import sys
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
    import pandas as pd  # type: ignore[import]
    import polars as pl  # type: ignore[import]
    from probeinterface import Probe  # type: ignore[import]
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
    * `SpikeGLX <https://github.com/jenniferColonell/SGLXMetaToCoords/blob/140452d43a55ea7c7904f09e03858bfe0d499df3/SGLXMetaToCoords.py#L79>`_

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

    @property
    def n_bank(self) -> int:
        """number of total banks"""
        return int(math.ceil(self.n_electrode_shank / self.n_channels))

    @property
    def n_block(self) -> int:
        """number of total blocks"""
        return self.n_electrode_shank // self.n_electrode_block

    @property
    def n_block_bank(self) -> int:
        """number of blocks in one bank"""
        return self.n_channels // self.n_electrode_block

    @classmethod
    def __class_getitem__(cls, item: int | str) -> ProbeType:
        return PROBE_TYPE[item]


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

    __slots__ = 'shank', 'column', 'row', 'in_used', 'ap_band_gain', 'lf_band_gain', 'ap_hp_filter'
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
    def of(cls, probe_type: ProbeType, reference: int) -> Self:
        """
        get information of reference value.

        :param probe_type:
        :param reference:
        :return:
        """
        if reference == 0:
            return ReferenceInfo(0, 'ext', 0, 0)

        n_shank = probe_type.n_shank
        ref_shank = probe_type.reference

        if reference < n_shank + 1:
            return ReferenceInfo(reference, 'tip', reference - 1, 0)

        x = reference - n_shank - 1
        s, i = divmod(x, len(ref_shank))
        c = ref_shank[i]
        return ReferenceInfo(reference, 'on-shank', s, c)


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
        """Parse imro table."""
        from .io import parse_imro
        return parse_imro(source)

    @classmethod
    def from_meta(cls, path: str | Path) -> Self:
        """Read imro table from meta file."""
        from .io import load_meta
        return load_meta(path)

    @classmethod
    def from_imro(cls, path: str | Path) -> Self:
        """Read imro file."""
        from .io import load_imro
        return load_imro(path)

    @classmethod
    def from_probe(cls, probe: Probe) -> Self:
        """From probeinterface.Probe"""
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
        """to probeinterface.Probe"""
        from .io import to_probe
        return to_probe(self)

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
        if not (0 <= value < ReferenceInfo.max_reference_value(self.probe_type)):
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
    def channel_shank(self) -> NDArray[np.float_]:
        """

        :return: Array[shank:int|NaN, C]
        """
        return np.array([it.shank if it is not None else np.nan for it in self.channels], dtype=float)

    @property
    def channel_pos_x(self) -> NDArray[np.float_]:
        """

        :return: Array[um:float, C]
        """
        return channel_coordinate(self, electrode_unit='xy')[:, 0]

    @property
    def channel_pos_y(self) -> NDArray[np.float_]:
        """

        :return: Array[um:float, C]
        """
        return channel_coordinate(self, electrode_unit='xy')[:, 1]

    @property
    def channel_pos(self) -> NDArray[np.float_]:
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
            case (shank, column, row) if all_int(shank, column, row):  # type: ignore
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
                       include_unused=False) -> NDArray[np.float_]:
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
        case (s, e) if all_int(s, e):  # type: ignore
            e = int(e)
        case (s, c, r) if all_int(s, c, r):  # type: ignore
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

    r, c = divmod(e, probe_type.n_col_shank)
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
        case (int(s), int(c), int(r)):
            pass
        case e if all_int(e):
            return int(e)
        case (c, r) if all_int(c, r):  # type: ignore
            c = int(c)
            r = int(r)
        case (s, c, r) if all_int(s, c, r):  # type: ignore
            c = int(c)
            r = int(r)
        case Electrode(column=c, row=r):
            pass
        case (c, r):
            c, r = align_arr(c, r)
        case [Electrode(), *_]:
            c = np.array([it.column for it in p])
            r = np.array([it.row for it in p])
        case _ if isinstance(p, np.ndarray):
            match p.shape:
                case (_, ):
                    return p
                case (_, 2):
                    c = p[:, 0]
                    r = p[:, 1]
                case _:
                    raise ValueError()
        case [*_]:
            return np.array(p)
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
        case int(electrode):
            shank = 0
        case (int(shank), int(electrode)):
            pass
        case (int(shank), int(column), int(row)):
            electrode = cr2e(probe_type, (column, row))
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

    block, index = divmod(channel, 48)

    if isinstance(shank, int):
        block = np.nonzero(ELECTRODE_MAP_24[shank] == block)[0][0]
    else:
        block, i = np.nonzero(ELECTRODE_MAP_24[shank].T == block)
        block = block[np.argsort(i)]

    return 384 * bank + 48 * block + index
