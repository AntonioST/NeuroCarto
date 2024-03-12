import math
import sys
from collections.abc import Callable
from typing import Literal, NamedTuple

import numpy as np
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from numpy.typing import NDArray

from neurocarto.probe import ElectrodeDesp
from neurocarto.util.util_numpy import index_of, closest_point_index, same_index
from neurocarto.util.utils import doc_link
from .desp import NpxProbeDesp
from .npx import ChannelMap, ProbeType, channel_coordinate, electrode_coordinate

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

__all__ = [
    'channel_coordinate',
    'electrode_coordinate',
    'plot_probe_shape',
    'plot_channelmap_block',
    'plot_channelmap_grid',
    'plot_channelmap_matrix',
    'plot_electrode_block',
    'plot_electrode_grid',
    'plot_electrode_matrix',
    'plot_category_area',
]

ELECTRODE_UNIT = Literal['cr', 'xy', 'raw']


class ElectrodeGridData(NamedTuple):
    """Electrode arrangement in boundary mode"""

    probe: ProbeType
    v_grid: np.ndarray  # Array[int, S, R, C+1]
    h_grid: np.ndarray  # Array[int, S, R+1, C]
    col: np.ndarray  # Array[int, C]
    row: np.ndarray  # Array[int, R]

    @classmethod
    def of(cls, probe: ProbeType,
           electrode: NDArray[np.int_],
           electrode_unit: ELECTRODE_UNIT = 'cr') -> Self:
        s_step = probe.s_space
        h_step = probe.c_space
        v_step = probe.r_space

        n_row = probe.n_row_shank
        n_col = probe.n_col_shank

        v_grid = np.zeros((probe.n_shank, n_row, n_col + 1), dtype=int)
        h_grid = np.zeros((probe.n_shank, n_row + 1, n_col), dtype=int)

        if electrode_unit == 'cr':
            s = electrode[:, 0].astype(int)
            c = electrode[:, 1].astype(int)
            r = electrode[:, 2].astype(int)
        elif electrode_unit == 'xy':
            x = electrode[:, 0]
            y = electrode[:, 1]

            if np.max(y) < 10:  # mm:
                x = (x * 1000)
                y = (y * 1000)

            x = x.astype(int)
            y = y.astype(int)

            s = x // s_step
            c = (x % s_step) // h_step
            r = y // v_step
        else:
            raise ValueError(f'unsupported electrode unit : {electrode_unit}')

        for ss, cc, rr in zip(s, c, r):
            v_grid[ss, rr, cc] += 1
            v_grid[ss, rr, cc + 1] += 1
            h_grid[ss, rr, cc] += 1
            h_grid[ss, rr + 1, cc] += 1

        return ElectrodeGridData(probe, v_grid, h_grid, np.arange(n_col), np.arange(n_row))

    @property
    def n_row(self) -> int:
        """
        :return: R
        """
        return len(self.row)

    @property
    def n_col(self) -> int:
        """
        :return: C
        """
        return len(self.col)

    @property
    def n_shank(self) -> int:
        """
        :return: S
        """
        return self.v_grid.shape[0]

    @property
    def shank_list(self) -> NDArray[np.int_]:
        return np.arange(self.n_shank)

    @property
    def y(self) -> NDArray[np.int_]:
        """Array[um, R]"""
        return self.row * self.probe.r_space

    @property
    def y_range(self) -> NDArray[np.int_]:
        """Array[um, (min, max)]"""
        return self.row[[0, -1]] * self.probe.r_space

    def with_height(self, height: float) -> Self:
        """

        :param height: um
        :return:
        """
        rx = np.max(np.nonzero(self.row * self.probe.r_space <= height)[0])
        return self._replace(
            v_grid=self.v_grid[:, :rx, :],
            h_grid=self.h_grid[:, :rx + 1, :],
            row=self.row[:rx]
        )

    def plot_shank(self, ax: Axes, shank: int,
                   cx: tuple[float, float, float] = None,
                   color: str = 'k',
                   shank_width_scale: float = 1,
                   **kwargs):
        """

        :param ax:
        :param shank:
        :param cx: (c0, cw) custom position for *shank*
        :param color:
        :param shank_width_scale: scaling the width of a shank for visualizing purpose.
        :param kwargs: pass to ax.plot(**kwargs)
        """
        s_step = self.probe.s_space
        h_step = self.probe.c_space * shank_width_scale
        v_step = self.probe.r_space

        if cx is None:
            cx = (
                (shank * s_step + 0 * h_step - h_step / 2) / 1000,
                h_step / 1000,
            )

        for y, x in zip(*np.nonzero(self.v_grid[shank] % 2 == 1)):
            x0 = cx[0] + x * cx[1]

            y0 = (y * v_step - v_step / 2) / 1000
            ax.plot([x0, x0], [y0, y0 + v_step / 1000], color=color, **kwargs)

        for y, x in zip(*np.nonzero(self.h_grid[shank] % 2 == 1)):
            x0 = cx[0] + x * cx[1]
            x1 = x0 + cx[1]
            y0 = (y * v_step - v_step / 2) / 1000
            ax.plot([x0, x1], [y0, y0], color=color, **kwargs)


class ElectrodeMatData(NamedTuple):
    """Electrode data in matrix form"""

    probe: ProbeType
    mat: NDArray[np.float_]  # Array[V:float, R, C]
    col: NDArray[np.int_]  # Array[int, C]
    row: NDArray[np.int_]  # Array[int, R]
    shank: NDArray[np.int_]  # Array[S:int, C]

    @classmethod
    def of(cls, probe: ProbeType,
           electrode: NDArray[np.float_],
           electrode_unit: ELECTRODE_UNIT = 'cr',
           reduce: Callable[[NDArray[np.float_]], float] = np.mean) -> Self:
        """
        Convert electrode array data into matrix data.

        :param probe: probe profile
        :param electrode: Array[float, E, (S, C, R, V?)] (electrode_unit='cr'),
                          Array[float, E, (X, Y, V?)] (electrode_unit='xy'), or
                          Array[V:float, S, C, R] (electrode_unit='raw')
        :param electrode_unit:
        :param reduce: function (Array[V, ?]) -> V used when data has same (s, x, y) position
        :return: ElectrodeMatData
        """
        if electrode_unit == 'raw':
            return cls._of_raw(probe, electrode)
        else:
            return cls._of(probe, electrode, electrode_unit, reduce)

    @classmethod
    def _of_raw(cls, probe: ProbeType,
                electrode: NDArray[np.float_]) -> Self:
        s, c, r = electrode.shape
        vmap = electrode.transpose(2, 0, 1).reshape(r, -1)  # Array[V:float, R, S*C]
        row = np.arange(r)
        col = np.tile(np.arange(c), s)  # [0,1,0,1, ...]
        shk = np.repeat(np.arange(s), c)  # [0,0,1,1,...]
        return ElectrodeMatData(probe, vmap, col=col, row=row, shank=shk)

    @classmethod
    def _of(cls, probe: ProbeType,
            electrode: NDArray[np.float_],
            electrode_unit: ELECTRODE_UNIT = 'cr',
            reduce: Callable[[NDArray[np.float_]], float] = np.mean) -> Self:
        nc = probe.n_col_shank
        s_step = probe.s_space
        h_step = probe.c_space
        v_step = probe.r_space

        if electrode_unit == 'cr':
            s = electrode[:, 0].astype(int)  # Array[s, E]
            c = electrode[:, 1].astype(int)  # Array[c, E]
            r = electrode[:, 2].astype(int)  # Array[r, E]
            v = electrode[:, 3]  # Array[float, E]

        elif electrode_unit == 'xy':
            x = electrode[:, 0]
            y = electrode[:, 1]
            v = electrode[:, 2]

            if np.max(y) < 10:  # mm:
                x = (x * 1000)
                y = (y * 1000)

            x = x.astype(int)
            y = y.astype(int)

            s = x // s_step
            c = (x % s_step) // h_step
            r = y // v_step

        else:
            raise ValueError(f'unsupported electrode unit : {electrode_unit}')

        sc = c + s * nc  # Array[sc, E]

        c0 = int(np.min(sc))
        c1 = int(np.max(sc))
        r0 = int(np.min(r))
        r1 = int(np.max(r))
        dc = c1 - c0
        dr = r1 - r0

        vmap = np.full((dr + 1, dc + 1), np.nan)
        vmap[r - r0, sc - c0] = v
        for i in same_index(np.vstack([c, r]).T):
            i0 = i[0]
            vmap[r[i0] - r0, sc[i0] - c0] = reduce(v[i])

        rr = np.arange(r0, r1 + 1)
        cc = np.arange(c0, c1 + 1)
        return ElectrodeMatData(probe, vmap, col=cc % nc, row=rr, shank=cc // nc)

    @property
    def n_row(self) -> int:
        """R """
        return len(self.row)

    @property
    def n_col(self) -> int:
        """len(unique(C)) """
        return len(np.unique(self.col))

    @property
    def n_shank(self) -> int:
        """len(unique(S)) """
        return len(np.unique(self.shank))

    @property
    def total_columns(self) -> int:
        """C """
        return len(self.col)

    @property
    def shank_list(self) -> NDArray[np.int_]:
        return np.unique(self.shank)

    @property
    def x(self) -> NDArray[np.int_]:
        """Array[um, C]"""
        return self.col * self.probe.c_space + self.shank * self.probe.s_space

    @property
    def y(self) -> NDArray[np.int_]:
        """Array[um, R]"""
        return self.row * self.probe.r_space

    @property
    def y_range(self) -> NDArray[np.int_]:
        """Array[um, (min, max)]"""
        return self.row[[0, -1]] * self.probe.r_space

    def irc(self, s: int, r: int, c: int) -> tuple[int, int]:
        """

        :param s:
        :param r:
        :param c:
        :return: (ir, ic)
        """
        ir = int(np.nonzero(self.row == r)[0][0])
        ic = int(np.nonzero((self.shank == s) & (self.col == c))[0][0])
        return ir, ic

    def src(self, ir: int, ic: int) -> tuple[int, int, int]:
        """
        :param ir: index of row
        :param ic: index of col
        :return: (shank, row, col)
        """
        return int(self.shank[ic]), int(self.row[ir]), int(self.col[ic])

    def with_shank(self, shank: int, nc: int | None = None) -> Self:
        """

        :param shank:
        :param nc: at least column number. 0 means allow empty result. None means use profile setting
        :return:
        """
        sc = self.shank == shank
        if nc is None:
            nc = self.probe.n_col_shank

        if np.any(sc) or nc == 0:
            return self._replace(mat=self.mat[:, sc], col=self.col[sc], shank=self.shank[sc])
        else:
            nr = self.n_row
            shape = (nr, nc)  # if self.mat.ndim == 2 else (nr, nc, self.n_sample)
            mat = np.full(shape, np.nan, dtype=self.mat.dtype)
            col = self.col
            for c in range(nc):
                cx = sc & (col == c)
                if np.any(cx):
                    mat[:, c] = self.mat[:, cx]
            return self._replace(mat=mat, col=np.arange(nc), shank=np.full((nc,), shank))

    def with_row(self, row: int | tuple[int, int] | NDArray[np.int_]) -> Self:
        if isinstance(row, int):
            row = np.arange(row + 1)
        elif isinstance(row, tuple):
            row = np.arange(row[0], row[1] + 1)

        if len(self.row) == len(row) and np.all(self.row == row):
            return self

        ri = index_of(self.row, row, missing=-1)
        mat = self.mat[ri].copy()
        mat[ri == -1] = np.nan

        return self._replace(mat=mat, row=row)

    def with_height(self, height: float | tuple[float, float]) -> Self:
        """

        :param height: max height in um or a range.
        :return:
        """
        y = self.y
        if isinstance(height, (int, float)):
            ri = y <= height
        elif isinstance(height, tuple):
            ri = np.logical_and(height[0] <= y, y <= height[1])
        else:
            raise TypeError()

        if np.all(ri):
            return self

        mat = self.mat[ri]
        row = self.row[ri]

        return self._replace(mat=mat, row=row)

    def sort_by_row(self) -> Self:
        i = np.argsort(self.row)
        return self._replace(mat=self.mat[i], row=self.row[i])

    def reorder_shank(self, shank_order: list[int] | NDArray[np.int_] | Literal['inc', 'dec']) -> Self:
        mat = np.empty_like(self.mat)
        col = np.empty_like(self.col)
        shank = np.empty_like(self.shank)
        used = np.zeros((self.total_columns,), dtype=bool)

        if isinstance(shank_order, str):
            if shank_order == 'inc':
                shank_order = list(np.unique(self.shank))
            elif shank_order == 'dec':
                shank_order = list(np.unique(self.shank)[::-1])
            else:
                raise ValueError()

        for s in shank_order:
            sc = np.nonzero(self.shank == s)[0]
            if (nc := len(sc)) == 0:
                continue

            oc = np.nonzero(~used)[0][:nc]
            mat[:, oc] = self.mat[:, sc]
            col[oc] = self.col[sc]
            shank[oc] = self.shank[sc]

            assert not np.any(used[oc])
            used[oc] = True

        if np.all(used):
            return self._replace(mat=mat, col=col, shank=shank)
        else:
            return self._replace(mat=mat[:, used], col=col[used], shank=shank[used])

    def extend_shank(self, shank: int, init: float = np.nan,
                     column_order: list[int] | NDArray[np.int_] | Literal['inc', 'dec'] = 'inc') -> Self:
        if np.any(self.shank == shank):
            raise RuntimeError()

        if isinstance(column_order, str):
            if column_order == 'inc':
                nc = self.probe.n_col_shank
                col = np.arange(nc)
            elif column_order == 'dec':
                nc = self.probe.n_col_shank
                col = np.arange(nc)[::-1]
            else:
                raise ValueError()
        else:
            col = np.array(column_order, dtype=int)
            nc = len(col)

        nr = self.n_row
        shape = (nr, nc)  # if self.mat.ndim == 2 else (nr, nc, self.n_sample)
        mat = np.full(shape, init, dtype=self.mat.dtype)
        snk = np.full((nc,), shank)

        return self._replace(
            mat=np.hstack([self.mat, mat]),
            col=np.concatenate([self.col, col]),
            shank=np.concatenate([self.shank, snk]),
        )

    def reorder_column(self, column_order: list[int] | NDArray[np.int_] | Literal['inc', 'dec']) -> Self:
        mat = np.empty_like(self.mat)
        col = np.empty_like(self.col)

        used = np.zeros((self.total_columns,), dtype=bool)
        for s in np.unique(self.shank):
            sc = np.nonzero(self.shank == s)[0]
            if len(sc) == 0:
                continue

            if isinstance(column_order, str):
                if column_order == 'inc':
                    oc = sc[np.argsort(self.col[sc])]
                elif column_order == 'dec':
                    oc = sc[np.argsort(-self.col[sc])]
                else:
                    raise ValueError()
            else:
                oc = sc[index_of(self.col[sc], column_order)]

            mat[:, oc] = self.mat[:, sc]
            col[oc] = self.col[sc]
            assert not np.any(used[oc])
            used[oc] = True

        if np.all(used):
            return self._replace(mat=mat, col=col)
        else:
            return self._replace(
                mat=mat[:, used],
                col=col[used],
                shank=self.shank[used]
            )

    @doc_link(interpolate_nan='neurocarto.util.util_numpy.interpolate_nan')
    def interpolate_nan(self, kernel: int | tuple[int, int] | Callable[[NDArray[np.float_]], NDArray[np.float_]]) -> Self:
        """

        :param kernel: interpolate missing data (NaN) between channels.
            It is pass to {interpolate_nan()}.
            Default (when use ``True``) is ``(0, 1)``.
        :return:
        """
        if not callable(kernel):
            from functools import partial
            from neurocarto.util.util_numpy import interpolate_nan
            kernel = partial(interpolate_nan, space=kernel, iteration=2)

        kernel: Callable
        nc = self.n_col
        tc = self.total_columns
        mat = self.mat.copy()
        for i in range(0, tc + 1, nc):  # foreach shanks
            ii = slice(i, i + nc)
            mat[:, ii] = kernel(mat[:, ii])
        return self._replace(mat=mat)


def plot_probe_shape(ax: Axes,
                     probe: ProbeType | ChannelMap,
                     height: float = 10,
                     color: str | None = 'k',
                     label_axis=False,
                     shank_width_scale: float = 1,
                     **kwargs):
    """
    Plot the probe shape.

    :param ax:
    :param probe: probe type
    :param height: max height (mm) of probe need to plot
    :param color: probe color
    :param label_axis: add labels on axes
    :param shank_width_scale: scaling the width of a shank for visualizing purpose.
    :param kwargs: pass to ax.plot(kwargs)
    """
    if isinstance(probe, ChannelMap):
        probe = probe.probe_type

    s_step = probe.s_space / 1000
    h_step = probe.c_space / 1000
    v_step = probe.r_space / 1000

    y0 = -v_step / 2
    y1 = -2 * v_step  # tip length
    w = h_step * shank_width_scale

    if color is not None:
        for sh in range(probe.n_shank):
            x0 = sh * s_step - w / 2
            x2 = x0 + probe.n_col_shank * w
            x1 = (x0 + x2) / 2

            ax.plot(
                [x0, x0, x1, x2, x2],
                [height, y0, y1, y0, height],
                color=color,
                **kwargs
            )

    if label_axis:
        ax.set_xticks(
            [(i * s_step + w / 2) for i in range(probe.n_shank)],
            [str(i) for i in range(probe.n_shank)],
        )
        ax.set_xlabel('Shanks')

        y_ticks = np.arange(int(height) + 1)
        ax.set_yticks(y_ticks, y_ticks)
        y_ticks = np.arange(int(math.ceil(height * 10))) * 0.1
        ax.set_yticks(y_ticks, minor=True)
        ax.set_ylabel('Distance from Tip (mm)')

    ax.set_xlim(-w, (probe.n_shank - 1) * s_step + probe.n_col_shank * w)
    ax.set_ylim(-0.3, height + 0.3)


def plot_channelmap_block(ax: Axes,
                          chmap: ChannelMap,
                          height: float = 10,
                          selection: Literal['used', 'unused', 'channel', 'disconnected', 'electrode'] = 'channel',
                          shank_width_scale: float = 1,
                          fill=True,
                          **kwargs):
    """

    :param ax:
    :param chmap: channelmap instance
    :param height: max height (mm) of probe need to plot
    :param selection: electrode selection
    :param shank_width_scale: scaling the width of a shank for visualizing purpose.
    :param fill: fill rectangle
    :param kwargs: pass to Rectangle(kwargs)
    """

    probe = chmap.probe_type
    if selection in ('channel', 'used', 'disconnected'):
        electrode = channel_coordinate(chmap, 'cr', include_unused=True)
        u = np.array([it.in_used for it in chmap.electrodes], dtype=bool)
        if selection == 'disconnected':
            electrode = electrode[~u]
        else:
            electrode = electrode[u]

    elif selection in ('electrode', 'unused'):
        electrode = electrode_coordinate(probe, 'cr')

        if selection == 'unused':
            i = []
            for q in channel_coordinate(chmap, 'cr'):
                if (j := closest_point_index(electrode, q, 0)) is not None:
                    i.append(j)
            electrode = np.delete(electrode, i, axis=0)

    else:
        raise ValueError()

    plot_electrode_block(ax, probe, electrode, 'cr',
                         height=height, fill=fill, shank_width_scale=shank_width_scale, **kwargs)


def plot_electrode_block(ax: Axes,
                         probe: ProbeType,
                         electrode: NDArray[np.float_] | ElectrodeMatData,
                         electrode_unit: ELECTRODE_UNIT | Literal['crv', 'xyv'] = 'cr', *,
                         height: float | None = 10,
                         shank_width_scale: float = 1,
                         fill=True,
                         **kwargs):
    """

    :param ax:
    :param probe: probe profile
    :param electrode: Array[float, E, (S, C, R, V?)] (electrode_unit='cr' or 'crv'),
                      Array[float, E, (X, Y, V?)] (electrode_unit='xy' or 'xyv'),
                      Array[V:float, S, C, r] (electrode_unit='raw'), or ElectrodeMatData
    :param electrode_unit:
    :param height: max height (mm) of probe need to plot
    :param shank_width_scale: scaling the width of a shank for visualizing purpose.
    :param fill: fill rectangle
    :param kwargs: pass to ``Rectangle(kwargs)`` or ``ax.imshow(kwargs)``.
    """
    from matplotlib.patches import Rectangle

    s_step = probe.s_space / 1000
    h_step = probe.c_space / 1000 * shank_width_scale
    v_step = probe.r_space / 1000

    if fill:
        w = h_step
        h = v_step
    else:
        w = int(h_step * 0.8)
        h = int(v_step * 0.8)

    x = y = data = None
    if isinstance(electrode, ElectrodeMatData):
        data = electrode
        electrode_unit = 'raw'
    elif electrode_unit == 'raw':
        data = ElectrodeMatData.of(probe, electrode, 'raw')
    elif electrode_unit == 'crv':
        data = ElectrodeMatData.of(probe, electrode, 'cr')
        electrode_unit = 'raw'
    elif electrode_unit == 'xyv':
        data = ElectrodeMatData.of(probe, electrode, 'xy')
        electrode_unit = 'raw'
    elif electrode_unit == 'cr':
        s = electrode[:, 0]
        x = electrode[:, 1] * h_step + s * s_step
        y = electrode[:, 2] * v_step
    elif electrode_unit == 'xy':
        x = electrode[:, 0]
        y = electrode[:, 1]
    else:
        raise ValueError(f'unsupported electrode unit : {electrode_unit}')

    if x is not None:
        if np.max(y, initial=0) > 10:  # um
            x /= 1000
            y /= 1000

        if height is not None:
            yx = y <= height
            x = x[yx]
            y = y[yx]
    else:
        if height is not None:
            data = data.with_height(height * 1000)

    if electrode_unit in ('cr', 'xy'):
        ret = []
        for i, j in zip(x - w / 2, y - h / 2):
            r = Rectangle((float(i), float(j)), w, h, lw=0, **kwargs)
            ret.append(r)
            ax.add_artist(r)
        return ret

    elif electrode_unit == 'raw':
        vmin = kwargs.pop('vmin', np.nanmin(data.mat))
        vmax = kwargs.pop('vmax', np.nanmax(data.mat))

        ret = []
        for s in data.shank_list:
            extent = [s * s_step - w / 2, s * s_step + h_step * (probe.n_col_shank - 1) + w / 2,
                      *data.y_range / 1000.0]

            im = ax.imshow(
                data.with_shank(s).mat,  # Array[V:float, R, C]
                origin='lower',
                extent=extent,
                aspect='auto',
                vmin=vmin,
                vmax=vmax,
                **kwargs
            )
            ret.append(im)
        return ret
    else:
        raise ValueError()


def plot_channelmap_grid(ax: Axes, chmap: ChannelMap, *,
                         height: float = 10,
                         shank_list: list[int] = None,
                         unit_column: bool = False,
                         unused=True,
                         half_as_full=False,
                         color: str = 'g',
                         **kwargs):
    """

    :param ax:
    :param chmap: channelmap  instance
    :param height: max height (mm) of probe need to plot
    :param shank_list: show shank in list
    :param unit_column: let one column as 1 mm
    :param unused: show disconnected channels
    :param half_as_full: make unused electrode which over half of surrounding electrode are read-out channels as a channel.
    :param color:
    :param kwargs: pass to ax.plot(kwargs)
    """
    probe = chmap.probe_type
    e = channel_coordinate(chmap, 'cr', include_unused=unused)

    if half_as_full:
        a = []
        for s in np.unique(e[:, 0]):
            sx = e[:, 0] == s
            c = e[sx, 1]  # [0, 1]
            r = e[sx, 2]  # [0, R]

            nc = probe.n_col_shank
            t = np.zeros((np.max(r) + 3, nc + 2), dtype=int)

            t[r, c + 1] += 1
            t[r + 2, c + 1] += 1
            t[r + 1, c] += 1
            t[r + 1, c + 2] += 1

            t[r + 1, c + 1] = 0  # unset used electrodes

            ar, ac = np.nonzero(t[1:, 1:] > 2)
            a.append(np.vstack([np.full_like(ar, s), ac, ar]).T)
        e = np.vstack([e, *a])

    plot_electrode_grid(ax, probe, e, 'cr',
                        shank_list=shank_list,
                        height=height,
                        unit_column=unit_column,
                        color=color,
                        **kwargs)


def plot_electrode_grid(ax: Axes,
                        probe: ProbeType,
                        electrode: NDArray[np.int_],
                        electrode_unit: ELECTRODE_UNIT = 'cr', *,
                        shank_list: list[int] = None,
                        height: float | None = 10,
                        unit_column: bool | tuple[float, ...] = False,
                        shank_width_scale: float = 1,
                        color: str = 'g',
                        label: str = None,
                        **kwargs):
    """

    :param ax:
    :param probe: probe type
    :param electrode: Array[int, E, (S, C, R)|(X, Y)]
    :param electrode_unit: 'xy'=(X,Y), 'cr'=(S,C,R)
    :param shank_list: show shank in list
    :param height: max height (mm) of probe need to plot
    :param unit_column: let one column as 1 mm, or (c0, c1, ..., cS, cw)
    :param shank_width_scale: scaling the width of a shank for visualizing purpose.
    :param color: grid line color
    :param label:
    :param kwargs: pass to ax.plot(kwargs)
    """
    data = ElectrodeGridData.of(probe, electrode, electrode_unit)

    if height is not None:
        data = data.with_height(height * 1000)

    for s in data.shank_list:
        cx = None
        if shank_list is None:
            si = s
        else:
            try:
                si = shank_list.index(s)
            except ValueError:
                continue

        if unit_column is True:
            cx = (si * data.probe.n_col_shank, 1)
        elif isinstance(unit_column, tuple):
            cx = (unit_column[si], unit_column[-1])

        data.plot_shank(ax, s, cx, color, shank_width_scale=shank_width_scale, **kwargs)

    if label is not None:
        ax.plot([np.nan], [np.nan], color=color, label=label, **kwargs)


@doc_link(interpolate_nan='neurocarto.util.util_numpy.interpolate_nan')
def plot_channelmap_matrix(ax: Axes,
                           chmap: ChannelMap,
                           data: NDArray[np.float_], *,
                           shank_list: list[int] = None,
                           kernel: int | tuple[int, int] | Callable[[NDArray[np.float_]], NDArray[np.float_]] | None = None,
                           reduce: Callable[[NDArray[np.float_]], float] = np.mean,
                           cmap='magma',
                           shank_gap_color: str | None = 'w',
                           **kwargs) -> ScalarMappable:
    """

    :param ax:
    :param chmap: channelmap instance
    :param data: Array[V:float, C] or Array[float, C', (C, V)]
    :param shank_list: show shank in order
    :param kernel: interpolate missing data (NaN) between channels.
        It is pass to {interpolate_nan()}.
        Default (when use ``True``) is ``(0, 1)``.
    :param reduce: function used when data has same (s, x, y) position
    :param cmap: colormap used in ax.imshow(cmap)
    :param shank_gap_color:
    :param kwargs: pass to ax.imshow(kwargs)
    """
    x = channel_coordinate(chmap, 'cr', include_unused=True).astype(float)  # Array[float, E, (S, C, R)]

    if data.ndim == 1:
        x = np.vstack([x, data.T])  # Array[float, E, (S, C, R, V)]
    elif data.ndim == 2:
        c = data[:, 0].astype(int)
        v = data[:, 1]
        x = np.vstack([x[c], v.T])  # Array[float, E, (S, C, R, V)]
    else:
        raise ValueError()

    return plot_electrode_matrix(
        ax, chmap.probe_type, x, 'cr',
        shank_list=shank_list,
        kernel=kernel,
        reduce=reduce,
        cmap=cmap,
        shank_gap_color=shank_gap_color,
        **kwargs
    )


@doc_link(interpolate_nan='neurocarto.util.util_numpy.interpolate_nan')
def plot_electrode_matrix(ax: Axes,
                          probe: ProbeType,
                          electrode: NDArray[np.float_] | ElectrodeMatData,
                          electrode_unit: ELECTRODE_UNIT = 'cr', *,
                          shank_list: list[int] = None,
                          kernel: int | tuple[int, int] | Callable[[NDArray[np.float_]], NDArray[np.float_]] | None = None,
                          reduce: Callable[[NDArray[np.float_]], float] = np.mean,
                          cmap='magma',
                          shank_gap_color: str | None = 'w',
                          **kwargs) -> ScalarMappable:
    """``ax.imshow`` the electrode data matrix.

    :param ax:
    :param probe: probe type
    :param electrode: Array[float, E, (S, C, R, V)] (electrode_unit='cr'),
                      Array[float, E, (X, Y, V)] (electrode_unit='xy'),
                      Array[V:float, S, C, r] (electrode_unit='raw'), or ElectrodeMatData
    :param electrode_unit:
    :param shank_list: show shank in order
    :param kernel: interpolate missing data (NaN) between channels.
        It is pass to {interpolate_nan()}.
        Default (when use ``True``) is ``(0, 1)``.
    :param reduce: function used when data has same (s, x, y) position
    :param cmap: colormap used in ax.imshow(cmap)
    :param shank_gap_color: color of shank gao line. Use None to disable plotting.
    :param kwargs: pass to ax.imshow(kwargs)
    """
    if isinstance(electrode, ElectrodeMatData):
        data = electrode
    else:
        data = ElectrodeMatData.of(probe, electrode, electrode_unit, reduce)

    if kernel is not None:
        data = data.interpolate_nan(kernel)

    nc = probe.n_col_shank

    if shank_list is not None:
        data = data.reorder_shank(shank_list)
        if shank_list[0] > shank_list[-1]:
            data = data.reorder_column('dec')

    y0, y1 = data.y_range / 1000
    im = ax.imshow(
        data.mat,
        extent=[0, data.total_columns, y0, y1],
        cmap=cmap, origin='lower', aspect='auto', **kwargs
    )

    shank_list = data.shank_list

    if shank_gap_color is not None:
        for i in range(0, len(shank_list) + 1):
            ax.axvline(i * nc, color=shank_gap_color, lw=2)

    ax.set_xlabel('Shanks')
    ax.set_xticks(np.arange(0, len(shank_list)) * nc + nc // 2,
                  shank_list)

    y_ticks = np.arange(int(math.ceil(y1)) + 1)
    ax.set_yticks(y_ticks, y_ticks)
    ax.set_yticks(np.arange(len(y_ticks) * 10) * 0.1, minor=True)
    ax.set_ylabel('Distance from Tip (mm)')

    return im


def plot_category_area(ax: Axes,
                       probe: ProbeType,
                       electrode: NDArray[np.int_] | list[ElectrodeDesp], *,
                       color: dict[int, str] = None,
                       **kwargs):
    """

    :param ax:
    :param probe:
    :param electrode: Array[category:int, N], Array[int, N, (S, C, R, category)] or list of ElectrodeDesp
    :param color:
    :param kwargs:
    :return:
    """
    if isinstance(electrode, list):
        _electrode = np.zeros((len(electrode), 4))
        for i, t in enumerate(electrode):  # type: int, ElectrodeDesp
            _electrode[i] = [*t.electrode, t.category]
        electrode = _electrode

    if color is None:
        color = {
            NpxProbeDesp.CATE_SET: 'green',
            NpxProbeDesp.CATE_FULL: 'green',
            NpxProbeDesp.CATE_HALF: 'orange',
            NpxProbeDesp.CATE_QUARTER: 'blue',
            NpxProbeDesp.CATE_FORBIDDEN: 'pink',
        }

    if electrode.ndim == 1:
        categories = electrode
        electrode = electrode_coordinate(probe, 'cr')
    elif electrode.ndim == 2:
        categories = electrode[:, 3]
        electrode = electrode[:, 0:3]
    else:
        raise ValueError()

    for category in np.unique(categories):
        if (c := color.get(int(category), None)) is not None:
            _electrode = electrode[categories == category]
            plot_electrode_block(ax, probe, _electrode, electrode_unit='cr', color=c, **kwargs)
