import math
import textwrap
from collections.abc import Callable
from typing import Literal

import numpy as np
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from numpy.typing import NDArray

from neurocarto.probe import ElectrodeDesp
from neurocarto.util.util_numpy import closest_point_index, interpolate_nan
from neurocarto.util.utils import doc_link
from .desp import NpxProbeDesp
from .npx import ChannelMap, ProbeType, channel_coordinate, electrode_coordinate

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
    'plot_electrode_curve',
    'plot_category_area',
    #
    'cast_electrode_data',
    'cast_electrode_grid',
    'cast_electrode_curve',
]

PROBE_TYPE = int | str | ChannelMap | ProbeType
ELECTRODE_UNIT = Literal['cr', 'xy', 'raw']


@doc_link()
def cast_probe_type(probe: PROBE_TYPE) -> ProbeType:
    """
    cast probe type identify to {ProbeType}

    :param probe: any probe type identify.
    :return: {ProbeType} instance
    """
    match probe:
        case int() | str():
            return ProbeType[probe]
        case ChannelMap(probe_type=ret):
            return ret
        case ProbeType():
            return probe
        case _:
            raise TypeError()


def cast_electrode_data(probe: PROBE_TYPE,
                        electrode: NDArray[np.float_],
                        electrode_unit: ELECTRODE_UNIT | Literal['crv', 'xyv']) -> NDArray[np.float_]:
    """
    **Data shape and units**

    the required shape of *electrode* is depended on *electrode_unit*.

    * ``cr``: column and row, require shape ``Array[float, E, (S, C, R)]``
    * ``crv``: column and row, require shape ``Array[float, E, (S, C, R, V)]``
    * ``xy``: x and y position, require shape ``Array[float, E, (X, Y)]``
    * ``xyv``: x and y position, require shape ``Array[float, E, (X, Y, V)]``
    * ``raw``:  require shape ``Array[V:float, E]`` or ``Array[V:float, S, C, R]``,

    where ``E=S*C*R`` means all electrodes for the *probe*.
    """
    probe: ProbeType = cast_probe_type(probe)
    shape = (probe.n_shank, probe.n_col_shank, probe.n_row_shank)

    match electrode_unit:
        case 'raw' if electrode.shape == shape:
            return electrode
        case 'raw' if electrode.ndim == 1:
            ret = np.full(shape, np.nan)
            for (s, c, r), v in zip(electrode_coordinate(probe, 'cr'), electrode):
                ret[s, c, r] = v
            return ret

        case 'raw':
            raise RuntimeError(f'electrode.ndim={electrode.ndim} not fit to raw unit')

        case 'cr':
            ret = np.full(shape, np.nan)
            for s, c, r in electrode.astype(int):
                ret[s, c, r] = 1
            return ret

        case 'crv':
            ret = np.full(shape, np.nan)
            for s, c, r, v in electrode:
                ret[int(s), int(c), int(r)] = v
            return ret

        case 'xy':
            ret = np.full(shape, np.nan)
            cr = electrode_coordinate(probe, 'cr')
            xy = electrode_coordinate(probe, 'xy')

            if np.max(electrode[:, 1]) < 10:  # mm
                electrode = electrode.copy()
                electrode[:, 0] *= 1000
                electrode[:, 1] *= 1000

            for x, y in electrode:
                if (i := closest_point_index(xy, [x, y], 0)) is not None:
                    ret[tuple(cr[i])] = 1
            return ret

        case 'xyv':
            ret = np.full(shape, np.nan)
            cr = electrode_coordinate(probe, 'cr')
            xy = electrode_coordinate(probe, 'xy')

            if np.max(electrode[:, 1]) < 10:  # mm
                electrode = electrode.copy()
                electrode[:, 0] *= 1000
                electrode[:, 1] *= 1000

            for x, y, v in electrode:
                if (i := closest_point_index(xy, [x, y], 0)) is not None:
                    ret[tuple(cr[i])] = v
            return ret

        case _:
            raise ValueError()


def cast_electrode_grid(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """

    :param data: Array[bool, S, C, R]
    :return: v_grid:Array[int, S, C+1, R] and h_grid:Array[int, S, C, R+1]
    """
    s, c, r = data.shape
    v_grid = np.zeros((s, c + 1, r), dtype=int)
    h_grid = np.zeros((s, c, r + 1), dtype=int)

    for ss, cc, rr in zip(*np.nonzero(data)):
        v_grid[ss, cc, rr] += 1
        v_grid[ss, cc + 1, rr] += 1
        h_grid[ss, cc, rr] += 1
        h_grid[ss, cc, rr + 1] += 1

    return v_grid, h_grid


def plot_probe_shape(ax: Axes,
                     probe: PROBE_TYPE,
                     height: float = 10,
                     color: str | None = 'k',
                     label_axis=False,
                     shank_width_scale: float = 1,
                     reverse_shank: bool = False,
                     **kwargs):
    """
    Plot the probe shape.

    :param ax:
    :param probe: probe type, see {cast_probe_type()}
    :param height: max height (mm) of probe need to plot
    :param color: probe color
    :param label_axis: add labels on axes
    :param shank_width_scale: scaling the width of a shank for visualizing purpose.
    :param reverse_shank:
    :param kwargs: pass to ax.plot(kwargs)
    """
    probe: ProbeType = cast_probe_type(probe)

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
        if reverse_shank:
            xticks = [str(probe.n_shank - i - 1) for i in range(probe.n_shank)]
        else:
            xticks = [str(i) for i in range(probe.n_shank)]

        ax.set_xticks(
            [(i * s_step + w / 2) for i in range(probe.n_shank)],
            xticks,
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
                          reverse_shank: bool = False,
                          **kwargs):
    """
    Plot channel in rectangles.

    :param ax:
    :param chmap: channelmap instance
    :param height: max height (mm) of probe need to plot
    :param selection: electrode selection
    :param shank_width_scale: scaling the width of a shank for visualizing purpose.
    :param fill: fill rectangle
    :param reverse_shank:
    :param kwargs: pass to Rectangle(kwargs)
    """

    probe = chmap.probe_type
    if selection == 'channel':
        electrode = channel_coordinate(chmap, 'cr', include_unused=True)
        u = np.isnan(electrode[:, 0])
        electrode = electrode[~u]
    elif selection == 'used':
        electrode = channel_coordinate(chmap, 'cr', include_unused=False)
        u = np.isnan(electrode[:, 0])
        electrode = electrode[~u]
    elif selection == 'disconnected':
        electrode = np.array([
            (it.shank, it.column, it.row) for it in chmap.electrodes
            if it is not None and not it.in_used
        ]).reshape(-1, 3)

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

    if reverse_shank:
        electrode[:, 0] = np.abs(probe.n_shank - electrode[:, 0] - 1)

    plot_electrode_block(ax, probe, electrode, 'cr',
                         height=height, fill=fill, shank_width_scale=shank_width_scale, **kwargs)


@doc_link(DOC=textwrap.dedent(cast_electrode_data.__doc__))
def plot_electrode_block(ax: Axes,
                         probe: PROBE_TYPE,
                         electrode: NDArray[np.float_],
                         electrode_unit: ELECTRODE_UNIT | Literal['crv', 'xyv'] = 'cr', *,
                         height: float | None = 10,
                         shank_width_scale: float = 1,
                         sparse=True,
                         fill=True,
                         cmap: str = None,
                         **kwargs):
    """
    Plot electrodes in rectangles per electrodes or per shanks.

    {DOC}

    :param ax:
    :param probe: probe type, see {cast_probe_type()}
    :param electrode: electrode data
    :param electrode_unit: electrode value unit
    :param sparse: If sparse, plot data block as rectangles (ignore V). use an image otherwise.
    :param height: max height (mm) of probe need to plot
    :param shank_width_scale: scaling the width of a shank for visualizing purpose.
    :param fill: fill rectangle
    :param cmap: colormap used by ``ax.imshow(kwargs)``.
    :param kwargs: pass to ``Rectangle(kwargs)`` or ``ax.imshow(kwargs)``.
    """
    from matplotlib.patches import Rectangle

    probe: ProbeType = cast_probe_type(probe)
    s_step = probe.s_space / 1000
    h_step = probe.c_space / 1000 * shank_width_scale
    v_step = probe.r_space / 1000

    if fill:
        w = h_step
        h = v_step
    else:
        w = int(h_step * 0.8)
        h = int(v_step * 0.8)

    data = cast_electrode_data(probe, electrode, electrode_unit)

    if sparse:
        s, c, r = np.nonzero(~np.isnan(data))
        x = c * h_step + s * s_step
        y = r * v_step

        if height is not None:
            yx = y <= height
            s = s[yx]
            c = c[yx]
            r = r[yx]
            x = x[yx]
            y = y[yx]

        kwargs.pop('lw', None)

        ret = []
        for ss, cc, rr, xx, yy in zip(s, c, r, x, y):
            a = Rectangle((float(xx - w / 2), float(yy - h / 2)), w, h, lw=0, **kwargs)
            ret.append(a)
            ax.add_artist(a)
        return ret

    else:
        y = np.arange(data.shape[2]) * v_step
        if height is not None:
            yx = y <= height
            y = y[yx]
            data = data[:, :, yx]

        vmin = kwargs.pop('vmin', np.nanmin(data))
        vmax = kwargs.pop('vmax', np.nanmax(data))

        ret = []
        for s in range(data.shape[0]):
            extent = [s * s_step - w / 2,
                      s * s_step + h_step * (data.shape[1] - 1) + w / 2,
                      0, y[-1]]

            im = ax.imshow(
                data[s].T,  # Array[V:float, R, C]
                origin='lower',
                extent=extent,
                aspect='auto',
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                **kwargs
            )
            ret.append(im)
        return ret


def plot_channelmap_grid(ax: Axes, chmap: ChannelMap,
                         height: float = 10,
                         shank_list: list[int] = None,
                         unused=True,
                         half_as_full=False,
                         color: str = 'g',
                         **kwargs):
    """
    Plot a channel map in grid block style.

    :param ax:
    :param chmap: channelmap  instance
    :param height: max height (mm) of probe need to plot
    :param shank_list: show shank in list
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
            a.append(np.column_stack([np.full_like(ar, s), ac, ar]))
        e = np.vstack([e, *a])

    plot_electrode_grid(ax, probe, e, 'cr',
                        shank_list=shank_list,
                        height=height,
                        color=color,
                        **kwargs)


def plot_electrode_grid(ax: Axes,
                        probe: PROBE_TYPE,
                        electrode: NDArray[np.int_],
                        electrode_unit: ELECTRODE_UNIT = 'cr', *,
                        shank_list: list[int] = None,
                        height: float | None = 10,
                        shank_width_scale: float = 1,
                        color: str = 'g',
                        label: str = None,
                        **kwargs):
    """
    Plot each electrode in grid rectangles.

    :param ax:
    :param probe: probe type, see {cast_probe_type()}
    :param electrode: Array[int, E, (S, C, R)|(X, Y)]
    :param electrode_unit: 'xy'=(X,Y), 'cr'=(S,C,R)
    :param shank_list: show shank in list
    :param height: max height (mm) of probe need to plot
    :param shank_width_scale: scaling the width of a shank for visualizing purpose.
    :param color: grid line color
    :param label:
    :param kwargs: pass to ``ax.plot(kwargs)``
    """
    probe: ProbeType = cast_probe_type(probe)
    s_step = probe.s_space / 1000
    h_step = probe.c_space / 1000 * shank_width_scale
    v_step = probe.r_space / 1000

    data = cast_electrode_data(probe, electrode, electrode_unit)
    v_grid, h_grid = cast_electrode_grid(data > 0)

    s, c, r = data.shape

    if height is not None:
        y = np.arange(r) * v_step
        yx = np.max(np.nonzero(y <= height)[0])
        v_grid = v_grid[:, :, :yx]
        h_grid = h_grid[:, :, :yx + 1]

    if shank_list is None:
        shank_list = list(range(s))

    for si, ss in enumerate(shank_list):
        cx = (
            si * s_step - h_step / 2,  # base
            h_step,  # step
        )

        for x, y in zip(*np.nonzero(v_grid[ss] % 2 == 1)):
            x0 = cx[0] + x * cx[1]
            y0 = (y * v_step - v_step / 2)
            ax.plot([x0, x0], [y0, y0 + v_step], color=color, **kwargs)

        for x, y in zip(*np.nonzero(h_grid[ss] % 2 == 1)):
            x0 = cx[0] + x * cx[1]
            x1 = x0 + cx[1]
            y0 = (y * v_step - v_step / 2)
            ax.plot([x0, x1], [y0, y0], color=color, **kwargs)

    if label is not None:
        ax.plot([np.nan], [np.nan], color=color, label=label, **kwargs)


@doc_link(interpolate_nan='neurocarto.util.util_numpy.interpolate_nan')
def plot_channelmap_matrix(ax: Axes,
                           chmap: ChannelMap,
                           data: NDArray[np.float_], *,
                           shank_list: list[int] = None,
                           kernel: int | tuple[int, int] | Callable[[NDArray[np.float_]], NDArray[np.float_]] | None = None,
                           cmap='magma',
                           shank_gap_color: str | None = 'w',
                           **kwargs) -> ScalarMappable:
    """
    Plot channel value.

    :param ax:
    :param chmap: channelmap instance
    :param data: Array[V:float, C] or Array[float, C', (C, V)]
    :param shank_list: show shank in order
    :param kernel: interpolate missing data (NaN) between channels.
        It is pass to {interpolate_nan()}.
        Default is ``(0, 1)`` on row-axis.
    :param cmap: colormap used in ax.imshow(cmap)
    :param shank_gap_color:
    :param kwargs: pass to ax.imshow(kwargs)
    """
    x = channel_coordinate(chmap, 'cr', include_unused=True).astype(float)  # Array[float, E, (S, C, R)]

    if data.ndim == 1:
        x = np.hstack([x, data[:, None]])  # Array[float, E, (S, C, R, V)]
    elif data.ndim == 2:
        c = data[:, 0].astype(int)
        v = data[:, [1]]  # Array[V:float, E, 1]
        x = np.hstack([x[c], v])  # Array[float, E, (S, C, R, V)]
    else:
        raise ValueError()

    return plot_electrode_matrix(
        ax, chmap, x, 'cr',
        shank_list=shank_list,
        kernel=kernel,
        cmap=cmap,
        shank_gap_color=shank_gap_color,
        **kwargs
    )


@doc_link(
    interpolate_nan='neurocarto.util.util_numpy.interpolate_nan',
    DOC=textwrap.dedent(cast_electrode_data.__doc__)
)
def plot_electrode_matrix(ax: Axes,
                          probe: PROBE_TYPE,
                          electrode: NDArray[np.float_],
                          electrode_unit: ELECTRODE_UNIT = 'cr', *,
                          shank_list: list[int] = None,
                          kernel: int | tuple[int, int] | Callable[[NDArray[np.float_]], NDArray[np.float_]] | None = None,
                          cmap='magma',
                          shank_gap_color: str | None = 'w',
                          **kwargs) -> ScalarMappable:
    """
    Plot electrode value.

    {DOC}

    :param ax:
    :param probe: probe type, see {cast_probe_type()}
    :param electrode: electrode data
    :param electrode_unit: electrode value unit
    :param shank_list: show shank in order
    :param kernel: interpolate missing data (NaN) between channels.
        It is pass to {interpolate_nan()}.
        Default is ``(0, 1)`` on row-axis.
    :param cmap: colormap used in ax.imshow(cmap)
    :param shank_gap_color: color of shank gao line. Use None to disable plotting.
    :param kwargs: pass to ax.imshow(kwargs)
    """
    probe: ProbeType = cast_probe_type(probe)
    data = cast_electrode_data(probe, electrode, electrode_unit)

    if kernel is not None:
        data = interpolate_nan(data, kernel)

    if shank_list is not None:
        data = data[shank_list]

    ns, nc, nr = data.shape
    # Array[V, S, C, R] -> Array[V, R, S, C] -> Array[V, R, S*C]
    flat_data = data.transpose(2, 0, 1).reshape(nr, ns * nc)

    _, _, r = np.nonzero(~np.isnan(data))
    r = np.max(r)
    y = r * probe.r_space / 1000
    flat_data = flat_data[np.arange(nr) <= r, :]

    im = ax.imshow(
        flat_data,
        extent=[0, ns * nc, 0, y],
        cmap=cmap, origin='lower', aspect='auto', **kwargs
    )

    if shank_gap_color is not None:
        for i in range(0, ns + 1):
            ax.axvline(i * nc, color=shank_gap_color, lw=2)

    ax.set_xlabel('Shanks')
    ax.set_xticks(np.arange(0, len(shank_list)) * nc + nc // 2, shank_list)

    y_ticks = np.arange(int(math.ceil(y)) + 1)
    ax.set_yticks(y_ticks, y_ticks)
    ax.set_yticks(np.arange(len(y_ticks) * 10) * 0.1, minor=True)
    ax.set_ylabel('Distance from Tip (mm)')

    return im


@doc_link(
    DOC=textwrap.dedent(cast_electrode_data.__doc__)
)
def cast_electrode_curve(probe: PROBE_TYPE,
                         electrode: NDArray[np.float_],
                         electrode_unit: ELECTRODE_UNIT = 'cr', *,
                         kernel: int | NDArray[np.float_] | Literal['norm'] = None) -> tuple[NDArray[np.float_], NDArray[np.float_]]:
    """
    smooth electrode data into 1d value data per shank.

    {DOC}

    :param probe: probe type, see {cast_probe_type()}
    :param electrode: electrode data
    :param electrode_unit: electrode value unit
    :param kernel: smoothing kernel ``Array[float, Y]``, where Y use 1-um bins
    :return: value Array[float, S, R] and y Array[um:float, R]
    """
    probe: ProbeType = cast_probe_type(probe)
    v_step = probe.r_space

    match kernel:
        case None:
            kernel = probe.r_space * 5
            kernel = np.ones((kernel,)) / kernel
        case int(kernel):
            kernel = np.ones((kernel,)) / kernel
        case 'norm':
            from scipy.stats import norm
            s = probe.r_space
            kernel = norm.pdf(np.linspace(-3 * s, 3 * s, 6 * s + 1), 0, s)
        case _ if isinstance(kernel, np.ndarray):
            pass
        case _:
            raise TypeError()

    data = cast_electrode_data(probe, electrode, electrode_unit)

    ns, nc, nr = data.shape
    y = np.arange(0, int(nr * v_step) + 1)  # um
    value = np.zeros((ns, len(y)), dtype=float)

    for s, c, r in zip(*np.nonzero(~np.isnan(data))):
        ri = np.searchsorted(y, r * v_step)
        value[s, ri] += data[s, c, r]

    k2 = len(kernel) // 2
    ks = slice(k2, k2 + len(y))
    for s in range(ns):
        value[s] = np.convolve(value[s], kernel, mode='full')[ks]

    return value, y


@doc_link(
    DOC=textwrap.dedent(cast_electrode_data.__doc__)
)
def plot_electrode_curve(ax: Axes,
                         probe: PROBE_TYPE,
                         electrode: NDArray[np.float_],
                         electrode_unit: ELECTRODE_UNIT = 'cr', *,
                         kernel: int | NDArray[np.float_] | Literal['norm'] = None,
                         shank_list: list[int] = None,
                         height: float | None = None,
                         direction: Literal['left', 'right'] = 'right',
                         vmax: float | None = None,
                         normalize: float | None = 1,
                         shank_width_scale: float = 1,
                         label: str = None,
                         **kwargs):
    """

    {DOC}

    :param ax:
    :param probe: probe type, see {cast_probe_type()}
    :param electrode: electrode data
    :param electrode_unit: electrode value unit
    :param kernel: smoothing kernel ``Array[float, Y]``, where Y use 1-um bins
    :param shank_list: show shank in order
    :param height: y limitation of curve.
    :param direction: the direction of the position value for the curves
    :param vmax: max value.
    :param normalize: normalize the max value to fit into the inter-shank space.
    :param shank_width_scale: scaling the width of a shank for visualizing purpose.
    :param label:
    :param kwargs: pass to ``ax.plot(kwargs)``
    """
    probe: ProbeType = cast_probe_type(probe)
    s_step = probe.s_space / 1000
    h_step = probe.c_space / 1000 * shank_width_scale
    v_step = probe.r_space / 1000

    match direction:
        case 'right':
            xa = probe.n_col_shank * h_step
            xb = s_step - (probe.n_col_shank + 1) * h_step
        case 'left':
            xa = -h_step / 2
            xb = -(s_step - (probe.n_col_shank + 1) * h_step)
        case _:
            raise ValueError()

    data = cast_electrode_data(probe, electrode, electrode_unit)
    ns, nc, nr = data.shape

    value, y = cast_electrode_curve(probe, data, 'raw', kernel=kernel)

    if normalize is not None:
        if vmax is None:
            vmax = np.max(value)
        value = value * normalize / vmax

    if shank_list is None:
        shank_list = list(range(ns))

    y = y / 1000  # mm
    if height is not None:
        yx = y <= height
        y = y[yx]
        value = value[:, yx]

    for si, ss in enumerate(shank_list):
        x = xa + si * s_step + value[ss] * xb
        ax.plot(x, y, **kwargs)

    if label is not None:
        ax.plot([np.nan], [np.nan], label=label, **kwargs)


@doc_link()
def plot_category_area(ax: Axes,
                       probe: PROBE_TYPE,
                       electrode: NDArray[np.int_] | list[ElectrodeDesp], *,
                       color: dict[int, str] = None,
                       **kwargs):
    """
    Plot category blocks for a blueprint *electrode*.

    :param ax:
    :param probe: probe type, see {cast_probe_type()}
    :param electrode: Array[category:int, N], Array[int, N, (S, C, R, category)] or list of ElectrodeDesp
    :param color: category mapping dictionary
    :param kwargs: pass to {plot_electrode_block()}
    """
    probe: ProbeType = cast_probe_type(probe)

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
            NpxProbeDesp.CATE_EXCLUDED: 'pink',
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
