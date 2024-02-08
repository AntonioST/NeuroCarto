from __future__ import annotations

import abc
import contextlib
import io
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import ContextManager, overload, Literal, Any, TYPE_CHECKING, NamedTuple

import matplotlib.pyplot as plt
import numpy as np
from bokeh.models import UIElement
from bokeh.plotting import figure as Figure
from numpy.typing import NDArray

from chmap.config import ChannelMapEditorConfig
from chmap.util.bokeh_app import run_timeout
from chmap.views.base import DynamicView, ViewBase
from chmap.views.image import ImageView, ImageHandler
from chmap.views.image_npy import NumpyImageHandler

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.transforms import BboxBase

__all__ = [
    'PltImageView',
    'Boundary',
    'get_current_plt_image',
    'get_current_plt_boundary'
]


class Boundary(NamedTuple):
    shape: tuple[int, int]
    bbox: tuple[float, float, float, float]  # boundary (left, bottom, right, top) in ratio.
    xlim: tuple[float, float]
    ylim: tuple[float, float]

    def as_um(self) -> Boundary:
        xlim = self.xlim
        if xlim[1] - xlim[0] < 50:
            xlim = (xlim[0] * 1000, xlim[1] * 1000)

        ylim = self.ylim
        if ylim[1] - ylim[0] < 50:
            ylim = (ylim[0] * 1000, ylim[1] * 1000)

        return self._replace(xlim=xlim, ylim=ylim)

    @property
    def fg_width_px(self) -> int:
        return self.shape[1]

    @property
    def fg_height_px(self) -> int:
        return self.shape[0]

    @property
    def ax_width(self) -> float:
        return self.xlim[1] - self.xlim[0]

    @property
    def ax_height(self) -> float:
        return self.ylim[1] - self.ylim[0]

    @property
    def origin_px(self) -> tuple[int, int]:
        """
        origin point position in figure.
        :return: (x, y) pixel
        """
        return self.point_px(0, 0)

    def point_px(self, x: float, y: float) -> tuple[int, int]:
        """
        point position in figure.

        :param x: x coordinate in axes
        :param y: y coordinate in axes
        :return: (x, y) pixel
        """
        h, w = self.shape
        x0, x1 = self.xlim
        y0, y1 = self.ylim
        a, b, c, d = self.bbox
        rx = (x - x0) * (c - a) / (x1 - x0) + a
        ry = (y - y0) * (d - b) / (y1 - y0) + c
        return int(rx * w), int(ry * h)

    @property
    def center(self) -> tuple[float, float]:
        """
        the position of the figure's center point.
        :return: (x, y)
        """
        h, w = self.shape
        return self.point(w // 2, h // 2)

    def point(self, x: int, y: int) -> tuple[float, float]:
        """

        :param x: x coordinate on figure in pixels.
        :param y: y coordinate on figure in pixels.
        :return: (x, y) in axis
        """
        h, w = self.shape
        x0, x1 = self.xlim
        y0, y1 = self.ylim
        a, b, c, d = self.bbox
        rx = (x / w - a) * (x1 - x0) / (c - a) + x0
        ry = (y / h - b) * (y1 - y0) / (d - b) + y0
        return rx, ry

    def _point_x(self, x: float) -> float:
        x0, x1 = self.xlim
        a, _, c, _ = self.bbox
        return (x - a) * (x1 - x0) / (c - a) + x0

    def _point_y(self, y: float) -> float:
        y0, y1 = self.ylim
        _, b, _, d = self.bbox
        return (y - b) * (y1 - y0) / (d - b) + y0

    @property
    def ax_extent(self) -> tuple[float, float, float, float]:
        """

        :return: (left, bottom, right, top)
        """
        return self.xlim[0], self.ylim[0], self.xlim[1], self.ylim[1]

    @property
    def ax_extent_px(self) -> tuple[int, int, int, int]:
        """

        :return: (left, bottom, right, top) in pixels
        """
        w = self.fg_width_px
        h = self.fg_height_px
        l, b, r, t = self.bbox
        return int(w * l), int(h * b), int(w * r), int(t * h)

    @property
    def fg_extent(self) -> tuple[float, float, float, float]:
        h, w = self.shape
        a, b = self.point(0, 0)
        c, d = self.point(w, h)
        return a, b, c, d

    @property
    def fg_width(self) -> float:
        a, _, c, _ = self.bbox
        return self.ax_width / (c - a)

    @property
    def fg_height(self) -> float:
        _, b, _, d = self.bbox
        return self.ax_height / (d - b)

    @property
    def fg_xlim(self) -> tuple[float, float]:
        return self._point_x(0), self._point_x(1)

    @property
    def fg_ylim(self) -> tuple[float, float]:
        return self._point_y(0), self._point_y(1)

    @property
    def scale(self) -> tuple[float, float]:
        return self.scale_x, self.scale_y

    @property
    def scale_x(self) -> float:
        return self.fg_width / self.fg_width_px

    @property
    def scale_y(self) -> float:
        return self.fg_height / self.fg_height_px


class PltImageView(ImageView, DynamicView, metaclass=abc.ABCMeta):
    """
    Use matplotlib to generate image.

    Example:

    .. code-block ::

        class PlotChannelMap(PltImageHandler):
            def on_probe_update(self, probe, chmap, e):
                if chmap is not None:
                    self.plot_channelmap(chmap)
                else:
                    self.set_image(None)

            def plot_channelmap(self, m):
                from chmap.probe_npx import plot

                with self.plot_figure() as ax:
                    plot.plot_channelmap_block(ax, chmap=m)
                    plot.plot_probe_shape(ax, m, color='k')
    """

    def __init__(self, config: ChannelMapEditorConfig, *,
                 logger: str | logging.Logger = 'chmap.view.plt'):
        super().__init__(config, logger=logger)

    @property
    def name(self) -> str:
        return type(self).__name__

    # ================ #
    # image properties #
    # ================ #

    def set_image_handler(self, image: ImageHandler | None):
        self._image = image
        if image is None:
            self.set_status(None)

    def set_image(self, image: NDArray[np.uint] | None,
                  boundary: Boundary = None,
                  offset: float | tuple[float, float] = 0):
        """
        Set image. Due to the figure origin point usually not the origin point in axes,
        you need to provide *boundary* to tell program how to align the image.

        :param image: image array
        :param boundary: image boundary
        :param offset: x or (x, y) offset. Once you don't want figure 100% aligned.
        """
        self.set_status('update image ...')
        self.set_image_handler(NumpyImageHandler(image))

        if boundary is None:
            self.update_boundary_transform()
        else:
            boundary = boundary.as_um()
            center = boundary.center
            if isinstance(offset, tuple):
                center = (center[0] + offset[0], center[1] + offset[1])
            else:
                center = (center[0] + offset, center[1])
            self.update_boundary_transform(p=center, s=boundary.scale)

        self.set_status('updated')
        run_timeout(3000, self.set_status, None)

    # ============= #
    # UI components #
    # ============= #

    def _setup_render(self, f: Figure, **kwargs):
        self.setup_image(f)

    def _setup_title(self, **kwargs) -> list[UIElement]:
        return ViewBase._setup_title(self, **kwargs)

    def _setup_content(self, **kwargs) -> list[UIElement]:
        return []

    # ================ #
    # updating methods #
    # ================ #

    def start(self):
        self.visible = False

    @overload
    def plot_figure(self,
                    nrows: int = 1, ncols: int = 1, *,
                    sharex: bool | Literal["none", "all", "row", "col"] = False,
                    sharey: bool | Literal["none", "all", "row", "col"] = False,
                    squeeze: bool = True,
                    width_ratios: Sequence[float] | None = None,
                    height_ratios: Sequence[float] | None = None,
                    subplot_kw: dict[str, Any] | None = None,
                    gridspec_kw: dict[str, Any] | None = None,
                    offset: float | tuple[float, float] = 0,
                    transparent: bool = True,
                    rc: str = None,
                    **kwargs) -> ContextManager[Axes]:
        pass

    @contextlib.contextmanager
    def plot_figure(self, **kwargs) -> ContextManager[Axes]:
        """
        A context manager of matplotlib axes.

        >>> with self.plot_figure() as ax:
        ...     ax.plot(...)

        Once context closed, call `set_image()` with parameters *image* and *boundary* filled.

        :param transparent: fig.savefig(transparent)
        :param rc: default is read from image_plt.matplotlibrc.
        :param kwargs: plt.subplots(**kwargs)
        :return: a context manger of Axes
        """
        self.set_status('computing...')

        rc_file = kwargs.pop('rc', 'image_plt.matplotlibrc')
        if '/' in rc_file:
            rc_file = Path(rc_file)
        else:
            rc_file = Path(__file__).with_name(rc_file)

        savefig_kw = dict(
            transparent=kwargs.pop('transparent', True),
        )

        offset = kwargs.pop('offset', 0)

        ax: Axes
        with plt.rc_context(fname=rc_file):
            fg, ax = plt.subplots(**kwargs)
            try:
                yield ax
            except BaseException as e:
                self.set_status('computing failed')
                self.logger.warning('plot fail', exc_info=e)
                image = None
                boundary = None
            else:
                self.set_status('computing done')
                boundary = get_current_plt_boundary(ax)
                image = get_current_plt_image(fg, **savefig_kw)
            finally:
                plt.close(fg)

        self.set_image(image, boundary, offset)


def get_current_plt_image(fg=None, **kwargs) -> NDArray[np.uint]:
    if fg is None:
        fg = plt.gcf()

    # https://stackoverflow.com/a/67823421
    with io.BytesIO() as buff:
        # force dpi as 'figure'. Otherwise, we will get wrong w and h.
        fg.savefig(buff, format='raw', dpi='figure', **kwargs)
        buff.seek(0)
        image = np.frombuffer(buff.getvalue(), dtype=np.uint8)

    w, h = fg.canvas.get_width_height()
    return np.flipud(image.view(dtype=np.uint32).reshape((int(h), int(w))))


def get_current_plt_boundary(ax: Axes = None) -> Boundary:
    if ax is None:
        ax = plt.gca()

    bbox: BboxBase = ax.get_position()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    w, h = ax.figure.canvas.get_width_height()
    return Boundary((h, w), tuple(bbox.extents), xlim, ylim)
