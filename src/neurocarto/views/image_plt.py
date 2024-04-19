from __future__ import annotations

import abc
import contextlib
import io
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import ContextManager, overload, Literal, Any, TYPE_CHECKING, NamedTuple, TypedDict

import matplotlib.pyplot as plt  # type: ignore[import]
import numpy as np
from bokeh.models import UIElement
from numpy.typing import NDArray

from neurocarto.config import CartoConfig
from neurocarto.util.utils import doc_link
from neurocarto.views.base import Figure, DynamicView, ViewBase, GlobalStateView
from neurocarto.views.image import ImageView, ImageHandler
from neurocarto.views.image_npy import NumpyImageHandler

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.transforms import BboxBase

__all__ = [
    'PltImageView',
    'Boundary',
    'RC_FILE',
    'get_current_plt_image',
    'get_current_plt_boundary'
]

RC_FILE = Path(__file__).with_name('image_plt.matplotlibrc')


class PltImageState(TypedDict, total=False):
    plt_rc_file: str


class Boundary(NamedTuple):
    """Matplotlib axes boundary in a figure."""

    shape: tuple[int, int]
    """Figure size in pixels."""

    bbox: tuple[float, float, float, float]
    """Axes boundary box (left, bottom, right, top) in ratio."""

    xlim: tuple[float, float]
    """Axes x-axis limits"""

    ylim: tuple[float, float]
    """Axes y-axis limits"""

    def as_um(self) -> Self:
        """Change value unit from mm to um."""
        xlim = self.xlim
        if xlim[1] - xlim[0] < 50:
            xlim = (xlim[0] * 1000, xlim[1] * 1000)

        ylim = self.ylim
        if ylim[1] - ylim[0] < 50:
            ylim = (ylim[0] * 1000, ylim[1] * 1000)

        return self._replace(xlim=xlim, ylim=ylim)

    # ================= #
    # Figure properties #
    # ================= #

    @property
    def fg_width_px(self) -> int:
        """Figure width in pixel"""
        return self.shape[1]

    @property
    def fg_height_px(self) -> int:
        """Figure height in pixel"""
        return self.shape[0]

    @property
    def fg_width(self) -> float:
        """Figure width in um"""
        a, _, c, _ = self.bbox
        return self.ax_width / (c - a)

    @property
    def fg_height(self) -> float:
        """Figure height in um."""
        _, b, _, d = self.bbox
        return self.ax_height / (d - b)

    @property
    def fg_xlim(self) -> tuple[float, float]:
        """Extended x-axis limits for the figure."""
        return self._point_x(0), self._point_x(1)

    @property
    def fg_ylim(self) -> tuple[float, float]:
        """Extended y-axis limits for the figure."""
        return self._point_y(0), self._point_y(1)

    @property
    def fg_extent(self) -> tuple[float, float, float, float]:
        """
        Figure's extent.

        :return: (left, bottom, right, top) in um
        """
        h, w = self.shape
        a, b = self.point(0, 0)
        c, d = self.point(w, h)
        return a, b, c, d

    # =============== #
    # Axes properties #
    # =============== #

    @property
    def ax_width(self) -> float:
        """Axes width in um."""
        return self.xlim[1] - self.xlim[0]

    @property
    def ax_height(self) -> float:
        """Axes height in um."""
        return self.ylim[1] - self.ylim[0]

    @property
    def ax_width_px(self) -> int:
        """Axes width in pixel"""
        w = self.fg_width_px
        l, _, r, _ = self.bbox
        return int(w * (r - l))

    @property
    def ax_height_px(self) -> int:
        """Axes height in pixel"""
        h = self.fg_height_px
        _, b, _, t = self.bbox
        return int(h * (t - b))

    @property
    def ax_extent(self) -> tuple[float, float, float, float]:
        """
        Axes's extent.

        :return: (left, bottom, right, top) in um
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

    # =============== #
    # point transform #
    # =============== #

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
        The center position of the figure.

        :return: (x, y) in um
        """
        h, w = self.shape
        return self.point(w // 2, h // 2)

    def point(self, x: int, y: int) -> tuple[float, float]:
        """
        The point position of the figure.

        :param x: x coordinate on figure in pixels.
        :param y: y coordinate on figure in pixels.
        :return: (x, y) in um
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

    # ======= #
    # Scaling #
    # ======= #

    @property
    def scale(self) -> tuple[float, float]:
        """scale for (x, y) in unit: um/pixels"""
        return self.scale_x, self.scale_y

    @property
    def scale_x(self) -> float:
        return self.fg_width / self.fg_width_px

    @property
    def scale_y(self) -> float:
        return self.fg_height / self.fg_height_px


class PltImageView(ImageView, DynamicView, GlobalStateView[PltImageState], metaclass=abc.ABCMeta):
    """
    Use matplotlib to generate an image.

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

    def __init__(self, config: CartoConfig, *,
                 logger: str | logging.Logger = 'neurocarto.view.plt'):
        super().__init__(config, logger=logger)
        self._plt_rc_file = RC_FILE

    @property
    def name(self) -> str:
        return type(self).__name__

    # ================ #
    # image properties #
    # ================ #

    def set_image_handler(self, image: ImageHandler | None):
        self._image = image

        # disable other components updating
        if image is None:
            self.set_status(None)

    @doc_link()
    def set_image(self, image: NDArray[np.uint] | None,
                  boundary: Boundary = None,
                  offset: float | tuple[float, float] = 0):
        """
        Set image. Due to the figure origin point usually not the origin point in axes,
        you need to provide *boundary* to tell program how to align the image with the probe.

        :param image: image array. May from {get_current_plt_image()}
        :param boundary: image boundary. May from {get_current_plt_boundary()}
        :param offset: x or (x, y) offset. When you don't want the image 100% aligned to the probe origin.
        """
        if image is None:
            self.set_status('clean image ...')
            self.set_image_handler(None)
        else:
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

        self.set_status('updated', decay=3)

    # ============= #
    # UI components #
    # ============= #

    def _setup_render(self, f: Figure, **kwargs):
        self.setup_image(f)

    def _setup_title(self, **kwargs) -> list[UIElement]:
        # we do not need other components. just origin title.
        return ViewBase._setup_title(self, **kwargs)

    def _setup_content(self, **kwargs) -> list[UIElement]:
        return []

    # ========= #
    # load/save #
    # ========= #

    def restore_state(self, state: PltImageState):
        if (rc_file := state.get('plt_rc_file', None)) is not None:
            self.logger.debug('read rc_file %s', rc_file)
            if (rc_file := Path(rc_file)).exists():
                self._plt_rc_file = rc_file
            else:
                self.logger.warning('rc_file not existed: %s', rc_file)

    # ================ #
    # updating methods #
    # ================ #

    def start(self):
        self.visible = False
        self.restore_global_state()

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

    # noinspection PyIncorrectDocstring
    @contextlib.contextmanager
    @doc_link()
    def plot_figure(self, **kwargs) -> ContextManager[Axes]:
        """
        A context manager of matplotlib axes.

        .. code-block:: python

            with self.plot_figure() as ax:
                ax.plot(...)

        Once context closed, call {#set_image()} with parameters *image*, *boundary* and *offset* filled.

        If a ``KeyboardInterrupt`` is raised, capture and clear the image.

        It an error except ``KeyboardInterrupt`` is raised. reraise it and do nothing.

        :param transparent: fig.savefig(transparent)
        :param rc: default is read from image_plt.matplotlibrc.
        :param offset: see *offset* in {#set_image()}
        :param kwargs: plt.subplots(kwargs)
        :return: a context manger carries {Axes}
        """
        self.set_status('computing...')

        rc_file = kwargs.pop('rc', None)
        if rc_file is None:
            rc_file = self._plt_rc_file
        elif isinstance(rc_file, str):
            if '/' in rc_file:
                rc_file = Path(rc_file)
            else:
                rc_file = Path(__file__).with_name(rc_file)
        elif isinstance(rc_file, Path):
            pass
        else:
            raise TypeError()

        savefig_kw = dict(
            transparent=kwargs.pop('transparent', True),
        )

        offset = kwargs.pop('offset', 0)

        ax: Axes
        with plt.rc_context(fname=rc_file):
            fg, ax = plt.subplots(**kwargs)
            try:
                yield ax
            except KeyboardInterrupt:
                self.logger.info('plot interrupted')
                image = None
                boundary = None
            except BaseException as e:
                self.set_status('computing failed')
                self.logger.warning('plot fail', exc_info=e)
                return
            else:
                self.set_status('computing done')
                boundary = get_current_plt_boundary(ax)
                image = get_current_plt_image(fg, **savefig_kw)
            finally:
                plt.close(fg)

        self.set_image(image, boundary, offset)


def get_current_plt_image(fg=None, **kwargs) -> NDArray[np.uint]:
    """
    Save matplotlib figure into numpy array.

    :param fg: matplotlib Figure.
    :param kwargs: ``fg.savefig(kwargs)``, except parameters *format* and *dpi*.
    :return: a numpy array.
    """
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
    """
    Get Axes boundary.

    :param ax: matplotlib Axes
    :return: a boundary
    """
    if ax is None:
        ax = plt.gca()

    bbox: BboxBase = ax.get_position()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    w, h = ax.figure.canvas.get_width_height()
    return Boundary((h, w), tuple(bbox.extents), xlim, ylim)
