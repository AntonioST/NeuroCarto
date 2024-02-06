from __future__ import annotations

import abc
import contextlib
import io
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import ContextManager, overload, Literal, Any, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from numpy._typing import NDArray

from chmap.views.base import DynamicView
from chmap.views.image_npy import NumpyImageHandler

if TYPE_CHECKING:
    from chmap.probe import ProbeDesp, M, E

__all__ = ['PltImageHandler', 'get_current_plt_image']


class PltImageHandler(NumpyImageHandler, DynamicView, metaclass=abc.ABCMeta):
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

    def __init__(self, *, logger: str | logging.Logger = 'chmap.view.plt'):
        super().__init__(type(self).__name__, logger=logger)

        # pre-set resolution for image_plt.matplotlibrc.
        self.resolution = (5, 5)

    @property
    def title(self) -> str | None:
        return f'<b>{type(self).__name__}</b>'

    @abc.abstractmethod
    def on_probe_update(self, probe: ProbeDesp[M, E], chmap: M | None, e: list[E] | None):
        pass

    def set_image(self, image: NDArray[np.uint] | None):
        super().set_image(image)
        self.update_boundary_transform()

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
                    dpi: float | Literal['figure'] = 'figure',
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

        :param dpi: fig.savefig(dpi)
        :param transparent: fig.savefig(transparent)
        :param rc: default is read from image_plt.matplotlibrc.
        :param kwargs: plt.subplots(**kwargs)
        :return:
        """
        rc_file = kwargs.pop('rc', 'image_plt.matplotlibrc')
        if '/' in rc_file:
            rc_file = Path(rc_file)
        else:
            rc_file = Path(__file__).with_name(rc_file)

        savefig_kw = dict(
            dpi=kwargs.pop('dpi', 'figure'),
            transparent=kwargs.pop('transparent', True),
        )

        with plt.rc_context(fname=rc_file):
            fg, ax = plt.subplots(**kwargs)
            try:
                yield ax
            except BaseException as e:
                self.logger.warning('plot fail', exc_info=e)
                image = None
            else:
                image = get_current_plt_image(fg, **savefig_kw)
            finally:
                plt.close(fg)

        self.set_image(image)


def get_current_plt_image(fg=None, **kwargs) -> NDArray[np.uint]:
    if fg is None:
        fg = plt.gcf()

    # https://stackoverflow.com/a/67823421
    with io.BytesIO() as buff:
        fg.savefig(buff, format='raw', **kwargs)
        buff.seek(0)
        image = np.frombuffer(buff.getvalue(), dtype=np.uint8)

    w, h = fg.canvas.get_width_height()
    return np.flipud(image.view(dtype=np.uint32).reshape((int(h), int(w))))
