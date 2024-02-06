from __future__ import annotations

import contextlib
import io
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import ContextManager, overload, Literal, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray

from chmap.views.base import DynamicView
from chmap.views.image import ImageHandler

__all__ = ['PltImageHandler']


class PltImageHandler(ImageHandler, DynamicView):
    logger: logging.Logger

    def __init__(self, *, logger: str = 'chmap.view.plt'):
        if isinstance(logger, str):
            self.logger = logging.getLogger(logger)
        elif isinstance(logger, logging.Logger):
            self.logger = logger

        if (logger := self.logger) is not None:
            logger.debug('init()')

        self.filename = type(self).__name__
        self._image: NDArray[np.uint] | None = None
        self._resolution = (5, 5)

    @property
    def title(self) -> str | None:
        return f'<b>{type(self).__name__}</b>'

    def __len__(self) -> int:
        if (image := self._image) is not None and image.ndim == 3:
            return len(image)
        else:
            return 1

    def __getitem__(self, index: int) -> NDArray[np.uint] | None:
        if (image := self._image) is None:
            return None
        elif image.ndim == 3:
            return image[index]
        else:
            return image

    @property
    def resolution(self) -> tuple[float, float]:
        return self._resolution

    @resolution.setter
    def resolution(self, resolution: float | tuple[float, float]):
        if not isinstance(resolution, tuple):
            resolution = float(resolution)
            resolution = (resolution, resolution)
        self._resolution = resolution

    @property
    def width(self) -> float:
        if (image := self._image) is None:
            return 0

        r = self.resolution[0]
        if image.ndim == 3:
            return image.shape[2] * r
        else:
            return image.shape[1] * r

    @property
    def height(self) -> float:
        if (image := self._image) is None:
            return 0

        r = self.resolution[1]
        if image.ndim == 3:
            return image.shape[1] * r
        else:
            return image.shape[0] * r

    @property
    def image(self) -> NDArray[np.uint] | None:
        return self._image

    def set_image(self, image: NDArray[np.uint] | None):
        self._image = image

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
                self.set_image(None)
                return
            else:
                # https://stackoverflow.com/a/67823421
                with io.BytesIO() as buff:
                    fg.savefig(buff, format='raw', **savefig_kw)
                    buff.seek(0)
                    image = np.frombuffer(buff.getvalue(), dtype=np.uint8)

                w, h = fg.canvas.get_width_height()
                image = image.view(dtype=np.uint32).reshape((int(h), int(w)))
                self.set_image(np.flipud(image))
