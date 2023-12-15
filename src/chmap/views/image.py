import abc
import math
from pathlib import Path
from typing import TypedDict, Final

import numpy as np
from bokeh.models import ColumnDataSource, GlyphRenderer, Select, Slider, UIElement
from bokeh.plotting import figure as Figure
from numpy.typing import NDArray
from scipy.ndimage import rotate

from chmap.util.utils import is_recursive_called


class ImageViewState(TypedDict):
    filename: Path
    index: int | None
    image_dx: float
    image_dy: float
    image_sx: float
    image_sy: float
    image_rt: float


class ImageOper(metaclass=abc.ABCMeta):

    @property
    @abc.abstractmethod
    def filename(self) -> Path:
        pass

    @property
    @abc.abstractmethod
    def __len__(self) -> int:
        pass

    @abc.abstractmethod
    def __getitem__(self, index: int) -> NDArray[np.uint]:
        pass

    @property
    @abc.abstractmethod
    def width(self) -> float:
        pass

    @property
    @abc.abstractmethod
    def height(self) -> float:
        pass

    @property
    @abc.abstractmethod
    def resolution(self) -> tuple[float, int]:
        pass


class ImageView:
    image: Final[ImageOper]

    data_image: ColumnDataSource
    data_image_boundary: ColumnDataSource

    render_image: GlyphRenderer
    render_image_boundary: GlyphRenderer

    def __init__(self, image: ImageOper):
        self.image = image
        self.data_brain = ColumnDataSource(data=dict(image=[], x=[], y=[], dw=[], dh=[]))
        self.data_brain_boundary = ColumnDataSource(data=dict(x=[], y=[], w=[], h=[], r=[]))

        self._index: int = 0
        self._dx: float = 0
        self._dy: float = 0
        self._sx: float = 1
        self._sy: float = 1
        self._rt: float = 0

    # ========= #
    # load/save #
    # ========= #

    def save_state(self) -> ImageViewState:
        return ImageViewState(
            filename=self.image.filename,
            index=self._index,
            image_dx=self._dx,
            image_dy=self._dy,
            image_sx=self._sx,
            image_sy=self._sy,
            image_rt=self._rt,
        )

    def restore_state(self, state: ImageViewState):
        pass

    # ============= #
    # UI components #
    # ============= #

    image_select: Select
    index_slider: Slider
    imr_slider: Slider
    ims_slider: Slider

    def setup(self, width: int = 300) -> list[UIElement]:
        pass

    # noinspection PyUnusedLocal
    def _on_image_selected(self, prop: str, old: str, s: str):
        if is_recursive_called():
            return

    # noinspection PyUnusedLocal
    def _on_index_changed(self, prop: str, old: int, s: int):
        if is_recursive_called():
            return

        self.update_image(s)

    # noinspection PyUnusedLocal
    def _on_image_rotate(self, prop: str, old: int, s: int):
        if is_recursive_called():
            return

        self.update_image_rotate(s)

    # noinspection PyUnusedLocal
    def _on_image_scale(self, prop: str, old: float, s: float):
        if is_recursive_called():
            return

        self.update_image_scale(math.pow(10, s))

    # noinspection PyUnusedLocal
    def _on_boundary_change(self, prop: str, old: dict, value: dict[str, list[float]]):
        if is_recursive_called():
            return

        try:
            x = float(value['x'][0])
        except IndexError:
            return

        y = float(value['y'][0])
        w = float(value['w'][0])
        h = float(value['h'][0])
        x -= w / 2
        y -= h / 2
        sx = w / self.width
        sy = h / self.height

        try:
            self.ims_slider.value = round(math.log10(min(sx, sy)), 2)
        except AttributeError:
            pass

        self.update_image_transform(p=(x, y), s=(sx, sy))

    def reset_imr(self):
        try:
            self.imr_slider.value = 0
        except AttributeError:
            self.update_image_rotate(0)

    def reset_ims(self):
        try:
            self.ims_slider.value = 0
        except AttributeError:
            self.update_image_scale(1)

    # ================= #
    # render components #
    # ================= #

    @property
    def visible(self) -> bool:
        try:
            return self.render_image.visible
        except AttributeError:
            return False

    @visible.setter
    def visible(self, v: bool):
        try:
            self.render_image.visible = v
            self.render_image_boundary.visible = v
        except AttributeError:
            pass

    def plot(self, f: Figure, palette: str = 'Greys256'):
        self.render_image = f.image(
            'image', x='x', y='y', dw='dw', dh='dh', source=self.data_image,
            palette=palette, level="image", global_alpha=0.5, syncable=False,
        )
        self.render_image_boundary = f.rect(
            'x', 'y', 'w', 'h', 'r', source=self.data_image_boundary,
            fill_alpha=0, angle_units='deg',
        )
        self.render_image_boundary.on_change('data', self._on_boundary_change)

    def boundary_tool(self):
        from bokeh.models import tools
        return tools.BoxEditTool(
            description='drag atlas brain image',
            renderers=[self.render_image_boundary], num_objects=1
        )

    # ============================= #
    # properties and update methods #
    # ============================= #

    @property
    def width(self) -> float:
        try:
            return self.image.width
        except TypeError:
            return 0

    @property
    def height(self) -> float:
        try:
            return self.image.height
        except TypeError:
            return 0

    @property
    def image_pos(self) -> tuple[float, float]:
        return self._dx, self._dy

    def update_image_pos(self, x: float, y: float):
        self.update_image_transform(p=(x, y))

    @property
    def image_scale(self) -> tuple[float, float]:
        return self._sx, self._sy

    def update_image_scale(self, s: float | tuple[float, float]):
        if isinstance(s, tuple):
            sx, sy = s
        else:
            sx = sy = float(s)

        self.update_image_transform(s=(sx, sy))

    @property
    def image_rotate(self) -> float:
        return self._rt

    def update_image_rotate(self, rt: float):
        self.update_image_transform(rt=rt)

    def update_image_transform(self, *,
                               p: tuple[float, float] = None,
                               s: float | tuple[float, float] = None,
                               rt: float = None,
                               update_image=True):
        if is_recursive_called():
            return

        if p is not None:
            self._dx, self._dy = p

        if s is not None:
            if isinstance(s, tuple):
                self._sx, self._sy = s
            else:
                self._sx = self._sy = float(s)

        if rt is not None:
            self._rt = rt

        w = self.width * self._sx
        h = self.height * self._sy
        x = self._dx
        y = self._dy

        try:
            self.ims_slider.value = round(math.log10(min(self._sx, self._sy)), 2)
        except AttributeError:
            pass

        try:
            self.imr_slider.value = self._rt
        except AttributeError:
            pass

        self.data_brain_boundary.data = dict(
            x=[x + w / 2], y=[y + h / 2], w=[w], h=[h], r=[self._rt]
        )

        if update_image:
            self.update_image(self.image[self._index])

    def update_image(self, image_data: int | NDArray[np.uint] | None):
        if is_recursive_called():
            return

        if isinstance(image_data, int) or isinstance(image_data, np.integer):
            self._index = index = int(image_data)
            image_data = self.image[index]

            try:
                self.index_slider.value = index
            except AttributeError:
                pass

        if image_data is None:
            self.data_brain.data = dict(image=[], dw=[], dh=[], x=[], y=[])
            self.data_brain_boundary.data = dict(x=[], y=[], w=[], h=[], r=[])
        else:
            w = self.image.width * self._sx
            h = self.image.height * self._sy
            x = self._dx
            y = self._dy
            image = np.flipud(image_data)

            if self._rt != 0:
                image = rotate(image, -self._rt, reshape=False)

            self.data_brain.data = dict(
                image=[image],
                dw=[w], dh=[h], x=[x], y=[y]
            )
