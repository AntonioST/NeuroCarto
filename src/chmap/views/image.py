import abc
import sys
from typing import TypedDict, Final

import numpy as np
from bokeh.models import ColumnDataSource, GlyphRenderer, Slider, UIElement, Div
from bokeh.plotting import figure as Figure
from numpy.typing import NDArray

from chmap.config import ChannelMapEditorConfig
from chmap.util.utils import is_recursive_called
from chmap.views.base import BoundView, StateView, BoundaryState

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

__all__ = ['ImageView', 'ImageViewState', 'ImageHandler']


class ImageViewState(TypedDict):
    filename: str
    index: int | None
    image_dx: float
    image_dy: float
    image_sx: float
    image_sy: float
    image_rt: float


class ImageHandler(metaclass=abc.ABCMeta):
    filename: str
    resolution: tuple[float, float]  # (x, y)
    width: float
    height: float

    @property
    @abc.abstractmethod
    def __len__(self) -> int:
        pass

    @abc.abstractmethod
    def __getitem__(self, index: int) -> NDArray[np.uint]:
        pass

    @classmethod
    def from_numpy(cls, filename: str, image: NDArray[np.uint] = None, *,
                   resolution: tuple[float, float] = (1, 1)) -> Self:
        from .image_npy import NumpyImageHandler
        return NumpyImageHandler(filename, image, resolution=resolution)

    @classmethod
    def from_file(cls, filename: str, *,
                  resolution: tuple[float, float] = (1, 1)) -> Self:
        """

        :param filename: any Pillow support image format
        :param resolution:
        :return:
        """
        from PIL import Image
        from .image_npy import NumpyImageHandler
        image = np.asarray(Image.open(filename, mode='r'))
        return NumpyImageHandler(filename, image, resolution=resolution)

    @classmethod
    def from_tiff(cls, filename: str, *,
                  resolution: tuple[float, float] = (1, 1)) -> Self:
        import tifffile
        from .image_npy import NumpyImageHandler
        image = tifffile.TiffFile(filename, mode='r').asarray()  # TODO memmap?
        return NumpyImageHandler(filename, image, resolution=resolution)


class ImageView(BoundView, StateView[ImageViewState]):
    image: Final[ImageHandler]

    data_image: ColumnDataSource

    render_image: GlyphRenderer

    def __init__(self, config: ChannelMapEditorConfig, image: ImageHandler):
        super().__init__(config)

        self.image = image
        self.data_brain = ColumnDataSource(data=dict(image=[], x=[], y=[], dw=[], dh=[]))

        self._index: int = 0

    # ========== #
    # properties #
    # ========== #

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
            self.render_boundary.visible = v
        except AttributeError:
            pass

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

    # ================= #
    # render components #
    # ================= #

    def plot(self, f: Figure, palette: str = 'Greys256',
             boundary_color: str = 'black',
             boundary_desp: str = 'drag image',
             **kwargs):
        self.render_image = f.image(
            'image', x='x', y='y', dw='dw', dh='dh', source=self.data_image,
            palette=palette, level="image", global_alpha=0.5, syncable=False,
        )

        super().plot(f, boundary_color=boundary_color, boundary_desp=boundary_desp, **kwargs)

    # ============= #
    # UI components #
    # ============= #

    index_slider: Slider

    def setup(self, slider_width: int = 300) -> list[UIElement]:
        self.index_slider = Slider(
            start=0,
            end=len(self.image),
            step=1,
            value=0,
            title='Index',
            width=slider_width,
            align='end',
            disabled=len(self.image) == 1
        )
        self.index_slider.on_change('value', self.on_index_changed)

        return [
            Div(text=f"<b>Image</b> {self.image.filename}"),
            self.index_slider,
            *self.setup_slider(slider_width=slider_width)
        ]

    # noinspection PyUnusedLocal
    def _on_image_selected(self, prop: str, old: str, s: str):
        if is_recursive_called():
            return

    # noinspection PyUnusedLocal
    def on_index_changed(self, prop: str, old: int, s: int):
        if is_recursive_called():
            return

        self.update_image(s)

    # ========= #
    # load/save #
    # ========= #

    def is_auto_saving(self) -> bool:
        return True

    def save_state(self) -> list[ImageViewState]:
        boundary = self.get_boundary_state()

        return [ImageViewState(
            filename=self.image.filename,
            index=self._index,
            image_dx=boundary['dx'],
            image_dy=boundary['dy'],
            image_sx=boundary['sx'],
            image_sy=boundary['sy'],
            image_rt=boundary['rt'],
        )]

    def restore_state(self, state: ImageViewState | list[ImageViewState]):
        if isinstance(state, list):
            for _state in state:  # type:ImageViewState
                if _state['filename'] == self.image.filename:
                    state = _state
                    break
            else:
                return
        elif state['filename'] != self.image.filename:
            raise RuntimeError()

        self.update_boundary_transform(p=(state['image_dx'], state['image_dy']),
                                       s=(state['image_sx'], state['image_sx']),
                                       rt=state['image_rt'])

        index = state['index']
        if index is None:
            index = 0

        self.update_image(index)

    # ================ #
    # updating methods #
    # ================ #

    def update(self):
        self.on_reset_boundary()

    def _update_boundary_transform(self, state: BoundaryState):
        super()._update_boundary_transform(state)

        try:
            image = self.image[self._index]
        except IndexError:
            pass
        else:
            self.update_image(image)

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
            self.render_boundary.visible = False
            self.data_brain.data = dict(image=[], dw=[], dh=[], x=[], y=[])
        else:
            self.render_boundary.visible = self.render_image.visible
            self.data_brain.data = self.transform_image_data(image_data)
