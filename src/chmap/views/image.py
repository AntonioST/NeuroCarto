import abc
import logging
import sys
from pathlib import Path
from typing import TypedDict

import numpy as np
from bokeh.models import ColumnDataSource, GlyphRenderer, Slider, UIElement, Div, TextInput, Tooltip
from bokeh.plotting import figure as Figure
from numpy.typing import NDArray

from chmap.config import ChannelMapEditorConfig
from chmap.util.bokeh_app import run_later
from chmap.util.bokeh_util import SliderFactory, is_recursive_called, PathAutocompleteInput, as_callback
from chmap.views.base import BoundView, StateView, BoundaryState

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

__all__ = ['ImageView', 'ImageViewState', 'ImageHandler', 'FileImageView']


class ImageViewState(TypedDict):
    filename: str
    index: int
    resolution_w: float
    resolution_h: float
    image_dx: float
    image_dy: float
    image_rt: float


class ImageHandler(metaclass=abc.ABCMeta):
    filename: str
    width: float
    height: float

    @abc.abstractmethod
    def __len__(self) -> int:
        pass

    @abc.abstractmethod
    def __getitem__(self, index: int) -> NDArray[np.uint]:
        pass

    @property
    @abc.abstractmethod
    def resolution(self) -> tuple[float, float]:
        pass

    @resolution.setter
    @abc.abstractmethod
    def resolution(self, value: float | tuple[float, float]):
        pass

    @classmethod
    def from_numpy(cls, filename: str | Path, image: NDArray[np.uint] = None) -> Self:
        logger = logging.getLogger('chmap.image')
        from .image_npy import NumpyImageHandler
        logger.debug('from numpy %s', image.shape)
        return NumpyImageHandler(filename, image)

    @classmethod
    def from_file(cls, filename: str | Path) -> Self:
        """

        :param filename: any Pillow support image format
        :return:
        """
        logger = logging.getLogger('chmap.image')
        from PIL import Image
        from .image_npy import NumpyImageHandler
        logger.debug('from file %s', filename)
        image = np.asarray(Image.open(filename, mode='r'))
        w, h, _ = image.shape
        image = np.flipud(image.view(dtype=np.uint32).reshape((w, h)))
        logger.debug('as image %s', image.shape)
        return NumpyImageHandler(filename, image)

    @classmethod
    def from_tiff(cls, filename: str | Path) -> Self:
        logger = logging.getLogger('chmap.image')
        import tifffile
        from .image_npy import NumpyImageHandler
        logger.debug('from file %s', filename)
        image = tifffile.TiffFile(filename, mode='r').asarray()  # TODO memmap?
        logger.debug('as image %s', image.shape)
        return NumpyImageHandler(filename, image)


class ImageView(BoundView, metaclass=abc.ABCMeta):
    data_image: ColumnDataSource
    render_image: GlyphRenderer

    def __init__(self, config: ChannelMapEditorConfig, image: ImageHandler = None):
        if self.logger is None:
            self.logger = logging.getLogger('chmap.view.image')

        super().__init__(config)

        self.data_image = ColumnDataSource(data=dict(image=[], x=[], y=[], dw=[], dh=[]))

        self._image = image
        self._index: int = 0

    @property
    def name(self) -> str:
        return 'Image'

    # ================ #
    # image properties #
    # ================ #

    @property
    def image(self) -> ImageHandler | None:
        return self._image

    @property
    def width(self) -> float:
        try:
            return self.image.width
        except (TypeError, AttributeError):
            return 0

    @property
    def height(self) -> float:
        try:
            return self.image.height
        except (TypeError, AttributeError):
            return 0

    def set_image_handler(self, image: ImageHandler | None):
        self._image = image

        if (slider := self.index_slider) is not None:
            if image is None or (n_image := len(image)) == 1:
                slider.end = 1
                slider.disabled = True
            else:
                slider.end = n_image
                slider.disabled = False

            if self.visible:
                slider.visible = not slider.disabled

    def save_current_state(self) -> ImageViewState | None:
        if (image := self.image) is None:
            return None

        self.logger.debug('save(%s)', image.filename)
        boundary = self.get_boundary_state()
        resolution = image.resolution

        return ImageViewState(
            filename=image.filename,
            index=self._index,
            resolution_w=resolution[0],
            resolution_h=resolution[1],
            image_dx=boundary['dx'],
            image_dy=boundary['dy'],
            image_rt=boundary['rt'],
        )

    def restore_current_state(self, state: ImageViewState):
        if (image := self.image) is None:
            return

        image.resolution = (state['resolution_w'], state['resolution_h'])
        self.update_boundary_transform(p=(state['image_dx'], state['image_dy']), rt=state['image_rt'])
        self.update_image(state['index'])

    # ============= #
    # UI components #
    # ============= #

    resolution_input: TextInput

    def _setup_render(self, f: Figure,
                      boundary_color: str = 'black',
                      boundary_desp: str = 'drag image',
                      **kwargs):
        # renders
        self.render_image = f.image_rgba(
            'image', x='x', y='y', dw='dw', dh='dh', source=self.data_image,
            global_alpha=0.5, syncable=False,
        )

        self.setup_boundary(f, boundary_color=boundary_color, boundary_desp=boundary_desp)

    def _setup_title(self, **kwargs) -> list[UIElement]:
        ret = super()._setup_title(**kwargs)

        if (image := self.image) is not None:
            self.view_title.text = f'<b>Image</b> {image.filename}'

        ret.append(Div(text='resolution:'))

        self.resolution_input = TextInput(max_width=100, description=Tooltip(content='image resolution. format "10" or "10,10"'))
        self.resolution_input.on_change('value', as_callback(self.on_resolution_changed))
        ret.append(self.resolution_input)

        return ret

    index_slider: Slider = None

    def _setup_content(self, slider_width: int = 300,
                       **kwargs) -> list[UIElement]:
        from bokeh.layouts import row

        ret = []
        new_slider = SliderFactory(width=slider_width, align='end')
        self.index_slider = new_slider('Index', (0, 1, 1, 0), self.on_index_changed)
        self.index_slider.visible = False
        ret.append(self.index_slider)
        ret.append(row(*self.setup_rotate_slider(new_slider=new_slider)))
        return ret

    def on_index_changed(self, s: int):
        if is_recursive_called():
            return

        self.update_image(s)

    def get_resolution_value(self, r: str = None) -> tuple[float, float] | None:
        if r is None:
            r = self.resolution_input.value

        if r == '':
            return None

        try:
            if ',' in r:
                f = r.partition(',')
                return float(f[0].strip()), float(f[2].strip())
            else:
                f = float(r)
                return f, f
        except ValueError:
            return None

    def on_resolution_changed(self, r: str | float):
        if is_recursive_called() or r == '':
            return

        if isinstance(r, str):
            f = self.get_resolution_value(r)
        else:
            f = float(r)
            f = (f, f)

        if f is None:
            if (image := self.image) is None:
                self.resolution_input.value = ''
            else:
                r = image.resolution
                self.resolution_input.value = f'{r[0]},{r[1]}'
        else:
            if (image := self.image) is not None:
                image.resolution = f

            self.update_boundary_transform(s=1)

    # ================ #
    # updating methods #
    # ================ #

    def start(self):
        if self.image is not None:
            self.visible = True
            self.set_image_handler(self.image)  # trigger setter
            self.on_reset_boundary()
        else:
            self.visible = False

    def _update_boundary_transform(self, state: BoundaryState):
        super()._update_boundary_transform(state)

        try:
            image = self.image[self._index]
        except (IndexError, TypeError):
            pass
        else:
            self.update_image(image)

    def update_image(self, image_data: int | NDArray[np.uint] | None):
        if is_recursive_called():
            return

        if isinstance(image_data, int) or isinstance(image_data, np.integer):
            self._index = index = int(image_data)

            try:
                image_data = self.image[index]
            except TypeError:
                return

            try:
                self.index_slider.value = index
            except (AttributeError, TypeError):
                pass

        if image_data is None:
            self.data_image.data = dict(image=[], dw=[], dh=[], x=[], y=[])
            self.visible = False
        else:
            self.data_image.data = self.transform_image_data(image_data)


class FileImageView(ImageView, StateView[list[ImageViewState]]):
    image_config: dict[str, ImageViewState]

    def __init__(self, config: ChannelMapEditorConfig):
        if self.logger is None:
            self.logger = logging.getLogger('chmap.view.file')

        super().__init__(config)
        self.image_root = Path('.')
        self.image_config = {}

    # ============= #
    # UI components #
    # ============= #

    image_input: PathAutocompleteInput

    def _setup_title(self, **kwargs) -> list[UIElement]:
        ret = super()._setup_title(**kwargs)

        self.view_title.text = '<b>Image Path</b>'
        self.image_input = PathAutocompleteInput(
            self.image_root,
            self.on_image_selected,
            mode='file',
            accept=['image/*'],
            # title='Image filepath',
            width=300,
        )
        ret.insert(2, self.image_input.input)

        return ret

    def on_image_selected(self, filename: Path | None):
        if is_recursive_called():
            return

        if (image := self.image) is not None:
            if (state := self.save_current_state()) is not None:
                self.image_config[state['filename']] = state

        if filename is None:
            self.set_image_handler(None)
            self.visible = False
        else:
            self.logger.debug('load(%s)', filename)
            self.set_image_handler(ImageHandler.from_file(filename))
            self.visible = True

        run_later(self.restore_current_state)

    # ========= #
    # load/save #
    # ========= #

    def save_state(self) -> list[ImageViewState]:
        self.logger.debug('save()')
        return list(self.image_config.values())

    def restore_state(self, state: list[ImageViewState]):
        self.logger.debug('restore()')
        for _state in state:  # type:ImageViewState
            _state = ImageViewState(**_state)
            self.image_config[_state['filename']] = _state

    def restore_current_state(self, state: ImageViewState = None):
        if (image := self.image) is None:
            return

        if state is None:
            try:
                state = self.image_config[image.filename]
                self.logger.debug('restore(%s)', image.filename)
            except KeyError:
                self.logger.info('fail restore(%s) ', image.filename)
                f = self.get_resolution_value()
                if f is None:
                    self.reset_boundary_transform()
                else:
                    image.resolution = f
                    self.update_boundary_transform(s=1)
                return

        super().restore_current_state(state)
