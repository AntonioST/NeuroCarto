from __future__ import annotations

import abc
import logging
import sys
from pathlib import Path
from typing import TypedDict, Final

import numpy as np
from bokeh.models import ColumnDataSource, GlyphRenderer, Slider, UIElement
from numpy.typing import NDArray

from neurocarto.config import CartoConfig
from neurocarto.util.bokeh_app import run_later
from neurocarto.util.bokeh_util import SliderFactory, is_recursive_called, PathAutocompleteInput, new_help_button
from neurocarto.views.base import Figure, BoundView, StateView, BoundaryState

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
    """
    Image information.
    """

    def __init__(self, filename: str | None):
        self.filename: Final[str | None] = filename
        self._resolution: tuple[float, float] = (1.0, 1.0)

    @abc.abstractmethod
    def __len__(self) -> int:
        """number of image slides."""
        pass

    @abc.abstractmethod
    def __getitem__(self, index: int) -> NDArray[np.uint] | None:
        """Get image slide at *index*."""
        pass

    @property
    @abc.abstractmethod
    def width(self) -> float:
        """image width in um."""
        pass

    @property
    @abc.abstractmethod
    def height(self) -> float:
        """image height in um."""
        pass

    @property
    def resolution(self) -> tuple[float, float]:
        """resolution (width, height) in unit um/pixel."""
        return self._resolution

    @resolution.setter
    def resolution(self, resolution: float | tuple[float, float]):
        if not isinstance(resolution, tuple):
            resolution = float(resolution)
            resolution = (resolution, resolution)

        self._resolution = resolution

    @classmethod
    def from_numpy(cls, filename: str | Path, image: NDArray[np.uint] = None) -> Self:
        logger = logging.getLogger('neurocarto.image')

        filename = str(filename)
        if image is None:
            image = np.load(filename)

        from .image_npy import NumpyImageHandler
        logger.debug('from numpy %s', image.shape)
        return NumpyImageHandler(image, filename)  # type: ignore[return-value]

    @classmethod
    def from_file(cls, filename: str | Path) -> Self:
        """

        :param filename: any Pillow support image format
        :return:
        """
        logger = logging.getLogger('neurocarto.image')
        from PIL import Image  # type: ignore[import]
        from .image_npy import NumpyImageHandler

        filename = str(filename)
        logger.debug('from file %s', filename)
        image = np.asarray(Image.open(filename, mode='r'))

        w, h, c = image.shape
        if c == 3:
            image = np.dstack([image, np.full((w, h), 255, dtype=np.uint8)])
        image = np.flipud(image.view(dtype=np.uint32).reshape((w, h)))

        logger.debug('as image %s', image.shape)
        return NumpyImageHandler(image, filename)  # type: ignore[return-value]

    @classmethod
    def from_tiff(cls, filename: str | Path) -> Self:
        logger = logging.getLogger('neurocarto.image')
        import tifffile
        from .image_npy import NumpyImageHandler

        filename = str(filename)
        logger.debug('from file %s', filename)
        image = tifffile.TiffFile(filename, mode='r').asarray()  # TODO memmap?

        logger.debug('as image %s', image.shape)
        return NumpyImageHandler(image, filename)  # type: ignore[return-value]


class ImageView(BoundView, metaclass=abc.ABCMeta):
    """
    Base of image view.
    """
    data_image: ColumnDataSource
    render_image: GlyphRenderer

    def __init__(self, config: CartoConfig, image: ImageHandler = None, *,
                 logger: str | logging.Logger = 'neurocarto.view.image'):
        super().__init__(config, logger=logger)

        self.data_image = ColumnDataSource(data=dict(image=[], x=[], y=[], dw=[], dh=[]))
        if image is not None:
            self.logger.debug('init(%s)', type(image).__name__)

        self._image = image
        self._index: int = 0

    @property
    def name(self) -> str:
        if (image := self._image) is None or image.filename is None:
            return '<b>Image</b>'
        else:
            return f'<b>Image</b> {image.filename}'

    # ================ #
    # image properties #
    # ================ #

    @property
    def image(self) -> ImageHandler | None:
        return self._image

    @property
    def width(self) -> float:
        try:
            return self._image.width  # type: ignore
        except (TypeError, AttributeError):
            return 0

    @property
    def height(self) -> float:
        try:
            return self._image.height  # type: ignore
        except (TypeError, AttributeError):
            return 0

    def set_image_handler(self, image: ImageHandler | None):
        self._image = image
        if image is not None and (filename := image.filename) is not None:
            self.set_status(filename)

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

        if (filename := image.filename) is None:
            return None

        self.logger.debug('save(%s)', filename)
        boundary = self.get_boundary_state()
        resolution = image.resolution

        return ImageViewState(
            filename=filename,
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

    def _setup_render(self, f: Figure,
                      boundary_color: str | None = 'black',
                      **kwargs):
        self.setup_image(f)

        desp = 'drag image'
        if (image := self._image) is not None:
            desp = f'drag {image.filename}'

        if boundary_color is not None:
            self.setup_boundary(f, boundary_color=boundary_color, boundary_desp=desp)

    def setup_image(self, f: Figure):
        self.render_image = f.image_rgba(
            'image', x='x', y='y', dw='dw', dh='dh', source=self.data_image,
            global_alpha=1, syncable=False,
        )

    index_slider: Slider = None
    alpha_slider: Slider = None

    def _setup_content(self, slider_width: int = 300,
                       support_index=True,
                       support_rotate=True,
                       support_scale=True,
                       support_alpha=False,
                       **kwargs) -> list[UIElement]:
        """

        :param slider_width:
        :param support_index: support image slide indexing.
        :param support_rotate: support image rotation
        :param support_scale: support image scaling
        :param kwargs:
        :return:
        """
        from bokeh.layouts import row

        new_slider = SliderFactory(width=slider_width, align='end')

        ret = []
        if support_index:
            ret.append(row(*self.setup_index_slider(new_slider=new_slider)))
        if support_alpha:
            ret.append(row(*self.setup_alpha_slider(new_slider=new_slider)))
        if support_rotate:
            ret.append(row(*self.setup_rotate_slider(new_slider=new_slider)))
        if support_scale:
            ret.append(row(*self.setup_scale_slider(new_slider=new_slider)))

        return ret

    def setup_index_slider(self, *,
                           new_slider: SliderFactory = None) -> list[UIElement]:
        if new_slider is None:
            new_slider = SliderFactory(width=300, align='end')

        self.index_slider = new_slider('Index', (0, 1, 1, 0), self._on_index_changed)
        self.index_slider.visible = False

        return [self.index_slider]

    def setup_alpha_slider(self, *,
                           new_slider: SliderFactory = None) -> list[UIElement]:
        if new_slider is None:
            new_slider = SliderFactory(width=300, align='end')

        self.alpha_slider = new_slider('Alpha', (0, 255, 1, 255), self._on_alpha_changed)

        return [self.alpha_slider]

    def _on_index_changed(self, s: int):
        if is_recursive_called():
            return

        self.update_image(s)

    def _on_alpha_changed(self, a: int):
        if is_recursive_called():
            return

        if (image := self._image) is None:
            return

        try:
            image.alpha = a
        except AttributeError:
            pass
        else:
            self.update_image(self.index_slider.value)

    # ================ #
    # updating methods #
    # ================ #

    def start(self):
        if self.image is not None:
            self.visible = True
            self.set_image_handler(self.image)  # trigger setter
            self.reset_boundary()
        else:
            self.visible = False

    def on_boundary_transform(self, state: BoundaryState):
        super().on_boundary_transform(state)
        self.update_image(self._index)

    def update_image(self, image_data: int | NDArray[np.uint] | None):
        if is_recursive_called():
            return

        if (image := self.image) is None:
            return

        if isinstance(image_data, int) or isinstance(image_data, np.integer):
            self._index = index = int(image_data)

            try:
                image_data = image[index]
            except (IndexError, TypeError) as e:
                return

            try:
                self.index_slider.value = index
            except (AttributeError, TypeError) as e:
                pass

        if image_data is None:
            self.data_image.data = dict(image=[], dw=[], dh=[], x=[], y=[])
        else:
            self.data_image.data = self.transform_image_data(image_data)


class FileImageView(ImageView, StateView[list[ImageViewState]]):
    """
    Load images from file.
    """
    image_config: dict[str, ImageViewState]

    def __init__(self, config: CartoConfig, *,
                 logger: str | logging.Logger = 'neurocarto.view.file'):
        super().__init__(config, logger=logger)
        self.image_root = Path('.')
        self.image_config = {}

    @property
    def name(self) -> str:
        return 'Image Path'

    # ============= #
    # UI components #
    # ============= #

    image_input: PathAutocompleteInput

    def _setup_title(self, **kwargs) -> list[UIElement]:
        ret = super()._setup_title(**kwargs)

        self.image_input = PathAutocompleteInput(
            self.image_root,
            self.on_image_selected,
            mode='file',
            accept=['image/*'],
            # title='Image filepath',
            width=300,
        )

        # visible_btn?, view_title, help?, image_input, help, resolution_input, help, status_div
        ret.insert(-3, self.image_input.input)
        ret.insert(-3, new_help_button('show image file', position='top'))

        return ret

    def _setup_content(self,
                       support_rotate=False,
                       **kwargs) -> list[UIElement]:
        return super()._setup_content(support_rotate=support_rotate, **kwargs)

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

        if state is None and image.filename is not None:
            try:
                state = self.image_config[image.filename]
                self.logger.debug('restore(%s)', image.filename)
            except KeyError:
                self.logger.info('fail restore(%s) ', image.filename)
                self.reset_boundary()
                return

        if state is not None:
            super().restore_current_state(state)
