from __future__ import annotations

import abc
import logging
import math
from pathlib import Path
from typing import TypeVar, Generic, TypedDict, Any, TYPE_CHECKING, cast

import numpy as np
from bokeh.models import UIElement, ColumnDataSource, GlyphRenderer, Slider, Switch, Div
from bokeh.plotting import figure as Figure
from numpy.typing import NDArray

from chmap.config import ChannelMapEditorConfig
from chmap.util.bokeh_util import ButtonFactory, SliderFactory, as_callback, is_recursive_called, is_image, new_help_button

if TYPE_CHECKING:
    from chmap.probe import ProbeDesp, M, E

__all__ = ['ViewBase', 'StateView', 'DynamicView', 'InvisibleView', 'BoundaryState', 'BoundView']


class ViewBase(metaclass=abc.ABCMeta):
    """
    View component base class.

    """

    logger: logging.Logger = None

    # noinspection PyUnusedLocal
    def __init__(self, config: ChannelMapEditorConfig, *, logger: str | logging.Logger = None):
        if isinstance(logger, str):
            self.logger = logging.getLogger(logger)
        elif isinstance(logger, logging.Logger):
            self.logger = logger
        else:
            raise TypeError()

        if (logger := self.logger) is not None:
            logger.debug('init()')

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """view name"""
        pass

    @property
    def description(self) -> str | None:
        """view description. show in help button."""
        return None

    view_title: Div
    view_content: UIElement

    def setup(self, f: Figure, **kwargs) -> list[UIElement]:
        """
        Setup controls and plotting.

        :param f: figure in middle panel
        :param kwargs: control or plotting related parameters.
        :return: row list.
        """
        if (logger := self.logger) is not None:
            logger.debug('setup()')

        self._setup_render(f, **kwargs)

        from bokeh.layouts import row, column
        ret = []
        title = row(self._setup_title(**kwargs))
        ret.append(title)

        content = self._setup_content(**kwargs)
        if content is not None:
            if isinstance(content, list):
                content = column(content)
            self.view_content = content
            ret.append(content)

        return ret

    def _setup_render(self, f: Figure, **kwargs):
        pass

    def _setup_title(self, **kwargs) -> list[UIElement]:
        ret = []
        if isinstance(self, InvisibleView):
            ret.append(self.setup_visible_switch())

        self.view_title = Div(text=f'<b>{self.name}</b>')
        ret.append(self.view_title)
        if (desp := self.description) is not None:
            ret.append(new_help_button(desp))
        return ret

    def _setup_content(self, **kwargs) -> UIElement | list[UIElement] | None:
        return None

    def start(self):
        """Invoked when figure is ready."""
        pass


def init_view(config: ChannelMapEditorConfig, view_type) -> ViewBase | None:
    """

    Recognised type:

    * `None` skip
    * `ViewBase` or `type[ViewBase]`
    * `ImageHandler`, wrap with ImageView.
    * literal 'file' for FileImageView
    * image filepath
    * `str` in pattern: `module.path:attribute` in type listed above.


    :param config:
    :param view_type:
    :return:
    """
    from chmap.views.image import ImageView, ImageHandler

    try:
        if isinstance(view_type, type) and issubclass(view_type, ViewBase):
            return view_type(config)

        elif isinstance(view_type, ViewBase):
            return view_type

        elif isinstance(view_type, ImageHandler):
            return ImageView(config, view_type)

        elif view_type == 'file':
            from chmap.views.image import FileImageView
            return FileImageView(config)

        elif isinstance(view_type, str) and is_image(image_file := Path(view_type)):
            from chmap.views.image import ImageView, ImageHandler
            return ImageView(config, ImageHandler.from_file(image_file))

        elif isinstance(view_type, str):
            return import_view(config, view_type)

    except BaseException as e:
        logging.getLogger('chmap.view').warning('init view fail', exc_info=e)
        pass

    return None


def import_view(config: ChannelMapEditorConfig, module_path: str) -> ViewBase | None:
    module, _, name = module_path.partition(':')
    if len(name) == 0:
        raise ValueError(f'not a module_path pattern "module_path:name" : {module_path}')

    import importlib
    module = importlib.import_module(module)

    return init_view(config, getattr(module, name))


class InvisibleView:
    """
    This view component's visible state is controlled by GUI.
    """

    visible_btn: Switch

    @property
    def visible(self) -> bool:
        try:
            return self.visible_btn.active
        except AttributeError:
            return True

    @visible.setter
    def visible(self, v: bool):
        try:
            self.visible_btn.active = v
        except AttributeError:
            pass

    def setup_visible_switch(self) -> Switch:
        """Setup visible switch control."""
        self.visible_btn = Switch(active=True)
        self.visible_btn.on_change('active', as_callback(self.on_visible))
        return self.visible_btn

    def on_visible(self, visible: bool):
        """visible state changed callback.

        :param visible: new visible state
        """
        if (logger := getattr(self, 'logger', None)) is not None:
            cast(logging.Logger, logger).debug(f'visible({visible})')

        try:
            self.view_content.visible = visible
        except AttributeError:
            pass

        for attr in dir(self):
            if attr.startswith('render_') and isinstance(render := getattr(self, attr, None), GlyphRenderer):
                render.visible = visible


S = TypeVar('S')


class StateView(Generic[S], metaclass=abc.ABCMeta):
    """
    This view component has something states can be saved and restored.

    :param S: stored information. type should be json-serialize.
    """

    @abc.abstractmethod
    def save_state(self) -> S:
        """
        Save current state into S.

        :return: json-serialize instance.
        """
        pass

    @abc.abstractmethod
    def restore_state(self, state: S):
        """
        Restore state from *state*.

        :param state: json-deserialize instance.
        """
        pass


class DynamicView:
    """
    This view component needs to be aware on modification of channelmap and electrodes.
    """

    def on_probe_update(self, probe: ProbeDesp[M, E], chmap: M | None, e: list[E] | None):
        """
        Invoked when channelmap is changed or electrode's policy is changed.

        :param probe: probe interface.
        :param chmap: channelmap instance.
        :param e: all electrodes.
        """
        pass


class BoundaryState(TypedDict):
    """Boundary parameters"""
    dx: float  # x moving
    dy: float  # y moving
    sx: float  # x scaling
    sy: float  # y scaling
    rt: float  # rotating (degree)


class BoundView(ViewBase, InvisibleView, metaclass=abc.ABCMeta):
    """
    This view component has draw a rectangle-like (shorten as *image*) on the plotting,
    and supporting moving, scaling and rotating. This class provide a framework for
    supporting image transforming.

    This class handle a rectangle as boundary. The image should follow the boundary updating.
    """

    data_boundary: ColumnDataSource  # boundary data
    render_boundary: GlyphRenderer  # boundary drawing

    def __init__(self, config: ChannelMapEditorConfig, *,
                 logger: str | logging.Logger = None):
        super().__init__(config, logger=logger)

        self.data_boundary = ColumnDataSource(data=dict(x=[0], y=[0], w=[0], h=[0], r=[0], sx=[1], sy=[1]))

    @property
    @abc.abstractmethod
    def width(self) -> float:
        """Width of image"""
        pass

    @property
    @abc.abstractmethod
    def height(self) -> float:
        """Height of image"""
        pass

    def setup_boundary(self, f: Figure, *,
                       boundary_color: str = 'black',
                       boundary_desp: str = None):
        """
        Setup boundary plotting in figure.

        :param f:
        :param boundary_color: boundary border color
        :param boundary_desp: figure tool hint description.
        """
        self.render_boundary = f.rect(
            'x', 'y', 'w', 'h', 'r', source=self.data_boundary,
            color=boundary_color, fill_alpha=0, angle_units='deg',
        )
        self.data_boundary.on_change('data', as_callback(self.on_boundary_change))

        from bokeh.models import tools
        f.tools.append(tools.BoxEditTool(
            description=boundary_desp,
            renderers=[self.render_boundary],
            num_objects=1
        ))

    boundary_rotate_slider: Slider
    boundary_scale_slider: Slider

    def setup_rotate_slider(self, *,
                            new_btn: ButtonFactory = None,
                            new_slider: SliderFactory = None) -> list[UIElement]:
        """
        Setup image rotating controls.

        :param new_btn: ButtonFactory
        :param new_slider: SliderFactory
        :return: row list.
        """
        if new_btn is None:
            new_btn = ButtonFactory(min_width=100, width_policy='min')
        if new_slider is None:
            new_slider = SliderFactory(width=300, align='end')

        self.boundary_rotate_slider = new_slider('image rotation (deg)', (-25, 25, 1, 0), self.on_boundary_rotate)

        reset_imr = new_btn('reset', self.on_reset_boundary_rotate)

        return [
            reset_imr, self.boundary_rotate_slider
        ]

    def setup_scale_slider(self, *,
                           new_btn: ButtonFactory = None,
                           new_slider: SliderFactory = None) -> list[UIElement]:
        """
        Setup image scaling controls.

        :param new_btn: ButtonFactory
        :param new_slider: SliderFactory
        :return: row list.
        """
        if new_btn is None:
            new_btn = ButtonFactory(min_width=100, width_policy='min')
        if new_slider is None:
            new_slider = SliderFactory(width=300, align='end')

        self.boundary_scale_slider = new_slider('image scale (log)', (-1, 1, 0.01, 0), self.on_boundary_scale)

        reset_ims = new_btn('reset', self.on_reset_boundary_scale)

        return [reset_ims, self.boundary_scale_slider]

    def on_boundary_rotate(self, s: int):
        if not is_recursive_called():
            self.update_boundary_transform(rt=s)

    def on_boundary_scale(self, s: float):
        if not is_recursive_called():
            self.update_boundary_transform(s=math.pow(10, s))

    def on_reset_boundary_rotate(self):
        try:
            self.boundary_rotate_slider.value = 0
        except AttributeError:
            self.update_boundary_transform(rt=0)

    def on_reset_boundary_scale(self):
        try:
            self.boundary_scale_slider.value = 0
        except AttributeError:
            self.update_boundary_transform(s=1)

    def on_reset_boundary(self):
        self.update_boundary_transform(p=(0, 0), s=1, rt=0)

    def on_boundary_change(self, value: dict[str, list[float]]):
        if is_recursive_called():
            return

        iw = self.width
        ih = self.height
        if (iw <= 0) or (ih <= 0):
            return

        try:
            x = float(value['x'][0])
        except IndexError:
            return

        y = float(value['y'][0])
        w = float(value['w'][0])
        h = float(value['h'][0])
        sx = w / iw
        sy = h / ih

        self.update_boundary_transform(p=(x, y), s=(sx, sy))

    def get_boundary_state(self) -> BoundaryState:
        """Get current boundary parameters."""
        data = self.data_boundary.data
        dx = float(data['x'][0])
        dy = float(data['y'][0])
        w = float(data['w'][0])
        h = float(data['h'][0])
        rt = float(data['r'][0])
        ow = self.width
        oh = self.height

        if ow == 0:
            sx = 1
        else:
            sx = w / ow

        if oh == 0:
            sy = 1
        else:
            sy = h / oh

        return BoundaryState(dx=dx, dy=dy, sx=sx, sy=sy, rt=rt)

    def reset_boundary_transform(self):
        self.data_boundary.data = dict(x=[0], y=[0], w=[0], h=[0], r=[0], sx=[1], sy=[1])
        # data -> update_boundary_transform
        #   -> _update_boundary_transform

    def update_boundary_transform(self, *,
                                  p: tuple[float, float] = None,
                                  s: float | tuple[float, float] = None,
                                  rt: float = None):
        """
        Image transforming updating handle.

        :param p: position (x, y)
        :param s: scaling (sx, sy)
        :param rt: rotating
        """
        if is_recursive_called():
            return

        old = self.get_boundary_state()

        if p is not None:
            x, y = p
        else:
            x = old['dx']
            y = old['dy']

        if s is not None:
            if isinstance(s, tuple):
                sx, sy = s
            else:
                sx = sy = float(s)
        else:
            sx = old['sx']
            sy = old['sy']

        if sx <= 0:
            sx = 1
        if sy <= 0:
            sy = 1

        if rt is None:
            rt = old['rt']

        w = self.width * sx
        h = self.height * sy

        self.data_boundary.data = dict(
            x=[x], y=[y], w=[w], h=[h], r=[rt], sx=[sx], sy=[sy]
        )

        self._update_boundary_transform(self.get_boundary_state())

    def _update_boundary_transform(self, state: BoundaryState):
        """
        Image transforming updating callback.

        :param state: updated boundary parameters.
        """
        try:
            sx = state['sx']
            sy = state['sy']
            s = min(sx, sy)
            if s > 0:
                s = round(math.log10(s), 2)
            else:
                s = 0

            self.boundary_scale_slider.value = s
        except AttributeError:
            pass

        try:
            self.boundary_rotate_slider.value = state['rt']
        except AttributeError:
            pass

    def transform_image_data(self, image: NDArray[np.uint], boundary: BoundaryState = None, *,
                             field_image='image', field_x='x', field_y='y', field_w='dw', field_h='dh') -> dict[str, Any]:
        """
        A helper method for transforming an image data.

        :param image: image data
        :param boundary: boundary parameters
        :param field_image: field name of image in ColumnDataSource
        :param field_x: field name of x
        :param field_y: field name of y
        :param field_w: field name of w
        :param field_h: field name of h
        :return: a dict which is ready for updating ColumnDataSource.
        """
        if boundary is None:
            boundary = self.get_boundary_state()

        w = self.width * boundary['sx']
        h = self.height * boundary['sy']
        x = boundary['dx'] - w / 2
        y = boundary['dy'] - h / 2

        if (rt := boundary['rt']) != 0:
            from scipy.ndimage import rotate
            image = rotate(image, -rt, reshape=False)

        return {field_image: [image], field_w: [w], field_h: [h], field_x: [x], field_y: [y]}
