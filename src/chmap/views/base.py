from __future__ import annotations

import abc
import logging
import math
from typing import TypeVar, Generic, TypedDict, Any, TYPE_CHECKING, cast, final

import numpy as np
from bokeh.models import UIElement, ColumnDataSource, GlyphRenderer, Slider, Switch, Div
from bokeh.models import tools
from bokeh.plotting import figure as Figure
from numpy.typing import NDArray

from chmap.config import ChannelMapEditorConfig
from chmap.util.bokeh_util import ButtonFactory, SliderFactory, as_callback, is_recursive_called, new_help_button
from chmap.util.utils import doc_link

if TYPE_CHECKING:
    from chmap.main_bokeh import ChannelMapEditorApp
    from chmap.probe import ProbeDesp, M, E

__all__ = [
    'ViewBase', 'ControllerView',
    'StateView', 'GlobalStateView',
    'DynamicView', 'EditorView',
    'InvisibleView',
    'BoundaryState',
    'BoundView',
]


class ViewBase(metaclass=abc.ABCMeta):
    """
    View component base class.

    """

    logger: logging.Logger

    # noinspection PyUnusedLocal
    def __init__(self, config: ChannelMapEditorConfig, *, logger: str | logging.Logger = None):
        if isinstance(logger, str):
            self.logger = logging.getLogger(logger)
        elif isinstance(logger, logging.Logger):
            self.logger = logger
        else:
            self.logger = logging.getLogger('chmap.view.' + type(self).__name__)

        self.logger.debug('init()')

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """view name"""
        pass

    @property
    def description(self) -> str | None:
        """view description. show in help button."""
        return None

    # ============= #
    # UI components #
    # ============= #

    view_title: Div
    view_status: Div
    view_content: UIElement

    def setup(self, f: Figure, **kwargs) -> list[UIElement]:
        """
        Setup controls and plotting.

        :param f: figure in middle panel
        :param kwargs: control or plotting related parameters.
        :return: row list.
        """
        self.logger.debug('setup()')

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
        """

        components in title::

            visible_btn?, view_title, help?, status_div

        :param component:
        :param kwargs:
        :return:
        """
        ret = []
        if isinstance(self, InvisibleView):
            ret.append(self.setup_visible_switch())

        name = self.name
        if not name.startswith('<b>'):
            name = f'<b>{name}</b>'
        self.view_title = Div(text=name)
        ret.append(self.view_title)

        if (desp := self.description) is not None:
            ret.append(new_help_button(desp, position='bottom'))

        self.status_div = Div(text='')
        ret.append(self.status_div)

        return ret

    def _setup_content(self, **kwargs) -> UIElement | list[UIElement] | None:
        return None

    # ================ #
    # updating methods #
    # ================ #

    def start(self):
        """Invoked when figure is ready."""
        pass

    def set_status(self, text: str | None):
        if text is None:
            self.status_div.text = ''
        else:
            self.status_div.text = text
            self.logger.info('status : %s', text)

    # =========== #
    # GUI methods #
    # =========== #

    # noinspection PyUnusedLocal
    @final
    @doc_link(ChannelMapEditorApp='chmap.main_bokeh.ChannelMapEditorApp')
    def log_message(self, *message: str, reset=False):
        """
        log message in {ChannelMapEditorApp}.

        Implement note:
           do not overwrite this function, because this method will be
           replaced by {ChannelMapEditorApp}.

        :param message: message in lines.
        :param reset: reset message area.
        """
        self.logger.info(' '.join(message))


V = TypeVar('V', bound=ViewBase)


class ControllerView:
    @final
    @doc_link(ChannelMapEditorApp='chmap.main_bokeh.ChannelMapEditorApp')
    def get_app(self) -> ChannelMapEditorApp:
        """

        Implement note:
            do not overwrite this function, because this method will be
            replaced by {ChannelMapEditorApp}.

        :return:
        """
        pass

    @final
    @doc_link(ChannelMapEditorApp='chmap.main_bokeh.ChannelMapEditorApp')
    def get_view(self, view_type: str | type[V]) -> V | None:
        """

        Implement note:
            do not overwrite this function, because this method will be
            replaced by {ChannelMapEditorApp}.

        :param view_type:
        :return:
        """
        return None


@doc_link(ChannelMapEditorApp='chmap.main_bokeh.ChannelMapEditorApp')
class InvisibleView:
    """
    This view component's visible state is controlled by {ChannelMapEditorApp}.
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

        if not visible:
            try:
                self.status_div.text = ''
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
    def save_state(self) -> S | None:
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


class GlobalStateView(StateView[S], Generic[S], metaclass=abc.ABCMeta):
    disable_save_global_state = False
    """disable {#restore_global_state()} for debugging purposes to prevent testing content from polluting config."""

    @abc.abstractmethod
    def save_state(self, local=True) -> S | None:
        """
        Save current state into S.

        :param local: Is this state saved into local config?
        :return: json-serialize instance.
        """
        pass

    @final
    @doc_link(ChannelMapEditorApp='chmap.main_bokeh.ChannelMapEditorApp')
    def save_global_state(self, state: S = None, *, sync=False, force=False):
        """
        save state into global config.

        Implement note:
            do not overwrite this function, because this method will be
            replaced by {ChannelMapEditorApp}.

        :param state: saved state. If None, use state from {StateView#save_state()}.
        :param sync: save all {GlobalStateView}. (ignore *state*).
        :param force: ignore {#disable_save_global_state}
        """
        pass

    @final
    @doc_link(ChannelMapEditorApp='chmap.main_bokeh.ChannelMapEditorApp')
    def restore_global_state(self, *, reload=False, force=False):
        """
        read global config and invoke {StateView#restore_state()}.

        Implement note:
            do not overwrite this function, because this method will be
            replaced by {ChannelMapEditorApp}.

        :param reload: reload config from disk
        :param force: force invoke {StateView#restore_state()} (give an empty dict)
        """
        pass


class DynamicView:
    """
    This view component needs to be aware on modification of channelmap and electrodes.
    """

    def on_probe_update(self, probe: ProbeDesp[M, E], chmap: M | None, electrodes: list[E] | None):
        """
        Invoked when channelmap is changed or electrode's category is changed.

        :param probe: probe interface.
        :param chmap: channelmap instance.
        :param electrodes: all electrodes.
        """
        pass


class EditorView(DynamicView):
    """
    This view provider the editing function on channelmap or blueprint.
    """

    @final
    @doc_link(ChannelMapEditorApp='chmap.main_bokeh.ChannelMapEditorApp')
    def update_probe(self):
        """
        notify GUI probe has updated.

        Implement note:
            do not call this method in {DynamicView#on_probe_update()}. It may cause
            recursive call.
        Implement note:
            do not overwrite this function, because this method will be
            replaced by {ChannelMapEditorApp}.
        """
        raise RuntimeError()


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

    Event Call chain
    ----------------

    ::

        (UI component)
            -/-> (event callback)
                -> update_boundary_transform()
                    -> on_boundary_transform()
    """

    data_boundary: ColumnDataSource  # boundary data
    tool_boundary: tools.BoxEditTool
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
        self.data_boundary.on_change('data', as_callback(self._on_boundary_change))

        self.tool_boundary = tools.BoxEditTool(description=boundary_desp, renderers=[self.render_boundary], num_objects=1)
        f.tools.append(self.tool_boundary)

    boundary_rotate_slider: Slider
    boundary_scale_slider: Slider

    def setup_rotate_slider(self, *,
                            new_btn: ButtonFactory = None,
                            new_slider: SliderFactory = None) -> list[UIElement]:
        """
        Setup image rotating controls.

        :param new_btn:
        :param new_slider:
        :return: row list.
        """
        if new_btn is None:
            new_btn = ButtonFactory(min_width=100, width_policy='min')
        if new_slider is None:
            new_slider = SliderFactory(width=300, align='end')

        self.boundary_rotate_slider = new_slider('image rotation (deg)', (-25, 25, 1, 0), self._on_boundary_rotate)

        reset_imr = new_btn('reset', self._on_reset_boundary_rotate)

        return [
            reset_imr, self.boundary_rotate_slider
        ]

    def setup_scale_slider(self, *,
                           new_btn: ButtonFactory = None,
                           new_slider: SliderFactory = None) -> list[UIElement]:
        """
        Setup image scaling controls.

        :param new_btn:
        :param new_slider:
        :return: row list.
        """
        if new_btn is None:
            new_btn = ButtonFactory(min_width=100, width_policy='min')
        if new_slider is None:
            new_slider = SliderFactory(width=300, align='end')

        self.boundary_scale_slider = new_slider('image scale (log)', (-1, 1, 0.01, 0), self._on_boundary_scale)

        reset_ims = new_btn('reset', self._on_reset_boundary_scale)

        return [reset_ims, self.boundary_scale_slider]

    def _on_boundary_rotate(self, s: int):
        if not is_recursive_called():
            self.update_boundary_transform(rt=s)

    def _on_boundary_scale(self, s: float):
        if not is_recursive_called():
            self.update_boundary_transform(s=math.pow(10, s))

    def _on_reset_boundary_rotate(self):
        try:
            self.boundary_rotate_slider.value = 0
        except AttributeError:
            self.update_boundary_transform(rt=0)

    def _on_reset_boundary_scale(self):
        try:
            self.boundary_scale_slider.value = 0
        except AttributeError:
            self.update_boundary_transform(s=1)

    def _on_boundary_change(self, value: dict[str, list[float]]):
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

    def reset_boundary(self):
        self.update_boundary_transform(p=(0, 0), s=1, rt=0)

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

        state = self.get_boundary_state()
        self.on_boundary_transform(state)

    def on_boundary_transform(self, state: BoundaryState):
        """
        Image transforming updating callback.

        :param state: updated boundary parameters.
        """
        if is_recursive_called():
            return

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

    @doc_link()
    def transform_image_data(self, image: NDArray[np.uint], boundary: BoundaryState = None) -> dict[str, Any]:
        """
        A helper method for transforming an image data.

        :param image: image data
        :param boundary: boundary parameters
        :return: a dict which is ready for updating {ColumnDataSource}.
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

        return dict(image=[image], dw=[w], dh=[h], x=[x], y=[y])
