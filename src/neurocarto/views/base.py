from __future__ import annotations

import abc
import logging
import math
import sys
from typing import TypeVar, Generic, TypedDict, Any, TYPE_CHECKING, cast, final, NamedTuple

import numpy as np
from bokeh.models import UIElement, ColumnDataSource, GlyphRenderer, Slider, Switch, Div, tools
from bokeh.plotting import figure as Figure
from numpy.typing import NDArray

from neurocarto.config import CartoConfig
from neurocarto.util.bokeh_app import run_timeout, remove_timeout
from neurocarto.util.bokeh_util import ButtonFactory, SliderFactory, as_callback, is_recursive_called, new_help_button
from neurocarto.util.utils import doc_link, SPHINX_BUILD

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

if TYPE_CHECKING:
    from neurocarto.main_app import CartoApp
    from neurocarto.probe import ProbeDesp, M, E
    from neurocarto.util.util_blueprint import BlueprintFunctions
elif SPHINX_BUILD:
    ProbeDesp = 'neurocarto.probe.ProbeDesp'
    CartoApp = 'neurocarto.main_app.CartoApp'
    BlueprintFunctions = 'neurocarto.util.util_blueprint.BlueprintFunctions'

__all__ = [
    'Figure',
    'ViewBase', 'ControllerView',
    'StateView', 'GlobalStateView',
    'DynamicView', 'EditorView',
    'ExtensionView', 'InvisibleView',
    'RecordView', 'RecordStep',
    'BoundaryState',
    'BoundView',
]


class ViewBase(metaclass=abc.ABCMeta):
    """
    View component base class.

    """

    logger: logging.Logger

    # noinspection PyUnusedLocal
    def __init__(self, config: CartoConfig, *, logger: str | logging.Logger | None = None):
        if isinstance(logger, str):
            self.logger = logging.getLogger(logger)
        elif isinstance(logger, logging.Logger):
            self.logger = logger
        else:
            self.logger = logging.getLogger('neurocarto.view.' + type(self).__name__)

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
        ret: list[UIElement] = []
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
        """
        Setup renders in the center figure.

        :param f: the center figure
        :param kwargs: custom parameters for subclass.
        """
        pass

    @doc_link()
    def _setup_title(self, **kwargs) -> list[UIElement]:
        """
        Setup title row.

        components in title::

            visible_btn?, view_title, help?, status_div

        where

        * visible_btn: a visible switch, shown when self is a {InvisibleView}.
        * view_title: title
        * help: help button, shown when {#description} is not ``None``
        * status_div: status message, used by {#set_status()}.

        :param kwargs: custom parameters for subclass.
        :return: list of components, which placed in a row.
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
        """
        setup content row.

        :param kwargs: custom parameters for subclass.
        :return: a layout, or list of components which will placed in a column. ``None`` for empty content.
        """
        return None

    # ================ #
    # updating methods #
    # ================ #

    def start(self):
        """Invoked when figure is ready."""
        pass

    # =========== #
    # GUI methods #
    # =========== #

    _status_decay_callback = None

    def set_status(self, text: str | None, *, decay: float = None):
        """
        Set status message, which is shown right beside the help button in the title row.

        :param text: message
        :param decay: after give seconds, clear the message.
        """
        if text is None:
            self.status_div.text = ''

            if (callback := self._status_decay_callback) is not None:
                remove_timeout(callback)
            self._status_decay_callback = None
        else:
            self.status_div.text = text
            self.logger.info('status : %s', text)
            if decay is not None:
                if (callback := self._status_decay_callback) is not None:
                    remove_timeout(callback)
                self._status_decay_callback = run_timeout(int(decay * 1000), self.set_status, None)

    # noinspection PyUnusedLocal
    @final
    @doc_link()
    def log_message(self, *message: str, reset=False):
        """
        log messages in {CartoApp}.

        Implement note:
           do not overwrite this function, because this method will be
           replaced by {CartoApp}.

        :param message: message in lines.
        :param reset: reset message area.
        """
        self.logger.info(' '.join(message))


V = TypeVar('V', bound=ViewBase)


class ControllerView:
    """
    This view component is a controller view that it purpose to
    control {CartoApp} or other {ViewBase}.
    """

    @final
    @doc_link()
    def get_app(self) -> CartoApp:
        """
        Get {CartoApp} instance.

        Implement note:
            do not overwrite this function, because this method will be
            replaced by {CartoApp}.

        :return:
        """
        raise RuntimeError()

    @final
    @doc_link()
    def get_view(self, view_type: str | type[V]) -> V | None:
        """
        Get corresponding {ViewBase} instance if activated.

        Implement note:
            do not overwrite this function, because this method will be
            replaced by {CartoApp}.

        :param view_type: view type or its type name.
        :return:
        """
        raise RuntimeError()

    @doc_link()
    def new_blueprint_function(self, probe: str | ProbeDesp, chmap: int | str | M | None = None) -> BlueprintFunctions:
        """
        Create a {BlueprintFunctions} with controller supports.

        :param probe: {ProbeDesp} or a module path
        :param chmap: channelmap instance.
        :return: an empty {BlueprintFunctions}
        """
        from neurocarto.util.util_blueprint import BlueprintFunctions
        ret = BlueprintFunctions(probe, chmap)
        ret._controller = self
        return ret


@doc_link(
    init_view='neurocarto.views.utils.init_view'
)
class ExtensionView:
    """
    This view component require some probe specific operations,
    and it is only enabled/activated when the probe fit the requirements.

    In general, it requires a {ProbeDesp} implement some functions or a Protocol class.
    {init_view()} use {#is_supported()} to determine whether it needs to initialize this view or not.

    """

    @classmethod
    def is_supported(cls, probe: ProbeDesp) -> bool:
        """Check whether a *probe* fit the requirements."""
        return False


@doc_link()
class InvisibleView:
    """
    This view component's visible state is controlled by {CartoApp}.
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


@doc_link()
class StateView(Generic[S], metaclass=abc.ABCMeta):
    """
    This view component can save a state with a channelmap into a project config.
    It also can read the config and restore the state.

    The state will be stored (invoke {#save_state()}) when a corresponding channelmap has saved.
    The state will be restored (invoke {#restore_state()}) when a corresponding channelmap has loaded.

    :param S: stored information. It should be json-serializable. We usually use a TypedDict.
    """

    @abc.abstractmethod
    def save_state(self) -> S | None:
        """
        Save current state into S.

        :return: json-serializable instance.
        """
        pass

    @abc.abstractmethod
    def restore_state(self, state: S):
        """
        Restore state from *state*.

        :param state: json-deserializable instance.
        """
        pass


@doc_link()
class GlobalStateView(StateView[S], Generic[S], metaclass=abc.ABCMeta):
    """
    This view component can save a **global** state into a user config, which is shared across projects (channelmaps).
    It also can read the config and restore the state.

    Due to it also inherit from {StateView}, it is able to save a **local** state into a project config.
    The state S can be either shared (user and project config) or separate.

    For a local state, storing and restoring are following a channelmap file (as same as {StateView}).
    However, for a global state, storing and restoring will not be triggered by {CartoApp} automatically.
    The view needs to explicitly use {#save_global_state()} or {#restore_global_state()} to make it happen.

    :param S: stored information. It should be json-serializable. We usually use a TypedDict.
    """

    disable_save_global_state = False
    """disable {#restore_global_state()} for debugging purposes to prevent testing content from polluting config."""

    def save_state(self, local=True) -> S | None:
        """
        Save current state into S.

        Default returning ``None``, which indicate this view only read config and never write config.

        :param local: Is this state going to be saved into a project config?
        :return: json-serializable instance.
        """
        return None

    @final
    @doc_link()
    def save_global_state(self, state: S = None, *, sync=False, force=False):
        """
        save **state** into user config.

        Although {CartoApp} will invoke {#save_state()} when the session is closed,
        the timing is not guarantied. It is better to call this method when any global
        option is modified.

        Implement note:
            do not overwrite this function, because this method will be
            replaced by {CartoApp}.

        :param state: saved state. If None, use state from {StateView#save_state()}.
        :param sync: save all {GlobalStateView}. (ignore *state*).
        :param force: ignore {#disable_save_global_state}
        """
        pass

    @final
    @doc_link()
    def restore_global_state(self, *, reload=False, force=False):
        """
        read user config and invoke {StateView#restore_state()}.

        Remember to invoke this method when {ViewBase#start()} to restore the global state.

        Implement note:
            do not overwrite this function, because this method will be
            replaced by {CartoApp}.

        :param reload: reload config from disk
        :param force: force invoke {StateView#restore_state()} (give an empty dict), even when the config is not existed.
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
    @doc_link()
    def update_probe(self):
        """
        notify GUI probe has updated.

        Implement note:
            do not call this method in {DynamicView#on_probe_update()}. It may cause
            recursive call.
        Implement note:
            do not overwrite this function, because this method will be
            replaced by {CartoApp}.
        """
        raise RuntimeError()


R = TypeVar('R')


# class RecordStep[R](NamedTuple): # python >= 3.11
class RecordStep(NamedTuple):
    source: str
    """(class) name of the source RecordView"""

    time_stamp: float
    """action time stamp in unix time"""

    category: str
    """action category"""

    description: str
    """action description"""

    record: R
    """json-serializable"""

    def __str__(self):
        return f'RecordStep({self.source})[{self.category}]{{{self.description}}}'

    def with_record(self, record: R) -> Self:
        return self._replace(record=record)

    def as_dict(self) -> dict:
        return self._asdict()

    @classmethod
    def from_dict(cls, record: dict) -> Self:
        return RecordStep(
            record['source'],
            record['time_stamp'],
            record['category'],
            record['description'],
            record['record'],
        )  # type: ignore[return-value]


class RecordView(Generic[R], metaclass=abc.ABCMeta):
    """
    This view can record each manipulating steps and also can replay them,

    :param R: RecordedAction. It should be json-serializable. We usually use a TypedDict.
    """

    @final
    @doc_link(RecordManager='neurocarto.views.record.RecordManager')
    def add_record(self, record: R, category: str, description: str):
        """
        add a record into history.

        Implement note:
            do not overwrite this function, because this method will be
            replaced by {RecordManager}.

        :param record: stored step. type should be json-serializable.
        :param category: step category
        :param description: step description
        """
        pass

    def filter_records(self, records: list[RecordStep], *, reset=False) -> list[RecordStep]:
        """
        Filter records which source came from itself.

        It is also used to modify the *records* and return the equivalence steps.

        :param records:
        :param reset: reset view to the initial state? If so, you can add extra action in return.
        :return: filtered records
        """
        name = type(self).__name__
        return [it for it in records if it.source == name]

    @abc.abstractmethod
    @doc_link(RecordManager='neurocarto.views.record.RecordManager')
    def replay_record(self, record: RecordStep):
        """
        Replay the record step.

        Use Note:
            Do not call this method directly, because it might cause
            {#add_record()} be invoked during the replay.
            Use {RecordManager#replay()} instead.

        :param record:
        """
        pass


class BoundaryState(TypedDict):
    """Boundary parameters"""

    dx: float
    """x moving (um)"""

    dy: float
    """y moving (um)"""

    sx: float
    """x scaling"""

    sy: float
    """y scaling"""

    rt: float
    """rotating (degree)"""


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

    def __init__(self, config: CartoConfig, *,
                 logger: str | logging.Logger | None = None):
        super().__init__(config, logger=logger)

        self.data_boundary = ColumnDataSource(data=dict(x=[0], y=[0], w=[0], h=[0], r=[0], sx=[1], sy=[1]))

    # ========== #
    # properties #
    # ========== #

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

    # ============= #
    # UI components #
    # ============= #

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

    # ================ #
    # boundary methods #
    # ================ #

    def get_boundary_state(self) -> BoundaryState:
        """Get current boundary parameters."""
        data = self.data_boundary.data
        dx = float(data['x'][0])  # type: ignore
        dy = float(data['y'][0])  # type: ignore
        w = float(data['w'][0])  # type: ignore
        h = float(data['h'][0])  # type: ignore
        rt = float(data['r'][0])  # type: ignore
        ow = self.width
        oh = self.height

        if ow == 0:
            sx = 1.0
        else:
            sx = w / ow

        if oh == 0:
            sy = 1.0
        else:
            sy = h / oh

        return BoundaryState(dx=dx, dy=dy, sx=sx, sy=sy, rt=rt)

    def reset_boundary(self):
        self.update_boundary_transform(p=(0, 0), s=1, rt=0)

    def set_anchor_to(self, p: tuple[float, float], a: tuple[float, float] = (0, 0)):
        """
        Update boundary transform to move *a* onto *p*.

        :param p: target point on figure. figure (probe) origin as origin.
        :param a: anchor point on image, center point as origin.
        """
        from neurocarto.util.probe_coor import prepare_affine_matrix

        state = self.get_boundary_state()
        t = prepare_affine_matrix(dx=0, dy=0, sx=state['sx'], sy=state['sy'], rt=state['rt'])
        q = t @ [a[0], a[1], 1]  # transformed anchor point
        dx = float(p[0] - q[0])
        dy = float(p[1] - q[1])
        self.update_boundary_transform(p=(dx, dy))

    def update_boundary_transform(self, *,
                                  p: tuple[float, float] = None,
                                  s: float | tuple[float, float] = None,
                                  rt: float = None):
        """
        Image transforming updating handle.

        :param p: center position (x, y)
        :param s: scaling (sx, sy)
        :param rt: rotating degree
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

    # ============== #
    # helper methods #
    # ============== #

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
            from scipy.ndimage import rotate  # type: ignore[import]
            image = rotate(image, -rt, reshape=False)

        return dict(image=[image], dw=[w], dh=[h], x=[x], y=[y])
