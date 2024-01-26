from __future__ import annotations

import abc
import math
from typing import TypeVar, Generic, TypedDict, Any, TYPE_CHECKING

import numpy as np
from bokeh.models import UIElement, ColumnDataSource, GlyphRenderer, Slider, Switch
from bokeh.plotting import figure as Figure
from numpy.typing import NDArray

from chmap.config import ChannelMapEditorConfig
from chmap.util.bokeh_util import ButtonFactory, SliderFactory, as_callback, is_recursive_called

if TYPE_CHECKING:
    from chmap.probe import ProbeDesp, M, E

__all__ = ['ViewBase', 'StateView', 'DynamicView', 'InvisibleView', 'BoundaryState', 'BoundView']


class ViewBase(metaclass=abc.ABCMeta):
    """
    View component base class.

    """

    # noinspection PyUnusedLocal
    def __init__(self, config: ChannelMapEditorConfig):
        pass

    @abc.abstractmethod
    def setup(self, **kwargs) -> list[UIElement]:
        """
        Setup controls.

        :param kwargs: control related parameters.
        :return: row list.
        """
        pass

    def plot(self, f: Figure, **kwargs):
        """
        Setup plotting in figure.

        :param f:
        :param kwargs: drawing related parameters.
        """
        pass

    def start(self):
        """Invoked when figure is ready."""
        pass


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
        pass


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

    def __init__(self, config: ChannelMapEditorConfig):
        super().__init__(config)

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

    def plot(self, f: Figure, *,
             boundary_color: str = 'black',
             boundary_desp: str = None,
             **kwargs):
        """
        Setup boundary plotting in figure.

        :param f:
        :param boundary_color: boundary border color
        :param boundary_desp: figure tool hint description.
        :param kwargs:
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

    def on_visible(self, visible: bool):
        self.render_boundary.visible = visible

    boundary_rotate_slider: Slider
    boundary_scale_slider: Slider

    def setup_slider(self, slider_width: int = 300) -> list[UIElement]:
        """
        Setup image transforming controls.

        :param slider_width:
        :return: row list.
        """
        new_btn = ButtonFactory(min_width=100, width_policy='min')
        new_slider = SliderFactory(width=slider_width, align='end')

        self.boundary_rotate_slider = new_slider('image rotation (deg)', (-25, 25, 1, 0), self.on_boundary_rotate)
        self.boundary_scale_slider = new_slider('image scale (log)', (-1, 1, 0.01, 0), self.on_boundary_scale)

        reset_imr = new_btn('reset', self.on_reset_boundary_rotate)
        reset_ims = new_btn('reset', self.on_reset_boundary_scale)

        from bokeh.layouts import row
        return [
            row(reset_imr, self.boundary_rotate_slider),
            row(reset_ims, self.boundary_scale_slider),
        ]

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

        try:
            x = float(value['x'][0])
        except IndexError:
            return

        y = float(value['y'][0])
        w = float(value['w'][0])
        h = float(value['h'][0])
        sx = w / self.width
        sy = h / self.height

        self.update_boundary_transform(p=(x, y), s=(sx, sy))

    def reset_boundary_transform(self):
        self.data_boundary.data = dict(x=[0], y=[0], w=[0], h=[0], r=[0], sx=[1], sy=[1])
        self._update_boundary_transform(self.get_boundary_state())

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
