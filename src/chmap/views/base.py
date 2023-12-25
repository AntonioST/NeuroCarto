import abc
import math
from typing import TypeVar, Generic, TypedDict, Any

import numpy as np
from bokeh.models import UIElement, ColumnDataSource, GlyphRenderer, Slider
from bokeh.plotting import figure as Figure
from numpy.typing import NDArray

from chmap.config import ChannelMapEditorConfig
from chmap.probe import ProbeDesp
from chmap.util.bokeh_util import ButtonFactory
from chmap.util.utils import is_recursive_called

__all__ = ['ViewBase', 'StateView', 'BoundaryState', 'BoundView']


class ViewBase(metaclass=abc.ABCMeta):
    visible: bool

    def __init__(self, config: ChannelMapEditorConfig):
        pass

    @abc.abstractmethod
    def setup(self, **kwargs) -> list[UIElement]:
        pass

    def plot(self, f: Figure, **kwargs):
        pass

    def update(self):
        pass


S = TypeVar('S')


class StateView(Generic[S], metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def save_state(self) -> S:
        """

        :return:
        """
        pass

    @abc.abstractmethod
    def restore_state(self, state: S):
        pass


class DynamicView:
    def on_probe_update(self, probe: ProbeDesp):
        pass


class BoundaryState(TypedDict):
    dx: float
    dy: float
    sx: float
    sy: float
    rt: float


class BoundView(ViewBase, metaclass=abc.ABCMeta):
    data_boundary: ColumnDataSource
    render_boundary: GlyphRenderer

    def __init__(self, config: ChannelMapEditorConfig):
        super().__init__(config)

        self.data_boundary = ColumnDataSource(data=dict(x=[0], y=[0], w=[0], h=[0], r=[0], sx=[1], sy=[1]))

    @property
    def visible(self) -> bool:
        try:
            return self.render_boundary.visible
        except AttributeError:
            return False

    @visible.setter
    def visible(self, v: bool):
        try:
            self.render_boundary.visible = v
        except AttributeError:
            pass

    @property
    @abc.abstractmethod
    def width(self) -> float:
        pass

    @property
    @abc.abstractmethod
    def height(self) -> float:
        pass

    def get_boundary_state(self) -> BoundaryState:
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
        self.render_boundary = f.rect(
            'x', 'y', 'w', 'h', 'r', source=self.data_boundary,
            color=boundary_color, fill_alpha=0, angle_units='deg',
        )
        self.data_boundary.on_change('data', self.on_boundary_change)

        from bokeh.models import tools
        f.tools.insert(-2, tools.BoxEditTool(
            description=boundary_desp,
            renderers=[self.render_boundary], num_objects=1
        ))

    boundary_rotate_slider: Slider
    boundary_scale_slider: Slider

    def setup_slider(self, slider_width: int = 300) -> list[UIElement]:
        new_btn = ButtonFactory(min_width=100, width_policy='min')

        self.boundary_rotate_slider = Slider(
            start=-25,
            end=25,
            step=1,
            value=0,
            title='image rotation (deg)',
            width=slider_width,
        )
        self.boundary_rotate_slider.on_change('value', self.on_boundary_rotate)

        #
        self.boundary_scale_slider = Slider(
            start=-1,
            end=1,
            step=0.01,
            value=0,
            title='image scale (log)',
            width=slider_width,
        )
        self.boundary_scale_slider.on_change('value', self.on_boundary_scale)

        reset_imr = new_btn('reset', self.on_reset_boundary_rotate)
        reset_ims = new_btn('reset', self.on_reset_boundary_scale)

        from bokeh.layouts import row
        return [
            row(reset_imr, self.boundary_rotate_slider),
            row(reset_ims, self.boundary_scale_slider),
        ]

    # noinspection PyUnusedLocal
    def on_boundary_rotate(self, prop: str, old: int, s: int):
        if is_recursive_called():
            return

        self.update_boundary_transform(rt=s)

    # noinspection PyUnusedLocal
    def on_boundary_scale(self, prop: str, old: float, s: float):
        if is_recursive_called():
            return

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

    # noinspection PyUnusedLocal
    def on_boundary_change(self, prop: str, old: dict, value: dict[str, list[float]]):
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
