import math
from typing import get_args, TypedDict

import numpy as np
from bokeh.models import ColumnDataSource, GlyphRenderer, Select, Slider
from bokeh.plotting import figure as Figure
from numpy.typing import NDArray
from scipy.ndimage import rotate

from chmap.util.atlas_brain import BrainGlobeAtlas
from chmap.util.atlas_slice import SlicePlane, SLICE, SliceView
from chmap.util.utils import is_recursive_called

__all__ = ['AtlasBrainView', 'AtlasBrainViewState']


class AtlasBrainViewState(TypedDict):
    atlas_brain: str
    brain_slice: SLICE | None
    slice_plane: int | None
    slice_rot_w: int | None
    slice_rot_h: int | None
    image_dx: float
    image_dy: float
    image_sx: float
    image_sy: float
    image_rt: float


class AtlasBrainView:
    data_brain: ColumnDataSource
    data_brain_boundary: ColumnDataSource

    render_brain: GlyphRenderer
    render_brain_boundary: GlyphRenderer

    def __init__(self, brain: BrainGlobeAtlas):
        self.brain: BrainGlobeAtlas = brain
        self.data_brain = ColumnDataSource(data=dict(image=[], x=[], y=[], dw=[], dh=[]))
        self.data_brain_boundary = ColumnDataSource(data=dict(x=[], y=[], w=[], h=[], r=[]))

        self._brain_view: SliceView | None = None
        self._brain_slice: SlicePlane | None = None

        self._dx: float = 0
        self._dy: float = 0
        self._sx: float = 1
        self._sy: float = 1
        self._rt: float = 0

    # ========= #
    # load/save #
    # ========= #

    def save_state(self) -> AtlasBrainViewState:
        return AtlasBrainViewState(
            atlas_brain=self.brain.atlas_name,
            brain_slice=None if (p := self._brain_view) is None else p.name,
            slice_plane=None if (p := self._brain_slice) is None else p.plane,
            slice_rot_w=None if p is None else p.dw,
            slice_rot_h=None if p is None else p.dh,
            image_dx=self._dx,
            image_dy=self._dy,
            image_sx=self._sx,
            image_sy=self._sy,
            image_rt=self._rt,
        )

    def restore_state(self, state: AtlasBrainViewState):
        if self.brain.atlas_name != state['atlas_brain']:
            raise RuntimeError()

        self.update_brain_view(state['brain_slice'], update_image=False)
        self.update_brain_slice(state['slice_plane'], update_image=False)

        dp = state['slice_plane']
        dw = state['slice_rot_w']
        dh = state['slice_rot_h']
        brain_slice = self.brain_view.plane_at(dp).with_offset(dw, dh)
        self.update_brain_slice(brain_slice, update_image=False)

        self.update_image_transform(p=(state['image_dx'], state['image_dy']), s=(state['image_sx'], state['image_sx']), rt=state['image_rt'])

    # ============= #
    # UI components #
    # ============= #

    slice_select: Select
    plane_slider: Slider
    rth_slider: Slider
    rtv_slider: Slider
    imr_slider: Slider
    ims_slider: Slider

    def setup(self, width: int = 300, rotate_steps=(-1000, 1000, 5)):
        #
        slice_view_options = list(get_args(SLICE))
        self.slice_select = Select(
            title='Slice view',
            value=slice_view_options[0],
            options=slice_view_options,
            width=100
        )
        self.slice_select.on_change('value', self._on_slice_selected)

        #
        self.plane_slider = Slider(
            start=0,
            end=1,
            step=1,
            value=0,
            title='Slice Plane',
            width=width,
            align='end',
        )
        self.plane_slider.on_change('value', self._on_slice_changed)

        #
        self.rth_slider = Slider(
            start=rotate_steps[0],
            end=rotate_steps[1],
            step=rotate_steps[2],
            value=0,
            title='horizontal rotation (um)',
            width=width,
        )
        self.rth_slider.on_change('value', self._on_diff_changed)

        #
        self.rtv_slider = Slider(
            start=rotate_steps[0],
            end=rotate_steps[1],
            step=rotate_steps[2],
            value=0,
            title='vertical rotation (um)',
            width=width,
        )
        self.rtv_slider.on_change('value', self._on_diff_changed)

        #
        self.imr_slider = Slider(
            start=-25,
            end=25,
            step=1,
            value=0,
            title='image rotation (deg)',
            width=width,
        )
        self.imr_slider.on_change('value', self._on_image_rotate)

        #
        self.ims_slider = Slider(
            start=-1,
            end=1,
            step=0.01,
            value=0,
            title='image scale (log)',
            width=width,
        )
        self.ims_slider.on_change('value', self._on_image_scale)

    # noinspection PyUnusedLocal
    def _on_slice_selected(self, prop: str, old: str, s: str):
        if is_recursive_called():
            return

        self.update_brain_view(s)

    # noinspection PyUnusedLocal
    def _on_slice_changed(self, prop: str, old: int, s: int):
        if is_recursive_called():
            return

        if (p := self._brain_slice) is not None:
            q = p.slice.plane_at(int(s)).with_offset(p.dw, p.dh)
            self.update_brain_slice(q)

    # noinspection PyUnusedLocal
    def _on_diff_changed(self, prop: str, old: int, s: int):
        if is_recursive_called():
            return

        if (p := self._brain_slice) is not None:
            r = p.slice.resolution
            x = int(self.rth_slider.value / r)
            y = int(self.rtv_slider.value / r)
            q = p.with_offset(x, y)
            self.update_brain_slice(q)

    def _on_image_rotate(self, prop: str, old: int, s: int):
        if is_recursive_called():
            return

        self.update_image_rotate(s)

    def _on_image_scale(self, prop: str, old: float, s: float):
        if is_recursive_called():
            return

        self.update_image_scale(math.pow(10, s))

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

    def reset_rth(self):
        try:
            self.rth_slider.value = 0
        except AttributeError:
            pass

    def reset_rtv(self):
        try:
            self.rtv_slider.value = 0
        except AttributeError:
            pass

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
            return self.render_brain.visible
        except AttributeError:
            return False

    @visible.setter
    def visible(self, v: bool):
        try:
            self.render_brain.visible = v
        except AttributeError:
            pass

    def plot(self, f: Figure, palette: str = 'Greys256'):
        self.render_brain = f.image(
            'image', x='x', y='y', dw='dw', dh='dh', source=self.data_brain,
            palette=palette, level="image", global_alpha=0.5, syncable=False,
        )
        self.render_brain_boundary = f.rect(
            'x', 'y', 'w', 'h', 'r', source=self.data_brain_boundary,
            fill_alpha=0, angle_units='deg',
        )
        self.data_brain_boundary.on_change('data', self._on_boundary_change)

    def boundary_tool(self):
        from bokeh.models import tools
        return tools.BoxEditTool(
            description='drag atlas brain image',
            renderers=[self.render_brain_boundary], num_objects=1
        )

    # ============================= #
    # properties and update methods #
    # ============================= #

    @property
    def width(self) -> float:
        try:
            return self._brain_slice.width
        except TypeError:
            return 0

    @property
    def height(self) -> float:
        try:
            return self._brain_slice.height
        except TypeError:
            return 0

    @property
    def brain_view(self) -> SliceView:
        return self._brain_view

    def update_brain_view(self, view: SLICE | SliceView, *,
                          update_image=True):
        if is_recursive_called():
            return

        if isinstance(view, str):
            view = SliceView(self.brain, view)

        self._brain_view = view

        try:
            self.slice_select.value = view.name
        except AttributeError:
            pass

        try:
            self.plane_slider.title = f'Slice Plane (1/{view.resolution} um)'
            self.plane_slider.end = view.n_plane
        except AttributeError:
            pass

        try:
            self.rth_slider.step = view.resolution
        except AttributeError:
            pass

        try:
            self.rtv_slider.step = view.resolution
        except AttributeError:
            pass

        if (p := self._brain_slice) is not None:
            self._brain_slice = None  # avoid self.plane_slider.value invoke updating methods
            p = view.plane_at(p.coor_on())

            try:
                self.plane_slider.value = p.plane
            except AttributeError:
                pass
        else:
            self._brain_slice = None  # avoid self.plane_slider.value invoke updating methods
            p = view.plane_at(view.n_plane // 2)

        self.update_brain_slice(p, update_image=update_image)
        self.update_image_transform()

    @property
    def brain_slice(self) -> SlicePlane | None:
        return self._brain_slice

    def update_brain_slice(self, plane: int | SlicePlane | None, *,
                           update_image=True):
        if is_recursive_called():
            return

        if isinstance(plane, int):
            plane = self.brain_view.plane(plane)

        self._brain_slice = plane

        try:
            self.plane_slider.value = plane.plane
        except AttributeError:
            pass
        try:
            self.rth_slider.value = plane.dw * plane.resolution
        except AttributeError:
            pass
        try:
            self.rtv_slider.value = plane.dh * plane.resolution
        except AttributeError:
            pass

        if update_image:
            if plane is None:
                self.update_image(None)
            else:
                self.update_image(plane.image)

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

        if update_image and (plane := self._brain_slice) is not None:
            self.update_image(plane.image)

    def update_image(self, image_data: NDArray[np.uint] | None):
        if image_data is None:
            self.data_brain.data = dict(image=[], dw=[], dh=[], x=[], y=[])
            self.data_brain_boundary.data = dict(x=[], y=[], w=[], h=[], r=[])
        else:
            w = self.width * self._sx
            h = self.height * self._sy
            x = self._dx
            y = self._dy
            image = np.flipud(image_data)

            if self._rt != 0:
                image = rotate(image, -self._rt, reshape=False)

            self.data_brain.data = dict(
                image=[image],
                dw=[w], dh=[h], x=[x], y=[y]
            )