from typing import get_args

import numpy as np
from bokeh.models import ColumnDataSource, GlyphRenderer, Select, Slider
from bokeh.plotting import figure as Figure
from numpy.typing import NDArray

from chmap.util.atlas_brain import BrainGlobeAtlas
from chmap.util.atlas_slice import SlicePlane, SLICE, SliceView


class AtlasBrainView:
    data_brain: ColumnDataSource
    render_brain: GlyphRenderer

    def __init__(self, brain: BrainGlobeAtlas):
        self.brain: BrainGlobeAtlas = brain
        self.data_brain = ColumnDataSource(data=dict(image=[], x=[], y=[], dw=[], dh=[]))
        self._brain_view: SliceView | None = None
        self._brain_slice: SlicePlane | None = None

        self._dx: float = 0
        self._dy: float = 0
        self._sx: float = 1
        self._sy: float = 1
        self._rt: float = 0

    # ============= #
    # UI components #
    # ============= #

    slice_select: Select
    plane_slider: Slider
    rth_slider: Slider
    rtv_slider: Slider

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

    # noinspection PyUnusedLocal
    def _on_slice_selected(self, prop: str, old: str, s: str):
        # if old != s:
        self.update_brain_view(s)

    # noinspection PyUnusedLocal
    def _on_slice_changed(self, prop: str, old: int, s: int):
        if (p := self._brain_slice) is not None:
            q = p.slice.plane_at(int(s)).with_offset(p.dw, p.dh)
            self.update_brain_slice(q)

    # noinspection PyUnusedLocal
    def _on_diff_changed(self, prop: str, old: int, s: int):
        if (p := self._brain_slice) is not None:
            r = p.slice.resolution
            x = int(self.rth_slider.value / r)
            y = int(self.rtv_slider.value / r)
            q = p.with_offset(x, y)
            self.update_brain_slice(q)

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
            palette=palette, level="image",
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

    def update_brain_view(self, view: SLICE | SliceView):
        if isinstance(view, str):
            print('update_brain_view', view)
            view = SliceView(view, self.brain.reference, self.brain.resolution[0])

        self._brain_view = view

        try:
            self.slice_select.value = view.name
        except AttributeError:
            pass

        try:
            self.plane_slider.end = view.n_plane
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

            try:
                p = self.plane_slider.value
            except AttributeError:
                p = view.n_plane // 2

            p = view.plane_at(p)

        self.update_brain_slice(p)

    @property
    def brain_slice(self) -> SlicePlane | None:
        return self._brain_slice

    def update_brain_slice(self, plane: SlicePlane | None):
        self._brain_slice = plane

        if plane is None:
            print('update_brain_slice', None)
            self.update_image(None)
        else:
            print('update_brain_slice', plane.plane)
            self.update_image(plane.image)

    @property
    def image_pos(self) -> tuple[float, float]:
        return self._dx, self._dy

    def update_image_pos(self, x: float, y: float):
        self._dx = x
        self._dy = y

        if (plane := self._brain_slice) is not None:
            self.update_image(plane.image)

    @property
    def image_scale(self) -> tuple[float, float]:
        return self._sx, self._sy

    def update_image_scale(self, s: float | tuple[float, float]):
        if isinstance(s, tuple):
            self._sx, self._sy = s
        else:
            self._sx = self._sy = float(s)

        if (plane := self._brain_slice) is not None:
            self.update_image(plane.image)

    @property
    def image_rotate(self) -> float:
        return self._rt

    def update_image_rotate(self, rt: float):
        self._rt = rt

        if (plane := self._brain_slice) is not None:
            self.update_image(plane.image)

    def update_image_transform(self, *,
                               p: tuple[float, float] = None,
                               s: float | tuple[float, float] = None,
                               rt: float = None):
        if p is not None:
            self._dx, self._dy = p

        if isinstance(s, tuple):
            self._sx, self._sy = s
        elif s is not None:
            self._sx = self._sy = float(s)

        if rt is not None:
            self._rt = rt

        if (plane := self._brain_slice) is not None:
            self.update_image(plane.image)

    def update_image(self, image_data: NDArray[np.int_] | None):
        if image_data is None:
            print('update image (Nan, NaN Nan, NaN)')
            self.data_brain.data = dict(image=[], dw=[], dh=[], x=[], y=[])
        else:
            w = self.width * self._sx
            h = self.height * self._sy
            x = self._dx
            y = self._dy
            print(f'update image ({x}, {y}, {w}, {h})')
            self.data_brain.data = dict(
                image=[np.flipud(image_data)],
                dw=[w], dh=[h], x=[x], y=[y]
            )
