from __future__ import annotations

from typing import get_args, TypedDict, Final

import numpy as np
from bokeh.models import ColumnDataSource, GlyphRenderer, Select, Slider, UIElement, MultiChoice, Div, CheckboxGroup
from bokeh.plotting import figure as Figure
from numpy.typing import NDArray

from chmap.config import ChannelMapEditorConfig
from chmap.util.atlas_brain import BrainGlobeAtlas, get_atlas_brain
from chmap.util.atlas_slice import SlicePlane, SLICE, SliceView
from chmap.util.atlas_struct import Structures
from chmap.util.bokeh_util import ButtonFactory, SliderFactory, as_callback, is_recursive_called, new_help_button
from chmap.views.base import StateView, BoundView, BoundaryState

__all__ = ['AtlasBrainView', 'AtlasBrainViewState']


class AtlasBrainViewState(TypedDict, total=False):
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
    regions: list[str]


class AtlasBrainView(BoundView, StateView[AtlasBrainViewState]):
    """
    Atlas mouse brain displaying view component.

    Used command-line arguments:
    * '--atlas' : brain name.
    * '--atlas-root' : data saving directory.

    Event Call chain
    ----------------

    (boundary UI components)

        -/-> (event callback)
            -> update_boundary_transform()
                -> on_boundary_transform()

    (atlas UI components)

        (slice_select) -/-> (event callback)
            -> update_brain_view()
                -> update_brain_slice(update_image=False)
                -> update_boundary_transform()
                    -> on_boundary_transform()
                        -> update_image()
                            -> update_region_image()
        (region_choose) -/-> (event callback)
            -> update_region_image()
        (other) -/-> (event callback)
            -> update_brain_slice()
                -> update_image()
                    -> update_region_image()

    """

    brain: Final[BrainGlobeAtlas]

    data_brain: ColumnDataSource
    render_brain: GlyphRenderer

    data_region: ColumnDataSource
    render_region: GlyphRenderer

    def __init__(self, config: ChannelMapEditorConfig, *, logger: str = 'chmap.view.atlas'):
        super().__init__(config, logger=logger)

        self.logger.debug('init(%s)', config.atlas_name)
        self.brain = get_atlas_brain(config.atlas_name, config.atlas_root)
        self._structure = Structures.of(self.brain)

        self.data_brain = ColumnDataSource(data=dict(image=[], x=[], y=[], dw=[], dh=[]))
        self.data_region = ColumnDataSource(data=dict(image=[], x=[], y=[], dw=[], dh=[]))

        self._brain_view: SliceView | None = None
        self._brain_slice: SlicePlane | None = None
        self._regions: dict[str, int] = {}

    @property
    def name(self) -> str:
        return 'Atlas brain'

    # ========== #
    # properties #
    # ========== #

    @property
    def width(self) -> float:
        try:
            return self._brain_slice.width
        except (TypeError, AttributeError):
            return 0

    @property
    def height(self) -> float:
        try:
            return self._brain_slice.height
        except (TypeError, AttributeError):
            return 0

    @property
    def brain_view(self) -> SliceView:
        return self._brain_view

    @property
    def brain_slice(self) -> SlicePlane | None:
        return self._brain_slice

    # ============= #
    # UI components #
    # ============= #

    slice_select: Select
    plane_slider: Slider
    rotate_hor_slider: Slider
    rotate_ver_slider: Slider
    region_choose: MultiChoice

    checkbox_group: CheckboxGroup
    checkbox_groups: dict[str, UIElement]

    def _setup_render(self, f: Figure,
                      palette: str = 'Greys256',
                      palette_region: str = 'Turbo256',
                      boundary_color: str = 'black',
                      **kwargs):
        self.render_brain = f.image(
            'image', x='x', y='y', dw='dw', dh='dh', source=self.data_brain,
            palette=palette, level="image", global_alpha=0.5, syncable=False,
        )

        self.render_region = f.image(
            'image', x='x', y='y', dw='dw', dh='dh', source=self.data_region,
            palette=palette_region, level="image", global_alpha=0.3, syncable=False,
        )

        self.setup_boundary(f, boundary_color=boundary_color, boundary_desp='drag atlas brain image')

    def _setup_title(self, **kwargs) -> list[UIElement]:
        ret = super()._setup_title(**kwargs)

        self.checkbox_group = CheckboxGroup(labels=['Shear', 'Rotation', 'Scaling', 'Masking'], inline=True)
        self.checkbox_group.on_change('active', as_callback(self._on_checkbox_active))
        ret.insert(-1, self.checkbox_group)

        return ret

    def _setup_content(self, slider_width: int = 300,
                       rotate_steps=(-1000, 1000, 5),
                       **kwargs) -> list[UIElement]:
        from bokeh.layouts import row, column

        new_btn = ButtonFactory(min_width=100, min_height=30, width_policy='min', height_policy='min')
        new_slider = SliderFactory(width=slider_width, align='end')
        self.checkbox_groups = {}

        # slicing
        slice_view_options = list(get_args(SLICE))
        self.slice_select = Select(
            value=slice_view_options[0],
            options=slice_view_options,
            width=100,
        )
        self.slice_select.on_change('value', as_callback(self._on_slice_selected))
        self.plane_slider = new_slider('Slice Plane', (0, 1, 1, 0), self._on_slice_changed)

        # shear
        self.rotate_hor_slider = new_slider('horizontal rotation (um)', rotate_steps, self._on_rotate_changed)
        self.rotate_ver_slider = new_slider('vertical rotation (um)', rotate_steps, self._on_rotate_changed)

        reset_rth = new_btn('reset', self._on_reset_rotate_horizontal)
        reset_rtv = new_btn('reset', self._on_reset_rotate_vertical)

        self.checkbox_groups['Shear'] = col_shear = column(
            row(reset_rth, self.rotate_hor_slider),
            row(reset_rtv, self.rotate_ver_slider),
        )

        # rotation
        self.checkbox_groups['Rotation'] = col_rot = row(*self.setup_rotate_slider(new_btn=new_btn, new_slider=new_slider))

        # scaling
        self.checkbox_groups['Scaling'] = col_scale = row(*self.setup_scale_slider(new_btn=new_btn, new_slider=new_slider))

        # masking
        self.region_choose = MultiChoice(
            options=[
                ((structure := self._structure[acronym]).acronym, f'{acronym} ({structure.name})')
                for acronym in sorted(self._structure.regions)
            ],
            width=350,
        )
        self.region_choose.on_change('value', as_callback(self._on_region_choose))
        mask_help = new_help_button('type region names to color label the corresponding areas.')
        self.checkbox_groups['Masking'] = col_mask = row(Div(text='mask'), self.region_choose, mask_help)

        return [
            row(self.slice_select, self.plane_slider),
            col_shear,
            col_rot,
            col_scale,
            col_mask,
        ]

    def _on_slice_selected(self, s: str):
        if is_recursive_called():
            return

        self.update_brain_view(s)

    def _on_slice_changed(self, s: int):
        if is_recursive_called():
            return

        if (p := self._brain_slice) is not None:
            q = p.slice.plane_at(int(s)).with_offset(p.dw, p.dh)
            self.update_brain_slice(q)

    def _on_checkbox_active(self, active: list[int]):
        for i, n in enumerate(self.checkbox_group.labels):
            if (ui := self.checkbox_groups.get(n, None)) is not None:
                ui.visible = i in active

    def _on_rotate_changed(self):
        if is_recursive_called():
            return

        if (p := self._brain_slice) is not None:
            r = p.slice.resolution
            x = int(self.rotate_hor_slider.value / r)
            y = int(self.rotate_ver_slider.value / r)
            q = p.with_offset(x, y)
            self.update_brain_slice(q)

    def _on_reset_rotate_horizontal(self):
        try:
            self.rotate_hor_slider.value = 0
        except AttributeError:
            pass

    def _on_reset_rotate_vertical(self):
        try:
            self.rotate_ver_slider.value = 0
        except AttributeError:
            pass

    def _on_region_choose(self, value: list[str]):
        self.logger.debug('choose(%s)', value)

        self._regions = {
            region: i + 1
            for i, region in enumerate(value)
        }

        self.update_region_image()

    # ========= #
    # load/save #
    # ========= #

    def save_state(self) -> AtlasBrainViewState:
        boundary = self.get_boundary_state()

        self.logger.debug('save()')
        return AtlasBrainViewState(
            atlas_brain=self.brain.atlas_name,
            brain_slice=None if (p := self._brain_view) is None else p.name,
            slice_plane=None if (p := self._brain_slice) is None else p.plane,
            slice_rot_w=None if p is None else p.dw,
            slice_rot_h=None if p is None else p.dh,
            image_dx=boundary['dx'],
            image_dy=boundary['dy'],
            image_sx=boundary['sx'],
            image_sy=boundary['sy'],
            image_rt=boundary['rt'],
            regions=list(self._regions)
        )

    def restore_state(self, state: AtlasBrainViewState):
        if self.brain.atlas_name != state['atlas_brain']:
            raise RuntimeError()

        self.logger.debug('restore()')
        self.update_brain_view(state['brain_slice'])

        dp = state['slice_plane']
        dw = state['slice_rot_w']
        dh = state['slice_rot_h']
        brain_slice = self.brain_view.plane_at(dp).with_offset(dw, dh)
        self.update_brain_slice(brain_slice, update_image=False)

        self.update_boundary_transform(p=(state['image_dx'], state['image_dy']), s=(state['image_sx'], state['image_sx']), rt=state['image_rt'])

        try:
            self.region_choose.value = state['regions']
        except (AttributeError, KeyError):
            pass

    # ================ #
    # updating methods #
    # ================ #

    def start(self):
        self._on_checkbox_active([])

        if self.brain_view is None:
            try:
                view = self.slice_select.value
            except AttributeError:
                view = 'coronal'

            self.update_brain_view(view)
        elif (plane := self._brain_slice) is not None:
            self.update_image(plane.image)

    # ================== #
    # SliceView updating #
    # ================== #

    def update_brain_view(self, view: SLICE | SliceView | str):
        if is_recursive_called():
            return

        old_state = self.get_boundary_state()

        if isinstance(view, str):
            try:
                view = SliceView(self.brain, view)
            except ValueError as e:
                self.logger.warning('slice_view(%s)', view, exc_info=e)
                view = SliceView(self.brain, 'coronal')

        self._brain_view = view
        self.logger.debug('slice_view(%s)', view.name)

        try:
            self.slice_select.value = view.name
            self.plane_slider.title = f'Slice Plane (1/{view.resolution} um)'
            self.plane_slider.end = view.n_plane
            self.rotate_hor_slider.step = view.resolution
            self.rotate_ver_slider.step = view.resolution
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

        self.update_brain_slice(p, update_image=False)
        self.update_boundary_transform(s=(old_state['sx'], old_state['sy']))

    # =================== #
    # SlicePlane updating #
    # =================== #

    def update_brain_slice(self, plane: int | SlicePlane | None, *,
                           update_image=True):
        if is_recursive_called():
            return

        if isinstance(plane, int):
            plane = self.brain_view.plane(plane)

        self._brain_slice: SlicePlane = plane
        # self.logger.debug('slice_plane(%d)', plane.plane)

        if plane is not None:
            try:
                self.plane_slider.value = plane.plane
                self.rotate_hor_slider.value = plane.dw * plane.resolution
                self.rotate_ver_slider.value = plane.dh * plane.resolution
            except AttributeError:
                pass

        if update_image:
            if plane is None:
                self.update_image(None)
            else:
                self.update_image(plane.image)

    # ================= #
    # boundary updating #
    # ================= #

    def on_boundary_transform(self, state: BoundaryState):
        super().on_boundary_transform(state)
        if (plane := self._brain_slice) is not None:
            self.update_image(plane.image)

    # ============== #
    # image updating #
    # ============== #

    def update_image(self, image_data: NDArray[np.uint] | None):
        if image_data is None:
            self.visible = False
            self.data_brain.data = dict(image=[], dw=[], dh=[], x=[], y=[])
        else:
            self.data_brain.data = self.transform_image_data(np.flipud(image_data))

        self.update_region_image()

    def update_region_image(self):
        if len(self._regions) == 0 or (plane := self._brain_slice) is None:
            self.data_region.data = dict(image=[], dw=[], dh=[], x=[], y=[])
        else:
            view = SliceView(self.brain, plane.slice.name, self.brain.annotation)
            plane = view.plane_at(plane)
            self.data_region.data = self.transform_image_data(self.process_image_data(np.flipud(plane.image)))

    def process_image_data(self, image: NDArray[np.uint]) -> NDArray[np.uint]:
        return self._structure.image_annotation(image, self._regions, 0)
