from typing import TypedDict, Final, get_args, Literal

import matplotlib.pyplot as plt
import numpy as np
from bokeh.models import ColumnDataSource, GlyphRenderer, UIElement, Select, Slider, MultiChoice, Div
from iblatlas.atlas import AllenAtlas
from iblatlas.regions import BrainRegions
from numpy.typing import NDArray

from chmap.config import parse_cli, ChannelMapEditorConfig
from chmap.util.bokeh_util import as_callback, ButtonFactory, SliderFactory, is_recursive_called, new_help_button
from chmap.views.base import Figure, BoundView, StateView, BoundaryState
from chmap.views.image_plt import get_current_plt_image

__all__ = ['IblAtlasBrainView', 'IblAtlasBrainViewState']

SLICE = Literal['coronal', 'sagittal', 'horizontal', 'top']


class IblAtlasBrainViewState(TypedDict, total=False):
    atlas_brain: str
    brain_slice: str | None
    slice_plane: int | None
    image_dx: float
    image_dy: float
    image_sx: float
    image_sy: float
    image_rt: float
    regions: list[str]


class IblAtlasBrainView(BoundView, StateView[IblAtlasBrainViewState]):
    brain: Final[AllenAtlas]

    data_brain: ColumnDataSource
    render_brain: GlyphRenderer

    def __init__(self, config: ChannelMapEditorConfig, *, logger: str = 'chmap.view.ibl'):
        super().__init__(config, logger=logger)
        self.logger.warning('it is an experimental feature.')

        self._brain_name: str = str(config.atlas_name)
        self.brain = self._init_allen_atlas(config)
        self.regions = BrainRegions()
        self.data_brain = ColumnDataSource(data=dict(image=[], x=[], y=[], dw=[], dh=[]))

    def _init_allen_atlas(self, config: ChannelMapEditorConfig) -> AllenAtlas:
        try:
            res_um = int(config.atlas_name)
        except ValueError:
            if config.atlas_name in ('Needles',):
                from iblatlas.atlas import NeedlesAtlas
                self.logger.debug('init(NeedlesAtlas)')
                return NeedlesAtlas()

            elif config.atlas_name in ('MRI', 'MRIToronto'):
                from iblatlas.atlas import MRITorontoAtlas
                self.logger.debug('init(MRITorontoAtlas)')
                return MRITorontoAtlas()

            # elif config.atlas_name in ('Franklin&Paxinos',):
            #     from iblatlas.atlas import FranklinPaxinosAtlas
            #     self.logger.debug('init(FranklinPaxinosAtlas)')
            #     self.brain = FranklinPaxinosAtlas()

            else:
                raise

        else:
            from iblatlas.atlas import AllenAtlas
            self.logger.debug('init(AllenAtlas(res_um=%d))', res_um)
            return AllenAtlas(res_um=res_um)

    @property
    def name(self) -> str:
        return 'IBL Atlas'

    # ========== #
    # properties #
    # ========== #

    @property
    def width(self) -> float:
        # (ap, ml, dv)
        match self.slice_select.value:
            case 'coronal':  # ml
                a, b = self.brain.bc.ylim
            case 'sagittal':  # ap
                a, b = self.brain.bc.xlim
            case 'horizontal':  # ml?
                a, b = self.brain.bc.ylim
            case 'top':  # ml?
                a, b = self.brain.bc.ylim
            case _:
                return 0

        return abs(b - a) * 1e6

    @property
    def height(self) -> float:
        # (ap, ml, dv)
        match self.slice_select.value:
            case 'coronal':  # dv
                a, b = self.brain.bc.zlim
            case 'sagittal':  # dv
                a, b = self.brain.bc.zlim
            case 'horizontal':  # ap?
                a, b = self.brain.bc.xlim
            case 'top':  # ap?
                a, b = self.brain.bc.xlim
            case _:
                return 0

        return abs(b - a) * 1e6

    # ============= #
    # UI components #
    # ============= #

    slice_select: Select
    volume_select: Select
    plane_slider: Slider
    region_choose: MultiChoice

    def _setup_render(self, f: Figure,
                      boundary_color: str = 'black',
                      **kwargs):
        self.render_brain = f.image_rgba(
            'image', x='x', y='y', dw='dw', dh='dh', source=self.data_brain,
            global_alpha=0.5, syncable=False,
        )

        self.setup_boundary(f, boundary_color=boundary_color, boundary_desp='drag atlas brain image')

    def _setup_content(self, slider_width: int = 300,
                       rotate_steps=(-1000, 1000, 5),
                       **kwargs) -> list[UIElement]:
        new_btn = ButtonFactory(min_width=100, min_height=30, width_policy='min', height_policy='min')
        new_slider = SliderFactory(width=slider_width, align='end')

        #
        slice_view_options = list(get_args(SLICE))
        self.slice_select = Select(
            value=slice_view_options[0],
            options=slice_view_options,
            width=100,
        )
        self.slice_select.on_change('value', as_callback(self._on_slice_selected))

        #
        self.volume_select = Select(
            value='image',
            options=['image', 'annotation', 'surface', 'boundary', 'volume', 'value'],
            width=100,
        )
        self.volume_select.on_change('value', as_callback(self.update_image))

        #
        self.plane_slider = new_slider('Slice Plane', (0, 1, 1, 0), self.update_image)

        #
        self.region_choose = MultiChoice(
            options=list(sorted(set(self.regions.acronym))),
            width=400,
        )
        self.region_choose.on_change('value', as_callback(self.update_image))

        #
        from bokeh.layouts import row
        return [
            row(self.slice_select, self.plane_slider, self.volume_select),
            # row(*self.setup_rotate_slider(new_btn=new_btn, new_slider=new_slider)),
            row(*self.setup_scale_slider(new_btn=new_btn, new_slider=new_slider)),
            row(Div(text='mask'), self.region_choose, new_help_button('type region names to color label the corresponding areas.'))
        ]

    def _on_slice_selected(self, s: str):
        if is_recursive_called():
            return

        old_state = self.get_boundary_state()

        match s:
            case 'coronal':
                a, b = self.brain.bc.xlim
            case 'sagittal':
                a, b = self.brain.bc.ylim
            case 'horizontal':
                a, b = self.brain.bc.zlim
            case 'top':
                a, b = self.brain.bc.zlim
            case _:
                return

        self.plane_slider.start = min(a, b) * 1e6
        self.plane_slider.end = max(a, b) * 1e6

        self.update_boundary_transform(s=(old_state['sx'], old_state['sy']))

    # ========= #
    # load/save #
    # ========= #

    def save_state(self) -> IblAtlasBrainViewState:
        boundary = self.get_boundary_state()

        self.logger.debug('save()')
        return IblAtlasBrainViewState(
            atlas_brain=self._brain_name,
            brain_slice=self.slice_select.value,
            slice_plane=self.plane_slider.value,
            image_dx=boundary['dx'],
            image_dy=boundary['dy'],
            image_sx=boundary['sx'],
            image_sy=boundary['sy'],
            image_rt=boundary['rt'],
            regions=list(self.region_choose.value)
        )

    def restore_state(self, state: IblAtlasBrainViewState):
        if self._brain_name != state['atlas_brain']:
            raise RuntimeError()

        self.logger.debug('restore()')
        self.slice_select.value = state['brain_slice']
        self.plane_slider.value = state['slice_plane']

        self.update_boundary_transform(
            p=(state['image_dx'], state['image_dy']),
            s=(state['image_sx'], state['image_sx']),
            # rt=state['image_rt'],
        )

        self.region_choose.value = state['regions']

    # ================= #
    # boundary updating #
    # ================= #

    def on_boundary_transform(self, state: BoundaryState):
        super().on_boundary_transform(state)

        self.update_image()

    # ============== #
    # image updating #
    # ============== #

    def start(self):
        self._on_slice_selected('coronal')
        self.update_boundary_transform()

    def update_image(self):
        # TODO width/height is inner axes size, not figure size.
        fg, ax = plt.subplots(gridspec_kw=dict(top=1, bottom=0.05, left=0.05, right=1))

        try:
            volume = self.volume_select.value
            if len(regions := self.region_choose.value) == 0:
                match self.slice_select.value:
                    case 'coronal':
                        p = self.plane_slider.value / 1e6
                        self.brain.plot_cslice(p, volume=volume, ax=ax)
                    case 'sagittal':
                        p = self.plane_slider.value / 1e6
                        self.brain.plot_sslice(p, volume=volume, ax=ax)
                    case 'horizontal':
                        p = self.plane_slider.value / 1e6
                        self.brain.plot_hslice(p, volume=volume, ax=ax)
                    case 'top':
                        self.brain.plot_top(volume=volume, ax=ax)
                    case _:
                        self._update_image(None)
                        return
            else:
                from iblatlas.plots import plot_scalar_on_slice
                plot_scalar_on_slice(
                    *self._collect_choose_region(regions),
                    coord=self.plane_slider.value,
                    slice=self.slice_select.value,
                    hemisphere='both',
                    background=volume,
                    brain_atlas=self.brain,
                    ax=ax,
                )

            self._update_image(get_current_plt_image(fg, dpi=100))
        finally:
            plt.close(fg)

    def _collect_choose_region(self, regions: list[str]) -> tuple[list[str], NDArray[np.int_]]:
        if len(regions) == 0:
            return [], []

        ret_r = []
        ret_v = []

        for i, region in enumerate(regions):
            sub = self.regions.descendants(self.regions.acronym2id(region))['acronym']
            ret_r.extend(list(sub))
            ret_v.extend([i] * len(sub))

        return ret_r, np.array(ret_v)

    def _update_image(self, image_data: NDArray[np.uint] | None):
        if image_data is None:
            self.visible = False
            self.data_brain.data = dict(image=[], dw=[], dh=[], x=[], y=[])
        else:
            self.data_brain.data = self.transform_image_data(image_data)


if __name__ == '__main__':
    import sys

    from chmap.main_bokeh import main

    main(parse_cli([
        *sys.argv[1:],
        '--debug',
        '--view=-',
        '--view=chmap.views.atlas_ibl:IblAtlasBrainView',
    ]))
