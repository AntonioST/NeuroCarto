from typing import TypedDict, Final, get_args

import matplotlib.pyplot as plt
import numpy as np
from bokeh.models import ColumnDataSource, GlyphRenderer, UIElement, Select, Slider
from bokeh.plotting import figure as Figure
from iblatlas.atlas import BrainAtlas
from numpy.typing import NDArray

from chmap.config import parse_cli, ChannelMapEditorConfig
from chmap.util.atlas_slice import SLICE
from chmap.util.bokeh_util import as_callback, ButtonFactory, SliderFactory, is_recursive_called
from chmap.views.base import BoundView, StateView, BoundaryState

__all__ = ['IblAtlasBrainView', 'IblAtlasBrainViewState']

from chmap.views.image_plt import get_current_plt_image


class IblAtlasBrainViewState(TypedDict, total=False):
    atlas_brain: str
    brain_slice: str | None
    slice_plane: int | None
    slice_rot_w: int | None
    slice_rot_h: int | None
    image_dx: float
    image_dy: float
    image_sx: float
    image_sy: float
    image_rt: float
    regions: list[str]


class IblAtlasBrainView(BoundView, StateView[IblAtlasBrainViewState]):
    brain: Final[BrainAtlas]

    data_brain: ColumnDataSource
    render_brain: GlyphRenderer

    def __init__(self, config: ChannelMapEditorConfig, *, logger: str = 'chmap.view.ibl'):
        super().__init__(config, logger=logger)

        try:
            res_um = int(config.atlas_name)
        except ValueError:
            if config.atlas_name in ('Needles',):
                from iblatlas.atlas import NeedlesAtlas
                self.logger.debug('init(NeedlesAtlas)')
                self.brain = NeedlesAtlas()

            elif config.atlas_name in ('MRI', 'MRIToronto'):
                from iblatlas.atlas import MRITorontoAtlas
                self.logger.debug('init(MRITorontoAtlas)')
                self.brain = MRITorontoAtlas()

            elif config.atlas_name in ('Franklin&Paxinos',):
                from iblatlas.atlas import FranklinPaxinosAtlas
                self.logger.debug('init(FranklinPaxinosAtlas)')
                self.brain = FranklinPaxinosAtlas()

            else:
                raise

        else:
            from iblatlas.atlas import AllenAtlas
            self.logger.debug('init(AllenAtlas(res_um=%d))', res_um)
            self.brain = AllenAtlas(res_um=res_um)

        self.data_brain = ColumnDataSource(data=dict(image=[], x=[], y=[], dw=[], dh=[]))

        self._axis = 0
        self._volume = 'image'

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
            case 'transverse':  # ml?
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
            case 'transverse':  # ap?
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
            options=['image', 'annotation'],
            width=100,
        )
        self.volume_select.on_change('value', as_callback(self.update_image))

        #
        self.plane_slider = new_slider('Slice Plane', (0, 1, 1, 0), self.update_image)

        #
        from bokeh.layouts import row
        return [
            row(self.slice_select, self.plane_slider, self.volume_select),
            row(*self.setup_rotate_slider(new_btn=new_btn, new_slider=new_slider)),
            row(*self.setup_scale_slider(new_btn=new_btn, new_slider=new_slider)),
        ]

    def _on_slice_selected(self, s: str):
        if is_recursive_called():
            return

        match s:
            case 'coronal':
                a, b = self.brain.bc.xlim
            case 'sagittal':
                a, b = self.brain.bc.ylim
            case 'transverse':
                a, b = self.brain.bc.zlim
            case _:
                return

        self.plane_slider.start = min(a, b) * 1e6
        self.plane_slider.end = max(a, b) * 1e6

        self.update_image()
        self.update_boundary_transform()  # TODO width/height not keep

    # ========= #
    # load/save #
    # ========= #

    def save_state(self) -> IblAtlasBrainViewState:
        return IblAtlasBrainViewState()

    def restore_state(self, state: IblAtlasBrainViewState):
        pass

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
        fg, ax = plt.subplots()

        try:
            volume = self.volume_select.value
            match self.slice_select.value:
                case 'coronal':
                    p = self.plane_slider.value / 1e6
                    self.brain.plot_cslice(p, volume=volume, ax=ax)
                case 'sagittal':
                    p = self.plane_slider.value / 1e6
                    self.brain.plot_sslice(p, volume=volume, ax=ax)
                case 'transverse':
                    p = self.plane_slider.value / 1e6
                    self.brain.plot_hslice(p, volume=volume, ax=ax)
                case _:
                    self._update_image(None)
                    return

            # plt.show()
            self._update_image(get_current_plt_image(fg))
        finally:
            plt.close(fg)

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
