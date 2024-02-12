from __future__ import annotations

from typing import TYPE_CHECKING, Any

from bokeh.models import ColumnDataSource, GlyphRenderer
from bokeh.plotting import figure as Figure

from chmap.config import parse_cli, ChannelMapEditorConfig
from chmap.util.util_blueprint import BlueprintFunctions
from chmap.views.base import ViewBase, InvisibleView, DynamicView

if TYPE_CHECKING:
    from chmap.probe_npx.npx import ChannelMap

__all__ = ['BlueprintView']


class BlueprintView(ViewBase, InvisibleView, DynamicView):
    data_blueprint: ColumnDataSource
    render_blueprint: GlyphRenderer

    def __init__(self, config: ChannelMapEditorConfig):
        super().__init__(config, logger='chmap.view.plot_blueprint')

        # xs = ys = [shape_group: [shape: [exterior:[p, ...], holes:[p, ...], ...]]]
        # we use as [[shape: [[p,...]] ]]
        self.data_blueprint = ColumnDataSource(data=dict(xs=[], ys=[], c=[]))

    @property
    def name(self) -> str | None:
        return "Blueprint"

    @property
    def description(self) -> str | None:
        return 'plot blueprint beside'

    # ============= #
    # UI components #
    # ============= #

    def _setup_render(self, f: Figure, **kwargs):
        self.render_blueprint = f.multi_polygons(
            xs='xs', ys='ys',
            source=self.data_blueprint, fill_color='c',
            line_width=0
        )

    # ======== #
    # updating #
    # ======== #

    cache_probe: Any
    cache_chmap: Any = None
    cache_blueprint: Any = None

    def on_visible(self, visible: bool):
        super().on_visible(visible)
        if visible and self.cache_chmap is not None:
            self.on_probe_update(self.cache_probe, self.cache_chmap, self.cache_blueprint)

    def on_probe_update(self, probe, chmap, electrodes):
        self.cache_probe = probe
        self.cache_chmap = chmap
        self.cache_blueprint = electrodes

        from chmap.probe_npx.npx import ChannelMap
        if not self.visible:
            return

        if isinstance(chmap, ChannelMap):
            bp = BlueprintFunctions.from_blueprint(electrodes, probe.all_possible_categories())
            self.data_blueprint.data = self.plot_npx_channelmap(bp, chmap)
        else:
            self.reset_blueprint()

    def reset_blueprint(self):
        self.data_blueprint.data = dict(xs=[], ys=[], c=[])

    def plot_npx_channelmap(self, bp: BlueprintFunctions, chmap: ChannelMap) -> dict:
        from chmap.probe_npx.desp import NpxProbeDesp

        probe_type = chmap.probe_type
        blueprint = bp.set(bp.blueprint(), NpxProbeDesp.CATE_SET, NpxProbeDesp.CATE_FULL)
        categories = [
            NpxProbeDesp.CATE_FULL, NpxProbeDesp.CATE_HALF, NpxProbeDesp.CATE_QUARTER, NpxProbeDesp.CATE_FORBIDDEN
        ]
        edges = bp.clustering_edges(blueprint, categories)

        w = probe_type.c_space // 2
        h = probe_type.r_space // 2
        edges = [it.set_corner((w, h)) for it in edges]

        xs = [[], [], [], []]
        ys = [[], [], [], []]
        color = ['green', 'orange', 'blue', 'pink']

        for edge in edges:
            i = categories.index(edge.category)
            xs[i].append([list(edge.x - probe_type.c_space * probe_type.n_col_shank)])
            ys[i].append([list(edge.y)])

        return dict(xs=xs, ys=ys, c=color)


if __name__ == '__main__':
    import sys

    from chmap.main_bokeh import main

    main(parse_cli([
        *sys.argv[1:],
        '-C', 'res',
        '--debug',
        '--view=-',
        '--view=chmap.views.blueprint:BlueprintView',
    ]))
