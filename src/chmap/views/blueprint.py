from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from bokeh.models import ColumnDataSource, GlyphRenderer, Div
from numpy.typing import NDArray

from chmap.config import parse_cli, ChannelMapEditorConfig
from chmap.util.util_blueprint import BlueprintFunctions
from chmap.views.base import Figure, ViewBase, InvisibleView, DynamicView

if TYPE_CHECKING:
    from chmap.probe_npx.npx import ChannelMap

__all__ = ['BlueprintView']


class BlueprintView(ViewBase, InvisibleView, DynamicView):
    """Show blueprint beside."""

    data_blueprint: ColumnDataSource
    render_blueprint: GlyphRenderer

    def __init__(self, config: ChannelMapEditorConfig):
        super().__init__(config, logger='chmap.view.plot_blueprint')

        # xs = ys = [shape_group: [shape: [exterior:[p, ...], holes:[p, ...], ...]]]
        # we use as [[shape: [[p,...]] ]]
        self.data_blueprint = ColumnDataSource(data=dict(xs=[], ys=[], c=[]))

    @property
    def name(self) -> str:
        return "Blueprint"

    @property
    def description(self) -> str | None:
        return 'plot blueprint beside'

    # ============= #
    # UI components #
    # ============= #

    category_legend_div: Div

    def _setup_render(self, f: Figure, **kwargs):
        self.render_blueprint = f.multi_polygons(
            xs='xs', ys='ys', fill_color='c', source=self.data_blueprint,
            line_width=0, fill_alpha=0.5,
        )

    def _setup_content(self, **kwargs):
        self.category_legend_div = Div(
            text="",
            stylesheets=["""
                div.chmap-blueprint-legend {
                    display: flex;
                    flex-direction: row; 
                    padding-left: 1em;
                }
                div.chmap-blueprint-legend div.chmap-blueprint-legend-name {
                    margin-left: 0.5em;
                }
                div.chmap-blueprint-legend div.chmap-blueprint-legend-color {
                    margin-left: 1em;
                    width: var(--line-height-computed, 14pt);
                    height: var(--line-height-computed, 14pt);
                }
            """]
        )
        return [self.category_legend_div]

    def set_category_colr(self, category: dict[int, str]):
        self.category_legend_div.text = "".join([
            '<div class="chmap-blueprint-legend">',
            *[
                (f'<div class="chmap-blueprint-legend-color" style="background-color: {color};"></div>'
                 f'<div class="chmap-blueprint-legend-name">{name}</div>')
                for code, color in category.items()
                if (name := code if isinstance(code, str) else self.cache_probe.category_description(code)) is not None
            ],
            '</div>'
        ])

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
            bp = BlueprintFunctions(probe, chmap)
            bp.set_blueprint(electrodes)
            self.data_blueprint.data = self.plot_channelmap_npx(bp)
        else:
            self.reset_blueprint()

    def reset_blueprint(self):
        self.data_blueprint.data = dict(xs=[], ys=[], c=[])

    def plot_channelmap(self, bp: BlueprintFunctions) -> dict:
        setting = {
            bp.CATE_SET: 'green',
            bp.CATE_FORBIDDEN: 'pink',
        }
        self.set_category_colr(setting)
        return self._plot_blueprint(bp, setting)

    def plot_channelmap_npx(self, bp: BlueprintFunctions) -> dict:
        from chmap.probe_npx.npx import ProbeType

        setting = {
            bp.CATE_FULL: 'green',
            bp.CATE_HALF: 'orange',
            bp.CATE_QUARTER: 'blue',
            bp.CATE_FORBIDDEN: 'pink',
        }
        self.set_category_colr(setting)

        probe_type: ProbeType = bp.channelmap.probe_type
        c_space = probe_type.c_space
        r_space = probe_type.r_space
        size = c_space // 2, r_space // 2
        offset = c_space + c_space * probe_type.n_col_shank

        blueprint = bp.set(bp.blueprint(), bp.CATE_SET, bp.CATE_FULL)
        return self._plot_blueprint(bp, setting, blueprint, size, offset)

    def _plot_blueprint(self, bp: BlueprintFunctions,
                        setting: dict[int, str],
                        blueprint: NDArray[np.int_] = None,
                        size: tuple[int, int] = None,
                        offset: int = None):
        categories = list(setting.keys())
        colors = list(setting.values())

        if blueprint is None:
            blueprint = bp.blueprint()

        if size is None:
            size = int(bp.dx), int(bp.dy)

        if offset is None:
            offset = 4 * bp.dx

        edges = bp.clustering_edges(blueprint, categories)
        edges = [it.set_corner(size) for it in edges]

        xs = [[], [], [], []]
        ys = [[], [], [], []]

        #
        # |<-c ->|<-2w->|
        # +------+------+
        # |<-0 ->|<-1 ->| column
        #    0            origin x

        for edge in edges:
            i = categories.index(edge.category)

            xs[i].append([edge.x + offset])
            ys[i].append([edge.y])

        return dict(xs=xs, ys=ys, c=colors)


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
