from __future__ import annotations

from typing import Any, Protocol, runtime_checkable, TypedDict

import numpy as np
from bokeh.models import ColumnDataSource, GlyphRenderer, Div, CheckboxGroup, UIElement
from numpy.typing import NDArray
from typing_extensions import NotRequired

from chmap.config import parse_cli, ChannelMapEditorConfig
from chmap.util.bokeh_util import as_callback
from chmap.util.util_blueprint import BlueprintFunctions
from chmap.util.utils import doc_link, SPHINX_BUILD
from chmap.views.base import Figure, ViewBase, InvisibleView, DynamicView

if SPHINX_BUILD:
    ProbeDesp = 'chmap.probe.ProbeDesp'

__all__ = [
    'BlueprintView',
    'ProbePlotBlueprintFunctor',
    'ProbePlotBlueprintReturn'
]


class ProbePlotBlueprintReturn(TypedDict):
    size: tuple[int, int]
    """electrode size (width, height)"""

    offset: int
    """x-axis offset"""

    categories: dict[int, str]
    """categories dict {C: html-color}"""

    blueprint: NotRequired[NDArray[np.int_]]
    """Array[C, N]"""

    legends: NotRequired[dict[str, str]]
    """categories legend dict {C-name: html-color}"""


@doc_link()
@runtime_checkable
class ProbePlotBlueprintFunctor(Protocol):
    """
    {ProbeDesp} extension protocol for plotting blueprint beside probe.
    """

    def view_ext_blueprint_view(self, chmap: Any, bp: BlueprintFunctions, options: list[str]) -> ProbePlotBlueprintReturn:
        """

        :param chmap:
        :param bp:
        :param options:
        :return:
        """
        pass


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

    checkbox_group: CheckboxGroup
    category_legend_div: Div

    def _setup_render(self, f: Figure, **kwargs):
        self.render_blueprint = f.multi_polygons(
            xs='xs', ys='ys', fill_color='c', source=self.data_blueprint,
            line_width=0, fill_alpha=0.5,
        )

    def _setup_title(self, **kwargs) -> list[UIElement]:
        ret = super()._setup_title(**kwargs)

        self.checkbox_group = CheckboxGroup(labels=['Conflict'], inline=True)
        self.checkbox_group.on_change('active', as_callback(self.update_blueprint))
        ret.insert(-1, self.checkbox_group)

        return ret

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

    def set_category_color(self, category: dict[int | str, str]):
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

        self.update_blueprint()

    def update_blueprint(self):
        if not self.visible:
            return

        labels = self.checkbox_group.labels
        options = [labels[it] for it in self.checkbox_group.active]

        if (chmap := self.cache_chmap) is None:
            self.reset_blueprint()
        else:
            bp = BlueprintFunctions(self.cache_probe, chmap)
            bp.set_blueprint(self.cache_blueprint)

            if isinstance((functor := self.cache_probe), ProbePlotBlueprintFunctor):
                ret: ProbePlotBlueprintReturn = functor.view_ext_blueprint_view(chmap, bp, options)

                setting = ret['categories']
                self.set_category_color(ret.get('legends', setting))
                data = self._plot_blueprint(bp, setting, ret.get('blueprint', None), ret['size'], ret['offset'])
            else:
                setting = {
                    bp.CATE_SET: 'green',
                    bp.CATE_FORBIDDEN: 'pink',
                }
                self.set_category_color(setting)
                data = self._plot_blueprint(bp, setting)

            self.data_blueprint.data = data

    def reset_blueprint(self):
        self.data_blueprint.data = dict(xs=[], ys=[], c=[])

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
            offset = int(4 * bp.dx)

        edges = bp.clustering_edges(blueprint, categories)
        edges = [it.set_corner(size) for it in edges]

        xs: list[list[list[NDArray[np.int_]]]] = [[] for _ in range(len(categories))]
        ys: list[list[list[NDArray[np.int_]]]] = [[] for _ in range(len(categories))]

        #
        # |<-c ->|<-2w->|
        # +------+------+
        # |<-0 ->|<-1 ->| column
        #    0            origin x

        for edge in edges:
            i = categories.index(edge.category)

            xs[i].append([edge.x + offset])  # type: ignore[operator]
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
