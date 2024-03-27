from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
from bokeh.models import ColumnDataSource, GlyphRenderer, Div, CheckboxGroup, UIElement
from numpy.typing import NDArray

from neurocarto.config import parse_cli, CartoConfig
from neurocarto.util.bokeh_util import as_callback
from neurocarto.util.util_blueprint import BlueprintFunctions
from neurocarto.util.utils import doc_link, SPHINX_BUILD
from neurocarto.views.base import Figure, ViewBase, InvisibleView, DynamicView

if SPHINX_BUILD:
    ProbeDesp = 'neurocarto.probe.ProbeDesp'

__all__ = [
    'BlueprintView',
    'ProbePlotBlueprintProtocol',
    'ProbePlotBlueprintCallback'
]


@doc_link()
class ProbePlotBlueprintCallback:
    """
    An interface to control {BlueprintView}.
    """

    bp: BlueprintFunctions

    channelmap: Any

    blueprint: NDArray[np.int_]

    options: list[str]
    """
    option keywords for different visualization purpose. Coule be:
    
    * (empty): default
    * ``Conflict``: show conflict areas.
    
    """

    def reset_blueprint(self):
        pass

    @doc_link()
    def plot_blueprint(self, colors: dict[int, str],
                       size: tuple[int, int] = None,
                       offset: int = None):
        """
        plot blueprint.

        :param colors: category colors
        :param size: electrode size (width, height)
        :param offset: x-axis offset
        """
        pass

    @doc_link()
    def set_category_legend(self, category: dict[str, str]):
        """
        change the category legend in {BlueprintView{

        :param category: legend dict {C-name: html-color}
        """
        pass


@doc_link()
@runtime_checkable
class ProbePlotBlueprintProtocol(Protocol):
    """
    {ProbeDesp} extension protocol for plotting blueprint beside probe.
    """

    def view_ext_plot_blueprint(self, callback: ProbePlotBlueprintCallback, chmap: Any):
        """
        :param callback:
        :param chmap:
        """
        pass


@doc_link()
class BlueprintView(ViewBase, InvisibleView, DynamicView):
    """
    Show blueprint beside.

    Check whether the {ProbeDesp} implement protocol {ProbePlotBlueprintProtocol}.
    """

    data_blueprint: ColumnDataSource
    render_blueprint: GlyphRenderer

    def __init__(self, config: CartoConfig):
        super().__init__(config, logger='neurocarto.view.plot_blueprint')

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
            text="", visible=False,
            stylesheets=["""
                div.carto-blueprint-legend {
                    display: flex;
                    flex-direction: row; 
                    padding-left: 1em;
                }
                div.carto-blueprint-legend div.carto-blueprint-legend-name {
                    margin-left: 0.5em;
                }
                div.carto-blueprint-legend div.carto-blueprint-legend-color {
                    margin-left: 1em;
                    width: var(--line-height-computed, 14pt);
                    height: var(--line-height-computed, 14pt);
                }
            """]
        )
        return [self.category_legend_div]

    def set_category_color(self, category: dict[str, str]):
        self.category_legend_div.text = "".join([
            '<div class="carto-blueprint-legend">',
            *[
                (f'<div class="carto-blueprint-legend-color" style="background-color: {color};"></div>'
                 f'<div class="carto-blueprint-legend-name">{name}</div>')
                for name, color in category.items()
            ],
            '</div>'
        ])
        self.category_legend_div.visible = len(self.category_legend_div.text) > 0

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
            impl = ProbePlotBlueprintCallbackImpl(self, bp, options)

            functor: ProbePlotBlueprintProtocol
            if isinstance((functor := self.cache_probe), ProbePlotBlueprintProtocol):
                functor.view_ext_plot_blueprint(impl, chmap)
            else:
                impl.set_category_legend({
                    'select': 'green',
                    'excluded': 'pink',
                })
                impl.plot_blueprint({
                    bp.CATE_SET: 'green',
                    bp.CATE_EXCLUDED: 'pink',
                })

    def reset_blueprint(self):
        self.data_blueprint.data = dict(xs=[], ys=[], c=[])

    def plot_blueprint(self, bp: BlueprintFunctions,
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

        self.data_blueprint.data = dict(xs=xs, ys=ys, c=colors)


class ProbePlotBlueprintCallbackImpl(ProbePlotBlueprintCallback):
    def __init__(self, view: BlueprintView, bp: BlueprintFunctions, options: list[str]):
        self.__view = view
        self.bp = bp
        self._channelmap = bp.channelmap
        self._blueprint = bp.blueprint()
        self.options = options

    @property
    def channelmap(self) -> Any:
        return self._channelmap

    @property
    def blueprint(self) -> NDArray[np.int_]:
        return self._blueprint

    @blueprint.setter
    def blueprint(self, value: NDArray[np.int_]):
        if len(self._blueprint) != len(value):
            raise ValueError()

        self._blueprint = value

    def reset_blueprint(self):
        self.__view.reset_blueprint()

    def plot_blueprint(self, colors: dict[int, str], size: tuple[int, int] = None, offset: int = None):
        self.__view.plot_blueprint(self.bp, colors, self.blueprint, size, offset)

    def set_category_legend(self, category: dict[str, str]):
        self.__view.set_category_color(category)


if __name__ == '__main__':
    import sys

    from neurocarto.main_app import main

    main(parse_cli([
        *sys.argv[1:],
        '-C', 'res',
        '--debug',
        '--view=-',
        '--view=neurocarto.views.blueprint:BlueprintView',
    ]))
