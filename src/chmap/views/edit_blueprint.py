from __future__ import annotations

import textwrap
from typing import Protocol, TypedDict, Literal, TYPE_CHECKING, cast

import numpy as np
from bokeh.models import Select, TextInput, PreText
from matplotlib.axes import Axes
from matplotlib.transforms import Affine2D
from numpy.typing import NDArray

from chmap.config import parse_cli, ChannelMapEditorConfig
from chmap.probe import ProbeDesp, M, E
from chmap.probe_npx import plot
from chmap.util.bokeh_app import run_later
from chmap.util.bokeh_util import ButtonFactory, as_callback
from chmap.util.util_blueprint import BlueprintFunctions
from chmap.util.utils import import_name
from chmap.views.base import EditorView, GlobalStateView, ControllerView
from chmap.views.data import DataHandler
from chmap.views.image_plt import PltImageView

if TYPE_CHECKING:
    from chmap.probe_npx.npx import ChannelMap, ProbeType

__all__ = [
    'BlueprintScriptView',
    'BlueScript',
    'BlueprintScriptState',
]

missing = object()
SCOPE = Literal['pure', 'parser', 'context']


class BlueScript(Protocol):
    def __call__(self, bp: BlueprintFunctions, arg: str):
        pass


class BlueprintScriptState(TypedDict):
    actions: dict[str, str]


class BlueprintScriptView(PltImageView, EditorView, DataHandler, ControllerView, GlobalStateView[BlueprintScriptState]):
    def __init__(self, config: ChannelMapEditorConfig):
        super().__init__(config, logger='chmap.view.blueprint_script')
        self.logger.warning('it is an experimental feature.')
        self.actions: dict[str, str | BlueScript] = {}

    @property
    def name(self) -> str:
        return 'Blueprint Script'

    # ============= #
    # UI components #
    # ============= #

    script_select: Select
    script_input: TextInput
    script_document: PreText

    def _setup_content(self, **kwargs):
        btn = ButtonFactory(min_width=50, width_policy='min')

        self.script_select = Select(
            value='', options=list(self.actions), width=100
        )
        self.script_select.on_change('value', as_callback(self._on_script_select))
        self.script_input = TextInput()
        self.script_document = PreText(text="", )

        from bokeh.layouts import row
        return [
            row(
                self.script_select, self.script_input,
                btn('run', self._on_run_script),
                btn('reset', self.reset_blueprint),
            ),
            self.script_document
        ]

    def _on_script_select(self, name: str):
        if len(name) == 0:
            self.script_document.text = ''

        try:
            script = self.get_script(name)
        except ImportError:
            self.script_document.text = 'Import Fail'
        else:
            head = f'{script.__name__}()'
            if (doc := script.__doc__) is not None:
                self.script_document.text = head + '\n' + textwrap.dedent(doc)
            else:
                self.script_document.text = head

    def _on_run_script(self):
        script = self.script_select.value
        arg = self.script_input.value_input
        if len(script):
            self.run_script(script, arg)

    def get_script(self, name: str) -> BlueScript:
        script = self.actions.get(name, name)
        if isinstance(script, str):
            self.logger.debug('load script(%s)', name)
            script = cast(BlueScript, import_name('blueprint script', script))
            if callable(script):
                self.actions[name] = script
            else:
                raise ImportError(f'script {name} not callable')
        return script

    def run_script(self, script: str, arg: str):
        if (probe := self.cache_probe) is None:
            self.log_message('no probe created')
            return
        if (chmap := self.cache_chmap) is None:
            self.log_message('no probe created')
            return
        if (blueprint := self.cache_blueprint) is None:
            self.log_message('no probe created')
            return

        self.logger.debug('run_script(%s)=%s', script, arg)
        self.set_status('run script ...')
        bp = BlueprintFunctions(probe, chmap)
        bp._controller = self

        try:
            self.logger.debug('run_script(%s) invoke', script)
            self.get_script(script)(bp, arg)
        except BaseException as e:
            self.logger.warning('run_script(%s) fail', script, exc_info=e)
            self.log_message(f'run script {script} fail', *e.args)
            return

        self.logger.debug('run_script(%s) done', script)
        self.set_status('run script done', decay=3000)
        bp.apply_blueprint(blueprint)

        self.logger.debug('run_script(%s) update', script)
        run_later(self.update_probe)

    def reset_blueprint(self):
        if (blueprint := self.cache_blueprint) is None:
            return

        self.logger.debug('reset blueprint')

        for e in blueprint:
            e.category = ProbeDesp.CATE_UNSET

        self.update_probe()
        self.log_message('reset blueprint')

    # ========= #
    # load/save #
    # ========= #

    def save_state(self, local=True) -> BlueprintScriptState | None:
        if local:
            return None

        return BlueprintScriptState(
            actions=self.actions
        )

    def restore_state(self, state: BlueprintScriptState):
        self.actions = state.get('actions', {})
        self.script_select.options = opts = list(sorted(self.actions))
        try:
            self.script_select.value = opts[0]
        except IndexError:
            self.script_select.value = ""

    def add_script(self, name: str, script: str):
        self.actions[name] = script
        self.script_select.options = list(sorted(self.actions))
        self.script_select.value = name

    # ================ #
    # updating methods #
    # ================ #

    def start(self):
        self.restore_global_state(force=True)

    cache_probe: ProbeDesp[M, E] = None
    cache_chmap: M | None
    cache_blueprint: list[E] | None
    cache_data: NDArray[np.float_] | None = None

    def on_probe_update(self, probe: ProbeDesp, chmap: M | None = None, electrodes: list[E] | None = None):
        self.cache_probe = probe
        self.cache_chmap = chmap
        self.cache_blueprint = electrodes

        from chmap.probe_npx.npx import ChannelMap
        if isinstance(chmap, ChannelMap):
            self.plot_npx_channelmap()
        else:
            self.set_image(None)

    def on_data_update(self, probe: ProbeDesp[M, E], e: list[E], data: NDArray[np.float_] | None):
        if self.cache_probe is None:
            self.cache_probe = probe

        if self.cache_blueprint is None:
            self.cache_blueprint = e

        if data is None:
            self.cache_data = None
        else:
            try:
                n = len(data)
            except TypeError as e:
                self.logger.warning('not a array', exc_info=e)
            else:
                if len(self.cache_blueprint) == n:
                    self.cache_data = np.asarray(data)

    # ================ #
    # plotting methods #
    # ================ #

    def plot_npx_channelmap(self):
        self.logger.debug('plot_npx_channelmap')

        chmap: ChannelMap = self.cache_chmap
        probe_type = chmap.probe_type

        offset = -50
        if self.cache_data is not None:
            offset = -100

        with self.plot_figure(gridspec_kw=dict(top=0.99, bottom=0.01, left=0, right=1),
                              offset=offset) as ax:
            self._plot_npx_blueprint(ax, probe_type, self.cache_blueprint)
            if self.cache_data is not None:
                self._plot_npx_electrode(ax, probe_type, self.cache_blueprint, self.cache_data,
                                         transform=Affine2D().translate(0.050, 0) + ax.transData)

            ax.set_xlabel(None)
            ax.set_xticks([])
            ax.set_xticklabels([])
            ax.set_ylabel(None)
            ax.set_yticks([])
            ax.set_yticklabels([])
            if self.cache_data is not None:
                xlim = ax.get_xlim()
                ax.set_xlim(xlim[0], xlim[1] + 0.1)

    def _plot_npx_blueprint(self, ax: Axes, probe_type: ProbeType, blueprint: list[E]):
        plot.plot_category_area(ax, probe_type, blueprint, shank_width_scale=0.5)
        plot.plot_probe_shape(ax, probe_type, color=None, label_axis=False)

    def _plot_npx_electrode(self, ax: Axes, probe_type: ProbeType, blueprint: list[E], value: NDArray[np.float_], **kwargs):
        data = np.vstack([
            [it.x for it in blueprint],
            [it.y for it in blueprint],
            value
        ]).T

        plot.plot_electrode_block(ax, probe_type, data, electrode_unit='xyv', shank_width_scale=0.5, **kwargs)


if __name__ == '__main__':
    import sys

    from chmap.main_bokeh import main

    main(parse_cli([
        *sys.argv[1:],
        '-C', 'res',
        '--debug',
        '--view=-',
        '--view=chmap.views.edit_blueprint:BlueprintScriptView',
    ]))
