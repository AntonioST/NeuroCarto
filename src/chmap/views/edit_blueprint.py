from __future__ import annotations

import inspect
from typing import TypedDict, TYPE_CHECKING, Generator, ClassVar

import numpy as np
from bokeh.models import Select, TextInput, Div, Button
from numpy.typing import NDArray
from typing_extensions import Required

from chmap.config import ChannelMapEditorConfig
from chmap.probe import ProbeDesp, M, E
from chmap.probe_npx import plot
from chmap.util.bokeh_app import run_later, run_timeout
from chmap.util.bokeh_util import ButtonFactory, as_callback
from chmap.util.edit.script import BlueprintScript, BlueprintScriptInfo, script_html_doc
from chmap.util.util_blueprint import BlueprintFunctions
from chmap.util.utils import doc_link
from chmap.views import RecordStep
from chmap.views.base import EditorView, GlobalStateView, ControllerView, RecordView
from chmap.views.data import DataHandler
from chmap.views.image_plt import PltImageView

if TYPE_CHECKING:
    from chmap.probe_npx.npx import ChannelMap
    from chmap.util.edit.checking import RequestChannelmapTypeRequest

__all__ = [
    'BlueprintScriptView',
    'BlueprintScriptState',
]


class BlueprintScriptState(TypedDict):
    clear: bool
    actions: dict[str, str]


class BlueprintScriptAction(TypedDict, total=False):
    action: Required[str]  # script, reset, interrupt

    # action=script, interrupt
    script_name: str

    # action=script
    script_args: str

    # action=interrupt
    interrupt: int

    # other
    description: str


class BlueprintScriptView(PltImageView, EditorView, DataHandler, ControllerView,
                          RecordView[BlueprintScriptAction], GlobalStateView[BlueprintScriptState]):
    BUILTIN_ACTIONS: ClassVar[dict[str, str]] = {
        'load': 'chmap.util.edit._actions:load_blueprint',
        'move': 'chmap.util.edit._actions:move_blueprint',
        'exchange': 'chmap.util.edit._actions:exchange_shank',
        'pre-select': 'chmap.util.edit._actions:enable_electrode_as_pre_selected',
        'single': 'chmap.util.edit._actions:npx24_single_shank',
        'stripe': 'chmap.util.edit._actions:npx24_stripe',
        'half': 'chmap.util.edit._actions:npx24_half_density',
        'quarter': 'chmap.util.edit._actions:npx24_quarter_density',
        '1-eighth': 'chmap.util.edit._actions:npx24_one_eighth_density',
        'label': 'chmap.util.edit._actions:atlas_label',
        'probe-coor': 'chmap.util.edit._actions:adjust_atlas_mouse_brain_to_probe_coordinate',
    }

    def __init__(self, config: ChannelMapEditorConfig):
        super().__init__(config, logger='chmap.view.blueprint_script')
        self.logger.warning('it is an experimental feature.')
        self.actions: dict[str, str | BlueprintScriptInfo] = dict(self.BUILTIN_ACTIONS)

        # long-running (as a generator) script control.
        self._running_script: dict[str, Generator | type[KeyboardInterrupt]] = {}

        self._script_input_cache: dict[str, str] = {}

    @property
    def name(self) -> str:
        return 'Blueprint Script'

    # ============= #
    # UI components #
    # ============= #

    script_select: Select
    script_input: TextInput
    script_run: Button
    script_document: Div

    def _setup_content(self, **kwargs):
        btn = ButtonFactory(min_width=50, width_policy='min')

        #
        self.script_select = Select(
            value='', options=list(self.actions), width=150,
            styles={'font-family': 'monospace'}
        )
        self.script_select.on_change('value', as_callback(self._on_script_select))

        #
        self.script_input = TextInput(
            width=400,
            styles={'font-family': 'monospace'}
        )
        # Mouse exit event also invoke updating handle, which this event doesn't we want
        # self.script_input.on_change('value', as_callback(self._on_run_script))

        #
        self.script_document = Div(
            text="",
            styles={'font-family': 'monospace'}
        )

        #
        self.script_run = btn('Run', self._on_run_script)

        from bokeh.layouts import row
        return [
            row(
                self.script_select, Div(text="(", css_classes=['chmap-large']),
                self.script_input, Div(text=")", css_classes=['chmap-large']),
                self.script_run,
                btn('Reset', self.reset_blueprint),
                stylesheets=['div.chmap-large {font-size: x-large;}']
            ),
            self.script_document
        ]

    def _on_script_select(self, old: str, name: str):
        if len(old) > 0:
            self._script_input_cache[old] = self.script_input.value_input

        if len(name) == 0:
            self.script_document.text = ''
            self.script_input.value_input = ''
            return

        try:
            script = self.get_script(name)
        except ImportError:
            self.script_input.value_input = ''
            self.script_document.text = 'Import Fail'
            return

        self.script_input.value_input = self._script_input_cache.get(name, '')
        self.script_document.text = script_html_doc(script)

        self._set_script_run_button_status(name)

    def _set_script_run_button_status(self, script: str):
        if script in self._running_script and script == self.script_select.value:
            self.script_run.label = 'Stop'
            self.script_run.button_type = 'danger'
        else:
            self.script_run.label = 'Run'
            self.script_run.button_type = 'default'

    def _on_run_script(self):
        script = self.script_select.value
        if script in self._running_script:
            self.logger.debug('run_script(%s) interrupt', script)
            self._running_script[script] = KeyboardInterrupt
        else:
            arg = self.script_input.value_input
            if len(script):
                self.run_script(script, arg)

    def reset_blueprint(self):
        if (blueprint := self.cache_blueprint) is None:
            return

        self.logger.debug('reset blueprint')

        for e in blueprint:
            e.category = ProbeDesp.CATE_UNSET

        self.log_message('reset blueprint')
        run_later(self.update_probe)
        self.add_record(BlueprintScriptAction(action='reset'),
                        'reset', 'reset blueprint')

    # ========= #
    # load/save #
    # ========= #

    def save_state(self, local=True) -> None:
        return None

    def restore_state(self, state: BlueprintScriptState):
        clear = state.get('clear', False)
        actions = state.get('actions', {})
        if clear:
            self.actions = actions
        else:
            self.actions.update(actions)

        self.update_actions_select()

    def add_script(self, name: str, script: str):
        self.actions[name] = script
        self.script_select.options = list(sorted(self.actions))
        self.script_select.value = name

    # ================ #
    # updating methods #
    # ================ #

    def start(self):
        self.restore_global_state(force=True)

        run_later(self._load_scripts)

    def _load_scripts(self):
        for action in self.actions:
            self.get_script(action)

    cache_probe: ProbeDesp[M, E] = None
    cache_chmap: M | None = None
    cache_blueprint: list[E] | None = None
    cache_data: NDArray[np.float_] | None = None

    def on_probe_update(self, probe: ProbeDesp, chmap: M | None = None, electrodes: list[E] | None = None):
        update_select = (
                (self.cache_probe is None)
                or (probe is None)
                or (self.cache_probe != probe)
                or (self.cache_chmap is None)
                or (chmap is None)
                or (self.cache_chmap != chmap)
        )

        self.cache_probe = probe
        self.cache_chmap = chmap
        self.cache_blueprint = electrodes

        if update_select:
            self.update_actions_select()

        from chmap.probe_npx.npx import ChannelMap
        if isinstance(chmap, ChannelMap):
            self.plot_npx_channelmap()
        else:
            self.set_image(None)

    @doc_link()
    def update_actions_select(self):
        """
        Update {#script_select}'s content that only keep probe-suitable blueprint scripts.
        """
        opts = []

        if (probe := self.cache_probe) is None:
            opts = list(self.actions)
        else:
            for action in self.actions:
                script = self.get_script(action)
                if (request := script.script_use_probe()) is None:
                    opts.append(action)
                elif request.match_probe(probe, self.cache_chmap):
                    opts.append(action)

        current_select = self.script_select.value
        self.script_select.options = opts
        try:
            if current_select not in opts:
                self.script_select.value = opts[0]
        except IndexError:
            self.script_select.value = ""

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
        """Plot Neuropixels blueprint and electrode data."""
        if (value := self.cache_data) is None:
            return

        self.logger.debug('plot_npx_channelmap')

        chmap: ChannelMap = self.cache_chmap
        probe_type = chmap.probe_type

        with self.plot_figure(gridspec_kw=dict(top=0.99, bottom=0.01, left=0, right=1), offset=-50) as ax:
            data = np.vstack([
                [it.x for it in self.cache_blueprint],
                [it.y for it in self.cache_blueprint],
                value
            ]).T

            plot.plot_electrode_block(ax, probe_type, data, electrode_unit='xyv', shank_width_scale=0.5)
            plot.plot_probe_shape(ax, probe_type, color=None, label_axis=False)

            ax.set_xlabel(None)
            ax.set_xticks([])
            ax.set_xticklabels([])
            ax.set_ylabel(None)
            ax.set_yticks([])
            ax.set_yticklabels([])

    # ========== #
    # run script #
    # ========== #

    @doc_link()
    def get_script(self, name: str | BlueprintScript | BlueprintScriptInfo) -> BlueprintScriptInfo:
        """
        Get and load {BlueprintScript}.

        If the corresponding module has changed, this function will try to reload it.

        :param name: script name.
        :return:
        """
        script = self.actions.get(name, name)

        if isinstance(name, str) and isinstance(script, str):
            self.logger.debug('load script(%s)', name)
            self.actions[name] = script = BlueprintScriptInfo.load(name, script)
            if script.filepath is not None:
                self.logger.debug('loaded script(%s) from %s', name, script.filepath)

        if not isinstance(script, BlueprintScriptInfo):
            if not callable(script):
                raise TypeError()

            script = BlueprintScriptInfo(script.__name__, '', None, None, script)

        if isinstance(name, str) and script.check_changed():
            self.logger.debug('reload script(%s)', name)
            self.actions[name] = script = script.reload()

        return script

    @doc_link()
    def run_script(self, script: str | BlueprintScript | BlueprintScriptInfo, script_input: str = None, *,
                   interrupt_at: int = None):
        """
        Run a blueprint script.

        :param script: script name, script path or a {BlueprintScript}
        :param script_input: script input text.
        :param interrupt_at: **Do not use**. Record replay used parameter.
        """
        probe = self.get_app().probe

        if isinstance(script, str):
            script_name = script
            self.logger.debug('run_script(%s)', script_name)
        elif isinstance(script, BlueprintScriptInfo):
            script_name = script.script_name()
            self.logger.debug('run_script(%s) -> %s', script.name, script_name)
        else:
            script_name = getattr(script, '__name__', str(script))
            self.logger.debug('run_script() -> %s', script_name)

        self.set_status('run script ...')

        try:
            script = self.get_script(script)
        except BaseException as e:
            self.logger.warning('run_script(%s) import fail', script_name, exc_info=e)
            self.log_message(f'run script {script_name} import fail')
            self.set_status(None)
            return

        if script_input is None:
            script_input = self.script_input.value_input

        self.add_record(BlueprintScriptAction(action='script', script_name=script.name, script_args=script_input),
                        "script", f"{script.name}({script_input})")

        try:
            bp = self._run_script(script, probe, self.cache_chmap, script_input, interrupt_at=interrupt_at)
        except BaseException as e:
            self.logger.warning('run_script(%s) fail', script.name, exc_info=e)
            self.log_message(f'run script {script.name} fail')
            self.set_status(None)
            return

        if bp is not None:
            self._run_script_done(bp, script)

    def _run_script_done(self, bp: BlueprintFunctions, script: BlueprintScriptInfo):
        if (blueprint := self.cache_blueprint) is not None:
            bp.apply_blueprint(blueprint)
            self.logger.debug('run_script(%s) update', script.name)
            run_later(self.update_probe)

    def _run_script(self, script: BlueprintScriptInfo,
                    probe: ProbeDesp[M, E],
                    chmap: M | None,
                    script_input: str, *,
                    interrupt_at: int = None) -> BlueprintFunctions | None:
        from chmap.util.edit.checking import RequestChannelmapTypeError, check_probe

        bp = BlueprintFunctions(probe, chmap)
        bp._controller = self
        if (blueprint := self.cache_blueprint) is not None:
            bp.set_blueprint(blueprint)

        request: RequestChannelmapTypeRequest | None
        if (request := script.script_use_probe()) is not None:
            if chmap is None and request.create:
                self.logger.debug('run_script(%s) create probe %s[%d]', script.name, request.probe_name, request.code)
                chmap = bp.new_channelmap(request.code)
                return self._run_script(script, probe, chmap, script_input)

            if request.check:
                check_probe(bp, request)

        try:
            self.logger.debug('run_script(%s)[%s]', script.name, script_input)
            ret = script(bp, script_input)
        except RequestChannelmapTypeError as e:
            request = e.request
            if chmap is None and request.match_probe(probe) and request.code is not None:
                pass
            else:
                raise
        else:
            if inspect.isgenerator(ret):
                self.logger.debug('run_script(%s) return generator', script.name)
                self._run_script_generator(bp, script, ret, interrupt_at=interrupt_at)
                return None
            else:
                self.logger.debug('run_script(%s) done', script.name)
                self.set_status(f'{script.name} finished', decay=3)
                return bp

        # from RequestChannelmapTypeError
        self.logger.debug('run_script(%s) request %s[%d]', script.name, request.probe_name, request.code)
        chmap = bp.new_channelmap(request.code)

        self.logger.debug('run_script(%s) rerun', script.name)
        return self._run_script(script, probe, chmap, script_input)

    def _run_script_generator(self, bp: BlueprintFunctions,
                              script: BlueprintScriptInfo,
                              gen: Generator[float | None, None, None],
                              counter: int = 0, *,
                              interrupt_at: int = None):
        is_interrupted = False
        if self._running_script.get(script.name, None) is KeyboardInterrupt:
            is_interrupted = True
        elif interrupt_at is not None and counter >= interrupt_at:
            is_interrupted = True

        try:
            if is_interrupted:
                # We are replaying in different update cycle, so RecordManager's internal flag is reset.
                # Thus, we need to block recording by myself.
                if interrupt_at is None:
                    self.add_record(BlueprintScriptAction(action='interrupt', script_name=script.name, interrupt=counter),
                                    "script", f"interrupt script {script.name}")

                gen.throw(KeyboardInterrupt())
            else:
                self._run_script_generator_next(bp, script, gen, counter, interrupt_at=interrupt_at)
                return

        except KeyboardInterrupt:
            return self._run_script_generator_interrupt(script)

        except StopIteration:
            pass

        self._run_script_generator_finish(bp, script)

    def _run_script_generator_next(self, bp: BlueprintFunctions,
                                   script: BlueprintScriptInfo,
                                   gen: Generator[float | None, None, None],
                                   counter: int = 0, *,
                                   interrupt_at: int = None):
        self._running_script[script.name] = gen
        self._set_script_run_button_status(script.name)

        ret = gen.send(None)
        self.logger.debug('run_script(%s) yield[%d]', script.name, counter)

        if ret is None:
            run_later(self._run_script_generator, bp, script, gen, counter + 1, interrupt_at=interrupt_at)
        else:
            run_timeout(int(ret * 1000), self._run_script_generator, bp, script, gen, counter + 1, interrupt_at=interrupt_at)

    def _run_script_generator_interrupt(self, script: BlueprintScriptInfo):
        try:
            del self._running_script[script.name]
        except KeyError:
            pass

        self.logger.debug('run_script(%s) interrupted', script.name)
        self.set_status(f'{script.name} interrupted', decay=10)
        self._set_script_run_button_status(script.name)

    def _run_script_generator_finish(self, bp: BlueprintFunctions, script: BlueprintScriptInfo):
        try:
            del self._running_script[script.name]
        except KeyError:
            pass

        self.logger.debug('run_script(%s) done', script.name)
        self.set_status(f'{script.name} finished', decay=3)
        self._set_script_run_button_status(script.name)

        self._run_script_done(bp, script)

    # ============= #
    # record replay #
    # ============= #

    def filter_records(self, records: list[RecordStep], *, reset=False) -> list[RecordStep]:
        source = type(self).__name__

        ret = []
        last = {}
        for record in records:
            if record.source == source:
                match record.record:
                    case {'action': 'reset'}:
                        ret.append(record)

                    case {'action': 'script', 'script_name': script_name}:
                        # copy, because we may modify 'interrupt' term.
                        record = record.with_record(dict(record.record))

                        ret.append(record)
                        last[script_name] = record

                    case {'action': 'interrupt', 'script_name': script_name, 'interrupt': interrupt_at}:
                        # merge interrupt action to the previous corresponding action.
                        try:
                            last[script_name].record['interrupt'] = interrupt_at
                        except KeyError:
                            pass

        return ret

    def replay_record(self, record: RecordStep):
        self.logger.debug('replay %s', record.description)

        match record.record:
            case {'action': 'reset'}:
                self.reset_blueprint()
            case {'action': 'script', 'script_name': script_name, 'script_args': script_args} as step:
                interrupt_at = step.get('interrupt', None)
                self.run_script(script_name, script_args, interrupt_at=interrupt_at)
