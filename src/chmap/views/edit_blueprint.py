from __future__ import annotations

import collections
import functools
import inspect
import re
import sys
import textwrap
from pathlib import Path
from typing import Protocol, TypedDict, Literal, TYPE_CHECKING, cast, NamedTuple

import numpy as np
from bokeh.models import Select, TextInput, Div
from numpy.typing import NDArray

from chmap.config import ChannelMapEditorConfig
from chmap.probe import ProbeDesp, M, E
from chmap.probe_npx import plot
from chmap.util.bokeh_app import run_later
from chmap.util.bokeh_util import ButtonFactory, as_callback
from chmap.util.util_blueprint import BlueprintFunctions
from chmap.util.utils import import_name, get_import_file, doc_link
from chmap.views.base import EditorView, GlobalStateView, ControllerView
from chmap.views.data import DataHandler
from chmap.views.image_plt import PltImageView

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

if TYPE_CHECKING:
    from chmap.probe_npx.npx import ChannelMap
    from chmap.util.edit.checking import RequestChannelmapTypeRequest

__all__ = [
    'BlueprintScriptView',
    'BlueprintScript',
    'BlueprintScriptState',
]

missing = object()
SCOPE = Literal['pure', 'parser', 'context']

EXAMPLE_DOCUMENT = """\
Document here.

:param bp:
:param a0: (type=default) parameter description
:param a1: (int=0) as example.
"""


def format_html_doc(doc: str) -> str:
    ret = doc.strip()
    ret = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', ret)
    ret = re.sub(r'\*(.+?)\*', r'<em>\1</em>', ret)
    ret = re.sub(r':param\s+bp:.*?\n?', '\n', ret)
    ret = re.sub(r'(?<=\n):param\s+(\w+):', r'<b>\1</b>:', ret)
    ret = re.sub(r'\n +', '</p><p style="text-indent:2em;">', ret)
    ret = re.sub(r'\n\n', '</p><br/><p>', ret)
    ret = '<p>' + ret.replace('\n', '</p><p>') + '</p>'
    return ret


EXAMPLE_DOCUMENT_HTML = format_html_doc(EXAMPLE_DOCUMENT)


class BlueprintScript(Protocol):
    """A protocol class  to represent a blueprint script function."""

    @doc_link(use_probe='chmap.util.edit.checking.use_probe')
    def __call__(self, bp: BlueprintFunctions, *args, **kwargs) -> None:
        """
        **Command structure**

        .. code-block:: python

            def example_script(bp: BlueprintFunctions):
                \"""Document\"""
                bp.check_probe() # checking used probe
                ...

        **Available decorators**

        {use_probe()}

        ***Generator script**

        (experimental feature)

        **Document script**

        We use reStructuredText format by default, so we format the document into html
        based on that rules.

        .. code-block:: python

            def example_script(bp: BlueprintFunctions, a0: str, a1:int=0):
                \"""
                {EXAMPLE_DOCUMENT}
                \"""

        * The line `:param bp:` will be removed.
        * The line `:param PARA:` will be replaced as a bold font (`<b>`).
        * The word `**WORD**` will be replaced as a bold font (`<b>`).
        * The word `*WORD*` will be replaced as an italic font (`<em>`).

        And it will look like:

        .. raw:: html

            <div class="highlight" style="padding: 10px;"><div style="font-family: monospace;">
            <p><b>example_script(a0, a1)</b></p>
            {EXAMPLE_DOCUMENT_HTML}
            </div></div>

        :param bp: script running context.
        :param args:
        :param kwargs:
        """
        pass


class BlueprintScriptInfo(NamedTuple):
    name: str
    module: str | None  # 'MODUlE:NAME'
    filepath: Path | None
    time_stamp: float | None
    script: BlueprintScript

    @classmethod
    def load(cls, name: str, module_path: str) -> Self:
        script = cast(BlueprintScript, import_name('blueprint script', module_path))
        if not callable(script):
            raise ImportError(f'script {name} not callable')

        script_file = get_import_file(module_path)
        time_stamp = None if script_file is None or not script_file.exists() else script_file.stat().st_mtime
        return BlueprintScriptInfo(name, module_path, script_file, time_stamp, script)

    def check_changed(self) -> bool:
        if self.filepath is None or self.time_stamp is None:
            return False
        if not self.filepath.exists():
            return True
        t = self.filepath.stat().st_mtime
        return self.time_stamp < t

    def reload(self) -> Self:
        if self.module is None:
            raise ImportError()

        script = cast(BlueprintScript, import_name('blueprint script', self.module, reload=True))
        if not callable(script):
            raise ImportError(f'script {self.name} not callable')

        script_file = self.filepath
        time_stamp = None if script_file is None or not script_file.exists() else script_file.stat().st_mtime
        return self._replace(time_stamp=time_stamp, script=script)

    def script_name(self) -> str:
        return self.script.__name__

    def script_signature(self) -> str:
        name = self.script.__name__
        p = ', '.join(self.script_parameters())
        return f'{name}({p})'

    def script_parameters(self) -> list[str]:
        s = inspect.signature(self.script)
        return [it for i, it in enumerate(s.parameters) if i != 0]

    def script_doc(self, html=False) -> str | None:
        if (doc := self.script.__doc__) is not None:
            ret = textwrap.dedent(doc)
            if html:
                ret = format_html_doc(ret)
            return ret
        return None

    def script_use_probe(self) -> RequestChannelmapTypeRequest | None:
        from chmap.util.edit.checking import get_use_probe
        return get_use_probe(self.script)

    def __call__(self, bp: BlueprintFunctions, script_input: str):
        """
        Eval *script_input* and call actual script function.

        Although *script_input* should be a valid Python code,
        we cheat the undefined variable name as a str.
        For example, the following *script_input* will be considered::

            (input) a,1,"b,3"
            (output) ("a", 1, "b,3")

        This function does not do the argument type validation.

        :param bp:
        :param script_input:
        """

        class Missing(collections.defaultdict):
            def __missing__(self, key):
                return key

        eval(f'__script_func__({script_input})', {}, Missing(__script_func__=functools.partial(self.script, bp)))


class BlueprintScriptState(TypedDict):
    clear: bool
    actions: dict[str, str]


class BlueprintScriptView(PltImageView, EditorView, DataHandler, ControllerView, GlobalStateView[BlueprintScriptState]):
    def __init__(self, config: ChannelMapEditorConfig):
        super().__init__(config, logger='chmap.view.blueprint_script')
        self.logger.warning('it is an experimental feature.')
        self.actions: dict[str, str | BlueprintScriptInfo] = {
            'load': 'chmap.util.edit._actions:load_blueprint',
            'move': 'chmap.util.edit._actions:move_blueprint',
            'exchange': 'chmap.util.edit._actions:exchange_shank',
            'pre-select': 'chmap.util.edit._actions:enable_electrode_as_pre_selected',
            'single': 'chmap.util.edit._actions:npx24_single_shank',
            'stripe': 'chmap.util.edit._actions:npx24_stripe',
            'half': 'chmap.util.edit._actions:npx24_half_density',
            'quarter': 'chmap.util.edit._actions:npx24_quarter_density',
            '1-eighth': 'chmap.util.edit._actions:npx24_one_eighth_density',
        }
        self._script_input_cache: dict[str, str] = {}

    @property
    def name(self) -> str:
        return 'Blueprint Script'

    # ============= #
    # UI components #
    # ============= #

    script_select: Select
    script_input: TextInput
    script_document: Div

    def _setup_content(self, **kwargs):
        btn = ButtonFactory(min_width=50, width_policy='min')

        self.script_select = Select(
            value='', options=list(self.actions), width=150,
            styles={'font-family': 'monospace'}
        )
        self.script_select.on_change('value', as_callback(self._on_script_select))

        self.script_input = TextInput(
            width=400,
            styles={'font-family': 'monospace'}
        )
        self.script_document = Div(
            text="",
            styles={'font-family': 'monospace'}
        )

        from bokeh.layouts import row
        return [
            row(
                self.script_select, Div(text="(", css_classes=['chmap-large']),
                self.script_input, Div(text=")", css_classes=['chmap-large']),
                btn('run', self._on_run_script),
                btn('reset', self.reset_blueprint),
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

        head = script.script_signature()
        if (doc := script.script_doc(html=True)) is not None:
            self.script_document.text = f'<p><b>{head}</b></p>' + doc
        else:
            self.script_document.text = f'<b>{head}</b>'

    def _on_run_script(self):
        script = self.script_select.value
        arg = self.script_input.value_input
        if len(script):
            self.run_script(script, arg)

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
    def run_script(self, script: str | BlueprintScript | BlueprintScriptInfo, script_input: str = None):
        """
        Run a blueprint script.

        :param script: script name, script path or a {BlueprintScript}
        :param script_input: script input text.
        """
        probe = self.get_app().probe

        if isinstance(script, str):
            script_name = script
        elif isinstance(script, BlueprintScriptInfo):
            script_name = script.script_name()
        else:
            script_name = getattr(script, '__name__', str(script))

        self.logger.debug('run_script(%s)', script_name)
        self.set_status('run script ...')

        try:
            bs = self.get_script(script)
        except BaseException as e:
            self.logger.warning('run_script(%s) import fail', script_name, exc_info=e)
            self.log_message(f'run script {script_name} import fail')
            self.set_status(None)
            return

        if script_input is None:
            script_input = self.script_input.value_input

        try:
            bp = self._run_script(bs, probe, self.cache_chmap, script_input)
        except BaseException as e:
            self.logger.warning('run_script(%s) fail', script_name, exc_info=e)
            self.log_message(f'run script {script_name} fail')
            self.set_status(None)
            return

        self.set_status('run script done', decay=3)

        if (blueprint := self.cache_blueprint) is not None:
            bp.apply_blueprint(blueprint)
            self.logger.debug('run_script(%s) update', script_name)
            run_later(self.update_probe)

    def _run_script(self, script: BlueprintScriptInfo,
                    probe: ProbeDesp[M, E],
                    chmap: M | None,
                    script_input: str) -> BlueprintFunctions:
        from chmap.util.edit.checking import RequestChannelmapTypeError, check_probe
        script_name = script.script_name()

        bp = BlueprintFunctions(probe, chmap)
        bp._controller = self
        if (blueprint := self.cache_blueprint) is not None:
            bp.set_blueprint(blueprint)

        request: RequestChannelmapTypeRequest | None
        if (request := script.script_use_probe()) is not None:
            if chmap is None and request.create:
                self.logger.debug('run_script(%s) create probe %s[%d]', script_name, request.probe_name, request.code)
                chmap = bp.new_channelmap(request.code)
                return self._run_script(script, probe, chmap, script_input)

            if request.check:
                check_probe(bp, request)

        try:
            self.logger.debug('run_script(%s)[%s]', script_name, script_input)
            script(bp, script_input)
        except RequestChannelmapTypeError as e:
            request = e.request
            if chmap is None and request.match_probe(probe) and request.code is not None:
                pass
            else:
                raise
        else:
            self.logger.debug('run_script(%s) done', script_name)
            return bp

        # from RequestChannelmapTypeError
        self.logger.debug('run_script(%s) request %s[%d]', script_name, request.probe_name, request.code)
        chmap = bp.new_channelmap(request.code)

        self.logger.debug('run_script(%s) rerun', script_name)
        return self._run_script(script, probe, chmap, script_input)

    def reset_blueprint(self):
        if (blueprint := self.cache_blueprint) is None:
            return

        self.logger.debug('reset blueprint')

        for e in blueprint:
            e.category = ProbeDesp.CATE_UNSET

        self.log_message('reset blueprint')
        run_later(self.update_probe)

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

        # load scripts
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

    def update_actions_select(self):
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

        self.script_select.options = opts
        try:
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
