from __future__ import annotations

import datetime
import functools
import logging
import textwrap
from pathlib import Path
from typing import Protocol, TypedDict, Generic, overload, Literal, get_args, TYPE_CHECKING

import numpy as np
from bokeh.models import TextAreaInput, Select
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
    'InitializeBlueprintView', 'InitializeBlueprintState',
    'CriteriaParser', 'CriteriaContext',
    'DataHandler', 'ExternalFunction'
]

missing = object()
SCOPE = Literal['pure', 'parser', 'context']


# noinspection PyUnusedLocal
class DataLoader(Protocol):
    def __call__(self, filepath: Path, probe: ProbeDesp[M, E], chmap: M) -> NDArray[np.float_] | None:
        """

        :param filepath: data file to load.
        :param probe: probe description
        :param chmap: channelmap type. It is a reference.
        :return: Array[float, E] for all available electrodes.
        """
        pass


# noinspection PyUnusedLocal
class ExternalFunction(Protocol):

    # SCOPE=pure
    @overload
    def __call__(self, args: list[str], expression: str = None):
        pass

    # SCOPE=parser
    @overload
    def __call__(self, parser: CriteriaParser, args: list[str], expression: str = None):
        pass

    # SCOPE=context
    @overload
    def __call__(self, context: CriteriaContext, args: list[str], expression: str = None):
        pass

    def __call__(self, *args):
        pass


def default_loader(filepath: Path, probe: ProbeDesp[M, E], chmap: M) -> NDArray[np.float_] | None:
    if not filepath.exists():
        return None

    data = np.load(filepath)
    s = probe.all_electrodes(chmap)

    for e in s:
        e.category = np.nan

    s = probe.load_blueprint(data, s)
    return np.array([it.category for it in s], dtype=float)


class InitializeBlueprintState(TypedDict):
    content: str  # last shown content
    contents: dict[str, str]


class InitializeBlueprintView(PltImageView, EditorView, DataHandler, ControllerView, GlobalStateView[InitializeBlueprintState]):

    def __init__(self, config: ChannelMapEditorConfig):
        super().__init__(config, logger='chmap.view.edit.blueprint')
        self.logger.warning('it is an experimental feature.')
        self.contents: dict[str, str] = {}

    @property
    def name(self) -> str:
        return 'Blueprint'

    # ============= #
    # UI components #
    # ============= #

    criteria_area: TextAreaInput
    script_select: Select

    def _setup_content(self, **kwargs):
        btn = ButtonFactory(min_width=50, width_policy='min')

        self.criteria_area = TextAreaInput(
            rows=20, cols=100,
            width=500,
            max_length=None,
            stylesheets=['textarea {font-family: monospace;}']
        )

        self.script_select = Select(
            value='', options=['', *list(self.contents)], width=100
        )
        self.script_select.on_change('value', as_callback(self.on_script_select))

        from bokeh.layouts import row
        return [
            self.criteria_area,
            row(
                btn('reset', self.reset_blueprint),
                btn('eval', self.eval_blueprint),
                self.script_select,
                btn('New', self.on_new_script),
                btn('Save', self.on_save_script),
                btn('Copy', self.on_copy_script),
                btn('Delete', self.on_delete_script),
            ),
        ]

    def on_new_script(self):
        if self.script_select.value != "":
            self.on_save_script()

        self.script_select.value = ""
        self.criteria_area.value = ""

    def on_script_select(self, name: str, value: str):
        if name != value and len(name) > 0:
            self.contents[name] = self.criteria_area.value

        try:
            self.criteria_area.value = self.contents[value]
        except KeyError:
            self.criteria_area.value = ''

    def on_save_script(self):
        if (name := self.script_select.value) != "":
            self.save_script(name)
        else:
            content = self.criteria_area.value

            if (name := CriteriaParser.find_func_save_script_config(content)) is not None:
                self.logger.debug('save content to %s', name)
                self.save_script(name, content)
                self.script_select.value = name

    def on_copy_script(self):
        if (name := self.script_select.value) != "":
            content = self.criteria_area.value
            self.script_select.value = ''
            self.criteria_area.value = content

    def on_delete_script(self):
        name = self.script_select.value
        self.criteria_area.value = ""
        self.script_select.value = ""

        if name != "":
            try:
                del self.contents[name]
            except KeyError as e:
                pass
            else:
                self.save_global_state()

        run_later(self._reload_script_select_content)

    def _reload_script_select_content(self):
        self.script_select.options = ['', *list(self.contents)]

    # ========= #
    # load/save #
    # ========= #

    def save_state(self) -> InitializeBlueprintState:
        if (name := self.script_select.value) == "":
            content = self.criteria_area.value
        else:
            content = ''

        return InitializeBlueprintState(
            content=content,
            contents=dict(self.contents),
        )

    def restore_state(self, state: InitializeBlueprintState):
        self.criteria_area.value = state['content']

        try:
            self.contents = dict(state['contents'])
        except KeyError:
            pass

        try:
            self.script_select.options = ['', *list(self.contents)]
        except AttributeError:
            pass

    # ================ #
    # updating methods #
    # ================ #

    def start(self):
        self.restore_global_state()

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

    def reset_blueprint(self):
        if (blueprint := self.cache_blueprint) is None:
            return

        self.logger.debug('reset blueprint')

        for e in blueprint:
            e.category = ProbeDesp.CATE_UNSET

        self.update_probe()
        self.log_message('reset blueprint')

    def eval_blueprint(self):
        self.cache_data = None
        if (probe := self.cache_probe) is None:
            return
        if (chmap := self.cache_chmap) is None:
            return
        if (blueprint := self.cache_blueprint) is None:
            return
        if len(content := self.criteria_area.value) == 0:
            return

        self.save_global_state()

        parser = CriteriaParser(self, probe, chmap)
        if not parser.parse_content(content):
            return

        parser.get_blueprint(blueprint)
        run_later(self.update_probe)

    def save_script(self, name: str, content: str = None):
        if content is None:
            content = self.criteria_area.value

        self.contents[name] = content
        self.save_global_state()
        run_later(self._reload_script_select_content)


class CriteriaParser(Generic[M, E]):
    """

    For given a ProbeDesp[M, E].

    **Require data**

    a numpy array that can be parsed by `ProbeDesp.load_blueprint`.
    The data value is read from `ElectrodeDesp.category` for electrodes.

    Because E's category is expected as an int, this view also take it as an int by default.

    For the Neuropixels, `NpxProbeDesp` use the numpy array in this form:

       Array[int, E, (shank, col, row, state, category)]

    **Data loader**

    If a data loader is given, it is imported by `import_name`.
    A loader should follow the signature `DataLoader`.

    There is a default loader `default_loader` which just convert the electrode category value
    to float number.

    **criteria_area**

    grammar ::

        root = statement*

        statement = comment
                 | block
        comment =  '#' .* '\\n'
        block =
           'file=' FILE comment? '\\n'
           ('loader=' LOADER comment? '\\n')?
           func_expression*
        func_expression = CATEGORY '=' EXPRESSION comment? '\\n'
                        | FUNC(args*) ('=' EXPRESSION)? comment? '\\n'
                        | '!' EXPRESSION comment? '\\n'
                        | '>' EXPRESSION comment? '\\n'
        args = VAR (',' VAR)*

    *   `FILE` is a file path. Could be 'None'.
    *   `LOADER` is a str match `import_name`'s required.
    *   `CATEGORY` is a str defined in ProbeDesp[M, E] or an int which work as a tmp category.

        it is short for `set(CATEGORY)=expression`

    *   `EXPRESSION` is a python expression with variables VAR.

        The expression variables are Array[float, E] and named in

        * s : shank
        * x : x pos in um
        * y : y pos in um
        * v : data value. use zero array when `file=None`.
        * p : previous block's sum result
        * np : numpy module
        * bp : `chmap.util.util_blueprint.BlueprintFunctions` instance
        * python builtin functions.
        * other variables set by func  'val(VAR)' or 'var(VAR)'

        The expression evaluated result should be an Array[bool, E].
        The latter expression does not overwrite the previous expression.

    *   `FUNC` is a function call. Here lists functions with simple description.
        For details, please check corresponding `func_FUNC` function.

        * `use(PROBE,...)` check current probe type (class name of `ProbeDesp`). Abort is not matched.
        * `run(FLAG,...)=FILE` run another file. FLAG could be 'inherit', 'abort'
        * `func(NAME, SCOPE?)=func_import_path` import external function.
        * `blueprint(*CATEGORY)=FILE` load blueprint file (npy or channelmap file), with CATEGORY only.
        * `check(CATEGORY,...)=ACTION?` check CATEGORY existed.
        * `set(CATEGORY)=EXPRESSION` set CATEGORY by a bool mask (EXPRESSION result)
        * `val(NAME)=EXPRESSION` set variable NAME
        * `var(NAME)=EXPRESSION` set temp variable NAME
        * `alias(NAME)=CATEGORY` give the CATEGORY an alias NAME
        * `save(FLAG)=FILE` save current result to file
        * `move(SHANK,...)=VALUE` move the blueprint up/down
        * `print(FLAG)=MESSAGE` print message
        * `show(FLAG)=EXPRESSION` show figure
        * `abort()` abort parsing

    *   short symbols

        * `!` for `eval()=`
        * `>` for `print(eval)=`

    The latter block overwrite previous blocks.

    Example::

        # It is a comment
        file=data.npy
        loader=default_loader
        FORBIDDEN=(y>6000)  # set all electrodes over 6mm to FORBIDDEN category
        FORBIDDEN=(s==0)    # set all electrodes in shank 0 to FORBIDDEN category
        FULL=v>1            # set all electrodes which value over 1 to FULL category
        LOW=np.one_like(s)  # set remained electrodes to LOW category

    Debug Example::

        file=None
        loader=another_loader # import test
        FORBIDDEN=(y>6000)
        LOW=1

    """

    def __init__(self, view: InitializeBlueprintView | None, probe: ProbeDesp[M, E], chmap: M):
        self.logger = logging.getLogger('chmap.blueprint.parser')

        self.view = view
        self.probe = probe
        self.chmap = chmap
        self.categories = probe.all_possible_categories()

        self.electrodes = probe.all_electrodes(chmap)
        self.variables = dict(
            s=np.array([it.s for it in self.electrodes]),
            x=np.array([it.x for it in self.electrodes]),
            y=np.array([it.y for it in self.electrodes]),
        )
        self.external_functions: dict[str, (SCOPE, ExternalFunction)] = {}

        self.blueprint_functions = BlueprintFunctions(self.variables['s'], self.variables['x'], self.variables['y'], self.categories)
        self.context: CriteriaContext | None = None

        self.current_line: int = 0
        self.current_content: str = ''

    def info(self, message: str):
        if self.view is None:
            self.logger.info(message)
        else:
            self.view.log_message(message)

    def warning(self, message: str, exc: BaseException = None):
        self.logger.warning('%s line:%d  %s', message, self.current_line, self.current_content, exc_info=exc)
        if self.view is not None:
            self.view.log_message(f'{message} @{self.current_line}', self.current_content)

    def clone(self, inherit=False) -> CriteriaParser:
        """

        :param inherit: inherit context
        :return:
        """
        ret = CriteriaParser(self.view, self.probe, self.chmap)
        if inherit:
            ret.context = self.context
        return ret

    def get_result(self) -> NDArray[np.int_]:
        if (context := self.context) is None:
            return CriteriaContext(self, None).result

        ret = context.merge_result().copy()
        mask = np.zeros_like(ret, dtype=bool)

        # mask valid category value
        for v in self.categories.values():
            np.logical_or(mask, ret == v, out=mask)

        ret[~mask] = ProbeDesp.CATE_UNSET
        return ret

    def get_category_value(self, category: str) -> int | None:
        if category.isalpha():
            try:
                return self.categories[category.upper()]
            except KeyError:
                pass

        try:
            return int(category)
        except ValueError:
            return None

    def get_blueprint(self, blueprint: list[E] = None, categories: NDArray[np.int_] = None) -> list[E]:
        if blueprint is None:
            blueprint = self.probe.all_electrodes(self.chmap)

        if categories is None:
            categories = self.get_result()

        c = {it.electrode: it for it in blueprint}
        for e, p in zip(self.probe.all_electrodes(self.chmap), categories):
            if (t := c.get(e.electrode, None)) is not None:
                t.category = int(p)

        return blueprint

    def set_blueprint(self, blueprint: list[E], categories: list[int] = tuple(), inherit=True):
        data = np.array([it.category for it in blueprint])

        if len(categories) > 0:
            mask = np.zeros_like(data, dtype=bool)
            for p in categories:
                np.logical_or(mask, data == p, out=mask)
            data[~mask] = ProbeDesp.CATE_UNSET

        context = CriteriaContext(self, self.context if inherit else None)
        context.result[:] = data
        self.context = context

    def eval_expression(self, expression: str, force_context=False):
        context = self.context

        if force_context:
            if context is None:
                raise RuntimeError()
            if context.data is None:
                raise RuntimeError()

            variables = context.variables
            blueprint = context.result
        elif context is not None:
            if context.data is None:
                raise RuntimeError()
            variables = context.variables
            blueprint = context.result
        else:
            variables = self.variables
            blueprint = None

        variables = dict(variables)
        variables.update(self.categories)

        self.blueprint_functions._blueprint = blueprint
        try:
            ret = eval(expression, dict(np=np, bp=self.blueprint_functions), variables)
        finally:
            new_blueprint = self.blueprint_functions._blueprint
            self.blueprint_functions._blueprint = None

        if new_blueprint is not None and (blueprint is None or np.any(new_blueprint != blueprint)):
            # blueprint changed
            if context is None:
                context = CriteriaContext(self, None)
                context.result[:] = new_blueprint
                self.context = context
            else:
                context.result[:] = new_blueprint

        return ret

    # ======= #
    # parsing #
    # ======= #

    def parse_content(self, content: str) -> bool:
        """

        :param content:
        :return: all content evaluated?
        """
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug('parse content\n%s', textwrap.indent('\n'.join([
                f'{i:03d} {line}'
                for i, line in enumerate(content.split('\n'), start=1)
            ]), '    '))

        try:
            for i, line in enumerate(content.split('\n'), start=1):
                self.current_line = i
                self.current_content = line

                line = line.strip()
                if '#' in line:
                    line = line[:line.index('#')].strip()

                if len(line) == 0:
                    continue

                self.parse_line(line)

        except KeyboardInterrupt:
            self.logger.debug('parse abort')
            return False
        except BaseException as e:
            self.warning('un-captured error', exc=e)
            return False
        else:
            self.logger.debug('parse done')
            return True

    def parse_line(self, line: str):
        self.logger.debug('parse %s', line)

        if line.startswith('!'):
            self.parse_call('eval', [], line[1:].strip())
        elif line.startswith('>'):
            self.parse_call('print', ['eval'], line[1:].strip())
        else:
            left, eq, expression = line.partition('=')

            if len(eq) == 0:
                if '(' in left and left.endswith(')'):
                    func, _, args = left[:-1].partition('(')
                    self.parse_call(func, self.parse_args(args))
                    return
                else:
                    self.warning('unknown content')
                    raise KeyboardInterrupt

            if '(' in left and left.endswith(')'):
                func, _, args = left[:-1].partition('(')
                self.parse_call(func, self.parse_args(args), expression)
            else:
                match left:
                    case 'file':
                        self.context = CriteriaContext(self, self.context)
                        self.context.set_file(expression)
                    case 'loader':
                        if self.context is not None:
                            self.context.set_loader(expression)
                        else:
                            self.warning('missing file=')
                    case _:
                        if self.context is not None and self.context.data is not None:
                            self.parse_call('set', [left], expression)

    @classmethod
    def find_func_save_script_config(cls, content: str) -> str | None:
        for i, line in enumerate(content.split('\n'), start=1):
            line = line.strip()
            if '#' in line:
                line = line[:line.index('#')].strip()

            if line.startswith('save('):
                left, eq, expression = line.partition('=')
                if '(' in left and left.endswith(')'):
                    func, _, args = left[:-1].partition('(')
                    assert func == 'save'
                    args = cls.parse_args(args)
                    if 'script' in args and 'config' in args:
                        return expression
        return None

    @classmethod
    def parse_args(cls, args: str) -> list[str]:
        args = args.strip()
        if len(args) == 0:
            return []

        if args.startswith('(') and args.endswith(')'):
            args = args[1:-1].strip()
        return [it.strip() for it in args.split(',')]

    def parse_call(self, func: str, args: list[str], expression: str | None = missing):
        if expression is missing:
            self.logger.debug('call %s(%s)', func, args)
        else:
            self.logger.debug('call %s(%s)=%s', func, args, expression)

        f: ExternalFunction = None
        scope = 'pure'

        try:
            scope, f = self.external_functions[func]
        except KeyError:
            pass

        if f is None:
            try:
                scope = 'pure'
                f = getattr(self, f'func_{func}')
            except AttributeError:
                pass

        if f is None:
            self.warning(f'unknown func {func}')
            return False

        if scope == 'parser':
            f = functools.partial(f, self)
        elif scope == 'context':
            if self.context is None:
                self.warning(f'call func {func}() without context')
                return

            f = functools.partial(f, self.context)

        if expression is missing:
            f(args)
        else:
            f(args, expression)

    # ========= #
    # functions #
    # ========= #

    def func_abort(self, args: list[str], expression: str = None):
        """

        :param args: ignore
        :param expression: ignore
        :raise: KeyboardInterrupt
        """
        raise KeyboardInterrupt

    def func_eval(self, args: list[str], expression: str):
        """

        :param args: ignored
        :param expression:
        """
        self.eval_expression(expression)

    def func_print(self, args: list[str], expression: str):
        """

        FLAG:

        * 'i', 'info' (default)
        * 'w', 'warn'
        * 'eval' : evaluate the expression before printing

        :param args: [FLAG]
        :param expression: message
        """
        need_eval = 'eval' in args
        level_info = 'i' in args or 'info' in args
        level_warn = 'w' in args or 'warn' in args

        if level_info and level_warn:
            level_info = False
        if not level_info and not level_warn:
            level_info = True

        if need_eval:
            expression = str(self.eval_expression(expression))

        if level_info:
            self.info(expression)
        elif level_warn:
            self.warning(expression)

    def func_use(self, args: list[str]):
        """
        check use probe type.

        :param args: [PROBE,...], class name of `ProbeDesp`. If empty, print current probe type.
        :return:
        """
        probe = type(self.probe).__name__

        if len(args) == 0:
            self.info(f'use({probe})')
        else:
            if probe not in args:
                self.warning(f'fail probe check : {probe}')
                raise KeyboardInterrupt

    def func_run(self, args: list[str], expression: str):
        """
        run file.

        flag:

        * 'inherit' : inherit current context
        * 'abort' : abort when any error. otherwise, skip.
        * 'no-function', 'xf' : do not import external function
        * 'no-variable', 'xv'  do not import variable
        * 'no-alias', 'xa' do not import category aliases
        * 'no-result', 'xr'  do not import category result

        :param args: [flag,...]
        :param expression: filepath
        """
        inherit = 'inherit' in args
        abort = 'abort' in args
        no_func = 'no-function' in args or 'xf' in args
        no_var = 'no-variable' in args or 'xv' in args
        no_res = 'no-result' in args or 'xr' in args
        no_ali = 'no-alias' in args or 'xa' in args

        file = Path(expression)
        if not file.exists():
            self.warning(f'file not found. {file}')
            if abort:
                raise KeyboardInterrupt
            else:
                return

        self.logger.debug('run() read %s', file)
        content = file.read_text()
        parser = self.clone(inherit)

        self.logger.debug('run() parse %s', file)
        if not parser.parse_content(content):
            self.logger.debug('run() fail %s', file)
            if abort:
                raise KeyboardInterrupt
            else:
                return
        self.logger.debug('run() exit %s', file)

        if not no_func:
            self.external_functions.update(parser.external_functions)
        if not no_var:
            self.variables.update(parser.variables)
        if not no_ali:
            for p, v in parser.categories.items():
                self.categories.setdefault(p, v)
        if not no_res:
            self.context = parser.context

    def func_check(self, args: list[str], expression: str | None = None):
        """

        check categories name.

        ACTION:

        * 'info' send message about unknown categories in INFO level
        * 'warn' send message about unknown categories in WARN level
        * 'error' send message about unknown categories in WARN level and abort parsing.

        :param args: categories values
        :param expression: ACTION, default 'info'
        """
        for category in args:
            if self.get_category_value(category) is None:
                match expression:
                    case None | 'info':
                        self.info(f'unknown category {category}')
                    case 'warn':
                        self.warning(f'unknown category {category}')
                    case 'error':
                        self.warning(f'unknown category {category}')
                        raise KeyboardInterrupt
                    case _:
                        self.warning(f'unknown action {expression}')
                        raise KeyboardInterrupt

    def func_alias(self, args: list[str], expression: str):
        """
        alias a category name to a new name.

        :param args: [name]
        :param expression: expression
        """
        match args:
            case [str(name)] if name.isalpha():
                if (category := self.get_category_value(expression)) is None:
                    self.warning(f'unknown category {expression}')
                    return

                self.categories[name.upper()] = category
            case _:
                self.warning(f'unknown args {args}')
                return

    def func_val(self, args: list[str], expression: str):
        """
        set parser variable.

        :param args: [name]
        :param expression: expression
        """
        match args:
            case [str(name)]:
                pass
            case _:
                self.warning(f'unknown args {args}')
                return

        value = self.eval_expression(expression)
        self.variables[name] = value
        if (context := self.context) is not None:
            context.variables[name] = value

    def func_var(self, args: list[str], expression: str):
        """
        set context variable

        :param args: [name]
        :param expression: expression
        :return:
        """
        match args:
            case [str(name)]:
                pass
            case _:
                self.warning(f'unknown args {args}')
                return

        if (context := self.context) is None:
            return

        value = self.eval_expression(expression, force_context=True)
        context.variables[name] = value

    def func_func(self, args: list[str], expression: str):
        """

        scope:

        * 'pure' function signature (args: list[str], expression: str?) -> None
        * 'parser' function signature (parser: CriteriaParser, args: list[str], expression: str?) -> None
        * 'context' function signature (context: CriteriaContext, args: list[str], expression: str?) -> None

        :param args: [name, scope?]
        :param expression: function module path
        :return:
        """
        match args:
            case [str(name)]:
                scope = 'pure'
            case [str(name), str(scope)]:
                if scope not in get_args(SCOPE):
                    self.warning(f'unknown args {scope}')
                    return
            case _:
                self.warning(f'unknown args {args}')
                return

        try:
            func = import_name('external function', expression)
        except BaseException as e:
            self.warning(f'load func fail. {expression}', exc=e)
            return

        self.external_functions[name] = (scope, func)

    def func_blueprint(self, args: list[str], expression: str):
        """
        load blueprint.

        :param args: [category,...] category mask.
        :param expression: channelmap (change suffix '.blueprint.npy') or blueprint (*.npy) filepath.
        :return:
        """
        categories = [self.get_category_value(it) for it in args]
        if None in categories:
            return

        file = Path(expression)
        if not file.exists():
            self.warning('file not found')

        chmap = self.chmap
        if file.suffix != '.npy':
            try:
                chmap = self.probe.load_from_file(file)
            except BaseException as e:
                self.warning(f'load channelmap fail. {file}', exc=e)
                return

            file = file.with_suffix('.blueprint.npy')

        if not file.exists():
            self.warning(f'file not found. {file}')

        try:
            self.info(f'load {file}')
            data = self.probe.load_blueprint(file, chmap)
        except BaseException as e:
            self.warning(f'load blueprint fail. {file}', exc=e)
            return

        self.set_blueprint(data, categories, inherit=True)

    def func_save(self, args: list[str], expression: str):
        """

        flags:

        * ['blueprint'] : save blueprint (default)
        * ['data', VAR] : save data into numpy array.
        * ['script'] : save script.
        * ['script', 'config'] : save script into config. expression as script name.
        * 'force': force overwrite.
        * 'date', 'datetime': add date/datetime suffix in filename.

        :param args: flags
        :param expression: filepath
        :return:
        """
        save_target = []
        opts = []
        for arg in args:
            if arg in ('blueprint', 'data', 'script'):
                if len(save_target) == 0:
                    save_target.append(arg)
                else:
                    self.warning(f'multiple save target : {save_target[0]}, {arg}')
                    return
            elif len(save_target) == 1 and save_target[0] == 'data':
                save_target.append(arg)
            else:
                opts.append(arg)

        match save_target:
            case [] | ['blueprint']:
                self._func_save_blueprint(opts, expression)
            case ['data']:
                self.warning('missing variable name')
            case ['data', var]:
                self._func_save_variable(opts, var, expression)
            case ['script']:
                self._func_save_script(opts, expression)
            case _:
                self.warning(f'unknown saving target : {save_target[0]}')

    def _func_save_filename(self, args: list[str], expression: str, ext: str) -> Path | None:
        force = 'force' in args
        add_date = ' date' in args
        add_datetime = ' datetime' in args

        if add_datetime:
            expression = expression + '_' + str(datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S'))
        elif add_date:
            expression = expression + '_' + str(datetime.date.today().strftime('%Y-%m-%d'))

        file = Path(expression).with_suffix(ext)
        if file.exists():
            if not force:
                self.warning(f'file exist. use "force" to force saving. {file}')
                return None

        return file

    def _func_save_blueprint(self, args: list[str], expression: str):
        if (file := self._func_save_filename(args, expression, '.blueprint.npy')) is None:
            return

        blueprint = self.probe.all_electrodes(self.chmap)
        self.get_blueprint(blueprint)
        data = self.probe.save_blueprint(blueprint)
        np.save(file, data)
        self.info(f'save {file}')

    def _func_save_variable(self, args: list[str], var: str, expression: str):
        try:
            if (context := self.context) is not None:
                data = context.variables[var]
            else:
                data = self.variables[var]
        except KeyError:
            self.warning(f'variable {var} is not defined')
            return

        if (file := self._func_save_filename(args, expression, '.npy')) is None:
            return

        np.save(file, data)
        self.info(f'save {file}')

    def _func_save_script(self, args: list[str], expression: str = None):
        as_config = 'config' in args

        if (view := self.view) is None:
            return

        content = view.criteria_area.value
        if as_config and expression is None:
            expression = view.script_select.value
            if len(expression) == 0:
                self.warning('missing save name')
            else:
                view.on_save_script()

        elif as_config and expression is not None:
            view.save_script(expression, content)

        elif expression is None:
            self.warning('missing save filename')

        else:
            if (file := self._func_save_filename(args, expression, '.txt')) is None:
                return

            with file.open('w') as _file:
                print(content, file=_file)
            self.info(f'save {file}')

    def func_set(self, args: list[str], expression: str):
        """

        :param args: [category]
        :param expression: expression
        :return:
        """
        match args:
            case [str(category)]:
                if (category := self.get_category_value(category)) is None:
                    self.warning(f'unknown category {category}')
                    return
            case _:
                self.warning(f'unknown args {args}')
                return

        if (context := self.context) is None:
            return

        result = np.asarray(self.eval_expression(expression, force_context=True), dtype=bool)
        context.update_result(category, result)

    def func_move(self, args: list[str], expression: str):
        """
        collect and generate a blueprint, modify it by move area, store in new context.

        `move(SHANKS)=[MX,]MY` short from ::

            eval()=bp.set_blueprint(bp.move(bp.blueprint(), tx=MY, ty=MY, shanks=SHANKS, init=bp.CATE_UNSET))

        :param args: [shank,...], empty for all shanks.
        :param expression: y movement in um. or 'x,y' 2d movement.
        """
        shanks = [int(it) for it in args]

        if ',' in expression:
            mx, _, my = expression.partition(',')
            mx = int(mx)
            my = int(my)
        else:
            mx = 0
            my = int(expression)

        if mx != 0 and abs(mx) < (dx := self.blueprint_functions.dx):
            self.info(f'x movement {mx} smaller than the dx {dx}')
        if my != 0 and abs(my) < (dy := self.blueprint_functions.dy):
            self.info(f'y movement {my} smaller than the dx {dy}')

        categories = self.get_result()
        new_categories = self.blueprint_functions.move(categories, tx=mx, ty=my, shanks=shanks, init=ProbeDesp.CATE_UNSET)

        context = CriteriaContext(self, None)
        context.result[:] = new_categories
        self.context = context

    def func_draw(self, args: list[str], expression: str = None):
        """

        flags:

        * 'clear'
        * 'target=VIEW' pass data to ViewBase that inherit DataHandler
        * Support pass to imshow() in the future.

        :param args: [] flags.
        :param expression: expression, default use v in context
        """
        if self.context is None:
            self.view.on_data_update(self.probe, self.electrodes, None)
            return

        if expression is None:
            expression = 'v'

        clear_view = False
        target_view = None
        for arg in args:
            match arg.partition('='):
                case ('clear', '', ''):
                    clear_view = True
                case ('target', _, target_view):
                    pass

        if clear_view:
            value = None
        else:
            value = self.eval_expression(expression, force_context=True)

        if target_view is None:
            self.view.on_data_update(self.probe, self.electrodes, value)
        elif isinstance(view := self.view.get_view(target_view), DataHandler):
            view.on_data_update(self.probe, self.electrodes, value)
        else:
            self.warning(f'view {target_view} not a DataHandler')


class CriteriaContext:
    def __init__(self, parser: CriteriaParser, previous: CriteriaContext | None):
        self.parser = parser

        self._file: Path | None = missing
        self._loader_path: str | None = None
        self._loader: DataLoader | BaseException | None = None

        self.variables = dict(parser.variables)
        if previous is None:
            self.variables['p'] = np.full((len(parser.electrodes),), ProbeDesp.CATE_UNSET)
        else:
            self.variables['p'] = previous.merge_result()

        self.result: NDArray[np.int_] = np.full((len(parser.electrodes),), ProbeDesp.CATE_UNSET)

        self._data: NDArray[np.float_] | None = missing

    def set_file(self, file: str):
        if file == 'None':
            self._file = None
        else:
            self._file = Path(file)

    def set_loader(self, loader: str):
        self._loader_path = loader

        try:
            loader = import_name('data loader', loader)
            if loader is None:
                raise TypeError('NoneType loader')
            if not callable(loader):
                raise TypeError('loader not callable')
        except BaseException as e:
            self.parser.warning(f'import loader fail', exc=e)
            loader = e

        self._loader = loader

    @property
    def loader(self) -> DataLoader | BaseException:
        if self._loader is None:
            self._loader = default_loader
        return self._loader

    @property
    def data(self) -> NDArray[np.float_] | None:
        if self._data is missing:
            file = self._file
            data = None
            if file is missing:
                self.parser.warning('missing file=')

            elif file is None:
                data = np.zeros_like(self.result, dtype=float)

            elif isinstance(self.loader, BaseException):
                pass

            elif isinstance(file, Path):
                if not file.exists():
                    self.parser.warning(f'file not found. {file}')
                else:
                    try:
                        data = self.loader(file, self.parser.probe, self.parser.chmap)
                    except BaseException as e:
                        self.parser.warning('load data fail', exc=e)
                        data = None
                    else:
                        if data is None or data.shape != self.result.shape:
                            self.parser.warning('incorrect data')
                            data = None

            else:
                self.parser.warning(f'TypeError file={file}')

            self._data = data
            if data is not None:
                self.variables['v'] = data

        return self._data

    def update_result(self, category: int, result: NDArray[np.bool_]):
        if result.ndim == 0:
            result = np.full_like(self.result, result, dtype=bool)

        # The latter does not overwrite the previous.
        previous = self.result
        self.result = np.where((previous == ProbeDesp.CATE_UNSET) & result, category, previous)

    def merge_result(self) -> NDArray[np.int_]:
        previous = self.variables['p']
        result = self.result
        # The latter result overwrite previous result.
        return np.where(result == ProbeDesp.CATE_UNSET, previous, result)


if __name__ == '__main__':
    import sys

    from chmap.main_bokeh import main

    main(parse_cli([
        *sys.argv[1:],
        '-C', 'res',
        '--debug',
        '--view=-',
        '--view=chmap.views.edit_blueprint:InitializeBlueprintView',
    ]))
