from __future__ import annotations

import functools
from pathlib import Path
from typing import Protocol, TypedDict, Generic, overload, Literal, get_args

import numpy as np
from bokeh.models import TextAreaInput
from numpy.typing import NDArray

from chmap.config import parse_cli, ChannelMapEditorConfig
from chmap.probe import ProbeDesp, M, E
from chmap.util.bokeh_util import ButtonFactory
from chmap.util.utils import import_name
from chmap.views.base import ViewBase, EditorView, GlobalStateView

__all__ = ['InitializeBlueprintView']

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
    s = probe.electrode_from_numpy(probe.all_electrodes(chmap), data)
    return np.array([it.policy for it in s], dtype=float)


class InitializeBlueprintState(TypedDict):
    content: str


class InitializeBlueprintView(ViewBase, EditorView, GlobalStateView[InitializeBlueprintState]):

    def __init__(self, config: ChannelMapEditorConfig):
        super().__init__(config, logger='chmap.view.edit.blueprint')

    @property
    def name(self) -> str:
        return 'Initialize Blueprint'

    # ============= #
    # UI components #
    # ============= #

    criteria_area: TextAreaInput

    def _setup_content(self, **kwargs):
        btn = ButtonFactory(min_width=50, width_policy='min')

        self.criteria_area = TextAreaInput(
            title='criteria',
            rows=20, cols=100,
            width=500,
        )

        from bokeh.layouts import row
        return [
            self.criteria_area,
            row(
                btn('reset', self.reset_blueprint),
                btn('eval', self.eval_blueprint),
                btn('clear', self.clear_input),
            ),
        ]

    # ========= #
    # load/save #
    # ========= #

    def save_state(self) -> InitializeBlueprintState:
        return InitializeBlueprintState(content=self.criteria_area.value)

    def restore_state(self, state: InitializeBlueprintState):
        self.criteria_area.value = state['content']

    # ================ #
    # updating methods #
    # ================ #

    def start(self):
        self.restore_global_state()

    cache_probe: ProbeDesp[M, E] = None
    cache_chmap: M | None
    cache_blueprint: list[E] | None

    def on_probe_update(self, probe: ProbeDesp, chmap: M | None = None, e: list[E] | None = None):
        self.cache_probe = probe
        self.cache_chmap = chmap
        self.cache_blueprint = e

    def reset_blueprint(self):
        if (blueprint := self.cache_blueprint) is None:
            return

        self.logger.debug('reset blueprint')

        for e in blueprint:
            e.policy = ProbeDesp.POLICY_UNSET

        self.update_probe()
        self.log_message('reset blueprint')

    def clear_input(self):
        self.logger.debug('clear input')
        self.criteria_area.value = ''
        self.save_global_state()

    def eval_blueprint(self):
        if (probe := self.cache_probe) is None:
            return
        if (chmap := self.cache_chmap) is None:
            return
        if (blueprint := self.cache_blueprint) is None:
            return
        if len(content := self.criteria_area.value) == 0:
            return

        self.logger.debug('eval\n%s', content)
        self.save_global_state()

        parser = CriteriaParser(self, probe, chmap)
        if not parser.parse_content(content):
            return

        parser.get_blueprint(blueprint)
        self.update_probe()


class CriteriaParser(Generic[M, E]):
    """

    For given a ProbeDesp[M, E].

    Require data
    ------------

    a numpy array that can be parsed by `ProbeDesp.electrode_from_numpy`.
    The data value is read from `ElectrodeDesp.policy` for electrodes.

    Because E's policy is expected as an int, this view also take it as an int by default.

    For the Neuropixels, `NpxProbeDesp` use the numpy array in this form:

       Array[int, E, (shank, col, row, state, policy)]

    Data loader
    -----------

    If a data loader is given, it is imported by `import_name`.
    A loader should follow the signature `DataLoader`.

    There is a default loader `default_loader` which just convert the electrode policy value
    to float number.

    criteria_area
    -------------

    grammar::

        root = statement*

        statement = comment
                 | block
        comment =  '#' .* '\n'
        block =
           'file=' FILE comment? '\n'
           ('loader=' LOADER comment? '\n')?
           assign_expression*
        assign_expression = POLICY '=' EXPRESSION comment? '\n'
                         | FUNC(args) ('=' EXPRESSION)? comment? '\n'
        args = VAR (',' VAR)*

    *   `FILE` is a file path. Could be 'None'.
    *   `LOADER` is a str match `import_name`'s required.
    *   `POLICY` is a str defined in ProbeDesp[M, E] or an int which work as a tmp policy.

        it is short for `policy(POLICY)=expression`

    *   `EXPRESSION` is a python expression with variables VAR.

        The expression variables are Array[float, E] and named in

        * s : shank
        * x : x pos in um
        * y : y pos in um
        * v : data value. use zero array when `file=None`.
        * p : previous block's sum result
        * other variables set by func  'val(VAR)' or 'var(VAR)'

        The expression evaluated result should be an Array[bool, E].
        The latter expression does not overwrite the previous expression.

    *   `FUNC` is function call

        * `use(PROBE,...)` check current probe type (class name of `ProbeDesp`). Abort is not matched.
        * `run(FLAG,...)=FILE` run another file. FLAG could be 'inherit', 'abort'
        * `func(NAME, SCOPE?)=func_import_path` import external function. SCOPE could be 'pure', 'parser', 'context'
        * `blueprint(*POLICY)=FILE` load blueprint file (npy or channelmap file), with POLICY ONLY
        * `check(POLICY,...)=ACTION?` check POLICY existed. ACTION could be: (omitted), 'info', 'warn', 'error'
        * `policy(POLICY)=EXPRESSION` set POLICY
        * `val(NAME)=EXPRESSION` set variable NAME
        * `var(NAME)=EXPRESSION` set temp variable NAME
        * `alias(NAME)=POLICY` give POLICY an alias NAME
        * `save()=FILE` save current result to file
        * `move(SHANK,...)=VALUE` move the blueprint up/down
        * `print(LEVEL)=MESSAGE`
        * `abort()`

    The latter block overwrite previous blocks.

    Example::

        # It is a comment
        file=data.npy
        loader=default_loader
        FORBIDDEN=(y>6000)  # set all electrodes over 6mm to FORBIDDEN policy
        FORBIDDEN=(s==0)    # set all electrodes in shank 0 to FORBIDDEN policy
        FULL=v>1            # set all electrodes which value over 1 to FULL policy
        LOW=np.one_like(s)  # set remained electrodes to LOW policy

    Debug Example::

        file=None
        loader=another_loader # import test
        FORBIDDEN=(y>6000)
        LOW=1

    """

    def __init__(self, view: InitializeBlueprintView, probe: ProbeDesp[M, E], chmap: M):
        self.view = view
        self.probe = probe
        self.chmap = chmap
        self.policies = {
            it[len('POLICY_'):]: int(getattr(probe, it))  # make sure policies are int?
            for it in dir(type(probe))
            if it.startswith('POLICY_')
        }

        self.electrodes = probe.all_electrodes(chmap)
        self.variables = dict(
            s=np.array([it.s for it in self.electrodes]),
            x=np.array([it.x for it in self.electrodes]),
            y=np.array([it.y for it in self.electrodes]),
        )
        self.external_functions: dict[str, (SCOPE, ExternalFunction)] = {}

        self.context: CriteriaContext | None = None

        self.current_line: int = 0
        self.current_content: str = ''

    def info(self, message: str):
        self.view.log_message(message)

    def warning(self, message: str, exc: BaseException = None):
        self.view.logger.warning('%s line:%d  %s', message, self.current_line, self.current_content, exc_info=exc)
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

        # mask valid policy value
        for v in self.policies.values():
            np.logical_or(mask, ret == v, out=mask)

        ret[~mask] = ProbeDesp.POLICY_UNSET
        return ret

    def get_policy_value(self, policy: str) -> int | None:
        if policy.isalpha():
            try:
                return self.policies[policy.upper()]
            except KeyError:
                pass

        try:
            return int(policy)
        except ValueError:
            return None

    def get_blueprint(self, blueprint: list[E] = None, policies: NDArray[np.int_] = None) -> list[E]:
        if blueprint is None:
            blueprint = self.probe.all_electrodes(self.chmap)

        if policies is None:
            policies = self.get_result()

        c = {it.electrode: it for it in blueprint}
        for e, p in zip(self.probe.all_electrodes(self.chmap), policies):
            if (t := c.get(e.electrode, None)) is not None:
                t.policy = int(p)

        return blueprint

    def set_blueprint(self, blueprint: list[E], policies: list[int] = tuple(), inherit=True):
        data = np.array([it.policy for it in blueprint])

        if len(policies) > 0:
            mask = np.zeros_like(data, dtype=bool)
            for p in policies:
                np.logical_or(mask, data == p, out=mask)
            data[~mask] = ProbeDesp.POLICY_UNSET

        context = CriteriaContext(self, self.context if inherit else None)
        context.result[:] = data
        self.context = context

    # ======= #
    # parsing #
    # ======= #

    def parse_content(self, content: str) -> bool:
        """

        :param content:
        :return: all content evaluated?
        """
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
            return False
        except BaseException as e:
            self.warning('un-captured error', exc=e)
            return False

        return True

    def parse_line(self, line: str):
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
                        self.parse_call('policy', [left], expression)

    def parse_args(self, args: str) -> list[str]:
        args = args.strip()
        if len(args) == 0:
            return []

        if args.startswith('(') and args.endswith(')'):
            args = args[1:-1].strip()
        return [it.strip() for it in args.split(',')]

    def parse_call(self, func: str, args: list[str], expression: str | None = missing):
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

    def func_print(self, args: list[str], expression: str):
        """

        level:

        * 'i', 'info'
        * 'w', 'warn'

        :param args: [LEVEL]
        :param expression: message
        """
        match args:
            case ['warn' | 'w']:
                self.warning(expression)
            case [] | ['info' | 'i']:
                self.info(expression)
            case _:
                self.warning(f'unknown level {args}')
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
        * 'no-alias', 'xa' do not import policy aliases
        * 'no-result', 'xr'  do not import policy result

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

        content = file.read_text()
        parser = self.clone(inherit)

        if not parser.parse_content(content):
            if abort:
                raise KeyboardInterrupt
            else:
                return

        if not no_func:
            self.external_functions.update(parser.external_functions)
        if not no_var:
            self.variables.update(parser.variables)
        if not no_ali:
            for p, v in parser.policies.items():
                self.policies.setdefault(p, v)
        if not no_res:
            self.context = parser.context

    def func_check(self, args: list[str], expression: str | None = None):
        """

        ACTION:

        * 'info' send message about unknown policies in INFO level
        * 'warn' send message about unknown policies in WARN level
        * 'error' send message about unknown policies in WARN level and abort parsing.

        :param args: policies values
        :param expression: ACTION, default 'info'
        """
        for policy in args:
            if self.get_policy_value(policy) is None:
                match expression:
                    case None | 'info':
                        self.info(f'unknown policy {policy}')
                    case 'warn':
                        self.warning(f'unknown policy {policy}')
                    case 'error':
                        self.warning(f'unknown policy {policy}')
                        raise KeyboardInterrupt
                    case _:
                        self.warning(f'unknown action {expression}')
                        raise KeyboardInterrupt

    def func_alias(self, args: list[str], expression: str):
        """
        alias a POLICY to a new name.

        :param args: [name]
        :param expression: expression
        """
        match args:
            case [str(name)]:
                if (policy := self.get_policy_value(expression)) is None:
                    self.warning(f'unknown policy {expression}')
                    return

                self.policies[name.upper()] = policy
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

        if (context := self.context) is not None:
            variables = context.variables
        else:
            variables = self.variables

        value = eval(expression, dict(np=np), variables)
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

        value = eval(expression, dict(np=np), context.variables)
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

        :param args: [policy,...] policy mask.
        :param expression: channelmap (change suffix '.policy.npy') or blueprint (*.npy) filepath.
        :return:
        """
        policies = [self.get_policy_value(it) for it in args]
        if None in policies:
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

            file = file.with_suffix('.policy.npy')

        if not file.exists():
            self.warning(f'file not found. {file}')

        try:
            self.info(f'load {file}')
            data = self.probe.electrode_from_numpy(self.probe.all_electrodes(chmap), np.load(file))
        except BaseException as e:
            self.warning(f'load policy fail. {file}', exc=e)
            return

        self.set_blueprint(data, policies, inherit=True)

    def func_save(self, args: list[str], expression: str):
        """

        :param args: []
        :param expression: filepath
        :return:
        """
        if len(args) != 0:
            self.warning(f'unknown args {args}')
            return

        file = Path(expression).with_suffix('.policy.npy')
        blueprint = self.probe.all_electrodes(self.chmap)
        self.get_blueprint(blueprint)
        data = self.probe.electrode_to_numpy(blueprint)
        np.save(file, data)
        self.info(f'save {file}')

    def func_policy(self, args: list[str], expression: str):
        """

        :param args: [policy]
        :param expression: expression
        :return:
        """
        match args:
            case [str(policy)]:
                if (value := self.get_policy_value(policy)) is None:
                    self.warning(f'unknown policy {policy}')
                    return
            case _:
                self.warning(f'unknown args {args}')
                return

        if (context := self.context) is None:
            return

        result = np.asarray(eval(expression, dict(np=np), context.variables), dtype=bool)
        context.update_result(value, result)

    def func_move(self, args: list[str], expression: str):
        """
        collect and generate a blueprint, modify it by move area, store in new context.

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

        blueprint = self.get_blueprint()
        dx = np.min(np.diff(np.unique([it.x for it in blueprint])))
        dy = np.min(np.diff(np.unique([it.y for it in blueprint])))

        if mx != 0 and abs(mx) < dx:
            self.info(f'x movement {mx} smaller than the dx {dx}')
        if my != 0 and abs(my) < dy:
            self.info(f'y movement {my} smaller than the dx {dy}')

        new_blueprint = self.probe.all_electrodes(self.chmap)
        i_position = {
            (it.s, int(it.x / dx), int(it.y / dy)): it
            for it in new_blueprint
        }

        for e in blueprint:
            if len(shanks) == 0 or e.s in shanks:
                i = e.s, int((e.x + mx) / dx), int((e.y + my) / dy)
            else:
                i = e.s, int(e.x / dx), int(e.y / dy)

            if (t := i_position.get(i, None)) is not None:
                t.policy = e.policy

        self.set_blueprint(new_blueprint, inherit=False)


class CriteriaContext:
    def __init__(self, parser: CriteriaParser, previous: CriteriaContext | None):
        self.parser = parser

        self._file: Path | None = missing
        self._loader_path: str | None = None
        self._loader: DataLoader | BaseException | None = None

        self.variables = dict(parser.variables)
        if previous is None:
            self.variables['p'] = np.full((len(parser.electrodes),), ProbeDesp.POLICY_UNSET)
        else:
            self.variables['p'] = previous.merge_result()

        self.result: NDArray[np.int_] = np.full((len(parser.electrodes),), ProbeDesp.POLICY_UNSET)

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

        return self._data

    def update_result(self, policy: int, result: NDArray[np.bool_]):
        if result.ndim == 0:
            result = np.full_like(self.result, result, dtype=bool)

        # The latter does not overwrite the previous.
        previous = self.result
        self.result = np.where((previous == ProbeDesp.POLICY_UNSET) & result, policy, previous)

    def merge_result(self) -> NDArray[np.int_]:
        previous = self.variables['p']
        result = self.result
        # The latter result overwrite previous result.
        return np.where(result == ProbeDesp.POLICY_UNSET, previous, result)


if __name__ == '__main__':
    import sys

    from chmap.main_bokeh import main

    main(parse_cli([
        *sys.argv[1:],
        '-C', 'res',
        '--debug',
        '--view=-',
        '--view=blueprint',
        '--view=chmap.views.edit_blueprint:InitializeBlueprintView',
    ]))
