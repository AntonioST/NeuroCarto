from pathlib import Path
from typing import Protocol, Literal, TypedDict

import numpy as np
from bokeh.models import TextAreaInput
from numpy.typing import NDArray

from chmap.config import parse_cli, ChannelMapEditorConfig
from chmap.probe import ProbeDesp, M, E
from chmap.util.bokeh_util import ButtonFactory
from chmap.util.utils import import_name
from chmap.views.base import ViewBase, EditorView, GlobalStateView

__all__ = ['InitializeBlueprintView']


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


def default_loader(filepath: Path, probe: ProbeDesp[M, E], chmap: M) -> NDArray[np.float_] | None:
    if not filepath.exists():
        return None

    data = np.load(filepath)
    s = probe.electrode_from_numpy(probe.all_electrodes(chmap), data)
    return np.array([it.policy for it in s], dtype=float)


class InitializeBlueprintState(TypedDict):
    content: str


class InitializeBlueprintView(ViewBase, EditorView, GlobalStateView[InitializeBlueprintState]):
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
            (POLICY '=' EXPRESSION comment? '\n')*

    *   `FILE` is a file path. Could be 'None'.
    *   `LOADER` is a str match `import_name`'s required.
    *   `POLICY` is a str defined in ProbeDesp[M, E] or an int which work as a tmp policy.
    *   `EXPRESSION` is a python expression with variables.

        The expression variables are Array[float, E] and named in

        * s : shank
        * x : x pos in um
        * y : y pos in um
        * v : data value. use zero array when `file=None`.
        * p : previous block's sum result

        The expression evaluated result should be an Array[bool, E].
        The latter expression does not overwrite the previous expression.

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

        if (policies := self.parse_criteria(content, probe, chmap)) is None:
            return

        valid_policies = set([getattr(probe, it) for it in dir(type(probe)) if it.startswith('POLICY_')])
        for e, p in zip(probe.all_electrodes(chmap), policies):
            if (p := int(p)) in valid_policies:
                if (t := probe.get_electrode(blueprint, e.electrode)) is not None:
                    t.policy = p

        self.update_probe()

    def parse_criteria(self, content: str, probe: ProbeDesp, chmap: M) -> NDArray[np.int_] | None:
        policies = [it[len('POLICY_'):] for it in dir(type(probe)) if it.startswith('POLICY_')]

        current_file: Path | Literal['None'] | None = None
        current_loader: str | DataLoader | None = None
        current_data: NDArray[np.float_] | None = None
        e = probe.all_electrodes(chmap)
        s = np.array([it.s for it in e])
        x = np.array([it.x for it in e])
        y = np.array([it.y for it in e])
        p = np.full_like(s, ProbeDesp.POLICY_UNSET)
        q: NDArray[np.int_] | None = None

        for i, raw_line in enumerate(content.split('\n'), start=1):
            line = raw_line.strip()
            if '#' in line:
                line = line[:line.index('#')].strip()
            if len(line) == 0:
                continue

            match line.partition('='):
                case ('file', '=', expression):
                    if q is not None:
                        p = self.merge_policy(p, q)
                        q = None

                    if expression == 'None':
                        current_file = 'None'
                    else:
                        current_file = Path(expression)

                    current_loader = None
                    current_data = None
                case ('loader', '=', expression):
                    if current_data is not None:
                        self.log_message(f'missing file @{i}:', raw_line)
                        current_file = None
                        current_loader = None
                    else:
                        current_loader = expression

                    current_data = None

                case (left, '=', expression):
                    left = left.upper()
                    if left in policies:
                        policy = getattr(probe, f'POLICY_{left}')
                    else:
                        try:
                            policy = int(left)
                        except ValueError:
                            self.log_message(f'unknown policy {left} line @{i}:', raw_line)
                            continue

                    if current_data is None and current_file is not None:
                        try:
                            current_file, current_data = self.load_data(current_file, current_loader, probe, chmap, len(s))
                        except BaseException as e:
                            self.log_message(f'load fail:', raw_line)
                            self.logger.warning(f'load fail, line @%d: %s', i - 1, current_loader, exc_info=e)
                            current_file = None
                            current_data = None

                    if current_data is not None:
                        try:
                            q = self.parse_expression(policy, expression, q, s, x, y, current_data, p)
                        except BaseException as e:
                            self.log_message(f'eval fail:', raw_line)
                            self.logger.warning(f'eval fail, line @%d: %s', i, raw_line, exc_info=e)
                case _:
                    self.log_message(f'unknown line @{i}:', raw_line)
                    return None

        if q is not None:
            p = self.merge_policy(p, q)

        return p

    def load_data(self, file: Path | Literal['None'],
                  loader: str | None,
                  probe: ProbeDesp,
                  chmap: M,
                  n: int) -> tuple[Path | Literal['None'] | None, NDArray[np.float_] | None]:
        if file is None:
            self.log_message(f'file not exist')
            return None, None
        elif file == 'None':
            pass
        elif not file.exists():
            self.log_message(f'file not exist:', str(file))
            return None, None

        if loader is None:
            loader = default_loader
        else:
            loader_path = loader
            loader = import_name('data loader', loader_path)
            if loader is None:
                raise TypeError('NoneType loader')
            if not callable(loader):
                raise TypeError('loader not callable')

        if file == 'None':
            data = np.zeros((n,), dtype=float)
        else:
            data = loader(file, probe, chmap)
            if data is not None and data.shape != (n,):
                return None, None

        return file, data

    def parse_expression(self, policy: int, expression: str, q: NDArray[np.int_] | None, s, x, y, v, p) -> NDArray[np.int_]:
        if q is None:
            q = np.full_like(s, ProbeDesp.POLICY_UNSET)

        result = np.asarray(eval(expression, dict(np=np), dict(s=s, x=x, y=y, v=v, p=p)), dtype=bool)
        if result.ndim == 0:
            result = np.full_like(q, result)

        return np.where((q == ProbeDesp.POLICY_UNSET) & result, policy, q)

    def merge_policy(self, p: NDArray[np.int_], q: NDArray[np.int_]) -> NDArray[np.int_]:
        return np.where(q == ProbeDesp.POLICY_UNSET, p, q)


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
