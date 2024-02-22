from __future__ import annotations

import abc
import sys
from collections.abc import Hashable, Iterable, Sequence
from pathlib import Path
from types import ModuleType
from typing import TypeVar, Generic, Any, ClassVar, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from chmap.util.utils import import_name, doc_link, SPHINX_BUILD

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

if TYPE_CHECKING:
    from chmap.config import ChannelMapEditorConfig
    from chmap.views.base import ViewBase
elif SPHINX_BUILD:
    ProbeDesp = 'chmap.probe.ProbeDesp'
    ChannelMapEditorApp = 'chmap.main_bokeh.ChannelMapEditorApp'

__all__ = ['ProbeDesp', 'ElectrodeDesp', 'get_probe_desp']


@doc_link()
def get_probe_desp(name: str) -> type[ProbeDesp]:
    """Get probe describer.

    Naming rules (finding in order):

    * a module path ``MODULE:NAME``, import MODULE and use NAME.
    * (following rule does not specific which {ProbeDesp} subtype, find the first matched.)
    * a probe family name, which can be found in module ``chmap.probe_NAME``.
    * a probe family name, which can be found in module ``chmap.NAME``.
    * a module name, which can be found in module ``NAME``.

    :param name: probe family name.
    :return: type of {ProbeDesp}.
    :raise RuntimeError: module not found; no {ProbeDesp} implementation found; not a {ProbeDesp} subtype.
    :see: {import_name()}
    """

    module = None
    if '.' not in name and ':' not in name:
        try:
            module = import_name('probe', f'chmap.probe_{name}:*')
        except ImportError:
            pass

    if module is None and ':' not in name:
        try:
            module = import_name('probe', f'chmap.{name}:*')
        except ImportError:
            pass

    if module is None and ':' not in name:
        try:
            module = import_name('probe', f'{name}:*')
        except ImportError:
            pass

    if module is None:
        try:
            module = import_name('probe', name)
        except ImportError:
            pass

    if module is None:
        raise RuntimeError(f'ProbeDesp[{name}] not found')

    if isinstance(module, ModuleType):
        for attr in dir(module):
            if not attr.startswith('_') and issubclass(desp := getattr(module, attr), ProbeDesp):
                return desp

        raise RuntimeError(f'ProbeDesp[{name}] not found')
    elif issubclass(module, ProbeDesp):
        return module

    raise RuntimeError(f"type of {type(module).__name__} not subclass of ProbeDesp")


class ElectrodeDesp:
    """An electrode interface for GUI interaction between different electrode implementations."""

    s: int  # shank
    x: float  # x position in um
    y: float  # y position in um
    electrode: Hashable  # for identify
    channel: Any  # for display in hover
    state: int = 0
    category: int = 0

    __match_args__ = 'electrode', 'channel', 'state', 'category'

    def copy(self, r: ElectrodeDesp, **kwargs) -> Self:
        """A copy helper function to move data from *r*.

        :param r: copy reference electrode
        :param kwargs: overwrite fields. If you want a deep copy for particular fields.
        :return: self
        """
        for attr in dir(r):
            if not attr.startswith('_'):
                if attr in kwargs:
                    setattr(self, attr, kwargs[attr])
                else:
                    setattr(self, attr, getattr(r, attr))
        return self

    def __hash__(self) -> int:
        return hash(self.electrode)

    def __eq__(self, other) -> bool:
        try:
            return self.electrode == other.electrode
        except AttributeError:
            return False

    def __str__(self):
        return f'Electrode[{self.electrode}]'

    def __repr__(self):
        pos = [self.x, self.y]
        pos = ','.join(map(str, pos))
        return f'Electrode[{self.channel}:{self.electrode}]({pos}){{state={self.state}, category={self.category}}}'


K = TypeVar('K')
E = TypeVar('E', bound=ElectrodeDesp)  # electrode
M = TypeVar('M')  # channelmap


@doc_link()
class ProbeDesp(Generic[M, E], metaclass=abc.ABCMeta):
    """A probe interface for GUI interaction between different probe implementations.

    :param M: channelmap, any class
    :param E: electrode, subclass of {ElectrodeDesp}
    """

    # predefined electrode states
    STATE_UNUSED: ClassVar = 0  # electrode is not used, and it is selectable.
    STATE_USED: ClassVar = 1  # electrode is selected.
    STATE_FORBIDDEN: ClassVar = 2  # electrode is not used, but it is not selectable.

    # predefined electrode categories
    CATE_UNSET: ClassVar = 0  # initial value
    CATE_SET: ClassVar = 1  # pre-selected
    CATE_FORBIDDEN: ClassVar = 2  # never be selected
    CATE_LOW: ClassVar = 3  # random selected, low priority

    @property
    @abc.abstractmethod
    @doc_link()
    def supported_type(self) -> dict[str, int]:
        """
        All supported probe type.

        Used in {ChannelMapEditorApp#install_right_panel_views()} for dynamic generating options.

        :return: dict of {description: code}, where code is used in new_channelmap(code)
        """
        pass

    def channelmap_description(self, code: int) -> str | None:
        for name, _code in self.supported_type.items():
            if code == _code:
                return name
        return None

    @property
    @abc.abstractmethod
    @doc_link()
    def possible_states(self) -> dict[str, int]:
        """
        All possible exported electrode state.

        Used in {ChannelMapEditorApp#install_right_panel_views()} for dynamic generating buttons.

        :return: dict of {description: state}
        """
        pass

    def state_description(self, state: int) -> str | None:
        for desp, _state in self.possible_states.items():
            if state == _state:
                return desp
        return None

    @property
    @abc.abstractmethod
    @doc_link()
    def possible_categories(self) -> dict[str, int]:
        """
        All possible exported electrode categories.

        Used in {ChannelMapEditorApp#install_right_panel_views()} for dynamic generating buttons.

        :return: dict of {description: category}
        """
        pass

    def category_description(self, code: int) -> str | None:
        for desp, cate in self.possible_categories.items():
            if cate == code:
                return desp
        return None

    @classmethod
    def all_possible_states(cls) -> dict[str, int]:
        """
        Implement note: It finds all class variable that its name starts with 'STATE_*'.

        :return: dict of {state_name: state_value}
        """
        return {
            it[6:]: int(getattr(cls, it))
            for it in dir(cls)
            if it.startswith('STATE_')
        }

    @classmethod
    def all_possible_categories(cls) -> dict[str, int]:
        """
        Implement note: It finds all class variable that its name starts with 'CATE_*'.

        :return: dict of {category_name: category_value}
        """
        return {
            it[5:]: int(getattr(cls, it))
            for it in dir(cls)
            if it.startswith('CATE_')
        }

    def extra_controls(self, config: ChannelMapEditorConfig) -> list[type[ViewBase]]:
        """
        Probe specific controls.

        :param config: application configurations.
        :return: list of ViewBase subtype.
        """
        return []

    @property
    @abc.abstractmethod
    def channelmap_file_suffix(self) -> list[str]:
        """
        The filename extension for supported channelmap.

        The first suffix in returned list is considered the primary one.

        :return: file extension, like ".imro".
        """
        pass

    @abc.abstractmethod
    def load_from_file(self, file: Path) -> M:
        """
        Load channelmap file.

        :param file: channelmap filepath
        :return: channelmap instance
        """
        pass

    @abc.abstractmethod
    def save_to_file(self, chmap: M, file: Path):
        """
        Save channelmap into file.

        :param chmap: channelmap instance
        :param file: channelmap filepath
        """
        pass

    @abc.abstractmethod
    @doc_link()
    def channelmap_code(self, chmap: Any | None) -> int | None:
        """
        identify a channelmap, and return corresponding code.

        :param chmap: Any instance. It could be a channelmap instance.
        :return: a code from {#supported_type}. None if *chmap* is unknown or not supported.
        """
        pass

    @abc.abstractmethod
    @doc_link()
    def new_channelmap(self, chmap: int | M) -> M:
        """
        Create a new, empty channelmap instance.

        If you want to copy a channelmap instance, use {#copy_channelmap()} instead.

        :param chmap: a code from {#supported_type} or a channelmap instance as probe type.
        :return: a channelmap instance
        """
        pass

    @abc.abstractmethod
    def copy_channelmap(self, chmap: M) -> M:
        """
        Copy a channelmap instance.

        :param chmap: channelmap instance as reference.
        :return: a channelmap instance
        """
        pass

    @abc.abstractmethod
    def channelmap_desp(self, chmap: M | None) -> str:
        """
        A description for displaying the status of a channelmap instance.

        :param chmap: a channelmap instance, or None when no probe (an initial description)
        :return: description.
        """
        pass

    @abc.abstractmethod
    @doc_link()
    def all_electrodes(self, chmap: int | M) -> list[E]:
        """
        Get all possible electrode set for the given channelmap kind.

        Implement Node:
            make sure the result is consistent in its ordering.

        :param chmap: a channelmap instance or a code from supported_type.
        :return: a list of {ElectrodeDesp}
        """
        pass

    @abc.abstractmethod
    @doc_link()
    def all_channels(self, chmap: M, electrodes: Iterable[E] = None) -> list[E]:
        """
        Selected electrode set in channelmap.

        :param chmap: a channelmap instance
        :param electrodes: restrict electrode set that the return set is its subset.
        :return: a list of {ElectrodeDesp}
        """
        pass

    @abc.abstractmethod
    @doc_link()
    def is_valid(self, chmap: M) -> bool:
        """
        Is it a valid channelmap?

        A valid channelmap means:

        * not an incomplete channelmap.
        * no electrode pair will break the probe restriction ({#probe_rule()}).
        * can be saved in file and read by other applications without error and mis-position.

        :param chmap: a channelmap instance
        :return:
        """
        pass

    @doc_link()
    def get_electrode(self, electrodes: Iterable[E], e: Hashable | E) -> E | None:
        """

        :param electrodes: an electrode set
        :param e: electrode identify, as same as {ElectrodeDesp#electrode}.
        :return: found electrode in *s*. None if not found.
        """
        if isinstance(e, ElectrodeDesp):
            e = e.electrode

        for ee in electrodes:
            if ee.electrode == e:
                return ee
        return None

    @abc.abstractmethod
    def add_electrode(self, chmap: M, e: E, *, overwrite=False):
        """
        Add an electrode *e* into *chmap*.

        An error raised when:

        * Either *chmap* or *e* is in incorrect state. For example, *e* is ``None``.
        * *chmap* is complete, and it doesn't allow to add any additional electrode.
        * *chmap* is incomplete, but it doesn't allow to add *e* due to probe restriction.

        It is better to raise an error instead of ignoring because an error can carry
        the message to frontend.

        :param chmap: a channelmap instance
        :param e: an electrode
        :param overwrite: force add electrode
        :raise: any error means the action was failed.
        """
        pass

    @abc.abstractmethod
    def del_electrode(self, chmap: M, e: E):
        """
        Remove an electrode *e* from the *chmap*.

        :param chmap: a channelmap instance
        :param e: an electrode
        :raise: any error means the action was failed.
        """
        pass

    def clear_electrode(self, chmap: M):
        """
        Remove all electrodes from the *chmap*.

        :param chmap: a channelmap instance
        """
        for e in self.all_channels(chmap):
            self.del_electrode(chmap, e)

    def copy_electrode(self, electrodes: Sequence[E]) -> list[E]:
        """
        Copy an electrode set, including **ALL** information for every electrode.

        The default implement only consider simple case, so it won't work once
        any following points break:

        * type E has no-arg ``__init__``
        * type E has any attribute name start with '_'
        * type E overwrite ``__getattr__``, ``__setattr__``

        :param electrodes:
        :return:
        """
        if len(electrodes) == 0:
            return []

        t: type[E] = type(electrodes[0])
        return [t().copy(it) for it in electrodes]

    @abc.abstractmethod
    def probe_rule(self, chmap: M, e1: E, e2: E) -> bool:
        """
        Does electrode *e1* and *e2* can be used in the same time?

        This method's implementation should follow the rules in most cases:

        * ``probe_rule(M, e, e)`` should return ``False``
        * ``probe_rule(M, e1, e2) == probe_rule(M, e2, e1)``

        :param chmap: channelmap type. It is a reference.
        :param e1: an electrode.
        :param e2: an electrode.
        :return: True when *e1* and *e2* are compatible.
        """
        pass

    @doc_link()
    def invalid_electrodes(self, chmap: M, e: E | Iterable[E], electrodes: Iterable[E]) -> list[E]:
        """
        Collect the invalid electrodes that an electrode from *s* will break the {#probe_rule()}
        with the electrode *e* (or any electrode from *e*).

        Note that *e* may also be contained in the result if *e* in *electrodes*.

        :param chmap: channelmap type. It is a reference.
        :param e: an electrode.
        :param electrodes: an electrode set.
        :return: an invalid electrode set from *electrodes*.
        """
        if isinstance(e, Iterable):
            return [it for it in electrodes if any([not self.probe_rule(chmap, ee, it) for ee in e])]
        else:
            return [it for it in electrodes if not self.probe_rule(chmap, e, it)]

    @abc.abstractmethod
    def select_electrodes(self, chmap: M, blueprint: list[E], **kwargs) -> M:
        """
        Selecting electrodes based on the electrode blueprint.

        :param chmap: channelmap type. It is a reference.
        :param blueprint: channelmap blueprint
        :param kwargs: other parameters.
        :return: generated channelmap
        """
        pass

    @abc.abstractmethod
    def save_blueprint(self, blueprint: list[E]) -> NDArray[np.int_]:
        """
        Store blueprint, included all electrode information into a numpy array.

        :param blueprint: blueprint.
        :return: blueprint matrix.
        """
        pass

    @abc.abstractmethod
    def load_blueprint(self, a: str | Path | NDArray[np.int_], chmap: int | M | list[E]) -> list[E]:
        """
        Restore blueprint, included all electrode information from a numpy array *a*.

        If *chmap* is a ``list[E]``, it indicates only restore the information only for
        this electrode subset.

        :param a: saved category matrix, or a saved '.npy' file path
        :param chmap: channelmap type, or an electrode set.
        :return: blueprint
        """
        pass
