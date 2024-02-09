from __future__ import annotations

import abc
import sys
from collections.abc import Hashable, Iterable, Sequence
from pathlib import Path
from typing import TypeVar, Generic, Any, ClassVar, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from chmap.config import ChannelMapEditorConfig
    from chmap.views.base import ViewBase

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

__all__ = ['ProbeDesp', 'ElectrodeDesp', 'get_probe_desp']


def get_probe_desp(name: str, package: str = 'chmap', describer: str = None) -> type[ProbeDesp]:
    """Get probe describer.

    Naming rules:

    * *name* correspond Python module name.
    * in `chmap` package, module name `probe_NAME` can shorten as `NAME`.

    :param name: probe family name.
    :param package: root package.
    :param describer: class name of ProbeDesp. If None, find the first found.
    :return: type of ProbeDesp.
    :raise ModuleNotFoundError: module *name* not found.
    :raise RuntimeError: no ProbeDesp subclass found in module.
    :raise TypeError: *describer* in module not a subclass of ProbeDesp
    """
    if package == 'chmap' and not name.startswith('probe_'):
        name = f'probe_{name}'

    import importlib
    module = importlib.import_module('.', package + '.' + name)

    if describer is None:
        for attr in dir(module):
            if not attr.startswith('_') and issubclass(desp := getattr(module, attr), ProbeDesp):
                return desp

        raise RuntimeError(f'ProbeDesp[{name}] not found')
    else:
        if issubclass(desp := getattr(module, describer), ProbeDesp):
            return desp

        raise TypeError(f"type of {type(desp).__name__} not subclass of ProbeDesp")


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


class ProbeDesp(Generic[M, E], metaclass=abc.ABCMeta):
    """A probe interface for GUI interaction between different probe implementations.

    :param M: channelmap, any class
    :param E: electrode, subclass of ElectrodeDesp
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
    def supported_type(self) -> dict[str, int]:
        """
        All supported probe type.

        Used in ChannelMapEditorApp._index_left_control() for dynamic generating options.

        :return: dict of {description: code}, where code is used in new_channelmap(code)
        """
        pass

    @property
    @abc.abstractmethod
    def possible_states(self) -> dict[str, int]:
        """
        All possible exported electrode state.

        Used in ChannelMapEditorApp._index_left_control() for dynamic generating buttons.

        :return: dict of {description: state}
        """
        pass

    @property
    @abc.abstractmethod
    def possible_categories(self) -> dict[str, int]:
        """
        All possible exported electrode categories.

        Used in ChannelMapEditorApp._index_left_control() for dynamic generating buttons.

        :return: dict of {description: category}
        """
        pass

    @classmethod
    def all_possible_states(cls) -> dict[str, int]:
        """
        Implement note: It finds all class variable that its name starts with 'STATE_'.

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
        Implement note: It finds all class variable that its name starts with 'CATE_'.

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
    def channelmap_file_suffix(self) -> str:
        """
        The filename extension for supported channelmap.

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
    def new_channelmap(self, chmap: int | M) -> M:
        """
        Create a new, empty channelmap instance.

        If you want to copy a channelmap instance, use `copy_channelmap` instead.

        :param chmap: a code from supported_type or a channelmap instance as probe type.
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
    def all_electrodes(self, chmap: int | M) -> list[E]:
        """
        Get all possible electrode set for the given channelmap kind.

        Implement Node:
            make sure the result is consistent in its ordering.

        :param chmap: a channelmap instance or a code from supported_type.
        :return: a list of ElectrodeDesp
        """
        pass

    @abc.abstractmethod
    def all_channels(self, chmap: M, electrodes: Iterable[E] = None) -> list[E]:
        """
        Selected electrode set in channelmap.

        :param chmap: a channelmap instance
        :param electrodes: restrict electrode set that the return set is its subset.
        :return: a list of ElectrodeDesp
        """
        pass

    @abc.abstractmethod
    def is_valid(self, chmap: M) -> bool:
        """
        Is it a valid channelmap?

        A valid channelmap means:

        * not an incomplete channelmap.
        * no electrode pair will break the probe restriction (probe_rule()).
        * can be saved in file and read by other applications without error and mis-position.

        :param chmap: a channelmap instance
        :return:
        """
        pass

    def get_electrode(self, electrodes: Iterable[E], e: Hashable | E) -> E | None:
        """

        :param electrodes: an electrode set
        :param e: electrode identify, as same as ElectrodeDesp.electrode.
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

        * Either *chmap* or *e* is in incorrect state. For example, *e* is `None`.
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
        Remove an electrode *e* from *chmap*.

        :param chmap: a channelmap instance
        :param e: an electrode
        :raise: any error means the action was failed.
        """
        pass

    def copy_electrode(self, electrodes: Sequence[E]) -> list[E]:
        """
        Copy an electrode set, including **ALL** information for every electrode.

        The default implement only consider simple case, so it won't work once
        any following points break:

        * type E has no-arg `__init__`
        * type E has any attribute name start with '_'
        * type E overwrite `__getattr__`, `__setattr__`

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

        * `probe_rule(M, e, e)` should return `False`
        * `probe_rule(M, e1, e2) == probe_rule(M, e2, e1)`

        :param chmap: channelmap type. It is a reference.
        :param e1: an electrode.
        :param e2: an electrode.
        :return: True when *e1* and *e2* are compatible.
        """
        pass

    def invalid_electrodes(self, chmap: M, e: E | Iterable[E], electrodes: Iterable[E]) -> list[E]:
        """
        Collect the invalid electrodes that an electrode from *s* will break the `probe_rule`
        with the electrode *e* (or any electrode from *e*).

        :param chmap: channelmap type. It is a reference.
        :param e: an electrode.
        :param electrodes: an electrode set.
        :return: an invalid electrode set from *s*.
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

        If *chmap* is a list[E], it indicates only restore the information only for
        this electrode subset.

        :param a: saved category matrix, or a saved '.npy' file path
        :param chmap: channelmap type, or an electrode set.
        :return: blueprint
        """
        pass
