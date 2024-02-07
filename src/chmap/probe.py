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

    x: float  # x position in um
    y: float  # y position in um
    electrode: Hashable  # for identify
    channel: Any  # for display in hover
    state: int = 0
    policy: int = 0

    __match_args__ = 'electrode', 'channel', 'state', 'policy'

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
        return f'Electrode[{self.channel}:{self.electrode}]({pos}){{state={self.state}, policy={self.policy}}}'


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

    # predefined electrode selecting policy
    POLICY_UNSET: ClassVar = 0  # initial value
    POLICY_SET: ClassVar = 1  # pre-selected
    POLICY_FORBIDDEN: ClassVar = 2  # never be selected
    POLICY_LOW: ClassVar = 3  # random selected, low priority

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
    def possible_policies(self) -> dict[str, int]:
        """
        All possible exported electrode policies.

        Used in ChannelMapEditorApp._index_left_control() for dynamic generating buttons.

        :return: :return: dict of {description: policy}
        """
        pass

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

        :param chmap: a channelmap instance or a code from supported_type.
        :return: a list of ElectrodeDesp
        """
        pass

    @abc.abstractmethod
    def all_channels(self, chmap: M, s: Iterable[E] = None) -> list[E]:
        """
        Selected electrode set in channelmap.

        :param chmap: a channelmap instance
        :param s: restrict electrode set that the return set is its subset.
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

    def get_electrode(self, s: Iterable[E], e: Hashable) -> E | None:
        """

        :param s: an electrode set
        :param e: electrode identify, as same as ElectrodeDesp.electrode.
        :return: found electrode in *s*. None if not found.
        """
        for ee in s:
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

    def copy_electrode(self, s: Sequence[E]) -> list[E]:
        """
        Copy an electrode set, including **ALL** information for every electrode.

        The default implement only consider simple case, so it won't work once
        any following points break:

        * type E has no-arg `__init__`
        * type E has any attribute name start with '_'
        * type E overwrite `__getattr__`, `__setattr__`

        :param s:
        :return:
        """
        if len(s) == 0:
            return []

        t: type[E] = type(s[0])
        return [t().copy(it) for it in s]

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

    def invalid_electrodes(self, chmap: M, e: E | Iterable[E], s: Iterable[E]) -> list[E]:
        """
        Collect the invalid electrodes that an electrode from *s* will break the `probe_rule`
        with the electrode *e* (or any electrode from *e*).

        :param chmap: channelmap type. It is a reference.
        :param e: an electrode.
        :param s: an electrode set.
        :return: an invalid electrode set from *s*.
        """
        if isinstance(e, Iterable):
            return [it for it in s if any([not self.probe_rule(chmap, ee, it) for ee in e])]
        else:
            return [it for it in s if not self.probe_rule(chmap, e, it)]

    @abc.abstractmethod
    def select_electrodes(self, chmap: M, s: list[E], **kwargs) -> M:
        """
        Selecting electrodes based on the electrode blueprint.

        :param chmap: channelmap type. It is a reference.
        :param s: channelmap blueprint
        :param kwargs: other parameters.
        :return: generated channelmap
        """
        pass

    @abc.abstractmethod
    def electrode_to_numpy(self, s: list[E]) -> NDArray[np.int_]:
        """
        Store electrode information into a numpy array for saving purpose.

        :param s: all electrode set.
        :return: policy matrix for saving.
        """
        pass

    @abc.abstractmethod
    def electrode_from_numpy(self, s: list[E], a: NDArray[np.int_]) -> list[E]:
        """
        Retrieve electrode information for an electrode set *s* from *a* for saving purpose.

        :param s: all electrode set.
        :param a: saved policy matrix
        :return: *s*
        """
        pass
