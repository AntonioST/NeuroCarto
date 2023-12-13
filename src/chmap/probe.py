from __future__ import annotations

import abc
from collections.abc import Hashable
from pathlib import Path
from typing import TypeVar, Generic, Any, ClassVar

import numpy as np
from numpy.typing import NDArray

__all__ = ['ProbeDesp', 'ElectrodeDesp', 'get_probe_desp']


def get_probe_desp(name: str, package: str = 'chmap') -> type[ProbeDesp]:
    if package == 'chmap' and not name.startswith('probe_'):
        name = f'probe_{name}'

    import importlib
    module = importlib.import_module('.', package + '.' + name)

    for attr in dir(module):
        if not attr.startswith('_') and issubclass(desp := getattr(module, attr), ProbeDesp):
            return desp

    raise RuntimeError(f'ProbeDesp[{name}] not found')


class ElectrodeDesp:
    """An electrode interface for GUI interaction between different electrode implements."""

    x: float  # x position in um
    y: float  # y position in um
    z: float | None  # z position in um
    electrode: Hashable  # for identify
    channel: Any  # for display
    state: int = 0
    policy: int = 0

    __match_args__ = 'electrode', 'channel', 'state', 'policy'

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
        if self.z is not None:
            pos.append(self.z)
        pos = ','.join(map(str, pos))
        return f'Electrode[{self.channel}:{self.electrode}]({pos}){{state={self.state}, policy={self.policy}}}'


K = TypeVar('K')
E = TypeVar('E', bound=ElectrodeDesp)  # electrode
M = TypeVar('M')  # channelmap


class ProbeDesp(Generic[M, E], metaclass=abc.ABCMeta):
    """A probe interface for GUI interaction between different probe implements."""

    STATE_UNUSED: ClassVar = 0
    STATE_USED: ClassVar = 1
    STATE_FORBIDDEN: ClassVar = 2

    POLICY_UNSET: ClassVar = 0
    POLICY_FORBIDDEN: ClassVar = 1
    POLICY_SPARSE: ClassVar = 2

    @property
    @abc.abstractmethod
    def possible_type(self) -> dict[str, int]:
        pass

    @property
    @abc.abstractmethod
    def possible_states(self) -> dict[str, int]:
        pass

    @property
    @abc.abstractmethod
    def possible_policy(self) -> dict[str, int]:
        pass

    @property
    @abc.abstractmethod
    def channelmap_file_suffix(self) -> str:
        pass

    @abc.abstractmethod
    def load_from_file(self, file: Path) -> M:
        pass

    @abc.abstractmethod
    def save_to_file(self, chmap: M, file: Path):
        pass

    @abc.abstractmethod
    def new_channelmap(self, chmap: int | M) -> M:
        """Create a new, empty channelmap"""
        pass

    @abc.abstractmethod
    def copy_channelmap(self, chmap: M) -> M:
        """Copy a channelmap."""
        pass

    @abc.abstractmethod
    def channelmap_desp(self, chmap: M | None) -> str:
        pass

    @abc.abstractmethod
    def all_electrodes(self, chmap: M) -> list[E]:
        """all possible electrode set for *chmap* kind probe."""
        pass

    @abc.abstractmethod
    def all_channels(self, chmap: M) -> list[E]:
        """all selected electrode set in *chmap*"""
        pass

    @abc.abstractmethod
    def is_valid(self, chmap: M) -> bool:
        """Is *chmap* a valid channelmap?"""
        pass

    def get_electrode(self, s: list[E], e: Hashable) -> E | None:
        for ee in s:
            if ee.electrode == e:
                return ee
        return None

    @abc.abstractmethod
    def add_electrode(self, chmap: M, e: E):
        """
        Add an electrode *e* into *chmap*.

        An error raised when:

        1. Either *chmap* or *e* is in incorrect state. For example, *e* is `None`.
        2. *chmap* doesn't allow to add any additional electrode.
        3. *chmap* doesn't allow to add *e* due to probe restriction.

        It is better to raise an error instead of ignoring because an error can carry
        the message to frontend.

        :param chmap:
        :param e:
        :raise: any error means the action was failed.
        """
        pass

    @abc.abstractmethod
    def del_electrode(self, chmap: M, e: E):
        """
        Remove an electrode *e* from *chmap*.

        :param chmap:
        :param e:
        :raise: any error means the action was failed.
        """
        pass

    def copy_electrode(self, s: list[E]) -> list[E]:
        """Copy an electrode set, including **ALL** information for every electrode.

        The default implement only consider simple case, so it won't work once
        any following points break:

        1. type E has no-arg `__init__`
        2. type E has any attribute name start with '_'
        3. type E overwrite `__getattr__`, `__setattr__`

        :param s:
        :return:
        """
        if len(s) == 0:
            return []

        t = type(s[0])
        return [self._copy_electrode(it, t()) for it in s]

    def _copy_electrode(self, r: E, e: E, **kwargs) -> E:
        """A copy helper function to move data from *s* to *e*.

        :param r: a reference electrode
        :param e: a new electrode.
        :param kwargs: overwrite fields. If you want a deep copy for particular fields.
        :return: e
        """
        for attr in dir(r):
            if not attr.startswith('_'):
                if attr in kwargs:
                    setattr(e, attr, kwargs[attr])
                else:
                    setattr(e, attr, getattr(r, attr))
        return e

    @abc.abstractmethod
    def probe_rule(self, chmap: M, e1: E, e2: E) -> bool:
        """Does electrode *e1* and *e2* can be used in the same time?

        :param chmap: channelmap type. It is a reference.
        :param e1:
        :param e2:
        :return:
        """
        pass

    def invalid_electrodes(self, chmap: M, e: E, s: list[E]) -> list[E]:
        """
        Picking an invalid electrode set from *s* once an electrode *e* is added into *chmap*,
        under the probe restriction.

        :param chmap: channelmap type. It is a reference.
        :param e: a reference electrode. Usually, it is a most-recent selected electrode.
        :param s: an electrode set. Usually, it is a candidate set.
        :return: an invalid electrode set
        """
        if isinstance(s, list):
            return [it for it in s if not self.probe_rule(chmap, e, it)]
        else:
            raise TypeError()

    @abc.abstractmethod
    def select_electrodes(self, chmap: M, s: list[E], **kwargs) -> M:
        """

        :param chmap: channelmap type. It is a reference.
        :param s: channelmap policy
        :return: generate channelmap
        """

    @staticmethod
    def electrode_diff(s: list[E], e: E | list[E]) -> list[E]:
        """set difference.

        :param s: an electrode set S
        :param e: an electrode or a set E
        :return: an electrode set S \\ E
        """
        match (s, e):
            case ([], list() | ElectrodeDesp()):
                return []
            case (list(), [] | {}):
                return list(s)
            case (_, list(e)):
                t = set([it.electrode for it in e])
            case (_, ElectrodeDesp(electrode=e)):
                t = {e}
            case _:
                raise TypeError('e=' + repr(e))

        if isinstance(s, list):
            return [it for it in s if it.electrode not in t]
        else:
            raise TypeError('s=' + repr(s))

    @staticmethod
    def electrode_union(s: list[E], e: E | list[E]) -> list[E]:
        """set union.

        :param s: an electrode set S
        :param e: an electrode set E
        :return: an electrode set S ⋃ E
        """
        match (s, e):
            case ([], list()):
                return list(e)
            case ([], ElectrodeDesp()):
                return [e]
            case (list(), []):
                return list(s)
            case (list(), list()):
                pass
            case (list(), ElectrodeDesp()):
                e = [e]
            case _:
                raise TypeError()

        r = list(s)
        t = set([it.electrode for it in s])

        for ee in e:
            if ee.electrode not in t:
                r.append(ee)

        return r

    @staticmethod
    def electrode_intersect(s: list[E], e: E | list[E]) -> list[E]:
        """set intersect.

        :param s: an electrode set S
        :param e: an electrode set E
        :return: an electrode set S ⋂ E
        """
        match (s, e):
            case ([], list() | ElectrodeDesp()):
                return []
            case (_, []):
                return []
            case (_, list()):
                pass
            case (_, ElectrodeDesp()):
                e = [e]
            case _:
                raise TypeError()

        r = []
        t = set([it.electrode for it in s])

        for ee in e:
            if ee.electrode in t:
                r.append(ee)

        return r

    @abc.abstractmethod
    def electrode_to_numpy(self, s: list[E]) -> NDArray[np.int_]:
        """Store electrode information into a numpy array for saving purpose.

        :param s:
        :return:
        """
        pass

    @abc.abstractmethod
    def electrode_from_numpy(self, s: list[E], a: NDArray[np.int_]) -> list[E]:
        """Retrieve electrode information for an electrode set *e* from *a* for saving purpose.

        :param s:
        :param a:
        :return: *e*
        """
        pass
