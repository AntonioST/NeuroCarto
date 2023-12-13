import abc
import importlib
from collections.abc import Hashable
from typing import TypeVar, Generic, Any

import numpy as np
from numpy.typing import NDArray

__all__ = ['ProbeDesp', 'ElectrodeDesp', 'get_probe_desp']


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


E = TypeVar('E', bound=ElectrodeDesp)  # electrode
M = TypeVar('M')  # channelmap


class ProbeDesp(Generic[M, E], metaclass=abc.ABCMeta):
    """A probe interface for GUI interaction between different probe implements."""

    @abc.abstractmethod
    def new_channelmap(self, *args, **kwargs) -> M:
        """Create a new, empty channelmap"""
        pass

    @abc.abstractmethod
    def copy_channelmap(self, chmap: M) -> M:
        """Copy a channelmap."""
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
    def add_electrode(self, chmap: M, e: E) -> M:
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
        :return: updated channelmap.
        :raise: any error means the action was failed.
        """
        pass

    @abc.abstractmethod
    def del_electrode(self, chmap: M, e: E) -> M:
        """
        Remove an electrode *e* from *chmap*.

        :param chmap:
        :param e:
        :return: updated channelmap.
        :raise: any error means the action was failed.
        """
        pass

    def copy_electrode(self, e: list[E]) -> list[E]:
        """Copy an electrode set, including **ALL** information for every electrode.

        The default implement only consider simple case, so it won't work once
        any following points break:

        1. type E has no-arg `__init__`
        2. type E has any attribute name start with '_'
        3. type E overwrite `__getattr__`, `__setattr__`

        :param e:
        :return:
        """
        if len(e) == 0:
            return []

        t = type(e[0])
        return [self._copy_electrode(it, t()) for it in e]

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
    def invalid_electrodes(self, chmap: M, e: E, s: list[E]) -> list[E]:
        """
        Picking an invalid electrode set from *s* once an electrode *e* is added into *chmap*,
        under the probe restriction.

        :param chmap:
        :param e: a reference electrode. Usually, it is a most-recent selected electrode.
        :param s: an electrode set. Usually, it is a candidate set.
        :return: an invalid electrode set
        """
        pass

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
            case (list(), []):
                return list(s)
            case (list(), list(e)):
                t = set([it.electrode for it in e])
            case (list(), ElectrodeDesp(electrode=e)):
                t = {e}
            case _:
                raise TypeError()

        return [it for it in s if it.electrode not in t]

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


def get_probe_desp(name: str, package: str = 'chmap') -> ProbeDesp:
    if package == 'chmap' and not name.startswith('probe_'):
        name = f'probe_{name}'

    module = importlib.import_module('.', package + '.' + name)

    for attr in dir(module):
        if not attr.startswith('_') and issubclass(desp := getattr(module, attr), ProbeDesp):
            return desp

    raise RuntimeError(f'ProbeDesp[{name}] not found')