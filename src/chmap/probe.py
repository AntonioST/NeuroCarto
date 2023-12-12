import abc
import importlib
from collections.abc import Hashable
from typing import TypeVar, Generic, Any

__all__ = ['ProbeDesp', 'ElectrodeDesp', 'get_probe_desp']


class ElectrodeDesp:
    x: float
    y: float
    z: float | None
    electrode: Hashable  # for identify
    channel: Any  # for display
    state: int = 0
    policy: float = 0


E = TypeVar('E', bound=ElectrodeDesp)  # electrode
M = TypeVar('M')  # channelmap


class ProbeDesp(Generic[M, E], metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def new_channelmap(self, *args, **kwargs) -> M:
        pass

    @abc.abstractmethod
    def copy_channelmap(self, chmap: M) -> M:
        pass

    @abc.abstractmethod
    def all_electrodes(self, chmap: M) -> list[E]:
        pass

    @abc.abstractmethod
    def all_channels(self, chmap: M) -> list[E]:
        pass

    @abc.abstractmethod
    def add_electrode(self, chmap: M, e: E) -> M:
        pass

    @abc.abstractmethod
    def del_electrode(self, chmap: M, e: E) -> M:
        pass

    @abc.abstractmethod
    def invalid_electrodes(self, chmap: M, e: E, s: list[E]) -> list[E]:
        pass

    @staticmethod
    def electrode_diff(s: list[E], e: E | list[E]) -> list[E]:
        if isinstance(e, list):
            t = set([it.electrode for it in e])
        else:
            t = {e.electrode}

        return [it for it in s if it.electrode not in t]

    @staticmethod
    def electrode_union(s: list[E], e: E | list[E]) -> list[E]:
        r = list(s)
        t = set([it.electrode for it in s])

        if isinstance(e, list):
            for ee in e:
                if ee.electrode not in t:
                    r.append(ee)
        else:
            if e.electrode not in t:
                r.append(e)

        return r

    @staticmethod
    def electrode_intersect(s: list[E], e: E | list[E]) -> list[E]:
        r = []
        t = set([it.electrode for it in s])

        if isinstance(e, list):
            for ee in e:
                if ee.electrode in t:
                    r.append(ee)
        else:
            if e.electrode in t:
                r.append(e)

        return r


def get_probe_desp(name: str, package: str = 'chmap') -> ProbeDesp:
    if package == 'chmap' and not name.startswith('probe_'):
        name = f'probe_{name}'

    module = importlib.import_module('.', package + '.' + name)

    for attr in dir(module):
        if not attr.startswith('_') and issubclass(desp := getattr(module, attr), ProbeDesp):
            return desp

    raise RuntimeError(f'ProbeDesp[{name}] not found')
