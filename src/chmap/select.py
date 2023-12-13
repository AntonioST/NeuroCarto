import abc
from typing import ClassVar, Generic

from .probe import ProbeDesp, M, E

__all__ = ['Selector']


class Selector(Generic[M, E], metaclass=abc.ABCMeta):
    STATE_UNUSED: ClassVar = 0
    STATE_USED: ClassVar = 1

    POLICY_UNSET: ClassVar = 10
    POLICY_FORBIDDEN: ClassVar = 11
    POLICY_SPARSE: ClassVar = 12

    def __init__(self, desp: ProbeDesp[M, E]):
        self.desp: ProbeDesp[M, E] = desp

    @property
    @abc.abstractmethod
    def possible_states(self) -> dict[str, int]:
        pass

    @property
    @abc.abstractmethod
    def possible_policy(self) -> dict[str, int]:
        pass

    @abc.abstractmethod
    def run(self, chmap: M, s: list[E]) -> M:
        pass
