import numpy as np
from numpy.typing import NDArray

__all__ = ['BlueprintFunctions']


# noinspection PyMethodMayBeStatic
class BlueprintFunctions:
    """
    Provide blueprint manipulating functions.
    It is used by `chmap.views.edit_blueprint.CriteriaParser`.
    """

    POLICY_UNSET: int
    POLICY_SET: int
    POLICY_FORBIDDEN: int
    POLICY_LOW: int

    def __init__(self, s: NDArray[np.int_], x: NDArray[np.int_], y: NDArray[np.int_],
                 policies: dict[str, int]):
        if s.ndim != 1 or x.ndim != 1 or y.ndim != 1:
            raise ValueError()
        if (n := len(s)) != len(x) or n != len(y) or n == 0:
            raise ValueError()

        self.s = s
        self.x = x
        self.y = y
        self.dx = np.min(np.diff(np.unique(x)))
        self.dy = np.min(np.diff(np.unique(y)))
        if self.dx <= 0 or self.dy <= 0:
            raise ValueError(f'dx={self.dx}, dy={self.dy}')

        self._policies = policies

        self._blueprint: NDArray[np.int_] = None

    def id(self, v):
        return v

    def __getattr__(self, item: str):
        if item.startswith('POLICY_'):
            if (ret := self._policies.get(item[len('POLICY_'):], None)) is not None:
                return ret

        raise AttributeError(item)

    def blueprint(self) -> NDArray[np.int_]:
        return self._blueprint

    def set_blueprint(self, blueprint: NDArray[np.int_]):
        if len(blueprint) != len(self.s):
            raise ValueError()

        self._blueprint = blueprint

    def move(self, a: NDArray, *, tx: int = 0, ty: int = 0, shanks: list[int] = None, axis: int = 0, init=0) -> NDArray:
        """
        Move blueprint

        :param a: Array[V, ..., N, ...], where N means electrodes
        :param tx:
        :param ty:
        :param shanks: move electrode only on given shanks
        :param axis: index off N
        :param init: initial value
        :return:
        """
        s = self.s
        x = self.x
        y = self.y
        dx = self.dx
        dy = self.dy

        if a.shape[axis] != len(s):
            raise RuntimeError()

        pos = {
            (int(s[i]), int(x[i] / dx), int(y[i] / dy)): i
            for i in range(len(s))
        }

        ii = []
        jj = []
        for i in range(len(s)):
            if shanks is None or s[i] in shanks:
                p = int(s[i]), int((x[i] + tx) / dx), int((y[i] + ty) / dy)
                j = pos.get(p, None)
            else:
                j = i

            if j is not None:
                ii.append(i)
                jj.append(j)

        ii = np.array(ii)
        jj = np.array(jj)

        if a.ndim > 1:
            _index = [slice(None)] * a.ndim
            _index[axis] = ii
            ii = tuple(_index)
            _index[axis] = jj
            jj = tuple(_index)

        ret = np.full_like(a, init)
        ret[jj] = a[ii]

        return ret
