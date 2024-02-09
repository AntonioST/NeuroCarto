from collections.abc import Callable

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

    def move(self, a: NDArray, *,
             tx: int = 0, ty: int = 0,
             shanks: list[int] = None,
             axis: int = 0,
             init: float = 0) -> NDArray:
        """
        Move blueprint

        :param a: Array[V, ..., N, ...], where N means electrodes
        :param tx: x movement in um.
        :param ty: y movement in um.
        :param shanks: move electrode only on given shanks
        :param axis: index off N
        :param init: initial value
        :return: moved a (copied)
        """
        s = self.s
        x = self.x
        y = self.y
        dx = self.dx
        dy = self.dy

        if a.shape[axis] != len(s):
            raise RuntimeError()

        if abs(tx) < dx and abs(ty) < dy:
            return a

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

    def move_i(self, a: NDArray, *,
               tx: int = 0, ty: int = 0,
               shanks: list[int] = None,
               axis: int = 0,
               init: float = 0) -> NDArray:
        """
        Move blueprint by step.

        :param a: Array[V, ..., N, ...], where N means electrodes
        :param tx: number of dx
        :param ty: number of dy
        :param shanks: move electrode only on given shanks
        :param axis: index off N
        :param init: initial value
        :return: moved a (copied)
        """
        if tx == 0 and ty == 0:
            return a
        return self.move(a, tx=tx * self.dx, ty=ty * self.dy, shanks=shanks, axis=axis, init=init)

    def merge(self, blueprint: NDArray[np.int_], other: NDArray[np.int_] = None) -> NDArray[np.int_]:
        """
        merge blueprint. The latter result overwrite former result.

        `merge(blueprint)` works like `merge(blueprint(), blueprint)`.

        :param blueprint:
        :param other:
        :return:
        """
        if other is None:
            if self._blueprint is None:
                return blueprint

            other = blueprint
            blueprint = self._blueprint

        return np.where(other == self.POLICY_UNSET, blueprint, other)

    def interpolate_nan(self, a: NDArray[np.float_],
                        kernel: int | tuple[int, int] = 1,
                        f: str | Callable[[NDArray[np.float_]], float] = 'mean',
                        iteration: int = 1,
                        init: float = np.nan) -> NDArray[np.float_]:
        if isinstance(f, str):
            if f == 'mean':
                f = np.nanmean
            elif f == 'median':
                f = np.nanmedian
            elif f == 'min':
                f = np.nanmin
            elif f == 'max':
                f = np.nanmax
            else:
                raise ValueError()

        match kernel:
            case int(y) if y > 0:
                kernel = (0, y)
            case (int(x), int(y)) if x > 0 and y > 0:
                pass
            case _:
                raise TypeError()

        for _ in range(iteration):
            r = []
            for tx in range(-kernel[0], kernel[0] + 1):
                for ty in range(-kernel[1], kernel[1] + 1):
                    r.append(self.move_i(a, tx=tx, ty=ty, init=init))

            a = f(r, axis=0)

        return a

    def fill(self, blueprint: NDArray[np.int_],
             policy: list[int] = None,
             threshold: int = None) -> NDArray[np.int_]:
        """
        make the area occupied by policies be filled as rectangle.

        :param blueprint:
        :param policy:
        :param threshold:
        :return:
        """
        pass
