from __future__ import annotations

import inspect

import numpy as np
from numpy.typing import NDArray

__all__ = ['all_int', 'align_arr', 'as_set', 'is_recursive_called']


def all_int(*x) -> bool:
    for xx in x:
        if not isinstance(xx, (int, np.integer)):
            return False
    return True


def align_arr(*x: int | NDArray[np.int_]) -> list[NDArray[np.int_]]:
    if len(x) < 2:
        raise RuntimeError('not enough inputs')

    ret = [np.asarray(it) for it in x]
    sz = set([it.shape for it in ret if it.ndim > 0])
    if len(sz) != 1:
        raise RuntimeError('input ndim not aligned')
    shape = list(sz)[0]

    return [np.full(shape, it) if it.ndim == 0 else it for it in ret]


def as_set(x, n: int) -> set[int]:
    if x is None:
        return set(range(n))
    if all_int(x):
        return {int(x)}
    elif isinstance(x, slice):
        return set(range(n)[x])
    elif isinstance(x, tuple):
        ret = set()
        for xx in x:
            ret.update(as_set(xx, n))
        return ret
    else:
        return set(map(int, x))


def is_recursive_called(limit=100) -> bool:
    stack = inspect.stack()
    caller = stack[1]

    for i, frame in enumerate(stack[2:]):
        if i < limit and frame.filename == caller.filename and frame.function == caller.function:
            return True
    return False
