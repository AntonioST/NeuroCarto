from __future__ import annotations

import sys
import time

import numpy as np
from numpy.typing import NDArray

__all__ = ['all_int', 'align_arr', 'as_set', 'import_name', 'TimeMaker']


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
    elif isinstance(x, range):
        n = n if x.stop is None else min(n, x.stop)
        return set(range(x.start, n, abs(x.step)))
    elif isinstance(x, tuple):
        ret = set()
        for xx in x:
            ret.update(as_set(xx, n))
        return ret
    else:
        return set(map(int, x))


def import_name(desp: str, module_path: str, root: str = None):
    """

    :param desp:
    :param module_path: '[ROOT:]MODULE:NAME'
    :param root: PYTHONPATH
    :return:
    """
    if module_path.count(':') > 1:
        root, _, module_path = module_path.partition(':')
        return import_name(desp, module_path, root)

    module, _, name = module_path.partition(':')
    if len(name) == 0:
        raise ValueError(f'not a {desp} pattern "module_path:name" : {module_path}')

    import importlib
    try:
        if root is not None:
            sys.path.insert(0, root)

        module = importlib.import_module(module)
    finally:
        if root is not None:
            sys.path.pop()

    return getattr(module, name)


class TimeMaker:
    def __init__(self):
        self.t = time.time()

    def reset(self):
        self.t = time.time()

    def __call__(self, message: str):
        t = time.time()
        d = t - self.t
        print(message, f'use {d:.2f}')
        self.t = t
