from __future__ import annotations

import functools
import inspect
import os
import re
import sys
import time
from collections.abc import Callable
from types import FunctionType
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

__all__ = [
    # pattern match utils
    'all_int',
    'align_arr', 'as_set',
    # dynamic import
    'import_name',
    # profile
    'TimeMarker',
    # documenting
    'doc_link'
]

T = TypeVar('T')


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

    Module Path: `[ROOT:]MODULE:NAME`, where

    ROOT:
        a filepath insert into `sys.path`.
    MODULE:
        module path
    NAME:
         variable name. Use '*' to return a module.


    :param desp:
    :param module_path:
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
            sys.path.pop(0)

    if name == '*':
        return module

    return getattr(module, name)


class TimeMarker:
    def __init__(self, disable=False):
        self.t = time.time()
        self.disable = disable

    def reset(self):
        self.t = time.time()

    def __call__(self, message: str = None) -> float:
        t = time.time()
        d = t - self.t
        self.t = t

        if message is not None and not self.disable:
            print(message, f'use {d:.2f}')

        return d


def doc_link(**kwargs: str) -> Callable[[T], T]:
    """
    A decorator to replace the text with pattern `{CLASS}`
    into sphinx cross-reference link (if environment variable `SPHINX_BUILD` is set)
    in the function document.

    Match rules:

    * `{class}` : `:class:~`
    * `{class#attr}` : `:attr:~`
    * `{class#meth()}` : `:meth:~`
    * `{#meth()}` : `:meth:~`
    * `{func()}` : `:func:~`
    * `{VAR}` : if VAR is a str, use its str content.
    * `{...}` : do not replace

    Limitation:

    * not support ForwardReference

    :param kwargs: extra
    :return:
    """
    g = dict(inspect.stack()[1].frame.f_locals)
    kwargs.update(g)

    def _decorator(func: T) -> T:
        if func.__doc__ is not None:
            func.__doc__ = replace_doc_link(kwargs, func.__doc__)
        return func

    return _decorator


def replace_doc_link(context: dict, doc: str) -> str:
    if len(os.environ.get('SPHINX_BUILD', '')):
        replace = functools.partial(sphinx_doc_link_replace, context)
        return re.sub(r'\{([a-zA-Z_.]+)?(#([a-zA-Z_]+))?(\(\))?}', replace, doc)

    return doc


def sphinx_doc_link_replace(context: dict, m: re.Match) -> str:
    k = m.group(1)
    attr = m.group(3)
    is_func = m.group(4)

    if k is not None:
        old_k = k
        try:
            k = context[k]
        except KeyError:
            pass
        else:
            if isinstance(k, type):
                module = k.__module__
                name = k.__name__
                k = f'~{module}.{name}'
            elif isinstance(k, FunctionType):
                module = k.__module__
                name = k.__name__
                k = f'~{module}.{name}'
            elif isinstance(k, str) and k.startswith('chmap'):
                k = f'~{k}'
            elif isinstance(k, str) and attr is None and is_func is None:
                return k
            else:
                k = old_k

    match (k, attr, is_func):
        case (func, None, '()'):
            return f':func:`{func}()`'
        case (None, func, '()'):
            return f':meth:`.{func}()`'
        case (name, None, None):
            return f':class:`{name}`'
        case (name, func, '()'):
            return f':meth:`{name}.{func}()`'
        case _:
            return m.group()  # do not replace
