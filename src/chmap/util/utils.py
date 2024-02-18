from __future__ import annotations

import functools
import inspect
import os
import re
import sys
import textwrap
import time
from collections.abc import Callable
from pathlib import Path
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
    'get_import_file',
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


def import_name(desp: str, module_path: str, root: str = None, *, reload=False):
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
    :param reload: reload the module.
    :return:
    """
    if module_path.count(':') > 1:
        root, _, module_path = module_path.partition(':')
        return import_name(desp, module_path, root, reload=reload)

    module, _, name = module_path.partition(':')
    if len(name) == 0:
        raise ValueError(f'not a {desp} pattern "module_path:name" : {module_path}')

    import importlib
    try:
        if root is not None:
            sys.path.insert(0, root)

        module = importlib.import_module(module)
        if reload:
            module = importlib.reload(module)
    finally:
        if root is not None:
            sys.path.pop(0)

    if name == '*':
        return module

    return getattr(module, name)


def get_import_file(module_path: str, root: str = None) -> Path | None:
    """
    Try to find correspond python module file.

    :param module_path:
    :param root:
    :return: found path.
    """
    if module_path.count(':') > 1:
        root, _, module_path = module_path.partition(':')
        return get_import_file(module_path, root)

    module, _, _ = module_path.partition(':')
    module_path = Path(module.replace('.', '/') + '.py')
    if root is not None:
        if (p := Path(root) / module_path).exists():
            return p
    else:
        for root in sys.path:
            if (p := Path(root) / module_path).exists():
                return p
    return None


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
    * `{#attr}` : `:attr:~`
    * `{#meth()}` : `:meth:~`
    * `{func()}` : `:func:~`
    * `{VAR}` : if VAR is a str, use its str content.
    * `{...}` : do not replace

    Limitation:

    * not support ForwardReference

    :param kwargs: extra
    :return:
    """
    stack = inspect.stack()
    g = {}
    g.update({  # function scope
        name: value
        for name, value in stack[1].frame.f_locals.items()
        if not name.startswith('_')
    })
    g.update({  # method scope
        name: value
        for name, value in stack[2].frame.f_locals.items()
        if not name.startswith('_')
    })
    kwargs.update(g)

    def _decorator(func: T) -> T:
        if func.__doc__ is not None:
            func.__doc__ = replace_doc_link(kwargs, func.__doc__)
        return func

    return _decorator


def replace_doc_link(context: dict, doc: str) -> str:
    if len(os.environ.get('SPHINX_BUILD', '')) or True:
        replace = functools.partial(sphinx_doc_link_replace_ref, context)
        doc = re.sub(r'\{(?P<module>[a-zA-Z_.]+)?(#(?P<attr>[a-zA-Z_]+))?(?P<func>\(\))?}', replace, doc)

        replace = functools.partial(sphinx_doc_link_replace_word, context)
        doc = re.sub(r'(?:(?<=^)|(?<=\n))(?P<indent> +)\{(?P<attr>[a-zA-Z_]+)}', replace, doc)

    return doc


def sphinx_doc_link_replace_word(context: dict, m: re.Match) -> str:
    indent = m.group('indent')
    attr = m.group('attr')

    try:
        value = context[attr]
    except KeyError:
        return m.group()

    if indent is None:
        return value

    return textwrap.indent(value, indent)


def sphinx_doc_link_replace_ref(context: dict, m: re.Match) -> str:
    k = m.group('module')
    attr = m.group('attr')
    is_func = m.group('func')

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
                return m.group()  # pass to sphinx_doc_link_replace_word
            else:
                k = old_k

    match (k, attr, is_func):
        case (func, None, '()'):
            return f':func:`{func}()`'
        case (None, func, '()'):
            return f':meth:`.{func}()`'
        case (None, attr, None):
            return f':attr:`.{attr}`'
        case (name, None, None):
            return f':class:`{name}`'
        case (name, func, '()'):
            return f':meth:`{name}.{func}()`'
        case (name, attr, None):
            return f':attr:`{name}.{attr}`'
        case _:
            return m.group()  # do not replace
