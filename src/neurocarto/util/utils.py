from __future__ import annotations

import functools
import inspect
import os
import re
import sys
import textwrap
from pathlib import Path
from types import FunctionType, ModuleType
from typing import TypeVar, Any, TypeGuard, overload

import numpy as np
from numpy.typing import NDArray

__all__ = [
    # pattern match utils
    'all_int', 'all_float',
    'align_arr', 'as_set',
    # dynamic import
    'import_name',
    'get_import_file',
    # documenting
    'SPHINX_BUILD',
    'doc_link'
]

T = TypeVar('T')

SPHINX_BUILD = len(os.environ.get('SPHINX_BUILD', '')) > 0


@overload
def all_int(x, /) -> TypeGuard[int]:
    pass


@overload
def all_int(*args) -> bool:
    pass


def all_int(*args) -> bool:
    for xx in args:
        if not isinstance(xx, (int, np.integer)):
            return False
    return True


@overload
def all_float(x, /) -> TypeGuard[float]:
    pass


@overload
def all_float(*args) -> bool:
    pass


def all_float(*args) -> bool:
    for xx in args:
        if not isinstance(xx, (int, float, np.number)):
            return False
    return True


def align_arr(*x: int | NDArray) -> list[NDArray]:
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
    load a symbol (could be a function, a class or module itself) from a module.

    Module Path: ``[ROOT:]MODULE:NAME``, where

    ROOT:
        a filepath prepend into ``sys.path``.
    MODULE:
        module path
    NAME:
         variable name. Use '*' to return a module.


    :param desp: description of the *module_path*. used in error message.
    :param module_path:
    :param root: PYTHONPATH
    :param reload: reload the module.
    :return:
    :raise ImportError:
    :raise ValueError: incorrect module path
    """
    if module_path.count(':') > 1:
        root, _, module_path = module_path.partition(':')
        return import_name(desp, module_path, root, reload=reload)

    module_path, _, name = module_path.partition(':')
    if len(name) == 0:
        raise ValueError(f'not a {desp} module path : {module_path}')

    import importlib
    try:
        if root is not None:
            sys.path.insert(0, root)

        module = importlib.import_module(module_path)
        if reload:
            module = importlib.reload(module)
    finally:
        if root is not None:
            sys.path.pop(0)

    if name == '*':
        return module

    try:
        return getattr(module, name)
    except AttributeError as e:
        raise ImportError(f"cannot load {desp} from {module_path}:{name}") from e


def get_import_file(module_path: str, root: str = None) -> Path | None:
    """
    Try to find python module file according to the *module_path*.

    :param module_path:
    :param root:
    :return: found filepath.
    """
    if module_path.count(':') > 1:
        root, _, module_path = module_path.partition(':')
        return get_import_file(module_path, root)

    module_path, _, _ = module_path.partition(':')
    module_file = Path(module_path.replace('.', '/') + '.py')
    if root is not None:
        if (p := Path(root) / module_file).exists():
            return p
    else:
        for root in sys.path:
            if (p := Path(root) / module_file).exists():
                return p
    return None


def doc_link(**kwargs: str):
    """
    A decorator to replace the text with pattern ``{CLASS}``
    into sphinx cross-reference link (if environment variable ``SPHINX_BUILD`` is set)
    in the function document.

    Match rules:

    * ``{class}`` : ``:class:~``
    * ``{class#attr}`` : ``class :attr:~``
    * ``{class#meth()}`` : ``class :meth:~``
    * ``{module#class}`` : ``:class:~``
    * ``{module#func()}`` : ``:func:~``
    * ``{#attr}`` : ``:attr:~``
    * ``{#meth()}`` : ``:meth:~``
    * ``{func()}`` : ``:func:~``
    * ``{VAR}`` : if VAR is a str, use its str content.
    * ``{...}`` : do not replace

    Limitation:

    * not support ForwardReference
    * not support attribute documents.

    :param kwargs: extra
    :return: decorator.
    """
    stack = inspect.stack()
    g = [kwargs, stack[1].frame.f_globals]

    def _decorator(func: T) -> T:
        if func.__doc__ is not None:
            func.__doc__ = replace_doc_link(g, func.__doc__)
        return func

    return _decorator


def replace_doc_link(context: list[dict], doc: str) -> str:
    if SPHINX_BUILD:
        replace = functools.partial(sphinx_doc_link_replace_ref, context)
        doc = re.sub(r'\{(?P<module>[a-zA-Z0-9_.]+)?(#(?P<attr>[a-zA-Z0-9_]+))?(?P<func>\(\))?}', replace, doc)

    replace = functools.partial(sphinx_doc_link_replace_word, context)
    doc = re.sub(r'(?:(?<=^)|(?<=\n))(?P<indent> *)\{(?P<attr>[a-zA-Z0-9_]+)}', replace, doc)

    return doc


def sphinx_doc_link_get(context: list[dict], attr: str) -> Any:
    for g in context:
        try:
            return g[attr]
        except KeyError:
            pass
    raise KeyError


def sphinx_doc_link_replace_word(context: list[dict], m: re.Match) -> str:
    indent = m.group('indent')
    attr = m.group('attr')

    try:
        value = sphinx_doc_link_get(context, attr)
    except KeyError:
        return m.group()

    if not isinstance(value, str):
        return m.group()

    if indent is None or len(indent) == 0:
        return value

    return textwrap.indent(value, indent)


def sphinx_doc_link_replace_ref(context: list[dict], m: re.Match) -> str:
    k = m.group('module')
    attr = m.group('attr')
    is_func = m.group('func')

    if k is not None:
        old_k = k
        try:
            k = sphinx_doc_link_get(context, old_k)
        except KeyError:
            pass
        else:
            if isinstance(k, ModuleType):
                module = k.__name__
                if attr is not None:
                    k = f'{module}.{attr}'
                    attr = None

            if isinstance(k, type):
                module = k.__module__
                name = k.__name__
                k = f'~{module}.{name}'
            elif isinstance(k, FunctionType):
                module = k.__module__
                name = k.__name__
                k = f'~{module}.{name}'
            elif isinstance(k, str) and k.startswith('neurocarto'):
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
            class_name = name[name.rfind('.') + 1:]
            return f'{class_name}. :meth:`{name}.{func}()`'
        case (name, attr, None):
            class_name = name[name.rfind('.') + 1:]
            return f'{class_name}.:attr:`{name}.{attr}`'
        case _:
            return m.group()  # do not replace
