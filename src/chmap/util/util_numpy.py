from collections.abc import Callable
from typing import Literal

import numpy as np
from numpy.typing import NDArray

__all__ = [
    'is_sorted',
    'same_index',
    'closest_point_index',
    'index_of',
    'interpolate_nan'
]


def is_sorted(a: NDArray[np.number], strict=False) -> bool:
    """
    [reference](https://stackoverflow.com/a/47004507)

    :param a:
    :param strict:
    :return:
    """
    if strict:
        return bool(np.all(a[:-1] < a[1:]))
    else:
        return bool(np.all(a[:-1] <= a[1:]))


def same_index(a: NDArray[np.number]) -> list[NDArray[np.int_]]:
    """

    :param a: Array[V, N] or Array[V, N, M]
    :return: list of Array[N, *]
    """
    if a.ndim not in (1, 2):
        raise ValueError(f'wrong dimension number : {a.ndim}')

    if np.issubdtype(a.dtype, int):
        if a.ndim == 1:
            d = np.subtract.outer(a, a)
            return [np.nonzero(a == a[k])[0] for k in _same_index_d(d)]
        else:
            d = np.sum(np.abs([
                np.subtract.outer(a[:, x], a[:, x])
                for x in range(a.shape[1])
            ]), axis=0)
            return [
                np.nonzero(np.sum(np.abs(a - a[k]), axis=1) == 0)[0]
                for k in _same_index_d(d)
            ]

    elif np.issubdtype(a.dtype, float):
        if a.ndim == 1:
            d = np.subtract.outer(a, a)
            return [np.nonzero(np.abs(a - a[k]) <= 1E-5)[0] for k in _same_index_d(d)]
        else:
            d = np.sum(np.abs([
                np.subtract.outer(a[:, x], a[:, x])
                for x in range(a.shape[1])
            ]), axis=0)
            return [
                np.nonzero(np.sum(np.abs(a - a[k]), axis=1) <= 1E-5)[0]
                for k in _same_index_d(d)
            ]
    else:
        raise ValueError(f'unsupported dtype : {a.dtype}')


def _same_index_d(d: NDArray[np.number]) -> NDArray[np.int_]:
    n = d.shape[0]
    a = np.arange(n)
    d[a, a] = 1

    j, k = np.nonzero(d == 0)
    if len(j) == 0:
        return np.array([])
    a[:] = n
    a[j] = j
    for i, kk in enumerate(k):
        j[i] = a[kk] = min(a[kk], j[i])

    return np.unique(j)


def closest_point_index(a: NDArray[np.float_], p: float | list[float] | NDArray[np.float_], v: float) -> int | None:
    """

    :param a: Array[V:float, N[, D]]
    :param p: V or Array[V:float, D]
    :param v: V threshold
    :return: N index
    """

    if a.ndim == 1:
        p = float(p)

    elif a.ndim == 2:
        p = np.asarray(p)

        an, ad = a.shape
        if p.shape != (ad,):
            raise ValueError(f'{a.shape=}[1] != {p.shape}')
    else:
        raise RuntimeError()

    if a.ndim == 1:
        o = np.abs(a - p)
    elif a.ndim == 2:
        o = np.sqrt(np.sum((a - p) ** 2, axis=1))
    else:
        raise RuntimeError()

    i = np.argmin(o)
    if o[i] <= v:
        return int(i)
    else:
        return None


def index_of(ref: NDArray[np.int_],
             a: int | list[int] | NDArray[np.int_],
             missing: int | Literal['error', 'drop'] = 'error') -> NDArray[np.int_]:
    """Get channels index in *ref* channel

    :param ref: reference array.
    :param a: value array.
    :param missing: {'error', 'drop'} or int
    :return: a value index in ref
    :raise TypeError: unknown 'missing'
    :raise ValueError: channel not in *ref*
    """
    if isinstance(missing, str):
        if missing not in ('error', 'drop'):
            raise TypeError(f'unknown missing : {missing}')
    elif not isinstance(missing, int):
        raise TypeError(f'unknown missing : {missing}')

    a = np.asarray(a)
    if len(_missing := np.setdiff1d(a, ref)) > 0:
        if missing == 'error':
            raise ValueError(f'do not contain all channels : {_missing}')

    if is_sorted(ref):
        idx = np.searchsorted(ref, a, side='left')
        if missing == 'drop':
            tmp = idx < len(ref)
            idx = idx[tmp]
            a = a[tmp]
            idx = idx[ref[idx] == a]
        elif isinstance(missing, int):
            idx[ref[idx % len(ref)] != a] = missing
            idx[idx >= len(ref)] = missing

    else:
        diff = np.subtract.outer(ref, a)
        idx, ord = np.nonzero(diff == 0)
        if missing == 'drop':
            idx = idx[np.argsort(ord)]
        elif isinstance(missing, int):
            tmp = np.full_like(a, missing)
            tmp[ord] = idx
            idx = tmp

    return np.atleast_1d(idx)


def interpolate_nan(a: NDArray[np.float_],
                    space: int | tuple[int, int] = 1,
                    iteration: int = 1,
                    f: str | Callable[[NDArray[np.float_]], float] = 'mean',
                    n: float = np.nan) -> NDArray[np.float_]:
    """

    :param a: image Array[float, Y, X]
    :param space: int or (sx:int, sy:int)
    :param iteration:
    :param f: interpolate function (Array[float]) -> float
    :param n: fill value after iteration
    :return: image array
    """
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

    y, x = a.shape
    if isinstance(space, tuple):
        sx, sy = space
        gy, gx = np.mgrid[-sy:sy + 1, -sx:sx + 1]
    else:
        space = int(space)
        gy, gx = np.mgrid[-space:space + 1, -space:space + 1]
    gx = gx.ravel()
    gy = gy.ravel()

    for _ in range(iteration):
        jj, ii = np.nonzero(np.isnan(a))
        if len(jj) == 0:
            break

        r = np.full_like(jj, np.nan, dtype=a.dtype)
        for k, j, i in zip(range(len(jj)), jj, ii):
            gj = gy + j
            gi = gx + i
            gj = np.where(gj < 0, j, gj)
            gi = np.where(gi < 0, i, gi)
            gj = np.where(gj >= y, j, gj)
            gi = np.where(gi >= x, i, gi)

            r[k] = f(a[gj, gi])

        a[jj, ii] = r

    if not np.isnan(n):
        a[np.isnan(a)] = n

    return a
