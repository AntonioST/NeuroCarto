from collections.abc import Callable, Sequence
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
    Is *a* sorted?

    `reference <https://stackoverflow.com/a/47004507>`_

    :param a: Array[number, N]
    :param strict: strict increase
    :return:
    """
    if strict:
        return bool(np.all(a[:-1] < a[1:]))
    else:
        return bool(np.all(a[:-1] <= a[1:]))


def same_index(a: NDArray[np.number]) -> list[NDArray[np.int_]]:
    """
    Get index of save value.

    :param a: Array[V, N] or Array[V, N, M]
    :return: list of index Array[N, *] for particular duplicated value V.
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


def closest_point_index(a: NDArray[np.float_], p: float | Sequence[float] | NDArray[np.float_], v: float) -> int | None:
    """
    Find the index of closed point in *a*.

    :param a: Array[V:float, N[, D]]
    :param p: V or Array[V:float, D]
    :param v: V threshold
    :return: index of N
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
             val: int | list[int] | NDArray[np.int_],
             missing: int | Literal['error', 'drop'] = 'error') -> NDArray[np.int_]:
    """
    Get index of *ref* for each value in *a*.

    :param ref: reference Array[V, A].
    :param val: value Array[V, B].
    :param missing: {'error', 'drop'} or int
    :return: index Array[A, B]
    :raise TypeError: unknown 'missing'
    :raise ValueError: channel not in *ref*
    """
    if isinstance(missing, str):
        if missing not in ('error', 'drop'):
            raise TypeError(f'unknown missing : {missing}')
    elif not isinstance(missing, int):
        raise TypeError(f'unknown missing : {missing}')

    val = np.asarray(val)
    if len(_missing := np.setdiff1d(val, ref)) > 0:
        if missing == 'error':
            raise ValueError(f'do not contain all channels : {_missing}')

    if is_sorted(ref):
        idx = np.searchsorted(ref, val, side='left')
        if missing == 'drop':
            tmp = idx < len(ref)
            idx = idx[tmp]
            val = val[tmp]
            idx = idx[ref[idx] == val]
        elif isinstance(missing, int):
            idx[ref[idx % len(ref)] != val] = missing
            idx[idx >= len(ref)] = missing

    else:
        diff = np.subtract.outer(ref, val)
        idx, ord = np.nonzero(diff == 0)
        if missing == 'drop':
            idx = idx[np.argsort(ord)]
        elif isinstance(missing, int):
            tmp = np.full_like(val, missing)
            tmp[ord] = idx
            idx = tmp

    return np.atleast_1d(idx)


def interpolate_nan(a: NDArray[np.float_],
                    kernel: int | tuple[int, int] = 1,
                    iteration: int = 1,
                    f: str | Callable[[NDArray[np.float_]], float] = 'mean',
                    n: float = np.nan) -> NDArray[np.float_]:
    """
    interpolate NaN value in *a*.

    :param a: image Array[float, (N,) Y, X]
    :param kernel: int or (sy:int, sx:int)
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

    if a.ndim == 2:
        y, x = a.shape
        z = 0
    elif a.ndim == 3:
        z, y, x = a.shape
    else:
        raise RuntimeError()

    match kernel:
        case (int(sy), int(sx)):
            gy, gx = np.mgrid[-sy:sy + 1, -sx:sx + 1]
        case int(kernel):
            gy, gx = np.mgrid[-kernel:kernel + 1, -kernel:kernel + 1]
        case _:
            raise TypeError(f'{kernel=}')

    gx = gx.ravel()
    gy = gy.ravel()

    for _ in range(iteration):
        if z == 0:
            if not _interpolate_nan(a, gy, gx, f):
                break
        else:
            false_count = 0
            for i in range(z):
                if not _interpolate_nan(a[i], gy, gx, f):
                    false_count += 1
            if false_count == n:
                break

    if not np.isnan(n):
        a[np.isnan(a)] = n

    return a


def _interpolate_nan(a: NDArray[np.float_], gy: NDArray[np.int_], gx: NDArray[np.int_], f) -> bool:
    y, x = a.shape

    jj, ii = np.nonzero(np.isnan(a))
    if len(jj) == 0:
        return False

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

    return True
