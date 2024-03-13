from __future__ import annotations

import sys
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from neurocarto.util.atlas_slice import SliceView, SlicePlane

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

__all__ = [
    'ProbeCoordinate',
    'get_plane_at',
    'prepare_affine_matrix',
    'prepare_affine_matrix_both',
    'project',
    'project_b2i',
    'project_i2b'
]


class ProbeCoordinate(NamedTuple):
    x: float
    """ap (um)"""
    y: float
    """dv (um)"""
    z: float
    """ml (um)"""
    s: int = 0
    """i-th shank"""

    rx: float = 0
    """shank s x-axis (ap) rotate degree"""
    ry: float = 0
    """shank s y-axis (dv) rotate degree"""
    rz: float = 0
    """shank s z-axis (ml) rotate degree"""

    depth: float = 0
    """shank s insert depth (um)"""

    @classmethod
    def from_bregma(cls, atlas_name: str, ap: float, dv: float, ml: float, ref: str = 'bregma', **kwargs) -> Self:
        """

        :param atlas_name: atlas brain name
        :param ap: um
        :param dv: um
        :param ml: um
        :param ref: reference origin, default use 'bregma'
        :param kwargs: {ProbeCoordinate}'s other parameters.
        :return:
        :raises KeyError:
        """
        from neurocarto.util.atlas_brain import REFERENCE
        bregma = REFERENCE[ref][atlas_name]

        x = bregma[0] - ap
        y = bregma[1] + dv
        z = bregma[2] + ml
        return ProbeCoordinate(x, y, z, **kwargs)  # type: ignore[return-value]


def get_plane_at(view: SliceView, pc: ProbeCoordinate) -> SlicePlane:
    a = np.deg2rad([pc.rx, pc.ry, pc.rz])
    dw, dh = view.angle_offset(tuple(a))
    return view.plane_at((pc.x, pc.y, pc.z), um=True).with_offset(dw, dh)


def prepare_affine_matrix(dx: float, dy: float, sx: float, sy: float, rt: float) -> NDArray[np.float_]:
    r"""

    :param dx: x-axis offset
    :param dy: y-axis offset
    :param sx: x-axis scaling
    :param sy: y-axis scaling
    :param rt: rotate in degree
    :return: :math:`A_{3 \times 3}`
    """
    rt = np.deg2rad(rt)

    td = np.array([
        [1, 0, dx],
        [0, 1, dy],
        [0, 0, 1],
    ], dtype=float)
    cos = np.cos(rt)
    sin = np.sin(rt)
    tr = np.array([
        [cos, -sin, 0],
        [sin, cos, 0],
        [0, 0, 1],
    ], dtype=float)
    ts = np.array([
        [sx, 0, 0],
        [0, sy, 0],
        [0, 0, 1],
    ], dtype=float)

    return td @ ts @ tr


def prepare_affine_matrix_both(dx: float, dy: float, sx: float, sy: float, rt: float) -> tuple[NDArray[np.float_], NDArray[np.float_]]:
    r"""

    :param dx: x-axis offset
    :param dy: y-axis offset
    :param sx: x-axis scaling
    :param sy: y-axis scaling
    :param rt: rotate in degree
    :return: tuple of (:math:`A_{3 \times 3}`, :math:`A_{3 \times 3}^{-1}`)
    """
    rt = np.deg2rad(rt)

    td = np.array([
        [1, 0, dx],
        [0, 1, dy],
        [0, 0, 1],
    ], dtype=float)
    td_ = np.array([
        [1, 0, -dx],
        [0, 1, -dy],
        [0, 0, 1],
    ], dtype=float)
    cos = np.cos(rt)
    sin = np.sin(rt)
    tr = np.array([
        [cos, -sin, 0],
        [sin, cos, 0],
        [0, 0, 1],
    ], dtype=float)
    tr_ = np.array([
        [cos, sin, 0],
        [-sin, cos, 0],
        [0, 0, 1],
    ], dtype=float)
    ts = np.array([
        [sx, 0, 0],
        [0, sy, 0],
        [0, 0, 1],
    ], dtype=float)
    ts_ = np.array([
        [1 / sx, 0, 0],
        [0, 1 / sy, 0],
        [0, 0, 1],
    ], dtype=float)
    return td @ ts @ tr, tr_ @ ts_ @ td_


def project(a: NDArray[np.float_], p: NDArray[np.float_]) -> NDArray[np.float_]:
    r"""
    project coordinate from image-origin to probe-origin, or opposite
    (depends on :math:`A_{3 \times 3}` to probe-origin or :math:`A_{3 \times 3}^{-1}` to image-origin).

    :param a: Array[float, 3, 3] affine transform matrix
    :param p: Array[float, (x,y[,1]) [,N]] position
    :return: transform position, same shape as *p*.
    """
    if a.shape != (3, 3):
        raise ValueError()

    match p.shape:
        case (2, ):
            return (a @ np.concatenate([p, [1]]))[[0, 1]]
        case (3, ):
            return a @ p
        case (2, n):
            q = np.vstack([p, np.ones_like((n,))])
            return (a @ q)[[0, 1], :]
        case (3, n):
            return a @ p
        case _:
            raise ValueError()


def project_b2i(bregma: tuple[float, float, float], view: SlicePlane, p: NDArray[np.float_], *,
                keep_plane: bool = False) -> NDArray[np.float_]:
    """
    project coordinate from bregma-origin to image-origin.

    :param bregma: bregma (ap, dv, ml) in um
    :param view: projection view
    :param p: Array[float, (ap, dv, ml) [,N]] position in um
    :param keep_plane: keep plane position (um) in return. Otherwise, p will be filled with 1.
    :return: Array[float, (x, y, p) [,N]] transform position in um
    """
    match p.ndim:
        case 1:
            q = np.empty((3,), dtype=float)
        case 2:
            n = p.shape[1]
            q = np.empty((3, n), dtype=float)
        case _:
            raise ValueError()

    cx = view.width / 2
    cy = view.height / 2

    q[0] = bregma[0] - p[0]
    q[1] = p[1] + bregma[1]
    q[2] = p[2] + bregma[2]

    q = q[view.slice.project_index, :]
    q = q[[1, 2, 0], :]

    q[0] = q[0] - cx
    q[1] = cy - q[1]
    if not keep_plane:
        q[2] = 1

    return q


def project_i2b(bregma: tuple[float, float, float], view: SlicePlane, p: NDArray[np.float_]) -> NDArray[np.float_]:
    """
    project coordinate from image-origin to bregma-origin.

    :param bregma: bregma (ap, dv, ml) in um
    :param view: projection view
    :param p: Array[float, (x,y[,1]) [,N]] position in um
    :return: Array[float, (ap, dv, ml) [,N]] transform position in um
    """
    match p.ndim:
        case 1:
            q = np.empty((3,), dtype=float)
        case 2:
            n = p.shape[1]
            q = np.empty((3, n), dtype=float)
        case _:
            raise ValueError()

    cx = view.width / 2
    cy = view.height / 2

    x = p[0] + cx
    y = cy - p[1]

    pi, xi, yi = view.slice.project_index
    q[xi] = x
    q[yi] = y
    q[pi] = view.plane_idx_at(x, y, um=True) * view.resolution

    q[0] = bregma[0] - q[0]
    q[1] -= bregma[1]
    q[2] -= bregma[2]

    return q
