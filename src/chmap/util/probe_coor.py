from __future__ import annotations

import sys
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from chmap.util.atlas_slice import SliceView, SlicePlane

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

__all__ = [
    'ProbeCoordinate',
    'get_plane_at',
    'prepare_affine_matrix',
    'prepare_affine_matrix_both',
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
        from chmap.util.atlas_brain import REFERENCE
        bregma = REFERENCE[ref][atlas_name]

        x = bregma[0] - ap
        y = bregma[1] + dv
        z = bregma[2] + ml
        return ProbeCoordinate(x, y, z, **kwargs)


def get_plane_at(view: SliceView, pc: ProbeCoordinate) -> SlicePlane:
    a = np.deg2rad([pc.rx, pc.ry, pc.rz])
    r = view.angle_offset(tuple(a))
    return view.plane_at((pc.x, pc.y, pc.z), um=True).with_rotate(r)


def prepare_affine_matrix(dx: float, dy: float, sx: float, sy: float, rt: float) -> NDArray[np.float_]:
    """

    :param dx: x-axis offset
    :param dy: y-axis offset
    :param sx: x-axis scaling
    :param sy: y-axis scaling
    :param rt: rotate in degree
    :return: A
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
    """

    :param dx: x-axis offset
    :param dy: y-axis offset
    :param sx: x-axis scaling
    :param sy: y-axis scaling
    :param rt: rotate in degree
    :return: tuple of (A, A^-1)
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
