from __future__ import annotations

import sys
from typing import NamedTuple, TYPE_CHECKING

import numpy as np

from chmap.util.atlas_slice import SLICE, SliceView, SlicePlane

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

if TYPE_CHECKING:
    from chmap.util.atlas_brain import BrainGlobeAtlas

__all__ = [
    'ProbeCoordinate',
    'get_slice_view',
    'new_slice_view',
    'get_plane_at'
]

BREGMA = (5400, 0, 5700)  # um
"""
TODO bregma 10um, (ap, dv, ml)=(540 0 570) index

"""


class ProbeCoordinate(NamedTuple):
    x: float  # ap (um)
    y: float  # dv (um)
    z: float  # ml (um)
    s: int = 0  # i-th shank

    rx: float = 0  # shank s x-axis rotate degree
    ry: float = 0  # shank s y-axis rotate degree
    rz: float = 0  # shank s z-axis rotate degree

    depth: float = 0  # shank s insert depth
    direction: tuple[float, float, float] | None = None  # shank direction (ap, dv, ml)

    def shank_at(self, s: int) -> Self:
        if self.direction is None:
            return self

        dx, dy, dz = self.direction
        dx *= s
        dy *= s
        dz *= s
        return self._replace(x=self.x + dx, y=self.y + dy, z=self.z + dz)

    def shank_0(self) -> Self:
        return self.shank_at(-self.s)

    @classmethod
    def from_bregma(cls, ap: float, ml: float, dv: float = 0, **kwargs) -> Self:
        """

        :param ap: um
        :param ml: um
        :param dv: um
        :param kwargs: {ProbeCoordinate}'s other parameters.
        :return:
        """
        x = BREGMA[0] - ap
        y = BREGMA[1] + dv
        z = BREGMA[2] - ml
        return ProbeCoordinate(x, y, z, **kwargs)


def get_slice_view(pc: ProbeCoordinate) -> SLICE | None:
    if pc.direction is None:
        return None

    pd = np.array(pc.direction)
    uc = np.array([0, 0, 1])
    us = np.array([1, 0, 0])
    ut = np.array([0, -1, 0])
    i = np.argmax(np.abs([
        np.dot(pd, uc),
        np.dot(pd, us),
        np.dot(pd, ut),
    ]))
    return ['coronal', 'sagittal', 'transverse'][i]


def new_slice_view(view: BrainGlobeAtlas | SliceView, pc: ProbeCoordinate) -> SliceView:
    if isinstance(view, SliceView):
        brain = view.brain
        _name = view.name
        reference = view.reference
    else:
        brain = view
        _name = 'coronal'
        reference = brain.reference

    if (name := get_slice_view(pc)) is None:
        name = _name

    return SliceView(brain, name, reference)


def get_plane_at(view: SliceView, pc: ProbeCoordinate) -> SlicePlane:
    a = np.deg2rad([pc.rx, pc.ry, pc.rz])
    r = view.angle_offset(tuple(a))
    return view.plane_at((pc.x, pc.y, pc.z), um=True).with_rotate(r)
