from __future__ import annotations

import abc
import math
import sys
from typing import Literal, TypeVar, Final, overload, NamedTuple, get_args, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from neurocarto.util.utils import all_int, align_arr, all_float

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

if TYPE_CHECKING:
    from neurocarto.util.atlas_brain import BrainGlobeAtlas

__all__ = ['SLICE', 'SliceView', 'SlicePlane']

SLICE = Literal['coronal', 'sagittal', 'transverse']
T = TypeVar('T')

XY = tuple[int, int]
PXY = tuple[int, int, int]  # (plane, x, y)
COOR = tuple[int, int, int] | tuple[float, float, float]  # (ap, dv, ml)


class SliceView(metaclass=abc.ABCMeta):
    """Atlas brain slice view. Here provide three kinds of view ('coronal', 'sagittal', 'transverse').
    """

    name: Final[SLICE]
    """view"""

    resolution: Final[int]
    """um/pixel"""

    reference: Final[NDArray[np.uint]]
    """reference brain volume with shape (AP, DV, ML)"""

    grid_x: Final[NDArray[np.int_]]
    grid_y: Final[NDArray[np.int_]]

    def __new__(cls, brain: BrainGlobeAtlas, name: SLICE, reference: NDArray[np.uint] = None):
        if name == 'coronal':
            return object.__new__(CoronalView)
        elif name == 'sagittal':
            return object.__new__(SagittalView)
        elif name == 'transverse':
            return object.__new__(TransverseView)
        else:
            raise ValueError()

    def __init__(self, brain: BrainGlobeAtlas, name: SLICE, reference: NDArray[np.uint] = None):
        """

        :param name: view
        :param reference: reference brain volume with shape (AP, DL, ML)
        :param resolution: um/pixel
        """
        if reference is not None:
            if reference.shape != brain.reference.shape:
                raise RuntimeError()
        else:
            reference = brain.reference

        self.brain = brain
        """Atlas brain"""

        self.name = name
        """slice plane projection"""

        self.reference = reference
        """Image Array[uint, AP, DV, ML]"""

        self.resolution = int(brain.resolution[get_args(SLICE).index(name)])  # FIXME change to (x,y,z)
        """um/pixel"""

        self.grid_y, self.grid_x = np.mgrid[0:self.height, 0:self.width]

    def __str__(self):
        return f'SliceView[{self.name}]'

    __repr__ = __str__

    @property
    def n_ap(self) -> int:
        """Number of slices in AP axis"""
        return self.reference.shape[0]

    @property
    def n_dv(self) -> int:
        """Number of slices in DV axis"""
        return self.reference.shape[1]

    @property
    def n_ml(self) -> int:
        """Number of slices in ML axis"""
        return self.reference.shape[2]

    @property
    @abc.abstractmethod
    def n_plane(self) -> int:
        """Number of plane (pixel) in this view"""
        pass

    @property
    @abc.abstractmethod
    def width(self) -> int:
        """width (pixel) in this view"""
        pass

    @property
    @abc.abstractmethod
    def height(self) -> int:
        """height (pixel) in this view"""
        pass

    @property
    def width_um(self) -> float:
        """width (um) in this view"""
        return self.width * self.resolution

    @property
    def height_um(self) -> float:
        """height (um) in this view"""
        return self.height * self.resolution

    @property
    @abc.abstractmethod
    def project_index(self) -> tuple[int, int, int]:
        """index order of (ap, dv, ml).

        :return: (p, x, y)
        """
        pass

    def plane(self, o: int | tuple[int, int, int] | NDArray[np.int_], image: NDArray[np.uint] = None) -> NDArray[np.uint]:
        """Get brain image on plane *o*.

        :param o: plane, tuple (plane, dh, dv) or Array[plane:int, H, W]
        :param image: brain volume with shape (AP, DL, ML)
        :return: brain slice image with shape (height, width)
        """
        match o:
            case o if all_int(o):
                o = np.full_like((self.height, self.width), o)
            case (plane, dh, dv) if all_int(plane, dh, dv):
                o = plane + self.offset(dh, dv)
            case _ if isinstance(o, np.ndarray):
                if o.shape != (self.height, self.width):
                    raise RuntimeError(f'shape mismatch : {o.shape} != {(self.height, self.width)}')
            case _:
                raise TypeError(repr(o))

        if image is not None:
            if image.shape != self.reference.shape:
                raise RuntimeError('shape of brain volume mismatch')
        else:
            image = self.reference

        o = np.clip(o, 0, self.n_plane - 1)
        return image[self.coor_on(o, (self.grid_x, self.grid_y))]

    @overload
    def coor_on(self, plane: int, o: XY | tuple[float, float], *, um=False) -> COOR:
        pass

    @overload
    def coor_on(self, plane: int | NDArray[np.int_], o: tuple[NDArray, NDArray], *, um=False) -> tuple[NDArray[np.int_], NDArray[np.int_], NDArray[np.int_]]:
        pass

    @overload
    def coor_on(self, plane: int | NDArray[np.int_], o: NDArray[np.int_], *, um=False) -> NDArray[np.int_]:
        pass

    def coor_on(self, plane, o, *, um=False):
        """map slice point (x, y) at plane *plane* back to volume point (ap, dv, ml).

        :param plane: plane number of array
        :param o: tuple of (x, y) or Array[int|float, N, (x, y)]
        :param um: use um?
        :return: index (ap, dv, ml) or Array[int, N, (ap, dv, ml)]
        """
        pidx, xidx, yidx = self.project_index
        match o:
            case (x, y) if not um and all_int(plane, x, y):
                ret = [0, 0, 0]
                ret[pidx] = plane
                ret[xidx] = x
                ret[yidx] = y
                return tuple(ret)

            case (x, y) if um and all_int(plane) and all_float(x, y):
                ret = [0, 0, 0]
                ret[pidx] = plane
                ret[xidx] = int(x / self.resolution)
                ret[yidx] = int(y / self.resolution)
                return tuple(ret)

            case (x, y):
                plane, x, y = align_arr(plane, x, y)
                if um:
                    plane = plane.astype(int)
                    x = (x / self.resolution).astype(int)
                    y = (y / self.resolution).astype(int)

                ret = [0, 0, 0]
                ret[pidx] = plane
                ret[xidx] = x
                ret[yidx] = y
                return tuple(ret)

            case _ if isinstance(o, np.ndarray):
                ret = np.zeros((len(o), 3), dtype=int)
                ret[:, pidx] = plane
                ret[:, xidx] = o[:, 0] if not um else (o[:, 0] / self.resolution).astype(int)
                ret[:, yidx] = o[:, 1] if not um else (o[:, 1] / self.resolution).astype(int)
                return ret
            case _:
                raise TypeError()

    @overload
    def project(self, t: COOR, *, um=False) -> PXY:
        pass

    @overload
    def project(self, t: NDArray, *, um=False) -> NDArray[np.int_]:
        pass

    def project(self, t, *, um=False):
        """project volume point (ap, dv, ml) onto slice point (plane, x, y)

        :param t:  (ap, dv, ml) or Array[int, [N,], (ap, dv, ml)].
        :param um: use um?
        :return: (plane, x, y) or Array[int, [N,], (plane, x, y)]
        """
        p, x, y = self.project_index
        match t:
            case (ap, dv, ml) if not um and all_int(ap, dv, ml):
                return int(t[p]), int(t[x]), int(t[y])
            case (ap, dv, ml) if um and all_float(ap, dv, ml):
                res = self.resolution
                return int(t[p] / res), int(t[x] / res), int(t[y] / res)
            case _ if isinstance(t, np.ndarray):
                if um:
                    t = (t / self.resolution).astype(int)
                match t.ndim:
                    case 1:
                        return t[((p, x, y),)]
                    case 2:
                        return t[:, (p, x, y)]
                    case _:
                        raise ValueError(f'wrong dimension : {t.ndim}')
            case _:
                raise TypeError(repr(t))

    def offset(self, h: int, v: int) -> NDArray[np.int_]:
        """plane index offset according to horizontal difference *h* and vertical difference *v*.

        :param h: horizontal plane diff to the center. right side positive.
        :param v: vertical plane diff to the center. bottom side positive.
        :return: Array[int, H, W] array
        """
        x_frame = np.round(np.linspace(-h, h, self.width)).astype(int)
        y_frame = np.round(np.linspace(-v, v, self.height)).astype(int)
        return np.add.outer(y_frame, x_frame)

    def angle_offset(self, a: tuple[float, float, float]) -> tuple[int, int]:
        """plane index offset according to angle difference *a*.

        :param a: radian rotation of (ap, dv, ml)-axis.
        :return: tuple of (dw, dh)
        """
        raise RuntimeError()

    def offset_angle(self, dw: int, dh: int) -> tuple[float, float, float]:
        """plane index offset according to angle difference *a*.

        :param dw:
        :param dh:
        :return: radian rotation of (ap, dv, ml)-axis.
        """
        raise RuntimeError()

    def plane_at(self, c: int | COOR | NDArray[np.int_] | SlicePlane, um=False) -> SlicePlane:
        """

        :param c: plane index (int) or volume point (ap, dv, ml)
        :param um: does the unit of the values used in *c* are um?
        :return: correspond slice view.
        """
        dw = dh = 0
        match c:
            case SlicePlane(plane, ax, ay, dw, dh, _):
                pass
            case c if all_int(c):
                if um:
                    c = int(c / self.resolution)
                plane = int(c)
                ax = int(self.width // 2)
                ay = int(self.height // 2)
            case (ap, dv, ml):
                c = np.array(c)
                if um:
                    c = np.round(c / self.resolution).astype(int)
                plane, ax, ay = self.project(tuple(c))
            case _ if isinstance(c, np.ndarray):
                if um:
                    c = np.round(c / self.resolution).astype(int)
                plane, ax, ay = self.project(tuple(c))
            case _:
                raise TypeError()

        if plane < 0:
            raise ValueError('negative plane index')
        return SlicePlane(plane, ax, ay, dw, dh, self)


class CoronalView(SliceView):

    @property
    def n_plane(self) -> int:
        return self.n_ap

    @property
    def width(self) -> int:
        return self.n_ml

    @property
    def height(self) -> int:
        return self.n_dv

    @property
    def project_index(self) -> tuple[int, int, int]:
        return 0, 2, 1  # p=AP, x=ML, y=DV

    def angle_offset(self, a: tuple[float, float, float]) -> tuple[int, int]:
        rx, ry, rz = a
        dw = int(-self.width * math.tan(ry) / 2)
        dh = int(self.height * math.tan(rz) / 2)
        return dw, dh

    def offset_angle(self, dw: int, dh: int) -> tuple[float, float, float]:
        ry = math.atan(-dw * 2 / self.width)
        rz = math.atan(dh * 2 / self.height)
        return 0, ry, rz


class SagittalView(SliceView):
    @property
    def n_plane(self) -> int:
        return self.n_ml

    @property
    def width(self) -> int:
        return self.n_ap

    @property
    def height(self) -> int:
        return self.n_dv

    @property
    def project_index(self) -> tuple[int, int, int]:
        return 2, 0, 1  # p=ML, x=AP, y=DV

    def angle_offset(self, a: tuple[float, float, float]) -> tuple[int, int]:
        rx, ry, rz = a
        dw = int(-self.width * math.tan(ry) / 2)
        dh = int(self.height * math.tan(rx) / 2)
        return dw, dh

    def offset_angle(self, dw: int, dh: int) -> tuple[float, float, float]:
        ry = math.atan(-dw * 2 / self.width)
        rx = math.atan(dh * 2 / self.height)
        return rx, ry, 0


class TransverseView(SliceView):
    @property
    def n_plane(self) -> int:
        return self.n_dv

    @property
    def width(self) -> int:
        return self.n_ml

    @property
    def height(self) -> int:
        return self.n_ap

    @property
    def project_index(self) -> tuple[int, int, int]:
        return 1, 2, 0  # p=DV, x=ML, y=AP

    def angle_offset(self, a: tuple[float, float, float]) -> tuple[int, int]:
        rx, ry, rz = a
        dw = int(-self.width * math.tan(rx) / 2)
        dh = int(self.height * math.tan(rz) / 2)
        return dw, dh

    def offset_angle(self, dw: int, dh: int) -> tuple[float, float, float]:
        rx = math.atan(-dw * 2 / self.width)
        rz = math.atan(dh * 2 / self.height)
        return rx, 0, rz


class SlicePlane(NamedTuple):
    """Just a wrapper of SliceView that keep the information of volume point (ap, dv, ml) and rotate (dw, dh)."""

    plane: int  # anchor frame
    ax: int  # anchor x index
    ay: int  # anchor y index
    dw: int  # plane difference on left/right edge and the center
    dh: int  # plane difference on top/bottom edge and the center
    slice: SliceView

    @property
    def slice_name(self) -> SLICE:
        return self.slice.name

    @property
    def resolution(self) -> int:
        return self.slice.resolution

    @property
    def width(self) -> float:
        """width (um) in this view"""
        return self.slice.width_um

    @property
    def height(self) -> float:
        """height (um) in this view"""
        return self.slice.height_um

    @property
    def image(self) -> NDArray[np.uint]:
        return self.slice.plane(self.plane_offset)

    def image_of(self, image: NDArray[np.uint]) -> NDArray[np.uint]:
        return self.slice.plane(self.plane_offset, image)

    @property
    def plane_offset(self) -> NDArray[np.int_]:
        offset = self.slice.offset(self.dw, self.dh)
        return self.plane + offset - offset[self.ay, self.ax]

    @overload
    def plane_idx_at(self, x: int | float, y: int | float, *, um=False) -> int:
        pass

    @overload
    def plane_idx_at(self, x: NDArray[np], y: NDArray[np], *, um=False) -> NDArray[np.int_]:
        pass

    def plane_idx_at(self, x, y, *, um=False):
        if self.dw == self.dh == 0:
            return self.plane

        match (x, y):
            case (x, y) if all_float(x, y):
                if not um:
                    res = self.resolution
                    x *= res
                    y *= res
            case _:
                x, y = align_arr(x, y)
                if not um:
                    res = self.resolution
                    x = x * res
                    y = y * res

        cx = self.width / 2
        cy = self.height / 2

        dw = self.dw / cx * (x - cx)
        dh = self.dh / cy * (y - cy)
        dp = self.plane + dw + dh

        if isinstance(dp, np.ndarray):
            return dp.astype(int)
        else:
            return int(dp)

    @overload
    def coor_on(self, o: XY | tuple[float, float] = None, *, um=False) -> COOR:
        pass

    @overload
    def coor_on(self, o: tuple[NDArray, NDArray], *, um=False) -> tuple[NDArray[np.int_], NDArray[np.int_], NDArray[np.int_]]:
        pass

    @overload
    def coor_on(self, o: NDArray, *, um=False) -> NDArray[np.int_]:
        pass

    def coor_on(self, o=None, *, um=False):
        """

        :param o: tuple (x,y) or Array[int|float, [N,], (x, y)] position
        :return:  index (ap, dv, ml) or Array[int, N, (ap, dv, ml)]
        """
        if o is None:
            return self.slice.coor_on(self.plane_idx_at(self.ax, self.ay), (self.ax, self.ay), um=um)
        elif isinstance(o, tuple):
            return self.slice.coor_on(self.plane_idx_at(o[0], o[1], um=um), o, um=um)
        else:
            return self.slice.coor_on(self.plane_idx_at(o[:, 0], o[:, 1], um=um), o, um=um)

    def offset_angle(self) -> tuple[float, float, float]:
        return self.slice.offset_angle(self.dw, self.dh)

    def with_plane(self, plane: int) -> Self:
        return self._replace(plane=plane)

    def with_anchor(self, x: int, y: int) -> Self:
        plane = self.plane_idx_at(x, y)
        return self._replace(plane=plane, ax=x, ay=y)

    def with_offset(self, dw: int, dh: int) -> Self:
        return self._replace(dw=dw, dh=dh)

    def with_rotate(self, a: tuple[float, float]) -> Self:
        """plane index offset according to angle difference *a*.

        :param a: (vertical, horizontal)-axis radian rotation.
        :return: tuple of (dw, dh) or (rot, dw, dh)
        """
        rx, ry = a
        dw = int(-self.width * math.tan(rx) / 2)
        dh = int(self.height * math.tan(ry) / 2)
        return self.with_offset(dw, dh)
