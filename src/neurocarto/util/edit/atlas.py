from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from neurocarto.util import probe_coor
from neurocarto.util.util_blueprint import BlueprintFunctions
from neurocarto.util.utils import SPHINX_BUILD, doc_link
from neurocarto.views.base import ControllerView

if TYPE_CHECKING:
    from neurocarto.views.atlas import AtlasBrainView, Label
    from neurocarto.util.atlas_struct import Structure
elif SPHINX_BUILD:
    Label = 'neurocarto.views.atlas.Label'
    Structure = 'neurocarto.util.atlas_struct.Structure'

__all__ = [
    'atlas_get_region_name',
    'atlas_get_region',
    'atlas_get_slice',
    'atlas_set_slice',
    'atlas_add_label',
    'atlas_get_label',
    'atlas_focus_label',
    'atlas_del_label',
    'atlas_clear_labels',
    'atlas_set_transform',
    'atlas_set_anchor',
    'atlas_new_probe',
    'atlas_current_probe',
    'atlas_set_anchor_on_probe',
    'atlas_coor_electrode',
    'atlas_mask_region',
]


def _get_atlas(controller: ControllerView | AtlasBrainView) -> AtlasBrainView | None:
    from neurocarto.views.atlas import AtlasBrainView
    if isinstance(controller, ControllerView):
        return controller.get_view(AtlasBrainView)
    elif isinstance(controller, AtlasBrainView):
        return controller
    else:
        raise TypeError()


@doc_link(DOC=textwrap.dedent(BlueprintFunctions.atlas_get_region_name.__doc__))
def atlas_get_region_name(controller: ControllerView | AtlasBrainView, region: int | str) -> str | None:
    """
    {DOC}
    :see: {BlueprintFunctions#atlas_get_region_name()}
    """
    if (structure := atlas_get_region(controller, region)) is not None:
        return structure.acronym
    return None


def atlas_get_region(controller: ControllerView | AtlasBrainView, region: int | str) -> Structure | None:
    if (atlas := _get_atlas(controller)) is None:
        return None

    try:
        structure = atlas._structure
    except:
        from neurocarto.util.atlas_struct import Structures
        structure = Structures.of(atlas.brain)

    try:
        return structure[region]
    except KeyError:
        pass

    if isinstance(region, int):
        return None

    region = region.lower()
    for s in structure:
        if s.acronym.lower().startswith(region):
            return s
        if s.name.lower().startswith(region):
            return s
    return None


@doc_link(DOC=textwrap.dedent(BlueprintFunctions.atlas_get_slice.__doc__))
def atlas_get_slice(controller: ControllerView | AtlasBrainView, *, um=False) -> tuple[str | None, int | None]:
    """
    {DOC}
    :see: {BlueprintFunctions#atlas_get_slice()}
    """
    if (atlas := _get_atlas(controller)) is None:
        return None, None

    name = atlas.brain_view.name
    if (plane := atlas.brain_slice) is None:
        return name, None

    index = plane.plane
    if um:
        index = atlas.get_plane_offset(index)

    return name, index


@doc_link(DOC=textwrap.dedent(BlueprintFunctions.atlas_set_slice.__doc__))
def atlas_set_slice(controller: ControllerView | AtlasBrainView,
                    view: str = None,
                    plane: int = None, *, um=False):
    """
    {DOC}
    :see: {BlueprintFunctions#atlas_set_slice()}
    """
    if (atlas := _get_atlas(controller)) is not None:
        if view is not None:
            atlas.update_brain_view(view)

        if plane is not None:
            if um:
                plane = atlas.get_plane_index(plane)

            atlas.update_brain_slice(plane)


@doc_link(DOC=textwrap.dedent(BlueprintFunctions.atlas_add_label.__doc__))
def atlas_add_label(controller: ControllerView | AtlasBrainView,
                    text: str,
                    pos: tuple[float, float] | tuple[float, float, float], *,
                    origin: str = 'bregma', color: str = 'cyan', replace=True) -> Label | None:
    """
    {DOC}
    :see: {BlueprintFunctions#atlas_add_label()}
    """
    if (view := _get_atlas(controller)) is not None:
        return view.add_label(text, pos, origin=origin, color=color, replace=replace)
    return None


@doc_link(DOC=textwrap.dedent(BlueprintFunctions.atlas_get_label.__doc__))
def atlas_get_label(controller: ControllerView | AtlasBrainView,
                    index: int | str) -> Label | None:
    """
    {DOC}
    :see: {BlueprintFunctions#atlas_get_label()}
    """
    if (view := _get_atlas(controller)) is not None:
        if isinstance(index, str):
            if (index := view.index_label(index)) is None:
                return None

        try:
            return view.get_label(index)
        except IndexError:
            pass

    return None


@doc_link(DOC=textwrap.dedent(BlueprintFunctions.atlas_focus_label.__doc__))
def atlas_focus_label(controller: ControllerView | AtlasBrainView,
                      label: int | str | Label):
    """
    {DOC}
    :see: {BlueprintFunctions#atlas_focus_label()}
    """
    if (view := _get_atlas(controller)) is not None:
        view.focus_label(label)


@doc_link(DOC=textwrap.dedent(BlueprintFunctions.atlas_del_label.__doc__))
def atlas_del_label(controller: ControllerView | AtlasBrainView,
                    i: int | str | Label | list[int | str | Label]):
    """
    {DOC}
    :see: {BlueprintFunctions#atlas_del_label()}
    """
    if (view := _get_atlas(controller)) is not None:
        view.del_label(i)


@doc_link(DOC=textwrap.dedent(BlueprintFunctions.atlas_clear_labels.__doc__))
def atlas_clear_labels(controller: ControllerView | AtlasBrainView):
    """
    {DOC}
    :see: {BlueprintFunctions#atlas_clear_labels()}
    """
    if (view := _get_atlas(controller)) is not None:
        view.clear_labels()


@doc_link(DOC=textwrap.dedent(BlueprintFunctions.atlas_set_transform.__doc__))
def atlas_set_transform(controller: ControllerView | AtlasBrainView,
                        p: tuple[float, float] = None,
                        s: float | tuple[float, float] = None,
                        rt: float = None):
    """
    {DOC}
    :see: {BlueprintFunctions#atlas_set_transform()}
    """
    if (view := _get_atlas(controller)) is not None:
        view.update_boundary_transform(p=p, s=s, rt=rt)


@doc_link(DOC=textwrap.dedent(BlueprintFunctions.atlas_set_anchor.__doc__))
def atlas_set_anchor(controller: ControllerView | AtlasBrainView,
                     p: tuple[float, float],
                     a: tuple[float, float] = (0, 0)):
    """
    {DOC}
    :see: {BlueprintFunctions#atlas_set_anchor()}
    """
    if (view := _get_atlas(controller)) is not None:
        view.set_anchor_to(p, a)


@doc_link(DOC=textwrap.dedent(BlueprintFunctions.atlas_new_probe.__doc__))
def atlas_new_probe(controller: ControllerView | AtlasBrainView,
                    ap: float, dv: float, ml: float,
                    shank: int = 0,
                    rx: float = 0, ry: float = 0, rz: float = 0,
                    depth: float = 0,
                    ref: str = 'bregma') -> probe_coor.ProbeCoordinate | None:
    """
    {DOC}
    :see: {BlueprintFunctions#atlas_new_probe()}
    """
    if (view := _get_atlas(controller)) is None:
        return None

    from neurocarto.util.atlas_brain import REFERENCE
    origin = REFERENCE[ref][view.brain_view.brain.atlas_name]

    # get probe coordinate instance
    return probe_coor.ProbeCoordinate(ap, dv, ml, shank, rx, ry, rz, depth, origin)


@doc_link(DOC=textwrap.dedent(BlueprintFunctions.atlas_current_probe.__doc__))
def atlas_current_probe(bp: BlueprintFunctions,
                        controller: ControllerView | AtlasBrainView,
                        shank: int = 0,
                        ref: str = 'bregma') -> probe_coor.ProbeCoordinate | None:
    """
    {DOC}
    :see: {BlueprintFunctions#atlas_current_probe()}
    """
    if (view := _get_atlas(controller)) is None:
        return None

    from neurocarto.util.atlas_brain import REFERENCE

    brain = view.brain_slice
    try:
        origin = REFERENCE[ref][brain.slice.brain.atlas_name]
    except KeyError:
        return None

    transform = view.get_boundary_state()
    a, a_ = probe_coor.prepare_affine_matrix_both(**transform)

    p = _electrode_coor(bp, bp.s == shank)  # Array[float, (x,y,1), N']
    q = probe_coor.project_i2b(origin, brain, a_ @ p)  # Array[float, (ap,dv,ml), N']

    i = np.argmin(np.abs(q[1]))
    ap, dv, ml = q[:, i]
    depth = np.max(q[1])

    # set slice rotation
    rot = list(np.rad2deg(brain.offset_angle()))
    match brain.slice_name:
        case 'coronal':
            rot[0] = -transform['rt']
        case 'sagittal':
            rot[2] = transform['rt']
        case 'transverse':
            rot[1] = -transform['rt']
        case _:
            raise RuntimeError('un-reachable')

    return probe_coor.ProbeCoordinate(ap, 0, ml, shank, rot[0], rot[1], rot[2], depth, origin)


@doc_link(DOC=textwrap.dedent(BlueprintFunctions.atlas_set_anchor_on_probe.__doc__))
def atlas_set_anchor_on_probe(bp: BlueprintFunctions,
                              controller: ControllerView | AtlasBrainView,
                              coor: probe_coor.ProbeCoordinate):
    """
    {DOC}
    :see: {BlueprintFunctions#atlas_set_anchor_on_probe()}
    """
    if (view := _get_atlas(controller)) is None:
        return

    # set brain slice to corresponding plane
    brain_slice = probe_coor.get_plane_at(view.brain_view, coor)
    view.update_brain_slice(brain_slice, update_image=False)

    # set slice rotation
    match brain_slice.slice_name:
        case 'coronal':
            rot = -coor.rx
        case 'sagittal':
            rot = coor.rz
        case 'transverse':
            rot = -coor.ry
        case _:
            raise RuntimeError('un-reachable')

    view.update_boundary_transform(rt=rot)

    # another slice on to probe
    if bp.channelmap is None:
        ex = ey = 0
    else:
        electrode = _electrode_coor(bp, bp.s == coor.s)  # Array[um:float, (x, y, 1), N]

        from neurocarto.util.util_numpy import closest_point_index
        if (i := closest_point_index(electrode[1], coor.depth, bp.dy * 2)) is not None:
            ex = electrode[0, i]
        else:
            # cannot find nearest electrode position
            ex = 0

        ey = coor.depth

    cx = brain_slice.width / 2
    cy = brain_slice.height / 2
    ax = brain_slice.ax * brain_slice.resolution - cx
    ay = cy - brain_slice.ay * brain_slice.resolution
    view.set_anchor_to((ex, ey), (ax, ay))


@doc_link(DOC=textwrap.dedent(BlueprintFunctions.atlas_coor_electrode.__doc__))
def atlas_coor_electrode(bp: BlueprintFunctions,
                         controller: ControllerView | AtlasBrainView,
                         coor: probe_coor.ProbeCoordinate = None,
                         electrode: NDArray[np.int_] | NDArray[np.bool_] | NDArray[np.float_] = None,
                         bregma: str | None = 'bregma') -> NDArray[np.float_]:
    """
    {DOC}
    :see: {BlueprintFunctions#atlas_coor_electrode()}
    """
    if (view := _get_atlas(controller)) is None:
        raise RuntimeError('cannot determine current atlas')

    if coor is None:
        if (coor := atlas_current_probe(bp, view)) is None:
            raise RuntimeError('Cannot determine current probe coordinate.')

    if bregma is not None:
        from neurocarto.util.atlas_brain import REFERENCE
        origin = REFERENCE[bregma][view.brain.atlas_name]
    else:
        origin = None

    a, a_ = probe_coor.prepare_affine_matrix_both(**view.get_boundary_state())
    p = _electrode_coor(bp, electrode)  # Array[um:float, (x, y, 1), N] probe-origin
    q = probe_coor.project_i2b(origin, probe_coor.get_plane_at(view.brain_view, coor), a_ @ p)
    return q.T  # Array[um:float, N, (ap, dv, ml)] bregma-origin


@doc_link(DOC=textwrap.dedent(BlueprintFunctions.atlas_mask_region.__doc__))
def atlas_mask_region(bp: BlueprintFunctions,
                      controller: ControllerView | AtlasBrainView,
                      region: str,
                      coor: probe_coor.ProbeCoordinate = None,
                      electrode: NDArray[np.int_] | NDArray[np.bool_] | NDArray[np.float_] = None) -> NDArray[np.bool_]:
    """
    {DOC}
    :see: {BlueprintFunctions#atlas_mask_region()}
    """
    if (view := _get_atlas(controller)) is None:
        raise RuntimeError('cannot determine current atlas')

    if coor is None:
        if (coor := atlas_current_probe(bp, view)) is None:
            raise RuntimeError('Cannot determine current probe coordinate.')

    p = atlas_coor_electrode(bp, view, coor, electrode, bregma=None)  # Array[um:float, N, (ap, dv, ml)] atlas-origin
    q = (p / view.brain.resolution).astype(int)  # Array[index:int, N, (ap, dv, ml)]
    r = view.brain.annotation[tuple(q.T)]  # Array[annotation:int, N]

    try:
        structure = view._structure
    except:
        from neurocarto.util.atlas_struct import Structures
        structure = Structures.of(view.brain)

    ret = np.zeros_like(r, dtype=bool)
    if (target := atlas_get_region(view, region)) is None:
        return ret

    for sub in structure.iter_subregions(target):
        np.logical_or(ret, r == sub.id, out=ret)
    return ret


def _electrode_coor(bp: BlueprintFunctions,
                    electrode: NDArray[np.int_] | NDArray[np.bool_] | NDArray[np.float_] = None) -> NDArray[np.float_]:
    """

    :param bp:
    :param electrode: electrode index (Array[int, N]), mask (Array[bool, E]) or position (Array[um:float, N, (x, y)])
    :return: electrode position on probe Array[um:float, (x, y, 1), N]
    """
    if electrode is None:
        return np.vstack([
            bp.x,
            bp.y,
            np.ones((len(bp.x),))
        ])  # Array[float, (x,y,1), N]
    elif electrode.ndim == 1:
        return np.vstack([
            (x := bp.x[electrode]),
            bp.y[electrode],
            np.ones((len(x),))
        ])
    elif electrode.ndim == 2:
        return np.vstack([
            electrode[:, 0],
            electrode[:, 1],
            np.ones((len(electrode),))
        ])
    else:
        raise ValueError('wrong electrode position dimension')
