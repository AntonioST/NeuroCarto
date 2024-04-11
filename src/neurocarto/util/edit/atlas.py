from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING

import numpy as np

from neurocarto.util import probe_coor
from neurocarto.util.util_blueprint import BlueprintFunctions
from neurocarto.util.utils import SPHINX_BUILD, doc_link
from neurocarto.views.base import ControllerView

if TYPE_CHECKING:
    from neurocarto.views.atlas import AtlasBrainView, Label
elif SPHINX_BUILD:
    Label = 'neurocarto.views.atlas.Label'

__all__ = [
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
]


@doc_link(DOC=textwrap.dedent(BlueprintFunctions.atlas_get_slice.__doc__))
def atlas_get_slice(controller: ControllerView, *, um=False) -> tuple[str | None, int | None]:
    """
    {DOC}
    :see: {BlueprintFunctions#atlas_get_slice()}
    """
    atlas: AtlasBrainView
    if (atlas := controller.get_view('AtlasBrainView')) is None:  # type: ignore[assignment]
        return None, None

    name = atlas.brain_view.name
    if (plane := atlas.brain_slice) is None:
        return name, None

    index = plane.plane
    if um:
        index = atlas.get_plane_offset(index)

    return name, index


@doc_link(DOC=textwrap.dedent(BlueprintFunctions.atlas_set_slice.__doc__))
def atlas_set_slice(controller: ControllerView,
                    view: str = None,
                    plane: int = None, *, um=False):
    """
    {DOC}
    :see: {BlueprintFunctions#atlas_set_slice()}
    """
    atlas: AtlasBrainView
    if (atlas := controller.get_view('AtlasBrainView')) is not None:  # type: ignore[assignment]
        if view is not None:
            atlas.update_brain_view(view)

        if plane is not None:
            if um:
                plane = atlas.get_plane_index(plane)

            atlas.update_brain_slice(plane)


@doc_link(DOC=textwrap.dedent(BlueprintFunctions.atlas_add_label.__doc__))
def atlas_add_label(controller: ControllerView, text: str,
                    pos: tuple[float, float] | tuple[float, float, float], *,
                    origin: str = 'bregma', color: str = 'cyan', replace=True) -> Label | None:
    """
    {DOC}
    :see: {BlueprintFunctions#atlas_add_label()}
    """
    view: AtlasBrainView
    if (view := controller.get_view('AtlasBrainView')) is not None:  # type: ignore[assignment]
        return view.add_label(text, pos, origin=origin, color=color, replace=replace)
    return None


@doc_link(DOC=textwrap.dedent(BlueprintFunctions.atlas_get_label.__doc__))
def atlas_get_label(controller: ControllerView, index: int | str) -> Label | None:
    """
    {DOC}
    :see: {BlueprintFunctions#atlas_get_label()}
    """
    view: AtlasBrainView
    if (view := controller.get_view('AtlasBrainView')) is not None:  # type: ignore[assignment]
        if isinstance(index, str):
            if (index := view.index_label(index)) is None:
                return None

        try:
            return view.get_label(index)
        except IndexError:
            pass

    return None


@doc_link(DOC=textwrap.dedent(BlueprintFunctions.atlas_focus_label.__doc__))
def atlas_focus_label(controller: ControllerView, label: int | str | Label):
    """
    {DOC}
    :see: {BlueprintFunctions#atlas_focus_label()}
    """
    view: AtlasBrainView
    if (view := controller.get_view('AtlasBrainView')) is not None:  # type: ignore[assignment]
        view.focus_label(label)


@doc_link(DOC=textwrap.dedent(BlueprintFunctions.atlas_del_label.__doc__))
def atlas_del_label(controller: ControllerView, i: int | str | Label | list[int | str | Label]):
    """
    {DOC}
    :see: {BlueprintFunctions#atlas_del_label()}
    """
    view: AtlasBrainView
    if (view := controller.get_view('AtlasBrainView')) is not None:  # type: ignore[assignment]
        view.del_label(i)


@doc_link(DOC=textwrap.dedent(BlueprintFunctions.atlas_clear_labels.__doc__))
def atlas_clear_labels(controller: ControllerView):
    """
    {DOC}
    :see: {BlueprintFunctions#atlas_clear_labels()}
    """
    view: AtlasBrainView
    if (view := controller.get_view('AtlasBrainView')) is not None:  # type: ignore[assignment]
        view.clear_labels()


@doc_link(DOC=textwrap.dedent(BlueprintFunctions.atlas_set_transform.__doc__))
def atlas_set_transform(controller: ControllerView,
                        p: tuple[float, float] = None,
                        s: float | tuple[float, float] = None,
                        rt: float = None):
    """
    {DOC}
    :see: {BlueprintFunctions#atlas_set_transform()}
    """
    view: AtlasBrainView
    if (view := controller.get_view('AtlasBrainView')) is not None:  # type: ignore[assignment]
        view.update_boundary_transform(p=p, s=s, rt=rt)


@doc_link(DOC=textwrap.dedent(BlueprintFunctions.atlas_set_anchor.__doc__))
def atlas_set_anchor(controller: ControllerView,
                     p: tuple[float, float],
                     a: tuple[float, float] = (0, 0)):
    """
    {DOC}
    :see: {BlueprintFunctions#atlas_set_anchor()}
    """
    view: AtlasBrainView
    if (view := controller.get_view('AtlasBrainView')) is not None:  # type: ignore[assignment]
        view.set_anchor_to(p, a)


@doc_link(DOC=textwrap.dedent(BlueprintFunctions.atlas_new_probe.__doc__))
def atlas_new_probe(controller: ControllerView,
                    ap: float, dv: float, ml: float,
                    shank: int = 0,
                    rx: float = 0, ry: float = 0, rz: float = 0,
                    depth: float = 0,
                    ref: str = 'bregma') -> probe_coor.ProbeCoordinate | None:
    """
    {DOC}
    :see: {BlueprintFunctions#atlas_new_probe()}
    """
    view: AtlasBrainView
    if (view := controller.get_view('AtlasBrainView')) is None:  # type: ignore[assignment]
        return None

    # get probe coordinate instance
    name = view.brain_view.brain.atlas_name
    return probe_coor.ProbeCoordinate.from_bregma(name, ap, dv, ml, s=shank, rx=rx, ry=ry, rz=rz, depth=depth, ref=ref)


@doc_link(DOC=textwrap.dedent(BlueprintFunctions.atlas_current_probe.__doc__))
def atlas_current_probe(bp: BlueprintFunctions,
                        controller: ControllerView,
                        shank: int = 0,
                        ref: str = 'bregma') -> probe_coor.ProbeCoordinate | None:
    """
    {DOC}
    :see: {BlueprintFunctions#atlas_current_probe()}
    """
    view: AtlasBrainView
    if (view := controller.get_view('AtlasBrainView')) is None:  # type: ignore[assignment]
        return None

    from neurocarto.util.atlas_brain import REFERENCE

    brain = view.brain_slice
    try:
        origin = REFERENCE[ref][brain.slice.brain.atlas_name]
    except KeyError:
        return None

    transform = view.get_boundary_state()
    a, a_ = probe_coor.prepare_affine_matrix_both(**transform)

    s = bp.s == shank
    p = np.vstack([
        bp.x[s],
        bp.y[s],
        np.ones((np.count_nonzero(s),)),
    ])  # Array[float, (x,y,1), N']
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

    return probe_coor.ProbeCoordinate(ap, 0, ml, shank, rot[0], rot[1], rot[2], depth)


@doc_link(DOC=textwrap.dedent(BlueprintFunctions.atlas_set_anchor_on_probe.__doc__))
def atlas_set_anchor_on_probe(bp: BlueprintFunctions,
                              controller: ControllerView,
                              coor: probe_coor.ProbeCoordinate):
    """
    {DOC}
    :see: {BlueprintFunctions#atlas_set_anchor_on_probe()}
    """
    view: AtlasBrainView
    if (view := controller.get_view('AtlasBrainView')) is None:  # type: ignore[assignment]
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
        electrode_s = bp.s == coor.s
        electrode_x = bp.x[electrode_s]  # Array[um:float, N]
        electrode_y = bp.y[electrode_s]  # Array[um:float, N]

        from neurocarto.util.util_numpy import closest_point_index
        if (i := closest_point_index(electrode_y, coor.depth, bp.dy * 2)) is not None:
            ex = electrode_x[i]
        else:
            # cannot find nearest electrode position
            ex = 0

        ey = coor.depth

    cx = brain_slice.width / 2
    cy = brain_slice.height / 2
    ax = brain_slice.ax * brain_slice.resolution - cx
    ay = cy - brain_slice.ay * brain_slice.resolution
    view.set_anchor_to((ex, ey), (ax, ay))
