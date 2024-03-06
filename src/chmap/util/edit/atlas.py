from __future__ import annotations

from typing import TYPE_CHECKING

from chmap.util import probe_coor
from chmap.util.util_blueprint import BlueprintFunctions
from chmap.util.utils import SPHINX_BUILD, doc_link
from chmap.views.base import ControllerView

if TYPE_CHECKING:
    from chmap.views.atlas import AtlasBrainView, Label
elif SPHINX_BUILD:
    BoundView = 'chmap.views.base.BoundView'
    AtlasBrainView = 'chmap.views.atlas.AtlasBrainView'
    Label = 'chmap.views.atlas.Label'

__all__ = [
    'atlas_get_slice',
    'atlas_set_slice',
    'atlas_add_label',
    'atlas_focus_label',
    'atlas_del_label',
    'atlas_clear_labels',
    'atlas_set_transform',
    'atlas_set_anchor',
    'atlas_new_probe',
    'atlas_set_anchor_on_probe',
]


@doc_link()
def atlas_get_slice(controller: ControllerView, *, um=False) -> tuple[str | None, int | None]:
    """

    :param controller:
    :param um: is plane index in return um? If so, then use bregma as origin.
    :return: tuple of (projection name, plane index)
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


@doc_link()
def atlas_set_slice(controller: ControllerView,
                    view: str = None,
                    plane: int = None, *, um=False):
    """

    :param controller:
    :param view: 'coronal', 'sagittal', or 'transverse'
    :param plane: plane index
    :param um: is *plane* um? If so, then use bregma as origin.
    :see: {AtlasBrainView#update_brain_view()}, {AtlasBrainView#update_brain_slice()}
    """
    atlas: AtlasBrainView
    if (atlas := controller.get_view('AtlasBrainView')) is not None:  # type: ignore[assignment]
        if view is not None:
            atlas.update_brain_view(view)

        if plane is not None:
            if um:
                plane = atlas.get_plane_index(plane)

            atlas.update_brain_slice(plane)


@doc_link()
def atlas_add_label(controller: ControllerView, text: str,
                    pos: tuple[float, float] | tuple[float, float, float], *,
                    origin: str = 'bregma', color: str = 'cyan', replace=True) -> Label | None:
    """{AtlasBrainView#add_label()}"""
    view: AtlasBrainView
    if (view := controller.get_view('AtlasBrainView')) is not None:  # type: ignore[assignment]
        return view.add_label(text, pos, origin=origin, color=color, replace=replace)
    return None


@doc_link()
def atlas_focus_label(controller: ControllerView, label: int | str | Label):
    """{AtlasBrainView#focus_label()}"""
    view: AtlasBrainView
    if (view := controller.get_view('AtlasBrainView')) is not None:  # type: ignore[assignment]
        view.focus_label(label)


@doc_link()
def atlas_del_label(controller: ControllerView, i: int | str | list[int | str]):
    """{AtlasBrainView#del_label()}"""
    view: AtlasBrainView
    if (view := controller.get_view('AtlasBrainView')) is None:  # type: ignore[assignment]
        return

    match i:
        case int(i):
            ii = [i]
        case list(tmp):
            ii = []
            for it in tmp:
                if isinstance(it, int):
                    ii.append(it)
                elif isinstance(it, str):
                    if (it := view.index_label(it)) is not None:
                        ii.append(it)
                else:
                    raise TypeError()
        case str(text):
            if (it := view.index_label(text)) is None:
                return

            ii = [it]
        case _:
            raise TypeError()

    view.del_label(ii)


@doc_link()
def atlas_clear_labels(controller: ControllerView):
    """{AtlasBrainView#clear_labels()}"""
    view: AtlasBrainView
    if (view := controller.get_view('AtlasBrainView')) is not None:  # type: ignore[assignment]
        view.clear_labels()


@doc_link()
def atlas_set_transform(controller: ControllerView,
                        p: tuple[float, float] = None,
                        s: float | tuple[float, float] = None,
                        rt: float = None):
    """{BoundView#update_boundary_transform()}"""
    view: AtlasBrainView
    if (view := controller.get_view('AtlasBrainView')) is not None:  # type: ignore[assignment]
        view.update_boundary_transform(p=p, s=s, rt=rt)


@doc_link()
def atlas_set_anchor(controller: ControllerView,
                     p: tuple[float, float],
                     a: tuple[float, float] = (0, 0)):
    """{BoundView#set_anchor_to()}"""
    view: AtlasBrainView
    if (view := controller.get_view('AtlasBrainView')) is not None:  # type: ignore[assignment]
        view.set_anchor_to(p, a)


@doc_link()
def atlas_new_probe(controller: ControllerView,
                    ap: float, dv: float, ml: float,
                    shank: int = 0,
                    rx: float = 0, ry: float = 0, rz: float = 0,
                    depth: float = 0,
                    ref: str = 'bregma') -> probe_coor.ProbeCoordinate | None:
    """

    :param controller:
    :param ap: um
    :param dv: dv
    :param ml: ml
    :param shank: shank index
    :param rx: ap rotate
    :param ry: dv rotate
    :param rz: ml rotate
    :param depth: insert depth
    :param ref: reference origin.
    :return: a probe coordinate. ``None`` if origin not set.
    """
    view: AtlasBrainView
    if (view := controller.get_view('AtlasBrainView')) is None:  # type: ignore[assignment]
        return None

    # get probe coordinate instance
    name = view.brain_view.brain.atlas_name
    return probe_coor.ProbeCoordinate.from_bregma(name, ap, dv, ml, s=shank, rx=rx, ry=ry, rz=rz, depth=depth, ref=ref)


@doc_link()
def atlas_set_anchor_on_probe(bp: BlueprintFunctions,
                              controller: ControllerView,
                              coor: probe_coor.ProbeCoordinate):
    """

    :param bp:
    :param controller:
    :param coor:
    :return:
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

        from chmap.util.util_numpy import closest_point_index
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
