from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from neurocarto.probe_npx import NpxProbeDesp, utils
from neurocarto.util.edit.checking import use_probe, use_view
from neurocarto.util.util_blueprint import BlueprintFunctions

if TYPE_CHECKING:
    from neurocarto.util.probe_coor import ProbeCoordinate
    from neurocarto.views.atlas import Label


@use_probe(NpxProbeDesp, 24)
def npx24_single_shank(bp: BlueprintFunctions, shank: int = 0, row: int = 0):
    """
    Make a block channelmap for 4-shank Neuropixels probe.

    :param bp:
    :param shank: (int=0) on which shank.
    :param row: (int=0) start row in um.
    """
    bp.set_channelmap(utils.npx24_single_shank(shank, row, um=True))


@use_probe(NpxProbeDesp, 24)
def npx24_stripe(bp: BlueprintFunctions, row: int = 0):
    """
    Make a block channelmap for 4-shank Neuropixels probe.

    :param bp:
    :param row: (int=0) start row in um.
    """
    bp.set_channelmap(utils.npx24_stripe(row, um=True))


@use_probe(NpxProbeDesp, 24)
def npx24_half_density(bp: BlueprintFunctions, shank: int | list[int] | Literal['selected'] = 0, row: int = 0):
    """
    Make a channelmap for 4-shank Neuropixels probe that uniformly distributes channels in *half* density.

    :param bp:
    :param shank: (int|[int, int]|'selected'=0) on which shank/s.
        Use 'select' for selected electrodes.
    :param row: (int=0) start row in um.
    """
    match shank:
        case 'selected':
            if len(electrodes := bp.captured_electrodes(all=True)) < 4:
                bp.log_message('need capture more electrodes')
                return

            # npx 24 arrange electrode in S-R-C ordering
            match row % 2:
                case 0:
                    mask = (electrodes % 4 == 0) | (electrodes % 4 == 3)
                case 1:
                    mask = (electrodes % 4 == 1) | (electrodes % 4 == 2)
                case _:
                    raise RuntimeError('unreachable')

            bp.add_electrodes(electrodes[mask])
            bp.clear_capture_electrode()

        case int() | [int(), int()] | (int(), int()):
            bp.set_channelmap(utils.npx24_half_density(shank, row, um=True))
        case _:
            raise ValueError()


@use_probe(NpxProbeDesp, 24)
def npx24_quarter_density(bp: BlueprintFunctions, shank: int | list[int] | Literal['selected'] | None = None, row: int = 0):
    """
    Make a channelmap for 4-shank Neuropixels probe that uniformly distributes channels in *quarter* density.

    :param bp:
    :param shank: (int|[int, int]|'selected'=None) on which shank/s.
        Use 'select' for selected electrodes.
        Use ``None`` for four shanks.
    :param row: (int=0) start row in um.
    """
    match shank:
        case 'selected':
            if len(electrodes := bp.captured_electrodes(all=True)) < 8:
                bp.log_message('need capture more electrodes')
                return

            # npx 24 arrange electrode in S-R-C ordering
            match row % 4:
                case 0:
                    mask = (electrodes % 8 == 0) | (electrodes % 8 == 5)
                case 1:
                    mask = (electrodes % 8 == 1) | (electrodes % 8 == 4)
                case 2:
                    mask = (electrodes % 8 == 2) | (electrodes % 8 == 7)
                case 3:
                    mask = (electrodes % 8 == 3) | (electrodes % 8 == 6)
                case _:
                    raise RuntimeError('unreachable')

            bp.add_electrodes(electrodes[mask])
            bp.clear_capture_electrode()

        case None | int() | [int(), int()] | (int(), int()):
            bp.set_channelmap(utils.npx24_quarter_density(shank, row, um=True))
        case _:
            raise ValueError()


@use_probe(NpxProbeDesp, 24)
def npx24_one_eighth_density(bp: BlueprintFunctions, row: int = 0):
    """
    Make a channelmap for 4-shank Neuropixels probe that uniformly distributes channels in *one-eighth* density.

    :param bp:
    :param row: (int=0) start row in um.
    """
    bp.set_channelmap(utils.npx24_one_eighth_density(row, um=True))


@use_probe()
def move_blueprint(bp: BlueprintFunctions, y: int, shank: list[int] = None, update=False):
    """
    Move blueprint upward or downward.

    :param bp:
    :param y: (int) um
    :param shank: (list[int]=None) only particular shanks.
    :param update: (bool) update channelmap to follow the blueprint change.
    """
    if shank is None:
        mask = None
    else:
        mask = np.zeros_like(bp.s, dtype=bool)
        for s in shank:
            np.logical_or(mask, bp.s == s, out=mask)

    bp.set_blueprint(bp.move(bp.blueprint(), ty=y, mask=mask, axis=0, init=bp.CATE_UNSET))
    if update:
        bp.refresh_selection()


@use_probe()
def exchange_shank(bp: BlueprintFunctions, shank: list[int], update=False):
    """
    Move blueprint between shanks.

    *Note*: Each shank requires the same electrode number, and electrodes are ordered consistently.

    :param bp:
    :param shank: (list[int]): For N shank probe, it is an N-length list.
        For example, ``[3, 2, 1, 0]`` gives a reverse-shank-ordered blueprint.
    :param update: (bool) update channelmap to follow the blueprint change.
    """
    ns = np.max(bp.s) + 1
    if len(shank) != np.max(ns):
        raise RuntimeError(f'not a {ns}-length list: {shank}')

    p = bp.blueprint()
    q = bp.new_blueprint()
    for i, s in enumerate(shank):
        q[bp.s == i] = p[bp.s == s]

    bp.set_blueprint(q)
    if update:
        bp.refresh_selection()


@use_probe()
def load_blueprint(bp: BlueprintFunctions, filename: str):
    """
    Load a blueprint file.

    :param bp:
    :param filename: (str) a numpy file '*.blueprint.npy'.
    """
    bp.set_blueprint(bp.probe.load_blueprint(filename, bp.channelmap))


@use_probe(NpxProbeDesp)
def optimize_channelmap(bp: BlueprintFunctions, sample_times: int = 100, *,
                        single_process=False,
                        n_worker: int = 1,
                        **kwargs):
    """
    Sample and find the optimized channelmap that has maxima channel efficiency.

    :param bp:
    :param sample_times: (int=100)
    :param single_process: (bool) debug use parameter
    :param kwargs: selector parameters
    """
    blueprint = bp.blueprint()
    if np.all(blueprint == bp.CATE_UNSET):
        bp.log_message('empty blueprint')
        return

    from .optimize import optimize_channelmap as _optimize_channelmap

    chmap = None
    aeff = 0
    ceff = 0

    if single_process:
        chmap, aeff, ceff = _optimize_channelmap(bp, bp.channelmap, blueprint,
                                                 sample_times=sample_times,
                                                 **kwargs)
    else:
        import threading

        def target():
            nonlocal chmap, aeff, ceff
            chmap, aeff, ceff = _optimize_channelmap(bp, bp.channelmap, blueprint,
                                                     sample_times=sample_times,
                                                     n_worker=n_worker,
                                                     **kwargs)

        thread = threading.Thread(target=target, daemon=True)
        thread.start()

        while thread.is_alive():
            yield 1
        thread.join()

    bp.set_status_line(f'finished. got max(Ceff)={100 * ceff:.2f}% and Aeff={100 * aeff:.2f}')
    bp.set_channelmap(chmap)


@use_view('AtlasBrainView')
def atlas_label(bp: BlueprintFunctions, *args, color='cyan'):
    """
    Set labels on atlas brain image.

    commands:
    * clear : clear labels
    * probe[,S][,TEXT] : current coordinate of probe insert point.
    * delete,I,... :  delete labels
    * AP,DV,ML,TEXT : add text on (ap,dv,ml) and use bregma as origin.
    * X,Y,TEXT[,REF] : add text on (x,y[,z]) and use reference as origin ('probe' as default).
    * move,I,[AP,DV,ML|X,Y] : move label to new place (reference not change)
    * text,I,TEXT : change label content
    * color,I,... : change label color

    reference:
    * 'bregma' : origin at bregma of the brain, use (ap,dv,ml) mm.
    * 'probe' : origin at shank-0-probe-tip, use (x,y) um.
    * 'image' : origin at center of the image, use (x,y) um.

    :param bp:
    :param args: command args
    :param color: label color
    """
    match args:
        case ():
            return

        case ('clear', ):
            bp.atlas_clear_labels()
        case ('clear', _):
            raise RuntimeError(f'unknown clear args : {args}')

        case ('probe', ):
            _atlas_label_probe(bp, bp.atlas_current_probe(0), color=color)
        case ('probe', int(shank)):
            _atlas_label_probe(bp, bp.atlas_current_probe(shank), color=color)
        case ('probe', str(text)):
            _atlas_label_probe(bp, bp.atlas_current_probe(0), text, color=color)
        case ('probe', int(shank), str(text)):
            _atlas_label_probe(bp, bp.atlas_current_probe(shank), text, color=color)

        case ('delete' | 'move' | 'text', ):
            bp.log_message('missing label index or text')

        case ('delete', arg):
            bp.atlas_del_label(arg)
        case ('delete', *args):
            bp.atlas_del_label(list(args))

        case ('move', _):
            bp.log_message('missing label new position')
        case ('move', index, *coor):
            if (label := bp.atlas_get_label(index)) is not None:
                _atlas_label_move(bp, label, coor)
            else:
                bp.log_message('label not found')

        case ('text', _):
            bp.log_message('missing label new content')
        case ('text', index, text):
            if (label := bp.atlas_get_label(index)) is not None:
                bp.atlas_del_label(label)
                bp.atlas_add_label(str(text), label.pos, origin=label.origin, color=label.color)
            else:
                bp.log_message('label not found')

        case ('color', ):
            _atlas_label_color(bp, color, None)
        case ('color', *index):
            _atlas_label_color(bp, color, index)

        case (int(ap) | float(ap), int(dv) | float(dv), int(ml) | float(ml), text):
            bp.atlas_add_label(str(text), (ap, dv, ml), origin='bregma', color=color)

        case (int(x) | float(x), int(y) | float(y), text):
            bp.atlas_add_label(str(text), (x, y), origin='probe', color=color)

        case (int(x) | float(x), int(y) | float(y), text, str(ref)):
            bp.atlas_add_label(str(text), (x, y), origin=ref, color=color)

        case (command, *_):
            raise RuntimeError(f'unknown command : {command}')


def _atlas_label_probe(bp: BlueprintFunctions, coor: ProbeCoordinate, text: str = 'probe', color: str = 'red'):
    x = round(coor.x / 1000, 1)
    y = round(coor.y / 1000, 1)
    z = round(coor.z / 1000, 1)
    bp.atlas_add_label(text, (x, y, z), origin='bregma', color=color)


def _atlas_label_move(bp: BlueprintFunctions, label: Label, coor: tuple[float, ...]):
    match coor:
        case (int() | float(), int() | float(), int() | float()) as pos if label.origin == 'bregma':
            bp.atlas_del_label(label)
            bp.atlas_add_label(label.text, pos, origin='bregma', color=label.color)
        case (int() | float(), int() | float()) as pos:
            bp.atlas_del_label(label)
            bp.atlas_add_label(label.text, pos, origin=label.origin, color=label.color)


def _atlas_label_color(bp: BlueprintFunctions, color: str, index: tuple[int, ...] = None):
    labels = []

    if index is None:
        _index = 0

        while True:
            if (label := bp.atlas_get_label(_index)) is not None:
                if label.color != color:
                    labels.append(label)
                _index += 1
    else:

        for _index in index:
            if (label := bp.atlas_get_label(_index)) is not None and label.color != color:
                labels.append(label)

    for label in labels:
        bp.atlas_del_label(label)
        bp.atlas_add_label(label.text, label.pos, origin=label.origin, color=color)


@use_view('AtlasBrainView')
def adjust_atlas_mouse_brain_to_probe_coordinate(bp: BlueprintFunctions,
                                                 ap: float = None, ml: float = None, dv: float = 0,
                                                 shank: int = 0,
                                                 rx: float = 0, ry: float = 0, rz: float = 0,
                                                 depth: float = 0,
                                                 ref: str = 'bregma',
                                                 label: str = None,
                                                 color: str = 'cyan'):
    """
    Adjust atlas mouse brain image to corresponding probe coordinate.

    If *ap* and *ml* are omitted, calculate current probe coordinate based on *shank* and show
    the result in input area.

    :param bp:
    :param ap: (mm:float) ap from the *ref*.
    :param ml: (mm:float) ml from the *ref*.
    :param dv: (mm:float=0) dv from the *ref*.
    :param shank: (int=0) s-th shank coordinate
    :param rx: (degree:float=0) ap rotating.
    :param ry: (degree:float=0) dv rotating.
    :param rz: (degree:float=0) ml rotating.
    :param depth: (mm:float=0) probe insert depth.
    :param ref: (str in ['bregma'] = 'bregma') origin reference.
    :param label: label text
    :param color: label color
    """
    if ap is None and ml is None:
        if (coor := bp.atlas_current_probe(shank, ref)) is None:
            bp.log_message('fail to figure current probe coordinate')
            return

        assert coor.bregma is not None
        bp.set_script_input(
            None,
            f'{coor.x / 1000:.1f}',
            f'{coor.z / 1000:.1f}',
            f'{coor.y / 1000:.1f}' if coor.y != 0 else None,
            f'shank={coor.s}',
            f'rx={coor.rx:.0f}' if coor.rx != 0 else None,
            f'ry={coor.ry:.0f}' if coor.ry != 0 else None,
            f'rz={coor.rz:.0f}' if coor.rz != 0 else None,
            f'depth={coor.depth / 1000:.1f}',
            f'ref={ref}' if ref != 'bregma' else None
        )
    else:
        coor = bp.atlas_new_probe(ap * 1000, dv * 1000, ml * 1000, shank=shank, rx=rx, ry=ry, rz=rz, depth=depth * 1000, ref=ref)
        if coor is None:
            return

        bp.atlas_set_anchor_on_probe(coor)

        if label is not None:
            bp.atlas_add_label(label, (ap, dv, ml), origin=ref, color=color)


@use_view('AtlasBrainView')
def highlight_electrode_inside_region(bp: BlueprintFunctions,
                                      region: str,
                                      mode: Literal['replace', 'append', 'exclude'] = 'replace'):
    """
    Capture electrodes inside a region.

    :param bp:
    :param region: (str) region ID, acronym or its partial description
    :param mode: (one of 'replace', 'append' or 'exclude') capture mode
    """
    if (region := bp.atlas_get_region_name(name := region)) is None:
        bp.log_message(f'region "{name}" not found')
        return

    try:
        mask = bp.atlas_mask_region(region)
    except BaseException as e:
        bp.log_message(repr(e))
        return

    if np.count_nonzero(mask) == 0:
        bp.log_message(f'no electrode inside "{region}"')
    else:
        bp.capture_electrode(mask, mode=mode)


@use_probe(NpxProbeDesp, create=False)
def blueprint_simple_init_script_from_activity_data_with_a_threshold(bp: BlueprintFunctions, filename: str, threshold: float):
    """
    Initial a blueprint based on the experimental activity data with a given threshold,
    which follows:

    * set NaN area as excluded zone.
    * set full-density zone where corresponding activity over the threshold.
    * make the full-density zone into rectangle by filling the gaps.
    * extend the full-density zone with half-density zone.

    :param bp:
    :param filename: a numpy filepath, which shape Array[int, N, (shank, col, row, state, value)]
    :param threshold: (float) activities threshold to set FULL category
    """
    bp.log_message(f'{filename=}', f'{threshold=}')
    data = bp.load_data(filename)
    data[data == 0] = np.nan

    F = NpxProbeDesp.CATE_FULL
    H = NpxProbeDesp.CATE_HALF
    Q = NpxProbeDesp.CATE_QUARTER
    L = NpxProbeDesp.CATE_LOW
    X = NpxProbeDesp.CATE_EXCLUDED

    bp.log_message(f'min={np.nanmin(data)}, max={np.nanmax(data)}')

    data = bp.interpolate_nan(data)
    bp.draw(data)

    bp.clear_blueprint()
    bp[np.isnan(data)] = X
    bp.reduce(X, 20, bi=False)
    bp.fill(X, gap=None, threshold=10, unset=True)

    bp[data >= threshold] = F

    bp.fill(F, gap=None)
    bp.fill(F, threshold=10, unset=True)

    bp.extend(F, 2, threshold=(0, 100))
    bp.extend(F, 10, H)
