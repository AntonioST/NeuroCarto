import sys
from collections.abc import Iterator
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from neurocarto.probe_npx.npx import ProbeType, Electrode, ChannelMap, PROBE_TYPE_NP24, ChannelHasUsedError
from neurocarto.util.utils import as_set, doc_link

__all__ = [
    'clone', 'clear', 'set_electrodes',
    'npx24_single_shank',
    'npx24_stripe',
    'npx24_half_density',
    'npx24_quarter_density',
    'npx24_one_eighth_density',
    'print_probe'
]


def clear(chmap: ChannelMap) -> ChannelMap:
    del chmap.channels[:]
    return chmap


def clone(chmap: ChannelMap) -> ChannelMap:
    return ChannelMap(chmap)


ITER = int | slice | range | list[int] | NDArray[np.int_]


# ITER = _ITER | tuple[_ITER, ...]


def iter_electrodes(probe_type: ProbeType,
                    shank: ITER = None,
                    column: ITER = None,
                    row: ITER = None,
                    block: ITER = None,
                    bank: ITER = None) -> Iterator[tuple[int, int, int]]:
    s = as_set(shank, probe_type.n_shank)
    c = as_set(column, probe_type.n_col_shank)

    match (bank, block):
        case (None, None):
            for ss in s:
                for rr in as_set(row, probe_type.n_row_shank):
                    for cc in c:
                        yield ss, cc, rr

        case (None, _):
            rk = probe_type.n_electrode_block // probe_type.n_col_shank
            r = as_set(row, rk)
            k = as_set(block, probe_type.n_block)

            for ss in s:
                for kk in k:
                    for rr in r:
                        for cc in c:
                            yield ss, cc, rr + kk * rk

        case (_, None):
            rb = probe_type.n_channels // probe_type.n_col_shank
            r = as_set(row, rb)
            b = as_set(bank, probe_type.n_bank)

            for ss in s:
                for bb in b:
                    for rr in r:
                        for cc in c:
                            yield ss, cc, rr + bb * rb

        case (_, _):
            rb = probe_type.n_channels // probe_type.n_col_shank
            rk = probe_type.n_electrode_block // probe_type.n_col_shank

            r = as_set(row, rk)
            k = as_set(block, probe_type.n_block_bank)
            b = as_set(bank, probe_type.n_bank)

            for ss in s:
                for bb in b:
                    for kk in k:
                        for rr in r:
                            for cc in c:
                                yield ss, cc, rr + bb * rb + kk * rk


def set_electrodes(chmap: ChannelMap,
                   shank: ITER = None,
                   column: ITER = None,
                   row: ITER = None,
                   block: ITER = None,
                   bank: ITER = None, *,
                   overwrite=False) -> list[Electrode]:
    """
    add electrodes.

    :param chmap: a channelmap instance
    :param shank: shank
    :param column: column
    :param row: row. It ranges from 0 to total number of rows on one shank.
        If *block* present, it ranges from 0 to total number of rows in a block.
        If only *bank* present, it ranges from 0 to total number of rows in a bank.
    :param block: block. If ranges from 0 to total number of blocks on one shank.
        If *bank* present. If ranges from 0 to total number of blocks in one bank.
    :param bank: bank. If ranges from 0 to total number of banks on one shank.
    :param overwrite: overwrite the used channel with new electrode.
    :return: added electrodes.
    """
    ret = []
    for s, c, r in iter_electrodes(chmap.probe_type, shank, column, row, block, bank):
        try:
            e = chmap.add_electrode((s, c, r))
        except ChannelHasUsedError as x:
            if overwrite:
                chmap.del_electrode(x.electrode)
                e = chmap.add_electrode((s, c, r))
            else:
                e = None

        if e is not None:
            ret.append(e)

    return ret


def npx24_single_shank(shank: int, row: int = 0, *, um: bool = False) -> ChannelMap:
    """
    Make a block channelmap for 4-shank Neuropixels probe.

    >>> print_probe(npx24_single_shank(0))
    624▕ ▏▕ ▏▕ ▏▕ ▏
    576▕ ▏▕ ▏▕ ▏▕ ▏
    528▕ ▏▕ ▏▕ ▏▕ ▏
    480▕ ▏▕ ▏▕ ▏▕ ▏
    432▕ ▏▕ ▏▕ ▏▕ ▏
    384▕ ▏▕ ▏▕ ▏▕ ▏
    336▕ ▏▕ ▏▕ ▏▕ ▏
    288▕ ▏▕ ▏▕ ▏▕ ▏
    240▕ ▏▕ ▏▕ ▏▕ ▏
    192▕ ▏▕ ▏▕ ▏▕ ▏
    144▕█▏▕ ▏▕ ▏▕ ▏
     96▕█▏▕ ▏▕ ▏▕ ▏
     48▕█▏▕ ▏▕ ▏▕ ▏
      0▕█▏▕ ▏▕ ▏▕ ▏
        ╹  ╹  ╹  ╹

    :param shank:
    :param row: beginning row from tip.
    :param um: use um in *row*.
    :return: npx 24 channelmap instance
    """
    probe_type = PROBE_TYPE_NP24

    if um:
        row = int(row / probe_type.r_space)

    if not (0 <= shank < probe_type.n_shank):
        raise ValueError(f'shank out of range : {shank}')

    nr = probe_type.n_channels // probe_type.n_col_shank
    if not (0 <= row < probe_type.n_row_shank - nr):
        if um:
            raise ValueError(f'top row out of range : {probe_type.r_space * (row + nr)}um')
        else:
            raise ValueError(f'top row out of range : {row + nr}')

    ret = ChannelMap(probe_type)
    set_electrodes(ret, shank, row=range(row, row + nr))
    return ret


def npx24_stripe(row: int = 0, *, um: bool = False) -> ChannelMap:
    """
    Make a block channelmap for 4-shank Neuropixels probe.

    >>> print_probe(npx24_stripe())
    624▕ ▏▕ ▏▕ ▏▕ ▏
    576▕ ▏▕ ▏▕ ▏▕ ▏
    528▕ ▏▕ ▏▕ ▏▕ ▏
    480▕ ▏▕ ▏▕ ▏▕ ▏
    432▕ ▏▕ ▏▕ ▏▕ ▏
    384▕ ▏▕ ▏▕ ▏▕ ▏
    336▕ ▏▕ ▏▕ ▏▕ ▏
    288▕ ▏▕ ▏▕ ▏▕ ▏
    240▕ ▏▕ ▏▕ ▏▕ ▏
    192▕ ▏▕ ▏▕ ▏▕ ▏
    144▕ ▏▕ ▏▕ ▏▕ ▏
     96▕ ▏▕ ▏▕ ▏▕ ▏
     48▕ ▏▕ ▏▕ ▏▕ ▏
      0▕█▏▕█▏▕█▏▕█▏
        ╹  ╹  ╹  ╹

    :param row: beginning row from tip.
    :param um: use um in *row*.
    :return: npx 24 channelmap instance
    """
    probe_type = PROBE_TYPE_NP24

    if um:
        row = int(row / probe_type.r_space)

    nr = probe_type.n_channels // probe_type.n_col_shank // probe_type.n_shank
    if not (0 <= row < probe_type.n_row_shank - nr):
        if um:
            raise ValueError(f'top row out of range : {probe_type.r_space * (row + nr)}um')
        else:
            raise ValueError(f'top row out of range : {row + nr}')

    ret = ChannelMap(probe_type)
    set_electrodes(ret, row=range(row, row + nr))
    return ret


def npx24_half_density(shank: int | tuple[int, int], row: int = 0, *, um: bool = False) -> ChannelMap:
    """
    Make a channelmap for 4-shank Neuropixels probe that uniformly distributes channels in half density.

    >>> print_probe([npx24_half_density(shank=0), npx24_half_density(shank=(0, 1))])
            :              :
            :              :
    386▕ ▏▕ ▏▕ ▏▕ ▏386▕ ▏▕ ▏▕ ▏▕ ▏
    384▕ ▏▕ ▏▕ ▏▕ ▏384▕ ▏▕ ▏▕ ▏▕ ▏
    382▕▚▏▕ ▏▕ ▏▕ ▏382▕ ▏▕ ▏▕ ▏▕ ▏
    380▕▚▏▕ ▏▕ ▏▕ ▏380▕ ▏▕ ▏▕ ▏▕ ▏
            :              :
    194▕▚▏▕ ▏▕ ▏▕ ▏194▕ ▏▕ ▏▕ ▏▕ ▏
    192▕▚▏▕ ▏▕ ▏▕ ▏192▕ ▏▕ ▏▕ ▏▕ ▏
    190▕▞▏▕ ▏▕ ▏▕ ▏190▕▞▏▕▚▏▕ ▏▕ ▏
    188▕▞▏▕ ▏▕ ▏▕ ▏188▕▞▏▕▚▏▕ ▏▕ ▏
            :              :
      6▕▞▏▕ ▏▕ ▏▕ ▏  6▕▞▏▕▚▏▕ ▏▕ ▏
      4▕▞▏▕ ▏▕ ▏▕ ▏  4▕▞▏▕▚▏▕ ▏▕ ▏
      2▕▞▏▕ ▏▕ ▏▕ ▏  2▕▞▏▕▚▏▕ ▏▕ ▏
      0▕▞▏▕ ▏▕ ▏▕ ▏  0▕▞▏▕▚▏▕ ▏▕ ▏
        ╹  ╹  ╹  ╹     ╹  ╹  ╹  ╹

    :param shank:
    :param row: beginning row from tip.
    :param um: use um in *row*.
    :return: npx 24 channelmap instance
    """
    probe_type = PROBE_TYPE_NP24

    if um:
        row = int(row / probe_type.r_space)

    ret = ChannelMap(probe_type)

    if isinstance(shank, int):
        r = row
        set_electrodes(ret, shank=shank, column=0, row=range(r, r + 192, 2))
        r += 1
        set_electrodes(ret, shank=shank, column=1, row=range(r, r + 192, 2))

        r = row + 192
        set_electrodes(ret, shank=shank, column=1, row=range(r, r + 192, 2))
        r += 1
        set_electrodes(ret, shank=shank, column=0, row=range(r, r + 192, 2))
    else:
        r = row
        set_electrodes(ret, shank=shank[0], column=0, row=range(r, r + 192, 2))
        r += 1
        set_electrodes(ret, shank=shank[0], column=1, row=range(r, r + 192, 2))

        r = row
        set_electrodes(ret, shank=shank[1], column=1, row=range(r, r + 192, 2))
        r += 1
        set_electrodes(ret, shank=shank[1], column=0, row=range(r, r + 192, 2))

    return ret


def npx24_quarter_density(shank: int | tuple[int, int] | None = None,
                          row: int = 0, *, um: bool = False) -> ChannelMap:
    """
    Make a channelmap for 4-shank Neuropixels probe that uniformly distributes channels in quarter density.

    >>> print_probe([npx24_quarter_density(shank=0), npx24_quarter_density(shank=(0, 2)), npx24_quarter_density(None)])
    638▕▘▏▕ ▏▕ ▏▕ ▏638▕ ▏▕ ▏▕ ▏▕ ▏638▕ ▏▕ ▏▕ ▏▕ ▏
    636▕▝▏▕ ▏▕ ▏▕ ▏636▕ ▏▕ ▏▕ ▏▕ ▏636▕ ▏▕ ▏▕ ▏▕ ▏
            :              :
            :              :
    386▕▝▏▕ ▏▕ ▏▕ ▏386▕ ▏▕ ▏▕ ▏▕ ▏386▕ ▏▕ ▏▕ ▏▕ ▏
    384▕▘▏▕ ▏▕ ▏▕ ▏384▕ ▏▕ ▏▕ ▏▕ ▏384▕ ▏▕ ▏▕ ▏▕ ▏
    382▕▖▏▕ ▏▕ ▏▕ ▏382▕▖▏▕ ▏▕▝▏▕ ▏382▕ ▏▕ ▏▕ ▏▕ ▏
    380▕▗▏▕ ▏▕ ▏▕ ▏380▕▗▏▕ ▏▕▘▏▕ ▏380▕ ▏▕ ▏▕ ▏▕ ▏
            :              :              :
    194▕▖▏▕ ▏▕ ▏▕ ▏194▕▖▏▕ ▏▕▝▏▕ ▏194▕ ▏▕ ▏▕ ▏▕ ▏
    192▕▗▏▕ ▏▕ ▏▕ ▏192▕▗▏▕ ▏▕▘▏▕ ▏192▕ ▏▕ ▏▕ ▏▕ ▏
    190▕▗▏▕ ▏▕ ▏▕ ▏190▕▗▏▕ ▏▕▘▏▕ ▏190▕▗▏▕▖▏▕▝▏▕▘▏
    188▕▖▏▕ ▏▕ ▏▕ ▏188▕▖▏▕ ▏▕▝▏▕ ▏188▕▖▏▕▗▏▕▘▏▕▝▏
            :              :              :
      6▕▗▏▕ ▏▕ ▏▕ ▏  6▕▗▏▕ ▏▕▘▏▕ ▏  6▕▗▏▕▖▏▕▝▏▕▘▏
      4▕▖▏▕ ▏▕ ▏▕ ▏  4▕▖▏▕ ▏▕▝▏▕ ▏  4▕▖▏▕▗▏▕▘▏▕▝▏
      2▕▗▏▕ ▏▕ ▏▕ ▏  2▕▗▏▕ ▏▕▘▏▕ ▏  2▕▗▏▕▖▏▕▝▏▕▘▏
      0▕▖▏▕ ▏▕ ▏▕ ▏  0▕▖▏▕ ▏▕▝▏▕ ▏  0▕▖▏▕▗▏▕▘▏▕▝▏
        ╹  ╹  ╹  ╹     ╹  ╹  ╹  ╹     ╹  ╹  ╹  ╹

    Note: ``npx24_quarter_density(shank=0)`` remain 64 electrodes unset.

    :param shank:
    :param row: beginning row from tip.
    :param um: use um in *row*.
    :return: npx 24 channelmap instance
    """
    probe_type = PROBE_TYPE_NP24

    if um:
        row = int(row / probe_type.r_space)

    ret = ChannelMap(probe_type)

    if shank is None:
        r = row
        set_electrodes(ret, shank=0, column=0, row=range(r, r + 192, 4))
        r += 2
        set_electrodes(ret, shank=0, column=1, row=range(r, r + 192, 4))

        r = row
        set_electrodes(ret, shank=1, column=1, row=range(r, r + 192, 4))
        r += 2
        set_electrodes(ret, shank=1, column=0, row=range(r, r + 192, 4))

        r = row + 1
        set_electrodes(ret, shank=2, column=0, row=range(r, r + 192, 4))
        r += 2
        set_electrodes(ret, shank=2, column=1, row=range(r, r + 192, 4))

        r = row + 1
        set_electrodes(ret, shank=3, column=1, row=range(r, r + 192, 4))
        r += 2
        set_electrodes(ret, shank=3, column=0, row=range(r, r + 192, 4))

    elif isinstance(shank, int):
        r = row
        set_electrodes(ret, shank=shank, column=0, row=range(r, r + 192, 4))
        r += 2
        set_electrodes(ret, shank=shank, column=1, row=range(r, r + 192, 4))

        r = row + 192
        set_electrodes(ret, shank=shank, column=1, row=range(r, r + 192, 4))
        r += 2
        set_electrodes(ret, shank=shank, column=0, row=range(r, r + 192, 4))

        r = row + 384 + 1
        set_electrodes(ret, shank=shank, column=0, row=range(r, r + 192, 4))
        r += 2
        set_electrodes(ret, shank=shank, column=1, row=range(r, r + 192, 4))

        r = row + 576 + 1
        set_electrodes(ret, shank=shank, column=1, row=range(r, r + 192, 4))
        r += 2
        set_electrodes(ret, shank=shank, column=0, row=range(r, r + 192, 4))

        # remain 64 electrodes
    else:
        r = row
        set_electrodes(ret, shank=shank[0], column=0, row=range(r, r + 192, 4))
        r += 2
        set_electrodes(ret, shank=shank[0], column=1, row=range(r, r + 192, 4))

        r = row + 192
        set_electrodes(ret, shank=shank[0], column=1, row=range(r, r + 192, 4))
        r += 2
        set_electrodes(ret, shank=shank[0], column=0, row=range(r, r + 192, 4))

        r = row + 1
        set_electrodes(ret, shank=shank[1], column=1, row=range(r, r + 192, 4))
        r += 2
        set_electrodes(ret, shank=shank[1], column=0, row=range(r, r + 192, 4))

        r = row + 192 + 1
        set_electrodes(ret, shank=shank[1], column=0, row=range(r, r + 192, 4))
        r += 2
        set_electrodes(ret, shank=shank[1], column=1, row=range(r, r + 192, 4))

    return ret


def npx24_one_eighth_density(row: int = 0, *, um: bool = False) -> ChannelMap:
    """
    Make a channelmap for 4-shank Neuropixels probe that uniformly distributes channels in one-eighth density.

    >>> print_probe(npx24_one_eighth_density())
            :
            :
    386▕ ▏▕ ▏▕ ▏▕ ▏
    384▕ ▏▕ ▏▕ ▏▕ ▏
    382▕ ▏▕ ▏▕▖▏▕▘▏
    380▕▖▏▕▘▏▕ ▏▕ ▏
            :
      6▕ ▏▕ ▏▕▗▏▕▝▏
      4▕▗▏▕▝▏▕ ▏▕ ▏
      2▕ ▏▕ ▏▕▖▏▕▘▏
      0▕▖▏▕▘▏▕ ▏▕ ▏
        ╹  ╹  ╹  ╹

    :param row: beginning row from tip.
    :param um: use um in *row*.
    :return: npx 24 channelmap instance
    """
    probe_type = PROBE_TYPE_NP24

    if um:
        row = int(row / probe_type.r_space)

    ret = ChannelMap(probe_type)

    r = row
    set_electrodes(ret, shank=0, column=0, row=range(r, r + 192, 8))
    r += 1
    set_electrodes(ret, shank=1, column=0, row=range(r, r + 192, 8))
    r += 1
    set_electrodes(ret, shank=2, column=0, row=range(r, r + 192, 8))
    r += 1
    set_electrodes(ret, shank=3, column=0, row=range(r, r + 192, 8))
    r += 1
    set_electrodes(ret, shank=0, column=1, row=range(r, r + 192, 8))
    r += 1
    set_electrodes(ret, shank=1, column=1, row=range(r, r + 192, 8))
    r += 1
    set_electrodes(ret, shank=2, column=1, row=range(r, r + 192, 8))
    r += 1
    set_electrodes(ret, shank=3, column=1, row=range(r, r + 192, 8))
    r = row + 192
    set_electrodes(ret, shank=0, column=1, row=range(r, r + 192, 8))
    r += 1
    set_electrodes(ret, shank=1, column=1, row=range(r, r + 192, 8))
    r += 1
    set_electrodes(ret, shank=2, column=1, row=range(r, r + 192, 8))
    r += 1
    set_electrodes(ret, shank=3, column=1, row=range(r, r + 192, 8))
    r += 1
    set_electrodes(ret, shank=0, column=0, row=range(r, r + 192, 8))
    r += 1
    set_electrodes(ret, shank=1, column=0, row=range(r, r + 192, 8))
    r += 1
    set_electrodes(ret, shank=2, column=0, row=range(r, r + 192, 8))
    r += 1
    set_electrodes(ret, shank=3, column=0, row=range(r, r + 192, 8))

    return ret


@doc_link()
def print_probe(chmap: ChannelMap | list[ChannelMap], *,
                file=None,
                truncate: Literal['none', 'top', 'both'] = 'none',
                um: bool = False) -> str | None:
    """
    print probe in string.

    :param chmap: a {ChannelMap} instance, or a list of {ChannelMap}
    :param file: io, or ``print``
    :param truncate: truncate rows without electrodes
    :param um: print depth from tip in um, instead of row.
    :return: a string when *file* is ``None``. Otherwise, return ``None``
    """
    return_str = file is None
    if file is None:
        import io
        file = io.StringIO()
    elif file is print:
        file = sys.stdout

    if isinstance(chmap, ChannelMap):
        ret = _print_probe(chmap, um=um)

    else:
        ret = []

        for i, _chmap in enumerate(chmap):
            if i == 0:
                ret = _print_probe(_chmap, um=um)
            else:
                res = _print_probe(_chmap, um=um)

                if (last := len(ret)) < len(res):
                    empty = ' ' * len(ret[last - 1])
                    for _ in range(last, len(res)):
                        ret.append((empty, 0))

                for j, (a2, c2) in enumerate(res):  # type: int, tuple[str, int]
                    a1, c1 = ret[j]
                    ret[j] = (a1 + a2, c1 + c2)

    if truncate in ('both', 'top'):
        while len(ret) > 0 and ret[-1][1] == 0:
            del ret[-1]

        if truncate == 'both':
            while len(ret) > 0 and ret[0][1] == 0:
                del ret[0]

    for i in range(len(ret)):
        print(ret[-1 - i][0], file=file)

    if return_str:
        return file.getvalue()

    return None


_PRINT_PROBE_UNICODE_SYMBOL = {
    'left': '▕',
    'right': '▏',
    # 0x02 0x08
    # 0x01 0x04
    0: ' ',
    1: '▖',
    2: '▘',
    3: '▌',
    4: '▗',
    5: '▄',
    6: '▚',
    7: '▙',
    8: '▝',
    9: '▞',
    10: '▀',
    11: '▛',
    12: '▐',
    13: '▟',
    14: '▜',
    15: '█',
}


def _print_probe(chmap: ChannelMap, *, um: bool = False) -> list[tuple[str, int]]:
    # TODO not test with n_column > 2
    symbol = _PRINT_PROBE_UNICODE_SYMBOL

    probe_type = chmap.probe_type
    arr = np.zeros((probe_type.n_shank, probe_type.n_row_shank // 2, probe_type.n_col_shank // 2))  # (S, R, C)
    for e in chmap.electrodes:
        s = e.shank
        ci, cj = divmod(e.column, 2)
        ri, rj = divmod(e.row, 2)
        arr[s, ri, ci] = int(arr[s, ri, ci]) | (1 if cj == 0 else 4) * (1 if rj == 0 else 2)

    ret = []
    for r in range(arr.shape[1]):  # R
        cols = arr[:, r, :]  # (S,  C)
        code = int(sum(cols))
        arts = ''.join([
            symbol['left'] + ''.join(map(symbol.__getitem__, col)) + symbol['right']
            for col in cols
        ])
        ret.append((arts, code))

    if um:
        rows = [str(r * probe_type.r_space) for r in range(0, probe_type.n_row_shank, 2)]
    else:
        rows = [str(r) for r in range(0, probe_type.n_row_shank, 2)]

    rown = max(map(len, rows))
    rowf = f'%{rown}s'
    for i in range(len(ret)):
        ret[i] = (rowf % rows[i] + ret[i][0], ret[i][1])

    if (c := chmap.probe_type.n_col_shank) == 2:
        tip = ' ╹ ' * probe_type.n_shank
    else:
        side = ' ' * (c // 2)
        tip = side + '◥◤' + side
        tip = tip * probe_type.n_shank
    ret.insert(0, (rowf % '' + tip, 0))

    return ret
