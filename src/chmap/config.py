import argparse
from pathlib import Path
from typing import cast

__all__ = [
    'ChannelMapEditorConfig',
    'new_parser',
    'parse_cli'
]


class ChannelMapEditorConfig:
    probe_family: str
    chmap_root: Path

    atlas_name: int | str
    atlas_root: Path | None

    no_open_browser: bool


def new_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()

    ap.add_argument('-P', '--probe', metavar='NAME', default='npx', dest='probe_family',
                    help='use probe family. default use "npx" (Neuropixels probe family).')

    gp = ap.add_argument_group('Source')
    gp.add_argument('-C', '--chmap-dir', metavar='PATH', type=Path, default=Path('.'), dest='chmap_root',
                    help='channel saving directory')

    gp = ap.add_argument_group('Atlas')
    gp.add_argument('--atlas', metavar='NAME', type=_try_int, default=25, dest='atlas_name',
                    help='atlas mouse brain name')
    gp.add_argument('--atlas-root', metavar='PATH', type=Path, default=None, dest='atlas_root',
                    help='atlas mouse brain download path')

    gp = ap.add_argument_group('Bokeh Application')
    gp.add_argument('--no-open-browser', action='store_true', dest='no_open_browser',
                    help='do not open browser when server starts')

    return ap


def parse_cli(args: list[str] = None) -> ChannelMapEditorConfig:
    return cast(ChannelMapEditorConfig, new_parser().parse_args(args))


def _try_int(a: str) -> int | str:
    try:
        return int(a)
    except ValueError:
        return a
