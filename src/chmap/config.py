import argparse
from pathlib import Path
from typing import cast

__all__ = [
    'ChannelMapEditorConfig',
    'new_parser',
    'parse_cli'
]


class ChannelMapEditorConfig:
    """Start configuration for ChannelMapEditorApp.

    It is a protocol class that wrap for argparse.Namespace.
    """

    probe_family: str
    selector: str | None
    chmap_root: Path
    extra_view: list[str]

    atlas_name: int | str
    atlas_root: Path | None

    no_open_browser: bool
    debug: bool


def new_parser() -> argparse.ArgumentParser:
    """Create a cli parse for ChannelMapEditorConfig."""
    ap = argparse.ArgumentParser(prog='chmap')

    ap.add_argument('-P', '--probe', metavar='NAME', default='npx', dest='probe_family',
                    help='use probe family. default use "npx" (Neuropixels probe family).')
    ap.add_argument('--selector', metavar='MODULE:NAME', default='default', dest='selector',
                    help='use which electrode selection method')

    ap.add_argument('--debug', action='store_true', dest='debug',
                    help=argparse.SUPPRESS)
    #
    gp = ap.add_argument_group('Source')
    gp.add_argument('-C', '--chmap-dir', metavar='PATH', type=Path, default=Path('.'), dest='chmap_root',
                    help='channel saving directory')

    #
    gp = ap.add_argument_group('View')
    gp.add_argument('--view', metavar='MODULE:NAME', type=str, default=list(), action='append', dest='extra_view',
                    help='install extra views in right panel')

    #
    gp = ap.add_argument_group('Atlas')
    gp.add_argument('--atlas', metavar='NAME', type=_try_int, default=25, dest='atlas_name',
                    help='atlas mouse brain name')
    gp.add_argument('--atlas-root', metavar='PATH', type=Path, default=None, dest='atlas_root',
                    help='atlas mouse brain download path')

    #
    gp = ap.add_argument_group('Bokeh Application')
    gp.add_argument('--no-open-browser', action='store_true', dest='no_open_browser',
                    help='do not open browser when server starts')

    return ap


def parse_cli(args: list[str] = None) -> ChannelMapEditorConfig:
    """
    Parse command-line arguments and return result.

    :param args: command-line arguments list. use sys.argv if None.
    :return: ChannelMapEditorConfig
    """
    return cast(ChannelMapEditorConfig, new_parser().parse_args(args))


def _try_int(a: str) -> int | str:
    try:
        return int(a)
    except ValueError:
        return a


def _list_type(a: str) -> list[str]:
    return a.split(',')
