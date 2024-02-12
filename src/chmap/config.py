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

    # Source
    chmap_root: Path

    # Probe
    probe_family: str
    selector: str | None

    # Atlas
    atlas_name: int | str
    atlas_root: Path | None

    # Application
    config_file: Path
    extra_view: list[str]
    no_open_browser: bool

    debug: bool

    # File
    open_file: str | None


def new_parser() -> argparse.ArgumentParser:
    """Create a cli parse for ChannelMapEditorConfig."""
    ap = argparse.ArgumentParser(prog='chmap')

    ap.add_argument(metavar='FILE', nargs='?', type=Path, default=None, dest='open_file',
                    help='open channelmap file.')
    #
    gp = ap.add_argument_group('Source')
    gp.add_argument('-C', '--chmap-dir', metavar='PATH', type=Path, default=Path('.'), dest='chmap_root',
                    help='channel saving directory')

    #
    gp = ap.add_argument_group('Probe')
    gp.add_argument('-P', '--probe', metavar='NAME', default='npx', dest='probe_family',
                    help='use probe family. default use "npx" (Neuropixels probe family).')
    gp.add_argument('--selector', metavar='MODULE:NAME', default='default', dest='selector',
                    help='use which electrode selection method')

    #
    gp = ap.add_argument_group('Atlas')
    gp.add_argument('--atlas', metavar='NAME', type=_try_int, default=25, dest='atlas_name',
                    help='atlas mouse brain name')
    gp.add_argument('--atlas-root', metavar='PATH', type=Path, default=None, dest='atlas_root',
                    help='atlas mouse brain download path')

    #
    gp = ap.add_argument_group('Bokeh Application')
    gp.add_argument('--config-file', metavar='FILE', type=Path, default=None, dest='config_file',
                    help='global config file.')
    gp.add_argument('--view', metavar='MODULE:NAME', type=str, default=list(), action='append', dest='extra_view',
                    help='install extra views in right panel')
    gp.add_argument('--no-open-browser', action='store_true', dest='no_open_browser',
                    help='do not open browser when server starts')
    gp.add_argument('--debug', action='store_true', dest='debug',
                    help=argparse.SUPPRESS)
    ap.add_mutually_exclusive_group()
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
