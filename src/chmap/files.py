import os
from pathlib import Path
from typing import Any

from .config import ChannelMapEditorConfig
from .probe import ProbeDesp
from .util.utils import doc_link

__all__ = [
    'user_config_file',
    'load_user_config',
    'save_user_config',
    'channelmap_root',
    'list_channelmap_files',
    'get_channelmap_file',
    'get_blueprint_file',
    'get_view_config_file',
]

USER_CONFIG_FILENAME = 'chmap.config.json'


@doc_link()
def user_config_file(config: ChannelMapEditorConfig) -> Path:
    """
    Get user config filepath.

    * When `--config-file` is given, use it.
    * When `--debug`, use `.{USER_CONFIG_FILENAME}` at current working directory.

    :return: filepath.
    """
    if (ret := config.config_file) is not None:
        if ret.is_dir():
            ret = ret / USER_CONFIG_FILENAME
        return ret

    if config.debug:
        return Path('.') / f'.{USER_CONFIG_FILENAME}'

    # https://wiki.archlinux.org/title/XDG_Base_Directory
    # https://stackoverflow.com/a/3250952
    if (d := os.environ.get('XDG_CONFIG_HOME', None)) is not None:
        return Path(d) / 'chmap' / USER_CONFIG_FILENAME
    elif (d := os.environ.get('APPDATA', None)) is not None:
        return Path(d) / 'chmap' / USER_CONFIG_FILENAME

    # https://stackoverflow.com/a/1857
    import platform
    match platform.system():
        case 'Linux':
            return Path.home() / '.config/chmap' / USER_CONFIG_FILENAME
        case 'Windows':
            pass
        case 'Darwin':
            pass

    return Path.home() / '.chmap' / USER_CONFIG_FILENAME


@doc_link()
def load_user_config(config: ChannelMapEditorConfig) -> dict[str, Any]:
    """

    :param config:
    :return: user config dictionary
    :raise FileNotFoundError: config file does not exist.
    :raise IOError: wrap json.JSONDecodeError
    :see: {#user_config_file()}
    """
    import json

    file = user_config_file(config)
    if not file.exists():
        raise FileNotFoundError(file)

    try:
        with file.open('r') as f:
            data = dict(json.load(f))
    except json.JSONDecodeError as e:
        raise IOError(file) from e

    return data


@doc_link()
def save_user_config(config: ChannelMapEditorConfig, user: dict[str, Any]) -> Path:
    """

    :param config:
    :param user: user config
    :return: saved user confile path
    :see: {#user_config_file()}
    """
    import json

    file = user_config_file(config)
    file.parent.mkdir(parents=True, exist_ok=True)
    with file.open('w') as f:
        json.dump(user, f, indent=2)
    return file


def channelmap_root(config: ChannelMapEditorConfig) -> Path:
    """
    channelmap root.

    :param config:
    :return: directory path
    """
    if (ret := config.chmap_root) is None:
        ret = Path('.')
    return ret


@doc_link()
def list_channelmap_files(config: ChannelMapEditorConfig, probe: ProbeDesp) -> list[Path]:
    """
    List channelmap files.

    :param config:
    :param probe:
    :return: list of files.
    :see: {channelmap_root()}
    """
    pattern = '*' + probe.channelmap_file_suffix
    return list(sorted(channelmap_root(config).glob(pattern), key=Path.name.__get__))


@doc_link()
def get_channelmap_file(config: ChannelMapEditorConfig, probe: ProbeDesp, filename: str) -> Path:
    """
    Get channelmap file path.

    :param config:
    :param probe:
    :param filename:
    :return: a channelmap file path
    :see: {channelmap_root()}
    """
    if '/' in filename:
        p = Path(filename)
    else:
        p = channelmap_root(config) / filename

    return p.with_suffix(probe.channelmap_file_suffix)


@doc_link()
def get_blueprint_file(config: ChannelMapEditorConfig, probe: ProbeDesp, chmap: str | Path) -> Path:
    """
    Get corresponded blueprint file from channelmap path.

    :param config:
    :param probe:
    :param chmap: a filename, or a channelmap file path.
    :return: a blueprint file path
    :see: {get_channelmap_file()}
    """

    if isinstance(chmap, str):
        chmap = get_channelmap_file(config, probe, chmap)
    elif isinstance(chmap, Path) and chmap.name.endswith('.blueprint.npy'):
        return chmap
    elif not isinstance(chmap, Path):
        raise TypeError(f'not a path : {chmap}')

    return chmap.with_suffix('.blueprint.npy')


@doc_link()
def get_view_config_file(config: ChannelMapEditorConfig, probe: ProbeDesp, chmap: str | Path) -> Path:
    """
    Get view components' configurations saving path.

    :param config:
    :param probe:
    :param chmap:
    :return: a view config path.
    :see: {get_channelmap_file()}
    """
    if isinstance(chmap, str):
        chmap = get_channelmap_file(config, probe, chmap)
    elif isinstance(chmap, Path) and chmap.name.endswith('.config.json'):
        return chmap
    elif not isinstance(chmap, Path):
        raise TypeError(f'not a path : {chmap}')

    return chmap.with_suffix('.config.json')
