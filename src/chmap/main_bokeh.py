from pathlib import Path

from chmap.gui_config import ChannelMapEditorConfig, parse_cli
from chmap.probe_npx.npx import ChannelMap
from chmap.util.atlas_brain import BrainGlobeAtlas, get_atlas_brain
from chmap.util.bokeh_app import BokehApplication, run_server


class ChannelMapEditorApp(BokehApplication):
    atlas_brain: BrainGlobeAtlas | None

    def __init__(self, config: ChannelMapEditorConfig):
        self.config = config
        self._setup_atlas_brain()

    def _setup_atlas_brain(self):
        try:
            atlas_brain = get_atlas_brain(self.config.atlas_name, self.config.atlas_root)
        except ImportError:
            atlas_brain = None

        self.atlas_brain = atlas_brain

    @property
    def title(self) -> str:
        return "Channel Map"

    # ==================== #
    # load/save imro files #
    # ==================== #

    def list_imro_files(self) -> list[Path]:
        return list(sorted([
            f
            for f in self.config.imro_root.iterdir()
            if f.suffix == '.imro'
        ], key=Path.name.__get__))

    def get_imro_file(self, name: str) -> Path:
        if '/' in name:
            return Path(name)
        else:
            return (self.config.imro_root / name).with_suffix('.imro')

    def load_imro(self, name: str | Path) -> ChannelMap:
        return ChannelMap.from_imro(self.get_imro_file(name))

    def save_imro(self, name: str | Path, chmap: ChannelMap):
        return chmap.save_imro(self.get_imro_file(name).with_suffix('.imro'))


def main(config: ChannelMapEditorConfig = None):
    if config is None:
        config = parse_cli()

    run_server(ChannelMapEditorApp(config),
               no_open_browser=config.no_open_browser)


if __name__ == '__main__':
    main()
