import functools
from pathlib import Path

import numpy as np
from bokeh.models import Div, Select, AutocompleteInput, Toggle
from bokeh.plotting import figure as Figure

from chmap.config import ChannelMapEditorConfig, parse_cli
from chmap.probe_npx.npx import ChannelMap
from chmap.select import Selector
from chmap.util.atlas_brain import BrainGlobeAtlas, get_atlas_brain
from chmap.util.bokeh_app import BokehApplication, run_server
from chmap.util.bokeh_util import ButtonFactory, SelectFactory
from chmap.util.bokeh_view import MessageLogArea, col_layout
from chmap.views.probe import ProbeView


class ChannelMapEditorApp(BokehApplication):
    atlas_brain: BrainGlobeAtlas | None

    def __init__(self, config: ChannelMapEditorConfig):
        self.config = config
        self._setup_atlas_brain()

        from .probe_npx.desp import NpxProbeDesp
        from .probe_npx.select import NpxSelector
        self._desp = NpxProbeDesp()
        self._selector: Selector = NpxSelector(self._desp)

        self._use_imro: Path | None = None

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
        pattern = '*' + self._desp.channelmap_file_suffix
        return list(sorted(self.config.imro_root.glob(pattern), key=Path.name.__get__))

    def get_imro_file(self, name: str) -> Path:
        if '/' in name:
            return Path(name)
        else:
            return (self.config.imro_root / name).with_suffix(self._desp.channelmap_file_suffix)

    def load_imro(self, name: str | Path) -> ChannelMap:
        return ChannelMap.from_imro(self.get_imro_file(name))

    def save_imro(self, name: str | Path, chmap: ChannelMap) -> Path:
        file = self.get_imro_file(name).with_suffix(self._desp.channelmap_file_suffix)
        chmap.save_imro(file)
        return file

    def get_policy_file(self, chmap: str | Path, name: str) -> Path:
        imro_file = self.get_imro_file(chmap)
        return imro_file.with_stem(imro_file.stem + '-' + name).with_suffix('.npy')

    def load_policy(self, chmap: str | Path, name: str) -> bool:
        if (electrodes := self.probe_view.electrodes) is None:
            return False

        data = np.load(self.get_policy_file(chmap, name))
        self._desp.electrode_from_numpy(electrodes, data)
        return True

    def save_policy(self, chmap: str | Path, name: str) -> Path | None:
        if (electrodes := self.probe_view.electrodes) is None:
            return None

        file = self.get_policy_file(chmap, name)
        data = self._desp.electrode_to_numpy(electrodes)
        np.save(file, data)
        return file

    # ============= #
    # UI components #
    # ============= #

    probe_fig: Figure
    probe_view: ProbeView
    probe_info: Div

    input_imro: Select
    output_imro: AutocompleteInput
    message_area: MessageLogArea

    auto_btn: Toggle

    def index(self):
        new_btn = ButtonFactory(min_width=150, width_policy='min')
        new_select = SelectFactory(width=300)

        def header(text: str):
            return Div(text=f'<b>{text}</b>')

        self.message_area = MessageLogArea("Log:", rows=10, cols=100, width=300)

        #
        self.input_imro = new_select('Input Imro', [])

        self.output_imro = AutocompleteInput(title='Output Imro', width=300,
                                             max_completions=5, case_sensitive=False,
                                             restrict=True)

        #
        self.input_imro = new_select('Input Imro', [], None)

        self.output_imro = AutocompleteInput(title='Output Imro', width=300,
                                             max_completions=5, case_sensitive=False,
                                             restrict=True)

        #
        self.probe_info = header("<b>Probe</b>")
        self.probe_fig = Figure(width=600, height=800, tools='', toolbar_location='above')
        self.probe_view = ProbeView(self._desp)
        self.probe_view.plot(self.probe_fig)

        state_btns = col_layout([
            new_btn(name, functools.partial(self.on_state_change, state=value))
            for name, value in self._selector.possible_states.items()
        ], 2)

        policy_btns = col_layout([
            new_btn(name, functools.partial(self.on_policy_change, policy=value))
            for name, value in self._selector.possible_policy.items()
        ], 2)

        refresh_btn = new_btn('Refresh', self.on_refresh)
        self.auto_btn = Toggle(label='Auto', active=False, min_width=150, width_policy='min')
        self.auto_btn.on_change('active', self.on_autoupdate)

        empty_btn = new_btn('New', self.on_new, min_width=100, align='end')
        load_btn = new_btn('Load', self.on_load, min_width=100, align='end')
        save_btn = new_btn('Save', self.on_save, min_width=100, align='end')

        return [  # column
            [  # row
                [
                    header("Imro File"),
                    self.input_imro,
                    [empty_btn, load_btn, save_btn],
                    self.output_imro,
                    header("State"),
                    *state_btns,
                    header("Policy"),
                    *policy_btns,
                    [self.auto_btn, refresh_btn],
                    self.message_area,
                ],
                [
                    self.probe_info,
                    self.probe_fig,
                ],
            ],
        ]

    def update(self):
        self.reload_input_imro_list()

        if (use_imro := self._use_imro) is not None:
            use_imro = use_imro.stem
            if use_imro in self.input_imro.options:
                self.input_imro.value = use_imro
                self.run_later(self.on_load)

    def update_probe_info(self):
        self.probe_info.text = self.probe_view.channelmap_desp()

    # ========= #
    # callbacks #
    # ========= #

    def on_new(self):
        self.probe_view.reset()
        self.probe_view.update()

        if len(self.output_imro.value_input) == 0:
            self.output_imro.value = "New"

        self.update_probe_info()

    def on_load(self):
        name: str = self.input_imro.value
        try:
            self.log_message(f'load {name}')
            file = self.get_imro_file(name)
            chmap = self.load_imro(file)
        except FileNotFoundError as x:
            self.log_message(repr(x))
        else:
            self._use_imro = file

            self.probe_view.reset(chmap)
            self.probe_view.update()

            self.output_imro.value = file.stem

            self.update_probe_info()

    def reload_input_imro_list(self, preselect: str = None):
        if preselect is None:
            preselect = self.input_imro.value

        imro_list = [f.stem for f in self.list_imro_files()]
        self.input_imro.options = imro_list

        if preselect in imro_list:
            self.input_imro.value = preselect
        self.output_imro.completions = imro_list

    def on_save(self):
        name = self.output_imro.value_input
        if len(name) == 0:
            self.log_message('empty output filename')
        else:
            chmap = self.probe_view.channelmap

            if self._desp.is_valid(chmap):
                self.log_message(f'incomplete channelmap')
            else:
                path = self.save_imro(name, chmap)
                self.log_message('save', path.name)
                self.output_imro.value = path.stem
                self.reload_input_imro_list(path.stem)

    def on_state_change(self, state: int):
        self.probe_view.set_state_for_selected(state)
        self.probe_view.update()
        self.update_probe_info()

    def on_policy_change(self, policy: int):
        self.probe_view.set_policy_for_selected(policy)
        self._update_channelmap()

    def _update_channelmap(self):
        if self.auto_btn.active:
            self.on_refresh()

    # noinspection PyUnusedLocal
    def on_autoupdate(self, prop: str, old: bool, active: bool):
        if active:
            self.on_refresh()

    def on_refresh(self):
        self.probe_view.refresh()
        self.probe_view.update()
        self.update_probe_info()

    def log_message(self, *message):
        self.message_area.log_message(*message)


def main(config: ChannelMapEditorConfig = None):
    if config is None:
        config = parse_cli()

    run_server(ChannelMapEditorApp(config),
               no_open_browser=config.no_open_browser)


if __name__ == '__main__':
    main()
