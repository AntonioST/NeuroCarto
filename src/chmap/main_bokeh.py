import functools
from pathlib import Path
from typing import Any

import numpy as np
from bokeh.events import MenuItemClick
from bokeh.models import Div, Select, AutocompleteInput, Toggle, Dropdown, tools
from bokeh.plotting import figure as Figure

from chmap.config import ChannelMapEditorConfig, parse_cli
from chmap.probe import get_probe_desp, ProbeDesp, M
from chmap.util.atlas_brain import BrainGlobeAtlas, get_atlas_brain
from chmap.util.bokeh_app import BokehApplication, run_server
from chmap.util.bokeh_util import ButtonFactory, SelectFactory
from chmap.util.bokeh_view import MessageLogArea, col_layout
from chmap.views.probe import ProbeView


class ChannelMapEditorApp(BokehApplication):
    probe: ProbeDesp[M, Any]
    atlas_brain: BrainGlobeAtlas | None

    def __init__(self, config: ChannelMapEditorConfig):
        self.config = config
        self._setup_atlas_brain()

        self.probe = get_probe_desp('npx')()

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
        pattern = '*' + self.probe.channelmap_file_suffix
        return list(sorted(self.config.imro_root.glob(pattern), key=Path.name.__get__))

    def get_imro_file(self, name: str) -> Path:
        if '/' in name:
            return Path(name)
        else:
            return (self.config.imro_root / name).with_suffix(self.probe.channelmap_file_suffix)

    def load_imro(self, name: str | Path) -> M:
        file = self.get_imro_file(name)
        ret = self.probe.load_from_file(file)
        self.log_message(f'load channelmap : {file.name}')
        self._use_imro = file
        return ret

    def save_imro(self, name: str | Path, chmap: M) -> Path:
        file = self.get_imro_file(name).with_suffix(self.probe.channelmap_file_suffix)
        self.probe.save_to_file(chmap, file)
        self.log_message(f'save channelmap : {file.name}')
        self._use_imro = file
        return file

    def get_policy_file(self, chmap: str | Path) -> Path:
        imro_file = self.get_imro_file(chmap)
        return imro_file.with_stem(imro_file.stem).with_suffix('.npy')

    def load_policy(self, chmap: str | Path) -> bool:
        if (electrodes := self.probe_view.electrodes) is None:
            return False

        file = self.get_policy_file(chmap)

        try:
            data = np.load(file)
        except FileNotFoundError:
            return False
        else:
            self.log_message(f'load policy : {file.name}')

        self.probe.electrode_from_numpy(electrodes, data)
        return True

    def save_policy(self, chmap: str | Path) -> Path | None:
        if (electrodes := self.probe_view.electrodes) is None:
            return None

        file = self.get_policy_file(chmap)
        data = self.probe.electrode_to_numpy(electrodes)

        np.save(file, data)
        self.log_message(f'save policy : {file.name}')

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
        self.input_imro = new_select('Input Channelmap file', [])

        self.output_imro = AutocompleteInput(title='Output Channelmap file', width=300,
                                             max_completions=5, case_sensitive=False,
                                             restrict=True)

        #
        self.probe_info = header("<b>Probe</b>")
        self.probe_fig = Figure(width=600, height=800, tools='', toolbar_location='above')
        self.probe_view = ProbeView(self.probe)
        self.probe_view.plot(self.probe_fig)

        self.probe_fig.tools.clear()
        self.probe_fig.add_tools(
            (t_drag := tools.PanTool(dimensions='height')),
            tools.BoxSelectTool(description='select electrode', renderers=list(self.probe_view.render_electrodes.values())),
            tools.WheelPanTool(dimension='height'),
            (t_scroll := tools.WheelZoomTool(dimensions='height')),
            tools.ResetTool()
        )
        self.probe_fig.toolbar.active_drag = t_drag
        self.probe_fig.toolbar.active_scroll = t_scroll

        state_btns = col_layout([
            new_btn(name, functools.partial(self.on_state_change, state=value))
            for name, value in self.probe.possible_states.items()
        ], 2)

        policy_btns = col_layout([
            new_btn(name, functools.partial(self.on_policy_change, policy=value))
            for name, value in self.probe.possible_policy.items()
        ], 2)

        refresh_btn = new_btn('Refresh', self.on_refresh)
        self.auto_btn = Toggle(label='Auto', active=False, min_width=150, width_policy='min')
        self.auto_btn.on_change('active', self.on_autoupdate)

        empty_btn = Dropdown(
            label='New',
            menu=list(self.probe.possible_type),
            min_width=100, align='end', width_policy='min',
            stylesheets=["div.bk-menu { width: 300%; }"]
        )
        empty_btn.on_click(self.on_new)
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

    def update_probe_info(self):
        self.probe_info.text = self.probe_view.channelmap_desp()

    # ========= #
    # callbacks #
    # ========= #

    def on_new(self, e: MenuItemClick):
        probe_type = self.probe.possible_type[e.item]

        self._use_imro = None
        self.probe_view.reset(probe_type)
        self.probe_view.update_electrode()

        if len(self.output_imro.value_input) == 0:
            self.output_imro.value = "New"

        self.update_probe_info()

    def on_load(self):
        name: str
        if len(name := self.input_imro.value) == 0:
            return

        try:
            file = self.get_imro_file(name)
            chmap = self.load_imro(file)
        except FileNotFoundError as x:
            self.log_message(repr(x))
        else:
            self.output_imro.value = file.stem

            self.probe_view.reset(chmap)
            self.load_policy(file)

            self.probe_view.update_electrode()

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
            return

        chmap = self.probe_view.channelmap
        if not self.probe.is_valid(chmap):
            self.log_message(f'incomplete channelmap')
            return

        path = self.save_imro(name, chmap)
        self.output_imro.value = path.stem
        self.reload_input_imro_list(path.stem)

    def on_state_change(self, state: int):
        self.probe_view.set_state_for_selected(state)
        self.probe_view.update_electrode()
        self.update_probe_info()

    def on_policy_change(self, policy: int):
        self.probe_view.set_policy_for_selected(policy)

        if self.auto_btn.active:
            self.on_refresh()

    # noinspection PyUnusedLocal
    def on_autoupdate(self, prop: str, old: bool, active: bool):
        if active:
            self.on_refresh()

    def on_refresh(self):
        self.probe_view.refresh_selection()
        self.probe_view.update_electrode()
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
