from pathlib import Path

from bokeh.models import Div, Select, AutocompleteInput, Toggle
from bokeh.plotting import figure as Figure

from chmap.config import ChannelMapEditorConfig, parse_cli
from chmap.probe import get_probe_desp
from chmap.probe_npx.npx import ChannelMap
from chmap.util.atlas_brain import BrainGlobeAtlas, get_atlas_brain
from chmap.util.bokeh_app import BokehApplication, run_server
from chmap.util.bokeh_util import ButtonFactory, SelectFactory
from chmap.util.bokeh_view import MessageLogArea
from chmap.views.probe import ProbeView


class ChannelMapEditorApp(BokehApplication):
    atlas_brain: BrainGlobeAtlas | None

    def __init__(self, config: ChannelMapEditorConfig):
        self.config = config
        self._setup_atlas_brain()
        self._desp = get_probe_desp('npx')

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
        return list(sorted(self.config.imro_root.glob('*.imro'), key=Path.name.__get__))

    def get_imro_file(self, name: str) -> Path:
        if '/' in name:
            return Path(name)
        else:
            return (self.config.imro_root / name).with_suffix('.imro')

    def load_imro(self, name: str | Path) -> ChannelMap:
        return ChannelMap.from_imro(self.get_imro_file(name))

    def save_imro(self, name: str | Path, chmap: ChannelMap):
        return chmap.save_imro(self.get_imro_file(name).with_suffix('.imro'))

    def get_policy_file(self, chmap: str | Path, name: str) -> Path:
        imro_file = self.get_imro_file(chmap)
        return imro_file.with_stem(imro_file.stem + '-' + name).with_suffix('.npy')

    def load_policy(self, chmap: str | Path, name: str):
        pass

    def save_policy(self, chmap: str | Path, name: str):
        pass

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

        enable_btn = new_btn('Enable', self.on_enable)
        disable_btn = new_btn('Disable', self.on_disable)
        unset_btn = new_btn('Unset', self.on_unset)
        set_btn = new_btn('Set', self.on_set)
        density2_btn = new_btn('Half Density', self.on_density_2)
        density4_btn = new_btn('Quarter Density', self.on_density_4)
        random_btn = new_btn('Random', self.on_random)
        forbidden_btn = new_btn('Forbidden', self.on_forbidden)
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
                    header("Direct"),
                    [enable_btn, disable_btn],
                    header("Random"),
                    [unset_btn, set_btn],
                    [density2_btn, density4_btn],
                    [forbidden_btn, random_btn],
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

        if (use_imro := self.main.use_imro()) is not None:
            if '/' in use_imro:
                use_imro = Path(use_imro).stem
            if use_imro in self.input_imro.options:
                self.input_imro.value = use_imro
                self.run_later(self.on_load)

    def update_probe_info(self):
        s = self.probe_view.shank_map
        if s is None:
            self.probe_info.text = f"<b>Probe</b> 0 / 0"
        else:
            t = s.n_channels
            c = len(s.electrodes)
            self.probe_info.text = f"<b>Probe</b> {c} / {t}"

    # ========= #
    # callbacks #
    # ========= #

    def on_new(self):
        self.probe_view.reset()
        self.probe_view.update_probe()

        if len(self.output_imro.value_input) == 0:
            self.output_imro.value = "New.imro"

        self.update_probe_info()

    def on_load(self):
        name: str = self.input_imro.value
        try:
            self.log_message(f'load {name}')
            sm = self.load_imro(name)
        except FileNotFoundError as x:
            self.log_message(repr(x))
        else:
            self.probe_view.reset(sm)
            self.probe_view.update_probe()

            self.output_imro.value = name.replace('.imro', '')

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
            sm = self.probe_view.shank_map
            if len(sm) < imec_imro.PROBE_TYPE[sm.probe_type].n_channels:
                self.log_message(f'incomplete imro : {len(sm)}')
            else:
                path = self.main.save_imro(name, sm)
                self.log_message('save', path.name)
                self.output_imro.value = path.stem
                self.reload_input_imro_list(path.stem)

    def on_enable(self):
        self.probe_view.set_state_for_selected(Electrode.STATE_USED)
        self.probe_view.update_probe()
        self.update_probe_info()

    def on_disable(self):
        self.probe_view.set_state_for_selected(Electrode.STATE_UNUSED)
        self.probe_view.update_probe()
        self.update_probe_info()

    def on_set(self):
        self.probe_view.set_possible_for_selected(Electrode.POSSIBLE_SET)
        self._update_channelmap()

    def on_unset(self):
        self.probe_view.set_possible_for_selected(Electrode.POSSIBLE_UNSET)
        self._update_channelmap()

    def on_forbidden(self):
        self.probe_view.set_possible_for_selected(Electrode.POSSIBLE_FORBIDDEN)
        self._update_channelmap()

    def on_density_2(self):
        self.probe_view.set_possible_for_selected(Electrode.POSSIBLE_D2)
        self._update_channelmap()

    def on_density_4(self):
        self.probe_view.set_possible_for_selected(Electrode.POSSIBLE_D4)
        self._update_channelmap()

    def on_random(self):
        self.probe_view.set_possible_for_selected(Electrode.POSSIBLE_RANDOM)
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
        self.probe_view.update_probe()
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
