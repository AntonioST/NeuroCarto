import functools
from pathlib import Path
from typing import Any

import numpy as np
from bokeh.events import MenuItemClick
from bokeh.layouts import row as Row, column as Column
from bokeh.models import Div, Select, AutocompleteInput, Toggle, Dropdown, tools, TextAreaInput, UIElement
from bokeh.plotting import figure as Figure

from chmap.config import ChannelMapEditorConfig, parse_cli
from chmap.probe import get_probe_desp, ProbeDesp, M
from chmap.util.bokeh_app import BokehApplication, run_server
from chmap.util.bokeh_util import ButtonFactory, col_layout
from chmap.views.base import ViewBase, StateView, DynamicView
from chmap.views.data import DataView
from chmap.views.probe import ProbeView

__all__ = ['ChannelMapEditorApp', 'main']


class ChannelMapEditorApp(BokehApplication):
    """Application of neural probe channelmap editor.

    The layout of this application is split into:
    * left: channelmap manipulating button groups.
    * center: figure that contains probe/electrodes, image and curves.
    * right: controls of image, curves that shown in center panel.


    """

    probe: ProbeDesp[M, Any]
    """probe describer"""

    right_view_config: dict[str, Any] = {}
    """view configuration"""

    right_view_type: list[ViewBase | type[ViewBase]]
    """install views on right panel"""

    def __init__(self, config: ChannelMapEditorConfig):
        self.config = config

        self.probe = get_probe_desp(config.probe_family)()

        self._use_chmap_path: Path | None = None

        self.right_view_type = []

        if config.atlas_name is not None:
            from chmap.views.atlas import AtlasBrainView
            self.right_view_type.append(AtlasBrainView)

        from chmap.views.data_density import ElectrodeDensityData
        self.right_view_type.append(DataView(config, ElectrodeDensityData()))

    @property
    def title(self) -> str:
        f = self.config.probe_family.upper()
        return f"{f} Channel Map Editor"

    # ==================== #
    # load/save imro files #
    # ==================== #

    def list_chmap_files(self) -> list[Path]:
        pattern = '*' + self.probe.channelmap_file_suffix
        return list(sorted(self.config.chmap_root.glob(pattern), key=Path.name.__get__))

    def get_chmap_file(self, name: str) -> Path:
        if '/' in name:
            p = Path(name)
        else:
            p = self.config.chmap_root / name

        return p.with_suffix(self.probe.channelmap_file_suffix)

    def load_chmap(self, name: str) -> M:
        file = self.get_chmap_file(name)
        ret = self.probe.load_from_file(file)
        self.log_message(f'load channelmap : {file.name}')
        self._use_chmap_path = file
        return ret

    def save_chmap(self, name: str, chmap: M) -> Path:
        file = self.get_chmap_file(name)
        self.probe.save_to_file(chmap, file)
        self.log_message(f'save channelmap : {file.name}')
        self._use_chmap_path = file
        return file

    def get_policy_file(self, chmap: str | Path) -> Path:
        if isinstance(chmap, str):
            imro_file = self.get_chmap_file(chmap)
        else:
            imro_file = chmap
        return imro_file.with_suffix('.npy')

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

    def get_view_config_file(self, chmap: str | Path) -> Path:
        if isinstance(chmap, str):
            imro_file = self.get_chmap_file(chmap)
        else:
            imro_file = chmap

        return imro_file.with_suffix('.json')

    def load_view_config(self, chmap: str | Path, *, reset=False) -> dict[str, Any]:
        file = self.get_view_config_file(chmap)
        if not file.exists():
            return self.right_view_config

        import json
        with file.open('r') as f:
            data = dict(json.load(f))
            self.log_message(f'load image config : {file.name}')

        if reset:
            self.right_view_config.update(data)
        else:
            self.right_view_config = data

        return self.right_view_config

    def save_view_config(self, chmap: str | Path, *, direct=False):
        if not direct:
            for view in self.right_views:
                if isinstance(view, StateView):
                    self.right_view_config[type(view).__name__] = view.save_state()

        import json
        file = self.get_view_config_file(chmap)
        with file.open('w') as f:
            json.dump(self.right_view_config, f, indent=2)
            self.log_message(f'save image config : {file.name}')

    # ============= #
    # UI components #
    # ============= #

    probe_fig: Figure
    probe_view: ProbeView
    probe_info: Div

    right_views: list[ViewBase]

    input_imro: Select
    output_imro: AutocompleteInput
    message_area: TextAreaInput

    auto_btn: Toggle

    def index(self):
        self.probe_info = Div(text="<b>Probe</b>")
        self.probe_fig = Figure(width=600, height=800, tools='', toolbar_location='above')
        self.probe_view = ProbeView(self.probe)
        self.probe_view.plot(self.probe_fig)

        self.probe_fig.tools.clear()
        self.probe_fig.add_tools(
            (t_drag := tools.PanTool(dimensions='both')),
            tools.BoxSelectTool(description='select electrode', renderers=list(self.probe_view.render_electrodes.values())),
            #
            tools.WheelPanTool(dimension='height'),
            (t_scroll := tools.WheelZoomTool(dimensions='both')),
            #
            tools.ResetTool()
        )
        self.probe_fig.toolbar.active_drag = t_drag
        self.probe_fig.toolbar.active_scroll = t_scroll

        return Row(
            Column(self._index_left_control()),
            Column(self.probe_info, self.probe_fig),
            Column(self._index_right_control())
        )

    def _index_left_control(self) -> list[UIElement]:
        new_btn = ButtonFactory(min_width=150, width_policy='min')

        #
        self.input_imro = Select(
            title='Input file',
            options=[], value="",
            width=300
        )

        self.output_imro = AutocompleteInput(
            title='Save filename',
            width=300, max_completions=5, case_sensitive=False, restrict=True
        )

        #
        state_btns = col_layout([
            new_btn(name, functools.partial(self.on_state_change, state=value))
            for name, value in self.probe.possible_states.items()
        ], 2)

        policy_btns = col_layout([
            new_btn(name, functools.partial(self.on_policy_change, policy=value))
            for name, value in self.probe.possible_policy.items()
        ], 2)

        #

        empty_btn = Dropdown(
            label='New',
            menu=list(self.probe.possible_type),
            min_width=100, align='end', width_policy='min',
            stylesheets=["div.bk-menu { width: 300%; }"]
        )
        empty_btn.on_click(self.on_new)

        load_btn = new_btn('Load', self.on_load, min_width=100, align='end')
        save_btn = new_btn('Save', self.on_save, min_width=100, align='end')

        #

        refresh_btn = new_btn('Refresh', self.on_refresh)
        self.auto_btn = Toggle(label='Auto', active=True, min_width=150, width_policy='min')
        self.auto_btn.on_change('active', self.on_autoupdate)

        #
        self.message_area = TextAreaInput(title="Log:", rows=10, cols=100, width=300, disabled=True)

        return [
            Div(text="<b>ChannelMap File</b>"),
            self.input_imro,
            Row(empty_btn, load_btn, save_btn),
            self.output_imro,
            Div(text="<b>State</b>"),
            *state_btns,
            Div(text="<b>Policy</b>"),
            *policy_btns,
            Row(self.auto_btn, refresh_btn),
            self.message_area,
        ]

    def _index_right_control(self) -> list[UIElement]:
        self.right_views = []

        ret = []
        for view_type in self.right_view_type:
            if isinstance(view_type, type):
                view = view_type(self.config)
            elif isinstance(view_type, ViewBase):
                view = view_type
            else:
                raise TypeError()

            view.plot(self.probe_fig)
            ret.extend(view.setup())
            self.right_views.append(view)

        return ret

    def start(self):
        self.reload_input_imro_list()
        for view in self.right_views:
            view.update()

    # ========= #
    # callbacks #
    # ========= #

    def on_new(self, e: MenuItemClick):
        if (item := e.item) is None:
            return

        probe_type = self.probe.possible_type[item]

        self._use_chmap_path = None
        self.probe_view.reset(probe_type)

        if len(self.output_imro.value_input) == 0:
            self.output_imro.value = "New"

        self.on_probe_update()

    def on_load(self):
        name: str
        if len(name := self.input_imro.value) == 0:
            return

        try:
            file = self.get_chmap_file(name)
            chmap = self.load_chmap(name)
        except FileNotFoundError as x:
            self.log_message(repr(x))
            return

        self.output_imro.value = file.stem

        self.probe_view.reset(chmap)
        self.load_policy(file)

        config = self.load_view_config(file, reset=True)
        for view in self.right_views:
            if isinstance(view, StateView):
                try:
                    _config = config[type(view).__name__]
                except KeyError:
                    pass
                else:
                    try:
                        view.restore_state(_config)
                    except RuntimeError as e:
                        self.log_message(type(view).__name__ + '::' + repr(e))

        self.on_probe_update()

    def reload_input_imro_list(self, preselect: str = None):
        if preselect is None:
            preselect = self.input_imro.value

        imro_list = [f.stem for f in self.list_chmap_files()]
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

        path = self.save_chmap(name, chmap)
        self.save_policy(path)
        self.save_view_config(path)

        self.output_imro.value = path.stem
        self.reload_input_imro_list(path.stem)

    def on_state_change(self, state: int):
        self.probe_view.set_state_for_selected(state)
        self.on_probe_update()

    def on_policy_change(self, policy: int):
        self.probe_view.set_policy_for_selected(policy)

        if self.auto_btn.active:
            self.on_refresh()

        self.on_probe_update()

    def on_probe_update(self):
        self.probe_info.text = self.probe_view.channelmap_desp()
        self.probe_view.update_electrode()

        for view in self.right_views:
            if isinstance(view, DynamicView):
                view.on_probe_update(self.probe, self.probe_view.channelmap, self.probe_view.electrodes)

    # noinspection PyUnusedLocal
    def on_autoupdate(self, prop: str, old: bool, active: bool):
        if active:
            self.on_refresh()

    def on_refresh(self):
        self.probe_view.refresh_selection()
        self.probe_view.update_electrode()
        self.on_probe_update()

    def log_message(self, *message, reset=False):
        area = self.message_area
        message = '\n'.join(message)
        area.disabled = False
        try:
            if reset:
                area.value = message
            else:
                text = area.value
                area.value = message + '\n' + text
        finally:
            area.disabled = True

    def log_clear(self):
        area = self.message_area
        area.disabled = False
        try:
            area.value = ""
        finally:
            area.disabled = True


def main(config: ChannelMapEditorConfig = None):
    if config is None:
        config = parse_cli()

    run_server(ChannelMapEditorApp(config),
               no_open_browser=config.no_open_browser)


if __name__ == '__main__':
    main()
