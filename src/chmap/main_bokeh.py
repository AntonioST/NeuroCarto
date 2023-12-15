import functools
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from bokeh.events import MenuItemClick
from bokeh.layouts import row as Row, column as Column
from bokeh.models import Div, Select, AutocompleteInput, Toggle, Dropdown, tools, Button, TextAreaInput, UIElement
from bokeh.plotting import figure as Figure

from chmap.config import ChannelMapEditorConfig, parse_cli
from chmap.probe import get_probe_desp, ProbeDesp, M
from chmap.util.atlas_brain import BrainGlobeAtlas, get_atlas_brain
from chmap.util.bokeh_app import BokehApplication, run_server
from chmap.views.atlas import AtlasBrainView
from chmap.views.probe import ProbeView


class ChannelMapEditorApp(BokehApplication):
    probe: ProbeDesp[M, Any]
    atlas_brain: BrainGlobeAtlas | None

    def __init__(self, config: ChannelMapEditorConfig):
        self.config = config
        self._setup_atlas_brain()

        self.probe = get_probe_desp(config.probe_family)()

        self._use_chmap_path: Path | None = None

    def _setup_atlas_brain(self):
        try:
            atlas_brain = get_atlas_brain(self.config.atlas_name, self.config.atlas_root)
        except ImportError as e:
            print(repr(e))
            atlas_brain = None

        self.atlas_brain = atlas_brain

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

    def get_policy_file(self, chmap: str) -> Path:
        imro_file = self.get_chmap_file(chmap)
        return imro_file.with_suffix('.npy')

    def load_policy(self, chmap: str) -> bool:
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

    brain_view: AtlasBrainView

    input_imro: Select
    output_imro: AutocompleteInput
    message_area: TextAreaInput

    auto_btn: Toggle

    def index(self):
        self.probe_info = Div(text="<b>Probe</b>")
        self.probe_fig = Figure(width=600, height=800, tools='', toolbar_location='above')
        self.probe_view = ProbeView(self.probe)
        self.probe_view.plot(self.probe_fig)

        dimensions = 'height'
        if self.atlas_brain is not None:
            dimensions = 'both'

        self.probe_fig.tools.clear()
        self.probe_fig.add_tools(
            (t_drag := tools.PanTool(dimensions=dimensions)),
            tools.BoxSelectTool(description='select electrode', renderers=list(self.probe_view.render_electrodes.values())),
            #
            tools.WheelPanTool(dimension='height'),
            (t_scroll := tools.WheelZoomTool(dimensions=dimensions)),
            #
            tools.ResetTool()
        )
        self.probe_fig.toolbar.active_drag = t_drag
        self.probe_fig.toolbar.active_scroll = t_scroll

        #
        row = [
            Column(self._index_left_control()),
            Column(self.probe_info, self.probe_fig)
        ]

        if len(c := self._index_right_control()) > 0:
            row.append(Column(c))

        return Row(row)

    def _index_left_control(self) -> list[UIElement]:
        new_btn = ButtonFactory(min_width=150, width_policy='min')

        #
        self.input_imro = Select(
            title='Input Channelmap file',
            options=[], value="",
            width=300
        )

        self.output_imro = AutocompleteInput(
            title='Output Channelmap file',
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
            Div(text="<b>Imro File</b>"),
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
        ret = []
        if self.atlas_brain is not None:
            ret.extend(self._index_brain_view(self.atlas_brain))
        return ret

    def _index_brain_view(self, atlas: BrainGlobeAtlas) -> list[UIElement]:
        new_btn = ButtonFactory(min_width=100, width_policy='min')

        self.brain_view = AtlasBrainView(atlas)
        self.brain_view.plot(self.probe_fig)
        self.brain_view.setup()

        reset_rth = new_btn('reset', self.brain_view.reset_rth)
        reset_rtv = new_btn('reset', self.brain_view.reset_rtv)
        reset_imr = new_btn('reset', self.brain_view.reset_imr)
        reset_ims = new_btn('reset', self.brain_view.reset_ims)

        self.probe_fig.tools.insert(-2, self.brain_view.boundary_tool())

        return [
            Div(text="<b>Brain Atlas</b>"),
            Row(self.brain_view.slice_select, self.brain_view.plane_slider),
            Row(reset_rth, self.brain_view.rth_slider),
            Row(reset_rtv, self.brain_view.rtv_slider),
            Row(reset_imr, self.brain_view.imr_slider),
            Row(reset_ims, self.brain_view.ims_slider),
        ]

    def update(self):
        self.reload_input_imro_list()
        self.update_brain_view()

    def update_brain_view(self):
        try:
            view = self.brain_view
        except AttributeError:
            pass
        else:
            if view.brain_view is None:
                view.update_brain_view(view.slice_select.value)

    def update_probe_info(self):
        self.probe_info.text = self.probe_view.channelmap_desp()

    # ========= #
    # callbacks #
    # ========= #

    def on_new(self, e: MenuItemClick):
        if (item := e.item) is None:
            return

        probe_type = self.probe.possible_type[item]

        self._use_chmap_path = None
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
            file = self.get_chmap_file(name)
            chmap = self.load_chmap(file)
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


class ButtonFactory(object):
    def __init__(self, **kwargs):
        self.__kwargs = kwargs

    def __call__(self, label: str, callback: Callable[..., None], **kwargs) -> Button:
        for k, v in self.__kwargs.items():
            kwargs.setdefault(k, v)
        btn = Button(label=label, **kwargs)
        btn.on_click(callback)
        return btn


def col_layout(model: list[UIElement], n: int) -> list[UIElement]:
    ret = []
    for i in range(0, len(model), n):
        ret.append(Row(model[i:i + n]))
    return ret


def main(config: ChannelMapEditorConfig = None):
    if config is None:
        config = parse_cli()

    run_server(ChannelMapEditorApp(config),
               no_open_browser=config.no_open_browser)


if __name__ == '__main__':
    main()
