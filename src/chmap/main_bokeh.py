import functools
import logging
from pathlib import Path
from typing import Any

import numpy as np
from bokeh.events import MenuItemClick
from bokeh.layouts import row as Row, column as Column
from bokeh.models import Div, Select, AutocompleteInput, Toggle, Dropdown, tools, TextAreaInput, UIElement
from bokeh.plotting import figure as Figure

from chmap.config import ChannelMapEditorConfig, parse_cli
from chmap.probe import get_probe_desp, ProbeDesp, M
from chmap.util.bokeh_app import BokehApplication, run_server, run_later
from chmap.util.bokeh_util import ButtonFactory, col_layout, as_callback
from chmap.views.base import ViewBase, StateView, DynamicView, init_view
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

    right_panel_views_config: dict[str, Any] = {}
    """view configuration"""

    def __init__(self, config: ChannelMapEditorConfig):
        super().__init__(logger='chmap.editor')
        self.config = config

        self.logger.debug('get get_probe_desp(%s)', config.probe_family)
        self.probe = get_probe_desp(config.probe_family)()
        self.logger.debug('get get_probe_desp() -> %s', type(self.probe).__name__)

    @property
    def title(self) -> str:
        f = self.config.probe_family.upper()
        return f"{f} Channel Map Editor"

    # ==================== #
    # load/save imro files #
    # ==================== #

    def list_chmap_files(self) -> list[Path]:
        """
        List channelmap files under chmap-dir (--chmap-dir).

        :return: list of files
        """
        pattern = '*' + self.probe.channelmap_file_suffix
        return list(sorted(self.config.chmap_root.glob(pattern), key=Path.name.__get__))

    def get_chmap_file(self, name: str) -> Path:
        """
        Get channelmap file path.

        :param name: filename. it should be the filename under chmap-dir.
            Otherwise, it needs to contain '/' to load from/save into other place.
        :return: saving path.
        """
        if '/' in name:
            p = Path(name)
        else:
            p = self.config.chmap_root / name

        return p.with_suffix(self.probe.channelmap_file_suffix)

    def load_chmap(self, name: str) -> M:
        """
        Load channelmap from file with *name*.

        :param name: filename. See get_chmap_file(filename) for more details.
        :return: channelmap instance.
        """
        file = self.get_chmap_file(name)
        ret = self.probe.load_from_file(file)
        self.log_message(f'load channelmap : {file.name}')
        return ret

    def save_chmap(self, name: str, chmap: M) -> Path:
        """
        Save channelmap into file.

        :param name: filename. See get_chmap_file(filename) for more details.
        :param chmap: channelmap instance.
        :return: saved path.
        """
        file = self.get_chmap_file(name)
        self.probe.save_to_file(chmap, file)
        self.log_message(f'save channelmap : {file.name}')
        return file

    def get_policy_file(self, chmap: str | Path) -> Path:
        """
        Get corresponded policy matrix saving path.

        :param chmap: filename. See get_chmap_file(filename) for more details.
        :return: saving path.
        """
        if isinstance(chmap, str):
            imro_file = self.get_chmap_file(chmap)
        else:
            imro_file = chmap
        return imro_file.with_suffix('.policy.npy')

    def load_policy(self, chmap: str | Path) -> bool:
        """
        Load channelmap *chmap* corresponded policy matrix.

        :param chmap: filename. See get_policy_file(filename) for more details.
        :return: True on success.
        """
        if (electrodes := self.probe_view.electrodes) is None:
            return False

        file = self.get_policy_file(chmap)

        try:
            data = np.load(file)
        except FileNotFoundError as e:
            self.log_message(f'File not found : {file}')
            self.logger.warning(f'policy file not found : %s', file, exc_info=e)
            return False
        else:
            self.log_message(f'load policy : {file.name}')

        self.probe.electrode_from_numpy(electrodes, data)
        return True

    def save_policy(self, chmap: str | Path) -> Path | None:
        """
       Save channelmap *chmap* corresponded policy matrix.

       :param chmap: filename. See get_policy_file(filename) for more details.
       :return: saving path. None on failure.
       """
        if (electrodes := self.probe_view.electrodes) is None:
            return None

        file = self.get_policy_file(chmap)
        data = self.probe.electrode_to_numpy(electrodes)

        np.save(file, data)
        self.log_message(f'save policy : {file.name}')

        return file

    def get_view_config_file(self, chmap: str | Path) -> Path:
        """
        Get view components' configurations saving path.

        :param chmap: filename. See get_chmap_file(filename) for more details.
        :return: saving path.
        """
        if isinstance(chmap, str):
            imro_file = self.get_chmap_file(chmap)
        else:
            imro_file = chmap

        return imro_file.with_suffix('.config.json')

    def load_view_config(self, chmap: str | Path, *, reset=False) -> dict[str, Any]:
        """
        Load view components' configurations.

        :param chmap: filename. See get_view_config_file(filename) for more details.
        :param reset: reset right_view_config or update.
        :return: configuration dict {type_name: Any}.
        """
        file = self.get_view_config_file(chmap)
        if not file.exists():
            self.log_message(f'File not found : {file}')
            return self.right_panel_views_config

        import json
        with file.open('r') as f:
            data = dict(json.load(f))
            self.log_message(f'load image config : {file.name}')

        if reset:
            self.right_panel_views_config.update(data)
        else:
            self.right_panel_views_config = data

        return self.right_panel_views_config

    def save_view_config(self, chmap: str | Path, *, direct=False):
        """
        Save view components' configurations.

        :param chmap: filename. See get_view_config_file(filename) for more details.
        :param direct: Do not update configurations from view components.
        """
        if not direct:
            for view in self.right_panel_views:
                if isinstance(view, StateView):
                    self.logger.debug('on_save() config %s', type(view).__name__)
                    self.right_panel_views_config[type(view).__name__] = view.save_state()

        import json
        file = self.get_view_config_file(chmap)
        with file.open('w') as f:
            json.dump(self.right_panel_views_config, f, indent=2)
            self.log_message(f'save image config : {file.name}')

    # ============= #
    # UI components #
    # ============= #

    probe_fig: Figure
    probe_view: ProbeView
    probe_info: Div

    right_panel_views: list[ViewBase]

    input_imro: Select
    output_imro: AutocompleteInput
    message_area: TextAreaInput

    auto_btn: Toggle

    def index(self):
        self.logger.debug('index')

        # index_middle_panel
        self.logger.debug('index figure')
        self.probe_info = Div(text="<b>Probe</b>")
        self.probe_fig = Figure(width=600, height=800, tools='', toolbar_location='above')
        self.probe_view = ProbeView(self.probe)
        self.probe_view.plot(self.probe_fig)

        # figure toolbar
        self.logger.debug('index figure toolbar')
        self.probe_fig.tools.clear()
        self.probe_fig.add_tools(
            tools.ResetTool(),
            (t_drag := tools.PanTool(dimensions='both')),
            *self.probe_view.setup_tools(),
            #
            tools.WheelPanTool(dimension='height'),
            (t_scroll := tools.WheelZoomTool(dimensions='both')),
        )
        self.probe_fig.toolbar.active_drag = t_drag
        self.probe_fig.toolbar.active_scroll = t_scroll

        return Row(
            Column(self._index_left_panel()),
            Column(self.probe_info, self.probe_fig),
            Column(self._index_right_panel())
        )

    def _index_left_panel(self) -> list[UIElement]:
        self.logger.debug('index left')
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

        # electrode states buttons
        state_btns = col_layout([
            new_btn(name, functools.partial(self.on_state_change, state=value))
            for name, value in self.probe.possible_states.items()
        ], 2)

        # electrode selecting policy buttons
        policy_btns = col_layout([
            new_btn(name, functools.partial(self.on_policy_change, policy=value))
            for name, value in self.probe.possible_policies.items()
        ], 2)

        # channelmap IO buttons
        empty_btn = Dropdown(
            label='New',
            menu=list(self.probe.supported_type),
            min_width=100, align='end', width_policy='min',
            stylesheets=["div.bk-menu { width: 300%; }"]
        )
        empty_btn.on_click(self.on_new)

        load_btn = new_btn('Load', self.on_load, min_width=100, align='end')
        save_btn = new_btn('Save', self.on_save, min_width=100, align='end')

        # electrode selecting helping buttons
        refresh_btn = new_btn('Refresh', self.on_refresh)
        self.auto_btn = Toggle(label='Auto', active=True, min_width=150, width_policy='min')
        self.auto_btn.on_change('active', as_callback(self.on_autoupdate))

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

    def _index_right_panel(self) -> list[UIElement]:
        self.logger.debug('index right')
        self.right_panel_views = []

        for view_type in self.install_right_panel_views(self.config):
            if (view := init_view(self.config, view_type)) is not None:
                self.right_panel_views.append(view)

        uis = []
        for view in self.right_panel_views:
            uis.extend(view.setup(self.probe_fig))

        return uis

    def install_right_panel_views(self, config: ChannelMapEditorConfig) -> list:
        """
        :param config:
        :return: list of item that recognised by `init_view`
        """
        ret = []
        ret.extend(self.probe.extra_controls(config))
        ret.extend(config.extra_view)
        ret.append('chmap.views.atlas:AtlasBrainView')
        return ret

    def start(self):
        self.logger.debug('start')
        self.reload_input_imro_list()
        for view in self.right_panel_views:
            self.logger.debug('start %s', type(view).__name__)
            view.start()

    # ========= #
    # callbacks #
    # ========= #

    def on_new(self, e: MenuItemClick):
        if (item := e.item) is None:
            self.logger.debug('on_new()')
            return

        probe_type = self.probe.supported_type[item]
        self.logger.debug('on_new(%d)=%s', probe_type, item)

        self.probe_view.reset(probe_type)

        if len(self.output_imro.value_input) == 0:
            self.output_imro.value = "New"

        self.on_probe_update()

    def on_load(self):
        name: str
        if len(name := self.input_imro.value) == 0:
            self.logger.debug('on_load()')
            return

        self.logger.debug('on_load(%s)', name)
        file = self.get_chmap_file(name)
        try:
            chmap = self.load_chmap(name)
        except FileNotFoundError as x:
            self.log_message(f'File not found : {file}')
            self.logger.warning(f'channelmap file not found : %s', file, exc_info=x)
            return

        self.output_imro.value = file.stem

        self.probe_view.reset(chmap)
        self.load_policy(file)

        config = self.load_view_config(file, reset=True)
        for view in self.right_panel_views:
            if isinstance(view, StateView):
                try:
                    _config = config[type(view).__name__]
                except KeyError:
                    pass
                else:
                    self.logger.debug('on_load() config %s', type(view).__name__)
                    try:
                        view.restore_state(_config)
                    except RuntimeError as e:
                        self.log_message(type(view).__name__ + '::' + repr(e))

        self.on_probe_update()

    def reload_input_imro_list(self, preselect: str = None):
        if preselect is None:
            preselect = self.input_imro.value

        imro_list = [f.stem for f in self.list_chmap_files()]
        self.input_imro.options = [""] + imro_list

        if preselect in imro_list:
            self.input_imro.value = preselect
        else:
            self.input_imro.value = ""

        self.output_imro.completions = imro_list

    def on_save(self):
        name = self.output_imro.value
        if len(name) == 0:
            self.logger.debug('on_save()')
            self.log_message('empty output filename')
            return

        self.logger.debug('on_save(%s)', name)
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
        for desp, code in self.probe.possible_states.items():
            if code == state:
                break
        else:
            desp = None

        if desp is not None:
            self.logger.debug('on_state_change(%d)=%s', state, desp)
        else:
            self.logger.debug('on_state_change(%d)', state)

        try:
            self.probe_view.set_state_for_selected(state)
        except BaseException:
            self.log_message(f'set state {desp} fail')
        else:
            self.on_probe_update()

    def on_policy_change(self, policy: int):
        for desp, code in self.probe.possible_policies.items():
            if code == policy:
                break
        else:
            desp = None

        if desp is not None:
            self.logger.debug('on_policy_change(%d)=%s', policy, desp)
        else:
            self.logger.debug('on_policy_change(%d)', policy)

        try:
            self.probe_view.set_policy_for_selected(policy)
        except BaseException:
            self.log_message(f'set policy {desp} fail')
            return

        if self.auto_btn.active:
            self.on_refresh()
        else:
            self.on_probe_update()

    def on_probe_update(self):
        self.probe_info.text = self.probe_view.channelmap_desp()
        self.probe_view.update_electrode()

        for view in self.right_panel_views:
            if isinstance(view, DynamicView):
                run_later(view.on_probe_update, self.probe, self.probe_view.channelmap, self.probe_view.electrodes)

    def on_autoupdate(self, active: bool):
        self.logger.debug('on_autoupdate(active=%s)', active)
        if active:
            self.on_refresh()

    def on_refresh(self):
        try:
            self.probe_view.refresh_selection()
        except BaseException:
            self.log_message('refresh fail')
        else:
            self.probe_view.update_electrode()
            self.on_probe_update()

    def log_message(self, *message, reset=False):
        area = self.message_area
        message = '\n'.join(message)
        self.logger.info(message)
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
    """Start channelmap editor application."""
    if config is None:
        config = parse_cli()

    logging.basicConfig(
        format='[%(levelname)s] %(name)s - %(message)s'
    )

    if config.debug:
        logging.getLogger('chmap').setLevel(logging.DEBUG)

    run_server(ChannelMapEditorApp(config),
               no_open_browser=config.no_open_browser)


if __name__ == '__main__':
    main()
