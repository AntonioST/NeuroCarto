import functools
import os
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from bokeh.application.application import SessionContext
from bokeh.events import MenuItemClick
from bokeh.io import curdoc
from bokeh.models import Div, Select, AutocompleteInput, Toggle, Dropdown, tools, TextAreaInput, UIElement
from bokeh.plotting import figure as Figure
from bokeh.themes import Theme

from chmap.config import ChannelMapEditorConfig, parse_cli, setup_logger
from chmap.probe import get_probe_desp, ProbeDesp, M
from chmap.util.bokeh_app import BokehApplication, run_server, run_later
from chmap.util.bokeh_util import ButtonFactory, col_layout, as_callback, new_help_button
from chmap.util.utils import TimeMarker, doc_link
from chmap.views import *

__all__ = ['ChannelMapEditorApp', 'main']


class ChannelMapEditorAppConfig(TypedDict, total=False):
    theme: str | dict[str, Any] | None
    views: list[str]


class ChannelMapEditorApp(BokehApplication):
    """Application of neural probe channelmap editor.

    The layout of this application is split into:

    * left: channelmap manipulating button groups.
    * center: figure that contains probe/electrodes, image and curves.
    * right: controls of image, curves that shown in center panel.

    """

    probe: ProbeDesp[M, Any]
    """probe describer"""

    global_views_config: dict[str, Any] = {}
    """view configuration"""

    right_panel_views_config: dict[str, Any] = {}
    """view configuration, channelmap depended"""

    def __init__(self, config: ChannelMapEditorConfig, *, logger: str = 'chmap.editor'):
        super().__init__(logger=logger)
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

    def global_config_file(self) -> Path:
        """
        Get global (user) config filepath.

        :return: filepath.
        """
        if (ret := self.config.config_file) is not None:
            if ret.is_dir():
                ret = ret / 'chmap.config.json'
            return ret

        if self.config.debug:
            return Path('.') / '.chmap.config.json'

        # https://stackoverflow.com/a/3250952
        if (d := os.environ.get('XDG_CONFIG_HOME', None)) is not None:
            return Path(d) / 'chmap/chmap.config.json'
        elif (d := os.environ.get('APPDATA', None)) is not None:
            return Path(d) / 'chmap/chmap.config.json'

        # https://stackoverflow.com/a/1857
        import platform
        match platform.system():
            case 'Linux':
                return Path.home() / '.config/chmap/chmap.config.json'
            case 'Windows':
                pass
            case 'Darwin':
                pass

        return Path.home() / '.chmap.config.json'

    @doc_link()
    def load_global_config(self, *, reset=False) -> dict[str, Any]:
        """
        load global (user) config.

        :param reset: reset global config. Otherwise, just update dict.
        :return:
        :see: {#global_config_file()}
        """
        file = self.global_config_file()
        if not file.exists():
            self.logger.debug('global config not found: %s', file)
            return self.global_views_config

        import json
        with file.open('r') as f:
            data = dict(json.load(f))
            self.logger.debug('load global config : %s', file)

        if reset:
            self.global_views_config.update(data)
        else:
            self.global_views_config = data

        return self.global_views_config

    @doc_link()
    def save_global_config(self, direct=False):
        """
        save global (user) config.

        :param direct: Do not update configurations from {GlobalStateView}.
        :see: {#global_config_file()}
        """
        if not direct:
            for view in self.right_panel_views:
                if isinstance(view, GlobalStateView) and (state := view.save_state(local=False)) is not None:
                    self.logger.debug('on_save() config %s', type(view).__name__)
                    self.global_views_config[type(view).__name__] = state

        import json
        file = self.global_config_file()
        file.parent.mkdir(parents=True, exist_ok=True)
        with file.open('w') as f:
            json.dump(self.global_views_config, f, indent=2)
            self.logger.debug(f'save global config : %s', file)

    def get_app_global_config(self) -> ChannelMapEditorAppConfig:
        """
        get app config from global (user) config.
        :return:
        """
        try:
            app_config: ChannelMapEditorAppConfig = self.global_views_config[type(self).__name__]
        except KeyError:
            self.global_views_config[type(self).__name__] = app_config = ChannelMapEditorAppConfig(theme=None, views=[])
            self.save_global_config(direct=True)

        return app_config

    def list_chmap_files(self) -> list[Path]:
        """
        List channelmap files under chmap-dir (`-C`, `--chmap-dir`).

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

    @doc_link()
    def load_chmap(self, name: str | Path) -> M:
        """
        Load channelmap from file with *name*.

        :param name: filename. See {#get_chmap_file()} for more details.
        :return: channelmap instance.
        """
        if isinstance(name, str):
            file = self.get_chmap_file(name)
        else:
            file = name

        ret = self.probe.load_from_file(file)
        self.log_message(f'load channelmap : {file.name}')
        return ret

    @doc_link()
    def save_chmap(self, name: str, chmap: M) -> Path:
        """
        Save channelmap into file.

        :param name: filename. See {#get_chmap_file()} for more details.
        :param chmap: channelmap instance.
        :return: saved path.
        """
        file = self.get_chmap_file(name)
        self.probe.save_to_file(chmap, file)
        self.log_message(f'save channelmap : {file.name}')
        return file

    @doc_link()
    def get_blueprint_file(self, chmap: str | Path) -> Path:
        """
        Get corresponded blueprint from channelmap saving path.

        :param chmap: filename. See {#get_chmap_file()} for more details.
        :return: saving path.
        """
        if isinstance(chmap, str):
            imro_file = self.get_chmap_file(chmap)
        else:
            imro_file = chmap
        return imro_file.with_suffix('.blueprint.npy')

    @doc_link()
    def load_blueprint(self, chmap: str | Path) -> bool:
        """
        Load channelmap *chmap* corresponded blueprint.

        :param chmap: filename. See {#get_blueprint_file()} for more details.
        :return: True on success.
        """
        if (electrodes := self.probe_view.electrodes) is None:
            return False

        file = self.get_blueprint_file(chmap)

        try:
            data = np.load(file)
        except FileNotFoundError as e:
            self.log_message(f'File not found : {file}')
            self.logger.warning(f'blueprint file not found : %s', file, exc_info=e)
            return False
        else:
            self.log_message(f'load blueprint : {file.name}')

        self.probe.load_blueprint(data, electrodes)
        return True

    @doc_link()
    def save_blueprint(self, chmap: str | Path) -> Path | None:
        """
        Save channelmap *chmap* corresponded blueprint.

        :param chmap: filename. See {#get_blueprint_file()} for more details.
        :return: saving path. None on failure.
        """
        if (electrodes := self.probe_view.electrodes) is None:
            return None

        file = self.get_blueprint_file(chmap)
        data = self.probe.save_blueprint(electrodes)

        np.save(file, data)
        self.log_message(f'save blueprint : {file.name}')

        return file

    @doc_link()
    def get_view_config_file(self, chmap: str | Path) -> Path:
        """
        Get view components' configurations saving path.

        :param chmap: filename. See {#get_chmap_file()} for more details.
        :return: saving path.
        """
        if isinstance(chmap, str):
            imro_file = self.get_chmap_file(chmap)
        else:
            imro_file = chmap

        return imro_file.with_suffix('.config.json')

    @doc_link()
    def load_view_config(self, chmap: str | Path, *, reset=False) -> dict[str, Any]:
        """
        Load view components' configurations.

        :param chmap: filename. See {#get_view_config_file()} for more details.
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
            self.log_message(f'load config : {file.name}')

        if reset:
            self.right_panel_views_config.update(data)
        else:
            self.right_panel_views_config = data

        return self.right_panel_views_config

    @doc_link()
    def save_view_config(self, chmap: str | Path, *, direct=False):
        """
        Save view components' configurations.

        :param chmap: filename. See {#get_view_config_file()} for more details.
        :param direct: Do not update configurations from view components.
        """
        if not direct:
            for view in self.right_panel_views:
                if isinstance(view, StateView) and (state := view.save_state()) is not None:
                    self.logger.debug('on_save() config %s', type(view).__name__)
                    self.right_panel_views_config[type(view).__name__] = state

        import json
        file = self.get_view_config_file(chmap)
        with file.open('w') as f:
            json.dump(self.right_panel_views_config, f, indent=2)
            self.log_message(f'save config : {file.name}')

    # ============= #
    # UI components #
    # ============= #

    probe_fig: Figure
    probe_view: ProbeView

    right_panel_views: list[ViewBase]

    input_imro: Select
    output_imro: AutocompleteInput
    message_area: TextAreaInput

    auto_btn: Toggle

    def index(self):
        self.logger.debug('index')
        self.load_global_config(reset=True)

        if (theme := self.get_app_global_config().get('theme', None)) is not None:
            try:
                self._set_theme(theme)
            except BaseException as e:
                self.logger.warning('set theme', exc_info=e)

        # log area goes first
        self.message_area = TextAreaInput(
            rows=10, cols=100, width=290,
            max_length=None, disabled=True
        )

        # middle figure
        self.logger.debug('index figure')
        self.probe_fig = Figure(
            width=600, height=800,
            x_axis_label='(um)', y_axis_label='(um)',
            tools='', toolbar_location='above'
        )
        # figure toolbar
        self.probe_fig.tools.clear()
        self.probe_fig.add_tools(
            (t_drag := tools.PanTool(dimensions='both')),
            tools.WheelPanTool(dimension='height'),
            (t_scroll := tools.WheelZoomTool(dimensions='both')),
            tools.ResetTool(),
        )

        self.probe_fig.toolbar.active_drag = t_drag
        self.probe_fig.toolbar.active_scroll = t_scroll

        # probe
        from chmap.views.utils import install_view
        self.probe_view = install_view(self, ProbeView(self.config, self.probe))
        probe_ui = self.probe_view.setup(self.probe_fig)

        from bokeh.layouts import row, column
        return row(
            column(self._index_left_panel()),
            column(*probe_ui, self.probe_fig),
            column(self._index_right_panel())
        )

    def _set_theme(self, theme: str | Path | dict[str, Any]):
        """

        Json Theme: https://docs.bokeh.org/en/latest/docs/reference/themes.html#theme

        :param theme:
        :return:
        """
        document = curdoc()
        if isinstance(theme, str):
            self.logger.debug('set theme %s', theme)
            if theme.endswith('.json'):
                document.theme = Theme(theme)
            else:
                document.theme = theme
        elif isinstance(theme, (Path, dict)):
            document.theme = Theme(theme)
        else:
            raise TypeError()

    def _index_left_panel(self) -> list[UIElement]:
        # 1/1, 1/2, 1/3, ...
        widths = (290, 140, 90, 30)

        self.logger.debug('index left')
        new_btn = ButtonFactory(min_width=widths[1], width_policy='min')

        #
        self.input_imro = Select(
            title='Input file',
            options=[], value="",
            width=widths[0]
        )

        self.output_imro = AutocompleteInput(
            title='Save filename',
            width=widths[0], max_completions=5, case_sensitive=False, restrict=True
        )

        # electrode states buttons
        state_btns = col_layout([
            new_btn(name, functools.partial(self.on_state_change, state=value))
            for name, value in self.probe.possible_states.items()
        ], 2)

        # electrode selecting category buttons
        category_btns = col_layout([
            new_btn(name, functools.partial(self.on_category_change, category=value))
            for name, value in self.probe.possible_categories.items()
        ], 2)

        # channelmap IO buttons
        empty_btn = Dropdown(
            label='New',
            menu=list(self.probe.supported_type),
            min_width=widths[2], align='end', width_policy='min',
            stylesheets=["div.bk-menu { width: 300%; }"]
        )
        empty_btn.on_click(self.on_new)

        load_btn = new_btn('Load', self.on_load, min_width=widths[2], align='end')
        save_btn = new_btn('Save', self.on_save, min_width=widths[2], align='end')

        # electrode selecting helping buttons
        refresh_btn = new_btn('Refresh', self.on_refresh)
        self.auto_btn = Toggle(label='Auto', active=True, min_width=widths[1], width_policy='min')
        self.auto_btn.on_change('active', as_callback(self.on_autoupdate))

        from bokeh.layouts import row
        return [
            # Channelmap IO
            Div(text="<b>ChannelMap File</b>"),
            self.input_imro,
            row(empty_btn, load_btn, save_btn),
            self.output_imro,
            # Electrode Select
            row(Div(text="<b>Electrode State</b>"),
                new_help_button('Manually select electrodes directly', position='right')),
            *state_btns,
            # Electrode Category
            row(Div(text="<b>Electrode Category</b>"),
                new_help_button('Set electrode category for programming electrode selection', position='right')),
            *category_btns,
            row(self.auto_btn, refresh_btn),
            # log message
            Div(text="<b>Log</b>"),
            self.message_area,
            new_btn('clear', self.log_clear, min_width=widths[3])
        ]

    def _index_right_panel(self) -> list[UIElement]:
        from chmap.views.utils import install_view

        self.logger.debug('index right')
        self.right_panel_views = []

        for view_type in self.install_right_panel_views(self.config):
            if (view := init_view(self.config, view_type)) is not None:
                self.right_panel_views.append(install_view(self, view))

        uis = []
        for view in self.right_panel_views:
            uis.extend(view.setup(self.probe_fig))

        return uis

    @doc_link()
    def install_right_panel_views(self, config: ChannelMapEditorConfig) -> list:
        """
        :param config:
        :return: list of item that recognised by {init_view()}
        """
        ret: list = self.probe.extra_controls(config)

        ext = []

        ext.extend(self.get_app_global_config().get('views', ['blueprint', 'atlas']))
        ext.extend(config.extra_view)

        if '-' in ext:
            ext = ext[ext.index('-') + 1:]

        ret.extend(ext)
        return ret

    def start(self):
        self.logger.debug('start')
        self.reload_input_imro_list()

        self.probe_view.start()

        for view in self.right_panel_views:
            self.logger.debug('start %s', type(view).__name__)
            view.start()

        if (open_file := self.config.open_file) is not None:
            run_later(self.load_file, open_file)

    def cleanup(self, context: SessionContext):
        super().cleanup(context)
        self.save_global_config()

    # ========= #
    # callbacks #
    # ========= #

    def on_new(self, e: int | str | MenuItemClick):
        if isinstance(e, MenuItemClick):
            if (item := e.item) is None:
                self.logger.debug('on_new()')
                return
        else:
            item = e

        if isinstance(item, str):
            probe_type = self.probe.supported_type[item]
            self.logger.debug('on_new(%d)=%s', probe_type, item)
            self.log_message(f'new probe({item})')
        elif isinstance(item, int):
            probe_type = item
            self.logger.debug('on_new(%d)', probe_type)
            self.log_message(f'new probe[{probe_type}]')
        else:
            raise TypeError()

        self.probe_view.reset(probe_type)

        if len(self.output_imro.value_input) == 0:
            self.output_imro.value_input = "New"

        self.on_probe_update()

    def on_load(self):
        if len(name := self.input_imro.value) == 0:
            self.logger.debug('on_load()')
            return

        self.load_file(name)

    def load_file(self, name: str | Path):
        self.logger.debug('on_load(%s)', name)

        file = name

        try:
            if isinstance(name, str):
                file = self.get_chmap_file(name)
                chmap = self.load_chmap(name)
            else:
                chmap = self.load_chmap(file)
        except FileNotFoundError as x:
            self.log_message(f'File not found : {file}')
            self.logger.warning(f'channelmap file not found : %s', file, exc_info=x)
            return

        self.output_imro.value_input = file.stem

        self.probe_view.reset(chmap)
        self.load_blueprint(file)

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
        name = self.output_imro.value_input
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
        self.save_blueprint(path)
        self.save_view_config(path)

        self.output_imro.value_input = path.stem
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
            self.probe_view.set_state_for_captured(state)
        except BaseException:
            self.log_message(f'set state {desp} fail')
        else:
            self.on_probe_update()

    def on_category_change(self, category: int):
        for desp, code in self.probe.possible_categories.items():
            if code == category:
                break
        else:
            desp = None

        if desp is not None:
            self.logger.debug('on_category_change(%d)=%s', category, desp)
        else:
            self.logger.debug('on_category_change(%d)', category)

        try:
            self.probe_view.set_category_for_captured(category)
        except BaseException as e:
            self.log_message(f'set category {desp} fail')
            self.logger.warning('set category %s fail', desp, exc_info=e)
            return

        if self.auto_btn.active:
            self.on_refresh()
        else:
            self.on_probe_update()

    def on_probe_update(self):
        self.probe_view.update_electrode()

        mark = TimeMarker()
        for view in self.right_panel_views:
            if isinstance(view, DynamicView):
                view_class = type(view).__name__

                mark.reset()
                view.on_probe_update(self.probe, self.probe_view.channelmap, self.probe_view.electrodes)
                t = mark()

                self.logger.debug('on_probe_update(%s) used %.2f sec', view_class, t)

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
            self.on_probe_update()

    def log_message(self, *message, reset=False):
        message = '\n'.join(message)
        self.logger.info(message)

        if reset:
            self.message_area.value = message
        else:
            text = self.message_area.value
            self.message_area.value = message + '\n' + text

    def log_clear(self):
        self.message_area.value = ""


def main(config: ChannelMapEditorConfig = None):
    """Start channelmap editor application."""
    if config is None:
        config = parse_cli()

    setup_logger(config)

    run_server(ChannelMapEditorApp(config), config)


if __name__ == '__main__':
    main()
