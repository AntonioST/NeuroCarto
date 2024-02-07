from typing import Any

from chmap.config import parse_cli
from chmap.probe import ProbeDesp, ElectrodeDesp
from chmap.probe_npx import ChannelMap
from chmap.util.bokeh_util import ButtonFactory
from chmap.views.base import ViewBase, EditorView

__all__ = ['InitializeBlueprintView']


class InitializeBlueprintView(ViewBase, EditorView):

    @property
    def name(self) -> str:
        return 'Initialize Blueprint'

    # ============= #
    # UI components #
    # ============= #

    def _setup_content(self, **kwargs):
        btn = ButtonFactory(min_width=50, width_policy='min')
        from bokeh.layouts import row
        return [
            row(btn('clear', self.clear_blueprint))
        ]

    # ================ #
    # updating methods #
    # ================ #

    cache_probe: ProbeDesp
    cache_chmap: Any
    cache_blueprint: list[ElectrodeDesp]

    def on_probe_update(self, probe: ProbeDesp, chmap, e):
        self.cache_probe = probe
        self.cache_chmap = chmap
        self.cache_blueprint = e

    def clear_blueprint(self):
        if (blueprint := self.cache_blueprint) is None:
            return

        for e in blueprint:
            e.policy = ProbeDesp.POLICY_UNSET

        self.update_probe()
        self.log_message('clear blueprint')

    def initialize_blueprint(self):
        if self.cache_blueprint is None:
            return
        if isinstance(self.cache_chmap, ChannelMap):
            pass


if __name__ == '__main__':
    import sys

    from chmap.main_bokeh import main

    main(parse_cli([
        *sys.argv[1:],
        '-C', 'res',
        '--debug',
        '--view=-',
        '--view=blueprint',
        '--view=chmap.views.edit_blueprint:InitializeBlueprintView',
    ]))
