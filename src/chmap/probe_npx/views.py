from typing import cast

from bokeh.models import UIElement, Select

from chmap.config import ChannelMapEditorConfig
from chmap.util.bokeh_util import as_callback
from chmap.views.base import ViewBase, DynamicView
from .desp import NpxProbeDesp
from .npx import ChannelMap, ProbeType, ReferenceInfo

__all__ = ['NpxReferenceControl']


class NpxReferenceControl(ViewBase, DynamicView):

    def __init__(self, config: ChannelMapEditorConfig):
        super().__init__(config, logger='chmap.view.npx_ref')

        self._probe: ProbeType | None = None
        self._chmap: ChannelMap | None = None
        self._references: dict[str, ReferenceInfo] = {}

    @property
    def name(self) -> str:
        return 'Neuropixels References'

    @property
    def description(self) -> str | None:
        return 'set reference site'

    reference_select: Select

    def _setup_title(self, **kwargs) -> list[UIElement]:
        ret = super()._setup_title(**kwargs)

        self.reference_select = Select(
            options=[], value='', width=100,
        )
        self.reference_select.on_change('value', as_callback(self.on_reference_select))
        ret.insert(-1, self.reference_select)

        return ret

    def on_probe_update(self, probe, chmap, electrodes):
        if isinstance(probe, NpxProbeDesp):
            if chmap is None:
                self._chmap = None
            else:
                chmap = cast(ChannelMap, chmap)
                update = self._probe is None or self._probe != chmap.probe_type

                self._probe = chmap.probe_type
                self._chmap = None  # avoid reference set during on_reference_select()

                if update:
                    self.update_reference_select_options()

                code = chmap.reference
                desp = [desp for desp, ref in self._references.items() if ref.code == code]
                if len(desp) > 0:
                    self.reference_select.value = desp[0]

                self._chmap = chmap

    def on_reference_select(self, item: str):
        try:
            ref = self._references[item]
        except KeyError:
            return

        if (chmap := self._chmap) is not None:
            self.log_message(f'set reference({item})')
            chmap.reference = ref.code

    def update_reference_select_options(self):
        if (probe := self._probe) is None:
            return

        self.logger.debug('update reference select(%d)', probe.code)
        self._references = {
            self.repr_reference_info(ref := ReferenceInfo.of(probe, code)): ref
            for code in range(ReferenceInfo.max_reference_value(probe))
        }

        self.reference_select.options = list(self._references)

    @classmethod
    def repr_reference_info(cls, ref: ReferenceInfo) -> str:
        match ref.type:
            case 'ext':
                return 'Ext'
            case 'tip':
                return f'Tip:{ref.shank}'
            case 'on-shank':
                return f'Int:({ref.shank}, {ref.code})'
            case _:
                return 'unknown'
