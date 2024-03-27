import re
from pathlib import Path
from typing import cast, Literal, TypedDict

from bokeh.models import UIElement, Select, TextInput
from typing_extensions import Required

from neurocarto.config import CartoConfig
from neurocarto.probe import ElectrodeDesp
from neurocarto.util.bokeh_util import as_callback, ButtonFactory, new_help_button
from neurocarto.views.base import ViewBase, DynamicView, InvisibleView, EditorView, RecordView, RecordStep
from .desp import NpxProbeDesp
from .npx import ChannelMap, ProbeType, ReferenceInfo

__all__ = ['NpxReferenceControl', 'NpxBadElectrodesView']


class NpxReferenceControl(ViewBase, DynamicView, RecordView[int]):

    def __init__(self, config: CartoConfig):
        super().__init__(config, logger='neurocarto.view.npx_ref')

        self._chmap: ChannelMap | None = None
        self._probe: ProbeType | None = None
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
        if isinstance(probe, NpxProbeDesp) and chmap is not None:
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
        else:
            self._chmap = None

    def on_reference_select(self, item: str):
        try:
            ref = self._references[item]
        except KeyError:
            return

        if (chmap := self._chmap) is not None:
            self.log_message(f'set reference({item})')
            chmap.reference = ref.code

            self.add_record(ref.code, 'reference', f'set reference to {item}')

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

    # ============= #
    # record replay #
    # ============= #

    def replay_record(self, record: RecordStep):
        self.logger.debug('replay %s', record.description)

        ref = int(record.record)
        self.reference_select.value = self.reference_select.options[ref]


class NpxBadElectrodesAction(TypedDict, total=False):
    action: Required[str]  # save,load,set

    # action=save,load,set?
    serial_number: str


class NpxBadElectrodesView(ViewBase, InvisibleView, EditorView, RecordView[NpxBadElectrodesAction]):
    def __init__(self, config: CartoConfig):
        super().__init__(config, logger='neurocarto.view.npx_bad')

        self._config = config
        self._bad_electrodes: dict[str, list[int]] = {}
        self._serial_number: str | None = None
        self._blueprint: list[ElectrodeDesp] | None = None

    @property
    def name(self) -> str:
        return 'Neuropixels Bad Electrodes'

    # ================== #
    # Database save/load #
    # ================== #

    def get_bad_electrodes_file(self) -> Path:
        from neurocarto.files import user_data_dir
        return user_data_dir(self._config) / 'npx_bad_electrodes.json'

    def load_bad_electrodes(self) -> dict[str, list[int]]:
        file = self.get_bad_electrodes_file()
        if not file.exists():
            self.logger.debug('file not found %s', file)
        else:
            import json
            self.logger.debug('load file %s', file)
            try:
                with file.open() as f:
                    data = json.load(f)
            except json.JSONDecodeError as e:
                self.logger.warning('load file fail %s', file, exc_info=e)
            else:
                self._bad_electrodes.update(data)

        return self._bad_electrodes

    def check_serial_number(self, serial_number: str) -> Literal['not-exist', 'bad-format', None]:
        if not re.match(r'\d{11}', serial_number):
            return 'bad-format'

        if len(self._bad_electrodes) == 0:
            self.load_bad_electrodes()

        if serial_number not in self._bad_electrodes:
            return 'not-exist'

    def get_bad_electrodes(self, serial_number: str) -> list[int]:
        if len(self._bad_electrodes) == 0:
            self.load_bad_electrodes()

        return self._bad_electrodes.get(serial_number, [])

    def save_bad_electrodes(self, serial_number: str, electrodes: list[int]):
        if len(self._bad_electrodes) == 0:
            self.load_bad_electrodes()

        self._bad_electrodes[serial_number] = electrodes

        file = self.get_bad_electrodes_file()
        file.parent.mkdir(parents=True, exist_ok=True)

        import json
        self.logger.debug('save file %s', file)
        with file.open('w') as f:
            json.dump(self._bad_electrodes, f)

    # ============= #
    # UI components #
    # ============= #

    serial_number_input: TextInput

    def _setup_content(self, **kwargs):
        new_btn = ButtonFactory(min_width=50, width_policy='min')

        self.serial_number_input = TextInput(max_length=11)
        self.serial_number_input.on_change('value', as_callback(self._on_serial_number))

        from bokeh.layouts import row
        return [
            row(
                self.serial_number_input, new_help_button('serial number. 11 digit numbers'),
                new_btn('Check', self._on_check), new_help_button('Check current channelmap has use any bad electrodes'),
                new_btn('Load', self._on_load), new_help_button('Load bad electrodes and set as excluded electrodes'),
                new_btn('Save', self._on_save), new_help_button('Save excluded electrodes as bad electrodes'),
            )
        ]

    def _on_check(self):
        if len(serial_number := self.serial_number_input.value_input) == 0:
            self.set_status('missing serial number', decay=5)
            return

        if (blueprint := self._blueprint) is None:
            self.set_status('missing probe', decay=5)
            return

        if len(bad_electrodes := self.get_bad_electrodes(serial_number)) == 0:
            self.set_status('no bad electrode is used', decay=10)
            return

        used_electrodes = [
            i for i, e in enumerate(blueprint)
            if e.state == NpxProbeDesp.STATE_USED and i in bad_electrodes
        ]
        if len(used_electrodes) == 0:
            self.set_status('no bad electrode is used', decay=10)
        else:
            self.set_status(f'{len(used_electrodes)} bad electrodes are used', decay=10)

    def _on_load(self):
        if len(serial_number := self.serial_number_input.value_input) == 0:
            self.set_status('missing serial number', decay=5)
            return

        if (blueprint := self._blueprint) is None:
            self.set_status('missing probe', decay=5)
            return

        if len(bad_electrodes := self.get_bad_electrodes(serial_number)) > 0:
            self.logger.debug('set %d bad electrodes as excluded electrodes', len(bad_electrodes))
            for electrode in bad_electrodes:
                blueprint[electrode].category = NpxProbeDesp.CATE_EXCLUDED

            self.set_status('loaded', decay=5)
            self.update_probe()

            self.add_record(NpxBadElectrodesAction(action='load', serial_number=serial_number),
                            'bad', 'set bad electrodes as excluded')
        else:
            self.logger.debug('nothing to set')

    def _on_save(self):
        if len(serial_number := self.serial_number_input.value_input) == 0:
            self.set_status('missing serial number', decay=5)
            return

        if (blueprint := self._blueprint) is None:
            self.set_status('missing probe', decay=5)
            return

        bad_electrodes = [i for i, e in enumerate(blueprint) if e.category == NpxProbeDesp.CATE_EXCLUDED]
        self.logger.debug('get %d excluded electrodes as bad electrodes', len(bad_electrodes))
        self.save_bad_electrodes(serial_number, bad_electrodes)
        self.set_status('saved', decay=5)

        self.add_record(NpxBadElectrodesAction(action='save', serial_number=serial_number),
                        'bad', 'save excluded as bad electrodes')

    def _on_serial_number(self, serial_number: str):
        if len(serial_number) == 0:
            self.add_record(NpxBadElectrodesAction(action='set'),
                            'set', 'clear probe serial number')

            return

        match self.check_serial_number(serial_number):
            case None:
                self.set_status('acknowledged', decay=5)
                self.add_record(NpxBadElectrodesAction(action='set', serial_number=serial_number),
                                'set', 'set probe serial number')
            case 'not-exist':
                self.set_status('No bad electrodes data')
            case 'bad-format':
                self.set_status('Not a correct serial number')

    # ======== #
    # updating #
    # ======== #

    def on_probe_update(self, probe, chmap, electrodes):
        if isinstance(chmap, ChannelMap):
            self._blueprint = electrodes
            if (meta := chmap.meta) is not None:
                match meta:
                    case {'serial_number': serial_number} if len(serial_number) > 0:
                        self.serial_number_input.value = serial_number
        else:
            self.serial_number_input.value = ""
            self._blueprint = None

    # ============= #
    # record replay #
    # ============= #

    def replay_record(self, record: RecordStep):
        self.logger.debug('replay %s', record.description)

        serial_number = record.record.get('serial_number', None)
        match record.record:
            case {'action': 'load'} if serial_number is not None:
                self.serial_number_input.value_input = serial_number
                self._on_load()
            case {'action': 'save'} if serial_number is not None:
                self.serial_number_input.value_input = serial_number
                self._on_save()
            case {'action': 'set'}:
                if serial_number is not None:
                    self.serial_number_input.value_input = serial_number
                else:
                    self.serial_number_input.value_input = ''
