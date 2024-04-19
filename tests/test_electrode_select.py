import unittest
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from neurocarto.probe_npx import NpxProbeDesp, ChannelMap, NpxElectrodeDesp
from neurocarto.probe_npx.select import load_select
from neurocarto.util.debug import Profiler
from neurocarto.util.util_blueprint import BlueprintFunctions

if (res := Path('res')).exists():
    RES = res
elif (res := Path('../res')).exists():
    RES = res
else:
    raise RuntimeError()


class ElectrodeSelectorBenchmark(unittest.TestCase):
    channelmap_file: Path

    PROBE: NpxProbeDesp
    CHANNELMAP: ChannelMap
    BLUEPRINT: NDArray[np.int_]
    SAMPLE_TIMES = 100

    @classmethod
    def setUpClass(cls):
        cls.channelmap_file = RES / 'Fig3_example.imro'
        cls.PROBE = NpxProbeDesp()
        cls.CHANNELMAP = cls.PROBE.load_from_file(cls.channelmap_file)
        bp = BlueprintFunctions(cls.PROBE, cls.CHANNELMAP)
        cls.BLUEPRINT = bp.load_blueprint(RES / 'Fig3_example.blueprint.npy')

    bp: BlueprintFunctions
    profile: Profiler | None

    def setUp(self):
        self.bp = BlueprintFunctions(self.PROBE, self.CHANNELMAP)
        self.bp.set_blueprint(self.BLUEPRINT)
        self.profile = None

    def tearDown(self):
        if (profile := self.profile) is not None and profile.enable:
            print(f'{profile.file} use {profile.duration:.2f} sec')

            if (e := profile.run_command()) is not None:
                print(repr(e))

            if (e := profile.exception) is not None:
                self.fail(repr(e))

    def profile_selector(self, channelmap: ChannelMap, electrodes: list[NpxElectrodeDesp],
                         selector: str, module_path: str = None, **kwargs) -> ChannelMap:
        if module_path is None:
            module_path = selector

        _selector = load_select(module_path)
        probe = self.PROBE

        with Profiler(f'bm-electrode-selector-{selector}.png', capture_exception=True) as profile:
            for _ in range(self.SAMPLE_TIMES):
                channelmap = _selector(probe, channelmap, electrodes, **kwargs)

        self.profile = profile
        return channelmap


def make_selector_test(selector: str, module_path: str = None, suffix: str = '', **kwargs):
    def test_selector(self):
        channelmap = self.PROBE.copy_channelmap(self.CHANNELMAP)
        electrodes = self.bp.apply_blueprint()
        self.profile_selector(channelmap, electrodes, selector, module_path, **kwargs)

    setattr(ElectrodeSelectorBenchmark, f'test_selector_{selector}{suffix}', test_selector)


make_selector_test('default')
make_selector_test('weaker')

if __name__ == '__main__':
    unittest.main()
