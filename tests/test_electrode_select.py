import cProfile
import contextlib
import subprocess
import time
import unittest
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from neurocarto.probe_npx import NpxProbeDesp, ChannelMap, NpxElectrodeDesp
from neurocarto.probe_npx.select import load_select
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

    profile: cProfile.Profile
    bp: BlueprintFunctions

    def setUp(self):
        self.profile = cProfile.Profile(time.perf_counter)
        self.bp = BlueprintFunctions(self.PROBE, self.CHANNELMAP)
        self.bp.set_blueprint(self.BLUEPRINT)

    def save_profile_data(self, filename: str):
        dat_file = Path(f'{filename}.dat')
        print(dat_file)
        self.profile.dump_stats(dat_file)

        png_file = Path(f'{filename}.png')
        print(png_file)

        command = f'python -m gprof2dot -f pstats {dat_file} | dot -T png -o {png_file}'
        try:
            subprocess.run(command, shell=True)
        except:
            print(command)

    @contextlib.contextmanager
    def profile_context(self, filename: str):
        t = time.time()
        self.profile.enable()
        try:
            yield
        finally:
            self.profile.disable()
            t = time.time() - t
            self.save_profile_data(filename)

        print(f'{filename} use {t:.2f} sec')

    def profile_selector(self, channelmap: ChannelMap, electrodes: list[NpxElectrodeDesp],
                         selector: str, **kwargs) -> ChannelMap:

        _selector = load_select(selector)
        probe = self.PROBE

        with self.profile_context(f'bm-electrode-selector-{selector}'):
            for _ in range(self.SAMPLE_TIMES):
                channelmap = _selector(probe, channelmap, electrodes, **kwargs)
        return channelmap

    def test_channelmap_copy(self):
        probe = self.PROBE
        channelmap = self.CHANNELMAP
        with self.profile_context(f'bm-electrode-add'):
            for _ in range(self.SAMPLE_TIMES):
                probe.copy_channelmap(channelmap)

    def test_selector_default(self):
        channelmap = self.PROBE.copy_channelmap(self.CHANNELMAP)
        electrodes = self.bp.apply_blueprint()
        self.profile_selector(channelmap, electrodes, 'default')

    def test_selector_weaker(self):
        channelmap = self.PROBE.copy_channelmap(self.CHANNELMAP)
        electrodes = self.bp.apply_blueprint()
        self.profile_selector(channelmap, electrodes, 'weaker')


if __name__ == '__main__':
    unittest.main()
