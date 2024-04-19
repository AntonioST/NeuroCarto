import unittest
from pathlib import Path

import numpy as np
from numpy.testing import assert_array_equal

from neurocarto.probe_npx import NpxProbeDesp, ChannelMap, ProbeType
from neurocarto.util.debug import Profiler
from neurocarto.util.util_blueprint import BlueprintFunctions

if (res := Path('res')).exists():
    RES = res
elif (res := Path('../res')).exists():
    RES = res
else:
    raise RuntimeError()


# noinspection PyMethodMayBeStatic
class ProbeChannelElectrodeMappingTest(unittest.TestCase):
    def test_np1(self):
        from neurocarto.probe_npx.npx import PROBE_TYPE_NP1, e2c0, c2e0
        t = PROBE_TYPE_NP1
        self.assertEqual(1, t.n_shank)

        e0 = np.arange(t.n_electrode_shank)
        c, b = e2c0(e0)
        e1 = c2e0(b, c)
        assert_array_equal(e0, e1)

        e2 = []
        for e in e0:
            c, b = e2c0(e)
            e2.append(c2e0(b, c))
        assert_array_equal(e0, np.array(e2))

    def test_np21(self):
        from neurocarto.probe_npx.npx import PROBE_TYPE_NP21, e2c21, c2e21
        t = PROBE_TYPE_NP21
        self.assertEqual(1, t.n_shank)

        e0 = np.arange(t.n_electrode_shank)
        c, b = e2c21(e0)
        e1 = c2e21(b, c)
        assert_array_equal(e0, e1)

        e2 = []
        for e in e0:
            c, b = e2c21(e)
            e2.append(c2e21(b, c))
        assert_array_equal(e0, np.array(e2))

    def test_np24(self):
        from neurocarto.probe_npx.npx import PROBE_TYPE_NP24, e2c24, c2e24
        t = PROBE_TYPE_NP24

        e0 = np.arange(t.n_electrode_shank)

        for s in range(t.n_shank):
            s = np.full_like(e0, s)
            c, b = e2c24(s, e0)
            e1 = c2e24(s, b, c)
            assert_array_equal(e0, e1)

        for s in range(t.n_shank):
            e2 = []
            for e in e0:
                c, b = e2c24(s, e)
                e2.append(c2e24(s, b, c))
            assert_array_equal(e0, np.array(e2))

    def test_np1_electrode_coordinate(self):
        from neurocarto.probe_npx.desp import NpxProbeDesp
        from neurocarto.probe_npx.npx import PROBE_TYPE_NP1, ChannelMap
        from neurocarto.probe_npx.plot import electrode_coordinate

        D = NpxProbeDesp()
        E = D.all_electrodes(ChannelMap(PROBE_TYPE_NP1))
        e0 = np.array([it.electrode for it in E])  # Array[E, (S,C,R)]
        e1 = electrode_coordinate(PROBE_TYPE_NP1, 'cr')
        assert_array_equal(e0, e1)

        e0 = np.array([(it.x, it.y) for it in E])  # Array[E, (X,Y)]
        e1 = electrode_coordinate(PROBE_TYPE_NP1, 'xy')
        assert_array_equal(e0, e1)

    def test_np21_electrode_coordinate(self):
        from neurocarto.probe_npx.desp import NpxProbeDesp
        from neurocarto.probe_npx.npx import PROBE_TYPE_NP21, ChannelMap
        from neurocarto.probe_npx.plot import electrode_coordinate

        D = NpxProbeDesp()
        E = D.all_electrodes(ChannelMap(PROBE_TYPE_NP21))
        e0 = np.array([it.electrode for it in E])  # Array[E, (S,C,R)]
        e1 = electrode_coordinate(PROBE_TYPE_NP21, 'cr')
        assert_array_equal(e0, e1)

        e0 = np.array([(it.x, it.y) for it in E])  # Array[E, (X,Y)]
        e1 = electrode_coordinate(PROBE_TYPE_NP21, 'xy')
        assert_array_equal(e0, e1)

    def test_np24_electrode_coordinate(self):
        from neurocarto.probe_npx.desp import NpxProbeDesp
        from neurocarto.probe_npx.npx import PROBE_TYPE_NP24, ChannelMap
        from neurocarto.probe_npx.plot import electrode_coordinate

        D = NpxProbeDesp()
        E = D.all_electrodes(ChannelMap(PROBE_TYPE_NP24))
        e0 = np.array([it.electrode for it in E])  # Array[E, (S,C,R)]
        e1 = electrode_coordinate(PROBE_TYPE_NP24, 'cr')
        assert_array_equal(e0, e1)

        e0 = np.array([(it.x, it.y) for it in E])  # Array[E, (X,Y)]
        e1 = electrode_coordinate(PROBE_TYPE_NP24, 'xy')
        assert_array_equal(e0, e1)

    def test_iter_electrodes(self):
        from neurocarto.probe_npx.utils import iter_electrodes

        probe = ProbeType(-1, 2, 2, 4, 8, 16, 4, 1, 1, 1, ())

        self.assertListEqual(list(iter_electrodes(probe)), [
            (0, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
            (0, 1, 1),
            (0, 0, 2),
            (0, 1, 2),
            (0, 0, 3),
            (0, 1, 3),
            (1, 0, 0),
            (1, 1, 0),
            (1, 0, 1),
            (1, 1, 1),
            (1, 0, 2),
            (1, 1, 2),
            (1, 0, 3),
            (1, 1, 3),
        ])

        self.assertListEqual(list(iter_electrodes(probe, shank=1)), [
            (1, 0, 0),
            (1, 1, 0),
            (1, 0, 1),
            (1, 1, 1),
            (1, 0, 2),
            (1, 1, 2),
            (1, 0, 3),
            (1, 1, 3),
        ])

        self.assertListEqual(list(iter_electrodes(probe, column=1)), [
            (0, 1, 0),
            (0, 1, 1),
            (0, 1, 2),
            (0, 1, 3),
            (1, 1, 0),
            (1, 1, 1),
            (1, 1, 2),
            (1, 1, 3),
        ])

        self.assertListEqual(list(iter_electrodes(probe, row=slice(1, 3))), [
            (0, 0, 1),
            (0, 1, 1),
            (0, 0, 2),
            (0, 1, 2),
            (1, 0, 1),
            (1, 1, 1),
            (1, 0, 2),
            (1, 1, 2),
        ])

        self.assertListEqual(list(iter_electrodes(probe, row=range(1, 3))), [
            (0, 0, 1),
            (0, 1, 1),
            (0, 0, 2),
            (0, 1, 2),
            (1, 0, 1),
            (1, 1, 1),
            (1, 0, 2),
            (1, 1, 2),
        ])

        self.assertListEqual(list(iter_electrodes(probe, row=[0, 3])), [
            (0, 0, 0),
            (0, 1, 0),
            (0, 0, 3),
            (0, 1, 3),
            (1, 0, 0),
            (1, 1, 0),
            (1, 0, 3),
            (1, 1, 3),
        ])


class NpxProbeBenchmark(unittest.TestCase):
    profile: Profiler | None

    def setUp(self):
        self.profile = None

    def tearDown(self):
        if (profile := self.profile) is not None and profile.enable:
            print(f'{profile.file} use {profile.duration:.2f} sec')

            if (e := profile.run_command()) is not None:
                print(repr(e))

            if (e := profile.exception) is not None:
                self.fail(repr(e))

    def test_all_electrode(self):
        desp = NpxProbeDesp()
        chmap = ChannelMap.from_imro(RES / 'Fig3_example.imro')

        with Profiler('all_electrodes', capture_exception=True) as profile:
            desp.all_electrodes(chmap)

        self.profile = profile

    def test_electrode_density(self):
        from neurocarto.probe_npx.stat import npx_electrode_density
        chmap = ChannelMap.from_imro(RES / 'Fig3_example.imro')

        with Profiler('npx_electrode_density', capture_exception=True) as profile:
            npx_electrode_density(chmap)

        self.profile = profile

    def test_channel_efficiency(self):
        from neurocarto.probe_npx.stat import npx_channel_efficiency

        chmap = ChannelMap.from_imro(RES / 'Fig3_example.imro')
        bp = BlueprintFunctions(NpxProbeDesp(), chmap)
        blueprint = bp.load_blueprint(RES / 'Fig3_example.blueprint.npy')

        with Profiler('npx_channel_efficiency', capture_exception=True) as profile:
            value = npx_channel_efficiency(bp, chmap, blueprint)

        self.profile = profile
        print(f'npx_channel_efficiency(Fig3_example.imro) = {value}')


if __name__ == '__main__':
    unittest.main()
