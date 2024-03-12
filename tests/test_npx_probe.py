import unittest

import numpy as np
from numpy.testing import assert_array_equal


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


if __name__ == '__main__':
    unittest.main()
