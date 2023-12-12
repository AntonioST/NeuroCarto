import unittest

import numpy as np
from numpy.testing import assert_array_equal


class ProbeChannelElectrodeMappingTest(unittest.TestCase):
    def test_np1(self):
        from chmap.probe_npx.npx_probe import PROBE_TYPE_NP1, e2c0, c2e0
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
        from chmap.probe_npx.npx_probe import PROBE_TYPE_NP21, e2c21, c2e21
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
        from chmap.probe_npx.npx_probe import PROBE_TYPE_NP24, e2c24, c2e24
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


if __name__ == '__main__':
    unittest.main()
