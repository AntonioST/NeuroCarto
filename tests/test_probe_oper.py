import unittest
from pathlib import Path
from typing import Generic

from neurocarto.probe import ProbeDesp, M, E
from neurocarto.probe_npx import NpxProbeDesp, ChannelMap

if (res := Path('res')).exists():
    RES = res
elif (res := Path('../res')).exists():
    RES = res
else:
    raise RuntimeError()


class ProbeDespTest(Generic[M, E]):
    probe: ProbeDesp[M, E]
    chmap_file: Path
    chmap: M

    def test_support_type(self):
        zelf: unittest.TestCase = self

        zelf.assertIsNone(self.probe.type_description(None))
        for desp, code in self.probe.supported_type.items():
            zelf.assertEqual(desp, self.probe.type_description(code))

    def test_channelmap_file_suffix(self):
        zelf: unittest.TestCase = self
        suffix = self.probe.channelmap_file_suffix
        zelf.assertTrue(len(suffix) >= 1)

        for i, s in enumerate(suffix):
            zelf.assertTrue(s.startswith('.'), f'channelmap_file_suffix[{i}]={s}')

    def test_load_file(self):
        zelf: unittest.TestCase = self

        zelf.assertEqual(self.chmap, self.probe.load_from_file(self.chmap_file))

    def test_new_channelmap(self):
        zelf: unittest.TestCase = self

        a = self.probe.channelmap_code(self.chmap)
        zelf.assertEqual(a, self.probe.channelmap_code(self.probe.new_channelmap(self.chmap)))
        zelf.assertEqual(a, self.probe.channelmap_code(self.probe.new_channelmap(a)))

    def test_channelmap_code(self):
        zelf: unittest.TestCase = self

        code = self.probe.channelmap_code(self.chmap)
        zelf.assertIsInstance(code, int)
        zelf.assertIn(code, self.probe.supported_type.values())

        desp = self.probe.type_description(code)
        zelf.assertIsInstance(desp, str)
        zelf.assertIn(desp, self.probe.supported_type)
        zelf.assertEqual(code, self.probe.supported_type[desp])

    def test_channelmap_code_unknown(self):
        zelf: unittest.TestCase = self

        zelf.assertIsNone(self.probe.channelmap_code(None))
        zelf.assertIsNone(self.probe.channelmap_code(object()))

    def test_channelmap_copy(self):
        zelf: unittest.TestCase = self

        a = self.chmap
        b = self.probe.copy_channelmap(a)

        zelf.assertIsNot(a, b)
        zelf.assertEqual(a, b)

    def test_all_electrodes(self):
        zelf: unittest.TestCase = self

        a = self.probe.all_electrodes(self.chmap)
        b = self.probe.all_electrodes(self.probe.channelmap_code(self.chmap))

        zelf.assertTrue(len(a) > 0)
        zelf.assertListEqual(a, b)

    def test_all_channels(self):
        zelf: unittest.TestCase = self

        a = self.probe.all_channels(self.chmap)
        zelf.assertEqual(len(self.chmap), len(a))

    def test_all_channels_subset(self):
        zelf: unittest.TestCase = self

        e = self.probe.all_electrodes(self.chmap)
        x = e[:len(e) // 2]

        for c in self.probe.all_channels(self.chmap, x):
            zelf.assertIn(c, x)

        x = e[len(e) // 2:]

        for c in self.probe.all_channels(self.chmap, x):
            zelf.assertIn(c, x)

    def test_is_valid(self):
        zelf: unittest.TestCase = self

        zelf.assertTrue(self.probe.is_valid(self.chmap))
        zelf.assertFalse(self.probe.is_valid(self.probe.new_channelmap(self.chmap)))

    def test_add_electrode(self):
        zelf: unittest.TestCase = self
        e = self.probe.all_channels(self.chmap)

        c = self.probe.new_channelmap(self.chmap)
        for ee in e:
            zelf.assertNotEquals(self.chmap, c)
            self.probe.add_electrode(c, ee)

        zelf.assertEqual(self.chmap, c)

    def test_clear_electrode(self):
        zelf: unittest.TestCase = self
        c = self.chmap
        self.probe.clear_electrode(c)
        zelf.assertEqual(0, len(c))


class NpxProbeDespTest(ProbeDespTest[NpxProbeDesp, ChannelMap], unittest.TestCase):

    def setUp(self):
        self.probe = NpxProbeDesp()
        self.chmap_file = RES / 'Fig3_example.imro'
        self.chmap = ChannelMap.from_imro(self.chmap_file)


if __name__ == '__main__':
    unittest.main()
