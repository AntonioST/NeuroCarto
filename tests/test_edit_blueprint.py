import unittest
from pathlib import Path

import numpy as np

from chmap.probe_npx import ChannelMap
from chmap.probe_npx.desp import NpxProbeDesp, NpxElectrodeDesp
from chmap.views.edit_blueprint import CriteriaParser, default_loader


class CriteriaParserTester(CriteriaParser[ChannelMap, NpxElectrodeDesp]):
    def __init__(self):
        super().__init__(None, NpxProbeDesp(), ChannelMap(24))
        self.message = []
        self.error = None

    def info(self, message: str):
        self.message.append(message)

    def warning(self, message: str, exc: BaseException = None):
        self.message.append(message)
        self.error = exc

    def func_call0(self, args: list[str]):
        self.message.append('call0')
        self.message.extend(args)

    def func_call1(self, args: list[str], expression: str):
        self.message.append('call1')
        self.message.extend(args)
        self.message.append(expression)

    def func_policy(self, args: list[str], expression: str):
        self.message.append('policy')
        self.message.extend(args)
        self.message.append(expression)
        super().func_policy(args, expression)


def test_loader(filepath, probe, chmap):
    return default_loader(filepath, probe, chmap)


class EditBlueprintTest(unittest.TestCase):
    parser: CriteriaParserTester

    def setUp(self):
        self.parser = CriteriaParserTester()

    def test_file(self):
        self.parser.parse_content("""
        file=test.npy
        """)

        self.assertIsNotNone(context := self.parser.context)
        self.assertEqual(Path('test.npy'), context._file)

    def test_comment(self):
        self.parser.parse_content("""
        file=test.npy # load file
        """)

        self.assertIsNotNone(context := self.parser.context)
        self.assertEqual(Path('test.npy'), context._file)

    def test_file_none(self):
        self.parser.parse_content("""
        file=None
        """)

        self.assertIsNotNone(context := self.parser.context)
        self.assertIsNone(context._file)

    def test_loader_default(self):
        self.parser.parse_content("""
        file=None
        """)

        self.assertIsNotNone(context := self.parser.context)
        self.assertIs(default_loader, context.loader)

    def test_loader(self):
        self.parser.parse_content("""
        file=None
        loader=tests:test_edit_blueprint:test_loader
        """)

        self.assertIsNotNone(context := self.parser.context)
        # because function is load from different context(?), they have different memory address
        self.assertIs(test_loader.__name__, context.loader.__name__)

    def test_loder_missing_file(self):
        self.parser.parse_content("""
        loader=tests:test_edit_blueprint:test_loader
        """)

        self.assertIsNone(self.parser.context)
        self.assertListEqual(['missing file='], self.parser.message)

    def test_func_call_no_expression(self):
        self.parser.parse_content("""
        call0(1,2,3)
        """)

        self.assertListEqual(['call0', '1', '2', '3'], self.parser.message)

    def test_func_call_unexpect_expression(self):
        self.parser.parse_content("""
        call0(1,2,3)=expression
        """)

        self.assertListEqual(['un-captured error'], self.parser.message)
        self.assertIsInstance(self.parser.error, TypeError)

    def test_func_call_with_expression(self):
        self.parser.parse_content("""
        call1(1,2,3)=expression
        """)

        self.assertListEqual(['call1', '1', '2', '3', 'expression'], self.parser.message)

    def test_func_call_forget_expression(self):
        self.parser.parse_content("""
        call1(1,2,3)
        """)

        self.assertListEqual(['un-captured error'], self.parser.message)
        self.assertIsInstance(self.parser.error, TypeError)

    def test_func_call_no_define(self):
        self.parser.parse_content("""
        call2(1,2,3)=expression
        """)

        self.assertListEqual(['unknown func call2'], self.parser.message)

    def test_func_check(self):
        self.parser.parse_content("""
        check(FULL,HALF,QUARTER)
        """)

        self.assertListEqual([], self.parser.message)

    def test_func_alias(self):
        self.assertIn('FORBIDDEN', self.parser.policies)
        self.assertNotIn('X', self.parser.policies)

        self.parser.parse_content("""
        alias(X)=FORBIDDEN
        """)

        self.assertIn('X', self.parser.policies)
        self.assertEqual(self.parser.policies['FORBIDDEN'], self.parser.policies['X'])

    def test_func_policy(self):
        self.parser.parse_content("""
        file=None
        FORBIDDEN=1
        """)

        self.assertListEqual(['policy', 'FORBIDDEN', '1'], self.parser.message)
        self.assertTrue(np.all(self.parser.get_result() == NpxProbeDesp.POLICY_FORBIDDEN))

    def test_func_policy_skip_when_file_error(self):
        self.parser.parse_content("""
        file=not_found.npy
        FORBIDDEN=1
        """)

        match self.parser.message:
            case [message]:
                self.assertEqual('file not found. not_found.npy', message)
            case _:
                self.fail()

    def test_variable(self):
        self.parser.parse_content("""
        file=None
        FORBIDDEN=s==0
        FORBIDDEN=s==1
        FORBIDDEN=s==2
        FORBIDDEN=s==3
        """)

        self.assertListEqual([
            'policy', 'FORBIDDEN', 's==0',
            'policy', 'FORBIDDEN', 's==1',
            'policy', 'FORBIDDEN', 's==2',
            'policy', 'FORBIDDEN', 's==3',
        ], self.parser.message)

        self.assertTrue(np.all(self.parser.get_result() == NpxProbeDesp.POLICY_FORBIDDEN))

    def test_variable_not_define(self):
        self.parser.parse_content("""
        file=None
        FORBIDDEN=k
        """)

        self.assertListEqual(['policy', 'FORBIDDEN', 'k', 'un-captured error'], self.parser.message)
        self.assertIsInstance(self.parser.error, NameError)

    def test_func_val(self):
        self.parser.parse_content("""
        val(k)=1
        file=None
        FORBIDDEN=k
        """)

        self.assertListEqual(['policy', 'FORBIDDEN', 'k'], self.parser.message)
        self.assertTrue(np.all(self.parser.get_result() == NpxProbeDesp.POLICY_FORBIDDEN))


if __name__ == '__main__':
    unittest.main()
