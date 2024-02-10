import time
import unittest
from pathlib import Path

import numpy as np
from numpy.testing import assert_array_equal

from chmap.probe_npx import ChannelMap
from chmap.probe_npx.desp import NpxProbeDesp, NpxElectrodeDesp
from chmap.views.edit_blueprint import CriteriaParser, default_loader, CriteriaContext


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

    def clone(self, inherit=False) -> CriteriaParser:
        ret = CriteriaParserTester()
        if inherit:
            ret.context = self.context
        ret.message = self.message
        return ret

    def func_call0(self, args: list[str]):
        self.message.append('call0')
        self.message.extend(args)

    def func_call1(self, args: list[str], expression: str):
        self.message.append('call1')
        self.message.extend(args)
        self.message.append(expression)

    def func_set(self, args: list[str], expression: str):
        self.message.append('category')
        self.message.extend(args)
        self.message.append(expression)
        super().func_set(args, expression)


def test_loader(filepath, probe, chmap):
    return default_loader(filepath, probe, chmap)


def external_func_pure(args: list[str], expression: str):
    print('external_func_pure')


def external_func_parser(parser: CriteriaParser, args: list[str], expression: str):
    parser.message.append('external_func_parser')
    parser.message.extend(args)
    parser.message.append(expression)


def external_func_context(context: CriteriaContext, args: list[str], expression: str):
    parser = context.parser
    parser.message.append('external_func_context')
    parser.message.extend(args)
    parser.message.append(expression)


def get_test_file(filename) -> Path:
    if not (p := Path('tests') / filename).exists():  # from command line
        if not (p := Path('.') / filename).exists():  # from IDE
            raise FileNotFoundError(f'file not found {filename}')
    return p


class TimeMaker:
    def __init__(self):
        self.t = time.time()

    def reset(self):
        self.t = time.time()

    def __call__(self, message: str):
        t = time.time()
        d = t - self.t
        print(message, f'use {d:.2f}')
        self.t = t


class EditBlueprintTest(unittest.TestCase):
    parser: CriteriaParserTester

    def setUp(self):
        self.parser = CriteriaParserTester()

    def assert_blueprint_equal(self, a: list[NpxElectrodeDesp], b: list[NpxElectrodeDesp]):
        p0 = np.array([it.category for it in a])
        p1 = np.array([it.category for it in b])

        assert_array_equal(p0, p1)

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

    def test_func_call_unexpected_expression(self):
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

    def test_func_print(self):
        self.parser.parse_content("""
        print()=test
        """)

        self.assertListEqual(['test'], self.parser.message)

    def test_func_print_eval(self):
        self.parser.parse_content("""
        print(eval)=f'test {1+1}'
        """)

        self.assertListEqual(['test 2'], self.parser.message)

    def test_func_use(self):
        ret = self.parser.parse_content("""
        use(NpxProbeDesp)
        """)

        self.assertTrue(ret)

    def test_func_use_fail(self):
        ret = self.parser.parse_content("""
        use(NoDefineProbeDesp)
        """)

        self.assertFalse(ret)
        self.assertListEqual(['fail probe check : NpxProbeDesp'], self.parser.message)

    def test_func_check(self):
        self.parser.parse_content("""
        check(FULL,HALF,QUARTER)
        """)

        self.assertListEqual([], self.parser.message)

    def test_func_check_info(self):
        self.assertNotIn('X', self.parser.categories)
        self.parser.parse_content("""
        check(FULL,HALF,QUARTER,X)=info
        alias(X)=FORBIDDEN
        """)

        self.assertListEqual(['unknown category X'], self.parser.message)
        self.assertIn('X', self.parser.categories)

    def test_func_check_error(self):
        self.assertNotIn('X', self.parser.categories)
        self.parser.parse_content("""
        check(FULL,HALF,QUARTER,X)=error
        alias(X)=FORBIDDEN
        """)

        self.assertListEqual(['unknown category X'], self.parser.message)
        self.assertNotIn('X', self.parser.categories)

    def test_external_func(self):
        self.parser.parse_content("""
        func(test)=tests:test_edit_blueprint:external_func_pure
        test(arg)=exp
        """)

        self.assertListEqual([], self.parser.message)

    def test_external_func_scope_parser(self):
        self.parser.parse_content("""
        func(test, parser)=tests:test_edit_blueprint:external_func_parser
        test(arg)=exp
        """)

        self.assertListEqual(['external_func_parser', 'arg', 'exp'], self.parser.message)

    def test_external_func_scope_context(self):
        self.parser.parse_content("""
        func(test, context)=tests:test_edit_blueprint:external_func_context
        file=None
        test(arg)=exp
        """)

        self.assertListEqual(['external_func_context', 'arg', 'exp'], self.parser.message)

    def test_external_func_scope_context_without_context(self):
        self.parser.parse_content("""
        func(test, context)=tests:test_edit_blueprint:external_func_context
        test(arg)=exp
        """)

        self.assertListEqual(['call func test() without context'], self.parser.message)

    def test_run_file(self):
        self.parser.parse_content(f"""
        run()={get_test_file('test_edit_blueprint.txt')}
        """)

        self.assertListEqual(['category', 'X', '1'], self.parser.message)
        self.assertIn('DB_func', self.parser.external_functions)
        self.assertIn('DB_var', self.parser.variables)
        self.assertIn('X', self.parser.categories)
        self.assertTrue(np.all(self.parser.get_result() == NpxProbeDesp.CATE_FORBIDDEN))

    def test_run_file_no_all(self):
        self.parser.parse_content(f"""
        run(xf,xv,xa,xr)={get_test_file('test_edit_blueprint.txt')}
        """)

        self.assertListEqual(['category', 'X', '1'], self.parser.message)
        self.assertNotIn('DB_func', self.parser.external_functions)
        self.assertNotIn('DB_var', self.parser.variables)
        self.assertNotIn('X', self.parser.categories)
        self.assertTrue(np.all(self.parser.get_result() == NpxProbeDesp.CATE_UNSET))

    def test_func_alias(self):
        self.assertIn('FORBIDDEN', self.parser.categories)
        self.assertNotIn('X', self.parser.categories)

        self.parser.parse_content("""
        alias(X)=FORBIDDEN
        """)

        self.assertIn('X', self.parser.categories)
        self.assertEqual(self.parser.categories['FORBIDDEN'], self.parser.categories['X'])

    def test_func_category(self):
        self.parser.parse_content("""
        file=None
        FORBIDDEN=1
        """)

        self.assertListEqual(['category', 'FORBIDDEN', '1'], self.parser.message)
        self.assertTrue(np.all(self.parser.get_result() == NpxProbeDesp.CATE_FORBIDDEN))

    def test_func_category_skip_when_file_error(self):
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
            'category', 'FORBIDDEN', 's==0',
            'category', 'FORBIDDEN', 's==1',
            'category', 'FORBIDDEN', 's==2',
            'category', 'FORBIDDEN', 's==3',
        ], self.parser.message)

        self.assertTrue(np.all(self.parser.get_result() == NpxProbeDesp.CATE_FORBIDDEN))

    def test_variable_not_define(self):
        self.parser.parse_content("""
        file=None
        FORBIDDEN=k
        """)

        self.assertListEqual(['category', 'FORBIDDEN', 'k', 'un-captured error'], self.parser.message)
        self.assertIsInstance(self.parser.error, NameError)

    def test_func_val(self):
        self.parser.parse_content("""
        val(k)=1
        file=None
        FORBIDDEN=k
        """)

        self.assertIn('k', self.parser.variables)
        self.assertListEqual(['category', 'FORBIDDEN', 'k'], self.parser.message)
        self.assertTrue(np.all(self.parser.get_result() == NpxProbeDesp.CATE_FORBIDDEN))

    def test_func_var(self):
        self.parser.parse_content("""
        file=None
        var(k)=1
        FORBIDDEN=k
        """)

        self.assertNotIn('k', self.parser.variables)
        self.assertListEqual(['category', 'FORBIDDEN', 'k'], self.parser.message)
        self.assertTrue(np.all(self.parser.get_result() == NpxProbeDesp.CATE_FORBIDDEN))

    def test_func_save(self):
        m = TimeMaker()
        self.parser.parse_content("""
        file=None
        FORBIDDEN=(y>6000)
        save()=./test-save
        """)
        m('parse_content')

        self.assertListEqual(['category', 'FORBIDDEN', '(y>6000)', 'save test-save.blueprint.npy'],
                             self.parser.message)
        m('assertListEqual')

        blueprint = self.parser.get_blueprint()
        m('set_blueprint')

        file = Path('.') / 'test-save.blueprint.npy'
        self.assertTrue(file.exists())
        m('assertTrue')

        probe = self.parser.probe
        chmap = self.parser.chmap
        test = probe.load_blueprint(file, chmap)
        m('electrode_from_numpy')

        self.assert_blueprint_equal(blueprint, test)
        m('assert_blueprint_equal')

    @unittest.skipIf(condition=not Path('test-save.blueprint.npy').exists(),
                     reason='test_func_save() need to run first')
    def test_func_blueprint(self):
        m = TimeMaker()
        self.parser.parse_content("""
        blueprint()=./test-save.blueprint.npy
        """)
        m('parse_content')

        self.assertListEqual(['load test-save.blueprint.npy'], self.parser.message)
        m('assertListEqual')

        blueprint = self.parser.get_blueprint()
        m('set_blueprint')

        probe = self.parser.probe
        chmap = self.parser.chmap
        test = probe.load_blueprint('test-save.blueprint.npy', chmap)
        m('electrode_from_numpy')

        self.assert_blueprint_equal(blueprint, test)
        m('assert_blueprint_equal')

    def test_category_setting(self):
        self.parser.parse_content("""
        file=None
        FORBIDDEN=(y>6000)
        LOW=1
        """)

        self.assertFalse(np.all(self.parser.get_result() == NpxProbeDesp.CATE_LOW))

        self.setUp()

        self.parser.parse_content("""
        file=None
        LOW=1
        FORBIDDEN=(y>6000)
        """)

        self.assertTrue(np.all(self.parser.get_result() == NpxProbeDesp.CATE_LOW))

    def test_block_setting(self):
        self.parser.parse_content("""
        file=None
        FORBIDDEN=(y>6000)
        
        file=None
        LOW=1
        """)

        self.assertTrue(np.all(self.parser.get_result() == NpxProbeDesp.CATE_LOW))

        self.setUp()

        self.parser.parse_content("""
        file=None
        LOW=1
        
        file=None
        FORBIDDEN=(y>6000)
        """)

        self.assertFalse(np.all(self.parser.get_result() == NpxProbeDesp.CATE_LOW))

    def test_func_move(self):
        self.parser.parse_content("""
        file=None
        FORBIDDEN=(s==0)&(y>6000)
        FORBIDDEN=(s==1)&(y>6000)
        FORBIDDEN=(s==2)&(y>7000)
        FORBIDDEN=(s==3)&(y>7000)
        """)

        expected_blueprint = self.parser.get_blueprint()

        self.setUp()

        self.parser.parse_content("""
        file=None
        FORBIDDEN=(y>6000)
        
        move(2,3)=1000
        """)

        blueprint = self.parser.get_blueprint()
        self.assert_blueprint_equal(expected_blueprint, blueprint)

    def test_bp_blueprint(self):
        self.parser.parse_content("""
        file=None
        FORBIDDEN=(y>6000)
        val(b)=bp.blueprint()
        """)

        assert_array_equal(self.parser.context.result, self.parser.variables['b'])

    def test_bp_set_blueprint(self):
        self.parser.parse_content("""
        file=None
        FORBIDDEN=(y>6000)
        """)
        expected_blueprint = self.parser.context.result

        self.setUp()
        self.parser.variables['b'] = expected_blueprint
        self.parser.parse_content("""
        file=None
        var(_)=bp.set_blueprint(b)
        """)
        blueprint = self.parser.context.result
        assert_array_equal(blueprint, expected_blueprint)

    def test_bp_move(self):
        self.parser.parse_content("""
        file=None
        FORBIDDEN=(y>6000)

        move(2,3)=1000
        """)

        expected_blueprint = self.parser.get_blueprint()

        self.setUp()
        self.parser.parse_content("""
        file=None
        FORBIDDEN=(y>6000)
        eval()=bp.set_blueprint(bp.move(bp.blueprint(), tx=0, ty=1000, shanks=[2,3], init=bp.CATE_UNSET))
        """)

        blueprint = self.parser.get_blueprint()
        self.assert_blueprint_equal(expected_blueprint, blueprint)


if __name__ == '__main__':
    unittest.main()
