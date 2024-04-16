import os
import textwrap
from pathlib import Path

src = Path('../src/neurocarto')
dst = Path('source/api')

src_files = ['neurocarto.rst']

CONTENT_FILE = """\
neurocarto.{module}
{module_len}

.. automodule:: neurocarto.{module}
   :members:
   :undoc-members:
"""

CONTENT_DIR = """\
neurocarto.{module}
{module_len}

.. automodule:: neurocarto.{module}
   :members:

.. toctree::
    :maxdepth: 1
    :caption: Modules:

{module_list}
"""

for p, ds, fs in os.walk(src):
    for f in fs:
        if not f.startswith('_') and f.endswith('.py'):
            r = (Path(p).relative_to(src) / f).with_suffix('')
            o = dst / str(r.with_suffix('.rst')).replace('/', '.')
            src_files.append(o.name)

            if not o.exists():
                print('new', o)
                k = '.'.join(r.parts)
                with open(o, 'w') as of:
                    print(CONTENT_FILE.format(module=k, module_len='=' * (len(k) + 11)), file=of)

    for d in ds:
        if d != '__pycache__':
            r = Path(p).relative_to(src) / d
            o = dst / str(r.with_suffix('.rst')).replace('/', '.')
            src_files.append(o.name)
            if not o.exists():
                print('new', o)
                k = '.'.join(r.parts)

                module_list = []
                for f in (Path(p) / d).iterdir():
                    if f.suffix == '.py' and not f.name.startswith('_'):
                        module_list.append(k + '.' + f.stem)
                    elif f.is_dir() and (f / '__init__.py').exists():
                        module_list.append(k + '.' + f.name)

                module_list.sort()
                module_list_content = textwrap.indent('\n'.join(module_list), '    ')
                with open(o, 'w') as of:
                    print(CONTENT_DIR.format(module=k, module_len='=' * (len(k) + 11), module_list=module_list_content), file=of)

for f in dst.iterdir():
    if f.suffix == '.rst':
        if f.name not in src_files:
            print('delete?', f.name)
