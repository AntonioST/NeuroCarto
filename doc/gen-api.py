import os
from pathlib import Path

src = Path('../src/neurocarto')
dst = Path('source/api')

src_files = ['neurocarto.rst']

for p, ds, fs in os.walk(src):
    for f in fs:
        if not f.startswith('_') and f.endswith('.py'):
            r = (Path(p).relative_to(src) / f).with_suffix('')
            o = dst / str(r.with_suffix('.rst')).replace('/', '.')
            src_files.append(o.name)

            if not o.exists():
                print('new', o)
                k = '.'.join(['neurocarto', *r.parts])
                with open(o, 'w') as of:
                    print(f"""\
{k}
{"=" * len(k)}

.. automodule:: {k}
   :members:
   :undoc-members:
""", file=of)

    for d in ds:
        if d != '__pycache__':
            r = Path(p).relative_to(src) / d
            o = dst / str(r.with_suffix('.rst')).replace('/', '.')
            src_files.append(o.name)
            if not o.exists():
                print('new', o)
                k = '.'.join(['neurocarto', *r.parts])
                with open(o, 'w') as of:
                    print(f"""\
{k}
{"=" * len(k)}

.. automodule:: {k}
    :members:
   
modules
-------
.. toctree::
    :maxdepth: 1
""", file=of)

for f in dst.iterdir():
    if f.suffix == '.rst':
        if f.name not in src_files:
            print('delete?', f.name)
