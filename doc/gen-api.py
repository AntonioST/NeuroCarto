import os
from pathlib import Path

src = Path('../src/chmap')
dst = Path('source/api')

for p, ds, fs in os.walk(src):
    for f in fs:
        if not f.startswith('_') and f.endswith('.py'):
            r = (Path(p).relative_to(src) / f).with_suffix('')
            o = dst / str(r.with_suffix('.rst')).replace('/', '.')
            if not o.exists():
                print('new', o)
                k = '.'.join(['chmap', *r.parts])
                with open(o, 'w') as of:
                    print(f"""\
{k}
{"=" * len(k)}

.. automodule:: {k}
   :members:
""", file=of)

    for d in ds:
        if d != '__pycache__':
            r = Path(p).relative_to(src) / d
            o = dst / str(r.with_suffix('.rst')).replace('/', '.')
            if not o.exists():
                print('new', o)
                k = '.'.join(['chmap', *r.parts])
                with open(o, 'w') as of:
                    print("""\
{k}
{"=" * len(k)}

.. automodule:: {k}
    :members:
   
modules
-------
.. toctree::
    :maxdepth: 1
""", file=of)
