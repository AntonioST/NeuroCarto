Installation
============

Install
-------

:warning:
    So far this packages doesn't upload to https://pypi.org/.

::

    pip install chmap

Optional dependency packages
----------------------------

* `bg-atlasapi` Atlas Brain background image supporting.
* `Pillow`, `tifffile` other background image format supporting.
* `probeinterface` probe/channelmap format import/export
* `pandas`, `polars` channelmap data export.

Build from source
-----------------

1. create python environment. Here use conda as example. ::

    conda create -n chmap python~=3.10.0
    conda activate chmap

2. clone repository. ::

    git clone

3, update pip. ::

    python -m pip install --upgrade pip
    python -m pip install --upgrade build

4. build. ::

    python -m build

5. install. ::

    pip install dist/chmap_editor-0.0.0-py3-none-any.whl

