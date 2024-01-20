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

    git clone https://github.com/AntonioST/chamap_editor.git

3. update pip. ::

    python -m pip install --upgrade pip
    python -m pip install --upgrade build

4. build. ::

    python -m build

5. install. ::

    pip install dist/chmap_editor-0.0.0-py3-none-any.whl


Build Document
--------------

1. install extra dependencies. ::

    pip install -r requirements-doc.txt

2. build. ::

    cd doc
    make html

3. open generated html at `doc/build/html/index.html`.

build github wiki
~~~~~~~~~~~~~~~~~

For contributors only.

1. (first time) initialize submodule and update it. ::

    git submodule init doc/wiki
    git submodule update doc/wiki

2. under `doc` directory. run make. ::

    make markdown

3. check output at `doc/wiki`. Note that the `rst` output may be not suitable for `md`.
   Check the git difference and modify the contents are necessary.

testing library versions
------------------------

::

    # environment
    pip                       23.1.2          py310h06a4308_0
    python                    3.10.13              h955ad1f_0
    # required
    bokeh                     3.3.2                    pypi_0    pypi
    matplotlib                3.7.1           py310h06a4308_1
    numpy                     1.24.3          py310h5f9d8c6_1
    scipy                     1.10.1          py310h5f9d8c6_1
    typing_extensions         4.6.3           py310h06a4308_0
    # optional
    bg-atlasapi               1.0.2                    pypi_0    pypi
    pillow                    9.4.0           py310h6a678d5_0
    pandas                    1.5.3           py310h1128e8f_0
    polars                    0.18.2                   pypi_0    pypi
    probeinterface            0.2.20                   pypi_0    pypi
    tifffile                  2023.12.9                pypi_0    pypi
    # document
    jupyter                   1.0.0           py310h06a4308_8
    jupyter-contrib-nbextensions 0.7.0                 pypi_0    pypi
    nbsphinx                  0.9.3                    pypi_0    pypi
    sphinx                    7.2.6                    pypi_0    pypi
    sphinx-markdown-builder   0.6.6                    pypi_0    pypi
    sphinx-rtd-theme          2.0.0                    pypi_0    pypi

