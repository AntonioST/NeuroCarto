Installation
============

Require
-------

NeuroCarto requires ``Python 3.11`` or newer. A Python virtual environment is recommended.

Install
-------

::

    pip install neurocarto

Run
---

::

    neurocarto

Optional dependency packages
----------------------------

* ``brainglobe-atlasapi`` for Atlas Brain background images.
* ``Pillow`` and ``tifffile`` for other background image formats.
* ``probeinterface`` for probe/channelmap import and export.
* ``pandas`` and ``polars`` for channelmap data export.

Build from source
-----------------

1. Create a Python environment. This example uses ``conda``. ::

    conda create -n neurocarto python~=3.11.0
    conda activate neurocarto

2. Clone the repository. ::

    git clone https://github.com/AntonioST/NeuroCarto.git
    cd NeuroCarto

3. Update pip. ::

    python -m pip install --upgrade pip
    python -m pip install --upgrade build

4. Build the package. ::

    python -m build

5. Install the package. ::

        pip install dist/neurocarto-0.0.0-py3-none-any.whl

   Or install it in development mode. ::

        pip install -e .

   Change version ``0.0.0`` to the latest version when needed.

Build Document
--------------

Make sure the package has been installed or added to ``PYTHONPATH``.

1. Install extra dependencies. ::

    pip install -r requirements-doc.txt

2. Build the documentation. ::

    cd doc
    make html

3. Open the generated HTML at ``doc/build/html/index.html``.

Build GitHub wiki
~~~~~~~~~~~~~~~~~

For contributors only.

1. On the first run, initialize and update the submodule. ::

    git submodule init doc/wiki
    git submodule update doc/wiki

2. Run make under the ``doc`` directory. ::

    make markdown

3. Check the output at ``doc/wiki``. Note that the ``rst`` output may not be suitable for ``md``.
   Check the git diff and modify the contents as needed.

Testing Library Versions
------------------------

on 2024/05.

::

    # environment
    pip                            23.3.1                        pypi_0     pypi
    python                         3.10.13                   h955ad1f_0         
    # required
    bokeh                          3.3.2                         pypi_0     pypi
    matplotlib                     3.7.1                py310h06a4308_1         
    numpy                          1.22.3               py310h4f1e569_0         
    scipy                          1.7.3                py310hac523dd_2         
    typing_extensions              4.6.3                py310h06a4308_0         
    # optional
    bg-atlasapi                    1.0.2                         pypi_0     pypi
    pillow                         9.4.0                py310h6a678d5_0         
    pandas                         1.5.3                py310h1128e8f_0         
    polars                         0.18.2                        pypi_0     pypi
    probeinterface                 0.2.21                        pypi_0     pypi
    tifffile                       2023.12.9                     pypi_0     pypi
    # document
    jupyter                        1.0.0                         pypi_0     pypi
    jupyter-contrib-nbextensions   0.7.0                         pypi_0     pypi
    nbsphinx                       0.9.3                         pypi_0     pypi
    sphinx                         7.2.6                         pypi_0     pypi
    sphinx-markdown-builder        0.6.6                         pypi_0     pypi
    sphinx-rtd-theme               2.0.0                         pypi_0     pypi
