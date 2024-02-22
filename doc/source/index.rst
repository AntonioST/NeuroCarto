.. ChmapEditor documentation master file, created by
   sphinx-quickstart on Fri Jan  5 04:31:12 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ChmapEditor's documentation!
=======================================

Neuropixels channelmap editor is an editor for generate a custom channelmap from
a given electrode-density blueprint.

Features
--------

- [x] Read/Visualize/Modify/Write Neuropixels channelmap files (``*.imro``).

  - [x] Read Neuropixels channelmap files from other supported file format in GUI.
  - [x] (Experimental) set bad electrodes

- [x] Read SpikeGLX meta file (``*.meta``).

  - [x] Read Neuropixels probe serial number

- [x] Read/Visualize/Modify/Write Blueprint (a blueprint for generating a channelmap by a programming way).
- [x] Show Atlas mouse brain as a background image.

  - [ ] utilities functions for controlling the atlas image.
  - [ ] probe coordinate functions

- [x] Customize electrode selection and probe kind.
- [x] Show channel efficiency and electrode density.
- [x] (Experimental) Show an image file as a background image (``--view=file``, ``--view=IMAGE``).

  - [ ] Read image resolution tags.

- [x] Show dynamic generated image (via ``matplotlib``) as a background image.
- [x] (Experimental) Run custom scripts (``--view=script``).

  - [x] give an example script to initial a blueprint based on an experimental data.

- [x] (Experimental) Record/Save/Load/Replay channelmap manipulate steps (``--view=history``)



Contents
--------

.. toctree::
    :maxdepth: 2

    install
    start
    views
    tutorial


Support formats
---------------

==================== ============================
package\\probe       Neuropixels family
==================== ============================
SpikeGLX ``*.imro``  read/write
SpikeGLX ``*.meta``  read
probeinterface [1]_  read/write
pandas [1]_          write [2]_
polars [1]_          write [2]_
==================== ============================

.. [1] optional dependency package
.. [2] convert channelmap as dataframe.

Support probe kinds
-------------------

Neuropixels probe family
~~~~~~~~~~~~~~~~~~~~~~~~

file format: ``*.imro``.

===== =============== =============== ======================= =================
rw\\p Neuropixels 1.0 Neuropixels 2.0 4-shank Neuropixels 2.0 Neuropixels Ultra
===== =============== =============== ======================= =================
read       yes             yes               yes                    no
write      yes             yes               yes                    no
===== =============== =============== ======================= =================



API Reference
-------------

.. toctree::
    :maxdepth: 1

    api/chmap.rst

Contact Us
----------


.. toctree::
    :maxdepth: 1

    contact


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
