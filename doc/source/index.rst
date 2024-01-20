.. ChmapEditor documentation master file, created by
   sphinx-quickstart on Fri Jan  5 04:31:12 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ChmapEditor's documentation!
=======================================

Neuropixels channelmap editor is an editor for generate a custom channelmap from
a given electrode-density blueprint.


Contents
--------

.. toctree::
    :maxdepth: 2

    install
    start
    tutorial


Support formats
---------------

==================== ============================
package\\probe       Neuropixels family
==================== ============================
native               `*.imro` (rw), `*.meta` (r)
probeinterface [1]_  read/write
pandas [1]_          write [2]_
polars [1]_          write [2]_
==================== ============================

.. [1] optional dependency package
.. [2] convert channelmap as dataframe.

Support probes
--------------

Neuropixels probe family
~~~~~~~~~~~~~~~~~~~~~~~~

native file format: `*.imro`.

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
