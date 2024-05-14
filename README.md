NeuroCarto: A Neuropixels Channelmap Editor
===========================================

[![PyPI - Version](https://img.shields.io/pypi/v/neurocarto)](https://pypi.org/project/neurocarto/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/neurocarto)
[![Documentation Status](https://readthedocs.org/projects/neurocarto/badge/?version=latest)](https://neurocarto.readthedocs.io/en/latest/?badge=latest)

NeuroCarto is a neural probe channel map editor for the Neuropixels probe family.
It allows user to create a blueprint for arranging electrodes in a desired density
and generate a custom channel map.

Features
--------

- Read/Visualize/Modify/Write Neuropixels channelmap files (`*.imro`).
- Read SpikeGLX meta file (`*.meta`).
- Read/Visualize/Modify/Write Blueprint (a blueprint for generating a channelmap by a programming way).
- Show Atlas mouse brain as a background image.
- Customize electrode selection and probe kind.
- Show channel efficiency and electrode density.

Relevant Papers
---------------

writing.

Documents
---------

Please check [Documentation](https://neurocarto.readthedocs.io/en/latest/) for more details.

Install and Run
---------------

### prepare environment.

Require `Python 3.10`.

### Install

```shell
pip install neurocarto
```

### Run

```shell
neurocarto
```

### Optional dependency

* `bg-atlasapi` Atlas Brain background image supporting.
* `Pillow`, `tifffile` other background image format supporting.
* `probeinterface` probe/channelmap format import/export
* `pandas`, `polars` channelmap data export.

Full optional dependencies are list in [requirements-opt.txt](requirements-opt.txt).

Build from source
-----------------

Please check [install](https://github.com/AntonioST/chmap_editor/wiki/install) page in wiki.


