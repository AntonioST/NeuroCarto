Channelmap Editor for the Neuropixels Family
============================================

Neuropixels channelmap editor is an editor for generate a custom channelmap from 
a given electrode-density blueprint.

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

Please check [wiki](https://github.com/AntonioST/chmap_editor/wiki) for more details.

Install and Run
---------------

### prepare environment.

Require `Python 3.10`.

### Install


This package will upload onto PyPI soon.

### Run

```shell
chmap
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


