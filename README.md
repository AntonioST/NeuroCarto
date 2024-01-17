Channelmap Editor for Neuropixels Family
=========================================

Relevant Papers
---------------

writing.

Documents
---------

wiki.

Install and Run
---------------

### Install

This package will upload onto PyPI soon.

### Run

```shell
python -m chmap
```

### Optional dependency

* `bg-atlasapi` Atlas Brain background image supporting.
* `Pillow`, `tifffile` other background image format supporting.
* `probeinterface` probe/channelmap format import/export
* `pandas`, `polars` channelmap data export.

Full optional dependencies are list in [requirements-opt.txt](requirements-opt.txt).

Build from source
-----------------

### Update pip

```shell
python -m pip install --upgrade pip
python -m pip install --upgrade build
```

### Build from source

```shell
python -m build
```

### Install from local

```shell
pip install dist/chmap_editor-0.0.0-py3-none-any.whl
```

### Build Document

Require install extra packages listed in [requirements-doc.txt](requirements-doc.txt)

```shell
cd doc
make html
```


