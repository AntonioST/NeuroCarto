# Changelog

## 0.2.2

2026/08/05

### Added

- Add user config `figure_size` for the main probe figure.
- Add IMRO IO and Neuropixels probe mapping tests.

### Changed

- Require Python 3.11 or newer.
- Pin optional `probeinterface` dependency to `0.2.21`.
- Preserve compatible Neuropixels probe codes when loading and displaying channel maps.
- Improve documentation and notebook text for clarity.

### Fixed

- Fix Bokeh dropdown menu CSS for newer Bokeh versions.
- Fix Neuropixels IMRO parsing and writing for gain validation, and NP24 channel output.
- Fix Neuropixels probe type metadata and aliases.
- Fix probe-interface import/export reference handling.
- Fix atlas integration for `brainglobe-atlasapi` version 3 while retaining compatibility with older versions.
- Improve atlas structure loading through BrainGlobe AtlasAPI and handle empty region overlays.

## 0.2.1

2026/03/23

### Fixed

- Fix NP1 read/write imro file.
- Fix NP21 read imro file.
- Document improve.

## 0.2

2025/11/17

### Added

- Add move probe types, including type with code `2003`, `2010`, `2013`.
- Gif demonstration on README page 

### Changed

- `CartoApp.load_use_config` and `load_view_config` add keyword parameter `reset`.
- probe io and electrode-channel-mapping functions.
- view config file save working process.
- Rename term usage from `slice` to `projection` in atlas-related modules.
- Change term usage in probe references.

### Fixed

- fix `probe_pnx.npx.e2c`
- Document improve.

### Removed

- some attributes and properties in `npx.ProbeType`


## 0.1

- Initial release

---------------------------------------

Template below

## VERSION

### Added

### Changed

### Fixed

### Removed
