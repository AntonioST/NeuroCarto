.. _atlas:

Atlas Mouse Brain
=================

Require optional dependency `bg-atlasapi`_.

.. _bg-atlasapi: https://github.com/brainglobe/bg-atlasapi

Install
-------

::

    pip install bg-atlasapi

Commandline options
-------------------

use `--atlas` to specific which atlases.
You can use `bg_atlasapi.show_atlases()` to show all available atlases.
For the mouse brain atlases, you can give `--atlas=25` to specific use `allen_mouse_25um` atlas.

For first running, it needs to download the corresponding atlas for a while.

Application View
----------------

Once `bg-atlasapi` installed, you can find a brain slice shown in the middle figure.

Controls
~~~~~~~~

* moving: use the box-edit tool |bk-tool-icon-box-edit| in figure toolbar.
* scaling:
* rotating:
* change slice:

.. |bk-tool-icon-box-edit| image:: _static/bk-tool-icon-box-edit.png
