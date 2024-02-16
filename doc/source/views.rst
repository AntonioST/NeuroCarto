View Components
===============

Custom view components can be enabled by commandline via ``--view`` or put it into user config.

User configuration
------------------

The application read file ``~/.config/chmap/chmap.config.json`` (linux), ``AppData\\chmap\\chmap.config.json`` (windows)
or ``~/.chmap/chmap.config.json``.
If ``--debug`` enable, use ``.chmap.config.json`` at current working directory.

The config json file be looked like ::

    {
      "ChannelMapEditorApp": {
        "theme": null,
        "views": []
      }
    }

You can put custom components in ``ChannelMapEditorApp::views``.

Builtin Components
------------------

.. toctree::
    :maxdepth: 2
    :caption: Common Components

    atlas
    blueprint

.. toctree::
    :maxdepth: 2
    :caption: Neuropixels specific Components

    npx_reference
    npx_density
    npx_ceff

.. toctree::
    :maxdepth: 2
    :caption: Experimental Components

    image_file
    image_plt
    blueprint_script

Customize Components
--------------------

.. toctree::
    :maxdepth: 2

    notebooks/Extending_View
