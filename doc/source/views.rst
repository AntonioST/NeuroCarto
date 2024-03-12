View Components
===============

Custom view components can be enabled by commandline via ``--view`` or put it into user config.

User configuration
------------------

The application read file ``~/.config/neurocarto/neurocarto.config.json`` (linux), ``AppData\\neurocarto\\neurocarto.config.json`` (windows)
or ``~/.neurocarto/neurocarto.config.json``.
If ``--debug`` enable, use ``.neurocarto.config.json`` at current working directory.

The config json file be looked like ::

    {
      "CartoApp": {
        "theme": null,
        "views": []
      }
    }

You can put custom components in ``CartoApp::views``.

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
