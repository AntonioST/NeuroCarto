View Components
===============

Custom view components can be enabled from the command line with ``--view`` or added to the user config.

User configuration
------------------

The application reads ``~/.config/neurocarto/neurocarto.config.json`` (Linux), ``AppData\\neurocarto\\neurocarto.config.json`` (Windows),
or ``~/.neurocarto/neurocarto.config.json``.
If ``--debug`` is enabled, it uses ``.neurocarto.config.json`` in the current working directory.

The config json file looks like this. ::

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
