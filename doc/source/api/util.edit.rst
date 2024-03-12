neurocarto.util.edit
====================

It is an internal package for providing functions to the :class:`~neurocarto.util.util_blueprint.BlueprintFunctions`


.. toctree::
    :maxdepth: 1
    :caption: Blueprint functions:

    util.edit.script
    util.edit.checking
    util.edit.clustering
    util.edit.data
    util.edit.category
    util.edit.moving
    util.edit.surrounding
    util.edit.probe
    util.edit.plot
    util.edit.atlas
    util.edit.actions
    util.edit.debug

How to write a blueprint script function
========================================

Real examples are:

* ``src/neurocarto/util/edit/_actions.py``
* ``tests/bp_example.py``

Prepare a python file ``example.py`` with following contents.

.. code-block:: python

    from neurocarto.util.util_blueprint import BlueprintFunctions
    def my_blueprint_script_function(bp: BlueprintFunctions, a0: str, a1: int):
        """
        Script Document, which is used in GUI.

        :param bp:
        :param a0: (str) the first string argument
        :param a1: (int) the second int argument
        """
        bp.log_message(f'{a0=}', f'{a1=}')

Add ``example.py`` into ``neurocarto.config.json``.

.. code-block:: json

    {
      "BlueprintScriptView": {
        "actions" : {
          "my": "example:my_blueprint_script_function"
        }
      }
    }

Start an application with the commandline option ``--view=script``,
then you can test your script.
