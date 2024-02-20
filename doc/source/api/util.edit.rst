chmap.util.edit
===============

It is an internal package for providing functions to the :class:`~chmap.util.util_blueprint.BlueprintFunctions`


.. toctree::
    :maxdepth: 1
    :caption: Blueprint functions:

    util.edit.checking
    util.edit.clustering
    util.edit.data
    util.edit.moving
    util.edit.surrounding
    util.edit.actions
    util.edit.debug

How to write a blueprint script function
========================================

Real examples are:

* `src/chmap/util/edit/_actions.py`
* `tests/bp_example.py`

Prepare a python file `example.py` with following contents.

.. code-block:: python

    from chmap.util.util_blueprint import BlueprintFunctions
    def my_blueprint_script_function(bp: BlueprintFunctions, arg: str):
        """\
        Script Document, which is used in GUI.
        input: "A0,A1"
        A0: (str) the first string argument
        A1: (int) the second int argument
        """
        # check script input: arg
        args = [it.strip() for it in arg.split(',')]
        a0 = bp.arg.get_value('a0', args, 0)
        a1 = bp.arg.get_value('a1', args, 1, int)
        bp.log_message(f'{a0=}', f'{a1=}')

        ...

Add `example.py` into `chmap.config.json`.

.. code-block:: json

    {
      "BlueprintScriptView": {
        "actions" : {
          "my": "example:my_blueprint_script_function"
        }
      }
    }

Start an application with the commandline option `--view=script`,
then you can test your script.
