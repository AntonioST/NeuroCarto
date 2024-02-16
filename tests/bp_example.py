"""

Global Config setting
---------------------

.. code-block:: json

    {
      "BlueprintScriptView": {
        "actions" : {
          "my": "tests:bp_example:my_blueprint_script_function"
        }
      }
    }
"""

from chmap.util.util_blueprint import BlueprintFunctions


def my_blueprint_script_function(bp: BlueprintFunctions, a0: str, a1: int):
    """\
    Script Document, which is used in GUI.

    :param a0: the first string argument
    :param a1: the second int argument
    """
    bp.check_probe("npx", 24)

    bp.log_message(f'{a0=}', f'{a1=}')

    bp.log_message('done')
