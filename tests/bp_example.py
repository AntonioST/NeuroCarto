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
