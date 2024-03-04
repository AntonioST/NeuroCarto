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
    """
    Script Document, which is used in GUI.

    :param bp:
    :param a0: the first string argument
    :param a1: the second int argument
    """
    bp.check_probe("npx", 24)

    bp.log_message(f'{a0=}', f'{a1=}')

    bp.log_message('done')


def example_generator_function(bp: BlueprintFunctions):
    """
    Generator script example.

    :param bp:
    """
    for i in range(10):
        bp.set_status_line(f'{i} second')
        yield 1

    bp.log_message('done')


def example_indirect_call(bp: BlueprintFunctions, script: str, *args, **kwargs):
    bp.call_script(script, *args, **kwargs)
