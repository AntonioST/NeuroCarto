Blueprint Script View
=====================

**It is an experimental feature.** Run custom script.

.. image:: _static/blueprint-script.png

Commandline options
-------------------

use ``--view=script`` to enable.

Configurations
--------------

It uses user config. It is looked like::

    {
      "BlueprintScriptView": {
        "actions": {
          "my": "tests:bp_example:my_blueprint_script_function"
        }
      }
    }


With above config, you have an action named ``my``, and it will run the function ``my_blueprint_script_function``.


How to create a script?
-----------------------

Please check the file ``tests/bp_example.py``

