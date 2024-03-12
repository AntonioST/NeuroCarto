NeuraCarto package
==================

.. toctree::
    :maxdepth: 1
    :caption: Modules:

    main_app
    main_index
    config
    files

.. toctree::
    :maxdepth: 1
    :caption: Probe Abstract:

    probe

.. toctree::
    :maxdepth: 1
    :caption: Probe Implementation:

    probe_npx

.. toctree::
    :maxdepth: 2
    :caption: UI components:

    views

.. toctree::
    :maxdepth: 2
    :caption: Utilities:

    util

Symbols
-------

The symbols and expressions used the document.

Numpy Array
~~~~~~~~~~~

`Array[Dtype, *Shapes]`

**Examples**

* `Array[int, 3, 2]` is a `3x2` int number array.
* `Array[index:int, N]` is a 1-d (`N`-length) int index array.
* `Array[int, N, (shank, col, row, state, category)]` is a `Nx5` int array with 5 columns (fields).
* `Array[float, [S,], R, C]` is either a 2-d (shape `(R, C)`) or a 3-d (shape `(S, R, C)`) float array.
* `Array[V, ..., N, ...]` is a multi-dimension `V`-domain array, which has one `N`-length axis.
* `Array[bool, N]` is a `bool` N-length array, often used as a mark.

**Use examples**

* An array is applied with a mask::

    a = np.arange(10)
    a = a[a > 5]

  We can document the `__getitem__` of a numpy array as::

    def __getitem__(self: Array[V, N], mark: Array[bool, N]) -> Array[V, M]:
        """
        V : any value type
        N : N-length array
        M = np.nonzero(mask)
        """
