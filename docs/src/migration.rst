Migrating from TorchANI 2.0
===========================

If you were using a previous version of TorchANI there may be some necessary
modifications to your code. We strive to keep backwards compatibility for the most part,
but some breaking changes were necessary in order to support improvements in the models,
the dataset loading and managing procedure, etc.

Minor versions changes of ``torchani`` will attempt to be fully backwards compatible
going forward, and breaking changes will be reserved for major releases.

Here we document the most important breaking changes, and what you can do to modify your
code to be compatible with version 3.0. Additionally, we provide recommendations to use
the new, more user-friendly API, when appropriate.

General usage of built-in ANI models
------------------------------------

If you were previously calling torchani models as:

.. code-block:: python
    
    import torchani
    model = torchani.models.ANI1x()
    species_indices = torch.tensor([[0, 1, 1, 0]])
    coords = torch.tensor(...)
    model = torchani.models.ANI1x()

    species_indices, energies = model((species_indices, coords))
    # or
    energies = model((species_indices, coords)).energies

You can now do:

.. code-block:: python
    
    import torchani
    m = torchani.models.ANI1x()
    atomic_nums = torch.tensor([[1, 6, 6, 1]])
    coords = torch.tensor(...)
    m = torchani.models.ANI1x()

    # Here "sp" stands for a single-point calculation (typical chemistry jargon)
    result = model.sp(atomic_nums, coords)
    energies = result["energies"]

This was changed since it allows models to output more than a single scalar value, which
is necessary e.g. for models that output charges. Additionally, the new version is
simpler and less error prone, and it allows for outputting forces and hessians without
any familiarity with torch (no need to do anything with the ``requires_grad`` flag of
tensors).

To output other quantities of interest use:

.. code-block:: python
    
    result = model.sp(atomic_nums, coords, forces=True, hessians=True)
    atomic_charges = result["atomic_charges"]  # Only for models that support this
    energies = result["energies"]
    forces = result["forces"]
    hessians = result["hessians"]

TODO: Add more stuff

The `AEVComputer` class
-----------------------

TODO: Add

The ``torchani.ANIModel`` class
-------------------------------

TODO: Add

Usage of ``torchani.data``
--------------------------

TODO: Add

