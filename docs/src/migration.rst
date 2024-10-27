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
    species_indices = torch.tensor([[0, 1, 1, 0]])
    coords = torch.tensor(...)
    model = torchani.models.ANI1x()

    species_indices, energies = model((species_indices, coords))
    # or
    energies = model((species_indices, coords)).energies

You can now do:

.. code-block:: python
    
    import torchani
    atomic_nums = torch.tensor([[1, 6, 6, 1]])
    coords = torch.tensor(...)
    model = torchani.models.ANI1x()

    result = model.sp(atomic_nums, coords)
    energies = result["energies"]

Here "sp" stands for a "single-point calculation" (typical chemistry jargon). This was
changed since it allows models to output more than a single scalar value, which is
necessary e.g. for models that output charges. Additionally, the new version is simpler
and less error prone, and it allows for outputting forces and hessians without any
familiarity with torch (no need to do anything with the ``requires_grad`` flag of
tensors). Calling a model directly is still possible, but *is strongly discouraged and
may be removed in the future*.

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

Creating models for training with ``torchani.nn.Sequential``
------------------------------------------------------------

The ``Sequential`` class is still available, but *its use is
highly discouraged*. If you want to create a custom model, we recommend you
create your ``torch.nn.Module``. This is much more flexible and less error
prone, and it avoids having to return irrelevant outputs and accept irrelevant inputs.

If you were previously doing:

.. code-block:: python

    import torchani
    aev_computer = torchani.AEVComputer(...)
    neural_networks = torchani.ANIModel(...)
    energy_shifter = torchani.EnergyShifter(...)
    model = torchani.nn.Sequential(aev_computer, neural_networks, energy_shifter)

You can now do:

.. code-block:: python

    from torch.nn import Module
    import torchani

    class Model(Module):
        def __init__(self):
            self.converter = torchani.SpeciesConverter(...)
            self.featurizer = torchani.AEVComputer(...)
            self.nn = torchani.ANIModel(...)
            self.adder = torchani.EnergyAdder(...)

        def forward(self, atomic_nums, coords, cell, pbc):
            elem_idxs = self.converter(atomic_nums)
            aevs = self.featurizer(elem_idxs, coords, cell, pbc)
            energies = self.nn(elem_idxs, aevs)
            return energies + self.adder(elem_idxs)

    model = Model()

Which will have the same effect, but is much more flexible. As an alternative, you can
use the torchani ``Assembler`` to create your model. For example, to create a model just
like ``ANI2x``, but with random weights, use:

.. code-block:: python

    import torchani
    from torchani.aev import StandardRadial, StandardAngular
    from torchani import atomics
    from torch.nn import Module

    asm = torchani.Assembler()
    asm.set_symbols(("H", "C", "N", "O"))
    asm.set_featurizer(
        radial_terms=StandardRadial.like_2x(),
        angular_terms=StandardAngular.like_2x(),
        compute_strategy="cuaev",  # Use the cuAEV extension for faster training
    )
    asm.set_atomic_networks(atomics.like_2x)
    # This will ensure the assembled model adds the ground state atomic energies
    # for this level of theory
    asm.set_gsaes_as_self_energies("wb97x-631gd")

    model = asm.assemble()  # The assembled model is ready to train
