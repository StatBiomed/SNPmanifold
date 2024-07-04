===
API
===

.. automodule:: SNPmanifold

Import SNPmanifold as::

   import SNPmanifold

Main Object
-----------

Objects of type :class:`~SNPmanifold.SNP_VAE` for clustering with binomial
mixture model

Attributes
~~~~~~~~~~~~~~~~~~~~

**cell_total** (integer) - total number of cells after filtering
**latent** (np.array of shape(cell_total, z)) - latent factors of all cells after filtering

.. autoclass:: SNPmanifold.SNP_VAE
   :members: 
