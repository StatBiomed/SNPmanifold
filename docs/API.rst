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

.. autoclass:: SNPmanifold.SNP_VAE
   :members: __init__, filtering, training, clustering, phylogeny

VAE module
----------

Objects of type :class:`~SNPmanifold.VAE_base` for clustering with binomial
mixture model

.. autoclass:: SNPmanifold.VAE_base
   :members: __init__, encode, decode, reparameterize, forward

