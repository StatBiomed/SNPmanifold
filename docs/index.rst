|PyPI| |Docs| |Build Status|

.. |PyPI| image:: https://img.shields.io/pypi/v/SNPmanifold.svg
    :target: https://pypi.org/project/SNPmanifold
.. |Docs| image:: https://readthedocs.org/projects/SNPmanifold/badge/?version=latest
   :target: https://SNPmanifold.readthedocs.io
.. |Build Status| image:: https://travis-ci.org/huangyh09/SNPmanifold.svg?branch=master
   :target: https://travis-ci.org/huangyh09/SNPmanifold
   
====
Home
====


About SNPmanifold
=================

SNPmanifold is a Python package that learns a representative manifold for single cells based on their SNPs (Single-Nucleotide Polymorphisms) using VAE (Variational AutoEncoder) and UMAP (Uniform Manifold Approximation and Projection). It takes AD matrix, DP matrix, and VCF (or variant_name.tsv) as inputs. You can compile them from bam file(s) either conveniently by cellSNP-lite or by your custom scripts.

SNPmanifold first performs simple filtering on AD matrix and DP matrix for high-quality cells and SNPs. It then trains VAE and UMAP to learn a representative manifold for single cells according to their allele frequency of different SNPs (AF = AD/DP). Finally, it classifies cells into clones and infer their phylogeny based on the manifold.


References
==========

* Hoi Man Chung and Yuanhua Huang. `Interpretable variational encoding of 
  genotypes identifies comprehensive clonality and lineages in single cells 
  geometrically <https://>`_.
  \ **BioRxiv**\ to appear.


.. toctree::
   :caption: Main
   :maxdepth: 1
   :hidden:
   
   index
   install
   API
   release

.. toctree::
   :caption: Examples
   :maxdepth: 1
   :hidden:

   SNPmanifold_demo

