# SNPmanifold

SNPmanifold is a Python package that learns a representative manifold for single cells based on their SNPs (Single-Nucleotide Polymorphisms) using VAE (Variational AutoEncoder) and UMAP (Uniform Manifold Approximation and Projection). It takes AD matrix, DP matrix, and VCF (or variant_name.tsv) as inputs. You can compile them from bam file(s) either conveniently by cellSNP-lite or by your custom scripts.

SNPmanifold first performs simple filtering on AD matrix and DP matrix for high-quality cells and SNPs. It then trains VAE and UMAP to learn a representative manifold for single cells according to their allele frequency of different SNPs (AF = AD/DP). Finally, it classifies cells into clones and infer their phylogeny based on the manifold. 

## Installation

Quick install can be achieved via pip

```bash
# for published version
pip install -U SNPmanifold

# or developing version
pip install -U git+https://github.com/StatBiomed/SNPmanifold
```

Or set a conda environment before installing (credits to Xinyi Lin).
Replace `$myenv` with the environment name you prefer.

```bash
conda create -n $myenv python=3.8
conda activate $myenv

pip install -U git+https://github.com/StatBiomed/SNPmanifold
```

## Quick Usage

Full documentation is at https://SNPmanifold.readthedocs.io. 

Here is a quick start:

1. Import SNPmanifold and create an object of the class SNP_VAE.

```python
from SNPmanifold import SNP_VAE
```

2. Run 4 methods (filtering, training, clustering, phylogeny) in order.

  Each method can rerun sperately without reruning prior methods. 

* The [demo page](https://snpmanifold.readthedocs.io/en/latest/SNPmanifold_demo.html) 
  and notebook [SNPmanifold_demo.ipynb](./SNPmanifold_demo.ipynb) show 
  a demo for quick usage of SNPmanifold on MKN45 cancer cell line using 
  mitochondrial SNPs.

* See how to use it via the [API page](https://snpmanifold.readthedocs.io/en/latest/API.html#main-object).
