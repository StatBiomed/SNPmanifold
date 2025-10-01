# SNPmanifold

SNPmanifold is a Python package that learns a representative manifold for single cells based on their SNPs (Single-Nucleotide Polymorphisms) using VAE (Variational AutoEncoder) and UMAP (Uniform Manifold Approximation and Projection). It takes AD matrix, DP matrix, and VCF (or variant_name.tsv) as inputs. You can compile them from bam file(s) either conveniently by cellSNP-lite or by your custom scripts.

SNPmanifold first performs simple filtering on AD matrix and DP matrix for high-quality cells and SNPs. It then trains VAE and UMAP to learn a representative manifold for single cells according to their allele frequency of different SNPs (AF = AD/DP). Finally, it classifies cells into clones and infer their phylogeny based on the manifold. 

## Installation

Quick install can be achieved via pip (python 3.8 needed)

```bash
# for published version
pip install -U SNPmanifold==0.0.9

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

## FAQ

1. How to choose the filtering criteria for high-quality cells and SNPs?

   The motivation of filtering is to reduce the amount of noisy low-quality cells and SNPs so that the resulting embedding is cleaner. The

2. How to re-display figures in higher dpi?

   You can use functions filtering_summary(dpi = 300), training_summary(dpi = 300), clustering_summary(dpi = 300), phylogeny_summary(dpi = 300).

3. What to do when the embedding of SNPmanifold fails to converge during training?

   You can tune hyperparameters of the optimizer to fix the problem.

## Citation
Chung, H., Huang, Y. SNPmanifold: detecting single-cell clonality and lineages from single-nucleotide variants using binomial variational autoencoder. Genome Biol 26, 309 (2025). https://doi.org/10.1186/s13059-025-03803-3
