# SNPmanifold
SNPmanifold is a Python package that learns a representative manifold for single cells based on their SNPs (Single-Nucleotide Polymorphisms) using VAE (Variational AutoEncoder). It takes AD matrix, DP matrix, and VCF (or variant_name.tsv) as inputs. You can compile them from bam file(s) either conveniently by cellSNP-lite or by your custom scripts.

SNPmanifold first performs simple filtering on AD matrix and DP matrix for high-quality cells and SNPs. It then trains a VAE to learn a representative manifold for single cells according to their SNP-allelic ratios (AD/DP). Finally, it classifies cells into clones and infer their phylogeny based on the manifold. 

## Installation
Credits to Xinyi Lin. 

```python3
conda create -n $myenv python=3.8
conda activate $myenv
conda install matplotlib networkx numpy pandas scipy seaborn scikit-learn pytorch torchvision torchaudio cpuonly -c pytorch
pip install umap-learn
```

Replace `$myenv` with the environment name you prefer.

## Quick Usage

1. Import SNPmanifold and create an object of the class SNP_VAE.

2. Run 4 methods (filtering, training, clustering, phylogeny) in order.

Each method can rerun sperately without reruning prior methods. `SNPmanifold_demo.ipynb` shows a demo for quick usage of SNPmanifold.

## Key Attributes
