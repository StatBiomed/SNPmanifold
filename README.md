# SNPmanifold

SNPmanifold is a Python package that learns a representative manifold for single cells based on their SNPs (Single-Nucleotide Polymorphisms) using VAE (Variational AutoEncoder) and UMAP (Uniform Manifold Approximation and Projection). It takes AD matrix, DP matrix, and VCF (or variant_name.tsv) as inputs. You can compile them from bam file(s) either conveniently by cellSNP-lite or by your custom scripts.

SNPmanifold first performs simple filtering on AD matrix and DP matrix for high-quality cells and SNPs. It then trains VAE and UMAP to learn a representative manifold for single cells according to their SNP-allelic ratios (AD/DP). Finally, it classifies cells into clones and infer their phylogeny based on the manifold. 

## Installation

Credits to Xinyi Lin. 

```python3
conda create -n $myenv python=3.8
conda activate $myenv
conda install matplotlib networkx numpy pandas scipy seaborn scikit-learn pytorch torchvision torchaudio cpuonly -c pytorch
pip install umap-learn
```

Replace `$myenv` with the environment name you prefer.

Alternatively, you can now install >=0.0.1 version via this command line:

```bash
pip install -U git+https://github.com/StatBiomed/SNPmanifold
```

## Quick Usage

1. Import SNPmanifold and create an object of the class SNP_VAE.

2. Run 4 methods (filtering, training, clustering, phylogeny) in order.

Each method can rerun sperately without reruning prior methods. `SNPmanifold_demo.ipynb` shows a demo for quick usage of SNPmanifold on MKN45 cancer cell line using mitochondrial SNPs.

## Key Parameters

```python
from SNPmanifold import SNP_VAE
```

### SNP_VAE():

`path`

`AD`

`DP` 

`VCF` 

`variant_name`

### training():

`num_epoch` 

`stepsize`

`z_dim`

`num_batch`

### clustering():

`max_cluster`

### phylogeny():

`cluster_no`

`SNP_no`

## Key Attributes

`cell_filter`

`SNP_filter`

`pc`

`embedding_2d`

`embedding_3d`

`assigned_label`
