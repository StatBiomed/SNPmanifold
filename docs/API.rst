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
-----------

After running SNP_VAE.filtering():
~~~~~~~~~~~~~~~~~~~~

**cell_filter** (np.array of booleans) - boolean filter for all input cells

**SNP_filter** (np.array of booleans) - boolean filter for all input SNPs

**cell_total** (integer) - total number of cells after filtering

**SNP_total** (integer) - total number of SNPs after filtering

**AD_filtered** (np.array with shape (cell_total, SNP_total)) - AD matrix after filtering

**DP_filtered** (np.array with shape (cell_total, SNP_total)) - DP matrix after filtering

**AF_filtered** (torch.tensor with shape (cell_total, SNP_total)) - AF matrix which is the input to VAE

**VCF_filtered** (pd.DataFrame) - VCF after filtering which contains variant names

After running SNP_VAE.training():
~~~~~~~~~~~~~~~~~~~~

**model** (VAE_normalized) - trained VAE model implemented in PyTorch

**latent** (np.array with shape (cell_total, z_dim)) - latent factors of all cells after filtering

**pc** (np.array with shape (cell_total, z_dim)) - principal components of PCA of the latent space

**embedding_2d** (np.array with shape (cell_total, 2)) - 2D UMAP embedding of the latent space

**embedding_3d** (np.array with shape (cell_total, 3)) - 3D UMAP embedding of the latent space

After running SNP_VAE.clustering() and SNP_VAE.phylogeny():
~~~~~~~~~~~~~~~~~~~~

**cluster_no** (integer) - total number of clusters 

**assigned_label** (np.array of integers with shape (cell_total)) - assigned cluster labels of all cells after filtering

**clusters** (list of np.arrays with length (cluster_no)) - np.where(assigned_label == r) for each cluster r

**colors** (np.array with shape (cluster_no, 4)) - colors of all clusters in figures

**edge** (list of tuples with length (cluster_no - 1)) - all connected edges in the cluster phylogenetic tree

**f_stat** (np.array with shape (SNP_total)) - F-statistics of all SNPs after filtering

**p_value** (np.array with shape (SNP_total)) - P-values of all SNPs after filtering

**rank_SNP** (np.array with shape (SNP_total)) - ranking of all SNPs after filtering from the lowest p-value to the highest

Functions
-----------

.. autoclass:: SNPmanifold.SNP_VAE
   :members: 
