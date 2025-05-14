import matplotlib as mpl

from .utils_io import load_data
from .utils_tools import filter_data, summary_filtering, train_VAE, summary_training, umap_retrain, latent_clustering, summary_clustering, tree, summary_phylogeny, scatter_AF, heatmap_SNP, heatmap_cluster

class SNP_VAE:
    
    def __init__(self, path = None, SNP_mask = [], AD = None, DP = None, VCF = None, variant_name = None, SNPread = "unnormalized", missing_value = 'neighbour', cell_weight = "unnormalized", prior = None, UMI_correction = None):
        
        """
        Load AD and DP matrices, VCF.gz file or variant_name.tsv file for subsequent analyses in SNP_VAE 

        Parameters
        ----------
        path: string
            path of cellSNP-lite output folder which contains cellSNP.tag.AD.mtx, cellSNP.tag.DP.mtx, and cellSNP.base.vcf.gz

        SNP_mask: list of string
            list of variant names to mask from VAE, please refer to the internal variant names: VCF['TEXT'], if you use VCF as input (default: [])

        AD: string
            path of AD matrix in scipy.sparse.coo_matrix format

        DP: string
            path of DP matrix in scipy.sparse.coo_matrix format

        VCF: string
            path of VCF.gz file

        variant_name: string
            path of variant_name.tsv file which is a list of custom variant name stored in pandas dataframe without header and index
            
        SNPread: string
            optional observed-SNP normalization, 'normalized' or 'unnormalized' (default: 'normalized')
        
        missing_value: float between 0 and 1, or string 'mean' or 'neighbour'
            impute value for missing allele frequency in AF matrix, i.e. DP = 0 (default: 0.5)
        
        cell_weight: string
            optional cost normalization for each cell, 'normalized' or 'unnormalized' (default: 'unnormalized')

        prior: string
            path of prior weights of mutation for each variant in csv format

        UMI_correction: string
            add pseudocounts to AD and DP matrices in the model, None or 'positive' or 'negative' (default: None)

        """

        self.SNPread = SNPread
        self.missing_value = missing_value
        self.cell_weight = cell_weight
        self.UMI_correction = UMI_correction
        load_data(self, path, SNP_mask, AD, DP, VCF, variant_name, prior)
        
    def filtering(self, save_memory = False, cell_SNPread_threshold = None, SNP_DPmean_threshold = None, SNP_logit_var_threshold = None, filtering_only = False, num_neighbour = 3, what_to_do = 'skip'):
        
        """
        Filter low quality cells and SNPs based on number of observed SNPs for each cell, mean coverage of each SNP, and logit-variance of each SNP

        Parameters
        ----------
        save_memory: boolean
            if True, raw matrices and VCF will be deleted from the object to save memory (default: False)

        cell_SNPread_threshold: float
            minimal number of observed SNPs for a cell to be included for analysis, input after showing the plot if None (default: None)

        SNP_DPmean_threshold: float
            minimal cell-average coverage for a SNP to be included for analysis, input after showing the plot if None (default: None)

        SNP_logit_var_threshold: float
            minimal logit-variance for a SNP to be included for analysis, input after showing the plot if None (default: None)

        filtering_only: boolean
            if True, it does not process AF matrices which are required for subsequent analyses in order to speed up (default: False)

        num_neighbour: integer
            for missing_value = neighbour only, number of neighbouring cells for imputation (default: 3)

        what_to_do: string
            what to do for cells with 0 oberserved SNPs after filtering (default: 'skip')
        """
        
        filter_data(self, save_memory, cell_SNPread_threshold, SNP_DPmean_threshold, SNP_logit_var_threshold, filtering_only, num_neighbour, what_to_do)
        
    def training(self, num_epoch = 2000, stepsize = 0.0001, z_dim = None, beta = 0, num_batch = 5, is_cuda = True):
        
        """
        Train VAE using Adam optimizer and visualize latent space using PCA and UMAP

        Parameters
        ----------
        num_epoch: integer
            number of epochs for training VAE (default: 2000)

        stepsize: float
            stepsize of Adam optimizer (default: 0.0001)

        z_dim: integer
            dimension of latent space (default: half of number of filtered SNPs)

        beta: float
            strength of standard Gaussian prior in cost of VAE (default: 0)

        num_batch: integer
            number of batchs for training VAE (default: 5)

        is_cuda: boolean
            Set True if you want to use CUDA, set False if you want to use CPU (default: True)

        """
        
        train_VAE(self, num_epoch, stepsize, z_dim, beta, num_batch, is_cuda)
        
    def retrain_umap(self):
        
        """Re-train UMAP in the same latent space of VAE"""
        
        umap_retrain(self)
        
    def clustering(self, algorithm = "leiden_full", max_cluster = 15, resolution = 1):
        
        """
        Cluster cells using k-means clustering or Leiden clustering in SCANPY, in either full-dimensional latent space or 3D UMAP

        Parameters
        ----------
        algorithm: string 
            'kmeans_umap3d', 'kmeans_full', 'leiden_umap3d', or 'leiden_full' (default: 'leiden_full')

        max_cluster: integer 
            for k-means clustering only, maximum number of clusters (default: 15)

        resolution: float 
            for Leiden clustering only, resolution of clusters (default: 1)

        """

        latent_clustering(self, algorithm, max_cluster, resolution)
        
    def phylogeny(self, cluster_no = 2, pair_no = 100, SNP_no = 50, bad_color = "blue", cmap_heatmap = mpl.colormaps['rocket'], SNP_ranking = 'AF_diff'):
        
        """
        Construct phylogenetic tree of cells in full-dimensional latent space and rank SNPs according to p-values

        Parameters
        ----------
        cluster_no: integer
            for k-means clustering only, number of clusters for phylogenetic tree construction and ranking of SNPs (default: 2)

        pair_no: integer
            number of pair of cells to consider between each pair of clusters when constructing phylogenetic tree (default: 100)

        SNP_no: integer
            number of top-ranked SNPs to be visualized in heatmap (default: 50)

        bad_color: string
            color of heatmap when allele frequency is missing, i.e. DP = 0 (default: 'blue')

        cmap_heatmap: mpl.colormaps
            colormap used for heatmap visualization (default: mpl.colormaps['rocket'])

        SNP_ranking: string
            method for ranking SNPs, 'variance' or 'AF_diff'

        """
        
        tree(self, cluster_no, pair_no, SNP_no, bad_color, cmap_heatmap, SNP_ranking)
        
    def filtering_summary(self, dpi = mpl.rcParams['figure.dpi']):
        
        """
        Re-display figures shown in filtering with higher dpi

        Parameters
        ----------
        dpi: float
            dpi resolution for figures

        """
        
        summary_filtering(self, dpi)
        
    def training_summary(self, dpi = mpl.rcParams['figure.dpi']):
        
        """
        Re-display figures shown in training with higher dpi

        Parameters
        ----------
        dpi: float
            dpi resolution for figures

        """
        
        summary_training(self, dpi)
        
    def clustering_summary(self, dpi = mpl.rcParams['figure.dpi']):
        
        """
        Re-display figures shown in clustering with higher dpi

        Parameters
        ----------
        dpi: float
            dpi resolution for figures

        """
        
        summary_clustering(self, dpi)
        
    def phylogeny_summary(self, SNP_no = None, dpi = mpl.rcParams['figure.dpi'], bad_color = "blue", fontsize_c = None, fontsize_x = None, fontsize_y = None, cmap_heatmap = mpl.colormaps['rocket'], SNP_ranking = 'AF_diff'):
        
        """
        Re-display figures shown in phylogeny with higher dpi, different number of SNPs, color and fontsizes

        Parameters
        ----------
        SNP_no: integer
            number of top-ranked SNPs to be visualized in heatmap (default: 50)

        dpi: float
            dpi resolution for figures

        bad_color: string
            color of heatmap when allele frequency is missing, i.e. DP = 0 (default: 'blue')

        fontsize_c: float
            fontsize of cluster labels on heatmap

        fontsize_x: float
            fontsize of cell labels on heatmap

        fontsize_y: float
            fontsize of SNP labels on heatmap

        cmap_heatmap:
            colormap used for heatmap visualization (default: mpl.colormaps['rocket'])

        SNP_ranking: string
            method for ranking SNPs, 'variance' or 'AF_diff' (default: 'AF_diff')

        """
        
        summary_phylogeny(self, SNP_no, dpi, bad_color, fontsize_c, fontsize_x, fontsize_y, cmap_heatmap, SNP_ranking)
        
    def AF_scatter(self, SNP_name, dpi = mpl.rcParams['figure.dpi']):
    
        """
        Visualize allele frequency of one particular SNP in latent space

        Parameters
        ----------
        SNP_name: string
            name of the SNP to visualize

        dpi: float
            dpi resolution for figure

        """
        
        scatter_AF(self, SNP_name, dpi)

    def SNP_heatmap(self, SNP_name, dpi = mpl.rcParams['figure.dpi'], bad_color = "blue", fontsize_c = None, fontsize_x = None, fontsize_y = None, cmap_heatmap = mpl.colormaps['rocket']):

        """
        Visualize allele frequency of specific SNPs in heatmap

        Parameters
        ----------
        SNP_name: list
            list of names of the SNPs to visualize

        dpi: float
            dpi resolution for figures

        bad_color: string
            color of heatmap when allele frequency is missing, i.e. DP = 0 (default: 'blue')

        fontsize_c: float
            fontsize of cluster labels on heatmap

        fontsize_x: float
            fontsize of cell labels on heatmap

        fontsize_y: float
            fontsize of SNP labels on heatmap

        cmap_heatmap:
            colormap used for heatmap visualization (default: mpl.colormaps['rocket'])
        """

        heatmap_SNP(self, SNP_name, dpi, bad_color, fontsize_c, fontsize_x, fontsize_y, cmap_heatmap)

    def cluster_heatmap(self, cluster_order, SNP_no = 50, dpi = mpl.rcParams['figure.dpi'], bad_color = "blue", fontsize_c = None, fontsize_x = None, fontsize_y = None, cmap_heatmap = mpl.colormaps['rocket'], SNP_ranking = 'AF_diff'):

        """
        Visualize allele frequency of specific clusters in heatmap

        Parameters
        ----------
        cluster_order: list
            list of clusters to visualize

        SNP_no: integer 
            number of top-ranked SNPs to be visualized in heatmap (default: 50)

        dpi: float
            dpi resolution for figures

        bad_color: string
            color of heatmap when allele frequency is missing, i.e. DP = 0 (default: 'blue')

        fontsize_c: float
            fontsize of cluster labels on heatmap

        fontsize_x: float
            fontsize of cell labels on heatmap

        fontsize_y: float
            fontsize of SNP labels on heatmap

        cmap_heatmap:
            colormap used for heatmap visualization (default: mpl.colormaps['rocket'])

        SNP_ranking: string
            method for ranking SNPs, 'variance' or 'AF_diff' (default: 'AF_diff')
        """

        heatmap_cluster(self, cluster_order, SNP_no, dpi, bad_color, fontsize_c, fontsize_x, fontsize_y, cmap_heatmap, SNP_ranking)
