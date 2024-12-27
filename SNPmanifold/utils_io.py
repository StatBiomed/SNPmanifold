import gzip
import io
import numpy as np
import pandas as pd
from scipy.io import mmread
import torch
import warnings

def read_VCF_gz(path):
    
    """
    Read VCF.gz file into pandas dataframe
    
    Parameters
    ----------
    path: string
        path of VCF.gz file

    """

    with gzip.open(path, 'rb') as f:
        
        lines = [l.decode('utf-8') for l in f if not l.startswith(b'##')]

    return pd.read_csv(
        io.StringIO(''.join(lines)),
        dtype={'#CHROM': str, 'POS': int, 'ID': str, 'REF': str, 'ALT': str,
               'QUAL': str, 'FILTER': str, 'INFO': str},
        sep='\t'
    ).rename(columns={'#CHROM': 'CHROM'})


def load_data(self, path, SNP_mask, AD, DP, VCF, variant_name, prior):
    
    """
    Load AD and DP matrices, VCF.gz file or variant_name.tsv file for subsequent analyses in SNP_VAE 
    
    Parameters
    ----------
    path: string
        path of cellSNP-lite output folder which contains cellSNP.tag.AD.mtx, cellSNP.tag.DP.mtx, and cellSNP.base.vcf.gz
        
    SNP_mask: list of string
        list of variant names to mask from VAE, please refer to the internal variant names: VCF['TEXT'], if you use VCF as input
    
    AD: string
        path of AD matrix in scipy.sparse.coo_matrix format
    
    DP: string
        path of DP matrix in scipy.sparse.coo_matrix format
    
    VCF: string
        path of VCF.gz file
    
    variant_name: string
        path of variant_name.tsv file which is a list of custom variant name stored in pandas dataframe without header and index

    SNPread: string
        optional observed-SNP normalization, 'normalized' or 'unnormalized'
        
    missing_value: float between 0 and 1, or string 'mean' or 'neighbour'
        impute value for missing allele frequency in AF matrix, i.e. DP = 0
        
    cell_weight: string
        optional cost normalization for each cell, 'normalized' or 'unnormalized'

    prior: string
        path of prior weights of mutation for each variant in csv format

    """
    
    print("Start loading raw data.")
    
    if path != None:
    
        VCF_raw = read_VCF_gz(path + "/cellSNP.base.vcf.gz")
        is_VCF = True
        
    elif path == None and VCF != None:
        
        VCF_raw = read_VCF_gz(VCF)
        is_VCF = True
        
    elif path == None and VCF == None:
        
        VCF_raw = pd.read_csv(variant_name, delimiter = "\t", header = None)
        is_VCF = False
    
    if path != None:
        
        AD_raw = mmread(path + '/cellSNP.tag.AD.mtx').toarray().T
        DP_raw = mmread(path + '/cellSNP.tag.DP.mtx').toarray().T
        
    elif path == None:
        
        AD_raw = mmread(AD).toarray().T
        DP_raw = mmread(DP).toarray().T

    if self.UMI_correction == 'negative':

        WT_raw = np.clip(DP_raw - AD_raw - 1, 0, None)
        AD_raw = np.clip(AD_raw - 1, 0, None)
        DP_raw = WT_raw + AD_raw
    
    with warnings.catch_warnings():
        
        warnings.filterwarnings("ignore", category = RuntimeWarning)
        AF_raw_missing_to_mean = AD_raw / DP_raw
        
    AF_mean = np.nanmean(AF_raw_missing_to_mean, 0)
    AF_mean[np.isnan(AF_mean)] = 0
    AF_raw_missing_to_mean[np.isnan(AF_raw_missing_to_mean)] = np.outer(np.ones(AF_raw_missing_to_mean.shape[0]), AF_mean)[np.isnan(AF_raw_missing_to_mean)]
    AF_raw_missing_to_mean = torch.tensor(AF_raw_missing_to_mean).float()

    if prior != None:

        is_prior = True
        prior_raw = np.genfromtxt(prior)
        self.prior_raw = prior_raw

    elif prior == None:
        
        is_prior = False
    
    self.path = path
    self.VCF_raw = VCF_raw
    self.SNP_mask = SNP_mask
    self.is_VCF = is_VCF
    self.AD_raw = AD_raw
    self.DP_raw = DP_raw
    self.AF_mean = AF_mean
    self.AF_raw_missing_to_mean = AF_raw_missing_to_mean
    self.is_prior = is_prior

    print("Finish loading raw data.")

