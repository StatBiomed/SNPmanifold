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


def load_data(self, path, mitoSNP_mask, AD, DP, VCF, variant_name):
    
    """
    Load AD and DP matrices, VCF.gz file or variant_name.tsv file for subsequent analyses in SNP_VAE 
    
    Parameters
    ----------
    path: string
        path of cellSNP-lite output folder which contains cellSNP.tag.AD.mtx, cellSNP.tag.DP.mtx, and cellSNP.base.vcf.gz
        
    mitoSNP_mask: list of integers
        positions of mitochondrial SNPs to be masked due to artefacts
    
    AD: string
        path of AD matrix in scipy.sparse.coo_matrix format
    
    DP: string
        path of DP matrix in scipy.sparse.coo_matrix format
    
    VCF: string
        path of VCF.gz file
    
    variant_name: string
        path of variant_name.tsv file which is a list of custom variant name stored in pandas dataframe without header and index

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
        
    if is_VCF == True:
    
        mitoSNP_filter = np.ones(VCF_raw.shape[0])
    
        for j in range(VCF_raw.shape[0]):

            if VCF_raw["CHROM"][j] == 'chrM' and VCF_raw["POS"][j] in mitoSNP_mask:

                mitoSNP_filter[j] = 0

            if VCF_raw["CHROM"][j] == 'chrMT' and VCF_raw["POS"][j] in mitoSNP_mask:

                mitoSNP_filter[j] = 0

            if VCF_raw["CHROM"][j] == 'M' and VCF_raw["POS"][j] in mitoSNP_mask:

                mitoSNP_filter[j] = 0

            if VCF_raw["CHROM"][j] == 'MT' and VCF_raw["POS"][j] in mitoSNP_mask:

                mitoSNP_filter[j] = 0

        mitoSNP_filter = mitoSNP_filter.astype(bool)

        VCF_raw = VCF_raw[mitoSNP_filter]
    
    if path != None:
        
        AD_raw = mmread(path + '/cellSNP.tag.AD.mtx').toarray()[mitoSNP_filter, :].T
        DP_raw = mmread(path + '/cellSNP.tag.DP.mtx').toarray()[mitoSNP_filter, :].T
        
    elif path == None:
        
        if is_VCF == True:
        
            AD_raw = mmread(AD).toarray()[mitoSNP_filter, :].T
            DP_raw = mmread(DP).toarray()[mitoSNP_filter, :].T
            
        elif is_VCF == False:
            
            AD_raw = mmread(AD).toarray().T
            DP_raw = mmread(DP).toarray().T
    
    with warnings.catch_warnings():
        
        warnings.filterwarnings("ignore", category = RuntimeWarning)
        AF_raw_missing_to_mean = AD_raw / DP_raw
        
    AF_mean = np.nanmean(AF_raw_missing_to_mean, 0)
    AF_raw_missing_to_mean[np.isnan(AF_raw_missing_to_mean)] = np.outer(np.ones(AF_raw_missing_to_mean.shape[0]), AF_mean)[np.isnan(AF_raw_missing_to_mean)]
    AF_raw_missing_to_mean = torch.tensor(AF_raw_missing_to_mean).float()
    
    self.path = path
    self.VCF_raw = VCF_raw
    self.mitoSNP_mask = mitoSNP_mask
    self.is_VCF = is_VCF
    
    if is_VCF == True:
        
        self.mitoSNP_filter = mitoSNP_filter
        
    self.AD_raw = AD_raw
    self.DP_raw = DP_raw
    self.AF_mean = AF_mean
    self.AF_raw_missing_to_mean = AF_raw_missing_to_mean

    print("Finish loading raw data.")

