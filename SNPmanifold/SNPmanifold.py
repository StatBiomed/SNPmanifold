#!/usr/bin/env python
# coding: utf-8

import gzip
import io
import matplotlib as mpl
import matplotlib.cm as cm 
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.io import mmread
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import umap.umap_ as umap
import warnings

from .VAE_models import VAE_base, VAE_normalized, VAE_unnormalized


def read_VCF_gz(path):

    with gzip.open(path, 'rb') as f:
        
        lines = [l.decode('utf-8') for l in f if not l.startswith(b'##')]

    return pd.read_csv(
        io.StringIO(''.join(lines)),
        dtype={'#CHROM': str, 'POS': int, 'ID': str, 'REF': str, 'ALT': str,
               'QUAL': str, 'FILTER': str, 'INFO': str},
        sep='\t'
    ).rename(columns={'#CHROM': 'CHROM'})


# In[ ]:


def moving_average(a, n):

    ret = np.cumsum(a)
    ret[n: ] = ret[n: ] - ret[ : -n]

    return ret[n-1: ] / n


# In[ ]:




# In[ ]:


def load_data(self, path, mitoSNP_mask, AD, DP, VCF, variant_name):
    
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


# In[ ]:


def filter_data(self):
    
    print("Start filtering low-quality cells and SNPs.")
    
    cell_SNPread = np.sum(np.array(self.DP_raw) > 0, 1)
    
    plt.figure(figsize = (10, 7))
    plt.title("Cells sorted by number of observed SNPs")
    plt.plot(np.arange(cell_SNPread.shape[0]) + 1, np.flip(np.sort(cell_SNPread)))
    plt.ylabel("Number of observed SNPs")
    plt.xlabel("Cell")
    plt.show()
    
    cell_SNPread_threshold = float(input("Please determine y-axis threshold in the plot to filter low-quality cells with low number of observed SNPs.   "))
    cell_filter = cell_SNPread > cell_SNPread_threshold
    cell_total = np.sum(cell_filter)
    
    SNP_DPmean = np.mean(self.DP_raw[cell_filter, :], 0)
    
    plt.figure(figsize = (10, 7))
    plt.title("Mean coverage of SNPs per high-quality cell")
    plt.plot(np.arange(SNP_DPmean.shape[0]) + 1, SNP_DPmean)
    plt.ylabel("Mean coverage")
    plt.xlabel("SNP")
    plt.show()
    
    SNP_DPmean_threshold = float(input("Please determine y-axis threshold in the plot to filter low-quality SNPs with low coverage.   "))
    SNP_DPmean_filter = SNP_DPmean > SNP_DPmean_threshold
    
    SNP_logit_var = torch.var(torch.logit(self.AF_raw_missing_to_mean[cell_filter, :], eps = 0.01), 0).numpy()
    
    plt.figure(figsize = (10, 7))
    plt.title("SNPs sorted by logit-variance")
    plt.plot(np.arange(np.sum(SNP_DPmean_filter)) + 1, np.flip(np.sort((SNP_logit_var[SNP_DPmean_filter]))))
    plt.ylabel("Logit-variance of SNP")
    plt.xlabel("SNP")
    plt.show()
    
    SNP_logit_var_threshold = float(input("Please determine y-axis threshold in the plot to filter low-quality SNPs with low logit-variance.   "))
    SNP_logit_var_filter = SNP_logit_var > SNP_logit_var_threshold
    SNP_filter = np.logical_and(SNP_DPmean_filter, SNP_logit_var_filter)
    cell_SNPread_filtered = np.sum(self.DP_raw[cell_filter, :][:, SNP_filter] > 0, 1)
    
    while (cell_SNPread_filtered == 0).any():
        
        SNP_logit_var_threshold = float(input(f"{np.sum(cell_SNPread_filtered == 0)} cells have 0 observed SNPs, please determine a lower y-axis threshold.   "))
        SNP_logit_var_filter = SNP_logit_var > SNP_logit_var_threshold
        SNP_filter = np.logical_and(SNP_DPmean_filter, SNP_logit_var_filter)
        cell_SNPread_filtered = np.sum(self.DP_raw[cell_filter, :][:, SNP_filter] > 0, 1)
    
    SNP_total = np.sum(SNP_filter)
    
    AD_filtered = self.AD_raw[cell_filter, :][:, SNP_filter]
    DP_filtered = self.DP_raw[cell_filter, :][:, SNP_filter]
    
    with warnings.catch_warnings():
        
        warnings.filterwarnings("ignore", category = RuntimeWarning)
        AF_filtered = AD_filtered / DP_filtered
        
    AF_filtered_mean = np.nanmean(AF_filtered, 0)
    AF_filtered_missing_to_zero = np.copy(AF_filtered)
    AF_filtered_missing_to_mean = np.copy(AF_filtered)
    AF_filtered_missing_to_half = np.copy(AF_filtered)
    
    if self.missing_value == "mean":
        
        AF_filtered[np.isnan(AF_filtered)] = np.outer(np.ones(cell_total), AF_filtered_mean)[np.isnan(AF_filtered)]
    
    else:
        
        AF_filtered[np.isnan(AF_filtered)] = self.missing_value
    
    AF_filtered = torch.tensor(AF_filtered).float()
    AF_filtered_missing_to_zero[np.isnan(AF_filtered_missing_to_zero)] = 0
    AF_filtered_missing_to_zero = torch.tensor(AF_filtered_missing_to_zero).float()
    AF_filtered_missing_to_mean[np.isnan(AF_filtered_missing_to_mean)] = np.outer(np.ones(cell_total), AF_filtered_mean)[np.isnan(AF_filtered_missing_to_mean)]
    AF_filtered_missing_to_mean = torch.tensor(AF_filtered_missing_to_mean).float()
    AF_filtered_missing_to_half[np.isnan(AF_filtered_missing_to_half)] = 0.5
    AF_filtered_missing_to_half = torch.tensor(AF_filtered_missing_to_half).float()
    
    if self.is_VCF == True:
        
        pd.options.mode.chained_assignment = None
        VCF_filtered = self.VCF_raw.iloc[SNP_filter, :]
        VCF_filtered["TEXT"] = "chr:" + VCF_filtered["CHROM"].astype(str) + ", " + VCF_filtered["POS"].astype(str) + VCF_filtered["REF"] + ">" + VCF_filtered["ALT"]
        pd.options.mode.chained_assignment = 'warn'
        
    elif self.is_VCF == False:
        
        VCF_filtered = self.VCF_raw.iloc[SNP_filter, :]
    
    self.cell_SNPread = cell_SNPread
    self.cell_SNPread_threshold = cell_SNPread_threshold
    self.cell_filter = cell_filter
    self.cell_total = cell_total
    self.SNP_DPmean = SNP_DPmean
    self.SNP_DPmean_threshold = SNP_DPmean_threshold
    self.SNP_DPmean_filter = SNP_DPmean_filter
    self.SNP_logit_var = SNP_logit_var
    self.SNP_logit_var_threshold = SNP_logit_var_threshold
    self.SNP_logit_var_filter = SNP_logit_var_filter
    self.SNP_filter = SNP_filter
    self.cell_SNPread_filtered = cell_SNPread_filtered
    self.SNP_total = SNP_total
    self.AD_filtered = AD_filtered
    self.DP_filtered = DP_filtered
    self.AF_filtered = AF_filtered
    self.AF_filtered_mean = AF_filtered_mean
    self.AF_filtered_missing_to_zero = AF_filtered_missing_to_zero
    self.AF_filtered_missing_to_mean = AF_filtered_missing_to_mean
    self.AF_filtered_missing_to_half = AF_filtered_missing_to_half
    self.VCF_filtered = VCF_filtered
    
    print(f"Finish filtering low-quality data, {cell_total} cells and {SNP_total} SNPs will be used for downstream analysis.")


# In[ ]:


def summary_filtering(self, dpi):
    
    print(f"Finish filtering low-quality data, {self.cell_total} cells and {self.SNP_total} SNPs will be used for downstream analysis.")
    
    plt.figure(figsize = (10, 7), dpi = dpi)
    plt.title("Cells sorted by number of observed SNPs")
    plt.plot(np.arange(self.cell_SNPread.shape[0]) + 1, np.flip(np.sort(self.cell_SNPread)))
    plt.ylabel("Number of observed SNPs")
    plt.xlabel("Cell")
    plt.axhline(y = self.cell_SNPread_threshold, color = 'r', linestyle = '-')
    plt.show()
    
    plt.figure(figsize = (10, 7), dpi = dpi)
    plt.title("Mean coverage of SNPs per high-quality cell")
    plt.plot(np.arange(self.SNP_DPmean.shape[0]) + 1, self.SNP_DPmean)
    plt.ylabel("Mean coverage")
    plt.xlabel("SNP")
    plt.axhline(y = self.SNP_DPmean_threshold, color = 'r', linestyle = '-')
    plt.show()
    
    plt.figure(figsize = (10, 7), dpi = dpi)
    plt.title("SNPs sorted by logit-variance")
    plt.plot(np.arange(np.sum(self.SNP_DPmean_filter)) + 1, np.flip(np.sort((self.SNP_logit_var[self.SNP_DPmean_filter]))))
    plt.ylabel("Logit-variance of SNP")
    plt.xlabel("SNP")
    plt.axhline(y = self.SNP_logit_var_threshold, color = 'r', linestyle = '-')
    plt.show()


# In[ ]:


def train_VAE(self, num_epoch, stepsize, z_dim, beta, num_batch):
    
    print("Start training VAE.")
    
    if z_dim == None:
        
        z_dim = int(np.min((np.ceil(self.cell_total / 2), np.ceil(self.SNP_total / 2))))
    
    loss_fn = nn.BCELoss(reduction = 'none')
    
    AF_DP_combined = torch.cat((self.AF_filtered, torch.tensor(self.DP_filtered)), 1).float()
    cell_SNPread_filtered = np.count_nonzero(self.DP_filtered, 1)
    cell_SNPread_weight = torch.tensor(np.outer(cell_SNPread_filtered, np.ones(z_dim))).float()
    
    data_loader = DataLoader(AF_DP_combined, int(np.ceil(self.cell_total / num_batch)))
    
    if self.SNPread == "normalized" and self.cell_weight == "unnormalized":
    
        model = VAE_normalized(self.SNP_total, z_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr = stepsize)

        cost_total = np.empty(num_epoch)
        cost_recon = np.empty(num_epoch)
        cost_div = np.empty(num_epoch)

        for epoch in range(num_epoch):

            for batch, x in enumerate(data_loader):

                cell_SNPread_filtered_batch = np.count_nonzero(x[:, self.SNP_total:].numpy(), 1)
                cell_SNPread_weight_batch = torch.tensor(np.outer(cell_SNPread_filtered_batch, np.ones(z_dim))).float()

                x_reconst_mu, mu, log_var = model(x[:, :self.SNP_total], cell_SNPread_weight_batch)
                kl_div = 0.5 * torch.mean(mu.pow(2))
                recon_loss = torch.sum(loss_fn(x_reconst_mu, x[:, :self.SNP_total]) * x[:, self.SNP_total:]) / torch.tensor(np.sum(cell_SNPread_filtered_batch))
                loss_total = recon_loss + beta * kl_div

                optimizer.zero_grad()
                loss_total.backward()
                optimizer.step()

            x_reconst_mu, mu, log_var = model(self.AF_filtered, cell_SNPread_weight)
            kl_div = 0.5 * torch.mean(mu.pow(2))
            recon_loss = torch.sum(loss_fn(x_reconst_mu, AF_DP_combined[:, :self.SNP_total]) * AF_DP_combined[:, self.SNP_total:]) / torch.tensor(np.sum(cell_SNPread_filtered))
            loss_total = recon_loss + beta * kl_div

            print("Epoch[{}/{}], Cost: {:.6f}".format(epoch + 1, num_epoch, loss_total))

            cost_total[epoch] = loss_total
            cost_recon[epoch] = recon_loss
            cost_div[epoch] = kl_div
        
        latent = model.encode(self.AF_filtered, cell_SNPread_weight)[0].detach().numpy()
            
    elif self.SNPread == "unnormalized" and self.cell_weight == "unnormalized":
        
        model = VAE_unnormalized(self.SNP_total, z_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr = stepsize)

        cost_total = np.empty(num_epoch)
        cost_recon = np.empty(num_epoch)
        cost_div = np.empty(num_epoch)

        for epoch in range(num_epoch):

            for batch, x in enumerate(data_loader):
                
                cell_SNPread_filtered_batch = np.count_nonzero(x[:, self.SNP_total:].numpy(), 1)

                x_reconst_mu, mu, log_var = model(x[:, :self.SNP_total])
                kl_div = 0.5 * torch.mean(mu.pow(2))
                recon_loss = torch.sum(loss_fn(x_reconst_mu, x[:, :self.SNP_total]) * x[:, self.SNP_total:]) / torch.tensor(np.sum(cell_SNPread_filtered_batch))
                loss_total = recon_loss + beta * kl_div

                optimizer.zero_grad()
                loss_total.backward()
                optimizer.step()

            x_reconst_mu, mu, log_var = model(self.AF_filtered)
            kl_div = 0.5 * torch.mean(mu.pow(2))
            recon_loss = torch.sum(loss_fn(x_reconst_mu, AF_DP_combined[:, :self.SNP_total]) * AF_DP_combined[:, self.SNP_total:]) / torch.tensor(np.sum(cell_SNPread_filtered))
            loss_total = recon_loss + beta * kl_div

            print("Epoch[{}/{}], Cost: {:.6f}".format(epoch + 1, num_epoch, loss_total))

            cost_total[epoch] = loss_total
            cost_recon[epoch] = recon_loss
            cost_div[epoch] = kl_div
            
        latent = model.encode(self.AF_filtered)[0].detach().numpy()
        
    elif self.SNPread == "normalized" and self.cell_weight == "normalized":
    
        model = VAE_normalized(self.SNP_total, z_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr = stepsize)

        cost_total = np.empty(num_epoch)
        cost_recon = np.empty(num_epoch)
        cost_div = np.empty(num_epoch)

        for epoch in range(num_epoch):

            for batch, x in enumerate(data_loader):

                cell_SNPread_filtered_batch = np.count_nonzero(x[:, self.SNP_total:].numpy(), 1)
                cell_SNPread_weight_batch = torch.tensor(np.outer(cell_SNPread_filtered_batch, np.ones(z_dim))).float()

                x_reconst_mu, mu, log_var = model(x[:, :self.SNP_total], cell_SNPread_weight_batch)
                kl_div = 0.5 * torch.mean(mu.pow(2))
                recon_loss = torch.mean(torch.sum(loss_fn(x_reconst_mu, x[:, :self.SNP_total]) * x[:, self.SNP_total:], 1) / torch.tensor(cell_SNPread_filtered_batch))
                loss_total = recon_loss + beta * kl_div

                optimizer.zero_grad()
                loss_total.backward()
                optimizer.step()

            x_reconst_mu, mu, log_var = model(self.AF_filtered, cell_SNPread_weight)
            kl_div = 0.5 * torch.mean(mu.pow(2))
            recon_loss = torch.mean(torch.sum(loss_fn(x_reconst_mu, AF_DP_combined[:, :self.SNP_total]) * AF_DP_combined[:, self.SNP_total:], 1) / torch.tensor(self.cell_SNPread_filtered))
            loss_total = recon_loss + beta * kl_div

            print("Epoch[{}/{}], Cost: {:.6f}".format(epoch + 1, num_epoch, loss_total))

            cost_total[epoch] = loss_total
            cost_recon[epoch] = recon_loss
            cost_div[epoch] = kl_div
        
        latent = model.encode(self.AF_filtered, cell_SNPread_weight)[0].detach().numpy()
        
    elif self.SNPread == "unnormalized" and self.cell_weight == "normalized":
        
        model = VAE_unnormalized(self.SNP_total, z_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr = stepsize)

        cost_total = np.empty(num_epoch)
        cost_recon = np.empty(num_epoch)
        cost_div = np.empty(num_epoch)

        for epoch in range(num_epoch):

            for batch, x in enumerate(data_loader):
                
                cell_SNPread_filtered_batch = np.count_nonzero(x[:, self.SNP_total:].numpy(), 1)

                x_reconst_mu, mu, log_var = model(x[:, :self.SNP_total])
                kl_div = 0.5 * torch.mean(mu.pow(2))
                recon_loss = torch.mean(torch.sum(loss_fn(x_reconst_mu, x[:, :self.SNP_total]) * x[:, self.SNP_total:], 1) / torch.tensor(cell_SNPread_filtered_batch))
                loss_total = recon_loss + beta * kl_div

                optimizer.zero_grad()
                loss_total.backward()
                optimizer.step()

            x_reconst_mu, mu, log_var = model(self.AF_filtered)
            kl_div = 0.5 * torch.mean(mu.pow(2))
            recon_loss = torch.mean(torch.sum(loss_fn(x_reconst_mu, AF_DP_combined[:, :self.SNP_total]) * AF_DP_combined[:, self.SNP_total:], 1) / torch.tensor(self.cell_SNPread_filtered))
            loss_total = recon_loss + beta * kl_div

            print("Epoch[{}/{}], Cost: {:.6f}".format(epoch + 1, num_epoch, loss_total))

            cost_total[epoch] = loss_total
            cost_recon[epoch] = recon_loss
            cost_div[epoch] = kl_div
            
        latent = model.encode(self.AF_filtered)[0].detach().numpy()

    self.num_epoch = num_epoch
    self.stepsize = stepsize
    self.z_dim = z_dim
    self.beta = beta
    self.num_batch = num_batch
    self.cell_SNPread_filtered = cell_SNPread_filtered
    self.model = model
    self.cost_total = cost_total
    self.cost_recon = cost_recon
    self.cost_div = cost_div
    self.latent = latent
    
    print("Finish training VAE, training curve will be shown below.")
    
    plt.figure(figsize = (10, 7))
    plt.title(f"Training curve of VAE in {num_epoch} epochs")
    plt.plot(np.arange(1, num_epoch + 1), cost_total)
    plt.xlabel("Epoch")
    plt.ylabel("Cost")
    plt.show()
    
    print("Start learning PCA and UMAP of latent space in VAE.")
    
    pca = PCA()
    pca.fit(self.latent)
    pc = pca.fit_transform(self.latent)
    
    reducer_2d = umap.UMAP(n_components = 2)
    embedding_2d = reducer_2d.fit_transform(self.latent)
    pair_embedding_2d = squareform(pdist(embedding_2d))
    
    if z_dim >= 3:
    
        reducer_3d = umap.UMAP(n_components = 3)
        embedding_3d = reducer_3d.fit_transform(self.latent)
        pair_embedding_3d = squareform(pdist(embedding_3d))
    
    print("Finish learning, PCA and UMAP of latent space will be shown below.")
    
    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(12, 5)
    
    axs[0].set_title("Scatter plot of PCA")
    axs[0].scatter(pc[:, 0], pc[:, 1], s = 5, color = "black")
    axs[0].set_xticks([])
    axs[0].set_yticks([])

    axs[1].set_title("Scatter plot of UMAP")
    axs[1].scatter(embedding_2d[:, 0], embedding_2d[:, 1], s = 5, color = "black")
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    
    plt.show()
    
    xlim_pc = axs[0].get_xlim()
    ylim_pc = axs[0].get_ylim()
    xlim_embedding_2d = axs[1].get_xlim()
    ylim_embedding_2d = axs[1].get_ylim()
    
    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(12, 5)

    axs[0].set_title("Density plot of PCA")
    axs[0].hist2d(pc[:, 0], pc[:, 1], bins = (200, 200), cmap = plt.cm.jet, range = np.array([xlim_pc, ylim_pc]))
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    
    H_pc = plt.hist2d(pc[:, 0], pc[:, 1], bins = (200, 200), cmap = plt.cm.jet, range = np.array([xlim_pc, ylim_pc]))[0]
    vmin_pc = np.min(H_pc)
    vmax_pc = np.max(H_pc)
    
    axs[1].set_title("Density plot of UMAP")
    axs[1].hist2d(embedding_2d[:, 0], embedding_2d[:, 1], bins = (200, 200), cmap = plt.cm.jet, range = np.array([xlim_embedding_2d, ylim_embedding_2d]))
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    
    H_embedding_2d = plt.hist2d(embedding_2d[:, 0], embedding_2d[:, 1], bins = (200, 200), cmap = plt.cm.jet, range = np.array([xlim_embedding_2d, ylim_embedding_2d]))[0]
    vmin_embedding_2d = np.min(H_embedding_2d)
    vmax_embedding_2d = np.max(H_embedding_2d)
    
    plt.show()
    
    self.pc = pc
    self.embedding_2d = embedding_2d
    self.pair_embedding_2d = pair_embedding_2d
    
    if z_dim >= 3:
        
        self.embedding_3d = embedding_3d
        self.pair_embedding_3d = pair_embedding_3d
        
    self.xlim_pc = xlim_pc
    self.ylim_pc = ylim_pc
    self.xlim_embedding_2d = xlim_embedding_2d
    self.ylim_embedding_2d = ylim_embedding_2d
    self.vmin_pc = vmin_pc
    self.vmax_pc = vmax_pc
    self.vmin_embedding_2d = vmin_embedding_2d
    self.vmax_embedding_2d = vmax_embedding_2d


# In[ ]:


def summary_training(self, dpi):
    
    print("Training curve will be shown below.")
    
    plt.figure(figsize = (10, 7), dpi = dpi)
    plt.title(f"Training curve of VAE in {self.num_epoch} epochs")
    plt.plot(np.arange(1, self.num_epoch + 1), self.cost_total)
    plt.xlabel("Epoch")
    plt.ylabel("Cost")
    plt.show()
    
    print("PCA and UMAP of latent space will be shown below.")
    
    fig, axs = plt.subplots(1, 2, dpi = dpi)
    fig.set_size_inches(12, 5)
    
    axs[0].set_title("Scatter plot of PCA")
    axs[0].scatter(self.pc[:, 0], self.pc[:, 1], s = 5, color = "black")
    axs[0].set_xticks([])
    axs[0].set_yticks([])

    axs[1].set_title("Scatter plot of UMAP")
    axs[1].scatter(self.embedding_2d[:, 0], self.embedding_2d[:, 1], s = 5, color = "black")
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    
    plt.show()
    
    fig, axs = plt.subplots(1, 2, dpi = dpi)
    fig.set_size_inches(12, 5)

    axs[0].set_title("Density plot of PCA")
    axs[0].hist2d(self.pc[:, 0], self.pc[:, 1], bins = (200, 200), cmap = plt.cm.jet, range = np.array([self.xlim_pc, self.ylim_pc]))
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    
    axs[1].set_title("Density plot of UMAP")
    axs[1].hist2d(self.embedding_2d[:, 0], self.embedding_2d[:, 1], bins = (200, 200), cmap = plt.cm.jet, range = np.array([self.xlim_embedding_2d, self.ylim_embedding_2d]))
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    
    plt.show()


# In[ ]:


def kmeans_clustering(self, max_cluster):
    
    print("Start clustering.")
    
    scores = []
    labels = []
    distortions = []
    centres = []
    
    if self.z_dim >= 3:
    
        if max_cluster < self.cell_total:

            for m in np.arange(2, max_cluster + 1):

                kmeans = KMeans(n_clusters = m, n_init = 1).fit(self.embedding_3d)
                scores.append(silhouette_score(self.pair_embedding_3d, kmeans.labels_, metric = "precomputed"))
                labels.append(kmeans.labels_)
                distortions.append(kmeans.inertia_)
                centres.append(kmeans.cluster_centers_)
                print("{} clusters, Distortion: {:.6f}".format(m, kmeans.inertia_))

        elif max_cluster == self.cell_total:

            for m in np.arange(2, max_cluster):

                kmeans = KMeans(n_clusters = m, n_init = 1).fit(self.embedding_3d)
                scores.append(silhouette_score(self.pair_embedding_3d, kmeans.labels_, metric = "precomputed"))
                labels.append(kmeans.labels_)
                distortions.append(kmeans.inertia_)
                centres.append(kmeans.cluster_centers_)
                print("{} clusters, Distortion: {:.6f}".format(m, kmeans.inertia_))
                
            scores.append(1)
            labels.append(np.arange(0, self.cell_total))
            distortions.append(0)
            centres.append(self.embedding_3d)
                
    elif self.z_dim == 2:
            
        if max_cluster < self.cell_total:

            for m in np.arange(2, max_cluster + 1):

                kmeans = KMeans(n_clusters = m, n_init = 1).fit(self.embedding_2d)
                scores.append(silhouette_score(self.pair_embedding_2d, kmeans.labels_, metric = "precomputed"))
                labels.append(kmeans.labels_)
                distortions.append(kmeans.inertia_)
                centres.append(kmeans.cluster_centers_)
                print("{} clusters, Distortion: {:.6f}".format(m, kmeans.inertia_))

        elif max_cluster == self.cell_total:

            for m in np.arange(2, max_cluster):

                kmeans = KMeans(n_clusters = m, n_init = 1).fit(self.embedding_2d)
                scores.append(silhouette_score(self.pair_embedding_2d, kmeans.labels_, metric = "precomputed"))
                labels.append(kmeans.labels_)
                distortions.append(kmeans.inertia_)
                centres.append(kmeans.cluster_centers_)
                print("{} clusters, Distortion: {:.6f}".format(m, kmeans.inertia_))
            
            scores.append(1)
            labels.append(np.arange(0, self.cell_total))
            distortions.append(0)
            centres.append(self.embedding_2d)
        
    self.max_cluster = max_cluster
    self.scores = scores
    self.labels = labels
    self.distortions = distortions
    self.centres = centres
    
    print("Finish clustering, PCA, UMAP, distortion, silhouette score of K-means clustering will be shown below.")
    
    fig, axs = plt.subplots(1, max_cluster - 1)
    fig.set_size_inches(6 * (max_cluster - 1), 5)
    fig.suptitle("Scatter plot of PCA")
    
    for m in np.arange(2, max_cluster + 1):
        
        clusters = []

        for g in range(m):

                clusters.append(np.where(labels[m - 2] == g)[0])

        colors = cm.rainbow(np.linspace(0, 1, m))
        
        axs[m - 2].set_title(str(m) + " clusters")
        axs[m - 2].set_xticks([])
        axs[m - 2].set_yticks([])
        
        for g in range(m):
    
            axs[m - 2].scatter(self.pc[clusters[g], 0], self.pc[clusters[g], 1], s = 5, color = colors[g])
        
    plt.show()
    
    fig, axs = plt.subplots(1, max_cluster - 1)
    fig.set_size_inches(6 * (max_cluster - 1), 5)
    fig.suptitle("Scatter plot of UMAP")
    
    for m in np.arange(2, max_cluster + 1):
        
        clusters = []

        for g in range(m):

                clusters.append(np.where(labels[m - 2] == g)[0])

        colors = cm.rainbow(np.linspace(0, 1, m))
        
        axs[m - 2].set_title(str(m) + " clusters")
        axs[m - 2].set_xticks([])
        axs[m - 2].set_yticks([])
        
        for g in range(m):
    
            axs[m - 2].scatter(self.embedding_2d[clusters[g], 0], self.embedding_2d[clusters[g], 1], s = 5, color = colors[g])
        
    plt.show()
    
    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(12, 5)
    
    axs[0].set_title("Distortion")
    axs[0].plot(np.arange(2, max_cluster + 1), distortions)
    axs[0].set_xlabel("Number of clusters")    
    
    axs[1].set_title("Silhouette score")
    axs[1].plot(np.arange(2, max_cluster + 1), scores)
    axs[1].set_xlabel("Number of clusters")

    plt.show()


# In[ ]:


def summary_clustering(self, dpi):
    
    print("PCA, UMAP, distortion, silhouette score of K-means clustering will be shown below.")
    
    fig, axs = plt.subplots(1, self.max_cluster - 1, dpi = dpi)
    fig.set_size_inches(6 * (self.max_cluster - 1), 5)
    fig.suptitle("Scatter plot of PCA")
    
    for m in np.arange(2, self.max_cluster + 1):
        
        clusters = []

        for g in range(m):

                clusters.append(np.where(self.labels[m - 2] == g)[0])

        colors = cm.rainbow(np.linspace(0, 1, m))
        
        axs[m - 2].set_title(str(m) + " clusters")
        axs[m - 2].set_xticks([])
        axs[m - 2].set_yticks([])
        
        for g in range(m):
    
            axs[m - 2].scatter(self.pc[clusters[g], 0], self.pc[clusters[g], 1], s = 5, color = colors[g])
        
    plt.show()
    
    fig, axs = plt.subplots(1, self.max_cluster - 1, dpi = dpi)
    fig.set_size_inches(6 * (self.max_cluster - 1), 5)
    fig.suptitle("Scatter plot of UMAP")
    
    for m in np.arange(2, self.max_cluster + 1):
        
        clusters = []

        for g in range(m):

                clusters.append(np.where(self.labels[m - 2] == g)[0])

        colors = cm.rainbow(np.linspace(0, 1, m))
        
        axs[m - 2].set_title(str(m) + " clusters")
        axs[m - 2].set_xticks([])
        axs[m - 2].set_yticks([])
        
        for g in range(m):
    
            axs[m - 2].scatter(self.embedding_2d[clusters[g], 0], self.embedding_2d[clusters[g], 1], s = 5, color = colors[g])
        
    plt.show()
    
    fig, axs = plt.subplots(1, 2, dpi = dpi)
    fig.set_size_inches(12, 5)
    
    axs[0].set_title("Distortion")
    axs[0].plot(np.arange(2, self.max_cluster + 1), self.distortions)
    axs[0].set_xlabel("Number of clusters")    
    
    axs[1].set_title("Silhouette score")
    axs[1].plot(np.arange(2, self.max_cluster + 1), self.scores)
    axs[1].set_xlabel("Number of clusters")

    plt.show()


# In[ ]:


def tree(self, cluster_no, pair_no, SNP_no):
    
    print("PCA and UMAP of individual clusters will be shown below.")
    
    assigned_label = self.labels[cluster_no - 2]
    
    clusters = []
    
    for g in range(cluster_no):

            clusters.append(np.where(assigned_label == g)[0])
    
    colors = cm.rainbow(np.linspace(0, 1, cluster_no))
    
    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(12, 5)
    
    axs[0].set_title("Scatter plot of PCA")
    axs[0].set_xticks([])
    axs[0].set_yticks([])

    for g in range(cluster_no):

        axs[0].scatter(self.pc[clusters[g], 0], self.pc[clusters[g], 1], s = 5, color = colors[g])

    axs[1].set_title("Scatter plot of UMAP")
    axs[1].set_xticks([])
    axs[1].set_yticks([])

    for g in range(cluster_no):

        axs[1].scatter(self.embedding_2d[clusters[g], 0], self.embedding_2d[clusters[g], 1], s = 5, color = colors[g])

    plt.show()
    
    fig, axs = plt.subplots(1, cluster_no)
    fig.set_size_inches(6 * cluster_no, 5)
    fig.suptitle("Scatter plot of PCA")
    
    for m in range(cluster_no):
        
        axs[m].set_title("Cluster " + str(m))
        axs[m].scatter(self.pc[:, 0], self.pc[:, 1], s = 5, color = 'black')
        axs[m].scatter(self.pc[clusters[m], 0], self.pc[clusters[m], 1], s = 5, color = colors[m])
        axs[m].set_xticks([])
        axs[m].set_yticks([])
        axs[m].set_xlim(self.xlim_pc)
        axs[m].set_ylim(self.ylim_pc)
        
    plt.show()
    
    fig, axs = plt.subplots(1, cluster_no)
    fig.set_size_inches(6 * cluster_no, 5)
    fig.suptitle("Scatter plot of UMAP")
    
    for m in range(cluster_no):
        
        axs[m].set_title("Cluster " + str(m))
        axs[m].scatter(self.embedding_2d[:, 0], self.embedding_2d[:, 1], s = 5, color = 'black')
        axs[m].scatter(self.embedding_2d[clusters[m], 0], self.embedding_2d[clusters[m], 1], s = 5, color = colors[m])
        axs[m].set_xticks([])
        axs[m].set_yticks([])
        axs[m].set_xlim(self.xlim_embedding_2d)
        axs[m].set_ylim(self.ylim_embedding_2d)
        
    plt.show()
    
    fig, axs = plt.subplots(1, cluster_no)
    fig.set_size_inches(6 * cluster_no, 5)
    fig.suptitle("Density plot of PCA")
    
    for m in range(cluster_no):
        
        axs[m].set_title("Cluster " + str(m))
        axs[m].hist2d(self.pc[clusters[m], 0], self.pc[clusters[m], 1], bins = (200, 200), cmap = plt.cm.jet, range = np.array([self.xlim_pc, self.ylim_pc]), vmin = self.vmin_pc, vmax = self.vmax_pc)
        axs[m].set_xticks([])
        axs[m].set_yticks([])
        
    plt.show()
    
    fig, axs = plt.subplots(1, cluster_no)
    fig.set_size_inches(6 * cluster_no, 5)
    fig.suptitle("Density plot of UMAP")
    
    for m in range(cluster_no):
        
        axs[m].set_title("Cluster " + str(m))
        axs[m].hist2d(self.embedding_2d[clusters[m], 0], self.embedding_2d[clusters[m], 1], bins = (200, 200), cmap = plt.cm.jet, range = np.array([self.xlim_embedding_2d, self.ylim_embedding_2d]), vmin = self.vmin_embedding_2d, vmax = self.vmax_embedding_2d)
        axs[m].set_xticks([])
        axs[m].set_yticks([])
        
    plt.show()
    
    if self.z_dim >= 3:
    
        fig = plt.figure()
        fig.set_size_inches(12, 5)
        axs1 = fig.add_subplot(121, projection = '3d')
        axs2 = fig.add_subplot(122, projection = '3d')

        axs1.set_title("Scatter plot of PCA")
        axs1.set_xticks([])
        axs1.set_yticks([])
        axs1.set_zticks([])

        for g in range(cluster_no):

            axs1.scatter(self.pc[clusters[g], 0], self.pc[clusters[g], 1], self.pc[clusters[g], 2], s = 5, color = colors[g])

        axs2.set_title("Scatter plot of UMAP")
        axs2.set_xticks([])
        axs2.set_yticks([])
        axs2.set_zticks([])

        for g in range(cluster_no):

            axs2.scatter(self.embedding_3d[clusters[g], 0], self.embedding_3d[clusters[g], 1], self.embedding_3d[clusters[g], 2], s = 5, color = colors[g])

        plt.show()
    
    self.cluster_no = cluster_no
    self.assigned_label = assigned_label
    self.clusters = clusters
    self.colors = colors
    
    pair_latent = squareform(pdist(self.latent))
    pair_latent_cluster = np.empty((cluster_no, cluster_no))
    pair_latent_cluster_neighbour = np.empty((cluster_no, cluster_no))

    for t in range(cluster_no):

        for g in range(cluster_no):

            pair_latent_cluster[t, g] = np.mean(pair_latent[clusters[t], :][:, clusters[g]].flatten())
            pair_latent_cluster_neighbour[t, g] = np.mean(np.sort(pair_latent[clusters[t], :][:, clusters[g]].flatten())[:pair_no])

    for w in range(cluster_no):

        pair_latent_cluster[w, w] = np.nan
        pair_latent_cluster_neighbour[w, w] = np.nan

    edge_length = []
    edge_length_neighbour = [] 
    edge = []
    connected = np.zeros(cluster_no)
    connect_order = []

    pair_nearest = np.unravel_index(np.nanargmin(pair_latent_cluster_neighbour), pair_latent_cluster_neighbour.shape)
    edge_length.append(pair_latent_cluster[pair_nearest])
    edge_length_neighbour.append(pair_latent_cluster_neighbour[pair_nearest])
    edge.append(pair_nearest)
    connected[pair_nearest[0]] = 1
    connected[pair_nearest[1]] = 1
    connect_order.append(pair_nearest[0])
    connect_order.append(pair_nearest[1])

    while np.sum(connected) < cluster_no:

        search_space = pair_latent_cluster_neighbour[np.where(connected == 1)[0], :][:, np.where(connected == 0)[0]]
        pair_nearest_search = np.unravel_index(np.argmin(search_space), search_space.shape)
        pair_nearest = (np.where(connected == 1)[0][pair_nearest_search[0]], np.where(connected == 0)[0][pair_nearest_search[1]])
        edge_length.append(pair_latent_cluster[pair_nearest])
        edge_length_neighbour.append(pair_latent_cluster_neighbour[pair_nearest])
        edge.append(pair_nearest)
        connected[pair_nearest[1]] = 1
        connect_order.append(pair_nearest[1])

    edge_weight = 1 / (np.array(edge_length) ** 2)
    edge_length_normalized = np.round(edge_length / np.sqrt(self.z_dim), decimals = 2)

    cluster_size = []

    for w in range(cluster_no):

        cluster_size.append(clusters[w].shape[0])
        
    cluster_size = np.array(cluster_size)

    print(f"Phylogenetic tree in latent space will be shown below.")

    graph = nx.Graph()

    for u in range(len(edge)):

        graph.add_edge(edge[u][0], edge[u][1], weight = edge_weight[u], length = edge_length_normalized[u])

    pos = nx.spring_layout(graph, iterations = 3000, weight = 'weight')

    plt.figure(figsize = (26, 14)) 
    plt.title("Phylogenetic tree")
    nx.draw(graph, pos, with_labels = True, font_weight = 'bold', node_size = cluster_size[connect_order], node_color = self.colors[connect_order])
    nx.draw_networkx_edge_labels(graph, pos, nx.get_edge_attributes(graph, 'length'), rotate = False, alpha = 0.75)
    
    plt.show()
    
    self.pair_no = pair_no
    self.pair_latent = pair_latent
    self.pair_latent_cluster = pair_latent_cluster
    self.pair_latent_cluster_neighbour = pair_latent_cluster_neighbour
    self.edge_length = edge_length
    self.edge_length_neighbour = edge_length_neighbour
    self.edge = edge
    self.connect_order = connect_order
    self.edge_weight = edge_weight
    self.edge_length_normalized = edge_length_normalized
    self.cluster_size = cluster_size
    
    SNP_no = np.min((self.SNP_total, SNP_no)).astype(int)
    
    SNP_cluster_logit_var = np.empty((cluster_no, self.SNP_total))
    SNP_cluster_AF_filtered_missing_to_zero = np.empty((cluster_no, self.SNP_total))
    centre_cluster = np.empty((cluster_no, self.z_dim))
    
    for m in range(cluster_no):
        
        SNP_cluster_logit_var[m, :] = torch.var(torch.logit(self.AF_filtered_missing_to_mean[clusters[m], :], eps = 0.01), 0).numpy()
        SNP_cluster_AF_filtered_missing_to_zero[m, :] = np.mean(self.AF_filtered_missing_to_zero.numpy()[clusters[m], :], 0)
        centre_cluster[m, :] = np.mean(self.latent[clusters[m], :], 0)
    
    ratio_logit_var = np.min(SNP_cluster_logit_var, 0) / self.SNP_logit_var[self.SNP_filter]
    SNP_cluster_AF_filtered_missing_to_zero_max = np.max(SNP_cluster_AF_filtered_missing_to_zero, 0)
    
    rank_SNP = np.argsort(ratio_logit_var)
    SNP_low_ratio_logit_var_total = np.sum(ratio_logit_var < 0.01)
    
    if SNP_low_ratio_logit_var_total > 1:
        
        SNP_low_ratio_logit_var = rank_SNP[:SNP_low_ratio_logit_var_total]
        
        rank_SNP_low_ratio_logit_var = np.flip(np.argsort(SNP_cluster_AF_filtered_missing_to_zero_max[SNP_low_ratio_logit_var]))
        rank_SNP[:SNP_low_ratio_logit_var_total] = rank_SNP[:SNP_low_ratio_logit_var_total][rank_SNP_low_ratio_logit_var]
        
        self.SNP_low_ratio_logit_var = SNP_low_ratio_logit_var
        self.rank_SNP_low_ratio_logit_var = rank_SNP_low_ratio_logit_var
        
    root = np.argmin(np.mean(centre_cluster ** 2, 1))
    move = 0
    edge_remain = edge.copy()
    current_pos = root
    cluster_order = [root]
    history = []

    while move < len(edge):

        current_move = move

        for w in range(len(edge_remain)):

            if current_pos in edge_remain[w]:

                history.append(current_pos)

                if edge_remain[w][0] == current_pos:

                    current_pos = edge_remain[w][1]

                elif edge_remain[w][1] == current_pos:

                    current_pos = edge_remain[w][0]

                cluster_order.append(current_pos)
                move = current_move + 1            
                edge_remain.remove(edge_remain[w])     
                break

        if current_move == move:

            current_pos = history[-1]
            del history[-1]
    
    cell_sorted = np.empty(0)

    for w in cluster_order:

        cell_sorted = np.concatenate((cell_sorted, clusters[w]), axis = None).astype(int)
        
    AF_sorted = self.AF_filtered_missing_to_half.numpy()[cell_sorted, :][:, rank_SNP].T
    
    print(f"SNP-allelic ratios of {self.cell_total} cells and {SNP_no} SNPs will be shown below.")
    
    clus_colors = pd.Series(assigned_label[cell_sorted]).map(dict(zip(np.arange(0, cluster_no), colors)))
    clus_colors.index = pd.RangeIndex(start = 1, stop = self.cell_total + 1, step = 1)
    
    if self.is_VCF == True:
        
        fig = sns.clustermap(pd.DataFrame(AF_sorted[:SNP_no, :], index = self.VCF_filtered["TEXT"].to_numpy()[rank_SNP][:SNP_no], columns = np.arange(1, self.cell_total + 1)), row_cluster = False, col_cluster = False, col_colors = clus_colors, figsize = (20, SNP_no * 0.6))
    
    elif self.is_VCF == False:
        
        fig = sns.clustermap(pd.DataFrame(AF_sorted[:SNP_no, :], index = self.VCF_filtered[0].to_numpy()[rank_SNP][:SNP_no], columns = np.arange(1, self.cell_total + 1)), row_cluster = False, col_cluster = False, col_colors = clus_colors, figsize = (20, SNP_no * 0.6))
    
    fig.ax_col_colors.set_xticks(moving_average(np.cumsum([0] + list(cluster_size[cluster_order])), 2))
    fig.ax_col_colors.set_xticklabels(np.array(cluster_order))
    fig.ax_col_colors.xaxis.set_tick_params(size = 0)
    fig.ax_col_colors.xaxis.tick_top()
    plt.show()
    
    ratio_logit_var_ranked = ratio_logit_var[rank_SNP]
    SNP_cluster_AF_filtered_missing_to_zero_max_ranked = SNP_cluster_AF_filtered_missing_to_zero_max[rank_SNP]
    
    print("Highest cluster specific SNP-allelic ratio of SNPs sorted by lowest cluster-to-bulk variance ratio will be shown below")
    
    fig, ax1 = plt.subplots()
    fig.set_size_inches(10, 7)
    fig.suptitle("SNPs sorted by lowest cluster-to-bulk variance ratio")

    ax1.set_xlabel('SNP')
    ax1.set_ylabel('Variance ratio', color = 'tab:red')
    ax1.plot(np.arange(self.SNP_total), ratio_logit_var_ranked, color = 'tab:red')
    ax1.set_ylim([- 0.05, 1.05])
    ax1.tick_params(axis = 'y', labelcolor = 'tab:red')

    ax2 = ax1.twinx()

    color = 'tab:blue'
    ax2.set_ylabel('SNP-allelic ratio', color = 'tab:blue')
    ax2.plot(np.arange(self.SNP_total), SNP_cluster_AF_filtered_missing_to_zero_max_ranked, color = 'tab:blue')
    ax2.set_ylim([- 0.05, 1.05])
    ax2.tick_params(axis = 'y', labelcolor = 'tab:blue')

    fig.tight_layout()
    plt.show()
    
    self.SNP_no = SNP_no
    self.SNP_cluster_logit_var = SNP_cluster_logit_var
    self.SNP_cluster_AF_filtered_missing_to_zero = SNP_cluster_AF_filtered_missing_to_zero
    self.centre_cluster = centre_cluster
    self.ratio_logit_var = ratio_logit_var
    self.SNP_cluster_AF_filtered_missing_to_zero_max = SNP_cluster_AF_filtered_missing_to_zero_max
    self.rank_SNP = rank_SNP
    self.SNP_low_ratio_logit_var_total = SNP_low_ratio_logit_var_total
    self.root = root
    self.cluster_order = cluster_order
    self.cell_sorted = cell_sorted
    self.AF_sorted = AF_sorted
    self.ratio_logit_var_ranked = ratio_logit_var_ranked
    self.SNP_cluster_AF_filtered_missing_to_zero_max_ranked = SNP_cluster_AF_filtered_missing_to_zero_max_ranked


# In[ ]:


def summary_phylogeny(self, SNP_no, dpi):
    
    if SNP_no == None:
        
        SNP_no = self.SNP_no
        
    print("PCA and UMAP of individual clusters will be shown below.")
    
    fig, axs = plt.subplots(1, 2, dpi = dpi)
    fig.set_size_inches(12, 5)
    
    axs[0].set_title("Scatter plot of PCA")
    axs[0].set_xticks([])
    axs[0].set_yticks([])

    for g in range(self.cluster_no):

        axs[0].scatter(self.pc[self.clusters[g], 0], self.pc[self.clusters[g], 1], s = 5, color = self.colors[g])

    axs[1].set_title("Scatter plot of UMAP")
    axs[1].set_xticks([])
    axs[1].set_yticks([])

    for g in range(self.cluster_no):

        axs[1].scatter(self.embedding_2d[self.clusters[g], 0], self.embedding_2d[self.clusters[g], 1], s = 5, color = self.colors[g])

    plt.show()
    
    fig, axs = plt.subplots(1, self.cluster_no, dpi = dpi)
    fig.set_size_inches(6 * self.cluster_no, 5)
    fig.suptitle("Scatter plot of PCA")
    
    for m in range(self.cluster_no):
        
        axs[m].set_title("Cluster " + str(m))
        axs[m].scatter(self.pc[:, 0], self.pc[:, 1], s = 5, color = 'black')
        axs[m].scatter(self.pc[self.clusters[m], 0], self.pc[self.clusters[m], 1], s = 5, color = self.colors[m])
        axs[m].set_xticks([])
        axs[m].set_yticks([])
        axs[m].set_xlim(self.xlim_pc)
        axs[m].set_ylim(self.ylim_pc)
        
    plt.show()
    
    fig, axs = plt.subplots(1, self.cluster_no, dpi = dpi)
    fig.set_size_inches(6 * self.cluster_no, 5)
    fig.suptitle("Scatter plot of UMAP")
    
    for m in range(self.cluster_no):
        
        axs[m].set_title("Cluster " + str(m))
        axs[m].scatter(self.embedding_2d[:, 0], self.embedding_2d[:, 1], s = 5, color = 'black')
        axs[m].scatter(self.embedding_2d[self.clusters[m], 0], self.embedding_2d[self.clusters[m], 1], s = 5, color = self.colors[m])
        axs[m].set_xticks([])
        axs[m].set_yticks([])
        axs[m].set_xlim(self.xlim_embedding_2d)
        axs[m].set_ylim(self.ylim_embedding_2d)
        
    plt.show()
    
    fig, axs = plt.subplots(1, self.cluster_no, dpi = dpi)
    fig.set_size_inches(6 * self.cluster_no, 5)
    fig.suptitle("Density plot of PCA")
    
    for m in range(self.cluster_no):
        
        axs[m].set_title("Cluster " + str(m))
        axs[m].hist2d(self.pc[self.clusters[m], 0], self.pc[self.clusters[m], 1], bins = (200, 200), cmap = plt.cm.jet, range = np.array([self.xlim_pc, self.ylim_pc]), vmin = self.vmin_pc, vmax = self.vmax_pc)
        axs[m].set_xticks([])
        axs[m].set_yticks([])
        
    plt.show()
    
    fig, axs = plt.subplots(1, self.cluster_no, dpi = dpi)
    fig.set_size_inches(6 * self.cluster_no, 5)
    fig.suptitle("Density plot of UMAP")
    
    for m in range(self.cluster_no):
        
        axs[m].set_title("Cluster " + str(m))
        axs[m].hist2d(self.embedding_2d[self.clusters[m], 0], self.embedding_2d[self.clusters[m], 1], bins = (200, 200), cmap = plt.cm.jet, range = np.array([self.xlim_embedding_2d, self.ylim_embedding_2d]), vmin = self.vmin_embedding_2d, vmax = self.vmax_embedding_2d)
        axs[m].set_xticks([])
        axs[m].set_yticks([])
        
    plt.show()
    
    if self.z_dim >= 3:
    
        fig = plt.figure(dpi = dpi)
        fig.set_size_inches(12, 5)
        axs1 = fig.add_subplot(121, projection = '3d')
        axs2 = fig.add_subplot(122, projection = '3d')

        axs1.set_title("Scatter plot of PCA")
        axs1.set_xticks([])
        axs1.set_yticks([])
        axs1.set_zticks([])

        for g in range(self.cluster_no):

            axs1.scatter(self.pc[self.clusters[g], 0], self.pc[self.clusters[g], 1], self.pc[self.clusters[g], 2], s = 5, color = self.colors[g])

        axs2.set_title("Scatter plot of UMAP")
        axs2.set_xticks([])
        axs2.set_yticks([])
        axs2.set_zticks([])

        for g in range(self.cluster_no):

            axs2.scatter(self.embedding_3d[self.clusters[g], 0], self.embedding_3d[self.clusters[g], 1], self.embedding_3d[self.clusters[g], 2], s = 5, color = self.colors[g])

        plt.show()

    print(f"Phylogenetic tree in latent space will be shown below.")

    graph = nx.Graph()

    for u in range(len(self.edge)):

        graph.add_edge(self.edge[u][0], self.edge[u][1], weight = self.edge_weight[u], length = self.edge_length_normalized[u])

    pos = nx.spring_layout(graph, iterations = 3000, weight = 'weight')

    plt.figure(figsize = (26, 14), dpi = dpi) 
    plt.title("Phylogenetic tree")
    nx.draw(graph, pos, with_labels = True, font_weight = 'bold', node_size = self.cluster_size[self.connect_order], node_color = self.colors[self.connect_order])
    nx.draw_networkx_edge_labels(graph, pos, nx.get_edge_attributes(graph, 'length'), rotate = False, alpha = 0.75)
    
    plt.show()
    
    SNP_no = np.min((self.SNP_total, SNP_no)).astype(int)
    
    print(f"SNP-allelic ratios of {self.cell_total} cells and {SNP_no} SNPs will be shown below.")
    
    clus_colors = pd.Series(self.assigned_label[self.cell_sorted]).map(dict(zip(np.arange(0, self.cluster_no), self.colors)))
    clus_colors.index = pd.RangeIndex(start = 1, stop = self.cell_total + 1, step = 1)
    
    if self.is_VCF == True:
    
        fig = sns.clustermap(pd.DataFrame(self.AF_sorted[:SNP_no, :], index = self.VCF_filtered["TEXT"].to_numpy()[self.rank_SNP][:SNP_no], columns = np.arange(1, self.cell_total + 1)), row_cluster = False, col_cluster = False, col_colors = clus_colors, figsize = (20, SNP_no * 0.6))
        
    elif self.is_VCF == False:
        
        fig = sns.clustermap(pd.DataFrame(self.AF_sorted[:SNP_no, :], index = self.VCF_filtered[0].to_numpy()[self.rank_SNP][:SNP_no], columns = np.arange(1, self.cell_total + 1)), row_cluster = False, col_cluster = False, col_colors = clus_colors, figsize = (20, SNP_no * 0.6))
    
    fig.ax_col_colors.set_xticks(moving_average(np.cumsum([0] + list(self.cluster_size[self.cluster_order])), 2))
    fig.ax_col_colors.set_xticklabels(np.array(self.cluster_order))
    fig.ax_col_colors.xaxis.set_tick_params(size = 0)
    fig.ax_col_colors.xaxis.tick_top()
    plt.gcf().set_dpi(dpi)
    plt.show()
    
    print("Highest cluster specific SNP-allelic ratio of SNPs sorted by lowest cluster-to-bulk variance ratio will be shown below")
    
    fig, ax1 = plt.subplots(dpi = dpi)
    fig.set_size_inches(10, 7)
    fig.suptitle("SNPs sorted by lowest cluster-to-bulk variance ratio")

    ax1.set_xlabel('SNP')
    ax1.set_ylabel('Variance ratio', color = 'tab:red')
    ax1.plot(np.arange(self.SNP_total), self.ratio_logit_var_ranked, color = 'tab:red')
    ax1.set_ylim([- 0.05, 1.05])
    ax1.tick_params(axis = 'y', labelcolor = 'tab:red')

    ax2 = ax1.twinx()

    color = 'tab:blue'
    ax2.set_ylabel('SNP-allelic ratio', color = 'tab:blue')
    ax2.plot(np.arange(self.SNP_total), self.SNP_cluster_AF_filtered_missing_to_zero_max_ranked, color = 'tab:blue')
    ax2.set_ylim([- 0.05, 1.05])
    ax2.tick_params(axis = 'y', labelcolor = 'tab:blue')

    fig.tight_layout()
    plt.show()


# In[ ]:


class SNP_VAE:
    
    def __init__(self, path = None, mitoSNP_mask = [3107, 310], AD = None, DP = None, VCF = None, variant_name = None, SNPread = "normalized", missing_value = 0.5, cell_weight = "unnormalized"):
        """
        TO ADD
        """
        self.SNPread = SNPread
        self.missing_value = missing_value
        self.cell_weight = cell_weight
        load_data(self, path, mitoSNP_mask, AD, DP, VCF, variant_name)
        
    def filtering(self):
        """TO ADD"""
        filter_data(self)
        
    def training(self, num_epoch = 1000, stepsize = 0.0001, z_dim = None, beta = 0, num_batch = 5):
        """TO ADD"""
        train_VAE(self, num_epoch, stepsize, z_dim, beta, num_batch)
        
    def clustering(self, max_cluster = 15):
        """TO ADD"""
        kmeans_clustering(self, max_cluster)
        
    def phylogeny(self, cluster_no, pair_no = 16, SNP_no = 50):
        """TO ADD"""
        tree(self, cluster_no, pair_no, SNP_no)
        
    def filtering_summary(self, dpi = mpl.rcParams['figure.dpi']):
        """TO ADD"""
        summary_filtering(self, dpi)
        
    def training_summary(self, dpi = mpl.rcParams['figure.dpi']):
        """TO ADD"""
        summary_training(self, dpi)
        
    def clustering_summary(self, dpi = mpl.rcParams['figure.dpi']):
        """TO ADD"""
        summary_clustering(self, dpi)
        
    def phylogeny_summary(self, SNP_no = None, dpi = mpl.rcParams['figure.dpi']):
        """TO ADD"""
        summary_phylogeny(self, SNP_no, dpi)

