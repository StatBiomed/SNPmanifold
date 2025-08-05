#!/bin/bash

mkdir /home/kevin/storage_kevin/past_figures/figures_final7/notebook/donor8/bmvae/D_3
python bmvae.py --input /home/kevin/storage_kevin/past_figures/figures_final7/notebook/donor8/bmvae/binary_mutation_donor8.tsv --output /home/kevin/storage_kevin/past_figures/figures_final7/notebook/donor8/bmvae/D_3 --Kmax 8 --dimension 3

mkdir /home/kevin/storage_kevin/past_figures/figures_final7/notebook/donor8/bmvae/D_10
python bmvae.py --input /home/kevin/storage_kevin/past_figures/figures_final7/notebook/donor8/bmvae/binary_mutation_donor8.tsv --output /home/kevin/storage_kevin/past_figures/figures_final7/notebook/donor8/bmvae/D_10 --Kmax 8 --dimension 10

mkdir /home/kevin/storage_kevin/past_figures/figures_final7/notebook/donor8/bmvae/D_50
python bmvae.py --input /home/kevin/storage_kevin/past_figures/figures_final7/notebook/donor8/bmvae/binary_mutation_donor8.tsv --output /home/kevin/storage_kevin/past_figures/figures_final7/notebook/donor8/bmvae/D_50 --Kmax 8 --dimension 50

mkdir /home/kevin/storage_kevin/past_figures/figures_final7/notebook/donor8/bmvae/D_93
python bmvae.py --input /home/kevin/storage_kevin/past_figures/figures_final7/notebook/donor8/bmvae/binary_mutation_donor8.tsv --output /home/kevin/storage_kevin/past_figures/figures_final7/notebook/donor8/bmvae/D_93 --Kmax 8 --dimension 93

mkdir /home/kevin/storage_kevin/past_figures/figures_final7/notebook/donor8/bmvae/D_5
python bmvae.py --input /home/kevin/storage_kevin/past_figures/figures_final7/notebook/donor8/bmvae/binary_mutation_donor8.tsv --output /home/kevin/storage_kevin/past_figures/figures_final7/notebook/donor8/bmvae/D_5 --Kmax 8 --dimension 5

mkdir /home/kevin/storage_kevin/past_figures/figures_final7/notebook/donor8/bmvae/D_7
python bmvae.py --input /home/kevin/storage_kevin/past_figures/figures_final7/notebook/donor8/bmvae/binary_mutation_donor8.tsv --output /home/kevin/storage_kevin/past_figures/figures_final7/notebook/donor8/bmvae/D_7 --Kmax 8 --dimension 7
