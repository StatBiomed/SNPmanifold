#!/bin/bash

mkdir /home/kevin/storage_kevin/past_figures/figures_final7/notebook/donor4/bmvae/D_3
python bmvae.py --input /home/kevin/storage_kevin/past_figures/figures_final7/notebook/donor4/bmvae/binary_mutation_donor4.tsv --output /home/kevin/storage_kevin/past_figures/figures_final7/notebook/donor4/bmvae/D_3 --Kmax 4 --dimension 3

mkdir /home/kevin/storage_kevin/past_figures/figures_final7/notebook/donor4/bmvae/D_10
python bmvae.py --input /home/kevin/storage_kevin/past_figures/figures_final7/notebook/donor4/bmvae/binary_mutation_donor4.tsv --output /home/kevin/storage_kevin/past_figures/figures_final7/notebook/donor4/bmvae/D_10 --Kmax 4 --dimension 10

mkdir /home/kevin/storage_kevin/past_figures/figures_final7/notebook/donor4/bmvae/D_50
python bmvae.py --input /home/kevin/storage_kevin/past_figures/figures_final7/notebook/donor4/bmvae/binary_mutation_donor4.tsv --output /home/kevin/storage_kevin/past_figures/figures_final7/notebook/donor4/bmvae/D_50 --Kmax 4 --dimension 50

mkdir /home/kevin/storage_kevin/past_figures/figures_final7/notebook/donor4/bmvae/D_191
python bmvae.py --input /home/kevin/storage_kevin/past_figures/figures_final7/notebook/donor4/bmvae/binary_mutation_donor4.tsv --output /home/kevin/storage_kevin/past_figures/figures_final7/notebook/donor4/bmvae/D_191 --Kmax 4 --dimension 191

mkdir /home/kevin/storage_kevin/past_figures/figures_final7/notebook/donor4/bmvae/D_5
python bmvae.py --input /home/kevin/storage_kevin/past_figures/figures_final7/notebook/donor4/bmvae/binary_mutation_donor4.tsv --output /home/kevin/storage_kevin/past_figures/figures_final7/notebook/donor4/bmvae/D_5 --Kmax 4 --dimension 5

mkdir /home/kevin/storage_kevin/past_figures/figures_final7/notebook/donor4/bmvae/D_7
python bmvae.py --input /home/kevin/storage_kevin/past_figures/figures_final7/notebook/donor4/bmvae/binary_mutation_donor4.tsv --output /home/kevin/storage_kevin/past_figures/figures_final7/notebook/donor4/bmvae/D_7 --Kmax 4 --dimension 7
