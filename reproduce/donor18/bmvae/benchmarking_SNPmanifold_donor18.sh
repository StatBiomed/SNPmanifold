#!/bin/bash

mkdir /home/kevin/storage_kevin/past_figures/figures_final7/notebook/donor18/bmvae/D_3
python bmvae.py --input /home/kevin/storage_kevin/past_figures/figures_final7/notebook/donor18/bmvae/binary_mutation_donor18.tsv --output /home/kevin/storage_kevin/past_figures/figures_final7/notebook/donor18/bmvae/D_3 --Kmax 18 --dimension 3

mkdir /home/kevin/storage_kevin/past_figures/figures_final7/notebook/donor18/bmvae/D_10
python bmvae.py --input /home/kevin/storage_kevin/past_figures/figures_final7/notebook/donor18/bmvae/binary_mutation_donor18.tsv --output /home/kevin/storage_kevin/past_figures/figures_final7/notebook/donor18/bmvae/D_10 --Kmax 18 --dimension 10

mkdir /home/kevin/storage_kevin/past_figures/figures_final7/notebook/donor18/bmvae/D_50
python bmvae.py --input /home/kevin/storage_kevin/past_figures/figures_final7/notebook/donor18/bmvae/binary_mutation_donor18.tsv --output /home/kevin/storage_kevin/past_figures/figures_final7/notebook/donor18/bmvae/D_50 --Kmax 18 --dimension 50

mkdir /home/kevin/storage_kevin/past_figures/figures_final7/notebook/donor18/bmvae/D_87
python bmvae.py --input /home/kevin/storage_kevin/past_figures/figures_final7/notebook/donor18/bmvae/binary_mutation_donor18.tsv --output /home/kevin/storage_kevin/past_figures/figures_final7/notebook/donor18/bmvae/D_87 --Kmax 18 --dimension 87

mkdir /home/kevin/storage_kevin/past_figures/figures_final7/notebook/donor18/bmvae/D_5
python bmvae.py --input /home/kevin/storage_kevin/past_figures/figures_final7/notebook/donor18/bmvae/binary_mutation_donor18.tsv --output /home/kevin/storage_kevin/past_figures/figures_final7/notebook/donor18/bmvae/D_5 --Kmax 18 --dimension 5

mkdir /home/kevin/storage_kevin/past_figures/figures_final7/notebook/donor18/bmvae/D_7
python bmvae.py --input /home/kevin/storage_kevin/past_figures/figures_final7/notebook/donor18/bmvae/binary_mutation_donor18.tsv --output /home/kevin/storage_kevin/past_figures/figures_final7/notebook/donor18/bmvae/D_7 --Kmax 18 --dimension 7
