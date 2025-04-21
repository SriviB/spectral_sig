#!/bin/bash

#SBATCH -A CCR23048
#SBATCH -J MLSEC           # Job name
#SBATCH -o status/cifar10.o%j       # Name of stdout output file
#SBATCH -e status/cifar10.e%j       # Name of stderr error file
#SBATCH -p gpu-a100-small       # Queue (partition) name
#SBATCH -N 1                # Total # of nodes (must be 1 for serial)
#SBATCH -n 1                # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 01:00:00         # Run time (hh:mm:ss)
#SBATCH --mail-type=all     # Send email at begin and end of job
#SBATCH --mail-user=saloni.a.modi@utexas.edu

set -e

cd $SCRATCH
eval "$(conda shell.bash hook)"
conda activate bb_audit_dpsgd

cd spectral_sig

python3 main.py --scoring_fn madry_compute_corr --n_epochs 100 --out cifar10/madry_compute_corr
python3 main.py --scoring_fn madry_compute_corr_theta --n_epochs 100 --out cifar10/madry_compute_corr_theta --pretrained_fp cifar10/madry_compute_corr/model.pth
python3 main.py --scoring_fn madry_compute_corr_scaled --n_epochs 100 --out cifar10/madry_compute_corr_scaled --pretrained_fp cifar10/madry_compute_corr/model.pth
python3 main.py --scoring_fn madry_compute_corr_scaled_theta --n_epochs 100 --out cifar10/madry_compute_corr_scaled_theta --pretrained_fp cifar10/madry_compute_corr/model.pth
python3 main.py --scoring_fn madry_compute_corr_all --n_epochs 100 --out cifar10/madry_compute_corr_all --pretrained_fp cifar10/madry_compute_corr/model.pth
python3 main.py --scoring_fn madry_compute_corr_all_theta --n_epochs 100 --out cifar10/madry_compute_corr_all_theta --pretrained_fp cifar10/madry_compute_corr/model.pth
python3 main.py --scoring_fn madry_compute_corr_all_scaled --n_epochs 100 --out cifar10/madry_compute_corr_all_scaled --pretrained_fp cifar10/madry_compute_corr/model.pth
python3 main.py --scoring_fn madry_compute_corr_all_scaled_theta --n_epochs 100 --out cifar10/madry_compute_corr_all_scaled_theta --pretrained_fp cifar10/madry_compute_corr/model.pth
python3 main.py --scoring_fn whitened_norm --n_epochs 100 --out cifar10/whitened_norm --pretrained_fp cifar10/madry_compute_corr/model.pth
python3 main.py --scoring_fn whitened_norm_all --out cifar10/whitened_norm_all --pretrained_fp cifar10/madry_compute_corr/model.pth
python3 main.py --scoring_fn norm --n_epochs 100 --out cifar10/norm --pretrained_fp cifar10/madry_compute_corr/model.pth
python3 main.py --scoring_fn norm_all --n_epochs 100 --out cifar10/norm_all --pretrained_fp cifar10/madry_compute_corr/model.pth
python3 main.py --scoring_fn norm_scaled --n_epochs 100 --out norm_scaled --pretrained_fp cifar10/madry_compute_corr/model.pth
python3 main.py --scoring_fn norm_all_scaled --n_epochs 100 --out cifar10/norm_all_scaled --pretrained_fp cifar10/madry_compute_corr/model.pth