#!/bin/bash
#SBATCH -A CCR23048 # project allocation
#SBATCH -J SpectralSigTrain # job name
#SBATCH -o status/spectral_sig.o%j # stdout output file
#SBATCH -e status/spectral_sig.e%j # stderr error file
#SBATCH -p gpu-a100-small # GPU partition
#SBATCH -N 1 # 1 node
#SBATCH -n 1 # 1 task
#SBATCH -t 7:00:00 # time out after that much
#SBATCH --mail-type=ALL
#SBATCH --mail-user=srivibalaji@utexas.edu

module load cuda

cd $WORK
conda init
conda activate bb_venv
cd spectral_sig/

python train.py --config config.json