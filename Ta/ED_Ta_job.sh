#!/bin/bash

#SBATCH --job-name=edrixsED
#SBATCH --mail-user=tren@bnl.gov
#SBATCH --mail-type=begin,end

#SBATCH --partition=thal5
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=1G
#SBATCH --time=0-08:00:00

python ED_Ta.py > ED_Ta.txt
python Fixed_N_Ta.py