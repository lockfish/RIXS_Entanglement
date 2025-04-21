#!/bin/bash

#SBATCH --job-name=edrixsCal
#SBATCH --mail-user=tren@bnl.gov
#SBATCH --mail-type=begin,end

#SBATCH --partition=thal5
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=1G
#SBATCH --time=0-04:00:00

python rixs_Ce_compare.py