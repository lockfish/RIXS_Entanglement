#!/bin/bash

#SBATCH --job-name=polCalc
#SBATCH --mail-user=tr2401@columbia.edu
#SBATCH --mail-type=begin,end

#SBATCH --partition=thal5
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=1G
#SBATCH --time=0-24:00:00

python python qfi_pol_Ta.py