#!/bin/bash

#SBATCH --job-name=sumCalc
#SBATCH --mail-user=tr2401@columbia.edu
#SBATCH --mail-type=begin,end

#SBATCH --partition=thal5
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=6G
#SBATCH --time=0-015:00:00

python sum_rixs_E.py
python sum_denom_E.py
python sum_FQ_E.py
python sum_rixs_M.py
python sum_denom_M.py
python sum_FQ_M.py