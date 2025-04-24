#!/bin/bash

#SBATCH --job-name=polCalc
#SBATCH --mail-user=tr2401@columbia.edu
#SBATCH --mail-type=begin,end

#SBATCH --partition=thal5
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=1G
#SBATCH --time=0-04:00:00

python pol_rixs_E.py
python pol_denom_E.py
python pol_FQ_E.py
python pol_rixs_M.py
python pol_denom_M.py
python pol_FQ_M.py