import numpy as np
import sys
sys.path.append('/tren/edrixs/scripts')
from witness import dimer_witness

witness = dimer_witness(folder='/project/tren/edrixs/calc_Ba3CeIr2O9')
witness.construct_basis(10, eg_max=1)
witness.produce_dipole()

emat_params = dict(tenDq=3.5, soc=0.33, theta=50.86, kappa=0.1, a0=0., Vdd_sigma=-1.4, Vdd_pi=-0.66, Vdd_delta=0.05)
umat_params = dict(U_dd=3., J_dd=0.4, U_q=2.5, F2_dp=1.03*0.9, G1_dp=0.889*0.9, G3_dp=0.528*0.9)
witness.load_params(**emat_params, **umat_params)
witness.perform_ED()
