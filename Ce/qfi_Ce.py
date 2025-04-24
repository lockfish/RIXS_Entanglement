import numpy as np
from witness import dimer_witness, QFI_denom_dimerCe

witness = dimer_witness(folder='/project/tren/edrixs/calc_Ba3CeIr2O9')

# polarized, Emap, denominator calculation
L = 18.94
Q = np.sqrt(1.228965**2/4+(L*0.426991866)**2)
tth = np.rad2deg(np.arcsin(1.1054837271511369*Q/4./np.pi)*2)
thin = tth/2 - np.rad2deg(np.arctan(1.228965/2/L/0.426991866))
omega_list = np.arange(-570,-540,0.3)
denom_list = []
for omega in omega_list:
	denom_params = dict(Gam_c=2.47, omega=omega, pol=[(0, 0), (0, np.pi/2.0)], T=9., folder=witness.folder)
	denom = QFI_denom_dimerCe(tth=tth, thin=thin, HKL=[-0.5,0.0,L], phi=10.0, **denom_params)
	denom_list.append(np.squeeze(denom))
denom_Emap = dict(omega=omega_list, pol=denom_params['pol'], denom=np.array(denom_list))
np.savez(witness.folder+'/pol/denom_Emap.npz', **denom_Emap)

# polarized, Mmap, denominator calculation
Ls = np.arange(0.0, 20.1, 0.1)
Qs = np.sqrt(1.228965**2/4+(Ls*0.426991866)**2)
tths = np.rad2deg(np.arcsin(1.1054837271511369*Qs/4./np.pi)*2)
thins = tths/2 - np.rad2deg(np.arctan(1.228965/2/Ls/0.426991866))
denom_list = []
for tth, thin, L in zip(tths, thins, Ls):
	denom_params = dict(Gam_c=2.47, omega=-558.5, pol=[(0, 0), (0, np.pi/2.0)], T=9., folder=witness.folder)
	denom = QFI_denom_dimerCe(tth=tth, thin=thin, HKL=[-0.5,0.0,L], phi=10.0, **denom_params)
	denom_list.append(np.squeeze(denom))
denom_Mmap = dict(momenta=Qs, pol=denom_params['pol'], denom=np.array(denom_list))
np.savez(witness.folder+'/pol/denom_Mmap.npz', **denom_Mmap)
