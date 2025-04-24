import numpy as np
import sys
sys.path.append('/tren/edrixs/scripts')
from witness import dimer_witness
from witness import calc_RIXS_dimerCe_sum, conjugate_RIXS_dimerCe_sum
from witness import QFI_denom_dimerCe_sum, calc_FQ_dimerCe_sum

witness = dimer_witness(folder='/project/tren/edrixs/calc_Ba3CeIr2O9')

# sum, Emap, rixs calcuation (with conjugate part)
L = 18.94
Q = np.sqrt(1.228965**2/4+(L*0.426991866)**2)
tth = np.rad2deg(np.arcsin(1.1054837271511369*Q/4./np.pi)*2)
thin = tth/2 - np.rad2deg(np.arctan(1.228965/2/L/0.426991866))
RIXS_params = dict(Gam_c=2.47, sigma=0.075, fraction=0., omega=np.arange(-570,-540,0.3), eloss=np.arange(0.,10,0.005),
                   pol=[(0, 0), (0, np.pi/2.0)], T=9., eline=None, folder=witness.folder)

rixs = calc_RIXS_dimerCe_sum(tth=tth, thin=thin, HKL=[-0.5,0.0,L], phi=10.0, **RIXS_params)
rixs_Emap = dict(omega=RIXS_params['omega'], eloss=RIXS_params['eloss'], rixs=np.squeeze(rixs))
np.savez(witness.folder+'/sum/rixs_Emap.npz', **rixs_Emap)

conjugate_rixs = conjugate_RIXS_dimerCe_sum(tth=tth, thin=thin, HKL=[-0.5,0.0,L], phi=10.0, **RIXS_params)
conjugate_rixs_Emap = dict(omega=RIXS_params['omega'], eloss=RIXS_params['eloss'], rixs=np.squeeze(conjugate_rixs))
np.savez(witness.folder+'/sum/conjugate_rixs_Emap.npz', **conjugate_rixs_Emap)

# sum, Mmap, rixs calculation (with conjugate part)
RIXS_params = dict(Gam_c=2.47, sigma=0.075, fraction=0., omega=-558.5, eloss=np.arange(0.,10,0.005),
                   pol=[(0, 0), (0, np.pi/2.0)], T=9., eline=None, folder=witness.folder)
Ls = np.arange(0.0, 20.1, 0.1)
Qs = np.sqrt(1.228965**2/4+(Ls*0.426991866)**2)
tths = np.rad2deg(np.arcsin(1.1054837271511369*Qs/4./np.pi)*2)
thins = tths/2 - np.rad2deg(np.arctan(1.228965/2/Ls/0.426991866))

rixs_all = []
for tth, thin, L in zip(tths, thins, Ls):
    rixs_all.append(calc_RIXS_dimerCe_sum(tth=tth, thin=thin, HKL=[-0.5,0.0,L], phi=10.0, **RIXS_params))
rixs_Mmap = dict(momenta=Qs, eloss=RIXS_params['eloss'], rixs=np.squeeze(rixs_all))
np.savez(witness.folder+'/sum/rixs_Mmap.npz', **rixs_Mmap)

conjugate_rixs_all = []
for tth, thin, L in zip(tths, thins, Ls):
    conjugate_rixs_all.append(conjugate_RIXS_dimerCe_sum(tth=tth, thin=thin, HKL=[-0.5,0.0,L], phi=10.0, **RIXS_params))
conjugate_rixs_Mmap = dict(momenta=Qs, eloss=RIXS_params['eloss'], rixs=np.squeeze(conjugate_rixs_all))
np.savez(witness.folder+'/sum/conjugate_rixs_Mmap.npz', **conjugate_rixs_Mmap)

# sum, Emap, denominator calculation
L = 18.94
Q = np.sqrt(1.228965**2/4+(L*0.426991866)**2)
tth = np.rad2deg(np.arcsin(1.1054837271511369*Q/4./np.pi)*2)
thin = tth/2 - np.rad2deg(np.arctan(1.228965/2/L/0.426991866))
omega_list = np.arange(-570,-540,0.3)
denom_list = []
for omega in omega_list:
	denom_params = dict(Gam_c=2.47, omega=omega, T=9., folder=witness.folder)
	denom = QFI_denom_dimerCe_sum(tth=tth, thin=thin, HKL=[-0.5,0.0,L], phi=10.0, **denom_params)
	denom_list.append(np.squeeze(denom))
denom_Emap = dict(omega=omega_list, denom=np.array(denom_list))
np.savez(witness.folder+'/sum/denom_Emap.npz', **denom_Emap)

# sum, Mmap, denominator calculation
Ls = np.arange(0.0, 20.1, 0.1)
Qs = np.sqrt(1.228965**2/4+(Ls*0.426991866)**2)
tths = np.rad2deg(np.arcsin(1.1054837271511369*Qs/4./np.pi)*2)
thins = tths/2 - np.rad2deg(np.arctan(1.228965/2/Ls/0.426991866))
denom_list = []
for tth, thin, L in zip(tths, thins, Ls):
	denom_params = dict(Gam_c=2.47, omega=-558.5, T=9., folder=witness.folder)
	denom = QFI_denom_dimerCe_sum(tth=tth, thin=thin, HKL=[-0.5,0.0,L], phi=10.0, **denom_params)
	denom_list.append(np.squeeze(denom))
denom_Mmap = dict(momenta=Qs, denom=np.array(denom_list))
np.savez(witness.folder+'/sum/denom_Mmap.npz', **denom_Mmap)

# sum, Emap, QFI calculation for pure states
L = 18.94
Q = np.sqrt(1.228965**2/4+(L*0.426991866)**2)
tth = np.rad2deg(np.arcsin(1.1054837271511369*Q/4./np.pi)*2)
thin = tth/2 - np.rad2deg(np.arctan(1.228965/2/L/0.426991866))
omega = np.arange(-570,-540,0.3)
FQ = np.zeros(len(omega), dtype=float)
for nom, om in enumerate(omega):
	FQ_params = dict(Gam_c=2.47, omega=om, pol=[(0,0), (0, np.pi/2.0)], folder=witness.folder)
	FQ[nom] = calc_FQ_dimerCe_sum(tth=tth, thin=thin, HKL=[-0.5,0.0,L], phi=10.0, **FQ_params)
FQ_Emap = dict(omega=omega, FQ=FQ)
np.savez(witness.folder+'/sum/FQ_Emap.npz', **FQ_Emap)

# sum, Mmap, QFI calculation for pure states
omega=-558.5
Ls = np.arange(0.0, 20.1, 0.1)
Qs = np.sqrt(1.228965**2/4+(Ls*0.426991866)**2)
tths = np.rad2deg(np.arcsin(1.1054837271511369*Qs/4./np.pi)*2)
thins = tths/2 - np.rad2deg(np.arctan(1.228965/2/Ls/0.426991866))
FQ = np.zeros(len(Ls), dtype=float)
for nth, (tth, thin, L) in enumerate(zip(tths, thins, Ls)):
    FQ_params = dict(Gam_c=2.47, omega=omega, pol=[(0,0), (0, np.pi/2.0)], folder=witness.folder)
    FQ[nth] = calc_FQ_dimerCe_sum(tth=tth, thin=thin, HKL=[-0.5,0.0,L], phi=10.0, **FQ_params)
FQ_Mmap = dict(momenta=Qs, FQ=FQ)
np.savez(witness.folder+'/sum/FQ_Mmap.npz', **FQ_Mmap)
