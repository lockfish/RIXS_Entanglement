import numpy as np
import sys
sys.path.append('/tren/edrixs/scripts')
from witness import dimer_witness
from witness import calc_RIXS_dimerCe, conjugate_RIXS_dimerCe
from witness import QFI_denom_dimerCe, calc_FQ_dimerCe

witness = dimer_witness(folder='/project/tren/edrixs/calc_Ba3CeIr2O9')

# polarized, Emap, rixs calculation (with conjugate part)
L = 18.94
Q = np.sqrt(1.228965**2/4+(L*0.426991866)**2)
tth = np.rad2deg(np.arcsin(1.1054837271511369*Q/4./np.pi)*2)
thin = tth/2 - np.rad2deg(np.arctan(1.228965/2/L/0.426991866))
RIXS_params = dict(Gam_c=2.47, sigma=0.075, fraction=0., omega=np.arange(-570,-540,0.3), eloss=np.arange(0.,10,0.005),
                   pol=[(0, 0), (0, np.pi/2.0)], T=9., eline=None, folder=witness.folder)

rixs = calc_RIXS_dimerCe(tth=tth, thin=thin, HKL=[-0.5,0.0,L], phi=10.0, **RIXS_params)
rixs_Emap = dict(pol=RIXS_params['pol'], omega=RIXS_params['omega'], eloss=RIXS_params['eloss'], rixs=np.squeeze(rixs))
np.savez(witness.folder+'/pol/rixs_Emap.npz', **rixs_Emap)

conjugate_rixs = conjugate_RIXS_dimerCe(tth=tth, thin=thin, HKL=[-0.5,0.0,L], phi=10.0, **RIXS_params)
conjugate_rixs_Emap = dict(pol=RIXS_params['pol'], omega=RIXS_params['omega'], eloss=RIXS_params['eloss'], rixs=np.squeeze(conjugate_rixs))
np.savez(witness.folder+'/pol/conjugate_rixs_Emap.npz', **conjugate_rixs_Emap)

# polarized, Mmap, rixs calculation (with conjugate part)
RIXS_params = dict(Gam_c=2.47, sigma=0.075, fraction=0., omega=-558.5, eloss=np.arange(0.,10,0.005),
                   pol=[(0, 0), (0, np.pi/2.0)], T=9., eline=None, folder=witness.folder)
Ls = np.arange(0.0, 20.1, 0.1)
Qs = np.sqrt(1.228965**2/4+(Ls*0.426991866)**2)
tths = np.rad2deg(np.arcsin(1.1054837271511369*Qs/4./np.pi)*2)
thins = tths/2 - np.rad2deg(np.arctan(1.228965/2/Ls/0.426991866))

rixs_all = []
for tth, thin, L in zip(tths, thins, Ls):
    rixs_all.append(calc_RIXS_dimerCe(tth=tth, thin=thin, HKL=[-0.5,0.0,L], phi=10.0, **RIXS_params))
rixs_Mmap = dict(momenta=Qs, pol=RIXS_params['pol'], eloss=RIXS_params['eloss'], rixs=np.squeeze(rixs_all))
np.savez(witness.folder+'/pol/rixs_Mmap.npz', **rixs_Mmap)

conjugate_rixs_all = []
for tth, thin, L in zip(tths, thins, Ls):
    conjugate_rixs_all.append(conjugate_RIXS_dimerCe(tth=tth, thin=thin, HKL=[-0.5,0.0,L], phi=10.0, **RIXS_params))
conjugate_rixs_Mmap = dict(momenta=Qs, pol=RIXS_params['pol'], eloss=RIXS_params['eloss'], rixs=np.squeeze(conjugate_rixs_all))
np.savez(witness.folder+'/pol/conjugate_rixs_Mmap.npz', **conjugate_rixs_Mmap)

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

# polarized, Emap, QFI for pure states
L = 18.94
Q = np.sqrt(1.228965**2/4+(L*0.426991866)**2)
tth = np.rad2deg(np.arcsin(1.1054837271511369*Q/4./np.pi)*2)
thin = tth/2 - np.rad2deg(np.arctan(1.228965/2/L/0.426991866))
omega = np.arange(-570,-540,0.3)
FQ = np.zeros((2, len(omega)), dtype=float)
for nom, om in enumerate(omega):
	for npol, pol in enumerate([(0,0), (0, np.pi/2.0)]):
		FQ_params = dict(Gam_c=2.47, omega=om, pol=pol, folder=witness.folder)
		FQ[npol][nom] = calc_FQ_dimerCe(tth=tth, thin=thin, HKL=[-0.5,0.0,L], phi=10.0, **FQ_params)
FQ_Emap = dict(omega=omega, FQ=FQ)
np.savez(witness.folder+'/pol/FQ_Emap.npz', **FQ_Emap)

# polarized Mmap, QFI for pure states
omega=-558.5
Ls = np.arange(0.0, 20.1, 0.1)
Qs = np.sqrt(1.228965**2/4+(Ls*0.426991866)**2)
tths = np.rad2deg(np.arcsin(1.1054837271511369*Qs/4./np.pi)*2)
thins = tths/2 - np.rad2deg(np.arctan(1.228965/2/Ls/0.426991866))

FQ = np.zeros((2, len(Ls)), dtype=float)
for nth, (tth, thin, L) in enumerate(zip(tths, thins, Ls)):
    for npol, pol in enumerate([(0,0), (0, np.pi/2.0)]):
        FQ_params = dict(Gam_c=2.47, omega=omega, pol=pol, folder=witness.folder)
        FQ[npol][nth] = calc_FQ_dimerCe(tth=tth, thin=thin, HKL=[-0.5,0.0,L], phi=10.0, **FQ_params)
FQ_Mmap = dict(momenta=Qs, FQ=FQ)
np.savez(witness.folder+'/pol/FQ_Mmap.npz', **FQ_Mmap)

