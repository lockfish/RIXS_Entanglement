import numpy as np
import sys
sys.path.append('/tren/edrixs/scripts')
from witness import dimer_witness, calc_RIXS_dimerCe

# for cutmap

witness = dimer_witness(folder='/project/tren/edrixs_compare/calc_Ba3CeIr2O9')
RIXS_params = dict(Gam_c=2.47, sigma=0.075, fraction=0., omega=-559, eloss=np.arange(-0.5,6.01,0.01),
                   pol=[(0, 0), (0, np.pi/2.0)], T=9., eline=None, folder=witness.folder)
rixs1 = calc_RIXS_dimerCe(tth=96.6, thin=19.75, HKL=[-3.3,0.,17.48], phi=10., **RIXS_params).sum(axis=(0,1))
rixs2 = calc_RIXS_dimerCe(tth=81.5, thin=7.62,  HKL=[-3.3,0.,14.57], phi=10., **RIXS_params).sum(axis=(0,1))
rixs3 = calc_RIXS_dimerCe(tth=82.3, thin=36.43, HKL=[-0.5,0.,17.48], phi=10., **RIXS_params).sum(axis=(0,1))
rixs4 = calc_RIXS_dimerCe(tth=66.6, thin=27.67, HKL=[-0.5,0.,14.57], phi=10., **RIXS_params).sum(axis=(0,1))
rixs5 = calc_RIXS_dimerCe(tth=82.3, thin=45.86, HKL=[0.5,0.,17.48],  phi=10., **RIXS_params).sum(axis=(0,1))
rixs6 = calc_RIXS_dimerCe(tth=66.6, thin=38.96, HKL=[-0.5,0.,14.57], phi=10., **RIXS_params).sum(axis=(0,1))
rixs = np.vstack([RIXS_params['eloss'], rixs1, rixs2, rixs3, rixs4, rixs5, rixs6])
np.save(witness.folder+'/compare/rixs.npy', rixs)

# for Emap
witness = dimer_witness(folder='/project/tren/edrixs/calc_Ba3CeIr2O9')
RIXS_params = dict(Gam_c=2.47, sigma=0.075, fraction=0., omega=np.arange(-570,-540.01,0.2), eloss=np.arange(-1.,10.01,0.005),
                   pol=[(0, 0), (0, np.pi/2.0)], T=9., eline=None, folder=witness.folder)
rixs = calc_RIXS_dimerCe(tth=90.89, thin=38.84, HKL=[-0.5,0.,18.94], phi=10., **RIXS_params)
rixs_Emap = dict(omega=RIXS_params['omega'], eloss=RIXS_params['eloss'], rixs=np.squeeze(rixs))
np.savez(witness.folder+'/compare/rixs_Emap.npz', **rixs_Emap)

# for Lmap
RIXS_params = dict(Gam_c=2.47, sigma=0.075, fraction=0., omega=-558.5, eloss=np.arange(-0.5,6.01,0.01),
                   pol=[(0, 0), (0, np.pi / 2.0)], T=9., eline=None, folder=witness.folder)
Ls = np.arange(6.,20.1,0.1)
Qs = np.sqrt(1.228965**2/4+(Ls*0.426991866)**2)
tths = np.rad2deg(np.arcsin(1.1054837271511369*Qs/4./np.pi)*2)
thins = tths/2 - np.rad2deg(np.arctan(1.228965/2/Ls/0.426991866))
rixs_all = []
for tth, thin, L in zip(tths, thins, Ls):
    rixs_all.append(calc_RIXS_dimerCe(tth=tth, thin=thin, HKL=[-0.5,0.,L], phi=10., **RIXS_params).sum(axis=(0,1)))
rixs_Ldep = dict(Ls=Ls, eloss=RIXS_params['eloss'], rixs=np.vstack(rixs_all))
np.savez(witness.folder+'/compare/rixs_Ldep.npz', **rixs_Ldep)
