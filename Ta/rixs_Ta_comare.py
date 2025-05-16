import numpy as np
import sys
sys.path.append('/tren/edrixs/scripts')
from witness import dimer_witness, calc_RIXS_dimerTa

witness = dimer_witness(folder='/project/tren/edrixs/calc_Ba3TaIr2O9')

# for cutmap
RIXS_params = dict(T=9., Gam_c=2.47, sigma0=0.15, sigmas=[0.022,0.022,0.022,0.022,0.054,0.054],
                   fraction=0.5, omega=-559, eloss=np.arange(-0.5,6.01,0.005), eline=None, folder=witness.folder)
HKLs = np.vstack([[-1.698, -1.102, 16.211],
                  [-1.724, -1.083, 13.522],
                  [-0.212, -0.314, 16.148],
                  [-0.185, -0.334, 18.837],
                  [2.017, 0.868, 16.054],
                  [1.3, 0.454, 18.775]])
tth = [85.46, 71.72, 76.88, 92.87, 85.45, 95.7]
th  = [15.53, 4.56, 30.14, 38.85, 63.51, 57.53]
chi = [-95.46, -96.02, -93.29, -93.21, -90.19, -91.42]
phi = [-10.]*6
rixs = calc_RIXS_dimerTa(tth=tth, th=th, chi=chi, phi=phi, HKL=HKLs, **RIXS_params).sum(axis=(0))
rixs = np.vstack([RIXS_params['eloss'], rixs])
np.save(witness.folder+'/compare/rixs.npy', rixs)

# for Emap
RIXS_params = dict(T=9.0, Gam_c=2.47, sigma0=0.15, sigmas=[0.022,0.022,0.022,0.022,0.054,0.054],
                   fraction=0.5, omega=np.arange(-570,-540.01,0.2), eloss=np.arange(-1.,10.01,0.005), eline=None, folder=witness.folder)
HKLs = [-5.00033147e-01,  1.34356932e-05,  1.83975559e+01]
tth = 90.1645
th  = 40.95725
chi = -91.83391
phi = 0.
rixs = calc_RIXS_dimerTa(tth=tth, th=th, chi=chi, phi=phi, HKL=HKLs, **RIXS_params)
rixs_Emap = dict(omega=RIXS_params['omega'], eloss=RIXS_params['eloss'], rixs=np.squeeze(rixs))
np.savez(witness.folder+'/compare/rixs_Emap.npz', **rixs_Emap)

# for Lmap
RIXS_params = dict(T=9., Gam_c=2.47, sigma0=0.15, sigmas=[0.022,0.022,0.022,0.022,0.054,0.054,0.171,0.171,0.237,0.237],
                   fraction=0.5, omega=-559, eloss=np.arange(-0.5,6.01,0.005), eline=None, folder=witness.folder)
motors = np.vstack([[-0.2773941488281869, -0.2659741791152685, 9.51042684165467, 43.264375, 9.81525, -93.663173, -10.0],
                    [-0.26764896087249423, -0.273302932756571, 10.51087405002826, 47.98975, 12.9845, -93.576479, -10.0],
                    [-0.2577883546729399, -0.2805792776763275, 11.510507877509921, 52.808, 16.063875, -93.504979, -10.0],
                    [-0.24777219675676215, -0.28779323774242505, 12.509375498043902, 57.73175, 19.0925, -93.445016, -10.0],
                    [-0.23796699084652193, -0.29510625975440596, 13.509375345735329, 62.78675, 22.101, -93.394073, -10.0],
                    [-0.22785287701054702, -0.30221098200350494, 14.507727481478634, 67.976875, 25.115875, -93.349547, -10.0],
                    [-0.22341693910140217, -0.30610439562840724, 15.009870255621223, 70.650125, 26.633375, -93.33021, -10.0],
                    [-0.2180022136685174, -0.30950033549205264, 15.507542476207073, 73.343, 28.1615, -93.31136, -10.0],
                    [-0.21346254040160886, -0.31339680456990965, 16.009171507913727, 76.10825, 29.703375, -93.294785, -10.0],
                    [-0.20821606842705367, -0.31680014487519276, 16.507547237513325, 78.907625, 31.262125, -93.277723, -10.0],
                    [-0.20356182986295102, -0.3206686216079151, 17.008601568605542, 81.781875, 32.841, -93.26326, -10.0],
                    [-0.19849298265545512, -0.32414055133336717, 17.50771083979936, 84.708125, 34.44375, -93.248066, -10.0],
                    [-0.19360491108270322, -0.32791598960334883, 18.00789856761717, 87.711875, 36.07425, -93.235229, -10.0],
                    [-0.18850285302241201, -0.33138267089740425, 18.506986904695648, 90.786375, 37.73675, -93.22166, -10.0],
                    [-0.18369770751773082, -0.3351495940271972, 19.007283337061917, 93.955875, 39.436125, -93.20996, -10.0],
                    [-0.17973557818122593, -0.3379546434534219, 19.406928947872096, 96.55675, 40.826125, -93.200047, -10.0]])
Ls = np.arange(9.6,19.4,0.1)
Hs = np.interp(Ls, motors[:,2], motors[:,0])
Ks = np.interp(Ls, motors[:,2], motors[:,1])
HKLs = np.vstack([Hs, Ks, Ls]).T
tth = np.interp(Ls, motors[:,2], motors[:,3])
th  = np.interp(Ls, motors[:,2], motors[:,4])
chi = np.interp(Ls, motors[:,2], motors[:,5])
phi = np.interp(Ls, motors[:,2], motors[:,6])
rixs = calc_RIXS_dimerTa(tth=tth, th=th, chi=chi, phi=phi, HKL=HKLs, **RIXS_params).sum(axis=(0))
rixs_Ldep = dict(Ls=Ls, eloss=RIXS_params['eloss'], rixs=rixs)
np.savez(witness.folder+'/compare/rixs_Ldep.npz', **rixs_Ldep)