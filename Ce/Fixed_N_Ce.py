import numpy as np
import sys
sys.path.append('/tren/edrixs/scripts')
from witness_utils import get_fock_bin_by_N_S, SparseBinary

folder='/project/tren/edrixs/calc_Ba3CeIr2O9'

all_basis = []
eg_max = 1
noccu = 10
for eg in np.arange(0,eg_max+1):
    all_basis.append(get_fock_bin_by_N_S(8,eg, 12,noccu-eg, dispinfo=False))
basis = SparseBinary(np.vstack(all_basis), shape=[None,20])
# change basis from the complex harmonic basis to the real harmonic basis
coo_table = np.hstack([0,1,6,7,10,11,16,17,2,3,4,5,8,9,12,13,14,15,18,19])
basis.change_coordinates(coo_table, shape=[None,20])

basis_64 = get_fock_bin_by_N_S(8,0,6,6,6,4,dispinfo=False)
basis_64 = SparseBinary(np.vstack(basis_64), shape=[None,20])
# change basis from the complex harmonic basis to the real harmonic basis
coo_table = np.hstack([0,1,6,7,10,11,16,17,2,3,4,5,8,9,12,13,14,15,18,19])
basis_64.change_coordinates(coo_table, shape=[None,20])

basis_55 = get_fock_bin_by_N_S(8,0,6,5,6,5,dispinfo=False)
basis_55 = SparseBinary(np.vstack(basis_55), shape=[None,20])
# change basis from the complex harmonic basis to the real harmonic basis
coo_table = np.hstack([0,1,6,7,10,11,16,17,2,3,4,5,8,9,12,13,14,15,18,19])
basis_55.change_coordinates(coo_table, shape=[None,20])

basis_46 = get_fock_bin_by_N_S(8,0,6,4,6,6,dispinfo=False)
basis_46 = SparseBinary(np.vstack(basis_46), shape=[None,20])
# change basis from the complex harmonic basis to the real harmonic basis
coo_table = np.hstack([0,1,6,7,10,11,16,17,2,3,4,5,8,9,12,13,14,15,18,19])
basis_46.change_coordinates(coo_table, shape=[None,20])

def getMatrixElement(state1, state2):
	for i in range(len(state1)):
		if state1[i] != state2[i]:
			return 0
	return 1


states_64 = np.zeros((15,1826))
for i in range(15):
	for j in range(66):
		state1 = basis_64.toarray()[i]
		state2 = basis.toarray()[j]
		states_64[i,j] = getMatrixElement(state1, state2)

states_55 = np.zeros((36,1826))
for i in range(36):
	for j in range(66):
		state1 = basis_55.toarray()[i]
		state2 = basis.toarray()[j]
		states_55[i,j] = getMatrixElement(state1, state2)

states_46 = np.zeros((15,1826))
for i in range(15):
	for j in range(66):
		state1 = basis_46.toarray()[i]
		state2 = basis.toarray()[j]
		states_46[i,j] = getMatrixElement(state1, state2)

np.save(folder+'/states_64.npy', states_64)
np.save(folder+'/states_55.npy', states_55)
np.save(folder+'/states_46.npy', states_46)
