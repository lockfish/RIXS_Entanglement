# Adapted from edrixs_utils in application to entanglement witness

import numpy as np
import itertools
import scipy.sparse as sparse
from scipy.sparse.linalg import eigs, eigsh
from scipy.linalg import eig, eigh
from collections import defaultdict
from edrixs import scattering_mat, dipole_polvec_rixs

__all__ = ['SparseBinary', 'Sparse4D', 'get_fock_bin_by_N_S', 'two_fermion_S', 'four_fermion_S',
           'calc_RIXS', 'calc_RIXSmagSq_Q0',
           'diag', 'diag_evals_diff', 'diag_n_dipole', 'calc_Q',
           'get_tss', 'get_tsp', 'get_tsd', 'get_tpp', 'get_tpd', 'get_tdd',
           'tmat_t2o', 'tmat_o2t', 'rmat_t2o', 'rmat_o2t']

class SparseBinary:
    """
    Binary sparse matrix used for Hatree Fock basis.
    All the elements are zero unless specified.
    
    Attributes
    ----------
    get_shape          :  Output the matrix shape.
    get_coordinates    :  Output the coordinates of nonzero elements.
    choose             :  Choose some rows of the matrix.
    change_coordinates :  Change coordinates of the nonzero elements.
    get_row            :  Output the row of matrix as conventional array.
    toarray            :  Output the matrix as conventional array.
    save               :  Save the matrix.
    """
    
    def __init__(self, coordinates, shape=None):
        """
        Class initialization.

        Parameters
        ----------
        coordinates : list of int, coordinates of non-zero elements.
              shape : 2-tuple of int, shape of the matrix (default: None).
        
        Notes
        -----
        [1] dtype is fixed as int.
        [2] One of the elements of shape can be given as None.
                
        Example
        -------
        >>> basis = SparseBinary([[3,5],[1,2]], shape=(None, 7))
        >>> basis.toarray()
        >>> array([[0, 0, 0, 1, 0, 1, 0],
                   [0, 1, 1, 0, 0, 0, 0]], dtype=int)
        """
        self.coordinates = np.array(coordinates).astype('int')
        
        # Determine the shape of the matrix
        default_shape = [self.coordinates.shape[0], np.max(self.coordinates)]
        if shape is not None:
            for shape_i in range(2):
                if shape[shape_i] is not None:
                    default_shape[shape_i] = shape[shape_i]
        self.shape = default_shape
    
    def get_shape(self):
        """
        Output the matrix shape.
        
        Example
        -------
        >>> basis.get_shape()
        
        Returns
        -------
        shape : 2-element list, shape of the matrix.
        """
        return self.shape
    
    def get_coordinates(self):
        """
        Output the coordinates of nonzero elements.
        
        Example
        -------
        >>> basis.get_coordinates()
        
        Returns
        -------
        coordinates : 2d int array, coordinates of the nonzero elements.
        """
        return self.coordinates
    
    def choose(self, choose):
        """
        Choose some rows of the matrix.
        
        Example
        -------
        >>> coordinates = basis.get_coordinates()
        >>> choose = [np.sum(data%2==0)==2 for data in coordinates] # spin-up
        >>> basis.choose(choose)
        
        Returns
        -------
        No return. The cooridnates and shape will be updated.
        """
        self.coordinates = self.coordinates[choose, :]
        self.shape[0] = self.coordinates.shape[0]
    
    def change_coordinates(self, coo_table, shape=None):
        """
        Change coordinates of the nonzero elements.
        
        Parameters
        ----------
        coo_table : list of int, coordinate table.
        shape : 2-tuple of int, shape of the matrix (default: None).
        
        Example
        -------
        >>> coo_table = [2,5,6]
        >>> basis.toarray()
        >>> array([[1,0,1],[1,1,0]])
        >>> basis.change_coordinates(coo_table, shape=(None, 7))
        >>> basis.toarray()
        >>> array([[0,0,1,0,0,0,1],[0,0,1,0,0,1,0]])
        
        Returns
        -------
        No return. The cooridnates and shape will be updated.
        """
        self.coordinates = np.vstack([coo_table[row] for row in self.coordinates])
        
        # Redetermine the shape of the matrix
        default_shape = [self.coordinates.shape[0], np.max(self.coordinates)]
        if shape is not None:
            for shape_i in range(2):
                if shape[shape_i] is not None:
                    default_shape[shape_i] = shape[shape_i]
        self.shape = default_shape
    
    def get_row(self, nrow):
        """
        Output the row of matrix as conventional array.
        
        Parameters
        ----------
        row : int, index of row.
        
        Example
        -------
        >>> basis.toarray()
        >>> array([[1,0,1],[1,1,0]])
        >>> basis.get_row(1)
        >>> array([1,1,0])
        
        Returns
        -------
        row : 1d array
        """
        mat_output = np.zeros(self.shape[1], dtype=int)
        for ncol in self.coordinates[nrow]:
            mat_output[ncol] = 1
        return mat_output
    
    def toarray(self):
        """
        Output the matrix as conventional array.
        
        Example
        -------
        >>> basis.toarray()
        >>> array([[1,0,1],[1,1,0]])
        
        Returns
        -------
        array : 2d array
        """
        mat_output = np.zeros(self.shape, dtype=int)
        for nrow in range(self.shape[0]):
            for ncol in self.coordinates[nrow]:
                mat_output[nrow, ncol] = 1
        return mat_output
    
    def save(self, filename):
        """
        Save the matrix.
        
        Parameters
        ----------
        filename : string, saved filename.
        
        Example
        -------
        >>> basis.save('basis_i.npz')
        
        Returns
        -------
        No return. Data saved as *.npz with keys of coordinates and shape.
        """
        np.savez(filename, coordinates=self.coordinates, shape=self.shape)

class Sparse4D:
    """
    Four dimentional sparse matrix.
    
    Attributes
    ----------
    get_shape       :  Output the matrix shape.
    get_coordinates :  Output the coordinates of nonzero elements.
    get_elements    : Output the nonzero elements.
    toarray         :  Output the matrix as conventional array.
    save            :  Save the matrix.
    """
    
    def __init__(self, values, coordinates, shape=None, dtype=float):
        """
        Class initialization.

        Parameters
        ----------
               data : 1d array, non-zero elements.
        coordinates : 2d 4 x N int array, coordinates of non-zero elements.
              shape : 4-tuple of int, shape of the matrix.
              dtype : dtype, data type of the matrix.
        
        Notes
        ----------
        [1] One or several of the elements of shape can be given as None.
            
        Example
        ----------
        >>> umat = Sparse4D([0.2,0.1], [[0,1],[0,1],[0,1],[0,1]], shape=(None, 7, 7, 7))
        """
        self.coordinates = np.array(coordinates).astype('int')
        self.elements = np.array(values).astype(dtype)
        self.dtype = dtype
        
        # Determine the shape of the matrix
        default_shape = [np.max(self.coordinates[ncoo,:])+1 for ncoo in range(4)]
        if shape is not None:
            for shape_i in range(4):
                if shape[shape_i] is not None:
                    default_shape[shape_i] = shape[shape_i]
        self.shape = default_shape
    
    def get_shape(self):
        """
        Output the matrix shape.
        
        Example
        -------
        >>> umat.get_shape()
        
        Returns
        -------
        shape : 4-element list, shape of the matrix.
        """
        return self.shape
    
    def get_coordinates(self):
        """
        Output the coordinates of nonzero elements.
        
        Example
        -------
        >>> umat.get_coordinates()
        
        Returns
        -------
        coordinates : 2d 4 x N int array, coordinates of the nonzero elements.
        """
        return self.coordinates
    
    def get_elements(self):
        """
        Output the nonzero elements.
        
        Example
        -------
        >>> umat.get_elements()
        
        Returns
        -------
        elements : 1d array.
        """
        return self.elements
    
    def toarray(self):
        """
        Output the matrix as conventional array.
        
        Example
        -------
        >>> umat.toarray()
        >>> array([[1,0,1],[1,1,0],[0,0,1],[1,1,1]])
        
        Returns
        -------
        array : 4d array.
        """
        mat_output = np.zeros(self.shape, dtype=self.dtype)
        for element, coordinates in zip(self.elements, self.coordinates.T):
            mat_output[coordinates[0], coordinates[1], coordinates[2], coordinates[3]] = element
        return mat_output
    
    def save(self, filename):
        """
        Save the matrix.
        
        Parameters
        ----------
        filename : string, saved filename.
        
        Example
        -------
        >>> umat.save('umat_i.npz')
        
        Returns
        -------
        No return. Data saved as *.npz with keys of elements, coordinates, shape and dtype.
        """
        np.savez(filename, elements=self.elements, coordinates=self.coordinates, shape=self.shape, dtype=self.dtype)

def fock_bin_S(*args):
    """
    Get binary form to represent a Fock state.
    
    Parameters
    ----------
    args: ints
        args[0]: number of orbitals for 1st-shell,
        args[1]: number of occupancy for 1st-shell,
        args[2]: number of orbitals for 2nd-shell,
        args[3]: number of occupancy for 2nd-shell,
        ...
        args[ :math:`2N-2`]: number of orbitals for :math:`N` th-shell,
        args[ :math:`2N-1`]: number of occupancy for :math:`N` th-shell.
    
    Returns
    -------
    result: Fock states in form of list of non-zero site index.
    """
    n = len(args)

    if n % 2 != 0:
        print("Error: number of arguments is not even")
        return

    if n == 2:
        state = itertools.combinations(np.arange(args[0]), args[1])
        state = np.vstack([list(state0) for state0 in state]) # turn it into an array
        return state, args[0] # return basis and the length of sites
    else:
        result = []
        state1, nsite1 = fock_bin_S(args[0], args[1])
        state2, nsite2 = fock_bin_S(*args[2:])
        nstate1 = state1.shape[0]
        nstate2 = state2.shape[0]
        state1 = np.repeat(state1, nstate2, axis=0)
        state2 = np.tile(state2+nsite1, (nstate1, 1))
        return np.hstack([state1, state2]), nsite1+nsite2
    
def get_fock_bin_by_N_S(*args, dispinfo=True):
    """
    Get binary form to represent a Fock state.
    
    Parameters
    ----------
    args: ints
        args[0]: number of orbitals for 1st-shell,
        args[1]: number of occupancy for 1st-shell,
        args[2]: number of orbitals for 2nd-shell,
        args[3]: number of occupancy for 2nd-shell,
        ...
        args[ :math:`2N-2`]: number of orbitals for :math:`N` th-shell,
        args[ :math:`2N-1`]: number of occupancy for :math:`N` th-shell.
    
    Returns
    -------
    result: Fock states in form of fake coordinate matrix listing the orbital index with non-zero elements.
    """
    if dispinfo:
        print('----> Constructing the Fock state')
    state, nsite = fock_bin_S(*args) # Fock state in form of list of non-zero site index
    state = np.vstack([np.sort(state0) for state0 in state])
    if dispinfo:
        print('----> {} States constructed with {} orbitals.'.format(state.shape[0], nsite))
    return state.astype('int32')

def two_fermion_S(emat, lb, rb=None, dtype=float):
    
    """
    Build matrix form of a two-fermionic operator in the given Fock basis,
    .. math::
        <F_{l}|\\sum_{ij}E_{ij}\\hat{f}_{i}^{\\dagger}\\hat{f}_{j}|F_{r}>
    Rewritten from edrixs.two_fermion to adapt sparse matrix and improve efficiency
    
    Parameters
    ----------
    emat : scipy coordinate matrix, the impurity matrix.
      lb : SparseBinary matrix, left fock basis :math:`<F_{l}|`.
      rb : SparseBinary matrix, right fock basis :math:`|F_{r}>`. rb = lb if rb is None (default: None).
    
    Returns
    -------
    hmat: scipy coordinate matrix, the matrix form of the two-fermionic operator.
    """
    if rb is None:
        rb = lb
    nr, nl = rb.get_shape()[0], lb.get_shape()[0]

    hmat_data, hmat_rows, hmat_cols = [], [], []
    emat_data, emat_rows, emat_cols = emat.data, emat.row, emat.col

    indx = defaultdict(lambda: -1)
    for i, tmp_basis in enumerate(lb.get_coordinates()):
        indx[tuple(np.sort(tmp_basis))] = i
    
    for icfg, tmp_basis in enumerate(rb.get_coordinates()):
        tmp = np.vstack([emat_rows, emat_cols, emat_data, np.tile(tmp_basis, (len(emat_data),1)).T]).T
        for iorb in range(tmp_basis.size):
            choose = tmp[:,1]==tmp_basis[iorb]
            tmp_hmat = tmp[choose,:]
            s1 = [(-1)**np.sum(nn[3:]<nn[1]) for nn in tmp_hmat]
            tmp_hmat[:,3+iorb] = np.inf
            for ntmp, tmp_hmat0 in enumerate(tmp_hmat):
                if ~np.isin(tmp_hmat0[0], tmp_hmat0[3:]):
                    s2 = (-1)**np.sum(tmp_hmat0[3:]<tmp_hmat0[0])
                    new_basis = tmp_hmat0[3:]
                    new_basis[iorb] = tmp_hmat0[0]
                    jcfg = indx[tuple(np.sort(new_basis))]
                    if jcfg != -1:
                        hmat_rows.append(jcfg)
                        hmat_cols.append(icfg)
                        hmat_data.append(tmp_hmat0[2] * s1[ntmp] * s2)
    hmat = sparse.coo_matrix((hmat_data, (hmat_rows, hmat_cols)), shape=(nl, nr), dtype=dtype)
    hmat.sum_duplicates()
    return hmat

def four_fermion_S(umat, lb, rb=None, dtype=float):
    """
    Build matrix form of a four-fermionic operator in the given Fock basis,
    .. math::
        <F_l|\\sum_{ij}U_{ijkl}\\hat{f}_{i}^{\\dagger}\\hat{f}_{j}^{\\dagger}
        \\hat{f}_{k}\\hat{f}_{l}|F_r>
    Rewritten from edrixs.four_fermion to adapt sparse matrix and improve efficiency.
    
    Parameters
    ----------
    umat : Sparse4D, the 4 index Coulomb interaction tensor.
      lb : SparseBinary matrix, left fock basis :math:`<F_{l}|`.
      rb : SparseBinary matrix, right fock basis :math:`|F_{r}>`. rb = lb if rb is None (default: None).
        
    Returns
    -------
    hmat: scipy coordinate matrix, the matrix form of the four-fermionic operator.
    """
    
    if rb is None:
        rb = lb
    nr, nl, norbs = rb.get_shape()[0], lb.get_shape()[0], rb.get_shape()[1]
    
    hmat_data, hmat_rows, hmat_cols = [], [], []
    
    indx = defaultdict(lambda: -1)
    for i, tmp_basis in enumerate(lb.get_coordinates()):
        indx[tuple(np.sort(tmp_basis))] = i
    
    if rb.get_coordinates().shape[1]<2:
        return sparse.coo_matrix(([], ([], [])), shape=(nl, nr), dtype=dtype)
    
    umat_indx = umat.get_coordinates()
    umat_vals = umat.get_elements()
    choose = np.logical_and(abs(umat_indx[3]-umat_indx[2])>0, abs(umat_indx[1]-umat_indx[0])>0)
    lorbs, korbs = umat_indx[0][choose], umat_indx[1][choose]
    jorbs, iorbs = umat_indx[2][choose], umat_indx[3][choose]
    data = umat_vals[choose]
    
    for icfg, tmp_basis in enumerate(rb.get_coordinates()):
        tmp_hmat = np.vstack([lorbs, korbs, jorbs, iorbs, data, np.tile(tmp_basis, (len(lorbs),1)).T]).T
        for iorb, jorb in itertools.permutations(np.arange(len(tmp_basis)), 2):
            choose = np.logical_and(tmp_hmat[:,3]==tmp_basis[iorb], tmp_hmat[:,2]==tmp_basis[jorb])
            tmp_hmat0 = tmp_hmat[choose,:]
            s1 = [(-1)**np.sum(tmp[5:]<tmp[3]) for tmp in tmp_hmat0]
            tmp_hmat0[:,5+iorb] = np.inf
            s2 = [(-1)**np.sum(tmp[5:]<tmp[2]) for tmp in tmp_hmat0]
            tmp_hmat0[:,5+jorb] = np.inf
            for ntmp, tmp_hmat00 in enumerate(tmp_hmat0):
                if ~np.isin(tmp_hmat00[1], tmp_hmat00[5:]) and ~np.isin(tmp_hmat00[0], tmp_hmat00[5:]):
                    s3 = (-1)**np.sum(tmp_hmat00[5:]<tmp_hmat00[1])
                    new_basis = tmp_hmat00[5:] + 0.
                    new_basis[new_basis==np.inf] = tmp_hmat00[:2]
                    s4 = (-1)**np.sum(new_basis<tmp_hmat00[0])
                    jcfg = indx[tuple(np.sort(new_basis.real.astype(int)))]
                    if jcfg != -1:
                        hmat_rows.append(jcfg)
                        hmat_cols.append(icfg)
                        hmat_data.append(tmp_hmat00[4] * s1[ntmp] * s2[ntmp] * s3 * s4)
    hmat = sparse.coo_matrix((hmat_data, (hmat_rows, hmat_cols)), shape=(nl, nr), dtype=dtype)
    hmat.sum_duplicates()
    return hmat

def diag(hmat, emax=None, emin=None, k=None, break_point=False, tol=1e-5, suffix='_i', folder=''):
    """
    Diagonalize the Hamiltonian.
    evals{suffix}.npy and evecs{suffix}.npy will be saved in the cluster folder.

    Parameters
    ----------
           hmat : 2d array, Hamiltonian to diagonalize.
              k : int, mMaximum cycle number to run in sparse diagonalization.
                  If None, scipy.linalg.eigh will be called (default: None).
           emax : float, the maximum energy difference w.r.t. the minimum energ to cover using sparse approach.
                  If None, all the eigen states will be calculated using sparse approach (default: None).
           emin : float, the minimum eigen value to start from using sparse approach.
                  If None, it will start from the minimum value of all the eigen values (default: None).
    break_point : bool, whether start from the break point.
                  If True, the evals{suffix}_i.npy and evecs{suffix}.npy will be loaded and continue the diagonalization (default: False).
            tol : float, tolerance for sparse approach (default: 1e-5).
         folder : string, folder to save the results (default: '').
    """
    if k is None: # solve all the eigen states
        print('----> Diagonalizing Hamiltonian...')
        evals_all, evecs_all = eigh(hmat.toarray())
        evals_all = evals_all.real
        np.save(folder+f'/evals{suffix}.npy', evals_all)
        np.save(folder+f'/evecs{suffix}.npy', evecs_all)
        print('----> {} states found between {:.2f} and {:.2f} eV'.format(evals_all.size, np.min(evals_all), np.max(evals_all)))
    elif break_point is False:
        print('----> Diagonalizing Hamiltonian...')
        if emax is None: # solve k eigen states
            if emin is None:
                evals_all, evecs_all = eigs(hmat, k=k, which='SR', tol=tol)
                evals_all = evals_all.real
            else:
                evals_all, evecs_all = eigs(hmat, k=k, sigma=emin, which='LR', tol=tol)
                evals_all = evals_all.real
            np.save(folder+f'/evals{suffix}.npy', evals_all)
            np.save(folder+f'/evecs{suffix}.npy', evecs_all)
            print('----> {} states found between {:.2f} and {:.2f} eV'.format(evals_all.size, np.min(evals_all), np.max(evals_all)))
        else: # solve eigen states up to emax
            if emin is None:
                evals_all, evecs_all = eigs(hmat, k=k, which='SR', tol=tol)
                evals_all = evals_all.real
            else:
                evals_all, evecs_all = eigs(hmat, k=k, sigma=emin, which='LR', tol=tol)
                evals_all = evals_all.real
            np.save(folder+f'/evals{suffix}.npy', evals_all)
            np.save(folder+f'/evecs{suffix}.npy', evecs_all)
            if evals_all[-1]-evals_all[0]>=emax: # all the required eigen states found
                print('----> {} states found between {:.2f} and {:.2f} eV'.format(evals_all.size, np.min(evals_all), np.max(evals_all)))
            else: # continue to find more eigen states
                print('    ----> {} states found between {:.2f} and {:.2f} eV'.format(evals_all.size, np.min(evals_all), np.max(evals_all)))
                diag(hmat, emax=emax, emin=None, k=k, break_point=True, tol=tol, suffix=suffix, folder=folder)
    else: # resume from diagonalization and find k new eigen states
        evals_raw = np.load(folder+f'/evals{suffix}.npy')
        evecs_raw = np.load(folder+f'/evecs{suffix}.npy')
        evals, evecs = eigs(hmat, k=k, sigma=np.max(evals_raw), which='LR', v0=evecs_raw[:,np.argmax(evals_raw)], tol=tol)
        evals = evals.real
        evals_all = np.hstack([evals_raw, evals])
        evecs_all = np.hstack([evecs_raw, evecs])
        np.save(folder+f'/evals{suffix}.npy', evals_all)
        np.save(folder+f'/evecs{suffix}.npy', evecs_all)
        if emax is None: # find k eigen states
            print('----> {} states found between {:.2f} and {:.2f} eV'.format(evals_all.size, np.min(evals_all), np.max(evals_all)))
        elif evals_all[-1]-evals_all[0]>=emax: # all the required eigen states found
            print('----> {} states found between {:.2f} and {:.2f} eV'.format(evals_all.size, np.min(evals_all), np.max(evals_all)))
        else: # continue to find more eigen states
            print('    ----> {} states found between {:.2f} and {:.2f} eV'.format(evals.size, np.min(evals), np.max(evals)))
            diag(hmat, emax=emax, emin=None, k=k, break_point=True, tol=tol, suffix=suffix, folder=folder)

def diag_evals_diff(mat):
    """
    Diagonalize the matrix in full, return the difference between the largest and smallest eigenvalues

    Parameters
    ----------
           mat : 2d array, non-sparse matrix to diagonalize.
    """
#     print('----> Diagonalizing the matrix...')
    evals_all = eigh(mat, eigvals_only=True)
    evals_all = evals_all.real
#     print('----> Diagonalization done')
    return(np.max(evals_all)-np.min(evals_all))
#     evals_min = eigsh(mat, 1, which='LA', return_eigenvectors=False)
#     evals_max = eigsh(mat, 1, which='SA', return_eigenvectors=False)
#     return(evals_max[0] - evals_min[0])
    
            
def construct_Tabs(evecs_i, evecs_n, dipole):
    """
    Construct the absorption transition operators.
    
    Parameters
    ----------
    evecs_i : 2d array, eigen vectors of initial states.
    evecs_n : 2d array, eigen vectors of intermediate states.
     dipole : list of sparse matrix, dipolar transition operator.
    
    Returns
    -------
    Tabs : list of 2d array, absorption operator.
    """
    T_abs = []
    for xyz in range(3):
        T_abs.append(dipole[xyz].transpose().dot(np.conj(evecs_n)).transpose().dot(evecs_i))
    return np.stack(T_abs)

def diag_n_dipole(hmat_n, emax=None, emin=None, k=None, break_point=False, tol=1e-5, folder=''):
    """
    Diagonalize the Hamiltonian with core hole but keep T_abs and delete evecs_n.
    evals_n.npy and T_abs.npy will be saved in the cluster folder.
    evecs_n.npy only save the latest eigen vector.
    evecs_i.npy and dipole.npz will be loaded from the folder.
    dipole.npz have keys of row1, row2, row3, col1, col2, col3, data1, data2, data3, shape

    Parameters
    ----------
         hmat_n : 2d array, Hamiltonian with core hole.
              k : int, maximum cycle number to run in sparse diagonalization.
                  If None, scipy.linalg.eigh will be called (default: None).
           emax : float, the maximum energy difference w.r.t. the minimum energ to cover using sparse approach.
                  If None, all the eigen states will be calculated using sparse approach (default: None).
           emin : float, the minimum eigen value to start from using sparse approach.
                  If None, it will start from the minimum value of all the eigen values (default: None).
    break_point : bool, whether start from the break point.
                  If True, the evals{suffix}_i.npy and evecs{suffix}.npy will be loaded and continue the diagonalization (default: False).
            tol : float, tolerance for sparse approach (default: 1e-5).
         folder : string, folder to save the results (default: '').
    """
    # Load evecs_i
    evecs_i = np.load(folder+'/evecs_i.npy')
    # Load dipole
    dipole_raw = dict(np.load(folder+'/dipole.npz'))
    dipole = []
    for xyz in range(3):
        dipole_tmp = sparse.coo_matrix((dipole_raw['data{}'.format(xyz)],
                                       (dipole_raw['row{}'.format(xyz)], dipole_raw['col{}'.format(xyz)])),
                                       shape=dipole_raw['shape'], dtype=complex)
        dipole.append(dipole_tmp)
    
    if k is None: # solve all the eigen states
        print('----> Diagonalizing Hamiltonian...')
        evals_n, evecs_n = eigh(hmat_n.toarray())
        evals_n = evals_n.real
        T_abs = construct_Tabs(evecs_i, evecs_n, dipole)
        np.save(folder+'/evals_n.npy', evals_n)
        np.save(folder+'/evecs_n.npy', evecs_n[:,-1])
        np.save(folder+'/T_abs.npy', T_abs)
        print('----> {} states found between {:.2f} and {:.2f} eV'.format(evals_n.size, np.min(evals_n), np.max(evals_n)))
    
    elif break_point is False:
        print('----> Diagonalizing Hamiltonian...')
        if emax is None: # solve k eigen states
            if emin is None:
                evals_n, evecs_n = eigs(hmat_n, k=k, which='SR', tol=tol)
                evals_n = evals_n.real
            else:
                evals_n, evecs_n = eigs(hmat_n, k=k, sigma=emin, which='LR', tol=tol)
                evals_n = evals_n.real
            T_abs = construct_Tabs(evecs_i, evecs_n, dipole)
            np.save(folder+'/evals_n.npy', evals_n)
            np.save(folder+'/evecs_n.npy', evecs_n[:,-1])
            np.save(folder+'/T_abs.npy', T_abs)
            print('----> {} states found between {:.2f} and {:.2f} eV'.format(evals_n.size, np.min(evals_n), np.max(evals_n)))
        else: # solve eigen states up to emax
            if emin is None:
                evals_n, evecs_n = eigs(hmat_n, k=k, which='SR', tol=tol)
                evals_n = evals_n.real
            else:
                evals_n, evecs_n = eigs(hmat_n, k=k, sigma=emin, which='LR', tol=tol)
                evals_n = evals_n.real
            T_abs = construct_Tabs(evecs_i, evecs_n, dipole)
            np.save(folder+'/evals_n.npy', evals_n)
            np.save(folder+'/evecs_n.npy', evecs_n[:,-1])
            np.save(folder+'/T_abs.npy', T_abs)
            if evals_n[-1]-evals_n[0]>=emax: # all the required eigen states found
                print('----> {} states found between {:.2f} and {:.2f} eV'.format(evals_n.size, np.min(evals_n), np.max(evals_n)))
            else: # continue to find more eigen states
                print('    ----> {} states found between {:.2f} and {:.2f} eV'.format(evals_n.size, np.min(evals_n), np.max(evals_n)))
                diag_n_dipole(hmat_n, emax=emax, emin=None, k=k, break_point=True, tol=tol, folder=folder)
    else: # resume from diagonalization and find k new eigen states
        evals_n_raw = np.load(folder+'/evals_n.npy')
        evecs_n_raw = np.load(folder+'/evecs_n.npy')
        T_abs_raw = np.load(folder+'/T_abs.npy')
        evals, evecs = eigs(hmat_n, k=k, sigma=np.max(evals_n_raw), which='LR', v0=evecs_n_raw, tol=tol)
        evals = evals.real
        evals_n = np.hstack([evals_n_raw, evals])
        T_abs_tmp = construct_Tabs(evecs_i, evecs, dipole)
        T_abs = np.concatenate([T_abs_raw, T_abs_tmp], axis=1)
        np.save(folder+'/evals_n.npy', evals_n)
        np.save(folder+'/evecs_n.npy', evecs[:,-1])
        np.save(folder+'/T_abs.npy', T_abs)
        if emax is None: # find k eigen states
            print('----> {} states found between {:.2f} and {:.2f} eV'.format(evals_n.size, np.min(evals_n), np.max(evals_n)))
        elif evals_n[-1]-evals_n[0]>=emax: # all the required eigen states found
            print('----> {} states found between {:.2f} and {:.2f} eV'.format(evals_n.size, np.min(evals_n), np.max(evals_n)))
        else: # continue to find more eigen states
            print('    ----> {} states found between {:.2f} and {:.2f} eV'.format(evals.size, np.min(evals), np.max(evals)))
            diag_n_dipole(hmat_n, emax=emax, emin=None, k=k, break_point=True, tol=tol, folder=folder)

def calc_Q(k=0.441, thin=15., tth=150., phi=0.):
    """
    Calculate Q in unit of \AA^{-1}.
    
    Parameters
    ----------
    k : float,
        Wavevector of x-ray in unit of \AA^{-1}.
    thin : 1d array or float,
        Incident angle w.r.t. sample surface in unit of degree.
    tth : float,
        Scattering angle in unit of degree.
    phi : float,
        Azimuthal angle in unit of degree.
    
    Returns
    -------
    Q : 2d array.
    """
    if isinstance(thin, list):
        thin = np.array(thin)
    elif not isinstance(thin, np.ndarray):
        thin = np.array([thin])
    kin  = np.vstack([k * np.array([np.cos(np.deg2rad(th)),0,-np.sin(np.deg2rad(th))]) for th in thin])
    kout = np.vstack([k * np.array([-np.cos(np.deg2rad(th+180.-tth)),0,np.sin(np.deg2rad(th+180.-tth))]) for th in thin])
    tmat = np.vstack([[np.cos(np.deg2rad(phi)),np.sin(np.deg2rad(phi)),0.],[np.sin(np.deg2rad(phi)),np.cos(np.deg2rad(phi)),0.],[0.,0.,1.]])
    Q = np.dot(kout - kin, tmat)
    return Q

def calc_RIXS(evals_i, evals_n, T_abs_all,
              Q=None, pos=None,
              Gam_c=0.6, sigma=0.1, fraction=0.5,
              omega=None, eloss=None,
              phi=0., thin=15., tth=150., pol=[(0, 0), (0, np.pi / 2.0)], scatter_axis=None,
              T=10., tol=1e-5, ngs=None):
    """
    Calculate the RIXS spectra for multi-site cluster.

    Parameters
    ----------
         evals_i : 1d array, eigen values for the initial states.
         evals_n : list of 1d array, list of eigen values for the intermediate states.
       T_abs_all : list of 2d array, list of absorption matrix.
               Q : list of 3-element array/list, Q vectors in unit of \AA^{-1}.
                   It can either have the same size of thin or just one Q (default: None).
             pos : list of 3-element array/list, positions of the sites.
                   If None, assume a chain along a direction (default: None).
           Gam_c : float, inverse core-hole lifetime in unit of eV (default: 0.6).
           sigma : float, sigma of PseudoVoigt profile (default: 0.1).
        fraction : float, fraction of PseudoVoigt profile (default: 0.5).
           omega : 1d array or float, incident energy. If None, determined by evals (default: None).
           eloss : 1d array, energy loss. If None, determined by evals (default: None).
             phi : float, azimuthal angle in unit of degree (default: 0).
            thin : 1d array or float, incident angle w.r.t. sample surface in unit of degree (default: 15).
             tth : float, scattering angle in unit of degree (default: 150).
    scatter_axis : 3x3 array, The local axis defining the scattering geometry. The scattering plane is defined in the local xz-plane.
                   It will be set to an identity matrix if None (default: None).
             pol : list of 2-element turple, x-ray polarization, (incident, emission), w.r.t the scattering plane.
              ei : 3-element list/array, incident polarization in sample coordinates.
                   If given, it will overwrite thin, tth, pol etc. (default: None).
              ef : 3-element list/array, emission polarization in sample coordinates.
                   If given, it will overwrite thin, tth, pol etc. (default: None).
               T : float, temperature in unit of Kelvin (default: 10).
             tol : float, tolerance of bose factor to include the initial states (default: 1e-3).
             ngs : int, number of initial states. If None it will be determined based on tol (default: None).

    Returns
    -------
    rixs[pol,omega,thin,eloss] : 4d array.

    Example
    -------
    >>> # rixs with sigma pol.
    >>> rixs = calc_RIXS(evals_i, evals_n, T_abs,
                         Q=None, pos=None,
                         Gam_c=0.6, sigma=0.1, fraction=0.5,
                         omega=np.linspace(-2.9, 8.9, 100), eloss=np.linspace(-0.5, 10., 1000),
                         phi=0., thin=15., tth=150., pol=[(np.pi/2., 0), (np.pi/2., np.pi/2.)])
    """
    # Determine the ground state occupation
    GS_energy = np.min(evals_i) ## ground state energy
    beta = 1.0 / 8.6173303E-5 / T
    if ngs is None:
        choose = np.exp(-beta * (evals_i - GS_energy)) >= tol
        gs = np.arange(len(evals_i))[choose]
    else:
        gs = np.arange(ngs)
    prob = np.exp(-beta * (evals_i[gs] - GS_energy)) / np.sum(np.exp(-beta * (evals_i[gs] - GS_energy)))
    
    # Determine omega
    if omega is None:
        omega_min = np.min(evals_n[0]) - GS_energy - 10.
        omega_max = np.max(evals_n[0]) - GS_energy + 10.
        omega_step = Gam_c / 10
        omega = np.arange(omega_min, omega_max, omega_step)
    
    # Determine eloss
    if eloss is None:
        eloss_min = -1.
        eloss_max = np.max(evals_i) - GS_energy + 1.
        eloss_step = sigma / 10
        eloss = np.arange(eloss_min, eloss_max, eloss_step)
    
    # Determine scattering axis
    if scatter_axis is None:
        scatter_axis = np.diag([1.]*3)
    
    # Reformulate the params
    if isinstance(omega, list):
        omega = np.array(omega)
    elif not isinstance(omega, np.ndarray):
        omega = np.array([omega])
    if isinstance(thin, list):
        thin = np.array(thin)
    elif not isinstance(thin, np.ndarray):
        thin = np.array([thin])
    if not isinstance(pol, list):
        pol = [pol]
    if pos is None:
        pos = [[0., 0., 0.] for _ in range(len(evals_n))]
    if Q is None:
        Q = [[0., 0., 0.] for _ in range(len(thin))]
    elif len(np.array(Q).shape)==1:
        Q = [Q] * len(thin)
    sigma_g = sigma / np.sqrt(2.*np.log(2))
    # nTM is the number of sites, TM stands for transition metal
    nTM = len(evals_n)
    
    # Calculate RIXS
    F_fis = np.zeros((nTM, len(omega), 3, 3, len(evals_i), len(gs)), dtype=complex)
    for nomega, om in enumerate(omega):
        # axis-0 labels which site possessing the core-hole
        for nnTM, (evals, T_abs) in enumerate(zip(evals_n, T_abs_all)):
            T_emi = np.stack([np.conj(np.transpose(T0)) for T0 in T_abs])
            F_fis[nnTM,nomega,:,:,:,:] += scattering_mat(evals_i[:], evals[:], T_abs[:, :, gs], T_emi, om, Gam_c)
    
    # Initialize
    rixs = np.zeros((len(pol), len(omega), len(thin), len(eloss)), dtype=float)
    for nomega, om in enumerate(omega):
        for npol, (alpha, beta) in enumerate(pol):
            for nth, th in enumerate(thin):
                ei, ef = dipole_polvec_rixs(th/180.*np.pi, (-th+tth)/180.*np.pi, phi/180.*np.pi, alpha, beta)
                ei = np.dot(scatter_axis, ei)
                ef = np.dot(scatter_axis, ef)
                F_mag = np.zeros((len(evals_i), len(gs)), dtype=complex)
                for nnTM in range(nTM):
                    phase = np.exp(-1j * np.dot(Q[nth], pos[nnTM]))
                    for m in range(3):
                        for n in range(3):
                            F_mag[:, :] += ef[m] * F_fis[nnTM, nomega, m, n] * phase * ei[n]
                for nngs, gs_id in enumerate(gs):
                    for n in range(len(evals_i)):
                        Gaussian = (1 - fraction) / sigma_g / np.sqrt(2. * np.pi) * np.exp(-(eloss - (evals_i[n] - evals_i[gs_id]))**2 / 2. / sigma_g**2)
                        Lorentzian = fraction / np.pi * sigma / ((eloss - (evals_i[n] - evals_i[gs_id]))**2 + sigma**2)
                        rixs[npol,nomega,nth,:] += prob[nngs] * np.abs(F_mag[n, nngs])**2 * (Gaussian + Lorentzian)
    
    return rixs

def calc_RIXSmagSq_Q0(evals_i, evals_n, T_abs,
                      Gam_c=0.6, omega=None,
                      phi=0., thin=15., tth=150., pol=[(0, 0), (0, np.pi / 2.0)], scatter_axis=None,
                      ei=None, ef=None,
                      T=10., tol=1e-5, ngs=None):
    """
    Calculate the F_magnitude**2 for Q=0 case.
    
    Parameters
    ----------
         evals_i : 1d array, eigen values for the initial states.
         evals_n : 1d array, eigen values for the intermediate states.
           T_abs : 2d array, the absorption matrix.
           Gam_c : float, inverse core-hole lifetime in unit of eV (default: 0.6).
           omega : 1d array or float, incident energy. If None, determined by evals (default: None).
             phi : float, azimuthal angle in unit of degree (default: 0).
            thin : 1d array or float, incident angle w.r.t. sample surface in unit of degree (default: 15).
             tth : float, scattering angle in unit of degree (default: 150).
    scatter_axis : 3x3 array, The local axis defining the scattering geometry. The scattering plane is defined in the local xz-plane.
                   It will be set to an identity matrix if None (default: None).
             pol : list of 2-element turple, x-ray polarization, (incident, emission), w.r.t the scattering plane.
              ei : 3-element list/array, incident polarization in sample coordinates.
                   If given, it will overwrite thin, tth, pol etc. (default: None).
              ef : 3-element list/array, emission polarization in sample coordinates.
                   If given, it will overwrite thin, tth, pol etc. (default: None).
               T : float, temperature in unit of Kelvin (default: 10).
             tol : float, tolerance of bose factor to include the initial states (default: 1e-3).
             ngs : int, number of initial states. If None it will be determined based on tol (default: None).

    Returns
    -------
    if ei/ef not given, rixs[pol,omega,thin,ngs,evals] : 5d array.
    if ei/ef given, rixs[omega,ngs,evals] : 3d array.

    Example
    -------
    >>> rixs = calc_RIXSmagSq_Q0(evals_i, evals_n, T_abs,
                                 Gam_c=0.6, omega=np.linspace(-2.9, 8.9, 100),
                                 phi=0., thin=15., tth=150., pol=[(0, 0), (0, np.pi/2.)])
    """
     # Determine the ground state occupation
    GS_energy = np.min(evals_i) ## ground state energy
    beta = 1.0 / 8.6173303E-5 / T
    if ngs is None:
        choose = np.exp(-beta * (evals_i - GS_energy)) >= tol
        gs = np.arange(len(evals_i))[choose]
    else:
        gs = np.arange(ngs)
    prob = np.exp(-beta * (evals_i[gs] - GS_energy)) / np.sum(np.exp(-beta * (evals_i[gs] - GS_energy)))
    
    # Determine omega
    if omega is None:
        omega_min = np.min(evals_n) - GS_energy - 10.
        omega_max = np.max(evals_n) - GS_energy + 10.
        omega_step = Gam_c / 10
        omega = np.arange(omega_min, omega_max, omega_step)
    
    # Determine scattering axis
    if scatter_axis is None:
        scatter_axis = np.diag([1.]*3)
    
    # Reformulate the params
    if isinstance(omega, list):
        omega = np.array(omega)
    elif not isinstance(omega, np.ndarray):
        omega = np.array([omega])
    if isinstance(thin, list):
        thin = np.array(thin)
    elif not isinstance(thin, np.ndarray):
        thin = np.array([thin])
    if not isinstance(pol, list):
        pol = [pol]
    T_emi = np.stack([np.conj(np.transpose(Tabs_tmp)) for Tabs_tmp in T_abs])
    
    # Calculate RIXS
    if ei is None or ef is None: ## ei/ef not given
        allFmagSq = np.zeros((len(pol), len(omega), len(thin), len(gs), len(evals_i)), dtype=float)
        for nomega, om in enumerate(omega):
            F_fi = scattering_mat(evals_i, evals_n, T_abs[:, :, gs], T_emi, om, Gam_c)
            for npol, (alpha, beta) in enumerate(pol):
                for nth, th in enumerate(thin):
                    ei, ef = dipole_polvec_rixs(th/180.*np.pi, (-th+tth)/180.*np.pi, phi/180.*np.pi, alpha, beta)
                    ei = np.dot(scatter_axis, ei)
                    ef = np.dot(scatter_axis, ef)
                    F_mag = np.zeros((len(evals_i), len(gs)), dtype=complex)
                    for m in range(3):
                        for n in range(3):
                            F_mag[:, :] += ef[m] * F_fi[m, n] * ei[n]
                    for nngs, gs_id in enumerate(gs):
                        allFmagSq[npol,nomega,nth,nngs,:] = prob[nngs] * np.abs(F_mag[:,gs_id])**2
    else: ## ei/ef given
        ei = np.array(ei) / np.linalg.norm(ei)
        ef = np.array(ef) / np.linalg.norm(ef)
        allFmagSq = np.zeros((len(omega), len(gs), len(evals_i)), dtype=float)
        for nomega, om in enumerate(omega):
            F_fi = scattering_mat(evals_i, evals_n, T_abs[:, :, gs], T_emi, om, Gam_c)
            F_mag = np.zeros((len(evals_i),  len(gs)), dtype=complex)
            for m in range(3):
                for n in range(3):
                    F_mag[:,:] += ef[m] * F_fi[m,n,:,:] * ei[n]
            for nngs, gs_id in enumerate(gs):
                allFmagSq[nomega,nngs,:] = prob[nngs] * np.abs(F_mag[:,nngs])**2
    
    return allFmagSq

def get_tss():
    """
    Get hopping integral for s orbitals.
    
    Returns
    -------
    tss=1. since Ess==Vss_sigma
    """
    return 1.

def get_tsp(l, m, n):
    """
    Get hopping integral between s and p orbitals.
    
    Parameters
    ----------
    l, m, n : float, direction cosines along x, y and z directions.
    
    Returns
    -------
    tsp_sigma : 1x3 2d arrays, dtype=float.
    
    Notes
    -----
    p orbital orders: px, py, pz
    """
    tsp_sigma = np.zeros((1, 3), dtype=float)
    tsp_sigma[0,0] = l
    tsp_sigma[0,1] = m
    tsp_sigma[0,2] = n
    return tsp_sigma

def get_tsd(l, m, n):
    """
    Get hopping integral between s and d orbitals.
    
    Parameters
    ----------
    l, m, n : float, direction cosines along x, y and z directions.
    
    Returns
    -------
    tsd_sigma : 1x5 2d arrays, dtype=float.
    
    Notes
    -----
    d orbital orders: d3z2-r2, dxz, dyz, dx2-y2, dxy.
    """
    tsd_sigma = np.zeros((1, 5), dtype=float)
    tsd_sigma[0,0] = n**2 - 0.5 * (l**2 + m**2)
    tsd_sigma[0,1] = np.sqrt(3) * l * n
    tsd_sigma[0,2] = np.sqrt(3) * m * n
    tsd_sigma[0,3] = np.sqrt(3)/2. * (l**2 - m**2)
    tsd_sigma[0,4] = np.sqrt(3) * l * m
    return tsd_sigma

def get_tpp(l, m, n):
    """
    Get hopping integral for p orbitals.
    
    Parameters
    ----------
    l, m, n : float, direction cosines along x, y and z directions.
    
    Returns
    -------
    tpp_sigma, tpp_pi : 3x3 2d arrays, dtype=float.
    
    Notes
    -----
    p orbital orders: px, py, pz.
    Hopping integral is positive for hole, negative for electron.
    """
    tpp_sigma = np.zeros((3, 3), dtype=float)
    tpp_pi    = np.zeros((3, 3), dtype=float)
    tpp_sigma[0,0] = l**2
    tpp_pi[0,0]    = 1. - l**2
    tpp_sigma[0,1] = tpp_sigma[1,0] = l * m
    tpp_pi[0,1]    = tpp_pi[1,0]    = -1. * l * m
    tpp_sigma[0,2] = tpp_sigma[2,0] = l * n
    tpp_pi[0,2]    = tpp_pi[2,0]    = -1. * l * n
    tpp_sigma[1,1] = m**2
    tpp_pi[1,1]    = 1. - m**2
    tpp_sigma[1,2] = tpp_sigma[2,1] = m * n
    tpp_pi[1,2]    = tpp_pi[2,1]    = -1. * m * n
    tpp_sigma[2,2] = n**2
    tpp_pi[2,2]    = 1. - n**2
    return -1*tpp_sigma, -1*tpp_pi

def get_tpd(l, m, n):
    """
    Get hopping integral between p and d orbitals.
    
    Parameters
    ----------
    l, m, n : float, direction cosines along x, y and z directions.
    
    Returns
    -------
    tpd_sigma, tpd_pi : 3x5 2d arrays, dtype=float.
    
    Notes
    -----
    p orbital orders: px, py, pz.
    d orbital orders: d3z2-r2, dxz, dyz, dx2-y2, dxy.
    Hopping integral is positive for hole, negative for electron.
    """
    tpd_sigma = np.zeros((3, 5), dtype=float)
    tpd_pi    = np.zeros((3, 5), dtype=float)
    tpd_sigma[0,0] = l * (n**2 - 0.5 * (l**2 + m**2))
    tpd_pi[0,0]    = -1. * np.sqrt(3) * l * n**2
    tpd_sigma[0,1] = np.sqrt(3) * l**2 * n
    tpd_pi[0,1]    = n * (1. - 2. * l**2)
    tpd_sigma[0,2] = np.sqrt(3) * l * m * n
    tpd_pi[0,2]    = -2. * l * m * n
    tpd_sigma[0,3] = np.sqrt(3)/2. * l * (l**2 - m**2)
    tpd_pi[0,3]    = l * (1. - l**2 + m**2)
    tpd_sigma[0,4] = np.sqrt(3) * l**2 * m
    tpd_pi[0,4]    = m * (1. - 2. * l**2)
    tpd_sigma[1,0] = m * (n**2 - 0.5 * (l**2 + m**2))
    tpd_pi[1,0]    = -1. * np.sqrt(3) * m * n**2
    tpd_sigma[1,1] = np.sqrt(3) * l * m * n #
    tpd_pi[1,1]    = -2. * l * m * n #
    tpd_sigma[1,2] = np.sqrt(3) * m**2 * n #
    tpd_pi[1,2]    = n * (1. - 2. * m**2) #
    tpd_sigma[1,3] = np.sqrt(3)/2. * m * (l**2 - m**2)
    tpd_pi[1,3]    = -1. * m * (1. + l**2 - m**2)
    tpd_sigma[1,4] = np.sqrt(3) * m**2 * l #
    tpd_pi[1,4]    = l * (1. - 2. * m**2) #
    tpd_sigma[2,0] = n * (n**2 - 0.5 * (l**2 + m**2))
    tpd_pi[2,0]    = np.sqrt(3) * n * (l**2 + m**2)
    tpd_sigma[2,1] = np.sqrt(3) * n**2 * l #
    tpd_pi[2,1]    = l * (1. - 2. * n**2) #
    tpd_sigma[2,2] = np.sqrt(3) * n**2 * m #
    tpd_pi[2,2]    = m * (1. - 2. * n**2) #
    tpd_sigma[2,3] = np.sqrt(3)/2. * n * (l**2 - m**2)
    tpd_pi[2,3]    = -1. * n * (l**2 - m**2)
    tpd_sigma[2,4] = np.sqrt(3) * l * m * n #
    tpd_pi[2,4]    = -2. * l * m * n #
    return -1*tpd_sigma, -1*tpd_pi

def get_tdd(l, m, n):
    """
    Get hopping integral for d orbitals.
    
    Parameters
    ----------
    l, m, n : float, direction cosines along x, y and z directions.
    
    Returns
    -------
    tdd_sigma, tdd_pi, tdd_delta : 5x5 2d arrays, dtype=float.
    
    Notes
    -----
    d orbital orders: d3z2-r2, dxz, dyz, dx2-y2, dxy.
    Hopping integral is positive for hole, negative for electron.
    """
    tdd_sigma = np.zeros((5, 5), dtype=float)
    tdd_pi    = np.zeros((5, 5), dtype=float)
    tdd_delta = np.zeros((5, 5), dtype=float)
    tdd_sigma[0,0] = (n**2 - 0.5 * (l**2 + m**2))**2
    tdd_pi[0,0]    = 3. * n**2 * (l**2 + m**2)
    tdd_delta[0,0] = 0.75 * (l**2 + m**2)**2
    tdd_sigma[0,1] = tdd_sigma[1,0] = np.sqrt(3) * l * n * (n**2 - 0.5 * (l**2 + m**2))
    tdd_pi[0,1]    = tdd_pi[1,0]    = np.sqrt(3) * l * n * (l**2 + m**2 - n**2)
    tdd_delta[0,1] = tdd_delta[1,0] = -np.sqrt(3)/2. * l * n * (l**2 + m**2)
    tdd_sigma[0,2] = tdd_sigma[2,0] = np.sqrt(3) * m * n * (n**2 - 0.5 * (l**2 + m**2))
    tdd_pi[0,2]    = tdd_pi[2,0]    = np.sqrt(3) * m * n * (l**2 + m**2 - n**2)
    tdd_delta[0,2] = tdd_delta[2,0] = -np.sqrt(3)/2. * m * n * (l**2 + m**2)
    tdd_sigma[0,3] = tdd_sigma[3,0] = np.sqrt(3)/2. * (l**2 - m**2) * (n**2 - 0.5 * (l**2 + m**2))
    tdd_pi[0,3]    = tdd_pi[3,0]    = np.sqrt(3) * n**2 * (m**2 - l**2)
    tdd_delta[0,3] = tdd_delta[3,0] = np.sqrt(3)/4. * (1. + n**2) * (l**2 - m**2)
    tdd_sigma[0,4] = tdd_sigma[4,0] = np.sqrt(3) * l * m * (n**2 - 0.5 * (l**2 + m**2))
    tdd_pi[0,4]    = tdd_pi[4,0]    = -2.*np.sqrt(3) * l * m * n**2
    tdd_delta[0,4] = tdd_delta[4,0] = np.sqrt(3)/2. * l * m * (1. + n**2)
    tdd_sigma[1,1] = 3. * l**2 * n**2 #
    tdd_pi[1,1]    = (l**2 + n**2 - 4. * l**2 * n**2) #
    tdd_delta[1,1] = (m**2 + l**2 * n**2) #
    tdd_sigma[1,2] = tdd_sigma[2,1] = 3. * l * m * n**2 #
    tdd_pi[1,2]    = tdd_pi[2,1]    = l * m * (1. - 4. * n**2) #
    tdd_delta[1,2] = tdd_delta[2,1] = l * m * (n**2 - 1.) #
    tdd_sigma[1,3] = tdd_sigma[3,1] = 1.5 * n * l * (l**2 - m**2)
    tdd_pi[1,3]    = tdd_pi[3,1]    = n * l * (1. - 2. * (l**2 - m**2))
    tdd_delta[1,3] = tdd_delta[3,1] = -1. * n * l * (1. - 0.5 * (l**2 - m**2))
    tdd_sigma[1,4] = tdd_sigma[4,1] = 3. * l**2 * m * n
    tdd_pi[1,4]    = tdd_pi[4,1]    = m * n * (1. - 4. * l**2)
    tdd_delta[1,4] = tdd_delta[4,1] = m * n * (l**2 - 1.)
    tdd_sigma[2,2] = 3. * m**2 * n**2 #
    tdd_pi[2,2]    = (m**2 + n**2 - 4. * m**2 * n**2) #
    tdd_delta[2,2] = (l**2 + m**2 * n**2) #
    tdd_sigma[2,3] = tdd_sigma[3,2] = 1.5 * m * n * (l**2 - m**2)
    tdd_pi[2,3]    = tdd_pi[3,2]    = -1. * m * n * (1. + 2. * (l**2 - m**2))
    tdd_delta[2,3] = tdd_delta[3,2] = m * n * (1. + 0.5 * (l**2 - m**2))
    tdd_sigma[2,4] = tdd_sigma[4,2] = 3. * l * m**2 * n
    tdd_pi[2,4]    = tdd_pi[4,2]    = l * n * (1. - 4. * m**2)
    tdd_delta[2,4] = tdd_delta[4,2] = l * n * (m**2 - 1.)
    tdd_sigma[3,3] = 0.75 * (l**2 - m**2)**2
    tdd_pi[3,3]    = l**2 + m**2 - (l**2 - m**2)**2
    tdd_delta[3,3] = n**2 + 0.25 * (l**2 - m**2)**2
    tdd_sigma[3,4] = tdd_sigma[4,3] = 1.5 * l * m * (l**2 - m**2)
    tdd_pi[3,4]    = tdd_pi[4,3]    = 2. * l * m * (m**2 - l**2)
    tdd_delta[3,4] = tdd_delta[4,3] = 0.5 * l * m * (l**2 - m**2)
    tdd_sigma[4,4] = 3. * l**2 * m**2
    tdd_pi[4,4]    = (l**2 + m**2 - 4. * l**2 * m**2)
    tdd_delta[4,4] = (n**2 + l**2 * m**2)
    return tdd_sigma, tdd_pi, tdd_delta

def tmat_t2o(index, spin=False):
    """
    Get transform matrix from trigonal to octehedral notations for face-sharing scenario.
    
    Parameters
    ----------
    index : int, index of transition metal (1 or 2).
    
    Returns
    -------
    tmat : 3x3 2d arrays, dtype=float.
    """
    if index%2==1:
        tmat = np.zeros((5,5), dtype=float)
        tmat[:,0] = [0., -1*np.sqrt(2/3), 0., np.sqrt(1/3), 0.]
        tmat[:,1] = [np.sqrt(1/3), -1/6*np.sqrt(2), -1/np.sqrt(6), -1/3, np.sqrt(1/3)]
        tmat[:,2] = [np.sqrt(1/3), -1/6*np.sqrt(2), 1/np.sqrt(6), -1/3, -np.sqrt(1/3)]
        tmat[:,3] = [0., 0., -1*np.sqrt(2/3), 0., -1*np.sqrt(1/3)]
        tmat[:,4] = [np.sqrt(1/3), np.sqrt(2/9), 0., 2/3, 0.]
    else:
        tmat = np.zeros((5,5), dtype=float)
        tmat[:,0] = [0., 1*np.sqrt(2/3), 0., np.sqrt(1/3), 0.]
        tmat[:,1] = [np.sqrt(1/3), 1/6*np.sqrt(2), 1/np.sqrt(6), -1/3, np.sqrt(1/3)]
        tmat[:,2] = [np.sqrt(1/3), 1/6*np.sqrt(2), -1/np.sqrt(6), -1/3, -np.sqrt(1/3)]
        tmat[:,3] = [0., 0., 1*np.sqrt(2/3), 0., -1*np.sqrt(1/3)]
        tmat[:,4] = [np.sqrt(1/3), -np.sqrt(2/9), 0., 2/3, 0.]
    if spin:
        new_tmat = np.zeros((10,10), dtype=complex)
        new_tmat[::2,::2] = tmat
        new_tmat[1::2,1::2] = tmat
        return new_tmat
    else:
        return tmat


def tmat_o2t(index, spin=False):
    """
    Get transform matrix from octehedral to trigonal notations for face-sharing scenario.
    
    Parameters
    ----------
    index : int, index of transition metal (1 or 2).
    
    Returns
    -------
    tmat : 3x3 2d arrays, dtype=float.
    """
    if index%2==1:
        tmat = np.zeros((5,5), dtype=float)
        tmat[:,0] = [0., 1/np.sqrt(3), 1/np.sqrt(3), 0., 1/np.sqrt(3)]
        tmat[:,1] = [-np.sqrt(2/3), -np.sqrt(1/18), -np.sqrt(1/18), 0., np.sqrt(2/9)]
        tmat[:,2] = [0., -np.sqrt(1/6), np.sqrt(1/6), -np.sqrt(2/3), 0.]
        tmat[:,3] = [1/np.sqrt(3), -1/3, -1/3, 0., 2/3]
        tmat[:,4] = [0., 1/np.sqrt(3), -1/np.sqrt(3), -1/np.sqrt(3), 0.]
    else:
        tmat = np.zeros((5,5), dtype=float)
        tmat[:,0] = [0., 1/np.sqrt(3), 1/np.sqrt(3), 0., 1/np.sqrt(3)]
        tmat[:,1] = [np.sqrt(2/3), np.sqrt(1/18), np.sqrt(1/18), 0., -np.sqrt(2/9)]
        tmat[:,2] = [0., np.sqrt(1/6), -np.sqrt(1/6), np.sqrt(2/3), 0.]
        tmat[:,3] = [1/np.sqrt(3), -1/3, -1/3, 0., 2/3]
        tmat[:,4] = [0., 1/np.sqrt(3), -1/np.sqrt(3), -1/np.sqrt(3), 0.]
    if spin:
        new_tmat = np.zeros((10,10), dtype=complex)
        new_tmat[::2,::2] = tmat
        new_tmat[1::2,1::2] = tmat
        return new_tmat
    else:
        return tmat

def rmat_t2o(index):
    """
    Get rotational matrix from trigonal to octehedral notations for face-sharing scenario.
    
    Parameters
    ----------
    index : int, index of transition metal (1 or 2).
    
    Returns
    -------
    rmat : 3x3 2d arrays, dtype=float.
    """
    if index%2==1:
        rmat = np.zeros((3,3), dtype=float)
        rmat[:,0] = [np.sqrt(1/6), -np.sqrt(1/2), np.sqrt(1/3)]
        rmat[:,1] = [np.sqrt(1/6), np.sqrt(1/2), np.sqrt(1/3)]
        rmat[:,2] = [-np.sqrt(2/3), 0., np.sqrt(1/3)]
    else:
        rmat = np.zeros((3,3), dtype=float)
        rmat[:,0] = [-np.sqrt(1/6), np.sqrt(1/2), np.sqrt(1/3)]
        rmat[:,1] = [-np.sqrt(1/6), -np.sqrt(1/2), np.sqrt(1/3)]
        rmat[:,2] = [np.sqrt(2/3), 0., np.sqrt(1/3)]
    return rmat

def rmat_o2t(index):
    """
    Get rotational matrix from octehedral to trigonal notations for face-sharing scenario.
    
    Parameters
    ----------
    index : int, index of transition metal (1 or 2).
    
    Returns
    -------
    rmat : 3x3 2d arrays, dtype=float.
    """
    return np.linalg.inv(rmat_t2o(index))
