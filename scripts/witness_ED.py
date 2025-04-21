# Adapted from dimer_cluster in application to entanglement witness

import numpy as np
import scipy.sparse as sparse
from edrixs import get_F0, get_spin_momentum, get_orb_momentum, atom_hsoc, get_trans_oper, get_umat_slater, scattering_mat, dipole_polvec_rixs
from edrixs import cb_op, transform_utensor, tmat_c2r, tmat_r2c, rmat_to_euler, dmat_spinor
from witness_utils import get_tpp, get_tpd, get_tdd, tmat_t2o, tmat_o2t, rmat_t2o, rmat_o2t
from witness_utils import SparseBinary, Sparse4D, get_fock_bin_by_N_S, two_fermion_S, four_fermion_S
from witness_utils import diag

class dimer_witness:
    """
    Container for face-sharing dimer cluster model at TM L edge.
    Real basis is used along with electron language.
    Four-fermion Hamiltonians are saved in folder/garage.
    
    Operations
    ----------
    construct_basis : Construct the Hartree Fock basis_i, basis_n1 and basis_n2.
    produce_dipole  : Construct dipolar transition operator for cluster calculation.
    load_params     : Load all the parameters used for cluster calculations in electrpon language.
    produce_emat    : Produce emat in hole language.
    produce_umat    : Produce umat in hole language.
    load_hmat_4f    : Load four-fermion Hamiltonian.
    perform_ED      : Perform calculations.
    """

    def __init__(self, folder=''):
        """
        Class initialization.

        Parameters
        ----------
        folder : str
            Working directory (default: '').
        
        Examples
        --------
        >>> witness = dimer_witness(folder='Ir_d10')
        """
        self.folder = folder
        self.params = None
    
    def __getitem__(self, key):
        """ Assign [] indexing method to try to return data associated with a key. """
        return self.index(key)
    
    def index(self, key):
        """ Index a particular key. """
        info = dict(folder=self.folder)
        if key in info.keys():
            return info[key]
        elif key=='basis':
            basis = dict(np.load(self.folder+'/basis.npz'))
            return SparseBinary(basis['coordinates'], basis['shape'])
        elif key=='basis_i':
            basis = dict(np.load(self.folder+'/basis_i.npz'))
            return SparseBinary(basis['coordinates'], basis['shape'])
        elif key=='basis_n1':
            basis = dict(np.load(self.folder+'/basis_n1.npz'))
            return SparseBinary(basis['coordinates'], basis['shape'])
        elif key=='basis_n2':
            basis = dict(np.load(self.folder+'/basis_n2.npz'))
            return SparseBinary(basis['coordinates'], basis['shape'])
        elif key=='dipole':
            dipole_raw = dict(np.load(self.folder+'/dipole.npz'))
            all_dipole = []
            for nn in range(2):
                dipole = []
                for nxyz, xyz in enumerate(['x','y','z']):
                    dipole.append(sparse.coo_matrix((dipole_raw[f'n{nn+1}{xyz}_data'],
                                                    (dipole_raw[f'n{nn+1}{xyz}_row'], dipole_raw[f'n{nn+1}{xyz}_col'])),
                                                    shape=dipole_raw[f'n{nn+1}{xyz}_shape'], dtype=complex))
                all_dipole.append(dipole)
            return all_dipole
        elif key=='emat':
            emat = self.produce_emat()
            return sparse.coo_matrix(emat)
        elif key=='emat0':
            emat = self.produce_emat()
            return sparse.coo_matrix(emat[:20,:20])
        elif key=='umat':
            umat = self.produce_umat()
            tol = 1e-8
            umat_elements = umat[abs(umat)>tol]
            umat_coordinates = np.where(abs(umat)>tol)
            return Sparse4D(umat_elements, umat_coordinates, shape=[32]*4, dtype=complex)
        elif key=='umat0':
            umat = self.produce_umat()[:20,:20,:20,:20]
            tol = 1e-8
            umat_elements = umat[abs(umat)>tol]
            umat_coordinates = np.where(abs(umat)>tol)
            return Sparse4D(umat_elements, umat_coordinates, shape=[20]*4, dtype=complex)
        elif key=='evals':
            return np.load(self.folder+'/evals.npy')
        elif key=='evecs':
            return np.load(self.folder+'/evecs.npy')
        elif key=='evals_i':
            return np.load(self.folder+'/evals_i.npy')
        elif key=='evecs_i':
            return np.load(self.folder+'/evecs_i.npy')
        elif key=='evals_n':
            return [np.load(self.folder+'/evals_n1.npy'), np.load(self.folder+'/evals_n2.npy')]
        elif key=='evecs_n':
            return [np.load(self.folder+'/evecs_n1.npy'), np.load(self.folder+'/evecs_n2.npy')]
        elif key=='T_abs':
            return dict(np.load(self.folder+'/T_abs.npz'))
    
    def construct_basis(self, noccu, eg_max=0):
        """
        Construct the Hartree Fock basis.

        Parameters
        ----------
        noccu : int
            Number of electrons occupying the valence states.
        eg_max : int
            Maximum number of electrons occupying the eg orbitals, for both the initial and intermediate states (default: 0).

        Returns
        -------
        The Hartree Fock basis in form of SparseBinary will be saved in folder.

        Examples
        --------
        >>> witness.construct_basis(10, eg_max=1)
        """
        all_basis = []
        for eg in np.arange(0,eg_max+1):
            all_basis.append(get_fock_bin_by_N_S(8,eg, 12,noccu-eg, dispinfo=False))
        basis = SparseBinary(np.vstack(all_basis), shape=[None,20])
        # change basis from the complex harmonic basis to the real harmonic basis
        coo_table = np.hstack([0,1,6,7,10,11,16,17,2,3,4,5,8,9,12,13,14,15,18,19])
        basis.change_coordinates(coo_table, shape=[None,20])
        print('basis shape without core levels: {}'.format(basis.get_shape()))
        basis.save(self.folder+'/basis.npz')

        all_basis = []
        for eg in np.arange(0,eg_max+1):
            all_basis.append(get_fock_bin_by_N_S(8,eg, 12,noccu-eg, 6,6, 6,6, dispinfo=False))
        basis = SparseBinary(np.vstack(all_basis), shape=[None,32])
        # change basis from the complex harmonic basis to the real harmonic basis
        coo_table = np.hstack([0,1,6,7,10,11,16,17,2,3,4,5,8,9,12,13,14,15,18,19,np.arange(20,32)])
        basis.change_coordinates(coo_table, shape=[None,32])
        print('basis shape for initial/final states: {}'.format(basis.get_shape()))
        basis.save(self.folder+'/basis_i.npz')

        all_basis = []
        for eg in np.arange(0,eg_max+1):
            all_basis.append(get_fock_bin_by_N_S(8,eg, 12,noccu-eg+1, 6,5, 6,6, dispinfo=False))
        basis = SparseBinary(np.vstack(all_basis), shape=[None,32])
        # change basis from the complex harmonic basis to the real harmonic basis
        coo_table = np.hstack([0,1,6,7,10,11,16,17,2,3,4,5,8,9,12,13,14,15,18,19,np.arange(20,32)])
        basis.change_coordinates(coo_table, shape=[None,32])
        print('basis shape for site 1 intermediate states: {}'.format(basis.get_shape()))
        basis.save(self.folder+'/basis_n1.npz')

        all_basis = []
        for eg in np.arange(0,eg_max+1):
            all_basis.append(get_fock_bin_by_N_S(8,eg, 12,noccu-eg+1, 6,6, 6,5, dispinfo=False))
        basis = SparseBinary(np.vstack(all_basis), shape=[None,32])
        # change basis from the complex harmonic basis to the real harmonic basis
        coo_table = np.hstack([0,1,6,7,10,11,16,17,2,3,4,5,8,9,12,13,14,15,18,19,np.arange(20,32)])
        basis.change_coordinates(coo_table, shape=[None,32])
        print('basis shape for site 2 intermediate states: {}'.format(basis.get_shape()))
        basis.save(self.folder+'/basis_n2.npz')
    
    def produce_dipole(self):
        """
        Construct dipolar transition operator between Hartree Fock basis.

        Returns
        -------
        Dipolar transition operator in form of a dictionary saved in folder.
        
        Examples
        --------
        >>> witness.construct_dipole()
        """
        basis = dict(np.load(self.folder+'/basis_i.npz'))
        basis_i = SparseBinary(basis['coordinates'], basis['shape'])
        tmp = cb_op(get_trans_oper('dp'), tmat_c2r('d', True), tmat_c2r('p', True))
        all_dipole = dict()
        basis = dict(np.load(self.folder+'/basis_n1.npz'))
        basis_n = SparseBinary(basis['coordinates'], basis['shape'])
        for nxyz, xyz in enumerate(['x','y','z']):
            dipole_tmp = np.zeros((32,32), dtype=complex)
            dipole_tmp[0:10, 20:26] = tmp[nxyz]
            dipole = two_fermion_S(sparse.coo_matrix(dipole_tmp), basis_n, basis_i, dtype=complex)
            all_dipole['n1{}_row'.format(xyz)] = dipole.row
            all_dipole['n1{}_col'.format(xyz)] = dipole.col
            all_dipole['n1{}_data'.format(xyz)] = dipole.data
            all_dipole['n1{}_shape'.format(xyz)] = dipole.shape
        basis = dict(np.load(self.folder+'/basis_n2.npz'))
        basis_n = SparseBinary(basis['coordinates'], basis['shape'])
        for nxyz, xyz in enumerate(['x','y','z']):
            dipole_tmp = np.zeros((32,32), dtype=complex)
            dipole_tmp[10:20, 26:32] = tmp[nxyz]
            dipole = two_fermion_S(sparse.coo_matrix(dipole_tmp), basis_n, basis_i, dtype=complex)
            all_dipole['n2{}_row'.format(xyz)] = dipole.row
            all_dipole['n2{}_col'.format(xyz)] = dipole.col
            all_dipole['n2{}_data'.format(xyz)] = dipole.data
            all_dipole['n2{}_shape'.format(xyz)] = dipole.shape
        np.savez(self.folder+'/dipole.npz', **all_dipole)
    
    def load_params(self, tenDq=0., theta=54.7, kappa=1., a0=0., soc=0., soc_c=1140.332,
                    Vdd_sigma=0., Vdd_pi=0., Vdd_delta=0.,
                    U_dd=2., J_dd=0.3, U_q=None, F2_dp=1.03, G1_dp=0.889, G3_dp=0.528):
        """
        Load all the parameters used for cluster calculations in electron language.

        Parameters
        ----------
        tenDq : float
            Crystal field splitting 10Dq (default: 0.).
        theta : float
            TM-TM-O bond angle, 54.7 deg in undistorted case (default=54.7).
            See K. I. Kugel et al., Phy. Rev. B 91, 155125 (2015).
        kappa : float
            Element dependent crystal field splitting parameter (default: 1.).
        a0 : float
            Trigonal distortion due to neighbored cations (default: 0.).
        soc : float
            Spin orbital coupling strength for valence bands (default: 0).
        soc_c : float
            Spin orbital coupling strength for core levels (default: 1140.332).
        Vdd_sigma, Vdd_pi, Vdd_delta: float
            Slater-Koster parameter for d-d hopping (default: 0.).
        U_dd : float
            TM intra-orbital Coulomb interaction: U_dd = F0 + 4./49. * F2 + 4./49. * F4 (default: 2).
        J_dd : float
            TM Hund's coupling: J_dd = (F2 + F4) / 14 (default: 0.3).
        U_q : float
            Core hole potential. If None, U_q = U_dd - 0.5 (default: None).
        F2_dp, G1_dp, G3_dp: float
            Other core hole potential parameters (default: F2_dp=1.03, G1_dp=0.889, G3_dp=0.528)
        """
        self.params = dict()
        self.params['tenDq'] = tenDq
        self.params['theta'] = theta
        self.params['kappa'] = kappa
        self.params['a0']    = a0
        self.params['soc']   = soc
        self.params['soc_c'] = soc_c
        self.params['Vdd_sigma'] = Vdd_sigma
        if Vdd_pi is None:
            self.params['Vdd_pi'] = -2/3 * Vdd_sigma
        else:
            self.params['Vdd_pi'] = Vdd_pi
        if Vdd_delta is None:
            self.params['Vdd_delta'] = 1/6 * Vdd_sigma
        else:
            self.params['Vdd_delta'] = Vdd_delta
        
        F4F2_ratio = 0.625
        F2_dd = J_dd * 14. / (1. + F4F2_ratio)
        F4_dd = F2_dd * F4F2_ratio
        F0_dd = U_dd - 4.0 / 49.0 * F2_dd - 4.0 / 49.0 * F4_dd
        if U_q is None:
            U_q = U_dd - 0.5
        F0_dp = get_F0('dp', G1_dp, G3_dp) + U_q
        self.params['U_dd']  = U_dd
        self.params['J_dd']  = J_dd
        self.params['U_q']   = U_q
        self.params['F0_dd'] = F0_dd
        self.params['F2_dd'] = F2_dd
        self.params['F4_dd'] = F4_dd
        self.params['F0_dp'] = F0_dp
        self.params['F2_dp'] = F2_dp
        self.params['G1_dp'] = G1_dp
        self.params['G3_dp'] = G3_dp
    
    def produce_emat(self):
        """
        Produce emat in electron language.
        
        Returns
        -------
        Two-fermion matrix in non-sparse format.
        
        Examples
        --------
        >>> witness.construct_emat()
        """
        emat = np.zeros((32,32), dtype=complex)

        # emat0: spin-unresolved emat in trigonal notation
        ## asuming nondistorted octahedra but including trigonal field in a2
        emat0 = np.zeros((10,10), dtype=complex)
        a2 = 27./35.*self.params['kappa']*(3.*np.cos(np.deg2rad(self.params['theta']))**2 - 1.) - self.params['a0']
        a4 = -1.5 * (2.5*np.cos(np.deg2rad(self.params['theta']))**4-15./7.*np.cos(np.deg2rad(self.params['theta']))**2+3./14.)
        b = 3. * np.sin(np.deg2rad(self.params['theta']))**3 * np.cos(np.deg2rad(self.params['theta']))
        emat_tmp = np.diag([(-18.*a4+10.*a2)/15., (12.*a4+5.*a2)/15., (12.*a4+5.*a2)/15., (-3.*a4-10.*a2)/15., (-3.*a4-10.*a2)/15.])
        emat_tmp[1,3] = emat_tmp[3,1] = -b/2
        emat_tmp[2,4] = emat_tmp[4,2] = b/2
        emat_tmp *= self.params['tenDq']
        emat0[:5,:5] += emat_tmp
        emat_tmp = np.diag([(-18.*a4+10.*a2)/15., (12.*a4+5.*a2)/15., (12.*a4+5.*a2)/15., (-3.*a4-10.*a2)/15., (-3.*a4-10.*a2)/15.])
        emat_tmp[1,3] = emat_tmp[3,1] = b/2
        emat_tmp[2,4] = emat_tmp[4,2] = -b/2
        emat_tmp *= self.params['tenDq']
        emat0[5:,5:] += emat_tmp

        # hybridization
        tdd_sigma, tdd_pi, tdd_delta = get_tdd(0., 0., 1.)
        emat0[5:,:5] = tdd_sigma*self.params['Vdd_sigma'] + tdd_pi*self.params['Vdd_pi'] + tdd_delta*self.params['Vdd_delta']
        emat0[:5,5:] = np.transpose(tdd_sigma*self.params['Vdd_sigma'] + tdd_pi*self.params['Vdd_pi'] + tdd_delta*self.params['Vdd_delta'])

        # transform to local axis
        tmat = np.zeros((10,10), dtype=float)
        tmat[:5,:5] = tmat_t2o(1)
        tmat[5:,5:] = tmat_t2o(2)
        emat0 = cb_op(emat0, tmat)
        emat[:20:2,:20:2] += emat0
        emat[1:20:2,1:20:2] += emat0

        # spin orbital coupling
        soc_mat =np.zeros((20,20), dtype=complex)
        soc_mat[:10,:10] = cb_op(atom_hsoc('d', self.params['soc']), tmat_c2r('d', True))
        soc_mat[10:,10:] = cb_op(atom_hsoc('d', self.params['soc']), tmat_c2r('d', True))

        # transform to local axis
        dmat = np.zeros((2, 2, 2), dtype=complex)
        ang1, ang2, ang3 = rmat_to_euler(rmat_o2t(1))
        dmat[0] = dmat_spinor(ang1, ang2, ang3)
        ang1, ang2, ang3 = rmat_to_euler(rmat_o2t(2))
        dmat[1] = dmat_spinor(ang1, ang2, ang3)
        t_spinor = np.zeros((20,20), dtype=complex)
        for nTM in range(2):
            for norb in range(5):
                t_spinor[(nTM*10+norb*2):(nTM*10+norb*2+2),(nTM*10+norb*2):(nTM*10+norb*2+2)] = dmat[nTM]
        soc_emat = cb_op(soc_mat, t_spinor)
        emat[:20,:20] += soc_emat

        # soc for core
        soc_c = 1140.332
        soc_mat =np.zeros((12,12), dtype=complex)
        soc_mat[:6,:6] = cb_op(atom_hsoc('p', self.params['soc_c']), tmat_c2r('p', True))
        soc_mat[6:,6:] = cb_op(atom_hsoc('p', self.params['soc_c']), tmat_c2r('p', True))
        t_spinor = np.zeros((12,12), dtype=complex)
        for nTM in range(2):
            for norb in range(3):
                t_spinor[(nTM*6+norb*2):(nTM*6+norb*2+2),(nTM*6+norb*2):(nTM*6+norb*2+2)] = dmat[nTM]
        soc_mat = cb_op(soc_mat, t_spinor)
        emat[20:,20:] += soc_mat

        return emat
    
    def produce_umat(self):
        """
        Produce umat in electron language.
        
        Returns
        -------
        Four-fermion matrix in non-sparse format.
        
        Examples
        --------
        >>> witness.construct_umat()
        """
        umat = np.zeros((32,32,32,32), dtype=complex)
        tmat = np.zeros((16,16), dtype=complex)
        tmat[:10,:10] = tmat_c2r('d', True)
        tmat[10:,10:] = tmat_c2r('p', True)
        umat_tmp = transform_utensor(get_umat_slater('dp', self.params['F0_dd'], self.params['F2_dd'], self.params['F4_dd'],
                                                     self.params['F0_dp'], self.params['F2_dp'], self.params['G1_dp'], self.params['G3_dp'], 0, 0), tmat)
        index = np.hstack([np.arange(10), np.arange(20,26)])
        for n1 in range(16):
            for n2 in range(16):
                for n3 in range(16):
                    for n4 in range(16):
                        umat[index[n1],index[n2],index[n3],index[n4]] += umat_tmp[n1,n2,n3,n4]
        index = np.hstack([np.arange(10,20), np.arange(26,32)])
        for n1 in range(16):
            for n2 in range(16):
                for n3 in range(16):
                    for n4 in range(16):
                        umat[index[n1],index[n2],index[n3],index[n4]] += umat_tmp[n1,n2,n3,n4]
        
        return umat
    
    def load_hmat_4f(self, rixs=True, overwrite=False):
        """
        Load four-fermion Hamiltonian from garage folder.
        If the file cannot be found, it will generate hmat and save it.
        Note that F2dp, G1dp, G3dp are not in the title.
        
        Parameters
        ----------
        RIXS : bool
            Whether to include core levels.
            If True, it loads hmat_i_4f and hmat_i_n1/hmat_i_n2.
            If False, it loads hmat_4f without core orbitals (default: True).
        overwrite : bool
            Whether to overwrite the existing hmat (default: False).
        
        Returns
        -------
        Four-fermion Hamiltonians in sparse matrix format.
        
        Examples
        --------
        >>> witness.load_hmat_4f()
        """
        filename = 'Udd_{:.2f}_Jdd_{:.2f}_Uq_{:.2f}.npz'.format(
            self.params['U_dd'], self.params['J_dd'], self.params['U_q'])
        
        if rixs:
            if overwrite:
                hmat_i = four_fermion_S(self.index('umat'), self.index('basis_i'), dtype=complex)
                sparse.save_npz(self.folder+'/garage/hmat_i_4f_'+filename, hmat_i)
                hmat_n1 = four_fermion_S(self.index('umat'), self.index('basis_n1'), dtype=complex)
                sparse.save_npz(self.folder+'/garage/hmat_n1_4f_'+filename, hmat_n1)
                hmat_n2 = four_fermion_S(self.index('umat'), self.index('basis_n2'), dtype=complex)
                sparse.save_npz(self.folder+'/garage/hmat_n2_4f_'+filename, hmat_n2)
            else:
                try:
                    hmat_i = sparse.coo_matrix(sparse.load_npz(self.folder+'/garage/hmat_i_4f_'+filename))
                except:
                    hmat_i = four_fermion_S(self.index('umat'), self.index('basis_i'), dtype=complex)
                    sparse.save_npz(self.folder+'/garage/hmat_i_4f_'+filename, hmat_i)
                try:
                    hmat_n1 = sparse.coo_matrix(sparse.load_npz(self.folder+'/garage/hmat_n1_4f_'+filename))
                except:
                    hmat_n1 = four_fermion_S(self.index('umat'), self.index('basis_n1'), dtype=complex)
                    sparse.save_npz(self.folder+'/garage/hmat_n1_4f_'+filename, hmat_n1)
                try:
                    hmat_n2 = sparse.coo_matrix(sparse.load_npz(self.folder+'/garage/hmat_n2_4f_'+filename))
                except:
                    hmat_n2 = four_fermion_S(self.index('umat'), self.index('basis_n2'), dtype=complex)
                    sparse.save_npz(self.folder+'/garage/hmat_n2_4f_'+filename, hmat_n2)
            return hmat_i, [hmat_n1, hmat_n2]
        else:
            if overwrite:
                hmat = four_fermion_S(self.index('umat0'), self.index('basis'), dtype=complex)
                sparse.save_npz(self.folder+'/garage/hmat_4f_'+filename, hmat)
            else:
                try:
                    hmat = sparse.coo_matrix(sparse.load_npz(self.folder+'/garage/hmat_4f_'+filename))
                except:
                    hmat = four_fermion_S(self.index('umat0'), self.index('basis'), dtype=complex)
                    sparse.save_npz(self.folder+'/garage/hmat_4f_'+filename, hmat)
            return hmat
    
    def perform_ED(self, rixs=True):
        """
        Perform ED and calculate absorption operator if needed.
        It will produce Hamiltonian and perform diagonalization.
        
        Parameters
        ----------
        RIXS : bool
            Whether to include core levels (default: True).
        
        Examples
        --------
        >>> witness.perform_ED()
        """
        if rixs:
            basis_i = self.index('basis_i')
            basis_n1 = self.index('basis_n1')
            basis_n2 = self.index('basis_n2')
            dipole_all = self.index('dipole')
            emat = self.index('emat')
            hmat0_i_4f, hmat0_n_4f = self.load_hmat_4f()
            
            hmat = two_fermion_S(emat, basis_i, dtype=complex) + hmat0_i_4f
            diag(hmat, folder=self.folder, suffix='_i')
            hmat = two_fermion_S(emat, basis_n1, dtype=complex) + hmat0_n_4f[0]
            diag(hmat, folder=self.folder, suffix='_n1')
            hmat = two_fermion_S(emat, basis_n2, dtype=complex) + hmat0_n_4f[1]
            diag(hmat, folder=self.folder, suffix='_n2')

            evecs_i = np.load(self.folder+'/evecs_i.npy')
            evecs_n = [np.load(self.folder+'/evecs_n1.npy'), np.load(self.folder+'/evecs_n2.npy')]
            T_abs_all  = []
            for nnsite in range(2):
                T_abs = []
                for xyz in range(3):
                    T_abs.append(dipole_all[nnsite][xyz].transpose().dot(np.conj(evecs_n[nnsite])).transpose().dot(evecs_i))
                T_abs_all.append(np.stack(T_abs))
            np.savez(self.folder+'/T_abs.npz', T_abs1=T_abs_all[0], T_abs2=T_abs_all[1])
        else:
            basis = self.index('basis')
            emat = self.index('emat0')
            hmat0_4f = self.load_hmat_4f(rixs=False)
            hmat = two_fermion_S(emat, basis, dtype=complex) + hmat0_4f
            diag(hmat, folder=self.folder, suffix='')
