import block2 as b
import numpy as np
from itertools import accumulate
from pyblock2.driver.core import SymmetryTypes

class CustomHamiltonian(b.sz.GeneralHamiltonian):
    # Now only support Sz 
    def __init__(self, driver, vacuum, n_sites, orb_sym, n_sites_fermion = 0, n_sites_boson = 0, nbcuts = []):
        b.sz.GeneralHamiltonian.__init__(self)
        self.driver = driver
        self.opf = driver.bw.bs.OperatorFunctions(driver.bw.bs.CG())
        self.vacuum = vacuum
        self.n_sites = n_sites
        self.n_sites_fermion = n_sites_fermion
        self.n_sites_boson = n_sites_boson
        self.nbcuts = nbcuts
        self.orb_sym = orb_sym

        # Add bases for Boson [Shuoxue]
        basis_fermion = [self.get_site_basis(m) for m in range(self.n_sites_fermion)]
        basis_boson = [self.get_site_basis_boson(self.nbcuts[k]) for k in range(self.n_sites_boson)]
        self.basis = self.driver.bw.bs.VectorStateInfo(basis_fermion + basis_boson)
        
        self.site_op_infos = self.driver.bw.bs.VectorVectorPLMatInfo([
            self.driver.bw.bs.VectorPLMatInfo() for _ in range(self.n_sites)
        ])
        self.site_norm_ops = self.driver.bw.bs.VectorMapStrSpMat([
            self.driver.bw.bs.MapStrSpMat() for _ in range(self.n_sites)
        ])
        self.init_site_ops()

    def get_site_basis(self, m):
        """Single site states."""
        bz = self.driver.bw.bs.StateInfo()
        bz.allocate(4)
        bz.quanta[0] = self.driver.bw.SX(0, 0, 0)
        bz.quanta[1] = self.driver.bw.SX(1, 1, self.orb_sym[m])
        bz.quanta[2] = self.driver.bw.SX(1, -1, self.orb_sym[m])
        bz.quanta[3] = self.driver.bw.SX(2, 0, 0)
        bz.n_states[0] = bz.n_states[1] = bz.n_states[2] = bz.n_states[3] = 1
        bz.sort_states()
        return bz

    # Site basis for Boson [Shuoxue]
    def get_site_basis_boson(self, nbcut):
        """Single-site Boson states with nbcut"""

        bz = self.driver.bw.bs.StateInfo()
        bz.allocate(1)
        bz.quanta[0] = self.driver.bw.SX(0,0,0)
        bz.n_states[0] = nbcut  # add the cutoff
        bz.sort_states()
        return bz

    def init_site_ops(self):
        """Initialize operator quantum numbers at each site (site_op_infos)
        and primitive (single character) site operators (site_norm_ops)."""
        i_alloc = self.driver.bw.b.IntVectorAllocator()
        d_alloc = self.driver.bw.b.DoubleVectorAllocator()
        # site op infos
        max_n, max_s = 10, 10
        max_n_odd, max_s_odd = max_n | 1, max_s | 1
        max_n_even, max_s_even = max_n_odd ^ 1, max_s_odd ^ 1
        for m in range(self.n_sites_fermion):
            qs = {self.vacuum}
            for n in range(-max_n_odd, max_n_odd + 1, 2):
                for s in range(-max_s_odd, max_s_odd + 1, 2):
                    qs.add(self.driver.bw.SX(n, s, self.orb_sym[m]))
            for n in range(-max_n_even, max_n_even + 1, 2):
                for s in range(-max_s_even, max_s_even + 1, 2):
                    qs.add(self.driver.bw.SX(n, s, 0))
            for q in sorted(qs):
                mat = self.driver.bw.bs.SparseMatrixInfo(i_alloc)
                mat.initialize(self.basis[m], self.basis[m], q, q.is_fermion)
                self.site_op_infos[m].append((q, mat))

        # Initialize site_op_infos for Boson [Shuoxue]
        for k in range(self.n_sites_boson):
            mat = self.driver.bw.bs.SparseMatrixInfo(i_alloc)
            mat.initialize(self.basis[k+self.n_sites_fermion], self.basis[k+self.n_sites_fermion], self.vacuum, self.vacuum.is_fermion)
            self.site_op_infos[k + self.n_sites_fermion].append((self.vacuum, mat))
        
        # prim ops
        for m in range(self.n_sites_fermion):

            # ident
            mat = self.driver.bw.bs.SparseMatrix(d_alloc)
            info = self.find_site_op_info(m, self.driver.bw.SX(0, 0, 0))
            mat.allocate(info)
            mat[info.find_state(self.driver.bw.SX(0, 0, 0))] = np.array([1.0])
            mat[info.find_state(self.driver.bw.SX(1, 1, self.orb_sym[m]))] = np.array([1.0])
            mat[info.find_state(self.driver.bw.SX(1, -1, self.orb_sym[m]))] = np.array([1.0])
            mat[info.find_state(self.driver.bw.SX(2, 0, 0))] = np.array([1.0])
            self.site_norm_ops[m][""] = mat

            # C alpha
            mat = self.driver.bw.bs.SparseMatrix(d_alloc)
            info = self.find_site_op_info(m, self.driver.bw.SX(1, 1, self.orb_sym[m]))
            mat.allocate(info)
            mat[info.find_state(self.driver.bw.SX(0, 0, 0))] = np.array([1.0])
            mat[info.find_state(self.driver.bw.SX(1, -1, self.orb_sym[m]))] = np.array([1.0])
            self.site_norm_ops[m]["c"] = mat

            # D alpha
            mat = self.driver.bw.bs.SparseMatrix(d_alloc)
            info = self.find_site_op_info(m, self.driver.bw.SX(-1, -1, self.orb_sym[m]))
            mat.allocate(info)
            mat[info.find_state(self.driver.bw.SX(1, 1, self.orb_sym[m]))] = np.array([1.0])
            mat[info.find_state(self.driver.bw.SX(2, 0, 0))] = np.array([1.0])
            self.site_norm_ops[m]["d"] = mat

            # C beta
            mat = self.driver.bw.bs.SparseMatrix(d_alloc)
            info = self.find_site_op_info(m, self.driver.bw.SX(1, -1, self.orb_sym[m]))
            mat.allocate(info)
            mat[info.find_state(self.driver.bw.SX(0, 0, 0))] = np.array([1.0])
            mat[info.find_state(self.driver.bw.SX(1, 1, self.orb_sym[m]))] = np.array([-1.0])
            self.site_norm_ops[m]["C"] = mat

            # D beta
            mat = self.driver.bw.bs.SparseMatrix(d_alloc)
            info = self.find_site_op_info(m, self.driver.bw.SX(-1, 1, self.orb_sym[m]))
            mat.allocate(info)
            mat[info.find_state(self.driver.bw.SX(1, -1, self.orb_sym[m]))] = np.array([1.0])
            mat[info.find_state(self.driver.bw.SX(2, 0, 0))] = np.array([-1.0])
            self.site_norm_ops[m]["D"] = mat

            # Nup * Ndn
            mat = self.driver.bw.bs.SparseMatrix(d_alloc)
            info = self.find_site_op_info(m, self.driver.bw.SX(0, 0, 0))
            mat.allocate(info)
            mat[info.find_state(self.driver.bw.SX(2, 0, 0))] = np.array([1.0])
            self.site_norm_ops[m]["N"] = mat

        for k in range(self.n_sites_boson):

            vac = self.driver.bw.SX(0, 0, 0)

            # Identity 
            mat = self.driver.bw.bs.SparseMatrix(d_alloc)
            info = self.find_site_op_info(k+self.n_sites_fermion, vac)
            mat.allocate(info)
            mat[info.find_state(vac)] = np.eye(self.nbcuts[k])
            self.site_norm_ops[k + self.n_sites_fermion][""] = mat

            # b^\dagger as E
            op_bdagger = np.zeros((self.nbcuts[k], self.nbcuts[k]))
            for i in range(self.nbcuts[k] - 1):
                op_bdagger[i+1, i] = np.sqrt(i+1)
            mat = self.driver.bw.bs.SparseMatrix(d_alloc)
            info = self.find_site_op_info(k+self.n_sites_fermion, vac)
            mat.allocate(info)
            mat[info.find_state(vac)] = op_bdagger
            self.site_norm_ops[k + self.n_sites_fermion]["E"] = mat

            # b as F
            op_b = np.zeros((self.nbcuts[k], self.nbcuts[k]))
            for i in range(self.nbcuts[k] - 1):
                op_b[i, i+1] = np.sqrt(i+1)
            mat = self.driver.bw.bs.SparseMatrix(d_alloc)
            info = self.find_site_op_info(k+self.n_sites_fermion, vac)
            mat.allocate(info)
            mat[info.find_state(vac)] = op_b
            self.site_norm_ops[k + self.n_sites_fermion]["F"] = mat

    def get_site_string_ops(self, m, ops):
        """Construct longer site operators from primitive ones."""
        d_alloc = self.driver.bw.b.DoubleVectorAllocator()
        for k in ops:
            if k in self.site_norm_ops[m]:
                ops[k] = self.site_norm_ops[m][k]
            else:
                xx = self.site_norm_ops[m][k[0]]
                for p in k[1:]:
                    xp = self.site_norm_ops[m][p]
                    q = xx.info.delta_quantum + xp.info.delta_quantum
                    mat = self.driver.bw.bs.SparseMatrix(d_alloc)
                    mat.allocate(self.find_site_op_info(m, q))
                    self.opf.product(0, xx, xp, mat)
                    xx = mat
                ops[k] = self.site_norm_ops[m][k] = xx
        return ops

    def init_string_quanta(self, exprs, term_l, left_vacuum):
        """Quantum number for string operators (orbital independent part)."""
        qs = {
            'N': self.driver.bw.SX(0, 0, 0),
            'c':  self.driver.bw.SX(1, 1, 0),
            'C':  self.driver.bw.SX(1, -1, 0),
            'd':  self.driver.bw.SX(-1, -1, 0),
            'D':  self.driver.bw.SX(-1, 1, 0),
            "E": self.driver.bw.SX(0, 0, 0),
            "F": self.driver.bw.SX(0, 0, 0)
        }
        return self.driver.bw.VectorVectorSX([self.driver.bw.VectorSX(list(accumulate(
            [qs['N']] + [qs[x] for x in expr], lambda x, y: x + y)))
            for expr in exprs
        ])

    def get_string_quanta(self, ref, expr, idxs, k):
        """Quantum number for string operators (orbital dependent part)."""
        l, r = ref[k], ref[-1] - ref[k]
        for j, (ex, ix) in enumerate(zip(expr, idxs)):
            ipg = self.orb_sym[ix]
            if ex == "N":
                pass
            elif j < k:
                l.pg = l.pg ^ ipg
            else:
                r.pg = r.pg ^ ipg
        return l, r

    def get_string_quantum(self, expr, idxs):
        """Total quantum number for a string operator."""
        qs = lambda ix: {
            'N': self.driver.bw.SX(0, 0, 0),
            'c':  self.driver.bw.SX(1, 1, self.orb_sym[ix]),
            'C':  self.driver.bw.SX(1, -1, self.orb_sym[ix]),
            'd':  self.driver.bw.SX(-1, -1, self.orb_sym[ix]),
            'D':  self.driver.bw.SX(-1, 1, self.orb_sym[ix]),
            "E": self.driver.bw.SX(0, 0, 0),
            "F": self.driver.bw.SX(0, 0, 0)
        }
        return sum([qs(0)['N']] + [qs(ix)[ex] for ex, ix in zip(expr, idxs)])

    def deallocate(self):
        """Release memory."""
        for ops in self.site_norm_ops:
            for p in ops.values():
                p.deallocate()
        for infos in self.site_op_infos:
            for _, p in infos:
                p.deallocate()
        for bz in self.basis:
            bz.deallocate()
