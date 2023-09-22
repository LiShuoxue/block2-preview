import numpy as np
import json
import sys, os

from pyscf import gto, scf, lo, ao2mo
from pyblock2.driver.core import DMRGDriver, SymmetryTypes
from customhamil import CustomHamiltonian

def get_data(bl=3.2):

    data = {"energies":[], "multiplicities": [], "nphotons": []}

    # the hartree-fock scheme
    mol = gto.M(atom="Li 0 0 0; H 0 0 {:.2f}".format(bl), basis='631g', unit='Bohr')
    print(mol.nao)
    hf = scf.RHF(mol)
    hf.max_cycle = 200
    hf.conv_tol = 1.e-8
    hf.diis_space = 10
    mf = hf.run()

    method = 'rhf'
    mol.build()
    xc = 'b3lyp'
    nmode = 1
    gfac = 0.01 # start with zero, test different strengths
    omega = np.zeros(nmode)

    vec = np.zeros((nmode,3))
    omega[0] = .1 
    vec[0,:] = [1., 1., 1.]  # \lambda_a
    
    ca = lo.orth_ao(mf) # meta lowdin
    
    gmat = np.loadtxt("gmat_sample.txt")
    gmat = gmat.reshape((mol.nao, mol.nao, 1))
    
    h1e = np.einsum("mi,mn,nj->ij", ca.conj(), mol.get_hcore(), ca)
    eri = ao2mo.kernel(mol, ca)
    eri = ao2mo.restore("1", eri, mol.nao)
    
    print(np.linalg.norm(gmat))
    
    nbasis_fermion = mol.nao
    assert nbasis_fermion == 11
    nbasis_boson = 1

    driver = DMRGDriver(scratch="./tmp", symm_type=SymmetryTypes.SZ, n_threads=4, stack_mem=int(2*1024**3))
    b = driver.expr_builder()
    driver.initialize_system(n_sites=nbasis_fermion+nbasis_boson, n_elec=4, spin=0, orb_sym=None)
    nbcuts = [4]

    driver.ghamil = CustomHamiltonian(driver, driver.vacuum, driver.n_sites,
                                      driver.orb_sym,
                                      n_sites_fermion=nbasis_fermion, n_sites_boson=nbasis_boson,
                                      nbcuts=nbcuts)
    mpo = driver.get_qc_eph_mpo(h1e, eri, gmat, omega,
                                iprint=2, ecore=mf.energy_nuc())
    
    ket = driver.get_random_mps(tag="GS", bond_dim=300, nroots=3)
    
    def run_dmrg(driver, mpo):
        bond_dims = [300] * 4 + [600] * 4
        noises = [1e-4] * 4 + [1e-5] * 4 + [0]
        thrds = [1e-9] * 8
        return driver.dmrg(
            mpo,
            ket,
            n_sweeps=100,
            bond_dims=bond_dims,
            noises=noises,
            thrds=thrds,
            iprint=2,
        )

    energies = run_dmrg(driver, mpo)

    data['energies'] = list(energies)
    
    h1e_ssq = .75 * np.eye(nbasis_fermion)
    eri_ssq = np.zeros((nbasis_fermion, ) * 4)
    for i in range(nbasis_fermion):
        for j in range(nbasis_fermion):
            eri_ssq[i,i,j,j] -= .5
            eri_ssq[i,j,j,i] -= 1.
    gmat_ssq = np.zeros_like(gmat)
    omega_ssq = np.zeros_like(omega)
    mpo_ssq = driver.get_qc_eph_mpo(h1e_ssq, eri_ssq, gmat_ssq, omega_ssq)        
        
    mpo_nph = driver.get_qc_eph_mpo(
        np.zeros_like(h1e_ssq), np.zeros_like(eri_ssq),
        np.zeros_like(gmat_ssq), np.ones_like(omega_ssq)
    )

    # summary of the observables
    
    for i in range(5):
        ket_i = driver.split_mps(ket, iroot=i, tag=f"iroot{i}")

        ssq = driver.expectation(ket_i, mpo_ssq, ket_i)
        nph = driver.expectation(ket_i, mpo_nph, ket_i)

        print(" == SUMMARY BL {:.1f} ID {:.0f} ENER {:.10f} SSQ {:.1f} NPH {:.10f} == ".format(bl, i, energies[i], ssq, nph))
        
        data["multiplicities"].append(ssq)
        data["nphotons"].append(nph)
        
    return data

print(get_data(bl=3.2))
