#!/usr/bin/env python

from functools import reduce
import numpy
import scipy.linalg
from pyscf.fci import cistring
from pyscf.fci import rdm
from pyscf.fci import direct_spin1
from pyscf.fci import spin_op
from pyscf.fci import addons

# JPCA, 118, 9925, eq 10


def make_dm12(ci0, norb, nelec):
    def des(ci0, norb, nelec, ap_id, spin):
        if spin == 0:
            ci1 = addons.des_a(ci0, norb, nelec, ap_id)
            ne = nelec[0]-1, nelec[1]
        else:
            ci1 = addons.des_b(ci0, norb, nelec, ap_id)
            ne = nelec[0], nelec[1]-1
        return ci1, ne
    def cre(ci0, norb, nelec, ap_id, spin):
        if spin == 0:
            ci1 = addons.cre_a(ci0, norb, nelec, ap_id)
            ne = nelec[0]+1, nelec[1]
        else:
            ci1 = addons.cre_b(ci0, norb, nelec, ap_id)
            ne = nelec[0], nelec[1]+1
        return ci1, ne

    dm1 = numpy.zeros((norb,2,norb,2))
    for i in range(norb):
        for j in range(norb):
            for i1 in range(2):
                for j1 in range(2):
                    if i1 == j1:
                        ne = nelec
                        ci1, ne = des(ci0, norb, ne, j, j1)
                        ci1, ne = cre(ci1, norb, ne, i, i1)
                        dm1[i,i1,j,j1] = numpy.dot(ci0.ravel(), ci1.ravel())

    dm2 = numpy.zeros((norb,2,norb,2,norb,2,norb,2))
    for i in range(norb):
        for j in range(norb):
            for k in range(norb):
                for l in range(norb):
                    for i1 in range(2):
                        for j1 in range(2):
                            for k1 in range(2):
                                for l1 in range(2):
                                    if i1 + j1 == k1 + l1:
                                        ci1, ne = ci0, nelec
                                        ci1, ne = des(ci1, norb, ne, k, k1)
                                        ci1, ne = des(ci1, norb, ne, l, l1)
                                        ci1, ne = cre(ci1, norb, ne, j, j1)
                                        ci1, ne = cre(ci1, norb, ne, i, i1)
                                        dm2[i,i1,j,j1,k,k1,l,l1] = numpy.dot(ci0.ravel(), ci1.ravel())
    if 0:
        dm1a = numpy.einsum('iajb->ij', dm1)
        dm2a = numpy.einsum('iajbkalb->ijkl', dm2)
        print abs(numpy.einsum('ipjp->ij', dm2a)/(sum(nelec)-1) - dm1a).sum()
        (dm1a, dm1b), (dm2aa, dm2ab, dm2bb) = \
                direct_spin1.make_rdm12s(ci0, norb, nelec)
        print abs(dm1a - dm1[:,0,:,0]).sum()
        print abs(dm2aa - dm2[:,0,:,0,:,0,:,0].transpose(0,2,1,3)).sum()
        print abs(dm2ab - dm2[:,0,:,1,:,0,:,1].transpose(0,2,1,3)).sum()
        print abs(dm2ab.transpose(2,3,0,1) - dm2[:,1,:,0,:,1,:,0].transpose(0,2,1,3)).sum()
        print abs(dm2bb - dm2[:,1,:,1,:,1,:,1].transpose(0,2,1,3)).sum()
        dm2baab = spin_op.make_rdm2_baab(ci0, norb, nelec)
        dm2abba = spin_op.make_rdm2_abba(ci0, norb, nelec)
        print abs(dm2baab - dm2[:,1,:,0,:,0,:,1].transpose(0,2,1,3)).sum()
        print abs(dm2abba - dm2[:,0,:,1,:,1,:,0].transpose(0,2,1,3)).sum()
    return dm1, dm2


# evaluate all on orthogonal basis
def dbg_ss_frac(fcivec, norb, nelec, mo_coeff, ovlp):
    dm1, dm2 = make_dm12(fcivec, norb, nelec)
    sinv = numpy.eye(norb)  # on orthogonal basis
    # Note JPCA, 118, 9925, eq 8 may be wrong
    dm2tilde = dm2 + numpy.einsum('jbkc,il,ad->iajbkcld', dm1, sinv, numpy.eye(2))

    sigmax = numpy.zeros((2,2))
    sigmay = numpy.zeros((2,2), dtype=numpy.complex128)
    sigmaz = numpy.zeros((2,2))
    sigmax[0,1] = sigmax[1,0] = 1
    sigmay[0,1] = -1j; sigmay[1,0] = 1j
    sigmaz[0,0] = 1; sigmaz[1,1] = -1
    sigma = numpy.array((sigmax,sigmay,sigmaz))

    sdots = numpy.einsum('xca,xdb->abcd', sigma, sigma).real * .25
    d2 = numpy.einsum('abcd,iajbkcld->ijkl', sdots, dm2tilde)

    def eval_ss_frac(range_A, range_B):
        s_Ax = numpy.zeros_like(ovlp); s_Ax[range_A] = ovlp[range_A]
        s_Bx = numpy.zeros_like(ovlp); s_Bx[range_B] = ovlp[range_B]
        s_Ax = reduce(numpy.dot, (mo_coeff.T, s_Ax, mo_coeff))
        s_Bx = reduce(numpy.dot, (mo_coeff.T, s_Bx, mo_coeff))
        s_xA = s_Ax.T
        s_xB = s_Bx.T

        val = numpy.einsum('ki,lj,ijkl', s_Ax, s_Bx, d2)
        val+= numpy.einsum('ki,lj,ijkl', s_xA, s_xB, d2)
        return val * .5
    return eval_ss_frac


def opt_ss_frac(fcivec, norb, nelec, mo_coeff, ovlp):
    (dm1a, dm1b), (dm2aa, dm2ab, dm2bb) = \
            direct_spin1.make_rdm12s(fcivec, norb, nelec, reorder=False)

    dm2baab = spin_op._make_rdm2_baab(fcivec, norb, nelec)
    dm2abba = spin_op._make_rdm2_abba(fcivec, norb, nelec)

    def _bi_trace(dm2, ovlp1, ovlp2):
        return numpy.einsum('jilk,ij,kl->', dm2, ovlp1, ovlp2)

    def eval_ss_frac(range_A, range_B):
        s_Ax = numpy.zeros_like(ovlp); s_Ax[range_A] = ovlp[range_A]
        s_Bx = numpy.zeros_like(ovlp); s_Bx[range_B] = ovlp[range_B]
        s_Ax = reduce(numpy.dot, (mo_coeff.T, s_Ax, mo_coeff))
        s_Bx = reduce(numpy.dot, (mo_coeff.T, s_Bx, mo_coeff))
        s_xA = s_Ax.T
        s_xB = s_Bx.T

        ssz =(_bi_trace(dm2aa, s_Ax, s_Bx)
            - _bi_trace(dm2ab, s_Ax, s_Bx)
            + _bi_trace(dm2bb, s_Ax, s_Bx)
            - _bi_trace(dm2ab.transpose(2,3,0,1), s_Ax, s_Bx)) * .25
        ssz+=(_bi_trace(dm2aa, s_xA, s_xB)
            - _bi_trace(dm2ab, s_xA, s_xB)
            + _bi_trace(dm2bb, s_xA, s_xB)
            - _bi_trace(dm2ab.transpose(2,3,0,1), s_xA, s_xB)) * .25
        ssxy =(_bi_trace(dm2abba, s_Ax, s_Bx)
             + _bi_trace(dm2baab, s_Ax, s_Bx)) * .5
        ssxy+=(_bi_trace(dm2abba, s_xA, s_xB)
             + _bi_trace(dm2baab, s_xA, s_xB)) * .5
        ss = ssxy + ssz
        return ss*.5
    return eval_ss_frac



if __name__ == '__main__':
    norb = 3
    nelec = (2,2)
    numpy.random.seed(10)
    na = cistring.num_strings(norb, nelec[0])
    nb = cistring.num_strings(norb, nelec[1])
    ci0 = numpy.random.random((na,nb))
    ovlp = numpy.random.random((10,10))
    ovlp = numpy.dot(ovlp,ovlp.T)
    mo_coeff = numpy.linalg.inv(scipy.linalg.sqrtm(ovlp))[:,:norb]

    f_dbg = dbg_ss_frac(ci0, norb, nelec, mo_coeff, ovlp)
    f_opt = opt_ss_frac(ci0, norb, nelec, mo_coeff, ovlp)
    print f_dbg(range(2), range(2)), f_opt(range(2), range(2))
    print f_dbg(range(2), range(2,4)), f_opt(range(2), range(2,4))

    # 5 atoms, 2 AO for each
    ss_dbg = 0
    ss_opt = 0
    for iA in range(0, 10, 2):
        for iB in range(0, 10, 2):
            ss_dbg += f_dbg(range(iA,iA+2), range(iB,iB+2))
            ss_opt += f_opt(range(iA,iA+2), range(iB,iB+2))
    ss_tot = spin_op.spin_square(ci0, norb, nelec)[0]
    print(ss_tot, ss_dbg, ss_opt)
