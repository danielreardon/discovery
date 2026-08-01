import functools
import warnings

from . import matrix

import jax

# these versions of the ORFs take only the angle, z = matrix.jnp.dot(pos1, pos2).
# They are elementwise, and at zero separation (z == 1) they return the
# cross-correlation limit plus a pulsar term: 0.5 + 0.5 for HD, and a small
# 1e-6 for dipole/monopole so the autocorrelation stays distinguishable.

def hd_orfa(z):
    omc2 = 0.5 * (1.0 - matrix.jnp.clip(z, -1.0, 1.0))
    safe = matrix.jnp.where(omc2 > 0.0, omc2, 1.0)   # keep log finite at z == 1
    hd = 1.5 * omc2 * matrix.jnp.log(safe) - 0.25 * omc2 + 0.5
    return matrix.jnp.where(omc2 > 0.0, hd, 1.0)

def dipole_orfa(z):
    z = matrix.jnp.clip(z, -1.0, 1.0)
    return matrix.jnp.where(z < 1.0, z, 1.0 + 1.0e-6)

def monopole_orfa(z):
    z = matrix.jnp.clip(z, -1.0, 1.0)
    return matrix.jnp.where(z < 1.0, 1.0, 1.0 + 1.0e-6)


# Relative size of a negative eigenvalue of S that we are willing to call
# round-off. Measured on the 83-pulsar MPTA array: 81 of 83 pulsars have a
# negative eigenvalue, all in the range -4.4e-17 .. -4.3e-14 of the largest,
# i.e. n*eps for n ~ 150. Anything materially larger is not cancellation, it is
# a broken model, and must not be silently repaired.
_PSD_TOL = 1e-10


def _psd(S, warn=True):
    """Symmetrise, and clip round-off-level negative eigenvalues of ``S``.

    ``S = T^T K^-1 T`` is PSD analytically but is formed as a Schur complement,
    so cancellation leaves eigenvalues a few ulp below zero. That is harmless
    for ``bs = tr(D_i D_j)`` -- a contraction over ngw^2 modes, measured
    min bs = 1.7e-21 with the negative part contributing at most 1.3e-13 of it,
    so no ``1/sqrt(bs)`` ever goes NaN -- but it does break ``cholesky``, which
    is why this is applied at the Cholesky call sites and NOT in kernelsolves.

    Keeping it out of kernelsolves matters twice over: the point estimate
    (os_rhosigma / os / mcos) then still sees the raw S, as it always has, and
    it avoids putting an eigendecomposition in the hottest path (measured +63%
    on os() at 83 pulsars).

    ``warn`` reports a clip too large to be cancellation. Without it a genuinely
    wrong S is silently repaired: with S perturbed by 164% the projection
    returns snr = -0.234162 against a true -0.234078, indistinguishable from
    correct, where the unprojected code returned a loud NaN.
    """
    S = 0.5 * (S + S.T)
    w, V = matrix.jnp.linalg.eigh(S)

    if warn:
        wmin, wmax = float(matrix.jnp.min(w)), float(matrix.jnp.max(matrix.jnp.abs(w)))
        if wmax > 0.0 and wmin < -_PSD_TOL * wmax:
            warnings.warn(
                f"OS: S has a negative eigenvalue {wmin:.3e}, which is "
                f"{abs(wmin) / wmax:.2e} of the largest -- far above the {_PSD_TOL:.0e} "
                "expected from Schur-complement cancellation. Clipping it to zero, but "
                "this suggests the kernel or the parameters are wrong, not round-off.",
                RuntimeWarning, stacklevel=2)

    return (V * matrix.jnp.maximum(w, 0.0)) @ V.T


def _ridge(S):
    """Nudge to lift the exact zeros _psd leaves, so cholesky(S) is defined.

    Purely relative, so the result is invariant under ``S -> lambda S`` (a
    change of time units). An earlier version used
    ``1e-12 * maximum(scale, 1.0)``; that absolute floor is unreachable for real
    data (max|diag S| is 1e14-1e15) but it destroyed the scale invariance of
    opQ / sample / sample_rhosigma_lowrank -- a lowrank draw moved 0.822 -> 5519
    as S was rescaled, where the relative form gives 0.822163 at every scale.
    """
    return 1e-12 * matrix.jnp.max(matrix.jnp.abs(matrix.jnp.diag(S)))


def _require_1d_phi(Phi, name):
    """Reject a 2-D GW Phi where an elementwise sqrt(Phi) is assumed."""
    if matrix.jnp.asarray(Phi).ndim > 1:
        raise NotImplementedError(
            f"OS.{name} assumes a diagonal (1-D) GW prior, but Phi is 2-D (e.g. "
            "makegp_fftcov). Use OS.os, OS.os_rhosigma or OS.mcos instead.")
    return Phi


class OS:
    def __init__(self, gbl):
        # list() so a generator gbl.psls is not consumed by the first
        # comprehension below, which used to leave pos and pairs silently empty
        self.psls = list(gbl.psls)

        # before touching self.gws[0], or zero pulsars gives IndexError instead
        if len(self.psls) < 2:
            raise ValueError(f"the OS needs at least two pulsars, got {len(self.psls)}.")

        try:
            self.gws = [psl.gw for psl in self.psls]
            self.pos = [matrix.jnparray(psl.gw.pos) for psl in self.psls]
        except AttributeError:
            raise AttributeError("I cannot find the common GW GP in the pulsar likelihood objects.")

        gwpars = [par for par in self.gws[0].gpcommon if 'log10_A' in par]
        if len(gwpars) != 1:
            raise ValueError("I need exactly one common GW log10_A parameter, found "
                             f"{gwpars}. The OS amplitude rescaling assumes Phi is "
                             "proportional to 10^(2 log10_A).")
        self.gwpar = gwpars[0]

        self.pairs = [(i1, i2) for i1 in range(len(self.pos)) for i2 in range(i1 + 1, len(self.pos))]
        self.angles = [matrix.jnp.dot(self.pos[i], self.pos[j]) for (i,j) in self.pairs]

    def invalidate(self):
        """Drop cached kernel solves. Call after replacing ``gbl.residuals``.

        The cached properties close over ``psl.y``, so without this an OS built
        before a residual swap keeps returning the old numbers.
        """
        for name in ['params', 'kernelsolves', 'Q', 'opQ', 'sample',
                     'os_rhosigma', 'os', 'mcos', 'scramble',
                     'os_rhosigma_complex', 'shift']:
            self.__dict__.pop(name, None)

    @functools.cached_property
    def params(self):
        return self.os_rhosigma.params

    @functools.cached_property
    def kernelsolves(self):
        """Per-pulsar ``k(params) -> (T^T K^-1 y, T^T K^-1 T)`` with ``T = gw.F``.

        Letting each kernel supply its own Woodbury reduction marginalises the
        whole nested kernel, including the constant GPs (SVD timing model, fixed
        ECORR) that ``psl.N.F``/``psl.N.P_var`` do not describe.

        S is returned RAW. The consumers that factorise it (Q, opQ, sample,
        sample_rhosigma_lowrank) call ``_psd`` themselves; os_rhosigma and
        everything built on it use S as the kernel produced it. See ``_psd``
        for why the projection is deliberately not done here.
        """
        return [psl.N.make_kernelsolve(psl.y, gw.F)
                for psl, gw in zip(self.psls, self.gws)]

    @functools.cached_property
    def Q(self):
        """``get_Q(params, orf) -> Q`` such that ``x^T Q x`` is the OS S/N.

        The null distribution of ``x^T Q x`` for standard-normal ``x`` is the
        generalized chi-squared that ``gx2cdf`` integrates.
        """
        kernelsolves = self.kernelsolves

        Phivar = self.psls[0].gw.Phi.getN

        def get_Q(params, orf=hd_orfa):
            sPhi = matrix.jnp.sqrt(_require_1d_phi(Phivar(params), 'Q'))

            # _psd here rather than in kernelsolves: these are the sites that
            # factorise S, so the ridge only has to lift the exact zeros the
            # projection leaves behind
            Ss = [_psd(k(params)[1]) for k in kernelsolves]
            As = [matrix.jnp.linalg.cholesky(S + _ridge(S) * matrix.jnp.eye(S.shape[0]))
                  for S in Ss]

            ngw = Ss[0].shape[0]
            cnt = len(Ss) * ngw
            inds = [slice(i * ngw, (i + 1) * ngw) for i in range(len(Ss))]

            Ds = [sPhi[:,matrix.jnp.newaxis] * S * sPhi[matrix.jnp.newaxis,:] for S in Ss]
            # Ds are symmetric, so tr(Ds[i] @ Ds[j]) == sum(Ds[i] * Ds[j]) (O(m^2), no m x m temporary)
            bs = [matrix.jnp.sum(Ds[i] * Ds[j]) for (i,j) in self.pairs]

            orfs = orf(matrix.jnparray(self.angles))
            # the loop below adds both Bij and Bij.T, hence the 2 here
            denom = 2.0 * matrix.jnp.sqrt(matrix.jnp.sum(orfs**2 * matrix.jnparray(bs)))

            Q = matrix.jnpzeros((cnt, cnt))

            A_scaled = [sPhi[:, None] * A for A in As]

            for w, (i, j) in zip(orfs, self.pairs):
                Bij = w * (A_scaled[i].T @ A_scaled[j])

                Q = Q.at[inds[i], inds[j]].add(Bij)
                Q = Q.at[inds[j], inds[i]].add(Bij.T)

            return Q / denom
        get_Q.params = self.os_rhosigma.params

        return get_Q

    @functools.cached_property
    def opQ(self):
        """``get_opQ(params, orf) -> op``, the matrix-free form of ``Q``."""
        kernelsolves = self.kernelsolves

        Phivar = self.psls[0].gw.Phi.getN

        def get_opQ(params, orf=hd_orfa):
            sPhi = matrix.jnp.sqrt(_require_1d_phi(Phivar(params), 'opQ'))

            # _psd here, not in kernelsolves: these are the sites that factorise S
            Ss = [_psd(k(params)[1]) for k in kernelsolves]
            As = [matrix.jnp.linalg.cholesky(S + _ridge(S) * matrix.jnp.eye(S.shape[0]))
                  for S in Ss]

            ngw = Ss[0].shape[0]
            inds = [slice(i * ngw, (i + 1) * ngw) for i in range(len(Ss))]

            Ds = [sPhi[:,matrix.jnp.newaxis] * S * sPhi[matrix.jnp.newaxis,:] for S in Ss]
            bs = [matrix.jnp.sum(Ds[i] * Ds[j]) for (i,j) in self.pairs]  # Ds symmetric: tr(A@B)==sum(A*B)

            orfs = orf(matrix.jnparray(self.angles))
            # the loop below applies both orderings, hence the 2 here
            denom = 2.0 * matrix.jnp.sqrt(matrix.jnp.sum(orfs**2 * matrix.jnparray(bs)))

            Bs = [sPhi[:, None] * A for A in As]   # B_i = diag(sPhi) @ A_i

            def op(x):
                zs = [B @ x[ii] for B, ii in zip(Bs, inds)]

                y = matrix.jnp.zeros_like(x)
                for w, (i, j) in zip(orfs, self.pairs):
                    y = y.at[inds[i]].add((w / denom) * (Bs[i].T @ zs[j]))
                    y = y.at[inds[j]].add((w / denom) * (Bs[j].T @ zs[i]))

                return y

            return op
        get_opQ.params = self.os_rhosigma.params

        return get_opQ

    @functools.cached_property
    def sample(self):
        """``get_sample(key, params, orf) -> snr``, one draw from the OS null."""
        kernelsolves = self.kernelsolves

        Phivar = self.psls[0].gw.Phi.getN

        def get_sample(key, params, orf=hd_orfa):
            sPhi = matrix.jnp.sqrt(_require_1d_phi(Phivar(params), 'sample'))

            # _psd here, not in kernelsolves: these are the sites that factorise S
            Ss = [_psd(k(params)[1]) for k in kernelsolves]
            As = [matrix.jnp.linalg.cholesky(S + _ridge(S) * matrix.jnp.eye(S.shape[0]))
                  for S in Ss]

            ngw = Ss[0].shape[0]
            cnt = len(Ss) * ngw
            inds = [slice(i * ngw, (i + 1) * ngw) for i in range(len(Ss))]

            Ds = [sPhi[:,matrix.jnp.newaxis] * S * sPhi[matrix.jnp.newaxis,:] for S in Ss]
            bs = [matrix.jnp.sum(Ds[i] * Ds[j]) for (i,j) in self.pairs]  # Ds symmetric: tr(A@B)==sum(A*B)

            xs = matrix.jnpnormal(key, cnt)
            uks = [sPhi * (A @ xs[ind]) for A, ind in zip(As, inds)]

            ts = matrix.jnparray([matrix.jnp.dot(uks[i], uks[j].T) for (i,j) in self.pairs])

            gwnorm = 10**(2.0 * params[self.gwpar])
            rhos = gwnorm * (matrix.jnparray(ts) / matrix.jnparray(bs))
            sigmas = gwnorm / matrix.jnp.sqrt(matrix.jnparray(bs))

            orfs = orf(matrix.jnparray(self.angles))

            os = matrix.jnp.sum(rhos * orfs / sigmas**2) / matrix.jnp.sum(orfs**2 / sigmas**2)
            os_sigma = 1.0 / matrix.jnp.sqrt(matrix.jnp.sum(orfs**2 / sigmas**2))
            snr = os / os_sigma

            return snr
        get_sample.params = self.os_rhosigma.params

        return get_sample

    def sample_rhosigma_lowrank(self, params, orf=hd_orfa):
        """``xs2snrs(xs) -> snr`` from GW-space normals, for batched null draws."""
        Phi = _require_1d_phi(self.psls[0].gw.Phi.getN(params), 'sample_rhosigma_lowrank')
        sPhi = matrix.jnp.sqrt(Phi)

        Ss = [_psd(k(params)[1]) for k in self.kernelsolves]
        As = [matrix.jnp.linalg.cholesky(S + _ridge(S) * matrix.jnp.eye(S.shape[0]))
              for S in Ss]

        Ds = [sPhi[:,matrix.jnp.newaxis] * S * sPhi[matrix.jnp.newaxis,:] for S in Ss]
        bs = [matrix.jnp.sum(Ds[i] * Ds[j]) for (i,j) in self.pairs]  # Ds symmetric: tr(A@B)==sum(A*B)

        inds, cnt = [], 0
        for A in As:
            inds.append(slice(cnt, cnt + A.shape[0])) # these are all the same length, could simplify
            cnt += A.shape[0]

        def xs2snrs(xs):
            uks = [sPhi * (A @ xs[ind]) for A, ind in zip(As, inds)]

            ts = matrix.jnparray([matrix.jnp.dot(uks[i], uks[j].T) for (i,j) in self.pairs])

            gwnorm = 10**(2.0 * params[self.gwpar])
            rhos = gwnorm * (matrix.jnparray(ts) / matrix.jnparray(bs))
            sigmas = gwnorm / matrix.jnp.sqrt(matrix.jnparray(bs))

            orfs = orf(matrix.jnparray(self.angles))

            os = matrix.jnp.sum(rhos * orfs / sigmas**2) / matrix.jnp.sum(orfs**2 / sigmas**2)
            os_sigma = 1.0 / matrix.jnp.sqrt(matrix.jnp.sum(orfs**2 / sigmas**2))
            snr = os / os_sigma

            return snr
        xs2snrs.cnt = cnt

        return xs2snrs

    def sample_rhosigma(self, params, orf=hd_orfa):
        """``xs2snrs(xs) -> snr`` from TOA-space normals, drawing y directly.

        Cannot go through ``kernelsolves``: it needs a generative square root of
        the kernel, which the kernel API does not expose, so it rebuilds a
        Woodbury by hand and inherits that approach's two restrictions below.
        Prefer ``sample`` or ``sample_rhosigma_lowrank``.
        """
        Phi = _require_1d_phi(self.psls[0].gw.Phi.getN(params), 'sample_rhosigma')
        sPhi = matrix.jnp.sqrt(Phi)

        Nmats, Fmats, Pmats, Tmats = zip(*[(psl.white_noise_matrix, psl.N.F, psl.N.P_var.getN(params), psl.gw.F) for psl in self.psls])

        # (1) the elementwise sqrt(Pmat) below is only valid for a diagonal prior
        if any(matrix.jnp.asarray(Pmat).ndim > 1 for Pmat in Pmats):
            raise NotImplementedError(
                "OS.sample_rhosigma assumes a diagonal (1-D) GP prior, but a pulsar has a "
                "2-D prior covariance (e.g. correlated Legendre ECORR). Use OS.os/os_rhosigma/"
                "mcos, or an uncorrelated ECORR model for sampling.")

        # (2) psl.N.F / psl.N.P_var describe only the outermost Woodbury layer, so
        # constant GPs folded into an inner layer would be silently dropped here
        for psl in self.psls:
            # variable-white-noise kernels store the inner layer as N_var, not N
            inner = getattr(psl.N, 'N', None)
            if inner is None:
                inner = getattr(psl.N, 'N_var', None)
            if inner is not None and not isinstance(inner, matrix.NoiseMatrix):
                raise NotImplementedError(
                    "OS.sample_rhosigma cannot handle a nested kernel: this pulsar's psl.N "
                    f"wraps a {type(inner).__name__}, i.e. constant GPs (timing model, fixed "
                    "ECORR) live in an inner Woodbury layer that psl.N.F/psl.N.P_var do not "
                    "describe, and this sampler would drop them. Use OS.sample or "
                    "OS.sample_rhosigma_lowrank, which go through make_kernelsolve.")

        Ks = [matrix.WoodburyKernel_novar(matrix.NoiseMatrix1D_novar(Nmat), Fmat, matrix.NoiseMatrix1D_novar(Pmat))
              for Nmat, Fmat, Pmat in zip(Nmats, Fmats, Pmats)]
        K1s = [K.make_solve_1d() for K in Ks]

        TtKmTs = [_psd(Tmat.T @ K.solve_2d(Tmat)[0]) for K, Tmat in zip(Ks, Tmats)]
        PsTtKmFsPs = [sPhi[:,matrix.jnp.newaxis] * (Tmat.T @ K.solve_2d(Fmat)[0]) * matrix.jnp.sqrt(Pmat)[matrix.jnp.newaxis,:]
                      for K, Tmat, Fmat, Pmat in zip(Ks, Tmats, Fmats, Pmats)]
        PsTts = [sPhi[:,matrix.jnp.newaxis] * Tmat.T for Tmat in Tmats]

        Ds = [sPhi[:,matrix.jnp.newaxis] * TtKmT * sPhi[matrix.jnp.newaxis,:] for TtKmT in TtKmTs]
        bs = [matrix.jnp.sum(Ds[i] * Ds[j]) for (i,j) in self.pairs]  # Ds symmetric: tr(A@B)==sum(A*B)

        cnt, iNs, iPs = 0, [], []
        for Nmat in Nmats:
            iNs.append(slice(cnt, cnt + Nmat.shape[0]))
            cnt += Nmat.shape[0]
        for Fmat in Fmats:
            iPs.append(slice(cnt, cnt + Fmat.shape[1]))
            cnt += Fmat.shape[1]

        def xs2snrs(xs):
            uks = [PsTt @ K1(matrix.jnp.sqrt(Nmat) * xs[iN])[0] + PsTtKmFsP @ xs[iP]
                   for PsTt, K1, Nmat, iN, PsTtKmFsP, iP in zip(PsTts, K1s, Nmats, iNs, PsTtKmFsPs, iPs)]

            # use with matrix.jnpnormal(key, cnt)
            ts = matrix.jnparray([matrix.jnp.dot(uks[i], uks[j].T) for (i,j) in self.pairs])

            gwnorm = 10**(2.0 * params[self.gwpar])
            rhos = gwnorm * (matrix.jnparray(ts) / matrix.jnparray(bs))
            sigmas = gwnorm / matrix.jnp.sqrt(matrix.jnparray(bs))

            orfs = orf(matrix.jnparray(self.angles))

            os = matrix.jnp.sum(rhos * orfs / sigmas**2) / matrix.jnp.sum(orfs**2 / sigmas**2)
            os_sigma = 1.0 / matrix.jnp.sqrt(matrix.jnp.sum(orfs**2 / sigmas**2))
            snr = os / os_sigma

            return snr
        xs2snrs.cnt = cnt

        return xs2snrs

    @functools.cached_property
    def os_rhosigma(self):
        kernelsolves = self.kernelsolves
        getN = self.gws[0].Phi.getN   # use GW prior from first pulsar, assume all GW GP are the same
        pairs = self.pairs

        # kernelsolves give kv_i = T_i* K_i^-1 y_i and km_i = T_i* K_i^-1 T_i;
        # with U* U = Phi, ts_ij = (U kv_i)* (U kv_j) and bs_ij = tr(U km_i U* U km_j U*),
        # so rho_ij = ts_ij / bs_ij and sigma_ij = 1 / sqrt(bs_ij)

        def get_rhosigma(params):
            N = getN(params)
            ks = [k(params) for k in kernelsolves]

            if N.ndim == 1:
                sN = matrix.jnp.sqrt(N)

                ts = [matrix.jnp.dot(sN * ks[i][0], sN * ks[j][0]) for (i,j) in pairs]
                ds = [sN[:,matrix.jnp.newaxis] * k[1] * sN[matrix.jnp.newaxis,:] for k in ks]

                bs = [matrix.jnp.sum(ds[i] * ds[j]) for (i,j) in pairs]  # ds symmetric: tr(A@B)==sum(A*B)
            else:
                U = matrix.jnp.linalg.cholesky(N, upper=True) # N = U^T U, so y = U^T x

                uks = [U @ k[0] for k in ks]
                ds = [U @ k[1] @ U.T for k in ks]

                ts = [matrix.jnp.dot(uks[i], uks[j].T) for (i,j) in pairs]
                bs = [matrix.jnp.sum(ds[i] * ds[j]) for (i,j) in pairs]  # ds symmetric: tr(A@B)==sum(A*B)

                # slower:
                # ts = [matrix.jnp.dot(U @ ks[i][0], U @ ks[j][0]) for (i,j) in pairs]
                # even slower, more explicit:
                # ts = [ks[i][0].T @ N @ ks[j][0] for (i,j) in pairs]

                # more explicit:
                # bs = [matrix.jnp.trace(ks[i][1] @ N @ ks[j][1] @ N) for (i,j) in pairs]

            return (matrix.jnparray(ts) / matrix.jnparray(bs),
                    1.0 / matrix.jnp.sqrt(matrix.jnparray(bs)))

        get_rhosigma.params = sorted(set.union(*[set(k.params) for k in kernelsolves], getN.params))

        return get_rhosigma

    @functools.cached_property
    def os(self):
        os_rhosigma = self.os_rhosigma    # getos will close on os_rhosigma
        gwpar, angles = self.gwpar, matrix.jnparray(self.angles)

        def get_os(params, orf=hd_orfa):
            rhos, sigmas = os_rhosigma(params)

            gwnorm = 10**(2.0 * params[gwpar])
            rhos, sigmas = gwnorm * rhos, gwnorm * sigmas

            orfs = orf(angles)

            os = matrix.jnp.sum(rhos * orfs / sigmas**2) / matrix.jnp.sum(orfs**2 / sigmas**2)
            os_sigma = 1.0 / matrix.jnp.sqrt(matrix.jnp.sum(orfs**2 / sigmas**2))
            snr = os / os_sigma

            return {'os': os, 'os_sigma': os_sigma, 'snr': snr, 'log10_A': params[gwpar]} # , 'rhos': rhos, 'sigmas': sigmas}

        get_os.params = os_rhosigma.params

        return get_os

    @functools.cached_property
    def mcos(self):
        """Multi-component OS: fit several ORFs at once by GLS.

        The returned ``cov`` carries the correlation between components, which
        calling ``os`` once per ORF does not.
        """
        os_rhosigma = self.os_rhosigma    # get_mcos will close on os_rhosigma
        gwpar, angles = self.gwpar, matrix.jnparray(self.angles)

        def get_mcos(params, orfs=(hd_orfa,)):
            rhos, sigmas = os_rhosigma(params)

            gwnorm = 10**(2.0 * params[gwpar])
            rhos, sigmas = gwnorm * rhos, gwnorm * sigmas

            # design matrix M_{pk} = orf_k(angle_p); weights w_p = 1/sigma_p^2
            M = matrix.jnp.stack([orf(angles) for orf in orfs], axis=1)
            w = 1.0 / sigmas**2

            MtW = M.T * w[matrix.jnp.newaxis, :]
            fisher = MtW @ M                              # (ncomp, ncomp)
            cov = matrix.jnp.linalg.inv(fisher)           # parameter covariance
            os = cov @ (MtW @ rhos)                       # amplitude estimates A_k^2
            os_sigma = matrix.jnp.sqrt(matrix.jnp.diag(cov))
            snr = os / os_sigma

            return {'os': os, 'os_sigma': os_sigma, 'cov': cov, 'snr': snr,
                    'log10_A': params[gwpar]}

        get_mcos.params = os_rhosigma.params

        return get_mcos

    @functools.cached_property
    def scramble(self):
        os_rhosigma = self.os_rhosigma    # getos will close on os_rhosigma
        gwpar, pairs = self.gwpar, self.pairs

        def get_scramble(params, pos, orf=hd_orfa):
            rhos, sigmas = os_rhosigma(params)

            gwnorm = 10**(2.0 * params[gwpar])
            rhos, sigmas = gwnorm * rhos, gwnorm * sigmas

            angles = matrix.jnparray([matrix.jnp.dot(pos[i], pos[j]) for (i,j) in pairs])
            orfs = orf(angles)

            os = matrix.jnp.sum(rhos * orfs / sigmas**2) / matrix.jnp.sum(orfs**2 / sigmas**2)
            os_sigma = 1.0 / matrix.jnp.sqrt(matrix.jnp.sum(orfs**2 / sigmas**2))
            snr = os / os_sigma

            return {'os': os, 'os_sigma': os_sigma, 'snr': snr, 'log10_A': params[gwpar]} #, 'rhos': rhos, 'sigmas': sigmas}

        get_scramble.params = os_rhosigma.params

        return get_scramble

    @functools.cached_property
    def os_rhosigma_complex(self):
        kernelsolves = self.kernelsolves
        getN = self.gws[0].Phi.getN
        pairs = self.pairs

        def get_rhosigma_complex(params):
            N = getN(params)
            ks = [k(params) for k in kernelsolves]

            if N.ndim == 2:
                raise NotImplementedError("Complex rhosigma not defined for 2D Phi.")

            sN = matrix.jnp.sqrt(N)

            tsf = [sN[::2] * (k[0][::2] + 1j * k[0][1::2]) for k in ks]
            ts = [tsf[i] * matrix.jnp.conj(tsf[j]) for (i,j) in pairs]

            ds = [sN[:,matrix.jnp.newaxis] * k[1] * sN[matrix.jnp.newaxis,:] for k in ks]
            bs = [matrix.jnp.sum(ds[i] * ds[j]) for (i,j) in pairs]  # ds symmetric: tr(A@B)==sum(A*B)

            # matrix.jnparray forces float64, which would discard the imaginary
            # part and leave `shift` computing Re(ts)cos(dphi) instead of
            # Re(ts e^{i dphi}) -- so use jnp.asarray for the complex numerator
            return (matrix.jnp.asarray(ts) / matrix.jnparray(bs)[:,matrix.jnp.newaxis],
                    1.0 / matrix.jnp.sqrt(matrix.jnparray(bs)))

        get_rhosigma_complex.params = sorted(set.union(*[set(k.params) for k in kernelsolves], getN.params))

        return get_rhosigma_complex

    @functools.cached_property
    def shift(self):
        os_rhosigma_complex = self.os_rhosigma_complex    # getos will close on os_rhosigma
        gwpar, pairs, angles = self.gwpar, self.pairs, matrix.jnparray(self.angles)

        def get_shift(params, phases, orf=hd_orfa):
            rhos_complex, sigmas = os_rhosigma_complex(params)

            # can't use matrix.jnparray or complex will be downcast
            phaseprod = matrix.jnp.array([matrix.jnp.exp(1j * (phases[i] - phases[j])) for i,j in pairs])
            rhos = matrix.jnp.sum(matrix.jnp.real(rhos_complex * phaseprod), axis=1)

            gwnorm = 10**(2.0 * params[gwpar])
            rhos, sigmas = gwnorm * rhos, gwnorm * sigmas

            orfs = orf(angles)

            os = matrix.jnp.sum(rhos * orfs / sigmas**2) / matrix.jnp.sum(orfs**2 / sigmas**2)
            os_sigma = 1.0 / matrix.jnp.sqrt(matrix.jnp.sum(orfs**2 / sigmas**2))
            snr = os / os_sigma

            return {'os': os, 'os_sigma': os_sigma, 'snr': snr, 'log10_A': params[gwpar]} #, 'rhos': rhos, 'sigmas': sigmas}

        get_shift.params = os_rhosigma_complex.params

        return get_shift

    def gx2cdf(self, params, osxs, *, orf=hd_orfa, cutoff=1e-6, limit=100, epsabs=1e-6):
        """Null CDF P(S/N <= osx) for each osx in ``osxs``; p-value is 1 - this.

        ``osxs`` are S/N values (``os['snr']``), not OS amplitudes, and ``orf``
        must match the one used for the point estimate -- the null depends on it
        (a dipole S/N read off the HD null is wrong by 3-4% in p).

        ``orf`` is keyword-only on purpose: it was added after ``osxs``, so
        accepting it positionally would silently reinterpret an existing
        ``gx2cdf(params, osxs, 1e-3)`` cutoff as an ORF.
        """
        eigx = matrix.jnp.linalg.eigh(self.Q(params, orf=orf))[0]

        return eig2cdf(osxs, eigx, cutoff=cutoff, limit=limit, epsabs=epsabs)


@jax.jit
def imhof(u, x, eigs):
    theta = 0.5 * matrix.jnp.sum(matrix.jnp.arctan(eigs * u), axis=0) - 0.5 * x * u
    rho = matrix.jnp.prod((1.0 + (eigs * u)**2)**0.25, axis=0)

    # the integrand has a removable 0/0 singularity at u=0 with finite limit
    # 1/2 (sum(eigs) - x); quadax may sample the lower endpoint, so return the
    # limit there rather than nan
    u0 = 0.5 * (matrix.jnp.sum(eigs, axis=0) - x)
    return matrix.jnp.where(u == 0.0, u0, matrix.jnp.sin(theta) / (u * rho))

def eig2cdf(osxs, eigs, cutoff=1e-6, limit=100, epsabs=1e-6):
    """Imhof CDF P(sum_j eig_j z_j^2 <= osx) for standard normal z."""
    # imported here rather than at module scope so that quadax stays an optional
    # dependency: only gx2cdf needs it, and discovery/__init__ star-imports this
    # module, so a top-level import would make the whole package unimportable
    import quadax

    eigs = matrix.jnp.asarray(eigs)
    if cutoff >= 1:
        # keep that many eigenvalues, largest |.| first. eigh returns them
        # ascending and Q is traceless, so plain slicing would keep only the
        # negative half and discard every positive eigenvalue.
        eigs = eigs[matrix.jnp.argsort(-matrix.jnp.abs(eigs))[:int(cutoff)]]
    else:
        eigs = eigs[matrix.jnp.abs(eigs) > cutoff * matrix.jnp.abs(eigs).max()]

    # quadax.quadgk is a JAX-transformable analog of scipy.integrate.quad,
    # so we can vmap over osxs and keep everything in jax
    def cdf(osx):
        integral = quadax.quadgk(imhof, [0.0, matrix.jnp.inf], args=(osx, eigs),
                                 epsabs=epsabs, max_ninter=limit)[0]
        return 0.5 - integral / matrix.jnp.pi

    return jax.vmap(cdf)(matrix.jnparray(osxs))
