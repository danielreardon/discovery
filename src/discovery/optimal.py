import functools
import warnings

from . import matrix

import jax

# these versions of the ORFs take only the angle, z = matrix.jnp.dot(pos1, pos2).
# They are elementwise, and at zero separation (z == 1) they return the
# cross-correlation limit plus a pulsar term: 0.5 + 0.5 for HD, and a small
# 1e-6 for dipole/monopole so the autocorrelation stays distinguishable.

def hd_orfa(z):
    omc2 = 0.5 * (1.0 - matrix.jnp.clip(matrix.jnp.asarray(z), -1.0, 1.0))
    safe = matrix.jnp.where(omc2 > 0.0, omc2, 1.0)   # keep log finite at z == 1
    hd = 1.5 * omc2 * matrix.jnp.log(safe) - 0.25 * omc2 + 0.5
    return matrix.jnp.where(omc2 > 0.0, hd, 1.0)

def dipole_orfa(z):
    z = matrix.jnp.clip(matrix.jnp.asarray(z), -1.0, 1.0)
    return matrix.jnp.where(z < 1.0, z, 1.0 + 1.0e-6)

def monopole_orfa(z):
    z = matrix.jnp.clip(matrix.jnp.asarray(z), -1.0, 1.0)
    return matrix.jnp.where(z < 1.0, 1.0, 1.0 + 1.0e-6)


def make2d(array):
    return matrix.jnp.diag(array) if array.ndim == 1 else array


# Warn when a pair overlap is within this many error-lengths of zero, where the
# error length is the pair's own negative Frobenius mass.
_OVERLAP_MARGIN = 1.0


@jax.custom_jvp
def _psd(S):
    """Symmetrise ``S`` and clip its negative eigenvalues to zero.

    Returns the nearest positive semi-definite matrix to ``S`` in Frobenius
    norm. Applied once in ``kernelsolves``, so every consumer shares one ``S``.
    """
    S = 0.5 * (S + S.T)
    w, V = matrix.jnp.linalg.eigh(S)
    return (V * matrix.jnp.maximum(w, 0.0)) @ V.T


@_psd.defjvp
def _psd_jvp(primals, tangents):
    """Exact derivative of the clip, avoiding eigh's 1/(w_i - w_j).

    The Frechet derivative of S -> V max(w,0) V^T is

        d(_psd)(S)[E] = V (L o (V^T E V)) V^T,
        L_ij = (max(w_i,0) - max(w_j,0)) / (w_i - w_j)  in [0, 1],

    with L_ii = 1 if w_i > 0 else 0. The divided difference is bounded, so it
    is finite at repeated eigenvalues.
    """
    (S,), (dS,) = primals, tangents
    w, V = matrix.jnp.linalg.eigh(0.5 * (S + S.T))
    wc = matrix.jnp.maximum(w, 0.0)
    dw = w[:, matrix.jnp.newaxis] - w[matrix.jnp.newaxis, :]
    nz = dw != 0.0
    L = matrix.jnp.where(nz,
                         (wc[:, matrix.jnp.newaxis] - wc[matrix.jnp.newaxis, :])
                         / matrix.jnp.where(nz, dw, 1.0),
                         (w > 0.0).astype(w.dtype))
    return ((V * wc) @ V.T,
            V @ (L * (V.T @ (0.5 * (dS + dS.T)) @ V)) @ V.T)


def _pair_overlaps(Ds):
    """All pair overlaps ``bs_ij = tr(D_i D_j)`` as one Gram matrix.

    ``D`` is symmetric, so ``tr(D_i D_j) = sum(D_i * D_j)``; stacking the
    flattened ``D`` gives every pair as ``G G^T``. ``||D_i||_F**2`` is the
    diagonal.
    """
    G = matrix.jnp.stack([D.reshape(-1) for D in Ds])
    return G @ G.T


def _ridge(S):
    """Relative nudge added to the diagonal so ``cholesky`` is defined.

    ``1e-12 * max|diag S|``, so invariant under ``S -> lambda S``. Never
    returns exactly zero.
    """
    scale = matrix.jnp.max(matrix.jnp.abs(matrix.jnp.diag(S)))
    tiny = matrix.jnp.finfo(matrix.jnp.asarray(S).dtype).tiny
    return matrix.jnp.where(scale > 0.0, 1e-12 * scale, tiny)


# Detection statistics of van Haasteren et al. 2025 (arXiv:2509.06489), all
# quadratic forms chi^T Q chi on the same whitened data. DFCC is the traditional
# OS S/N; DF adds the auto-correlation blocks; NP applies the Neyman-Pearson
# filter (I + B)^-1 B, which reintroduces auto-correlation through the inverse;
# NPMV strips those blocks back out.
_DSTYPES = ('dfcc', 'df', 'np', 'npmv')


def _check_dstype(dstype):
    ds = str(dstype).lower()
    if ds not in _DSTYPES:
        raise ValueError(f"unknown detection statistic {dstype!r}, expected one of {_DSTYPES}.")
    return ds


def _require_1d_phi(Phi, name):
    """Reject a 2-D GW Phi where an elementwise sqrt(Phi) is assumed."""
    if matrix.jnp.asarray(Phi).ndim > 1:
        raise NotImplementedError(
            f"OS.{name} assumes a diagonal (1-D) GW prior, but Phi is 2-D (e.g. "
            "makegp_fftcov). Use OS.os, OS.os_rhosigma or OS.mcos instead.")
    return Phi


class OS:
    def __init__(self, gbl):
        # list() so a generator gbl.psls is not consumed by the comprehensions below
        self.psls = list(gbl.psls)

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
        # concrete, not a cached property: must not first be built inside a jit trace
        self.angles = matrix.jnparray([matrix.jnp.dot(self.pos[i], self.pos[j])
                                       for (i, j) in self.pairs])

        # built eagerly, for the same reason as self.angles
        self._kernelsolves_raw = self._build_kernelsolves()

    def invalidate(self):
        """Drop cached kernel solves. Call after replacing ``gbl.residuals``.

        Clears every ``cached_property`` on the class and rebuilds the kernel
        solves eagerly.
        """
        for name, value in vars(type(self)).items():
            if isinstance(value, functools.cached_property):
                self.__dict__.pop(name, None)

        self._kernelsolves_raw = self._build_kernelsolves()

    def validate(self, params, margin=_OVERLAP_MARGIN):
        """Check the two model assumptions the OS cannot enforce inside a trace.

        1. Every pulsar's GW ``Phi`` is the same vector. ``os_rhosigma`` uses
           pulsar 0's for every pair, so a per-pulsar Tspan or a differing
           template is silently wrong.
        2. Every pair overlap ``bs_ij = tr(D_i D_j)`` is positive, where
           ``D_i = diag(sqrt(Phi)) S_i diag(sqrt(Phi))``. The OS divides by
           ``sqrt(bs_ij)``, so a non-positive overlap makes ``os()`` NaN.

        Reads the raw ``S``, before ``_psd``. Not called automatically.

        :param params: parameter dict at which to check.
        :param margin: warn when ``cos_ij / (eta_i + eta_j)`` falls below this
                       for any pair.
        :returns: dict with ``min_bs``, ``min_cos``, ``worst_pair``, ``margin``
                  (the smallest such ratio observed)
                  and ``eta`` (per-pulsar relative negative Frobenius mass).
        :raises ValueError: if a Phi differs, or if any ``bs_ij <= 0``.
        """
        import numpy as np

        phis = [np.asarray(gw.Phi.getN(params)) for gw in self.gws]
        bad = [i for i, ph in enumerate(phis[1:], start=1)
               if ph.shape != phis[0].shape
               or not np.allclose(ph, phis[0], rtol=1e-12, atol=0)]
        if bad:
            raise ValueError(
                f"the GW Phi of pulsar(s) {bad} differs from pulsar 0's, but the OS "
                "uses pulsar 0's Phi for every pair. Give every gw GP the same "
                "template and the same Tspan.")

        sPhi = np.sqrt(_require_1d_phi(phis[0], 'validate'))
        S = np.array([np.asarray(k(params)[1]) for k in self._kernelsolves_raw])
        D = sPhi[None, :, None] * (0.5 * (S + np.swapaxes(S, 1, 2))) * sPhi[None, None, :]

        bs = np.asarray(_pair_overlaps(D))
        nrm = np.sqrt(np.abs(np.diag(bs)))
        cos = bs / np.outer(nrm, nrm)

        # eta_i is the relative negative Frobenius mass of D_i, zero for a PSD S_i
        w = np.linalg.eigvalsh(D)
        eta = (np.sqrt((np.minimum(w, 0.0) ** 2).sum(axis=1))
               / np.sqrt((w ** 2).sum(axis=1)))

        i, j = np.triu_indices(len(D), 1)
        cij, bij, esum = cos[i, j], bs[i, j], eta[i] + eta[j]
        ratio = np.where(esum > 0.0, cij / np.where(esum > 0.0, esum, 1.0), np.inf)
        worst = int(np.argmin(ratio))
        pair = (int(i[worst]), int(j[worst]))

        if bij.min() <= 0.0:
            k = int(np.argmin(bij))
            raise ValueError(
                f"{int((bij <= 0).sum())} of {len(bij)} pulsar pairs have a "
                f"non-positive overlap bs_ij = tr(D_i D_j); worst is pair "
                f"{(int(i[k]), int(j[k]))} at {bij[k]:.3e}. The OS divides by "
                "sqrt(bs_ij), so os() is NaN without the _psd projection. An "
                f"indefinite S (negative Frobenius mass up to {eta.max():.2e}) has "
                f"flipped a nearly orthogonal pair (overlap down to {cij.min():.2e}).")

        if ratio[worst] < margin:
            warnings.warn(
                f"OS: pair {pair} has a normalised overlap {cij[worst]:.2e}, only "
                f"{ratio[worst]:.2f} error-lengths from zero (that pair's negative "
                f"Frobenius mass is {esum[worst]:.2e}). The sign of bs_ij is not "
                "resolved, so sigma_ij for this pair is unreliable and a small "
                "parameter change can make os() NaN. Steeper gw_gamma makes this "
                "worse.", RuntimeWarning, stacklevel=2)

        return {'min_bs': float(bij.min()), 'min_cos': float(cij.min()),
                'worst_pair': pair, 'margin': float(ratio[worst]),
                'eta': eta, 'phi_consistent': True}

    @functools.cached_property
    def params(self):
        return self.os_rhosigma.params

    def _build_kernelsolves(self):
        """Unprojected kernel solves, built eagerly (see ``invalidate``)."""
        return [psl.N.make_kernelsolve(psl.y, gw.F)
                for psl, gw in zip(self.psls, self.gws)]

    @functools.cached_property
    def kernelsolves(self):
        """Per-pulsar ``k(params) -> (T^T K^-1 y, T^T K^-1 T)`` with ``T = gw.F``.

        Each kernel supplies its own Woodbury reduction, so the whole nested
        kernel is marginalised, including constant GPs. ``S`` is PSD-projected
        here, once, so every consumer shares it; ``validate`` reads
        ``_kernelsolves_raw`` instead.
        """
        def wrap(k):
            def kernelsolve(params):
                kv, km = k(params)
                return kv, _psd(km)
            kernelsolve.params = k.params
            return kernelsolve

        return [wrap(k) for k in self._kernelsolves_raw]

    @functools.cached_property
    def _whitened(self):
        """``get(params, orf, dstype) -> (Q, chi)`` with ``chi^T Q chi`` the statistic.

        ``chi_a = A_a^-1 T^T K^-1 y`` where ``S_a = A_a A_a^T``, so ``chi`` is
        standard normal under the null. Each ``Q`` is scaled to unit null
        variance, ``2 tr Q^2 == 1``. The whitening is fixed only up to a
        per-pulsar rotation, under which every dstype is invariant.
        """
        kernelsolves = self.kernelsolves

        Phivar = self.psls[0].gw.Phi.getN

        def get(params, orf=hd_orfa, dstype='dfcc'):
            dstype = _check_dstype(dstype)
            sPhi = matrix.jnp.sqrt(_require_1d_phi(Phivar(params), 'Q'))

            ks = [k(params) for k in kernelsolves]
            Ss = [k[1] for k in ks]
            As = [matrix.jnp.linalg.cholesky(S + _ridge(S) * matrix.jnp.eye(S.shape[0]))
                  for S in Ss]
            chi = matrix.jnp.concatenate(
                [jax.scipy.linalg.solve_triangular(A, k[0], lower=True) for A, k in zip(As, ks)])

            npsr, ngw = len(Ss), Ss[0].shape[0]
            cnt = npsr * ngw
            inds = [slice(i * ngw, (i + 1) * ngw) for i in range(npsr)]

            orfs = orf(self.angles)

            Q = matrix.jnpzeros((cnt, cnt))

            A_scaled = [sPhi[:, None] * A for A in As]

            for w, (i, j) in zip(orfs, self.pairs):
                Bij = w * (A_scaled[i].T @ A_scaled[j])

                Q = Q.at[inds[i], inds[j]].add(Bij)
                Q = Q.at[inds[j], inds[i]].add(Bij.T)

            if dstype == 'dfcc':
                Ds = [sPhi[:,matrix.jnp.newaxis] * S * sPhi[matrix.jnp.newaxis,:] for S in Ss]
                # Ds are symmetric, so tr(Ds[i] @ Ds[j]) == sum(Ds[i] * Ds[j]) (O(m^2), no m x m temporary)
                bs = [matrix.jnp.sum(Ds[i] * Ds[j]) for (i,j) in self.pairs]

                # the loop above adds both Bij and Bij.T, hence the 2 here
                denom = 2.0 * matrix.jnp.sqrt(matrix.jnp.sum(orfs**2 * matrix.jnparray(bs)))

                return Q / denom, chi

            # auto blocks included: I + B is the data covariance under the signal hypothesis
            wauto = orf(matrix.jnparray([1.0]))[0]
            for i in range(npsr):
                Q = Q.at[inds[i], inds[i]].add(wauto * (A_scaled[i].T @ A_scaled[i]))

            if dstype in ('np', 'npmv'):
                Q = matrix.jnp.linalg.solve(matrix.jnp.eye(cnt) + Q, Q)
                Q = 0.5 * (Q + Q.T)

                if dstype == 'npmv':
                    # zero the per-pulsar diagonal blocks the inverse reintroduces
                    blk = matrix.jnp.repeat(matrix.jnp.arange(npsr), ngw)
                    Q = matrix.jnp.where(blk[:, matrix.jnp.newaxis] != blk[matrix.jnp.newaxis, :],
                                         Q, 0.0)

            return Q / matrix.jnp.sqrt(2.0 * matrix.jnp.sum(Q * Q)), chi
        get.params = self.os_rhosigma.params

        return get

    @functools.cached_property
    def Q(self):
        """``get_Q(params, orf, dstype) -> Q`` such that ``x^T Q x`` is the statistic.

        The null distribution of ``x^T Q x`` for standard-normal ``x`` is the
        generalized chi-squared that ``gx2cdf`` integrates. See ``_DSTYPES``.
        """
        whitened = self._whitened

        def get_Q(params, orf=hd_orfa, dstype='dfcc'):
            return whitened(params, orf, dstype)[0]
        get_Q.params = self.os_rhosigma.params

        return get_Q

    @functools.cached_property
    def dstat(self):
        """``get_dstat(params, orf, dstype) -> dict`` for one detection statistic.

        ``snr`` is the statistic, scaled to unit variance under the null.
        ``os`` and ``os_sigma`` are NaN for every dstype but DFCC, the only one
        that estimates an amplitude. Goes through the dense ``Q``, costing
        O((npsr*ngw)^3) against the O(npsr^2 ngw^2) of ``os``.
        """
        whitened, gwpar = self._whitened, self.gwpar

        def get_dstat(params, orf=hd_orfa, dstype='dfcc'):
            Q, chi = whitened(params, orf, dstype)
            snr = chi @ Q @ chi

            # same keys as os(); no string entries, so the dict stays vmappable
            nan = matrix.jnp.nan * snr
            return {'os': nan, 'os_sigma': nan, 'snr': snr, 'log10_A': params[gwpar]}
        get_dstat.params = self._whitened.params

        return get_dstat

    @functools.cached_property
    def opQ(self):
        """``get_opQ(params, orf) -> op``, the matrix-free form of ``Q``.

        DFCC only: NP and NPMV invert ``I + B``, which needs the whole matrix.
        """
        kernelsolves = self.kernelsolves

        Phivar = self.psls[0].gw.Phi.getN

        def get_opQ(params, orf=hd_orfa):
            sPhi = matrix.jnp.sqrt(_require_1d_phi(Phivar(params), 'opQ'))

            Ss = [k(params)[1] for k in kernelsolves]
            As = [matrix.jnp.linalg.cholesky(S + _ridge(S) * matrix.jnp.eye(S.shape[0]))
                  for S in Ss]

            ngw = Ss[0].shape[0]
            inds = [slice(i * ngw, (i + 1) * ngw) for i in range(len(Ss))]

            Ds = [sPhi[:,matrix.jnp.newaxis] * S * sPhi[matrix.jnp.newaxis,:] for S in Ss]
            bs = [matrix.jnp.sum(Ds[i] * Ds[j]) for (i,j) in self.pairs]  # Ds symmetric: tr(A@B)==sum(A*B)

            orfs = orf(self.angles)
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

            Ss = [k(params)[1] for k in kernelsolves]
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

            orfs = orf(self.angles)

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

        Ss = [k(params)[1] for k in self.kernelsolves]
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

            orfs = orf(self.angles)

            os = matrix.jnp.sum(rhos * orfs / sigmas**2) / matrix.jnp.sum(orfs**2 / sigmas**2)
            os_sigma = 1.0 / matrix.jnp.sqrt(matrix.jnp.sum(orfs**2 / sigmas**2))
            snr = os / os_sigma

            return snr
        xs2snrs.cnt = cnt

        return xs2snrs

    def sample_rhosigma(self, params, orf=hd_orfa):
        """``xs2snrs(xs) -> snr`` from TOA-space normals, drawing y directly.

        Rebuilds a Woodbury by hand rather than using ``kernelsolves``, and so
        carries the restrictions below. Prefer ``sample`` or
        ``sample_rhosigma_lowrank``.
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

        # (2) psl.N.F / psl.N.P_var describe only the outermost Woodbury layer
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

        # (3) a variable white noise leaves white_noise_matrix a callable
        bad = [i for i, Nmat in enumerate(Nmats) if callable(Nmat)]
        if bad:
            raise NotImplementedError(
                f"OS.sample_rhosigma needs a constant white noise, but pulsar(s) {bad} "
                "still have free efac/equad parameters. Use OS.sample or "
                "OS.sample_rhosigma_lowrank, which go through make_kernelsolve.")

        Ks = [matrix.WoodburyKernel_novar(matrix.NoiseMatrix1D_novar(Nmat), Fmat, matrix.NoiseMatrix1D_novar(Pmat))
              for Nmat, Fmat, Pmat in zip(Nmats, Fmats, Pmats)]
        K1s = [K.make_solve_1d() for K in Ks]

        TtKmTs = [_psd(Tmat.T @ K.solve_2d(Tmat)[0]) for K, Tmat in zip(Ks, Tmats)]   # not from kernelsolves
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

            orfs = orf(self.angles)

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
        """``get_os(params, orf, dstype) -> dict`` with ``os`` (A^2), ``os_sigma`` and ``snr``.

        ``dstype`` selects the detection statistic; the default DFCC is the
        traditional OS. The other three delegate to ``dstat`` and return the
        same keys with ``os`` and ``os_sigma`` NaN.
        """
        os_rhosigma = self.os_rhosigma    # getos will close on os_rhosigma
        gwpar, angles = self.gwpar, self.angles
        dstat = self.dstat

        def get_os(params, orf=hd_orfa, dstype='dfcc'):
            if _check_dstype(dstype) != 'dfcc':
                return dstat(params, orf, dstype)

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
        gwpar, angles = self.gwpar, self.angles

        def get_mcos(params, orfs=(hd_orfa,)):
            rhos, sigmas = os_rhosigma(params)

            gwnorm = 10**(2.0 * params[gwpar])
            rhos, sigmas = gwnorm * rhos, gwnorm * sigmas

            # design matrix M_{pk} = orf_k(angle_p); weights w_p = 1/sigma_p^2.
            # angles is concrete, so the rank check below also runs under jit/vmap.
            with jax.ensure_compile_time_eval():
                M = matrix.jnp.stack([orf(angles) for orf in orfs], axis=1)
                rank = int(matrix.jnp.linalg.matrix_rank(M))

            if rank < M.shape[1]:
                names = [getattr(orf, '__name__', str(orf)) for orf in orfs]
                raise ValueError(
                    f"the ORFs {names} are collinear over these {M.shape[0]} pulsar "
                    f"pairs (design matrix has rank {rank} < {M.shape[1]}), so the GLS "
                    "normal matrix is singular. Drop a component, or use pulsars whose "
                    "angular separations tell the ORFs apart.")

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
        gwpar, pairs, npsr = self.gwpar, self.pairs, len(self.pos)

        def get_scramble(params, pos, orf=hd_orfa):
            # positions must be unit vectors, one per pulsar
            pos = matrix.jnparray(pos)
            if pos.shape != (npsr, 3):
                raise ValueError(f"scramble needs one unit position vector per pulsar, "
                                 f"i.e. shape ({npsr}, 3), got {pos.shape}.")
            pos = pos / matrix.jnp.linalg.norm(pos, axis=-1, keepdims=True)

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

            # jnp.asarray, not matrix.jnparray: the numerator stays complex
            return (matrix.jnp.asarray(ts) / matrix.jnparray(bs)[:,matrix.jnp.newaxis],
                    1.0 / matrix.jnp.sqrt(matrix.jnparray(bs)))

        get_rhosigma_complex.params = sorted(set.union(*[set(k.params) for k in kernelsolves], getN.params))

        return get_rhosigma_complex

    @functools.cached_property
    def shift(self):
        os_rhosigma_complex = self.os_rhosigma_complex    # getos will close on os_rhosigma
        gwpar, pairs, angles = self.gwpar, self.pairs, self.angles

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

    def gx2cdf(self, params, osxs, *, orf=hd_orfa, dstype='dfcc',
               cutoff=1e-6, limit=100, epsabs=1e-9):
        """Null CDF P(S/N <= osx) for each osx in ``osxs``; p-value is 1 - this.

        :param osxs: S/N values (``os['snr']``), not OS amplitudes.
        :param orf: keyword-only. Must match the ORF used for the point
                    estimate; the null depends on it.
        :param dstype: must match the dstype used for the point estimate.
        :param cutoff, limit, epsabs: passed to ``eig2cdf``.
        """
        eigx = matrix.jnp.linalg.eigh(self.Q(params, orf=orf, dstype=dstype))[0]

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

def eig2cdf(osxs, eigs, cutoff=1e-6, limit=100, epsabs=1e-9):
    """Imhof CDF P(sum_j eig_j z_j^2 <= osx) for standard normal z.

    :param cutoff: if >= 1, keep that many eigenvalues by largest magnitude;
                   otherwise drop those below ``cutoff * max|eig|``.
    :param limit: maximum number of quadrature subintervals.
    :param epsabs: absolute quadrature tolerance.
    """
    # imported lazily: quadax is an optional dependency, and this module is
    # star-imported by discovery/__init__
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
        integral, info = quadax.quadgk(imhof, [0.0, matrix.jnp.inf], args=(osx, eigs),
                                       epsabs=epsabs, max_ninter=limit)
        return 0.5 - integral / matrix.jnp.pi, info.status, info.err

    vals, status, err = jax.vmap(cdf)(matrix.jnparray(osxs))

    # warn on a CDF outside [0, 1], which is the observable symptom of a
    # quadrature that did not converge
    import numpy as _np
    raw = _np.asarray(vals)
    bad = (raw < -1e-6) | (raw > 1.0 + 1e-6)
    if bad.any():
        warnings.warn(
            f"eig2cdf: the quadrature did not converge for {int(bad.sum())} of "
            f"{bad.size} value(s): the CDF came out at {float(raw[bad][0]):.6f}, outside [0, 1]. "
            "The result is clipped to [0, 1] but is not trustworthy -- this happens "
            "when few eigenvalues survive `cutoff`, where the integrand has a heavy "
            "tail. Keep more eigenvalues, or use a Monte Carlo of x^T Q x.",
            RuntimeWarning, stacklevel=2)

    # a CDF that rounds to exactly 1.0 gives p = 0, which is not distinguishable
    # from an infinitely significant detection
    exhausted = (raw >= 1.0) & ~bad
    if exhausted.any():
        warnings.warn(
            f"eig2cdf: {int(exhausted.sum())} of {exhausted.size} value(s) sit further "
            "into the tail than the quadrature resolves, so the CDF rounded to 1 and the "
            f"p-value to 0. Read those as p < ~1e-9 at epsabs={epsabs:g}, not as p == 0.",
            RuntimeWarning, stacklevel=2)

    return matrix.jnp.clip(vals, 0.0, 1.0)
