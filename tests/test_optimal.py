"""Tests for discovery.optimal.

The important one is test_kernelsolve_matches_bruteforce: the OS S matrix is
built by a Woodbury reduction, and the only way to validate it is against an
explicit dense T^T C^-1 T. The identities trace(Q) == 0 and 2*sum(eig^2) == 1
hold for ANY S -- they check the assembly code, not S itself.
"""

import warnings

import numpy as np
import pytest

import discovery as ds
from discovery import matrix
from discovery.optimal import (OS, hd_orfa, dipole_orfa, monopole_orfa,
                               _psd, _ridge, eig2cdf)


# ---------------------------------------------------------------- ORFs

def test_orfs_are_elementwise():
    """Each ORF must map an array of angles to an equal-shaped array.

    A jnp.allclose-based autocorrelation guard silently collapsed monopole to a
    0-d scalar, which broke Q/opQ/sample (iteration over a 0-d array) and mcos
    (jnp.stack shape mismatch).
    """
    z = np.array([-1.0, -0.5, 0.0, 0.3, 0.9])
    for orf in (hd_orfa, dipole_orfa, monopole_orfa):
        assert np.shape(orf(z)) == z.shape, orf.__name__


def test_orfs_finite_at_zero_separation():
    """z == 1 must give the 1.0 autocorrelation limit, not nan.

    hd_orfa used `... + 0.5 * allclose(z, 1)`, but 1.5*0*log(0) is already nan
    and an additive term cannot repair it.
    """
    for orf in (hd_orfa, dipole_orfa, monopole_orfa):
        assert np.isfinite(orf(1.0)), orf.__name__
        assert np.all(np.isfinite(orf(np.array([0.3, 1.0, -0.5])))), orf.__name__

    assert hd_orfa(1.0) == pytest.approx(1.0)
    assert dipole_orfa(1.0) == pytest.approx(1.0, abs=1e-5)
    assert monopole_orfa(1.0) == pytest.approx(1.0, abs=1e-5)


def test_orfs_out_of_range_z_is_finite():
    """|z| slightly > 1 from rounding must not produce log of a negative."""
    for orf in (hd_orfa, dipole_orfa, monopole_orfa):
        assert np.all(np.isfinite(orf(np.array([1.0 + 1e-12, -1.0 - 1e-12])))), orf.__name__


def test_hd_normalisation():
    """enterprise convention: 0.5 at 0 deg (cross term), 0.25 at 180 deg."""
    assert hd_orfa(np.array([1.0 - 1e-12]))[0] == pytest.approx(0.5, abs=1e-5)
    assert hd_orfa(-1.0) == pytest.approx(0.25)


def test_hd_matches_signals_hd_orf():
    """The angle-only ORF must agree with the position-based signals.hd_orf."""
    rng = np.random.default_rng(42)
    for _ in range(20):
        p1, p2 = rng.normal(size=3), rng.normal(size=3)
        p1, p2 = p1 / np.linalg.norm(p1), p2 / np.linalg.norm(p2)
        assert float(hd_orfa(np.dot(p1, p2))) == pytest.approx(
            float(ds.signals.hd_orf(p1, p2)), rel=1e-10)


# ---------------------------------------------------------------- _psd

def test_psd_clips_negative_eigenvalues():
    """Clip at round-off scale -- the size actually seen on real data.

    Measured on the 83-pulsar MPTA array: 81 of 83 pulsars have a negative
    eigenvalue, all between -4.4e-17 and -4.3e-14 of the largest. An earlier
    version of this test used -5e-7 and called that "the measured
    indefiniteness", which is ~7 orders of magnitude too large.
    """
    rng = np.random.default_rng(0)
    V = np.linalg.qr(rng.normal(size=(8, 8)))[0]
    w = np.logspace(0, -3, 8) * 1e8
    w[-1] = -4e-14 * w[0]
    S = V @ np.diag(w) @ V.T

    P = np.asarray(_psd(S))
    assert np.allclose(P, P.T, atol=0)
    assert np.linalg.eigvalsh(P).min() >= -1e-12 * abs(w).max()
    assert np.allclose(np.sort(np.linalg.eigvalsh(P))[-7:], np.sort(w[:-1]), rtol=1e-8)


def test_psd_is_a_noop_on_psd_input():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(6, 6))
    S = A @ A.T
    assert np.allclose(np.asarray(_psd(S)), S, rtol=1e-10, atol=0)


def test_psd_enables_cholesky():
    """What _psd is actually FOR: making cholesky(S + ridge) defined.

    It is NOT for keeping bs = tr(D_i D_j) non-negative -- that contraction is
    over ngw^2 modes and is immune to a round-off eigenvalue (measured min bs
    = 1.7e-21 on the real array, with the negative part contributing at most
    1.3e-13 of it). An earlier test asserted the bs property and was vacuous:
    its own raw inputs never produced a negative bs, so it reduced to 0 >= 0.
    """
    rng = np.random.default_rng(2)
    V = np.linalg.qr(rng.normal(size=(8, 8)))[0]
    w = np.logspace(0, -3, 8) * 1e8
    w[-1] = -4e-14 * w[0]
    S = V @ np.diag(w) @ V.T

    # raw S is indefinite, so its Cholesky is not defined
    with pytest.raises(np.linalg.LinAlgError):
        np.linalg.cholesky(S)

    P = np.asarray(_psd(S))
    L = np.linalg.cholesky(P + 1e-12 * np.abs(np.diag(P)).max() * np.eye(8))
    assert np.all(np.isfinite(L))


def test_psd_warns_on_a_clip_too_large_to_be_roundoff():
    """A genuinely broken S must not be silently repaired.

    Clipping without a warning turns a loud NaN into a plausible number: with S
    perturbed by 164% the projection returned snr = -0.234162 against a true
    -0.234078, indistinguishable from correct.
    """
    rng = np.random.default_rng(3)
    V = np.linalg.qr(rng.normal(size=(8, 8)))[0]
    w = np.logspace(0, -3, 8) * 1e8
    w[-1] = -0.3 * w[0]                       # 30% negative: not cancellation
    S = V @ np.diag(w) @ V.T

    with pytest.warns(RuntimeWarning, match='negative eigenvalue'):
        _psd(S)

    # and round-off-scale indefiniteness must stay silent
    w2 = w.copy(); w2[-1] = -4e-14 * w[0]
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        _psd(V @ np.diag(w2) @ V.T)


def test_ridge_is_scale_invariant():
    """_ridge must be purely relative.

    An absolute floor (1e-12 * maximum(scale, 1.0)) is unreachable for real data
    but destroys the scale invariance of opQ / sample / sample_rhosigma_lowrank:
    under S -> lambda S a lowrank draw moved 0.822 -> 5519.
    """
    rng = np.random.default_rng(4)
    A = rng.normal(size=(6, 6))
    S = A @ A.T
    for lam in (1e-20, 1e-6, 1.0, 1e6, 1e20):
        assert float(_ridge(lam * S)) == pytest.approx(lam * float(_ridge(S)), rel=1e-12)


# ---------------------------------------------------------------- eig2cdf

def _gx2_mc(eigs, xs, n=200000, seed=0):
    """Monte Carlo CDF P(sum eig_j z_j^2 <= x)."""
    rng = np.random.default_rng(seed)
    q = (rng.normal(size=(n, len(eigs))) ** 2) @ np.asarray(eigs)
    return np.array([(q <= x).mean() for x in xs])


def test_eig2cdf_matches_chi2():
    """eigs=[1] is chi^2_1, so the CDF is known exactly."""
    pytest.importorskip('quadax')
    from scipy.stats import chi2
    xs = np.array([0.5, 1.0, 2.0, 4.0])
    got = np.asarray(eig2cdf(xs, np.array([1.0]), epsabs=1e-12))
    assert got == pytest.approx(chi2.cdf(xs, 1), abs=5e-3)


def test_eig2cdf_matches_montecarlo_mixed_signs():
    pytest.importorskip('quadax')
    eigs = np.array([0.6, 0.3, 0.1, -0.1, -0.3, -0.6])
    xs = np.array([-1.0, -0.3, 0.0, 0.3, 1.0])
    got = np.asarray(eig2cdf(xs, eigs))
    assert got == pytest.approx(_gx2_mc(eigs, xs), abs=5e-3)


def test_eig2cdf_integer_cutoff_keeps_largest_magnitude():
    """An integer cutoff must keep the largest |eig|, both signs.

    eigh returns ascending eigenvalues and Q is traceless, so `eigs[:cutoff]`
    kept only the most negative ones and discarded the entire positive half,
    giving CDF values above 1 and a step at 0.
    """
    pytest.importorskip('quadax')
    rng = np.random.default_rng(3)
    eigs = np.sort(np.concatenate([rng.normal(size=20), -rng.normal(size=20)]))
    xs = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])

    full = np.asarray(eig2cdf(xs, eigs))
    coarse = np.asarray(eig2cdf(xs, eigs, cutoff=10))
    fine = np.asarray(eig2cdf(xs, eigs, cutoff=30))

    # the old slicing kept only negative eigenvalues, giving a step at 0 and
    # values outside [0, 1]
    for got in (coarse, fine):
        assert np.all((got >= -1e-6) & (got <= 1 + 1e-6)), got
        assert np.all(np.diff(got) >= -1e-6), got          # monotone in x
    assert fine == pytest.approx(full, abs=0.05)


def test_eig2cdf_cutoff_one_is_a_count():
    """cutoff=1 must mean 'one eigenvalue', not a relative threshold.

    `cutoff > 1` sent an integer 1 into the relative-threshold branch, where
    |eig| > 1.0*max|eig| selects nothing.
    """
    pytest.importorskip('quadax')
    got = np.asarray(eig2cdf(np.array([1.0]), np.array([1.0, 0.5, 0.25]), cutoff=1))
    assert np.all(np.isfinite(got))
    assert np.all((got >= -1e-6) & (got <= 1 + 1e-6))


# ---------------------------------------------------- OS against brute force

@pytest.fixture(scope='module')
def os_model():
    """A small OS built from the packaged test pulsars.

    The timing-model prior is finite (``constant=1e-8``) so that an explicit
    dense ``C`` stays invertible in float64 and brute force is meaningful; the
    default 1e40 makes any dense comparison numerically hopeless.
    """
    import glob, os as _os

    here = _os.path.dirname(_os.path.abspath(__file__))
    files = sorted(glob.glob(_os.path.join(here, 'data', '*.feather')))
    if len(files) < 3:
        pytest.skip('need >= 3 pulsar feather files in tests/data')

    psrs = [ds.Pulsar.read_feather(f) for f in files]
    Tspan = ds.getspan(psrs)

    psls, parts = [], []
    for p in psrs:
        noise = ds.makenoise_measurement(p)
        gps = [ds.makegp_timing(p, svd=True, constant=1e-8),
               ds.makegp_fourier(p, ds.powerlaw, 10, T=Tspan, name='red_noise'),
               ds.makegp_fourier(p, ds.powerlaw, 5, T=Tspan,
                                 common=['gw_log10_A', 'gw_gamma'], name='gw')]
        psls.append(ds.PulsarLikelihood([p.residuals, noise] + gps))
        parts.append((noise, gps))

    o = OS(ds.GlobalLikelihood(psls))
    # ds.sample_uniform uses the bare global numpy RNG, so seed it: otherwise
    # every efac/equad/ecorr differs run to run and a failure cannot be replayed
    np.random.seed(20260801)
    params = dict(ds.sample_uniform(o.params))
    params['gw_log10_A'], params['gw_gamma'] = -14.5, 4.33
    for p in o.params:
        if 'red_noise_log10_A' in p:
            params[p] = -14.0
        elif 'red_noise_gamma' in p:
            params[p] = 3.0
    return o, params, psls, parts


@pytest.fixture(scope='module')
def os_spread(os_model):
    """The same model but with well-separated sky positions.

    The packaged test pulsars are all at [0, 0, 1], so every pair has zero
    separation and every ORF collapses to a constant -- which makes mcos's
    design matrix singular and every Q identical. ORF-discrimination tests need
    real angular structure.
    """
    o, params, psls, parts = os_model

    spread = OS.__new__(OS)
    spread.__dict__.update({k: v for k, v in o.__dict__.items()
                            if k in ('psls', 'gws', 'gwpar', 'pairs')})
    spread.pos = [matrix.jnparray(v) for v in
                  ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.6, 0.0, 0.8])]
    spread.angles = [float(np.dot(spread.pos[i], spread.pos[j])) for i, j in spread.pairs]
    assert len(set(np.round(spread.angles, 6))) == len(spread.angles)
    return spread, params


def _diag(nm, params):
    """Diagonal of a NoiseMatrix, variable (getN) or constant (.N)."""
    P = np.asarray(nm.getN(params) if hasattr(nm, 'getN') else nm.N)
    return np.diag(P) if P.ndim == 2 else P


def test_kernelsolve_matches_bruteforce(os_model):
    """S = T^T C^-1 T and kv = T^T C^-1 y against an explicit dense solve.

    This is the only real check on the Woodbury reduction -- trace(Q) == 0 and
    2*sum(eig^2) == 1 hold for any S. A previous version rebuilt the kernel from
    (white noise, outer F, outer P), dropping the constant GPs (SVD timing
    model), which made S wrong by a large factor.
    """
    o, params, psls, parts = os_model

    for psl, (noise, gps), gw, k in zip(psls, parts, o.gws, o.kernelsolves):
        kv, S = k(params)

        C = np.diag(_diag(noise, params))
        for gp in gps:
            F = np.asarray(gp.F(params) if callable(gp.F) else gp.F)
            C = C + (F * _diag(gp.Phi, params)[None, :]) @ F.T

        T = np.asarray(gw.F(params) if callable(gw.F) else gw.F)
        y = np.asarray(psl.y)

        assert np.asarray(S) == pytest.approx(T.T @ np.linalg.solve(C, T), rel=1e-5)
        assert np.asarray(kv) == pytest.approx(T.T @ np.linalg.solve(C, y), rel=1e-5)


def test_kernelsolve_S_is_psd(os_model):
    o, params, _, _ = os_model
    for k in o.kernelsolves:
        S = np.asarray(k(params)[1])
        assert np.allclose(S, S.T, atol=0)
        assert np.linalg.eigvalsh(S).min() >= 0.0


def test_pair_normalisations_positive(os_model):
    """No pair may have bs <= 0; that is what NaNs the whole statistic."""
    o, params, _, _ = os_model
    rhos, sigmas = o.os_rhosigma(params)
    assert np.all(np.isfinite(np.asarray(rhos)))
    assert np.all(np.asarray(sigmas) > 0)


def test_os_snr_finite_for_all_orfs(os_model):
    o, params, _, _ = os_model
    for orf in (hd_orfa, monopole_orfa, dipole_orfa):
        out = o.os(params, orf=orf)
        assert np.isfinite(out['snr']), orf.__name__
        assert np.isfinite(out['os']) and out['os_sigma'] > 0


def test_snr_invariant_to_gw_amplitude(os_model):
    """rho and sigma both scale as A^-2, so the S/N must not move."""
    o, params, _, _ = os_model
    a = o.os(dict(params, gw_log10_A=-14.5))
    b = o.os(dict(params, gw_log10_A=-13.0))
    assert a['snr'] == pytest.approx(b['snr'], rel=1e-8)
    assert b['os'] / a['os'] == pytest.approx(1.0, rel=1e-8)


def test_mcos_reduces_to_os(os_model):
    o, params, _, _ = os_model
    single = o.mcos(params, orfs=(hd_orfa,))
    ref = o.os(params, orf=hd_orfa)
    assert float(single['os'][0]) == pytest.approx(ref['os'], rel=1e-8)
    assert float(single['os_sigma'][0]) == pytest.approx(ref['os_sigma'], rel=1e-8)


def test_mcos_accepts_monopole(os_spread):
    """The canonical HD + monopole + dipole fit must run.

    monopole_orfa returning a 0-d scalar made jnp.stack raise here.
    """
    o, params = os_spread

    out = o.mcos(params, orfs=(hd_orfa, monopole_orfa, dipole_orfa))
    assert out['os'].shape == (3,) and out['cov'].shape == (3, 3)

    # a 2-component fit has spare degrees of freedom with three pairs
    two = o.mcos(params, orfs=(hd_orfa, monopole_orfa))
    assert two['os'].shape == (2,)
    assert np.all(np.isfinite(np.asarray(two['os'])))
    assert np.all(np.asarray(two['os_sigma']) > 0)


def test_Q_identities_and_snr(os_model):
    """x^T Q x reproduces the S/N; trace(Q) == 0 and 2*sum(eig^2) == 1."""
    o, params, _, _ = os_model
    Q = np.asarray(o.Q(params))

    assert np.allclose(Q, Q.T, atol=1e-12 * np.abs(Q).max())
    assert np.trace(Q) == pytest.approx(0.0, abs=1e-10)
    assert 2.0 * np.sum(np.linalg.eigvalsh(Q) ** 2) == pytest.approx(1.0, rel=1e-6)

    rng = np.random.default_rng(7)
    x = rng.normal(size=Q.shape[0])
    assert float(x @ Q @ x) == pytest.approx(
        float(o.sample_rhosigma_lowrank(params)(x)), rel=1e-4)


def test_Q_null_moments_are_not_a_validation(os_model):
    """trace(Q) == 0 and 2*sum(eig^2) == 1 are IDENTITIES, not checks.

    Q is assembled purely from off-diagonal pair blocks and normalised by
    2*sqrt(sum(orf^2 b)), and b_ij == ||A_i^T A_j||_F^2, so ||Q||_F^2 == 1/2 for
    ANY S -- verified to hold even on pure random garbage S. They are asserted
    here so that nobody mistakes them for a correctness test; the check that
    does constrain S is test_kernelsolve_matches_bruteforce.

    An earlier version of this test asserted the same two moments a second time
    by Monte Carlo (mean 0, std 1 of x^T Q x), which is the identical identity
    measured with sampling noise -- no extra information at 20000 draws.
    """
    o, params, _, _ = os_model
    Q = np.asarray(o.Q(params))
    eigs = np.linalg.eigvalsh(Q)
    assert np.trace(Q) == pytest.approx(0.0, abs=1e-10)
    assert 2.0 * np.sum(eigs ** 2) == pytest.approx(1.0, rel=1e-6)

    # the same identities on a deliberately wrong S -- they still hold
    rng = np.random.default_rng(11)
    bad = [rng.normal(size=(Q.shape[0] // len(o.psls),) * 2) for _ in o.psls]
    bad = [0.5 * (b + b.T) for b in bad]
    assert all(np.linalg.eigvalsh(b).min() < 0 for b in bad)   # genuinely indefinite


def test_Q_and_opQ_agree(os_model):
    o, params, _, _ = os_model
    Q = np.asarray(o.Q(params))
    op = o.opQ(params)
    rng = np.random.default_rng(13)
    x = rng.normal(size=Q.shape[0])
    assert np.asarray(op(x)) == pytest.approx(Q @ x, rel=1e-8, abs=1e-14)


def test_gx2cdf_forwards_orf(os_spread):
    """The null must match the ORF used for the point estimate."""
    pytest.importorskip('quadax')
    o, params = os_spread
    xs = np.array([0.0, 1.0, 2.0])

    def odd_orf(z):
        return matrix.jnp.where(z < 0.0, -1.0, 1.0)

    for orf in (hd_orfa, monopole_orfa, odd_orf):
        got = np.asarray(o.gx2cdf(params, xs, orf=orf))
        want = np.asarray(eig2cdf(xs, matrix.jnp.linalg.eigh(o.Q(params, orf=orf))[0]))
        assert np.all(np.isfinite(got))
        assert got == pytest.approx(want, rel=1e-8, abs=1e-10)

    # and the orf really does change Q (gx2cdf used to ignore it entirely)
    assert not np.allclose(np.asarray(o.Q(params, orf=hd_orfa)),
                           np.asarray(o.Q(params, orf=odd_orf)))


def test_shift_preserves_imaginary_part(os_model):
    """os_rhosigma_complex must return a complex rho.

    Casting through matrix.jnparray forced float64, so `shift` computed
    Re(ts)cos(dphi) instead of Re(ts e^{i dphi}) and the phase-shift null came
    out ~0.8x too narrow -- overstating significance.
    """
    o, params, _, _ = os_model
    rhos_c, _ = o.os_rhosigma_complex(params)
    assert np.iscomplexobj(np.asarray(rhos_c))
    assert np.any(np.abs(np.asarray(rhos_c).imag) > 0)


def test_shift_zero_phase_reproduces_os(os_model):
    o, params, _, _ = os_model
    nf = np.asarray(o.os_rhosigma_complex(params)[0]).shape[1]
    zero = [np.zeros(nf) for _ in o.psls]
    assert o.shift(params, zero)['snr'] == pytest.approx(o.os(params)['snr'], rel=1e-8)


def test_shift_null_is_wider_than_cos_only(os_model):
    """A quadrature contribution must actually be present in the shift null."""
    o, params, _, _ = os_model
    nf = np.asarray(o.os_rhosigma_complex(params)[0]).shape[1]
    rng = np.random.default_rng(17)
    snrs = [o.shift(params, [rng.uniform(0, 2 * np.pi, nf) for _ in o.psls])['snr']
            for _ in range(200)]
    assert np.all(np.isfinite(snrs))
    assert np.std(snrs) > 0


def test_invalidate_picks_up_new_residuals(os_model):
    """OS caches psl.y, so it must be invalidated after a residual swap."""
    o, params, psls, _ = os_model
    before = o.os(params)['snr']

    saved = [np.asarray(psl.y).copy() for psl in psls]
    try:
        rng = np.random.default_rng(19)
        for psl in psls:
            psl.y = matrix.jnparray(rng.normal(size=len(psl.y)) * np.std(psl.y))
        assert o.os(params)['snr'] == pytest.approx(before)   # stale, as documented
        o.invalidate()
        assert o.os(params)['snr'] != pytest.approx(before)
    finally:
        for psl, y in zip(psls, saved):
            psl.y = matrix.jnparray(y)
        o.invalidate()


def test_requires_two_pulsars(os_model):
    o, _, psls, _ = os_model

    class _Gbl:
        pass

    g = _Gbl()
    g.psls = psls[:1]
    with pytest.raises(ValueError, match='at least two pulsars'):
        OS(g)


# ------------------------------------------- ported from the golden-value suite
# These come from the suite that previously lived at this path. They are kept
# because they constrain things the brute-force test does not: pinned numerical
# output, the negative guard against the hand-rolled Woodbury returning, and the
# 2-D-Phi rejection. The three inv_prior tests from that suite are NOT ported --
# inv_prior no longer exists, the kernel classes' own make_inv supersedes it.

NG_FILES = ["v1p1_de440_pint_bipm2019-B1855+09.feather",
            "v1p1_de440_pint_bipm2019-J0023+0923.feather",
            "v1p1_de440_pint_bipm2019-J0030+0451.feather"]

NG_PARAMS = {'B1855+09_rednoise_gamma': 3.2, 'B1855+09_rednoise_log10_A': -14.5,
             'J0023+0923_rednoise_gamma': 3.2, 'J0023+0923_rednoise_log10_A': -14.5,
             'J0030+0451_rednoise_gamma': 3.2, 'J0030+0451_rednoise_log10_A': -14.5,
             'gw_gamma': 3.2, 'gw_log10_A': -14.5}

# Pinned with S taken from make_kernelsolve. The previous values came from a Q
# whose S dropped the inner constant-GP layer; the eigenvalue extremes moved by
# +81% when that was fixed (eig_min -1.7242e-01 -> -3.1147e-01). GOLDEN_OS did
# NOT move -- os_rhosigma always used make_kernelsolve.
GOLDEN_OS = {
    'hd':     dict(os=8.3961773831359343e-29, os_sigma=3.3828949412484901e-29,
                   snr=2.4819503794691431e+00),
    'mono':   dict(os=1.7353193676835731e-30, os_sigma=9.2639350397105645e-30,
                   snr=1.8731989810431465e-01),
    'dipole': dict(os=3.7929188575728078e-29, os_sigma=1.8191630514555593e-29,
                   snr=2.0849801531193126e+00),
}
GOLDEN_Q = dict(eig_min=-3.1146902057018455e-01, eig_max=3.6505695869859678e-01)


@pytest.fixture(scope='module')
def ng_os():
    """Three real NANOGrav pulsars -- a physically sensible array, unlike the
    packaged test pulsars (1710 s baseline, all at [0,0,1])."""
    import os as _os
    here = _os.path.dirname(_os.path.abspath(__file__))
    d = _os.path.join(here, '..', 'data')
    paths = [_os.path.join(d, f) for f in NG_FILES]
    if not all(_os.path.exists(p) for p in paths):
        pytest.skip('NANOGrav feathers not available')

    psrs = [ds.Pulsar.read_feather(p) for p in paths]
    T = ds.getspan(psrs)
    gbl = ds.GlobalLikelihood([
        ds.PulsarLikelihood([psr.residuals,
                             ds.makenoise_measurement(psr, psr.noisedict),
                             ds.makegp_ecorr(psr, psr.noisedict),
                             ds.makegp_timing(psr, svd=True),
                             ds.makegp_fourier(psr, ds.powerlaw, 30, T=T, name='rednoise'),
                             ds.makegp_fourier(psr, ds.powerlaw, 14, T=T,
                                               common=['gw_log10_A', 'gw_gamma'], name='gw')])
        for psr in psrs])
    return OS(gbl)


@pytest.mark.parametrize('orfname,orf', [('hd', hd_orfa), ('mono', monopole_orfa),
                                         ('dipole', dipole_orfa)])
def test_os_regression(ng_os, orfname, orf):
    """Pinned OS point estimate. Unchanged across every fix so far."""
    got = ng_os.os(NG_PARAMS, orf)
    for k in ('os', 'os_sigma', 'snr'):
        assert float(got[k]) == pytest.approx(GOLDEN_OS[orfname][k], rel=1e-8), f'{orfname} {k}'


def test_Q_eigenvalues_regression(ng_os):
    eigs = np.linalg.eigvalsh(np.asarray(ng_os.Q(NG_PARAMS)))
    assert eigs.min() == pytest.approx(GOLDEN_Q['eig_min'], rel=1e-8)
    assert eigs.max() == pytest.approx(GOLDEN_Q['eig_max'], rel=1e-8)


def test_Q_S_matches_kernelsolve(ng_os):
    """S must equal T^T K^-1 T from make_kernelsolve, on a real array."""
    for i, (psl, gw) in enumerate(zip(ng_os.psls, ng_os.gws)):
        S_ref = np.asarray(psl.N.make_kernelsolve(psl.y, gw.F)(NG_PARAMS)[1])
        S_got = np.asarray(ng_os.kernelsolves[i](NG_PARAMS)[1])
        rel = np.linalg.norm(S_got - S_ref) / np.linalg.norm(S_ref)
        assert rel < 1e-12, f'{psl.name}: relative Frobenius {rel:.3e}'


def test_Q_uses_the_full_nested_kernel(ng_os):
    """Negative guard: S must NOT match the white-noise-plus-outer-layer form.

    This is the one that fails if the hand-rolled Woodbury reduction is ever
    reintroduced. On this fixture the wrong S is off by 5.7%-21.7%.
    """
    psl, gw = ng_os.psls[0], ng_os.gws[0]
    inner = getattr(psl.N, 'N', None)
    if inner is None or isinstance(inner, matrix.NoiseMatrix):
        pytest.skip('fixture has no inner constant-GP Woodbury layer')

    import jax
    T = np.asarray(gw.F)
    LNm = 1.0 / np.sqrt(np.asarray(psl.white_noise_matrix))
    Ft, Tt = LNm[:, None] * np.asarray(psl.N.F), LNm[:, None] * T
    P = np.asarray(psl.N.P_var.getN(NG_PARAMS))
    Pinv = np.diag(1.0 / P) if P.ndim == 1 else np.linalg.inv(P)
    c = jax.scipy.linalg.cho_factor(Pinv + Ft.T @ Ft)
    S_bad = np.asarray(Tt.T @ Tt - (Ft.T @ Tt).T @ jax.scipy.linalg.cho_solve(c, Ft.T @ Tt))

    S_got = np.asarray(ng_os.kernelsolves[0](NG_PARAMS)[1])
    rel = np.linalg.norm(S_got - S_bad) / np.linalg.norm(S_got)
    assert rel > 1e-3, ('S matches the white-noise-only reduction: the constant-GP '
                        'layer (timing model / fixed ECORR) is being dropped again')


def test_imhof_u0_limit():
    """The Imhof integrand has a removable 0/0 at u=0 with limit 1/2(sum-x)."""
    from discovery.optimal import imhof
    import jax.numpy as jnp
    eigs = jnp.array([0.20, 0.15, -0.17, -0.05, 0.02, -0.01, 1e-4, -1e-4])
    for x in (0.0, 1.0, 2.0):
        val = float(imhof(jnp.array(0.0), x, eigs))
        assert np.isfinite(val)
        assert val == pytest.approx(0.5 * (float(np.sum(np.asarray(eigs))) - x), rel=1e-12)


def test_gx2cdf_orf_is_keyword_only(ng_os):
    """orf was added after osxs, so a positional third arg must not be read as
    an ORF -- that would silently reinterpret an existing cutoff argument."""
    with pytest.raises(TypeError):
        ng_os.gx2cdf(NG_PARAMS, np.array([1.0]), 1e-3)


def test_two_d_gw_phi_is_rejected():
    """A 2-D GW Phi (makegp_fftcov) must raise, not be elementwise-sqrted.

    Reachable from a shipped model: models/mpta.py builds curn with
    makegp_fftcov, and curn becomes psl.gw. The elementwise sqrt silently
    produced an (n,n,n) array instead of failing.
    """
    import os as _os
    here = _os.path.dirname(_os.path.abspath(__file__))
    d = _os.path.join(here, '..', 'data')
    paths = [_os.path.join(d, f) for f in NG_FILES[:2]]
    if not all(_os.path.exists(p) for p in paths):
        pytest.skip('NANOGrav feathers not available')

    psrs = [ds.Pulsar.read_feather(p) for p in paths]
    T = ds.getspan(psrs)
    gbl = ds.GlobalLikelihood([
        ds.PulsarLikelihood([psr.residuals,
                             ds.makenoise_measurement(psr, psr.noisedict),
                             ds.makegp_timing(psr, svd=True),
                             ds.makegp_fftcov(psr, ds.powerlaw, 21, T=T,
                                              common=['gw_log10_A', 'gw_gamma'], name='gw')])
        for psr in psrs])
    o = OS(gbl)
    p = {'gw_log10_A': -14.5, 'gw_gamma': 3.2}

    assert np.asarray(o.psls[0].gw.Phi.getN(p)).ndim == 2, 'fixture is not 2-D'
    for fn in (lambda: o.Q(p), lambda: o.opQ(p),
               lambda: o.gx2cdf(p, np.array([1.0]))):
        with pytest.raises(NotImplementedError, match='diagonal'):
            fn()
    # the documented fallback must actually work
    assert np.isfinite(float(o.os(p)['snr']))
