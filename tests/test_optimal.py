"""Tests for discovery.optimal."""

import functools

import jax
import numpy as np
import pytest

import discovery as ds
from discovery import matrix
from discovery.optimal import (OS, hd_orfa, dipole_orfa, monopole_orfa,
                               _psd, _ridge, eig2cdf, _DSTYPES)


# ---------------------------------------------------------------- ORFs

def test_orfs_are_elementwise():
    """Each ORF must map an array of angles to an equal-shaped array."""
    z = np.array([-1.0, -0.5, 0.0, 0.3, 0.9])
    for orf in (hd_orfa, dipole_orfa, monopole_orfa):
        assert np.shape(orf(z)) == z.shape, orf.__name__


def test_orfs_finite_at_zero_separation():
    """z == 1 must give the 1.0 autocorrelation limit, not nan."""
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
    """Clip at the round-off scale, and leave the positive spectrum alone."""
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
    """One of the two things _psd is for: making cholesky(S + ridge) defined."""
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


def test_psd_is_pure_and_jax_traceable():
    """_psd must not branch on a traced value; reporting lives in OS.validate."""
    rng = np.random.default_rng(3)
    A = rng.normal(size=(6, 6))
    S = A @ A.T
    jax.jit(_psd)(S)
    jax.vmap(_psd)(np.stack([S, S]))
    jax.grad(lambda M: _psd(M).sum())(S)


def test_psd_gradient_survives_repeated_eigenvalues():
    """What the custom_jvp is for: eigh's own VJP divides by (w_i - w_j)."""
    for S in (2.0 * np.eye(4), np.diag([1.0, 1.0, 2.0, 3.0])):
        g = np.asarray(jax.grad(lambda M: _psd(M).sum())(S))
        assert np.all(np.isfinite(g))
        # S is already PSD, so _psd is the identity there and d(sum)/dS is all ones
        assert np.allclose(g, np.ones_like(g))


def test_psd_prevents_negative_bs():
    """_psd keeps bs = tr(D_i D_j) non-negative for an anti-aligned pair under a steep Phi."""
    ncomp, k = 5, 10
    f = np.repeat(np.arange(1, ncomp + 1), 2).astype(float)
    sPhi = np.sqrt(f ** -8.8 / (1.0 ** -8.8))          # steep: weight on f1

    rng = np.random.default_rng(5)
    raw = []
    for _ in range(40):
        d = np.repeat(np.logspace(-6, 0, ncomp), 2) * 1e9   # info grows with f
        d[0] = d[1] = 1e-6 * 1e9 * rng.choice([-1.0, 1.0])  # f1 is round-off
        raw.append(np.diag(d))

    def nneg(Ss):
        return sum(np.sum((sPhi[:, None] * Ss[i] * sPhi[None, :])
                          * (sPhi[:, None] * Ss[j] * sPhi[None, :])) <= 0
                   for i in range(len(Ss)) for j in range(i + 1, len(Ss)))

    assert nneg(raw) > 0                                    # the bug
    assert nneg([np.asarray(_psd(S)) for S in raw]) == 0     # the fix


def test_ridge_is_scale_invariant():
    """_ridge scales with max|diag S|, so it is invariant under S -> lambda S."""
    rng = np.random.default_rng(4)
    A = rng.normal(size=(6, 6))
    S = A @ A.T
    for lam in (1e-20, 1e-6, 1.0, 1e6, 1e20):
        # abs=0.0: approx's tolerance is max(rel*expected, abs), and the default
        # abs=1e-12 swamps the whole comparison once lam*_ridge(S) falls below it
        assert float(_ridge(lam * S)) == pytest.approx(lam * float(_ridge(S)),
                                                       rel=1e-12, abs=0.0)


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
    # a CDF is a CDF: never outside [0, 1], never decreasing
    assert np.all((got >= 0.0) & (got <= 1.0))
    assert np.all(np.diff(got) >= -1e-9)


def test_eig2cdf_matches_montecarlo_mixed_signs():
    pytest.importorskip('quadax')
    eigs = np.array([0.6, 0.3, 0.1, -0.1, -0.3, -0.6])
    xs = np.array([-1.0, -0.3, 0.0, 0.3, 1.0])
    got = np.asarray(eig2cdf(xs, eigs))
    assert got == pytest.approx(_gx2_mc(eigs, xs), abs=5e-3)


def test_eig2cdf_integer_cutoff_keeps_largest_magnitude():
    """An integer cutoff must keep the largest |eig|, both signs."""
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
    """cutoff=1 must mean 'one eigenvalue', not a relative threshold."""
    pytest.importorskip('quadax')
    got = np.asarray(eig2cdf(np.array([1.0]), np.array([1.0, 0.5, 0.25]), cutoff=1))
    assert np.all(np.isfinite(got))
    assert np.all((got >= -1e-6) & (got <= 1 + 1e-6))


# ---------------------------------------------------- OS against brute force

@pytest.fixture(scope='module')
def os_model():
    """A small OS built from the packaged test pulsars."""
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
    """The same model but with well-separated sky positions."""
    o, params, psls, parts = os_model

    spread = OS.__new__(OS)
    spread.__dict__.update({k: v for k, v in o.__dict__.items()
                            if k in ('psls', 'gws', 'gwpar', 'pairs',
                                     '_kernelsolves_raw')})
    spread.pos = [matrix.jnparray(v) for v in
                  ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.6, 0.0, 0.8])]
    # an array, matching what OS.__init__ now guarantees -- it materialises
    # self.angles once so that os/mcos/shift cannot cache a tracer
    spread.angles = matrix.jnparray([float(np.dot(spread.pos[i], spread.pos[j]))
                                     for i, j in spread.pairs])
    _a = np.round(np.asarray(spread.angles), 6).tolist()
    assert len(set(_a)) == len(_a)
    return spread, params


@pytest.fixture(scope='module')
def os_novar_timing():
    """No constant GP, so sample_rhosigma gets past its nested-kernel guard."""
    import glob, os as _os

    here = _os.path.dirname(_os.path.abspath(__file__))
    files = sorted(glob.glob(_os.path.join(here, 'data', '*.feather')))
    if len(files) < 3:
        pytest.skip('need >= 3 pulsar feather files in tests/data')

    psrs = [ds.Pulsar.read_feather(f) for f in files]
    Tspan = ds.getspan(psrs)
    psls = [ds.PulsarLikelihood([p.residuals, ds.makenoise_measurement(p, p.noisedict),
                                 ds.makegp_fourier(p, ds.powerlaw, 5, T=Tspan,
                                                   common=['gw_log10_A', 'gw_gamma'],
                                                   name='gw')])
            for p in psrs]
    o = OS(ds.GlobalLikelihood(psls))
    np.random.seed(20260801)
    params = dict(ds.sample_uniform(o.params))
    params['gw_log10_A'], params['gw_gamma'] = -14.5, 4.33
    return o, params


def _diag(nm, params):
    """Diagonal of a NoiseMatrix, variable (getN) or constant (.N)."""
    P = np.asarray(nm.getN(params) if hasattr(nm, 'getN') else nm.N)
    return np.diag(P) if P.ndim == 2 else P


def test_kernelsolve_matches_bruteforce(os_model):
    """S = T^T C^-1 T and kv = T^T C^-1 y against an explicit dense solve."""
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
    """kernelsolves projects, so its S is PSD to machine precision."""
    o, params, _, _ = os_model
    for k, raw in zip(o.kernelsolves, o._kernelsolves_raw):
        S = np.asarray(k(params)[1])
        assert np.allclose(S, S.T, atol=0)
        assert np.linalg.eigvalsh(S).min() >= 0.0

        R = np.asarray(raw(params)[1])
        R = 0.5 * (R + R.T)
        assert np.linalg.eigvalsh(R).min() >= -1e-10 * np.abs(np.linalg.eigvalsh(R)).max()


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
    """The canonical HD + monopole + dipole fit must run."""
    o, params = os_spread

    out = o.mcos(params, orfs=(hd_orfa, monopole_orfa, dipole_orfa))
    assert out['os'].shape == (3,) and out['cov'].shape == (3, 3)

    # a 2-component fit has spare degrees of freedom with three pairs
    two = o.mcos(params, orfs=(hd_orfa, monopole_orfa))
    assert two['os'].shape == (2,)
    assert np.all(np.isfinite(np.asarray(two['os'])))
    assert np.all(np.asarray(two['os_sigma']) > 0)


# --------------------------------------------------- detection statistics

def test_dfcc_reproduces_the_traditional_os(os_spread):
    """DFCC is the OS S/N written as a quadratic form -- an exact identity."""
    o, params = os_spread
    assert float(o.dstat(params, hd_orfa, 'dfcc')['snr']) == pytest.approx(
        float(o.os(params)['snr']), rel=1e-10)


@pytest.mark.parametrize('dstype', _DSTYPES)
def test_every_dstype_has_unit_null_variance(os_spread, dstype):
    """x^T Q x for standard normal x has mean tr(Q) and variance 2 tr(Q^2)."""
    o, params = os_spread
    Q = np.asarray(o.Q(params, hd_orfa, dstype))
    assert np.allclose(Q, Q.T, atol=1e-10 * np.abs(Q).max())
    assert 2.0 * np.sum(np.linalg.eigvalsh(Q) ** 2) == pytest.approx(1.0, rel=1e-6)


def test_only_the_cross_only_dstypes_are_traceless(os_spread):
    """DF and NP keep the auto blocks, so they are not zero-mean under the null."""
    o, params = os_spread
    tr = {d: np.trace(np.asarray(o.Q(params, hd_orfa, d))) for d in _DSTYPES}
    assert tr['dfcc'] == pytest.approx(0.0, abs=1e-10)
    assert tr['npmv'] == pytest.approx(0.0, abs=1e-10)
    assert abs(tr['df']) > 1e-3
    assert abs(tr['np']) > 1e-3


def test_npmv_removes_the_auto_blocks_that_np_reintroduces(os_spread):
    """The whole point of NPMV."""
    o, params = os_spread
    npsr = len(o.psls)

    def auto_mass(dstype):
        Q = np.asarray(o.Q(params, hd_orfa, dstype))
        ngw = Q.shape[0] // npsr
        return sum(np.abs(Q[i*ngw:(i+1)*ngw, i*ngw:(i+1)*ngw]).sum() for i in range(npsr))

    assert auto_mass('np') > 0.0
    assert auto_mass('npmv') == 0.0
    assert auto_mass('dfcc') == 0.0
    assert auto_mass('df') > 0.0


def test_np_reduces_to_df_when_the_deflection_is_negligible(os_spread):
    """(I + B)^-1 B -> B as B -> 0, so the NP filter is the identity there."""
    o, params = os_spread
    val = {d: float(o.dstat(params, hd_orfa, d)['snr']) for d in _DSTYPES}
    assert val['np'] == pytest.approx(val['df'], rel=1e-9)
    assert val['npmv'] == pytest.approx(val['dfcc'], rel=1e-9)
    assert val['df'] != pytest.approx(val['dfcc'], rel=1e-3)   # auto blocks still differ


def test_dstypes_disagree_on_a_real_array(ng_os):
    """A switch that silently returns the same number would be worse than none."""
    vals = [float(ng_os.dstat(NG_PARAMS, hd_orfa, d)['snr']) for d in _DSTYPES]
    assert len(set(np.round(vals, 6))) == len(_DSTYPES), vals


def test_os_dstype_switch_reports_no_amplitude(os_spread):
    """Only DFCC estimates A^2; the rest must say so rather than invent one."""
    o, params = os_spread

    keys = set(o.os(params, hd_orfa, 'dfcc'))
    assert np.isfinite(float(o.os(params, hd_orfa, 'dfcc')['os']))

    for dstype in ('df', 'np', 'npmv'):
        out = o.os(params, hd_orfa, dstype)
        assert set(out) == keys              # same keys, so callers need no branch
        assert np.isnan(float(out['os'])) and np.isnan(float(out['os_sigma']))
        assert np.isfinite(float(out['snr']))


@pytest.mark.parametrize('dstype', _DSTYPES)
def test_os_dict_stays_vmappable(os_spread, dstype):
    """The dict returned by os() contains no string entries, so it survives vmap."""
    o, params = os_spread
    batch = {k: matrix.jnparray([v, v]) for k, v in params.items()}

    out = jax.vmap(lambda q: o.os(q, hd_orfa, dstype))(batch)
    assert np.asarray(out['snr']).shape == (2,)


def test_dstype_respects_the_orf(os_spread):
    """The null and the estimate both depend on the ORF, for every dstype."""
    o, params = os_spread
    for dstype in _DSTYPES:
        hd = float(o.dstat(params, hd_orfa, dstype)['snr'])
        dp = float(o.dstat(params, dipole_orfa, dstype)['snr'])
        assert hd != pytest.approx(dp, rel=1e-6), dstype


def test_unknown_dstype_raises(os_spread):
    o, params = os_spread
    for call in (lambda: o.dstat(params, hd_orfa, 'nvmp'),
                 lambda: o.Q(params, hd_orfa, 'NPVM'),
                 lambda: o.os(params, hd_orfa, 'pfos')):
        with pytest.raises(ValueError, match='unknown detection statistic'):
            call()


def test_dstype_is_case_insensitive(os_spread):
    o, params = os_spread
    assert float(o.dstat(params, hd_orfa, 'NPMV')['snr']) == pytest.approx(
        float(o.dstat(params, hd_orfa, 'npmv')['snr']), rel=1e-12)


@pytest.mark.parametrize('dstype', _DSTYPES)
def test_gx2cdf_matches_the_sampled_null(os_spread, dstype):
    """The p-value chain end to end: eigenvalues of Q -> Imhof -> CDF."""
    o, params = os_spread
    w = np.linalg.eigvalsh(np.asarray(o.Q(params, hd_orfa, dstype)))

    rng = np.random.default_rng(5)
    q = (rng.normal(size=(60000, len(w))) ** 2) @ w
    xs = np.percentile(q, [25, 50, 75, 90, 99])

    mc = np.array([(q <= x).mean() for x in xs])
    gx2 = np.asarray(o.gx2cdf(params, xs, orf=hd_orfa, dstype=dstype))
    assert np.max(np.abs(mc - gx2)) < 0.01      # 60k draws: MC noise alone is ~4e-3


def test_eig2cdf_warns_when_the_p_value_underflows():
    """p == 0 must not be silent -- it reads as an infinitely significant detection."""
    pytest.importorskip('quadax', exc_type=ImportError)
    eigs = np.linspace(-1.0, 1.0, 40)
    eigs = eigs / np.sqrt(2.0 * np.sum(eigs ** 2))

    # a batch, not one hand-picked x: out here the quadrature error dominates the
    # answer, so which particular x rounds to 1 is not stable
    xs = np.array([10.0, 20.0, 40.0, 60.0, 80.0, 120.0, 160.0, 240.0])
    with pytest.warns(RuntimeWarning, match='not as p == 0'):
        cdf = np.asarray(eig2cdf(xs, eigs))
    assert (cdf == 1.0).any()

    # and not on an ordinary value
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter('error')
        eig2cdf(np.array([0.5]), eigs)


def test_mcos_refuses_collinear_orfs(os_model):
    """Zero-separation pulsars make every ORF the same column."""
    o, params, _, _ = os_model      # all three pulsars at [0, 0, 1]
    with pytest.raises(ValueError, match='collinear'):
        o.mcos(params, orfs=(hd_orfa, monopole_orfa, dipole_orfa))
    # named so the caller can tell which component to drop
    with pytest.raises(ValueError, match='monopole_orfa'):
        o.mcos(params, orfs=(hd_orfa, monopole_orfa))


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
    """trace(Q) == 0 and 2*sum(eig^2) == 1 are IDENTITIES, not checks."""
    o, params, _, _ = os_model
    Q = np.asarray(o.Q(params))
    eigs = np.linalg.eigvalsh(Q)
    assert np.trace(Q) == pytest.approx(0.0, abs=1e-10)
    assert 2.0 * np.sum(eigs ** 2) == pytest.approx(1.0, rel=1e-6)

    # the same two identities on a Q ASSEMBLED FROM GARBAGE S: they still hold,
    # which is the point -- they test the assembly code, not S.
    ngw = Q.shape[0] // len(o.psls)
    rng = np.random.default_rng(11)
    bad = [rng.normal(size=(ngw, ngw)) for _ in o.psls]
    bad = [b @ b.T for b in bad]                      # PSD but unrelated to the data
    orfs = np.asarray(hd_orfa(matrix.jnparray(o.angles)))
    sPhi = np.ones(ngw)

    Ds = [sPhi[:, None] * b * sPhi[None, :] for b in bad]
    bs = np.array([np.sum(Ds[i] * Ds[j]) for i, j in o.pairs])
    denom = 2.0 * np.sqrt(np.sum(orfs ** 2 * bs))
    As = [np.linalg.cholesky(b) for b in bad]
    Qbad = np.zeros_like(Q)
    for w, (i, j) in zip(orfs, o.pairs):
        Bij = w * (As[i].T @ As[j])
        Qbad[i * ngw:(i + 1) * ngw, j * ngw:(j + 1) * ngw] += Bij
        Qbad[j * ngw:(j + 1) * ngw, i * ngw:(i + 1) * ngw] += Bij.T
    Qbad /= denom

    assert np.trace(Qbad) == pytest.approx(0.0, abs=1e-10)
    assert 2.0 * np.sum(np.linalg.eigvalsh(Qbad) ** 2) == pytest.approx(1.0, rel=1e-6)


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
    """os_rhosigma_complex must return a complex rho."""
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
    """The shifted null is wider than one built from the real part of ts alone."""
    o, params, _, _ = os_model
    rhos_c, sigmas = o.os_rhosigma_complex(params)
    rhos_c = np.asarray(rhos_c)
    nf = rhos_c.shape[1]
    gwnorm = 10 ** (2.0 * params[o.gwpar])
    sig = gwnorm * np.asarray(sigmas)
    orfs = np.asarray(hd_orfa(matrix.jnparray(o.angles)))
    w = orfs ** 2 / sig ** 2

    def snr_from(rhos):
        rhos = gwnorm * rhos
        os_ = np.sum(rhos * orfs / sig ** 2) / np.sum(w)
        return os_ / (1.0 / np.sqrt(np.sum(w)))

    rng = np.random.default_rng(17)
    correct, cos_only = [], []
    for _ in range(300):
        ph = np.array([rng.uniform(0, 2 * np.pi, nf) for _ in o.psls])
        dphi = np.array([ph[i] - ph[j] for i, j in o.pairs])
        correct.append(snr_from(np.sum(np.real(rhos_c * np.exp(1j * dphi)), axis=1)))
        cos_only.append(snr_from(np.sum(rhos_c.real * np.cos(dphi), axis=1)))

    assert np.all(np.isfinite(correct))
    # the buggy null is narrower -- that is what overstated the significance
    assert np.std(cos_only) < np.std(correct)

    # and the shipped shift() matches the correct branch, not the buggy one
    ph = rng.uniform(0, 2 * np.pi, (len(o.psls), nf))
    got = o.shift(params, [row for row in ph])['snr']
    dphi = np.array([ph[i] - ph[j] for i, j in o.pairs])
    assert got == pytest.approx(snr_from(np.sum(np.real(rhos_c * np.exp(1j * dphi)), axis=1)),
                                rel=1e-8)


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
    """Three real pulsars with distinct sky positions and realistic baselines."""
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
    """Negative guard: S must NOT match the white-noise-plus-outer-layer form."""
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
    """orf is keyword-only: a positional third argument must raise."""
    with pytest.raises(TypeError):
        ng_os.gx2cdf(NG_PARAMS, np.array([1.0]), 1e-3)


def test_two_d_gw_phi_is_rejected():
    """A 2-D GW Phi (makegp_fftcov) must raise, not be elementwise-sqrted."""
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
               lambda: o.gx2cdf(p, np.array([1.0])),
               lambda: o.validate(p)):
        with pytest.raises(NotImplementedError, match='diagonal'):
            fn()
    # the documented fallback must actually work
    assert np.isfinite(float(o.os(p)['snr']))


# ------------------------------------------------- _ridge, validate, jit safety

def test_ridge_is_never_zero():
    """A zero ridge makes jnp.linalg.cholesky return NaN silently."""
    assert float(_ridge(np.zeros((4, 4)))) > 0.0

    rng = np.random.default_rng(6)
    V = np.linalg.qr(rng.normal(size=(6, 6)))[0]
    allneg = V @ np.diag(-np.logspace(0, -3, 6)) @ V.T
    clipped = np.asarray(_psd(allneg))
    assert np.abs(clipped).max() == pytest.approx(0.0, abs=1e-30)
    assert float(_ridge(clipped)) > 0.0
    L = np.asarray(np.linalg.cholesky(clipped + float(_ridge(clipped)) * np.eye(6)))
    assert np.all(np.isfinite(L))

    # off-diagonal-only S: the diagonal is the wrong scale proxy on its own
    assert float(_ridge(np.array([[0.0, 1.0], [1.0, 0.0]]))) > 0.0


def test_Q_and_sample_are_jit_traceable(os_model):
    """Q, opQ and sample stay traceable under jit and vmap on a cold OS."""
    _, params, psls, _ = os_model
    o = OS(ds.GlobalLikelihood(psls))
    jax.jit(lambda p: o.Q(p))(params)
    jax.jit(lambda p: o.os(p)['snr'])(params)
    keys = jax.random.split(jax.random.PRNGKey(0), 4)
    out = jax.vmap(lambda kk: o.sample(kk, params))(keys)
    assert np.all(np.isfinite(np.asarray(out)))


def test_validate_reports_a_healthy_array(os_model):
    o, params, _, _ = os_model
    info = o.validate(params)
    assert info['phi_consistent'] is True
    assert info['min_bs'] > 0.0                    # no pair overlap has flipped
    assert info['min_cos'] > 0.0
    assert info['eta'].shape == (len(o.psls),)
    # inf when every S is PSD: no negative mass means no error length
    assert info['margin'] > 1.0


def test_validate_raises_when_a_pair_overlap_is_non_positive(os_model):
    """The criterion must key on the failure itself, not on a proxy."""
    o, params, _, _ = os_model
    n = np.asarray(o._kernelsolves_raw[0](params)[1]).shape[0]

    def planted(S):
        k = lambda prm, S=S: (np.zeros(n), S)
        k.params = []
        return k

    # pulsar 0 carries its power on mode 0 with a small NEGATIVE eigenvalue on
    # mode 1; pulsar 1 carries its power on mode 1. The overlap is then
    # A*E - N*B < 0: nearly orthogonal, and flipped by the negative eigenvalue.
    e0, e1 = np.zeros(n), np.zeros(n)
    e0[0], e1[1] = 1.0, 1.0
    S0 = 1.0 * np.outer(e0, e0) - 1e-6 * np.outer(e1, e1)
    S1 = 1.0 * np.outer(e1, e1) + 1e-12 * np.outer(e0, e0)

    raw = o._kernelsolves_raw
    try:
        o._kernelsolves_raw = [planted(S0), planted(S1)] + list(raw[2:])
        with pytest.raises(ValueError, match='non-positive overlap'):
            o.validate(params)
    finally:
        o._kernelsolves_raw = raw


def test_psd_projection_is_shared_by_every_consumer(os_model):
    """_psd belongs in kernelsolves, not at the Cholesky sites."""
    o, params, _, _ = os_model
    n = np.asarray(o._kernelsolves_raw[0](params)[1]).shape[0]

    def planted(S):
        k = lambda prm, S=S: (np.zeros(n), S)
        k.params = []
        return k

    e0, e1 = np.zeros(n), np.zeros(n)
    e0[0], e1[1] = 1.0, 1.0
    S0 = np.outer(e0, e0) - 1e-6 * np.outer(e1, e1)
    S1 = np.outer(e1, e1) + 1e-12 * np.outer(e0, e0)

    raw = o._kernelsolves_raw
    try:
        o.invalidate()      # rebuilds _kernelsolves_raw, so plant AFTER it
        o._kernelsolves_raw = [planted(S0), planted(S1)] + list(raw[2:])

        # structural: what kernelsolves hands every consumer must already be PSD
        S = np.asarray(o.kernelsolves[0](params)[1])
        assert np.linalg.eigvalsh(0.5 * (S + S.T)).min() >= 0.0

        # behavioural: the Cholesky-free path must still get finite sigmas
        _, sigmas = o.os_rhosigma(params)
        assert np.all(np.asarray(sigmas) > 0.0)
    finally:
        o._kernelsolves_raw = raw
        o.invalidate()


def test_validate_rejects_inconsistent_gw_phi(os_model):
    """The OS uses pulsar 0's Phi for every pair, so they must all match."""
    o, params, _, _ = os_model

    class _FakePhi:
        def __init__(self, inner, scale):
            self._inner, self._scale = inner, scale
            self.getN = lambda pars: self._scale * np.asarray(inner.getN(pars))
            self.getN.params = inner.getN.params

    class _FakeGw:
        def __init__(self, gw, scale):
            self.F, self.pos, self.gpcommon = gw.F, gw.pos, gw.gpcommon
            self.Phi = _FakePhi(gw.Phi, scale)

    saved = o.gws
    try:
        o.gws = [saved[0]] + [_FakeGw(g, 2.0) for g in saved[1:]]
        with pytest.raises(ValueError, match='differs from pulsar 0'):
            o.validate(params)
    finally:
        o.gws = saved


def test_invalidate_covers_every_cached_property(os_model):
    """Every cached_property must actually be cleared, and the raw solves rebuilt."""
    o, params, _, _ = os_model
    cached = {n for n, v in vars(OS).items() if isinstance(v, functools.cached_property)}
    assert cached                                  # sanity: there are some

    for name in cached:
        getattr(o, name)                           # populate every one
    assert cached <= set(o.__dict__), sorted(cached - set(o.__dict__))

    raw_before = o._kernelsolves_raw
    o.invalidate()

    assert not (cached & set(o.__dict__)), sorted(cached & set(o.__dict__))
    assert o._kernelsolves_raw is not raw_before   # rebuilt, not merely kept
    # and the object still works afterwards
    assert np.isfinite(float(o.os(params)['snr']))


# ------------------------------------------------------- eig2cdf convergence

def test_eig2cdf_warns_and_clips_when_quadrature_fails():
    """Discarding quadgk's status let CDF > 1 through, i.e. a NEGATIVE p-value."""
    pytest.importorskip('quadax')
    xs = np.array([25.0, 36.0, 49.0])
    with pytest.warns(RuntimeWarning, match='did not converge'):
        got = np.asarray(eig2cdf(xs, np.array([1.0])))
    assert np.all((got >= 0.0) & (got <= 1.0))


# ----------------------------------------------- previously untested samplers

def test_sample_draws_a_finite_snr(os_model):
    o, params, _, _ = os_model
    vals = [float(o.sample(jax.random.PRNGKey(i), params)) for i in range(20)]
    assert np.all(np.isfinite(vals))
    assert np.std(vals) > 0


def test_sample_null_has_unit_variance(os_model):
    """x^T Q x has unit null variance, so sample() must too."""
    o, params, _, _ = os_model
    keys = jax.random.split(jax.random.PRNGKey(1), 4000)
    vals = np.asarray(jax.vmap(lambda kk: o.sample(kk, params))(keys))
    assert np.all(np.isfinite(vals))
    assert vals.mean() == pytest.approx(0.0, abs=0.08)
    assert vals.std() == pytest.approx(1.0, abs=0.08)


def test_sample_rhosigma_lowrank_matches_Q(os_model):
    o, params, _, _ = os_model
    Q = np.asarray(o.Q(params))
    f = o.sample_rhosigma_lowrank(params)
    rng = np.random.default_rng(8)
    x = rng.normal(size=Q.shape[0])
    assert float(f(x)) == pytest.approx(float(x @ Q @ x), rel=1e-4)


def test_sample_rhosigma_refuses_clearly(os_model, os_novar_timing):
    """Every restriction must surface as NotImplementedError, not a raw TypeError."""
    o, params, _, _ = os_model
    with pytest.raises(NotImplementedError, match='nested kernel'):
        o.sample_rhosigma(params)

    o2, params2 = os_novar_timing
    with pytest.raises(NotImplementedError, match='constant white noise'):
        o2.sample_rhosigma(params2)


def test_scramble_changes_the_answer(os_spread):
    """scramble must actually use the positions it is handed."""
    o, params = os_spread
    true = o.scramble(params, o.pos)['snr']
    assert true == pytest.approx(o.os(params)['snr'], rel=1e-8)

    shuffled = [o.pos[i] for i in (1, 2, 0)]
    assert o.scramble(params, shuffled)['snr'] != pytest.approx(true, rel=1e-6)


def test_scramble_normalises_positions(os_spread):
    """A non-unit position must not silently change the answer."""
    o, params = os_spread
    true = o.scramble(params, o.pos)['snr']
    scaled = [3.0 * p for p in o.pos]
    assert o.scramble(params, scaled)['snr'] == pytest.approx(true, rel=1e-10)


@pytest.mark.parametrize('shape', [(4, 3), (2, 3), (3, 2)])
def test_scramble_rejects_the_wrong_position_shape(os_spread, shape):
    """A position array of the wrong shape raises rather than being clamped."""
    o, params = os_spread
    with pytest.raises(ValueError, match='unit position vector'):
        o.scramble(params, np.ones(shape))


def test_two_d_phi_rejected_by_every_1d_consumer(os_model):
    """_require_1d_phi guards six entry points; all must refuse a 2-D Phi."""
    o, params, _, _ = os_model
    saved = o.psls[0].gw.Phi.getN

    class _Phi2D:
        def __init__(self, inner):
            self._inner = inner
            self.params = inner.params
        def __call__(self, pars):
            v = np.asarray(self._inner(pars))
            return np.diag(v)

    try:
        o.psls[0].gw.Phi.getN = _Phi2D(saved)
        o.invalidate()
        for name, call in [('Q', lambda: o.Q(params)),
                           ('opQ', lambda: o.opQ(params)),
                           ('sample', lambda: o.sample(jax.random.PRNGKey(0), params)),
                           ('lowrank', lambda: o.sample_rhosigma_lowrank(params))]:
            with pytest.raises(NotImplementedError, match='1-D'):
                call()
    finally:
        o.psls[0].gw.Phi.getN = saved
        o.invalidate()
