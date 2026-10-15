"""Modulated Legendre ECORR: j(y) = m(y) u(y), m(y) = 1 + sum_d b_d P_d(y).

Covers signals.makegp_ecorr_legendre_modulated, its prior, and the ecorr_modulation option of
mpta.single_pulsar_noise and its detection in mpta.common_noise.
"""
import json
from pathlib import Path

import numpy as np
import jax
import pytest
from numpy.polynomial import legendre as L

import discovery as ds
from discovery import prior, signals, likelihood, matrix

DATA = Path(__file__).resolve().parent.parent / "data"
BANDS = {'LOW': (700.0, 1000.0), 'HIGH': (900.0, 1700.0)}


def _sim_pulsar(nepoch=10, nchan=30, seed=0, name='J0000+0000'):
    """Epochs alternate between two backends with different bands; nchan TOAs per epoch."""
    rng = np.random.default_rng(seed)
    toas, freqs, flags = [], [], []
    for e in range(nepoch):
        backend = 'LOW' if e % 2 == 0 else 'HIGH'
        lo, hi = BANDS[backend]
        toas += list(86400.0 * (55000.0 + 30.0 * e) + np.arange(nchan) * 1e-3)
        freqs += list(np.sort(rng.uniform(lo, hi, nchan)))
        flags += [backend] * nchan
    psr = ds.Pulsar()
    psr.name = name
    psr.toas = np.array(toas)
    psr.stoas = psr.toas.copy()
    psr.freqs = np.array(freqs)
    psr.toaerrs = rng.uniform(0.5e-6, 1.5e-6, len(toas))
    psr.backend_flags = np.array(flags)
    psr.flags = {'be': psr.backend_flags.copy()}
    t = (psr.toas - psr.toas.mean()) / psr.toas.std()
    psr.Mmat = np.column_stack([np.ones_like(t), t, t**2, (psr.backend_flags == 'HIGH') * 1.0])
    psr.residuals = 1e-6 * rng.standard_normal(len(toas))
    psr.pos = np.array([1.0, 0.0, 0.0])
    psr.mintoa, psr.maxtoa = psr.toas.min(), psr.toas.max()
    return psr


@pytest.fixture(scope="module")
def sim():
    return _sim_pulsar()


@pytest.fixture(scope="module")
def j0030():
    f = DATA / "v1p1_de440_pint_bipm2019-J0030+0451.feather"
    if not f.exists():
        pytest.skip("pulsar data fixture missing")
    return ds.Pulsar.read_feather(f)


def _white(psr):
    nd = {f'{psr.name}_{b}_efac': 1.1 for b in set(psr.backend_flags.tolist())}
    nd.update({f'{psr.name}_{b}_log10_t2equad': -7.0 for b in set(psr.backend_flags.tolist())})
    return signals.makenoise_measurement(psr, noisedict=nd)


def _likelihood(psr, gp):
    return likelihood.PulsarLikelihood([psr.residuals, signals.makegp_timing(psr, svd=True),
                                        _white(psr), gp])


def _random_point(params, rng, b_scale=1.0):
    out = {}
    for p in params:
        lo, hi = prior.getprior_uniform(p)
        if 'log10_ecorr' in p:
            lo, hi = -7.5, -5.5
        out[p] = float(rng.uniform(lo, hi) * (b_scale if 'mod_b' in p else 1.0))
    return out


def _y(psr):
    """Band coordinate per TOA, from each backend's log-frequency edges."""
    y = np.zeros(len(psr.freqs))
    for b in set(psr.backend_flags.tolist()):
        mask = psr.backend_flags == b
        lf = np.log(psr.freqs[mask])
        y[mask] = (lf - 0.5 * (lf.max() + lf.min())) / (0.5 * (lf.max() - lf.min()))
    return y


# --- the GP --------------------------------------------------------------------------------

def test_parameters_and_basis_at_zero_modulation(sim):
    gp = signals.makegp_ecorr_legendre_modulated(sim, nmodes=3, modulation=2)
    base = signals.makegp_ecorr_legendre(sim, nmodes=3, variable=True)
    mods = sorted(f'{sim.name}_{b}_ecorr_mod_b{d}' for b in BANDS for d in (1, 2))
    assert sorted(gp.F.params) == mods
    assert sorted(gp.Phi.params) == sorted(base.Phi.params)
    zero = {p: 0.0 for p in mods}
    assert np.array_equal(np.asarray(gp.F(zero)), np.asarray(base.F))
    assert all(prior.getprior_uniform(p) == [-0.4, 0.4] for p in mods)


@pytest.mark.parametrize("kw", [dict(modulation=0), dict(modulation=-1), dict(modulation=1.5),
                                dict(modulation=True), dict(nmodes=0)])
def test_refuses_bad_orders(sim, kw):
    with pytest.raises(ValueError):
        signals.makegp_ecorr_legendre_modulated(sim, **{'nmodes': 3, 'modulation': 2, **kw})


def test_refuses_a_fixed_profile_outside_its_prior(sim):
    with pytest.raises(ValueError, match="outside the prior"):
        signals.makegp_ecorr_legendre_modulated(
            sim, nmodes=2, modulation=1, noisedict={f'{sim.name}_LOW_ecorr_mod_b1': 0.5})


def test_fixed_profile_gives_a_constant_basis(sim):
    nd = {f'{sim.name}_{b}_ecorr_mod_b{d}': 0.1 * d for b in BANDS for d in (1, 2)}
    gp = signals.makegp_ecorr_legendre_modulated(sim, nmodes=2, modulation=2, noisedict=nd)
    free = signals.makegp_ecorr_legendre_modulated(sim, nmodes=2, modulation=2)
    assert not callable(gp.F)
    assert np.array_equal(np.asarray(gp.F), np.asarray(free.F(nd)))


# --- 1. nesting ----------------------------------------------------------------------------

@pytest.mark.parametrize("nmodes", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("modulation", [1, 2])
def test_zero_profile_reproduces_the_diagonal_model(sim, nmodes, modulation):
    mod = _likelihood(sim, signals.makegp_ecorr_legendre_modulated(sim, nmodes=nmodes,
                                                                   modulation=modulation))
    diag = _likelihood(sim, signals.makegp_ecorr_legendre(sim, nmodes=nmodes))
    fm, fd = jax.jit(mod.logL), jax.jit(diag.logL)
    rng = np.random.default_rng(nmodes * 10 + modulation)
    for _ in range(20):
        x = _random_point(diag.logL.params, rng)
        z = dict(x, **{p: 0.0 for p in mod.logL.params if 'mod_b' in p})
        a, b = float(fd(x)), float(fm(z))
        assert abs(b - a) <= 1e-8 * abs(a)


# --- 2. closed form ------------------------------------------------------------------------

def test_epoch_covariance_matches_the_closed_form(sim):
    nmodes, D = 4, 2
    gp = signals.makegp_ecorr_legendre_modulated(sim, nmodes=nmodes, modulation=D)
    y = _y(sim)
    rng = np.random.default_rng(5)
    for _ in range(5):
        x = _random_point(gp.F.params + gp.Phi.params, rng)
        G = np.asarray(gp.F(x))
        phi = np.asarray(gp.Phi.getN(x))
        for e in range(4):
            rows = slice(30 * e, 30 * (e + 1))
            backend = sim.backend_flags[30 * e]
            sig2 = np.array([10 ** (2 * x[f'{sim.name}_{backend}_log10_ecorr' + ('' if k == 0 else f'_k{k}')])
                             for k in range(nmodes)])
            P = L.legvander(y[rows], nmodes - 1)
            m = 1.0 + L.legvander(y[rows], D)[:, 1:] @ np.array(
                [x[f'{sim.name}_{backend}_ecorr_mod_b{d}'] for d in range(1, D + 1)])
            want = np.outer(m, m) * ((P * sig2) @ P.T)
            got = (G[rows] * phi) @ G[rows].T
            assert np.max(np.abs(got - want)) <= 1e-10 * np.max(np.abs(want))


# --- 3. brute force ------------------------------------------------------------------------

def _dense_logl(psr, K, U, var=1e40):
    """-1/2 r^T C^-1 r - 1/2 log|C| for C = K + U var U^T, by Woodbury on the dense K."""
    r = np.asarray(psr.residuals)
    cK = np.linalg.cholesky(K)
    Kr = np.linalg.solve(K, r)
    KU = np.linalg.solve(K, U)
    S = np.eye(U.shape[1]) / var + U.T @ KU
    cS = np.linalg.cholesky(S)
    quad = r @ Kr - (U.T @ Kr) @ np.linalg.solve(S, U.T @ Kr)
    logdet = (2 * np.sum(np.log(np.diag(cK))) + U.shape[1] * np.log(var)
              + 2 * np.sum(np.log(np.diag(cS))))
    return -0.5 * quad - 0.5 * logdet


def test_logl_matches_the_dense_gaussian(sim):
    gp = signals.makegp_ecorr_legendre_modulated(sim, nmodes=3, modulation=2)
    m = _likelihood(sim, gp)
    noise = _white(sim)
    U = np.linalg.svd(np.asarray(sim.Mmat), full_matrices=False)[0]
    rng = np.random.default_rng(11)
    for _ in range(5):
        x = _random_point(m.logL.params, rng)
        G = np.asarray(gp.F(x))
        K = np.diag(np.asarray(noise.N)) + (G * np.asarray(gp.Phi.getN(x))) @ G.T
        assert abs(float(m.logL(x)) - _dense_logl(sim, K, U)) < 1e-8


# --- 4. gradients --------------------------------------------------------------------------

def test_gradients_match_finite_differences(sim):
    m = _likelihood(sim, signals.makegp_ecorr_legendre_modulated(sim, nmodes=3, modulation=2))
    f, g = jax.jit(m.logL), jax.jit(jax.grad(m.logL))
    rng = np.random.default_rng(3)
    h = 1e-4
    for _ in range(10):
        x = _random_point(m.logL.params, rng)
        grad = g(x)
        for p in m.logL.params:
            vals = [float(f(dict(x, **{p: x[p] + k * h}))) for k in (-2, -1, 1, 2)]
            fd = (vals[0] - 8 * vals[1] + 8 * vals[2] - vals[3]) / (12 * h)
            assert np.isfinite(float(grad[p]))
            assert abs(float(grad[p]) - fd) <= 1e-5 * max(abs(fd), 1.0), p


# --- the model option ----------------------------------------------------------------------

KW = dict(fftint=False, max_cadence_days=30, background=False, red=True, dm=True, chrom=False,
          sw=False)


def _model(psr, **kw):
    from discovery.models import mpta
    return mpta.single_pulsar_noise(psr, **{**KW, **kw})


def _point(m, seed=0):
    rng = np.random.default_rng(seed)
    return {p: float(rng.uniform(*prior.getprior_uniform(p))) for p in m.logL.params}


def test_option_adds_exactly_the_profile_parameters(j0030):
    base = _model(j0030, ecorr_nmodes=3)
    mod = _model(j0030, ecorr_nmodes=3, ecorr_modulation=2)
    added = sorted(set(mod.logL.params) - set(base.logL.params))
    backends = sorted(set(np.asarray(j0030.backend_flags).tolist()))
    assert added == sorted(f'{j0030.name}_{b}_ecorr_mod_b{d}' for b in backends for d in (1, 2))
    assert set(base.logL.params) <= set(mod.logL.params)


@pytest.mark.parametrize("kw", [dict(ecorr_modulation=1),
                                dict(ecorr_modulation=1, ecorr_nmodes=3, ecorr_correlated=True),
                                dict(ecorr_modulation=1, ecorr_nmodes=3, quadratic=True),
                                dict(ecorr_modulation=1, ecorr_nmodes=3, ecorr=False),
                                dict(ecorr_modulation=1.0, ecorr_nmodes=3),
                                dict(ecorr_modulation=-1, ecorr_nmodes=3),
                                dict(ecorr_modulation=True, ecorr_nmodes=3)])
def test_option_refuses_what_it_cannot_modulate(j0030, kw):
    with pytest.raises(ValueError):
        _model(j0030, **kw)


def test_option_off_is_the_current_model(j0030):
    base = _model(j0030, ecorr_nmodes=3)
    off = _model(j0030, ecorr_nmodes=3, ecorr_modulation=0)
    x = _point(base)
    assert off.logL.params == base.logL.params
    assert float(off.logL(x)) == float(base.logL(x))


# --- 6. rebuild ----------------------------------------------------------------------------

def test_rebuild_from_feather_and_json_settings(j0030, tmp_path):
    settings = dict(KW, ecorr_nmodes=4, ecorr_modulation=2)
    m = _model(j0030, **settings)
    f = tmp_path / "psr.feather"
    j0030.save_feather(str(f))
    from discovery.models import mpta
    rebuilt = mpta.single_pulsar_noise(ds.Pulsar.read_feather(str(f)),
                                       **json.loads(json.dumps(settings)))
    x = _point(m, seed=4)
    assert rebuilt.logL.params == m.logL.params
    assert abs(float(rebuilt.logL(x)) - float(m.logL(x))) <= 1e-10 * abs(float(m.logL(x)))


def _chain(psr, nmodes, modulation=0, correlated=False, value=-0.1):
    import pandas as pd
    backends = sorted(set(np.asarray(psr.backend_flags).tolist()))
    cols = {f'{psr.name}_red_noise_log10_A': -14.0, f'{psr.name}_red_noise_gamma': 3.0}
    for b in backends:
        cols[f'{psr.name}_{b}_efac'] = 1.0
        cols[f'{psr.name}_{b}_log10_ecorr'] = -6.5
        for k in range(1, nmodes):
            cols[f'{psr.name}_{b}_log10_ecorr_k{k}'] = -7.0
        for d in range(1, modulation + 1):
            cols[f'{psr.name}_{b}_ecorr_mod_b{d}'] = value
        if correlated:
            cols[f'{psr.name}_{b}_ecorr_corr_k1k0'] = 0.1
    df = pd.DataFrame({c: np.full(4, v) for c, v in cols.items()})
    df.attrs['noisedict'] = {}
    return df


@pytest.mark.parametrize("nmodes, modulation", [(3, 2), (1, 1)])
def test_common_noise_detects_the_variant_from_the_chain(j0030, capsys, nmodes, modulation):
    from discovery.models import mpta
    j0030.telescope = np.array(['meerkat'] * len(j0030.toas))
    kw = dict(fftInt=False, fd=False, noise_point='median')
    mod = mpta.common_noise([j0030], [_chain(j0030, nmodes, modulation)], **kw)
    assert f'modulated to degree {modulation}' in capsys.readouterr().out
    zero = mpta.common_noise([j0030], [_chain(j0030, nmodes, modulation, value=0.0)], **kw)
    plain = mpta.common_noise([j0030], [_chain(j0030, nmodes)], **kw)
    x = _point(mod)
    assert float(zero.logL(x)) == pytest.approx(float(plain.logL(x)), rel=1e-12, abs=0)
    assert float(mod.logL(x)) != float(plain.logL(x))


def test_common_noise_refuses_a_chain_with_both_variants(j0030):
    from discovery.models import mpta
    j0030.telescope = np.array(['meerkat'] * len(j0030.toas))
    with pytest.raises(ValueError, match="both ecorr_corr_k and ecorr_mod_b"):
        mpta.common_noise([j0030], [_chain(j0030, 3, 2, correlated=True)], fftInt=False,
                          fd=False, noise_point='median')


def test_conditional_carries_the_gp(sim):
    m = _likelihood(sim, signals.makegp_ecorr_legendre_modulated(sim, nmodes=3, modulation=2))
    x = _random_point(m.logL.params, np.random.default_rng(2))
    _, coeffs = m.sample_conditional(jax.random.PRNGKey(0), x)
    assert list(coeffs) == [f'{sim.name}_modecorrGPleg_coefficients(30)']


# --- review follow-ups ---------------------------------------------------------------------

def test_partly_fixed_profiles_match_the_dense_gaussian(sim):
    U = np.linalg.svd(np.asarray(sim.Mmat), full_matrices=False)[0]
    noise = _white(sim)
    rng = np.random.default_rng(21)
    free = signals.makegp_ecorr_legendre_modulated(sim, nmodes=3, modulation=2)
    x = _random_point(free.F.params + free.Phi.params, rng)
    amps_fixed = {p: x[p] for p in free.Phi.params}
    one_b_fixed = {f'{sim.name}_LOW_ecorr_mod_b2': x[f'{sim.name}_LOW_ecorr_mod_b2']}
    for nd in (amps_fixed, one_b_fixed, {**amps_fixed, **one_b_fixed}):
        gp = signals.makegp_ecorr_legendre_modulated(sim, nmodes=3, modulation=2, noisedict=nd)
        m = _likelihood(sim, gp)
        G = np.asarray(free.F(x))
        K = np.diag(np.asarray(noise.N)) + (G * np.asarray(free.Phi.getN(x))) @ G.T
        assert abs(float(m.logL({p: x[p] for p in m.logL.params})) - _dense_logl(sim, K, U)) < 1e-8


def test_zero_profile_reproduces_the_diagonal_model_with_sampled_white_noise(sim):
    def build(gp):
        return likelihood.PulsarLikelihood([sim.residuals, signals.makegp_timing(sim, svd=True),
                                            signals.makenoise_measurement(sim), gp])
    mod = build(signals.makegp_ecorr_legendre_modulated(sim, nmodes=4, modulation=2))
    diag = build(signals.makegp_ecorr_legendre(sim, nmodes=4))
    rng = np.random.default_rng(8)
    for _ in range(5):
        x = _random_point(diag.logL.params, rng)
        z = dict(x, **{p: 0.0 for p in mod.logL.params if 'mod_b' in p})
        assert float(mod.logL(z)) == float(diag.logL(x))


def test_conditional_uses_the_sampled_profile(sim):
    free = signals.makegp_ecorr_legendre_modulated(sim, nmodes=3, modulation=2)
    x = _random_point(free.F.params + free.Phi.params, np.random.default_rng(4))
    fixed = signals.makegp_ecorr_legendre_modulated(
        sim, nmodes=3, modulation=2, noisedict={p: x[p] for p in free.F.params})
    mf, mr = _likelihood(sim, free), _likelihood(sim, fixed)
    mu, _ = mf.conditional(x)
    mu_ref, _ = mr.conditional({p: x[p] for p in mr.logL.params})
    assert np.allclose(np.asarray(mu), np.asarray(mu_ref), rtol=1e-10, atol=0)
    mu0, _ = mf.conditional(dict(x, **{p: 0.0 for p in free.F.params}))
    assert not np.allclose(np.asarray(mu0), np.asarray(mu), rtol=1e-3)


def test_option_accepts_a_modulated_gp_through_extra_gps(j0030):
    gp = signals.makegp_ecorr_legendre_modulated(j0030, nmodes=2, modulation=1)
    m = _model(j0030, ecorr=False, extra_gps=[gp])
    assert any('ecorr_mod_b1' in p for p in m.logL.params)


def test_option_refuses_profile_values_it_would_ignore(j0030):
    be = sorted(set(np.asarray(j0030.backend_flags).tolist()))[0]
    with pytest.raises(ValueError, match="ecorr_modulation"):
        _model(j0030, ecorr_nmodes=3, noisedict={f'{j0030.name}_{be}_ecorr_mod_b1': 0.1})


def test_common_noise_refuses_a_chain_missing_profile_columns(j0030):
    from discovery.models import mpta
    j0030.telescope = np.array(['meerkat'] * len(j0030.toas))
    df = _chain(j0030, 3, 2)
    df = df.drop(columns=[c for c in df.columns if c.endswith('_ecorr_mod_b2')][:1])
    with pytest.raises(ValueError, match="lacks"):
        mpta.common_noise([j0030], [df], fftInt=False, fd=False, noise_point='median')


@pytest.mark.parametrize("fftInt", [False, True])
def test_common_noise_stacked_path_matches_the_global_path(j0030, fftInt):
    from discovery.models import mpta
    j0030.telescope = np.array(['meerkat'] * len(j0030.toas))
    kw = dict(fftInt=fftInt, fd=False, noise_point='median', fix_chrom_alpha=True)
    glob = mpta.common_noise([j0030], [_chain(j0030, 3, 2)], **kw)
    stack = mpta.common_noise([j0030], [_chain(j0030, 3, 2)], use_commongp=True, **kw)
    x = _point(glob)
    assert type(stack).__name__ == 'ArrayLikelihood'
    assert abs(float(stack.logL(x)) - float(glob.logL(x))) <= 1e-10 * abs(float(glob.logL(x)))
