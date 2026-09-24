"""Tests for the frequency-dependent delay under a power-law prior in log-frequency, and the
node-space projection it shares with the Matern prior."""

import types
from pathlib import Path

import numpy as np
import pytest

import discovery as ds
from discovery import signals as s


DATA = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture(scope="module")
def psr():
    f = DATA / "v1p1_de440_pint_bipm2019-J0030+0451.feather"
    if not f.exists():
        pytest.skip("pulsar data fixture missing")
    return ds.Pulsar.read_feather(f)


@pytest.fixture
def restore_priors():
    from discovery import prior
    saved = dict(prior.priordict_standard)
    yield prior
    prior.priordict_standard.clear()
    prior.priordict_standard.update(saved)


def _names(psr):
    return (f'{psr.name}_fd_gp_log10_sigma', f'{psr.name}_fd_gp_gamma',
            f'{psr.name}_fd_gp_log10_sigma_ind')


def _C(gp, p):
    F = np.asarray(gp.F)
    return F @ np.asarray(gp.Phi.getN(p)) @ F.T


def _rel(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)


# --- recognising the frequency-only timing-model columns -----------------------------------

def test_frequency_only_columns_are_recognised_from_their_values_alone():
    """Offset, FDk polynomials and a constant nu^-2 count; anything that varies in time does
    not. Their node patterns are the column functions evaluated at the nodes."""
    rng = np.random.default_rng(3)
    n = 3000
    freqs = rng.uniform(800.0, 1700.0, n)
    t = rng.uniform(0.0, 1.0, n)
    x = np.log(freqs / 1000.0)
    cols = {'Offset': np.ones(n), 'FD1': x, 'FD2': x ** 2, 'FD5': x ** 5, 'DM': (freqs / 1000.0) ** -2,
            'F0': t, 'DM1': t * (freqs / 1000.0) ** -2, 'PX': np.sin(6.0 * t)}
    fake = types.SimpleNamespace(name='FAKE', freqs=freqs, Mmat=np.column_stack(list(cols.values())),
                                 fitpars=list(cols))
    q = np.log(np.linspace(820.0, 1680.0, 24))
    fmat_raw = np.zeros((n, 24))
    pats, labels = s._fd_frequency_only_patterns(fake, fmat_raw, q)
    assert labels == ['Offset', 'FD1', 'FD2', 'FD5', 'DM']
    xq = q - np.log(1000.0)
    for pat, want in zip(pats, (np.ones(24), xq, xq ** 2, xq ** 5, np.exp(-2.0 * xq))):
        assert np.allclose(pat, want, rtol=1e-8, atol=1e-10)


def test_the_node_projection_removes_the_recognised_directions_exactly(psr):
    x = np.log(np.asarray(psr.freqs, dtype=np.float64))
    fmat_raw, q = s._fd_piecewise_block(psr, x, np.ones(len(x), bool), 16, 'quantile', 'fd_gp')
    pats, labels = s._fd_frequency_only_patterns(psr, fmat_raw, q)
    gp = s.makegp_fd_piecewise_powerlaw(psr, nodes=16)
    B = gp.node_basis

    assert labels == gp.node_removed and labels
    assert np.allclose(B.T @ B, np.eye(B.shape[1]), rtol=0.0, atol=1e-12)
    assert B.shape[1] == 16 - np.linalg.matrix_rank(np.column_stack(pats))
    for g in pats:
        assert np.linalg.norm(B.T @ g) < 1e-12 * np.linalg.norm(g)


def test_without_the_node_projection_every_direction_stays(psr):
    gp = s.makegp_fd_piecewise_powerlaw(psr, nodes=16, project_nodes=False)
    assert np.asarray(gp.F).shape[1] == 16 and gp.node_removed == []


def test_a_selection_makes_group_offsets_representable(psr):
    """Per-backend blocks can carry a jump between backends, so it is removed as well."""
    one = s.makegp_fd_piecewise_powerlaw(psr, nodes=16)
    per = s.makegp_fd_piecewise_powerlaw(psr, nodes=16, selection=s.selection_backend_flags)
    assert set(one.node_removed) < set(per.node_removed)


# --- the kernel ------------------------------------------------------------------------------

def test_the_kernel_is_the_band_limited_power_law():
    """Exact at gamma = 0, where the step spectrum is the spectrum; within 1e-3 of k(0) at
    steep gamma on the default bin count."""
    from scipy.integrate import quad
    f_lo, f_hi = 0.8, 27.0
    lags = np.concatenate([[0.0], np.linspace(0.005, 0.6, 40)])
    for gamma, tol in ((0.0, 1e-12), (4.0, 1e-3), (8.0, 1e-3)):
        Z = quad(lambda f: f ** -gamma, f_lo, f_hi, limit=400)[0]
        exact = np.array([quad(lambda f: f ** -gamma * np.cos(2 * np.pi * f * L), f_lo, f_hi,
                               limit=400)[0] / Z for L in lags[1:]])
        k = np.asarray(s._fd_powerlaw_correlation(lags, f_lo, f_hi, 128)(gamma))[0, 1:]
        assert np.max(np.abs(k - exact)) < tol


def test_the_band_follows_the_node_grid(psr):
    gp = s.makegp_fd_piecewise_powerlaw(psr, nodes=16, flo_pad=3.0)
    q = np.log(np.asarray(gp.fd_nodes))
    f_lo, f_hi = gp.fd_bands[0]
    assert f_hi == pytest.approx(1.0 / (2.0 * np.median(np.diff(np.sort(q)))), rel=1e-12)
    assert f_lo == pytest.approx(1.0 / (3.0 * (q.max() - q.min())), rel=1e-12)


def test_steeper_spectra_correlate_neighbouring_nodes_more(psr):
    sig, gam, _ = _names(psr)
    gp = s.makegp_fd_piecewise_powerlaw(psr, nodes=32, spacing='log', project_nodes=False)
    lag1 = []
    for g in (0.0, 2.0, 4.0):
        Phi = np.asarray(gp.Phi.getN({sig: -7.0, gam: g}))
        d = np.sqrt(np.diag(Phi))
        lag1.append(np.mean(np.diag(Phi / np.outer(d, d), 1)))
    assert lag1[0] < lag1[1] < lag1[2]


# --- the prior -------------------------------------------------------------------------------

@pytest.mark.parametrize("kw", [dict(nodes=16), dict(nodes=32, spacing='log'),
                                dict(nodes=16, selection=s.selection_backend_flags)])
def test_sigma_is_the_rms_delay_per_toa_after_projection(psr, kw):
    sig, gam, ind = _names(psr)
    gp = s.makegp_fd_piecewise_powerlaw(psr, **kw)
    gi = s.makegp_fd_piecewise_powerlaw(psr, independent=True, **kw)
    for g in (0.0, 3.0, 8.0):
        C = _C(gp, {sig: -7.0, gam: g})
        assert np.sqrt(np.mean(np.diag(C))) == pytest.approx(1e-7, rel=1e-10)
        C = _C(gi, {sig: -7.0, gam: g, ind: -7.3})
        assert np.mean(np.diag(C)) == pytest.approx(1e-14 + 10 ** -14.6, rel=1e-10)


def test_the_independent_term_is_additive(psr):
    sig, gam, ind = _names(psr)
    gi = s.makegp_fd_piecewise_powerlaw(psr, nodes=16, independent=True)
    both = np.asarray(gi.Phi.getN({sig: -7.0, gam: 3.0, ind: -7.3}))
    alone = (np.asarray(gi.Phi.getN({sig: -7.0, gam: 3.0, ind: -40.0}))
             + np.asarray(gi.Phi.getN({sig: -40.0, gam: 3.0, ind: -7.3})))
    assert np.allclose(both, alone, rtol=1e-10, atol=0.0)


def test_the_prior_is_positive_definite_across_its_box(psr):
    sig, gam, _ = _names(psr)
    for kw in (dict(nodes=16), dict(nodes=64, spacing='quantile')):
        gp = s.makegp_fd_piecewise_powerlaw(psr, **kw)
        for g in (0.0, 4.0, 8.0):
            w = np.linalg.eigvalsh(np.asarray(gp.Phi.getN({sig: -7.0, gam: g})))
            assert w.min() > 0.0 and w.max() / w.min() < 1e10


def test_the_jitter_floor_scales_with_the_projected_prior(psr):
    """Relative to the power the projection leaves, a 1e-8 floor moves the model by ~1e-8
    even at the steepest index, where the surviving power is a tiny part of k(0)."""
    sig, gam, _ = _names(psr)
    p = {sig: -7.0, gam: 8.0}
    a = s.makegp_fd_piecewise_powerlaw(psr, nodes=64, jitter=1e-8)
    b = s.makegp_fd_piecewise_powerlaw(psr, nodes=64, jitter=1e-14)
    assert _rel(_C(a, p), _C(b, p)) < 1e-6


def test_gradients_reach_every_hyperparameter(psr):
    import jax
    sig, gam, ind = _names(psr)
    gi = s.makegp_fd_piecewise_powerlaw(psr, nodes=16, independent=True)
    psl = ds.PulsarLikelihood([psr.residuals, ds.makenoise_measurement(psr), gi])
    p = {k: (1.0 if k.endswith('efac') else -6.5 if 'equad' in k else -7.0) for k in psl.logL.params}
    p[gam] = 3.0
    g = jax.grad(psl.logL)(p)
    for k in (sig, gam, ind):
        assert np.isfinite(float(g[k])) and float(g[k]) != 0.0


# --- the Matern flag --------------------------------------------------------------------------

def test_the_matern_flag_defaults_off(psr):
    a = s.makegp_fd_piecewise_matern(psr, nodes=16)
    b = s.makegp_fd_piecewise_matern(psr, nodes=16, project_nodes=False)
    p = {f'{psr.name}_fd_gp_log10_sigma': -7.0, f'{psr.name}_fd_gp_log10_ell': 0.0}
    assert np.array_equal(np.asarray(a.F), np.asarray(b.F))
    assert np.array_equal(np.asarray(a.Phi.getN(p)), np.asarray(b.Phi.getN(p)))
    assert np.asarray(a.F).shape[1] == 16


def test_the_matern_flag_projects_the_same_directions(psr):
    m = s.makegp_fd_piecewise_matern(psr, nodes=16, normalise=True, project_nodes=True)
    pl = s.makegp_fd_piecewise_powerlaw(psr, nodes=16)
    assert m.node_removed == pl.node_removed
    assert np.array_equal(m.node_basis, pl.node_basis)
    p = {f'{psr.name}_fd_gp_log10_sigma': -7.0, f'{psr.name}_fd_gp_log10_ell': 0.0}
    assert np.sqrt(np.mean(np.diag(_C(m, p)))) == pytest.approx(1e-7, rel=1e-10)


# --- the model -------------------------------------------------------------------------------

def _mpta(psr, **kw):
    from discovery.models import mpta
    return mpta.single_pulsar_noise(psr, fftint=False, max_cadence_days=30, background=False,
                                    red=True, dm=True, chrom=False, sw=False, fd=True, fd_nodes=16, **kw)


def test_mpta_builds_the_power_law_with_exactly_its_parameters(psr, restore_priors):
    sig, gam, ind = _names(psr)
    m = _mpta(psr, fd_prior='powerlaw')
    assert {p for p in m.logL.params if '_fd_gp_' in p} == {sig, gam}
    m = _mpta(psr, fd_prior='powerlaw', fd_independent=True)
    assert {p for p in m.logL.params if '_fd_gp_' in p} == {sig, gam, ind}


def test_mpta_sets_the_fd_priors(psr, restore_priors):
    from discovery.models import mpta
    sig, gam, ind = _names(psr)
    _mpta(psr, fd_prior='powerlaw', fd_independent=True)
    assert restore_priors.getprior_uniform(sig) == [-10, -4]
    assert restore_priors.getprior_uniform(gam) == [0, 8]
    assert restore_priors.getprior_uniform(ind) == [-10, -4]
    from discovery.models import ppta  # noqa: F401 -- its priors are front-inserted on import
    ppta.update_priordict_standard_ppta()
    restore_priors.priordict_standard['(.*_)?fd_gp_log10_sigma_ind'] = [-12.0, -3.0]
    assert restore_priors.getprior_uniform(ind) == [-12.0, -3.0]
    assert restore_priors.getprior_uniform(sig) == [-10, -4]
    shadowed, _ = restore_priors.shadowed_priors()
    assert not [x for x in shadowed if 'fd_gp' in x[0]]

    _mpta(psr, fd_prior='powerlaw', fd_gamma_max=6.0)
    assert restore_priors.getprior_uniform(gam) == [0.0, 6.0]

    gp = s.makegp_fd_piecewise_matern(psr, nodes=16)
    mpta._set_fd_priors(psr, gp)
    span = np.log(gp.fd_nodes.max()) - np.log(gp.fd_nodes.min())
    lo, hi = restore_priors.getprior_uniform(f'{psr.name}_fd_gp_log10_ell')
    assert lo == pytest.approx(np.log10(0.1 * span)) and hi == pytest.approx(np.log10(3.0 * span))


def test_signals_registers_no_fd_prior(psr, restore_priors):
    before = dict(restore_priors.priordict_standard)
    s.makegp_fd_piecewise_matern(psr, nodes=16)
    s.makegp_fd_piecewise_powerlaw(psr, nodes=16, independent=True)
    assert dict(restore_priors.priordict_standard) == before


@pytest.mark.parametrize("prior, extra, match", [
    ('matern', dict(fd_independent=True), "powerlaw"),
    ('matern', dict(fd_flo_pad=4.0), "powerlaw"),
    ('improper', dict(fd_gamma_max=6.0), "powerlaw"),
    ('improper', dict(fd_project_nodes=True), "improper"),
    ('gaussian', {}, "must be"),
    ('powerlaw', dict(fd_normalise=False), "normalised by construction"),
    ('independent', dict(fd_normalise=False), "normalised by construction"),
])
def test_mpta_refuses_switches_the_prior_cannot_use(psr, restore_priors, prior, extra, match):
    with pytest.raises(ValueError, match=match):
        _mpta(psr, fd_prior=prior, **extra)


def test_mpta_checks_the_fd_parameters_it_built(psr):
    from discovery.models import mpta
    want = mpta._fd_parameters(psr, True, 'powerlaw', False)
    mpta._check_fd_parameters(psr, sorted(want) + ['other'], want, 'x')
    with pytest.raises(ValueError, match="switches ask for"):
        mpta._check_fd_parameters(psr, sorted(want) + [f'{psr.name}_fd_gp_log10_ell'], want, 'x')


def test_mpta_power_law_gives_a_finite_posterior(psr, restore_priors):
    import jax
    from discovery import prior as _p
    m = _mpta(psr, fd_prior='powerlaw', fd_independent=True)
    t = _p.makelogtransform_uniform(m.logL)
    ys = np.zeros(len(t.params))
    assert np.isfinite(float(t(ys)))
    assert np.isfinite(np.asarray(jax.grad(t)(ys))).all()


def test_mpta_independent_only_samples_one_amplitude(psr, restore_priors):
    sig, gam, ind = _names(psr)
    m = _mpta(psr, fd_prior='independent')
    assert {p for p in m.logL.params if '_fd_gp_' in p} == {ind}
    with pytest.raises(ValueError, match="powerlaw"):
        _mpta(psr, fd_prior='independent', fd_gamma_max=6.0)
    with pytest.raises(ValueError, match="no term"):
        s.makegp_fd_piecewise_powerlaw(psr, nodes=16, powerlaw=False)


def test_the_independent_only_prior_is_the_independent_term(psr):
    sig, gam, ind = _names(psr)
    gi = s.makegp_fd_piecewise_powerlaw(psr, nodes=16, independent=True)
    io = s.makegp_fd_piecewise_powerlaw(psr, nodes=16, independent=True, powerlaw=False)
    assert np.array_equal(np.asarray(gi.F), np.asarray(io.F))
    assert np.allclose(np.asarray(io.Phi.getN({ind: -7.3})),
                       np.asarray(gi.Phi.getN({sig: -40.0, gam: 3.0, ind: -7.3})), rtol=1e-10, atol=1e-30)


def test_every_fd_prior_projects_the_same_span_out_of_the_chromatic_polynomial(psr, restore_priors):
    """The improper chrom_poly basis is projected against the whole FD span, not the reduced
    basis the node projection leaves, so the improper rank is the same under every prior."""
    from discovery.models import mpta
    bases = {}
    for prior in ('matern', 'powerlaw', 'independent'):
        _, comps = mpta.single_pulsar_noise(psr, fftint=False, max_cadence_days=30, background=False,
                                            red=True, dm=True, chrom=True, sw=False, chrom_poly=True,
                                            fd=True, fd_nodes=16, fd_prior=prior, return_components=True)
        cp = [np.asarray(c.F({f'{psr.name}_chrom_gp_alpha': 4.0})) for c in comps
              if getattr(c, 'gpname', '') == 'chrom_gp' and callable(getattr(c, 'F', None))]
        bases[prior] = sorted(cp, key=lambda F: F.shape[1])
    assert [F.shape[1] for F in bases['matern']][0] == 3
    for prior in ('powerlaw', 'independent'):
        assert len(bases[prior]) == len(bases['matern'])
        assert all(np.array_equal(a, b) for a, b in zip(bases[prior], bases['matern']))


def test_the_independent_term_alone_needs_no_power_law_band(psr):
    """Two nodes carry independent offsets, though they cannot support the power-law band."""
    ind = _names(psr)[2]
    with pytest.raises(ValueError, match="Nyquist"):
        s.makegp_fd_piecewise_powerlaw(psr, nodes=2, project_nodes=False)
    gp = s.makegp_fd_piecewise_powerlaw(psr, nodes=2, independent=True, powerlaw=False, project_nodes=False)
    assert np.asarray(gp.Phi.getN({ind: -7.0})).shape == (2, 2)


def test_a_projection_that_leaves_few_directions_warns(psr):
    with pytest.warns(UserWarning, match="Use more nodes"):
        s.makegp_fd_piecewise_powerlaw(psr, nodes=8)
