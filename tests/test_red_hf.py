"""Tests for the achromatic process above a sampled outer scale, correlated across all epochs."""

from pathlib import Path

import numpy as np
import pytest

import discovery as ds
from discovery import signals as s
from discovery import likelihood as dl


DATA = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture(scope="module")
def psr():
    f = DATA / "v1p1_de440_pint_bipm2019-B1855+09.feather"
    if not f.exists():
        pytest.skip("pulsar data fixture missing")
    return ds.Pulsar.read_feather(f)


def _range(psr):
    T = s.getspan(psr)
    K = int(T / (14 * 86400))
    return T, K, ((K + 0.5) / T, 1.0 / 3600.0), s.withinday_resolution(psr)


def _ecorr_names(psr):
    return [f'{psr.name}_{b}_log10_ecorr'
            for b in sorted(set(np.asarray(s.selection_backend_flags(psr)).tolist())) if b != '']


def _params(psr, log10_sigma=-6.5, gamma=2.5, log10_flo=-5.5, log10_ecorr=-7.0):
    p = {f'{psr.name}_red_hf_log10_sigma': log10_sigma, f'{psr.name}_red_hf_gamma': gamma,
         f'{psr.name}_red_hf_log10_flo': log10_flo}
    p.update({e: log10_ecorr for e in _ecorr_names(psr)})
    return p


def _model(psr, *gps):
    return dl.PulsarLikelihood([psr.residuals, s.makenoise_measurement(psr, {}),
                                s.makegp_timing(psr, svd=True), *gps])


def _logl(m, p):
    full = {k: (1.0 if k.endswith('efac') else p.get(k, -13.5 if k.endswith('log10_A')
                                                    else 2.0 if k.endswith('gamma') else -7.0))
            for k in m.logL.params}
    return float(m.logL(full))


def _Pnorm(f, gamma, f_lo, f_hi):
    """f**-gamma on [f_lo, f_hi], normalised to unit integral."""
    from scipy.integrate import quad
    return f ** (-gamma) / quad(lambda x: x ** (-gamma), f_lo, f_hi, limit=200)[0]


# --- the kernel ---------------------------------------------------------------------

def test_the_variance_is_sigma_squared_for_any_slope_and_outer_scale(psr):
    _, _, rng, f_hi = _range(psr)
    c = s.make_band_powerlaw(rng, f_hi)
    for g in (0.0, 2.5, 7.0):
        for lflo in (np.log10(rng[0]), -5.0, np.log10(rng[1])):
            k0 = float(c(np.zeros((1, 1)), -6.5, g, lflo)[0, 0])
            assert k0 == pytest.approx(10 ** -13.0, rel=1e-12)


def test_the_kernel_converges_to_the_continuous_covariance(psr):
    """k(tau) against sigma^2 int P(f) cos(2 pi f tau) df, at lags from zero to two
    months and outer scales across the box, to 2e-3 of k(0) at the default resolution."""
    from scipy.integrate import quad
    _, _, rng, f_hi = _range(psr)
    c = s.make_band_powerlaw(rng, f_hi)
    for g in (1.0, 3.0):
        for f_lo in (rng[0], 1.0 / 86400.0, rng[1]):
            edges = np.exp(np.linspace(np.log(f_lo), np.log(f_hi), 200))

            def exact(tau):
                kw = dict(epsabs=0, epsrel=1e-10, limit=400)
                if tau:
                    kw.update(weight='cos', wvar=2 * np.pi * tau)
                return sum(quad(_Pnorm, a, b, args=(g, f_lo, f_hi), **kw)[0]
                           for a, b in zip(edges[:-1], edges[1:]))
            for tau in (0.0, 600.0, 3 * 3600.0, 86400.0, 14 * 86400.0, 60 * 86400.0):
                k = float(c(np.array([[tau]]), 0.0, g, np.log10(f_lo))[0, 0])
                assert abs(k - exact(tau)) < 2e-3


def test_the_spectrum_integrates_to_the_variance_and_vanishes_below_the_outer_scale(psr):
    _, _, rng, f_hi = _range(psr)
    c = s.make_band_powerlaw(rng, f_hi)
    lflo = -5.2
    fm, dfk = (np.asarray(x) for x in c.modes_at(lflo))
    S = np.asarray(c.spectrum(fm, -6.5, 2.5, lflo))
    assert float(np.sum(S * dfk)) == pytest.approx(10 ** -13.0, rel=1e-12)
    below = np.asarray(c.spectrum(np.array([0.5 * 10 ** lflo, 0.99 * 10 ** lflo]), -6.5, 2.5, lflo))
    assert np.all(below == 0.0)
    a, b = np.asarray(c.spectrum(np.array([2e-5, 2e-4]), -6.5, 2.5, lflo))
    above = np.asarray(c.spectrum(np.array([1.01 * f_hi]), -6.5, 2.5, lflo))
    assert np.all(above == 0.0)
    assert b / a == pytest.approx(10.0 ** -2.5, rel=1e-12)


def test_the_outer_scale_sets_where_correlation_is_lost(psr):
    """Raising f_lo removes the slow power, so correlation at a day's lag falls."""
    _, _, rng, f_hi = _range(psr)
    c = s.make_band_powerlaw(rng, f_hi)
    tau = np.array([[0.0, 86400.0]])
    wide = np.asarray(c(tau, -6.5, 2.5, np.log10(rng[0])))[0]
    narrow = np.asarray(c(tau, -6.5, 2.5, np.log10(rng[1])))[0]
    assert wide[1] / wide[0] > 0.5 and abs(narrow[1] / narrow[0]) < 0.05


def test_the_kernel_is_positive_semidefinite_on_real_epochs(psr):
    _, _, rng, _ = _range(psr)
    gp = s.makegp_withinday(psr, covariance=s.make_band_powerlaw(rng, s.withinday_resolution(psr)),
                            day=np.inf, resolution_day=86400.0, ecorr=False, name='red_hf')
    for g, lflo in ((0.0, np.log10(rng[1])), (2.5, np.log10(rng[0])), (7.0, -5.0)):
        w = np.linalg.eigvalsh(np.asarray(gp.Phi.getN(_params(psr, gamma=g, log10_flo=lflo))))
        assert w.min() > -1e-10 * w.max()


def test_the_dense_block_equals_a_brute_force_covariance(psr):
    _, _, rng, f_hi = _range(psr)
    nf, lflo = 64, -5.3
    gp = s.makegp_red_hf(psr, rng, nf=nf)
    Phi = np.asarray(gp.Phi.getN(_params(psr, log10_flo=lflo, log10_ecorr=-6.8)))

    edges = np.exp(np.linspace(lflo * np.log(10.0), np.log(f_hi), nf + 1))
    fm, dfk = 0.5 * (edges[1:] + edges[:-1]), np.diff(edges)
    w = fm ** -2.5 * dfk
    w *= 10 ** -13.0 / w.sum()
    tau = np.abs(gp.wd_epochs[:, None] - gp.wd_epochs[None, :])
    K = (w[:, None, None] * np.cos(2 * np.pi * fm[:, None, None] * tau)
         * np.sinc(dfk[:, None, None] * tau)).sum(0) + np.eye(len(tau)) * (10 ** -6.8) ** 2
    assert np.allclose(Phi, K, rtol=1e-10, atol=0.0)


def test_one_block_spans_every_epoch(psr):
    _, _, rng, _ = _range(psr)
    gp = s.makegp_red_hf(psr, rng)
    assert gp.wd_nsingle == 0
    assert gp.wd_blocks == [(0, np.asarray(gp.F).shape[1], False)]


# --- the model ------------------------------------------------------------------------

def _install_priors(psr, gp):
    """The model's priors for a within-day-family GP: the module boxes and the per-pulsar keys."""
    from discovery.models import mpta
    mpta.update_priordict_standard_mpta()
    mpta._set_withinday_priors(psr, gp)


def test_it_adds_three_parameters_with_their_own_boxes(psr):
    from discovery import prior
    T, K, rng, _ = _range(psr)
    red = s.makegp_fourier(psr, s.powerlaw, K, T=T, name='red_noise')
    hf = s.makegp_red_hf(psr, rng)
    _install_priors(psr, hf)
    m = _model(psr, red, hf)
    ref = _model(psr, red, s.makegp_ecorr(psr, {}))
    new = {f'{psr.name}_red_hf_log10_sigma', f'{psr.name}_red_hf_gamma', f'{psr.name}_red_hf_log10_flo'}
    assert set(m.logL.params) - set(ref.logL.params) == new
    assert set(ref.logL.params) <= set(m.logL.params)

    def resolved(name):
        return prior.getprior_uniform(name)
    assert resolved(f'{psr.name}_red_hf_log10_sigma') == [-9.0, -5.0]
    assert resolved(f'{psr.name}_red_hf_gamma') == [0.0, 7.0]
    lo, hi = resolved(f'{psr.name}_red_hf_log10_flo')
    assert lo == pytest.approx(np.log10(rng[0])) and hi == pytest.approx(np.log10(rng[1]))


def test_a_vanishing_process_reproduces_the_red_plus_ecorr_likelihood(psr):
    T, K, rng, _ = _range(psr)
    red = s.makegp_fourier(psr, s.powerlaw, K, T=T, name='red_noise')
    m = _model(psr, red, s.makegp_red_hf(psr, rng))
    ref = _model(psr, red, s.makegp_ecorr(psr, {}))
    b = _logl(ref, _params(psr))
    assert abs(_logl(m, _params(psr, log10_sigma=-16.0)) - b) < 1e-6
    assert abs(_logl(m, _params(psr, log10_sigma=-6.5)) - b) > 1.0


def test_the_gradient_reaches_every_parameter(psr):
    import jax
    T, K, rng, _ = _range(psr)
    red = s.makegp_fourier(psr, s.powerlaw, K, T=T, name='red_noise')
    m = _model(psr, red, s.makegp_red_hf(psr, rng))
    p = {k: (1.0 if k.endswith('efac') else -6.5 if 'equad' in k else -7.0 if 'ecorr' in k
             else -13.5 if k.endswith('log10_A') else 2.0 if k.endswith('red_noise_gamma')
             else -6.5 if k.endswith('log10_sigma') else 2.5 if k.endswith('red_hf_gamma')
             else -5.3) for k in m.logL.params}
    g = jax.grad(m.logL)(p)
    for k in (f'{psr.name}_red_hf_log10_sigma', f'{psr.name}_red_hf_gamma',
              f'{psr.name}_red_hf_log10_flo'):
        assert np.isfinite(float(g[k])) and float(g[k]) != 0.0


def test_the_prior_stays_factorisable_with_ecorr_at_its_floor(psr):
    _, _, rng, _ = _range(psr)
    gp = s.makegp_red_hf(psr, rng)
    for ls in (-9.0, -5.0):
        for g in (0.0, 2.5, 7.0):
            for lflo in (np.log10(rng[0]), np.log10(rng[1])):
                np.linalg.cholesky(np.asarray(gp.Phi.getN(
                    _params(psr, log10_sigma=ls, gamma=g, log10_flo=lflo, log10_ecorr=-10.0))))


def test_the_outer_scale_range_must_lie_below_the_resolution(psr):
    f_res = s.withinday_resolution(psr)
    with pytest.raises(ValueError, match="not above the top of the outer-scale range"):
        s.makegp_red_hf(psr, (1e-6, 2.0 * f_res))
    with pytest.raises(ValueError, match="must lie below"):
        s.make_band_powerlaw((1e-4, 1e-5), f_res)


def test_single_pulsar_noise_builds_it_in_place_of_ecorr(psr):
    from discovery.models import mpta
    from discovery import prior
    kw = dict(fftint=False, dm=False, chrom=False, sw=False, chrom_poly=False, background=False)
    ref = mpta.single_pulsar_noise(psr, **kw)
    m = mpta.single_pulsar_noise(psr, red_hf=True, **kw)
    assert set(m.logL.params) - set(ref.logL.params) == {
        f'{psr.name}_red_hf_log10_sigma', f'{psr.name}_red_hf_gamma', f'{psr.name}_red_hf_log10_flo'}
    T, K, rng, _ = _range(psr)
    name = f'{psr.name}_red_hf_log10_flo'
    lo, hi = prior.getprior_uniform(name)
    assert lo == pytest.approx(np.log10((K + 0.5) / T)) and hi == pytest.approx(np.log10(1.0 / 3600.0))
    for bad, match in ((dict(fftint=True), "fftint=False"), (dict(red_tail=True), "red_tail"),
                       (dict(withinday=True), "withinday"), (dict(ecorr_nmodes=2), "Legendre")):
        with pytest.raises(ValueError, match=match):
            mpta.single_pulsar_noise(psr, red_hf=True, **{**kw, **bad})


def test_the_outer_scale_box_follows_the_red_gps_grid_with_max_cadence_none(psr):
    from discovery.models import mpta
    from discovery import prior
    _, comps = mpta.single_pulsar_noise(psr, fftint=False, dm=False, chrom=False, sw=False,
                                        chrom_poly=False, background=False, red_hf=True,
                                        max_cadence_days=None, return_components=True)
    red = [c for c in comps if getattr(c, 'gpname', None) == 'red_noise']
    K, T = np.asarray(red[0].F).shape[1] // 2, s.getspan(psr)
    assert K != int(T / (14 * 86400))
    lo, _ = prior.getprior_uniform(f'{psr.name}_red_hf_log10_flo')
    assert lo == pytest.approx(np.log10((K + 0.5) / T), rel=1e-12)
