"""Tests for the red-noise spectrum continued past a Fourier GP as a time-domain kernel."""

from pathlib import Path

import numpy as np
import pytest

import discovery as ds
from discovery import const
from discovery import signals as s
from discovery import likelihood as dl


DATA = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture(scope="module")
def psr():
    f = DATA / "v1p1_de440_pint_bipm2019-B1855+09.feather"
    if not f.exists():
        pytest.skip("pulsar data fixture missing")
    return ds.Pulsar.read_feather(f)


def _grid(psr):
    T = s.getspan(psr)
    K = int(T / (14 * 86400))
    return T, K, (K + 0.5) / T, s.withinday_resolution(psr)


def _P(f, log10_A, gamma):
    return 10 ** (2 * log10_A) / 12 / np.pi ** 2 * const.fyr ** (gamma - 3) * f ** (-gamma)


def _ecorr_names(psr):
    return [f'{psr.name}_{b}_log10_ecorr'
            for b in sorted(set(np.asarray(s.selection_backend_flags(psr)).tolist())) if b != '']


def _params(psr, log10_A=-13.0, gamma=1.6, log10_ecorr=-7.0, extra={}):
    p = {f'{psr.name}_red_noise_log10_A': log10_A, f'{psr.name}_red_noise_gamma': gamma}
    p.update({e: log10_ecorr for e in _ecorr_names(psr)})
    p.update(extra)
    return p


def _model(psr, *gps):
    return dl.PulsarLikelihood([psr.residuals, s.makenoise_measurement(psr, {}),
                                s.makegp_timing(psr, svd=True), *gps])


def _logl(m, psr, p):
    full = {k: (1.0 if k.endswith('efac') else p.get(k, -7.0)) for k in m.logL.params}
    return float(m.logL(full))


# --- the partition --------------------------------------------------------------------

def test_the_tail_starts_at_the_upper_edge_of_the_last_fourier_bin(psr):
    """Bins are centred on k/T with width 1/T, so the last one ends at (K + 1/2)/T."""
    T, K, _, f_res = _grid(psr)
    gp = s.makegp_red_tail(psr, s.powerlaw, K, T)
    f, df, _ = s.fourierbasis(psr, K, T)
    assert gp.tail_band[0] == pytest.approx(f.max() + df[-1] / 2, rel=1e-14)
    assert gp.tail_band[1] == pytest.approx(f_res, rel=1e-14)


def test_the_tail_variance_is_the_integral_of_the_psd_over_its_band(psr):
    _, _, f_lo, f_hi = _grid(psr)
    c = s.make_red_tail(s.powerlaw, f_lo, f_hi)
    for gamma in (1.05, 1.64, 3.0, 7.0):
        k0 = float(c(np.zeros((1, 1)), -13.0, gamma)[0, 0])
        exact = (10 ** -26 / 12 / np.pi ** 2 * const.fyr ** (gamma - 3)
                 * (f_hi ** (1 - gamma) - f_lo ** (1 - gamma)) / (1 - gamma))
        assert k0 == pytest.approx(exact, rel=1e-5)


# --- the kernel ---------------------------------------------------------------------

def test_the_kernel_converges_to_the_continuous_covariance(psr):
    """k(tau) against int P(f) cos(2 pi f tau) df by an oscillatory quadrature, at lags
    from zero to two months, to 1e-3 of k(0) at the default resolution."""
    from scipy.integrate import quad
    _, _, f_lo, f_hi = _grid(psr)
    c = s.make_red_tail(s.powerlaw, f_lo, f_hi)
    edges = np.exp(np.linspace(np.log(f_lo), np.log(f_hi), 200))
    for gamma in (1.05, 1.64):
        def exact(tau):
            kw = dict(epsabs=0, epsrel=1e-10, limit=400)
            if tau:
                kw.update(weight='cos', wvar=2 * np.pi * tau)
            return sum(quad(_P, a, b, args=(-13.0, gamma), **kw)[0]
                       for a, b in zip(edges[:-1], edges[1:]))
        k0 = exact(0.0)
        for tau in (0.0, 600.0, 86400.0, 14 * 86400.0, 60 * 86400.0):
            k = float(c(np.array([[tau]]), -13.0, gamma)[0, 0])
            assert abs(k - exact(tau)) < 1e-3 * k0


def test_the_kernel_is_positive_semidefinite_on_real_epochs(psr):
    T, K, _, _ = _grid(psr)
    gp = s.makegp_withinday(psr, covariance=s.make_red_tail(s.powerlaw, (K + 0.5) / T,
                                                            s.withinday_resolution(psr)),
                            day=np.inf, resolution_day=86400.0, ecorr=False, name='red_tail')
    Phi = np.asarray(gp.Phi.getN(_params(psr)))
    w = np.linalg.eigvalsh(Phi)
    assert w.min() > -1e-10 * w.max()


def test_the_dense_block_equals_a_brute_force_covariance(psr):
    """Every epoch against every other, plus the per-backend ECORR diagonal."""
    T, K, f_lo, f_hi = _grid(psr)
    nf = 64
    gp = s.makegp_red_tail(psr, s.powerlaw, K, T, nf=nf)
    p = _params(psr, log10_ecorr=-6.8)
    Phi = np.asarray(gp.Phi.getN(p))

    edges = np.exp(np.linspace(np.log(f_lo), np.log(f_hi), nf + 1))
    fm, dfk = 0.5 * (edges[1:] + edges[:-1]), np.diff(edges)
    w = dfk / 6 * (_P(edges[:-1], -13.0, 1.6) + 4 * _P(fm, -13.0, 1.6) + _P(edges[1:], -13.0, 1.6))
    t = gp.wd_epochs
    tau = np.abs(t[:, None] - t[None, :])
    K_ = (w[:, None, None] * np.cos(2 * np.pi * fm[:, None, None] * tau)
          * np.sinc(dfk[:, None, None] * tau)).sum(0)
    F = np.asarray(gp.F)
    be = np.asarray(s.selection_backend_flags(psr))
    col_be = [be[F[:, j] > 0][0] for j in range(F.shape[1])]
    K_ += np.diag([(10 ** -6.8) ** 2 for _ in col_be])
    assert len(set(col_be)) > 1
    assert np.allclose(Phi, K_, rtol=1e-10, atol=0.0)


def test_one_block_spans_every_epoch(psr):
    T, K, _, _ = _grid(psr)
    gp = s.makegp_red_tail(psr, s.powerlaw, K, T)
    assert gp.wd_nsingle == 0
    assert gp.wd_blocks == [(0, np.asarray(gp.F).shape[1], False)]


# --- sharing the spectrum -------------------------------------------------------------

def test_the_tail_adds_no_parameters_to_the_red_noise_model(psr):
    T, K, _, _ = _grid(psr)
    red = s.makegp_fourier(psr, s.powerlaw, K, T=T, name='red_noise')
    m = _model(psr, red, s.makegp_red_tail(psr, s.powerlaw, K, T))
    ref = _model(psr, red, s.makegp_ecorr(psr, {}))
    assert sorted(m.logL.params) == sorted(ref.logL.params)


def test_a_tail_with_no_power_reproduces_the_red_plus_ecorr_likelihood(psr):
    """The machinery -- one block over all epochs, ECORR as its diagonal -- is exact."""
    T, K, _, f_res = _grid(psr)

    def zero(tau, log10_A, gamma):
        return 0.0 * tau
    zero.stems = {'log10_A': 'red_noise', 'gamma': 'red_noise'}
    zero.prior_boxes = {}
    zero.f_hi = f_res

    red = s.makegp_fourier(psr, s.powerlaw, K, T=T, name='red_noise')
    m = _model(psr, red, s.makegp_withinday(psr, covariance=zero, day=np.inf,
                                            resolution_day=86400.0, name='red_tail'))
    ref = _model(psr, red, s.makegp_ecorr(psr, {}))
    real = _model(psr, red, s.makegp_red_tail(psr, s.powerlaw, K, T))
    for lA, g in ((-13.5, 1.0), (-12.7, 2.5)):
        p = _params(psr, log10_A=lA, gamma=g)
        a, b = _logl(m, psr, p), _logl(ref, psr, p)
        assert abs(a - b) < 1e-9 * abs(b)
        assert abs(_logl(real, psr, p) - b) > 1.0


def test_the_real_tail_over_a_vanishing_band_reproduces_it_too(psr):
    T, K, _, f_res = _grid(psr)
    red = s.makegp_fourier(psr, s.powerlaw, K, T=T, name='red_noise')
    tail = s.makegp_withinday(psr, covariance=s.make_red_tail(s.powerlaw, f_res / (1 + 1e-7),
                                                              f_res, nf=4),
                              day=np.inf, resolution_day=86400.0, name='red_tail')
    p = _params(psr)
    assert abs(_logl(_model(psr, red, tail), psr, p)
               - _logl(_model(psr, red, s.makegp_ecorr(psr, {})), psr, p)) < 1e-6


def test_the_psd_parameters_move_the_tail(psr):
    """Shared, not merely named alike: the tail's variance follows the red-noise values."""
    T, K, _, _ = _grid(psr)
    gp = s.makegp_red_tail(psr, s.powerlaw, K, T)
    a = np.asarray(gp.Phi.getN(_params(psr, log10_A=-13.0)))
    b = np.asarray(gp.Phi.getN(_params(psr, log10_A=-12.5)))
    e = (10 ** -7.0) ** 2
    assert np.allclose(np.diag(b) - e, 10.0 * (np.diag(a) - e), rtol=1e-10, atol=0.0)


# --- the break ------------------------------------------------------------------------

def test_the_break_reduces_to_one_slope_when_the_indices_agree(psr):
    """Up to the bin holding the break, integrated as two Simpson panels rather than one."""
    _, _, f_lo, f_hi = _grid(psr)
    one = s.make_red_tail(s.powerlaw, f_lo, f_hi)
    two = s.make_red_tail(s.powerlaw, f_lo, f_hi, broken=True)
    tau = np.array([[0.0, 600.0], [3600.0, 86400.0]])
    for lfb in (np.log10(f_lo) + 0.5, np.log10(f_hi) - 0.5):
        assert np.allclose(np.asarray(two(tau, -13.0, 1.6, 1.6, lfb)),
                           np.asarray(one(tau, -13.0, 1.6)), rtol=1e-8, atol=0.0)
        assert not np.allclose(np.asarray(two(tau, -13.0, 1.6, 4.0, lfb)),
                               np.asarray(one(tau, -13.0, 1.6)), rtol=1e-3, atol=0.0)


def test_the_break_is_continuous(psr):
    """At the break the two branches give the same power, so moving it has no jump."""
    _, _, f_lo, f_hi = _grid(psr)
    two = s.make_red_tail(s.powerlaw, f_lo, f_hi, broken=True)
    fm, _ = two.modes
    lfb = float(np.log10(fm[40]))
    below = float(two(np.zeros((1, 1)), -13.0, 1.6, 5.0, lfb - 1e-9)[0, 0])
    above = float(two(np.zeros((1, 1)), -13.0, 1.6, 5.0, lfb + 1e-9)[0, 0])
    assert abs(below - above) < 1e-8 * below


def test_the_break_has_no_kinks_for_a_gradient_sampler(psr):
    """The slope in log10_fbreak is continuous where the break crosses a bin midpoint or a
    bin edge; assigning whole bins to one side of the break leaves a jump of order 1/nf."""
    c = s.make_red_tail(s.powerlaw, 1e-8, 1e-5, nf=128, broken=True)
    fm, dfk = c.modes

    def k0(lfb):
        return float(c(np.zeros((1, 1)), -13.0, 1.0, 4.0, lfb)[0, 0])
    mean = (k0(-5.0 - 1e-3) - k0(-8.0 + 1e-3)) / (3.0 - 2e-3)
    h = 1e-7
    for x in (np.log10(fm[40]), np.log10(fm[40] - dfk[40] / 2)):
        left, right = (k0(x) - k0(x - h)) / h, (k0(x + h) - k0(x)) / h
        assert abs(right - left) < 1e-3 * abs(mean)


def _install_priors(psr, gp):
    """The model's priors for a within-day-family GP: the module boxes and the per-pulsar keys."""
    from discovery.models import mpta
    mpta.update_priordict_standard_mpta()
    mpta._set_withinday_priors(psr, gp)


def test_break_parameters_resolve_to_their_own_boxes(psr):
    """red_noise_gamma's prior key ends in '.*', so a break index under that stem would
    silently take the red-noise box; checked on the box the sampler would resolve."""
    from discovery import prior
    T, K, f_lo, f_hi = _grid(psr)
    _install_priors(psr, s.makegp_red_tail(psr, s.powerlaw, K, T, broken=True))

    def resolved(name):
        return prior.getprior_uniform(name)

    assert resolved(f'{psr.name}_red_tail_gamma_hi') == [0.0, 10.0]
    lo, hi = resolved(f'{psr.name}_red_tail_log10_fbreak')
    assert lo == pytest.approx(np.log10(f_lo)) and hi == pytest.approx(np.log10(f_hi))


def test_the_break_is_bounded_by_fbreak_max(psr):
    from discovery import prior
    T, K, f_lo, f_hi = _grid(psr)
    _install_priors(psr, s.makegp_red_tail(psr, s.powerlaw, K, T, broken=True, fbreak_max=1.0 / 3600.0))
    name = f'{psr.name}_red_tail_log10_fbreak'
    box = prior.getprior_uniform(name)
    assert box[0] == pytest.approx(np.log10(f_lo)) and box[1] == pytest.approx(np.log10(1.0 / 3600.0))
    for bad in (2.0 * f_hi, 0.5 * f_lo):
        with pytest.raises(ValueError, match="fbreak_max"):
            s.make_red_tail(s.powerlaw, f_lo, f_hi, broken=True, fbreak_max=bad)


def _bin_integrals(c, args, split=None):
    from scipy.integrate import quad
    fm, dfk = c.modes
    return np.array([quad(lambda f: float(c.spectrum(np.array([f]), *args)[0]), a, a + d,
                          points=[split] if split is not None and a < split < a + d else None,
                          epsabs=0, epsrel=1e-12)[0] for a, d in zip(fm - dfk / 2, dfk)])


def test_the_spectrum_is_the_power_the_kernel_carries(psr):
    """The bin weights are the spectrum's integrals over the bins and sum to k(0); below
    the tail the spectrum is the Fourier GP's PSD; with a break it is continuous there and
    falls as gamma_hi above, and the bin holding the break is split at it."""
    _, _, f_lo, f_hi = _grid(psr)
    one = s.make_red_tail(s.powerlaw, f_lo, f_hi)
    k0 = float(one(np.zeros((1, 1)), -13.0, 1.6)[0, 0])
    w = np.asarray(one.weights(-13.0, 1.6))
    assert float(np.sum(w)) == pytest.approx(k0, rel=1e-12)
    assert np.allclose(w, _bin_integrals(one, (-13.0, 1.6)), rtol=1e-5, atol=0.0)
    f = np.array([0.1 * f_lo, 0.5 * f_lo])
    assert np.allclose(np.asarray(one.spectrum(f, -13.0, 1.6)), _P(f, -13.0, 1.6), rtol=1e-12, atol=0.0)

    two = s.make_red_tail(s.powerlaw, f_lo, f_hi, broken=True)
    fb = 1.0 / 86400.0
    lo, at, hi = np.asarray(two.spectrum(np.array([fb * (1 - 1e-9), fb, 10 * fb]),
                                         -13.0, 1.6, 3.5, np.log10(fb)))
    assert lo == pytest.approx(at, rel=1e-8)
    assert hi == pytest.approx(at * 10.0 ** -3.5, rel=1e-12)
    for lfb in (np.log10(fb), np.log10(two.modes[0][40]) + 1e-3, np.log10(f_lo) + 0.02):
        args = (-13.0, 1.6, 7.0, lfb)
        w = np.asarray(two.weights(*args))
        assert float(np.sum(w)) == pytest.approx(float(two(np.zeros((1, 1)), *args)[0, 0]), rel=1e-12)
        assert np.allclose(w, _bin_integrals(two, args, split=10 ** lfb), rtol=1e-4, atol=0.0)


def test_the_gradient_reaches_every_tail_parameter(psr):
    import jax
    T, K, _, _ = _grid(psr)
    red = s.makegp_fourier(psr, s.powerlaw, K, T=T, name='red_noise')
    m = _model(psr, red, s.makegp_red_tail(psr, s.powerlaw, K, T, broken=True))
    p = {k: (1.0 if k.endswith('efac') else -6.5 if 'tnequad' in k else -7.0 if 'ecorr' in k
             else -13.0 if k.endswith('log10_A') else 1.6 if k.endswith('_gamma')
             else 3.0 if k.endswith('gamma_hi') else -4.0) for k in m.logL.params}
    g = jax.grad(m.logL)(p)
    for k in (f'{psr.name}_red_noise_log10_A', f'{psr.name}_red_noise_gamma',
              f'{psr.name}_red_tail_gamma_hi', f'{psr.name}_red_tail_log10_fbreak'):
        assert np.isfinite(float(g[k])) and float(g[k]) != 0.0


# --- conditioning and guards -----------------------------------------------------------

def test_the_prior_stays_factorisable_with_ecorr_at_its_floor(psr):
    T, K, _, _ = _grid(psr)
    gp = s.makegp_red_tail(psr, s.powerlaw, K, T)
    for lA in (-18.0, -11.0):
        for g in (0.0, 7.0):
            Phi = np.asarray(gp.Phi.getN(_params(psr, log10_A=lA, gamma=g, log10_ecorr=-10.0)))
            np.linalg.cholesky(Phi)


def test_noisedict_holds_ecorr_fixed_in_the_withinday_prior(psr):
    names = _ecorr_names(psr)
    free = s.makegp_withinday(psr)
    fixed = s.makegp_withinday(psr, noisedict={n: -6.9 for n in names})
    assert set(names) <= set(free.Phi.params)
    assert not set(names) & set(fixed.Phi.params)
    p = {k: (-6.5 if k.endswith('log10_sigma') else 1.0) for k in fixed.Phi.params}
    assert np.allclose(np.asarray(fixed.Phi.getN(p)),
                       np.asarray(free.Phi.getN({**p, **{n: -6.9 for n in names}})), rtol=1e-12, atol=0.0)
    assert not np.allclose(np.asarray(fixed.Phi.getN(p)),
                           np.asarray(free.Phi.getN({**p, **{n: -6.5 for n in names}})), rtol=1e-3, atol=0.0)


def test_no_band_left_raises(psr):
    T, _, _, f_res = _grid(psr)
    with pytest.raises(ValueError, match="no band left"):
        s.makegp_red_tail(psr, s.powerlaw, int(2 * f_res * T), T)


def test_single_pulsar_noise_builds_the_tail_in_place_of_ecorr(psr):
    from discovery.models import mpta
    kw = dict(fftint=False, dm=False, chrom=False, sw=False, chrom_poly=False, background=False)
    ref = mpta.single_pulsar_noise(psr, **kw)
    m = mpta.single_pulsar_noise(psr, red_tail=True, **kw)
    assert sorted(m.logL.params) == sorted(ref.logL.params)
    broken = mpta.single_pulsar_noise(psr, red_tail=True, red_tail_broken=True,
                                      red_tail_fbreak_max=1.0 / 3600.0, **kw)
    assert set(broken.logL.params) - set(ref.logL.params) == {
        f'{psr.name}_red_tail_gamma_hi', f'{psr.name}_red_tail_log10_fbreak'}
    from discovery import prior
    name = f'{psr.name}_red_tail_log10_fbreak'
    box = prior.getprior_uniform(name)
    assert box[1] == pytest.approx(np.log10(1.0 / 3600.0))
    with pytest.raises(ValueError, match="fftint=False"):
        mpta.single_pulsar_noise(psr, red_tail=True, fftint=True)
    with pytest.raises(ValueError, match="cannot be combined with withinday"):
        mpta.single_pulsar_noise(psr, red_tail=True, withinday=True, **kw)
    with pytest.raises(ValueError, match="Legendre"):
        mpta.single_pulsar_noise(psr, red_tail=True, ecorr_nmodes=2, **kw)


def test_ecorr_correlated_without_modes_is_inert_under_the_tail(psr):
    """ecorr_correlated selects a Legendre variant only when ecorr_nmodes is set, as
    in the ECORR branch the tail replaces."""
    from discovery.models import mpta
    kw = dict(fftint=False, dm=False, chrom=False, sw=False, chrom_poly=False, background=False,
              red_tail=True)
    a = mpta.single_pulsar_noise(psr, **kw)
    b = mpta.single_pulsar_noise(psr, ecorr_correlated=True, **kw)
    assert sorted(a.logL.params) == sorted(b.logL.params)
    p = {k: (1.0 if k.endswith('efac') else -13.0 if k.endswith('log10_A')
             else 1.6 if k.endswith('gamma') else -7.0) for k in a.logL.params}
    assert float(a.logL(p)) == float(b.logL(p))


# --- options threaded through single_pulsar_noise ---------------------------------------

def _spn(psr, **extra):
    from discovery.models import mpta
    kw = dict(fftint=False, dm=False, chrom=False, sw=False, chrom_poly=False, background=False,
              red_tail=True, return_components=True)
    m, comps = mpta.single_pulsar_noise(psr, **{**kw, **extra})
    return m, [c for c in comps if hasattr(c, 'tail_band')][0], comps


def _fill(params, **fixed):
    return {k: fixed.get(k, 1.0 if k.endswith('efac') else -6.5 if 'equad' in k
                         else -13.5 if k.endswith('log10_A') else 2.0 if k.endswith('gamma') else -7.0)
            for k in params}


def test_the_tail_basis_takes_ecorr_tobs_scale_as_makegp_ecorr_does(psr):
    _, tail, _ = _spn(psr, ecorr_tobs_scale=True)
    weighted = np.asarray(s.makegp_ecorr(psr, {}, tobs_scale=True).F)
    assert np.array_equal(np.asarray(tail.F), weighted)
    assert not np.array_equal(weighted, np.asarray(s.makegp_ecorr(psr, {}).F))


def test_noisedict_holds_ecorr_fixed_through_single_pulsar_noise(psr):
    names = _ecorr_names(psr)
    fixed, _, _ = _spn(psr, noisedict={n: -6.9 for n in names})
    free, _, _ = _spn(psr)
    assert not set(names) & set(fixed.logL.params) and set(names) <= set(free.logL.params)
    p = _fill(fixed.logL.params)
    assert float(fixed.logL(p)) == pytest.approx(float(free.logL({**p, **{n: -6.9 for n in names}})), rel=1e-12)
    assert float(fixed.logL(p)) != pytest.approx(float(free.logL({**p, **{n: -6.5 for n in names}})), rel=1e-9)


def test_a_turnover_psd_is_the_tails_too(psr):
    """One log10_fc, shared, and it moves the tail when the corner lies inside its band."""
    m, tail, _ = _spn(psr, turnover='red')
    fc = [k for k in m.logL.params if k.endswith('log10_fc')]
    assert fc == [f'{psr.name}_red_noise_log10_fc'] and fc[0] in tail.Phi.params
    p = _fill(tail.Phi.params)
    below, inside = (np.asarray(tail.Phi.getN({**p, fc[0]: v})) for v in (-9.0, -4.0))
    assert not np.allclose(below, inside, rtol=1e-3, atol=0.0)


def test_the_tail_starts_at_the_red_gps_own_last_bin_with_max_cadence_none(psr):
    _, tail, comps = _spn(psr, max_cadence_days=None)
    red = [c for c in comps if getattr(c, 'gpname', None) == 'red_noise']
    assert len(red) == 1
    K, T = np.asarray(red[0].F).shape[1] // 2, s.getspan(psr)
    assert K == s.psr_components(psr.toas, dt=1.0, Tspan=T) != int(T / (14 * 86400))
    assert tail.tail_band[0] == pytest.approx((K + 0.5) / T, rel=1e-14)
