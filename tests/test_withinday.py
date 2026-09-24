"""Tests for the within-day correlated GP that replaces the diagonal ECORR prior."""

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


def _pars(psr, log10_sigma=-6.8, gamma=1.0, log10_ecorr=-7.0):
    gp = s.makegp_withinday(psr)
    return {p: (log10_sigma if p.endswith('log10_sigma') else
                gamma if p.endswith('gamma') else log10_ecorr) for p in gp.Phi.params}


def test_basis_is_the_ecorr_basis(psr):
    """Same epoch columns as makegp_ecorr, only reordered so day blocks are contiguous."""
    wd = np.asarray(s.makegp_withinday(psr).F)
    ec = np.asarray(s.makegp_ecorr(psr, {}).F)
    assert wd.shape == ec.shape
    assert np.all(wd.sum(axis=1) == 1)
    a = sorted(tuple(np.flatnonzero(c)) for c in wd.T)
    b = sorted(tuple(np.flatnonzero(c)) for c in ec.T)
    assert a == b


def test_ecorr_parameter_names_are_unchanged(psr):
    """A chain fitted with makegp_ecorr supplies the diagonal directly."""
    wd = set(s.makegp_withinday(psr).Phi.params)
    ec = set(s.makegp_ecorr(psr, {}).Phi.getN.params)
    assert ec < wd
    assert wd - ec == {f'{psr.name}_withinday_gp_log10_sigma',
                       f'{psr.name}_withinday_gp_gamma'}


def test_the_basis_carries_no_frequency_weight(psr):
    """Achromatic by construction: TOAs sharing an epoch get identical basis rows."""
    gp = s.makegp_withinday(psr)
    F = np.asarray(gp.F)
    bins = s.quantize(np.asarray(psr.toas), 1.0)
    freq = np.asarray(psr.freqs)
    for b in np.unique(bins)[:20]:
        m = bins == b
        if m.sum() < 2 or len(set(freq[m])) < 2:
            continue
        assert np.all(F[m] == F[m][0])


def test_prior_is_block_diagonal_by_day(psr):
    """Zero correlation between days, by construction."""
    gp = s.makegp_withinday(psr)
    Phi = np.asarray(gp.Phi.getN(_pars(psr)))
    day = np.floor(gp.wd_epochs / 86400.0)
    cross = day[:, None] != day[None, :]
    assert np.abs(Phi[cross]).max() == 0.0
    assert np.allclose(Phi, Phi.T, rtol=1e-12, atol=0.0)


def test_block_diagonal_factorisation_is_exact(psr):
    """The cheap per-block inverse must equal the dense one it replaces."""
    gp = s.makegp_withinday(psr)
    p = _pars(psr)
    Phi = np.asarray(gp.Phi.getN(p))
    Pinv, logdet = gp.Phi.make_inv()(p)
    _, dense = np.linalg.slogdet(Phi)
    assert abs(float(logdet) - dense) < 1e-6 * abs(dense)
    assert np.allclose(np.asarray(Pinv) @ Phi, np.eye(Phi.shape[0]), atol=1e-6)


def test_single_epoch_days_become_one_diagonal_chunk(psr):
    """They carry only the diagonal, so factorising them one by one is wasted work."""
    gp = s.makegp_withinday(psr)
    day = np.floor(gp.wd_epochs / 86400.0)
    _, inv = np.unique(day, return_inverse=True)
    singles = int(np.sum(np.bincount(inv)[inv] == 1))
    assert gp.wd_nsingle == singles
    diag = [b for b in gp.wd_blocks if b[2]]
    assert len(diag) == (1 if singles else 0)
    if singles:
        assert diag[0] == (0, singles, True)
    assert sum(n for _, n, _ in gp.wd_blocks) == np.asarray(gp.F).shape[1]


def test_sigma_is_the_within_day_rms_whatever_the_slope(psr):
    """k(0) = sigma**2, so log10_sigma and gamma are not degenerate through the variance."""
    k = s.make_withinday_powerlaw(f_hi=1 / 120.0)
    tau = np.zeros((1, 1))
    for gamma in (0.0, 1.0, 2.5):
        assert abs(float(k(tau, -6.5, gamma)[0, 0]) - (10 ** -6.5) ** 2) < 1e-20


def test_a_steeper_slope_decorrelates_more_slowly(psr):
    """gamma is the PSD slope, so more low-frequency power holds correlation to longer lag.

    Measured on the decorrelation timescale, not on rho at one lag: with little
    low-frequency power rho oscillates about zero and is not monotone in gamma pointwise.
    """
    k = s.make_withinday_powerlaw(f_hi=1 / 120.0)
    taus = np.logspace(0.5, 4.9, 300)
    half = []
    for g in (0.0, 1.0, 2.0, 3.0):
        rho = np.array([float(k(np.array([[t]]), -6.5, g)[0, 0]) for t in taus]) / (10 ** -6.5) ** 2
        half.append(taus[int(np.argmax(rho < 0.5))])
    assert half[0] < half[1] < half[2] < half[3]
    assert half[1] < 3600.0 < half[3]


def test_the_kernel_is_positive_semidefinite_on_real_epochs(psr):
    """It is C diag(w) C^T + S diag(w) S^T, so PSD holds whatever the sampling."""
    gp = s.makegp_withinday(psr, ecorr=False)
    Phi = np.asarray(gp.Phi.getN(_pars(psr)))
    w = np.linalg.eigvalsh(Phi)
    assert w.min() > -1e-12 * w.max()


def test_the_ecorr_diagonal_is_the_nugget_a_densely_sampled_day_needs(psr):
    """The bare kernel loses rank once a day holds more epochs than it spans.

    The bundled fixture never reaches that regime (7 epochs in its busiest day), so the
    threshold is shown on a constructed day instead -- which is the case that matters,
    since J1902-5105 reaches 174.
    """
    k = s.make_withinday_powerlaw(f_hi=1 / 120.0)
    nugget = (93.1e-9) ** 2
    cond = {}
    for n in (4, 16, 64, 174):
        t = np.arange(n) * 120.0
        K = np.asarray(k(np.abs(t[:, None] - t[None, :]), -6.5, 1.0))
        w = np.linalg.eigvalsh(K)
        wn = np.linalg.eigvalsh(K + np.eye(n) * nugget)
        cond[n] = (w.max() / max(w.min(), 1e-300), wn.max() / wn.min())
    assert cond[4][0] < 1e3 and cond[16][0] < 1e3
    assert cond[64][0] > 1e6 and cond[174][0] > 1e6
    assert all(c[1] < 1e4 for c in cond.values())

    with pytest.raises(np.linalg.LinAlgError):
        t = np.arange(174) * 120.0
        np.linalg.cholesky(np.asarray(k(np.abs(t[:, None] - t[None, :]), -6.5, 1.0)))


def test_prior_boxes_are_registered(psr):
    import re
    from discovery import prior
    s.makegp_withinday(psr)
    e = re.escape(psr.name)
    assert prior.priordict_standard[f'{e}_withinday_gp_log10_sigma'] == [-9.0, -5.5]
    assert prior.priordict_standard[f'{e}_withinday_gp_gamma'] == [0.0, 4.0]


def test_a_custom_covariance_is_used_and_names_its_own_parameters(psr):
    gp = s.makegp_withinday(psr, covariance=s.squared_exponential)
    assert f'{psr.name}_withinday_gp_log10_ell' in gp.Phi.params
    assert f'{psr.name}_withinday_gp_log10_sigma' in gp.Phi.params


def test_it_evaluates_in_a_likelihood(psr):
    gp = s.makegp_withinday(psr)
    m = dl.PulsarLikelihood([psr.residuals, s.makenoise_measurement(psr, {}),
                             s.makegp_timing(psr, svd=True), gp])
    base = {k: (1.0 if k.endswith('efac') else -6.8 if k.endswith('log10_sigma')
                else 1.0 if k.endswith('gamma') else -7.0) for k in m.logL.params}
    lo = float(m.logL({**base, f'{psr.name}_withinday_gp_log10_sigma': -12.0}))
    hi = float(m.logL(base))
    assert np.isfinite(lo) and np.isfinite(hi)
    assert abs(hi - lo) > 1.0


def test_single_pulsar_noise_swaps_ecorr_for_the_within_day_prior(psr):
    from discovery.models import mpta

    plain = mpta.single_pulsar_noise(psr, fftint=False, dm=False, chrom=False, sw=False,
                                     chrom_poly=False, background=False, red=False)
    wd = mpta.single_pulsar_noise(psr, fftint=False, dm=False, chrom=False, sw=False,
                                  chrom_poly=False, background=False, red=False, withinday=True)
    assert f'{psr.name}_withinday_gp_log10_sigma' in wd.logL.params
    assert f'{psr.name}_withinday_gp_log10_sigma' not in plain.logL.params
    ec = {p for p in plain.logL.params if p.endswith('log10_ecorr')}
    assert ec and ec <= set(wd.logL.params)


def test_single_pulsar_noise_refuses_the_legendre_combination(psr):
    from discovery.models import mpta

    with pytest.raises(ValueError, match="cannot be combined with the Legendre"):
        mpta.single_pulsar_noise(psr, fftint=False, withinday=True, ecorr_nmodes=3)


# --- broken power law -------------------------------------------------------------

def _bp(psr=None, **kw):
    return s.make_withinday_brokenpowerlaw(**kw)


def test_broken_reduces_to_the_single_power_law_when_the_slopes_are_equal(psr):
    """The extra degree of freedom must be exactly that, and nothing else."""
    band = dict(f_lo=1.0 / 86400.0, f_hi=1.0 / 16.0, nf=96)
    one = s.make_withinday_powerlaw(**band)
    two = s.make_withinday_brokenpowerlaw(**band)
    tau = np.array([[0.0, 600.0], [600.0, 0.0]])
    for g in (1.5, 2.55, 3.0):
        a = np.asarray(one(tau, -6.3, g))
        b = np.asarray(two(tau, -6.3, g, g, -3.0))
        assert np.allclose(a, b, rtol=1e-12, atol=0.0)


def test_broken_normalisation_is_the_variance(psr):
    two = s.make_withinday_brokenpowerlaw(f_hi=1 / 16.0)
    for g_lo, g_hi in ((2.55, 1.5), (1.0, 3.0)):
        k0 = float(two(np.zeros((1, 1)), -6.3, g_lo, g_hi, -3.0)[0, 0])
        assert abs(k0 - (10 ** -6.3) ** 2) < 1e-24


def test_the_low_slope_sets_the_long_lag_correlation(psr):
    """gamma_lo is the slope below the break, so it governs the decorrelation timescale."""
    two = s.make_withinday_brokenpowerlaw(f_hi=1 / 16.0)
    taus = np.logspace(1, 5, 300)
    half = []
    for g_lo in (2.0, 2.5, 3.0, 3.5):
        rho = np.array([float(two(np.array([[t]]), -6.3, g_lo, 1.5, np.log10(1 / 1200.))[0, 0])
                        for t in taus]) / (10 ** -6.3) ** 2
        half.append(taus[int(np.argmax(rho < 0.5))])
    assert half[0] < half[1] < half[2] < half[3]


def test_the_high_slope_sets_the_short_lag_correlation(psr):
    """gamma_hi is what prices the power that would otherwise be absorbed by ECORR."""
    two = s.make_withinday_brokenpowerlaw(f_hi=1 / 16.0)
    rho = [float(two(np.array([[30.0]]), -6.3, 2.55, g, np.log10(1 / 1200.))[0, 0]) / (10 ** -6.3) ** 2
           for g in (0.5, 1.0, 1.5, 2.0)]
    assert rho[0] < rho[1] < rho[2] < rho[3]


def test_the_break_prior_stays_inside_the_modelled_band(psr):
    """Outside it one of the two slopes stops being identified."""
    two = s.make_withinday_brokenpowerlaw(f_lo=1 / 86400.0, f_hi=1 / 16.0)
    lo, hi = two.prior_boxes['log10_fbreak']
    assert np.log10(1 / 86400.0) < lo < hi < np.log10(1 / 16.0)


def test_both_kernels_default_to_the_band_the_epochs_resolve(psr):
    """f_hi is derived from the data, not chosen: None binds it in makegp_withinday."""
    import inspect
    for factory in (s.make_withinday_powerlaw, s.make_withinday_brokenpowerlaw):
        assert inspect.signature(factory).parameters['f_hi'].default is None
        assert getattr(factory(), 'unbound', False)


def test_the_covariance_supplies_its_own_prior_boxes(psr):
    import re
    from discovery import prior
    gp = s.makegp_withinday(psr, covariance=s.make_withinday_brokenpowerlaw())
    e = re.escape(psr.name)
    for arg in ('log10_sigma', 'gamma_lo', 'gamma_hi', 'log10_fbreak'):
        assert f'{psr.name}_withinday_gp_{arg}' in gp.Phi.params
        assert f'{e}_withinday_gp_{arg}' in prior.priordict_standard


def test_the_broken_kernel_evaluates_in_a_likelihood(psr):
    gp = s.makegp_withinday(psr, covariance=s.make_withinday_brokenpowerlaw())
    m = dl.PulsarLikelihood([psr.residuals, s.makenoise_measurement(psr, {}),
                             s.makegp_timing(psr, svd=True), gp])
    base = {k: (1.0 if k.endswith('efac') else -6.3 if k.endswith('log10_sigma')
                else 2.55 if k.endswith('gamma_lo') else 1.5 if k.endswith('gamma_hi')
                else -3.0 if k.endswith('log10_fbreak') else -7.0) for k in m.logL.params}
    assert np.isfinite(float(m.logL(base)))



# --- the band edge set from the epoch sampling ---------------------------------------

def test_resolution_is_half_the_median_within_day_epoch_gap(psr):
    """Checked against an independent construction of the epochs."""
    t_ = np.asarray(psr.toas)
    be = np.asarray(s.selection_backend_flags(psr))
    epochs = []
    for b in sorted(set(be.tolist())):
        if b == '':
            continue
        bins = s.quantize(t_ * (be == b), 1.0)
        first = 0 if len(set(be.tolist())) == 1 else 1
        for i in range(first, bins.max() + 1):
            m = bins == i
            if m.any():
                epochs.append(t_[m].mean())
    epochs = np.sort(np.array(epochs))
    day = np.floor(epochs / 86400.0)
    gaps = np.concatenate([np.diff(epochs[day == d]) for d in np.unique(day)
                           if (day == d).sum() > 1])
    assert np.isclose(s.withinday_resolution(psr), 1.0 / (2.0 * np.median(gaps)), rtol=1e-8, atol=0.0)


def test_the_default_kernel_is_bound_to_the_resolution(psr):
    gp = s.makegp_withinday(psr)
    assert np.isclose(gp.wd_fhi, s.withinday_resolution(psr), rtol=1e-12, atol=0.0)


def test_an_unbound_kernel_refuses_direct_evaluation(psr):
    with pytest.raises(ValueError, match="f_hi is None"):
        s.make_withinday_powerlaw()(np.zeros((1, 1)), -6.3, 2.55)
    with pytest.raises(ValueError, match="f_hi is None"):
        s.make_withinday_brokenpowerlaw()(np.zeros((1, 1)), -6.3, 2.55, 1.5, -3.0)


def test_a_bound_kernel_is_the_explicit_one_at_the_resolution(psr):
    """Binding must build exactly the kernel an explicit f_hi would."""
    f = s.withinday_resolution(psr)
    bound = s.make_withinday_powerlaw().bind(f)
    explicit = s.make_withinday_powerlaw(f_hi=f)
    tau = np.array([[0.0, 900.0], [900.0, 0.0]])
    assert np.array_equal(np.asarray(bound(tau, -6.3, 2.55)), np.asarray(explicit(tau, -6.3, 2.55)))


def test_a_kernel_past_the_resolution_warns(psr, capsys):
    """Above the edge the kernel is degenerate with ECORR, so its prior sets the split."""
    s.makegp_withinday(psr, covariance=s.make_withinday_powerlaw(f_hi=1 / 16.0))
    assert 'degenerate with ECORR' in capsys.readouterr().out
    s.makegp_withinday(psr, covariance=s.make_withinday_powerlaw(f_hi=s.withinday_resolution(psr)))
    assert 'degenerate with ECORR' not in capsys.readouterr().out


def test_a_band_with_no_width_raises(psr):
    with pytest.raises(ValueError, match="has no band"):
        s.make_withinday_powerlaw(f_lo=1 / 600.0, f_hi=1 / 3600.0)
    with pytest.raises(ValueError, match="has no band"):
        s.make_withinday_brokenpowerlaw(f_lo=1 / 600.0, f_hi=1 / 3600.0)


def test_a_pulsar_with_no_multi_epoch_day_raises(psr):
    """There nothing within a day is measurable and the component is exactly ECORR."""
    class OneEpochPerDay:
        name = psr.name
        toas = np.arange(20) * 86400.0 + 3600.0
        freqs = np.full(20, 1400.0)
        backend_flags = np.array(['X'] * 20)
        flags = {}
    with pytest.raises(ValueError, match="no day holding more than one epoch"):
        s.withinday_resolution(OneEpochPerDay(), selection=lambda p: p.backend_flags)
