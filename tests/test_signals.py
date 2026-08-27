"""Tests for make_combined_crn signature merging and numerical correctness."""

import inspect
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import pytest

import discovery as ds
from discovery.signals import make_combined_crn


# A PSD with non-overlapping parameter names, for testing the no-rename path.
def _alt_psd(f, df, alpha, log10_ref):
    return (10.0 ** (2.0 * log10_ref)) * f ** (-alpha) * df


def _make_freqs(n_total=30, tspan_years=20):
    """Return (f, df) arrays with sin/cos pairs (2*n_total elements)."""
    tspan = tspan_years * 365.25 * 86400
    f = jnp.repeat(jnp.arange(1, n_total + 1) / tspan, 2)
    df = jnp.ones_like(f) / tspan
    return f, df


# ---------------------------------------------------------------------------
# Signature tests
# ---------------------------------------------------------------------------

class TestMakeCombinedCrnSignature:

    def test_same_function_default_prefix(self):
        """Overlapping params get crn_ prefix when same function is passed twice."""
        combined, crn_params = make_combined_crn(14, ds.powerlaw, ds.powerlaw)
        args = inspect.getfullargspec(combined).args
        assert args == ['f', 'df', 'log10_A', 'gamma', 'crn_log10_A', 'crn_gamma'], \
            f"Got args: {args}"
        assert crn_params == ['crn_log10_A', 'crn_gamma'], \
            f"Got crn_params: {crn_params}"

    def test_same_function_no_prefix_ties_params(self):
        """crn_prefix=None with same function: params are tied, no duplication."""
        combined, crn_params = make_combined_crn(14, ds.powerlaw, ds.powerlaw, crn_prefix=None)
        args = inspect.getfullargspec(combined).args
        assert args == ['f', 'df', 'log10_A', 'gamma'], f"Got args: {args}"
        assert crn_params == ['log10_A', 'gamma'], f"Got crn_params: {crn_params}"

    def test_no_overlap_no_rename(self):
        """Non-overlapping param names require no renaming."""
        combined, crn_params = make_combined_crn(14, ds.powerlaw, _alt_psd)
        args = inspect.getfullargspec(combined).args
        assert args == ['f', 'df', 'log10_A', 'gamma', 'alpha', 'log10_ref'], \
            f"Got args: {args}"
        assert crn_params == ['alpha', 'log10_ref'], f"Got crn_params: {crn_params}"

    def test_custom_prefix(self):
        """Custom prefix is applied to overlapping CRN param names."""
        combined, crn_params = make_combined_crn(14, ds.powerlaw, ds.powerlaw, crn_prefix='gw_')
        args = inspect.getfullargspec(combined).args
        assert args == ['f', 'df', 'log10_A', 'gamma', 'gw_log10_A', 'gw_gamma'], \
            f"Got args: {args}"
        assert crn_params == ['gw_log10_A', 'gw_gamma'], f"Got crn_params: {crn_params}"


# ---------------------------------------------------------------------------
# Numerical correctness tests
# ---------------------------------------------------------------------------

class TestMakeCombinedCrnValues:

    def test_same_function_separate_params(self):
        """phi = irn(A1,g1) + crn(A2,g2) on CRN bins; irn(A1,g1) elsewhere."""
        n_crn = 14
        combined, _ = make_combined_crn(n_crn, ds.powerlaw, ds.powerlaw)
        f, df = _make_freqs()

        log10_A, gamma = -14.5, 4.3
        crn_log10_A, crn_gamma = -15.0, 13 / 3

        phi = combined(f, df, log10_A, gamma, crn_log10_A, crn_gamma)
        irn = ds.powerlaw(f, df, log10_A, gamma)
        crn = ds.powerlaw(f[:2 * n_crn], df[:2 * n_crn], crn_log10_A, crn_gamma)

        np.testing.assert_allclose(phi[:2 * n_crn], irn[:2 * n_crn] + crn, rtol=1e-6)
        np.testing.assert_allclose(phi[2 * n_crn:], irn[2 * n_crn:], rtol=1e-6)

    def test_same_function_tied_params(self):
        """crn_prefix=None + same function: CRN bins = 2 * irn; rest unchanged."""
        n_crn = 14
        combined, _ = make_combined_crn(n_crn, ds.powerlaw, ds.powerlaw, crn_prefix=None)
        f, df = _make_freqs()

        log10_A, gamma = -14.5, 4.3
        phi = combined(f, df, log10_A, gamma)
        irn = ds.powerlaw(f, df, log10_A, gamma)

        # Both PSDs receive identical params -> CRN contribution doubles the IRN value
        np.testing.assert_allclose(phi[:2 * n_crn], 2.0 * irn[:2 * n_crn], rtol=1e-6)
        np.testing.assert_allclose(phi[2 * n_crn:], irn[2 * n_crn:], rtol=1e-6)

    def test_no_overlap_values(self):
        """Non-overlapping PSDs: CRN bins = irn + alt_psd; rest = irn only."""
        n_crn = 14
        combined, _ = make_combined_crn(n_crn, ds.powerlaw, _alt_psd)
        f, df = _make_freqs()

        log10_A, gamma = -14.5, 4.3
        alpha, log10_ref = 3.0, -14.0

        phi = combined(f, df, log10_A, gamma, alpha, log10_ref)
        irn = ds.powerlaw(f, df, log10_A, gamma)
        crn = _alt_psd(f[:2 * n_crn], df[:2 * n_crn], alpha, log10_ref)

        np.testing.assert_allclose(phi[:2 * n_crn], irn[:2 * n_crn] + crn, rtol=1e-6)
        np.testing.assert_allclose(phi[2 * n_crn:], irn[2 * n_crn:], rtol=1e-6)

    def test_n_crn_boundary(self):
        """CRN only affects exactly the first 2*n_crn bins."""
        n_crn = 5
        combined, _ = make_combined_crn(n_crn, ds.powerlaw, ds.powerlaw)
        f, df = _make_freqs()

        log10_A, gamma = -14.5, 4.3
        crn_log10_A, crn_gamma = -15.0, 13 / 3

        phi = combined(f, df, log10_A, gamma, crn_log10_A, crn_gamma)
        irn = ds.powerlaw(f, df, log10_A, gamma)

        # Bins beyond n_crn are untouched
        np.testing.assert_allclose(phi[2 * n_crn:], irn[2 * n_crn:], rtol=1e-6)
        # Bins within n_crn are strictly larger than IRN alone
        assert np.all(phi[:2 * n_crn] > irn[:2 * n_crn])


# --- low-frequency turnover --------------------------------------------------

from discovery import const, signals as s


def _band(nyears=6.0, nc=156):
    T = nyears * 365.25 * 86400.0
    f = np.arange(1, nc + 1) / T
    return T, f, np.full_like(f, 1.0 / T)


def test_turnover_is_flat_below_the_corner_and_a_power_law_above():
    f = np.logspace(-11, -6, 600)
    df = np.gradient(f)
    P = np.asarray(s.turnover(f, df, -14.0, 4.33, -8.0)) / df

    lo = np.polyfit(np.log(f[:60]), np.log(P[:60]), 1)[0]
    hi = np.polyfit(np.log(f[-60:]), np.log(P[-60:]), 1)[0]

    assert abs(lo) < 0.01
    assert abs(hi + 4.33) < 0.01


def test_turnover_reduces_to_a_power_law_below_the_band():
    """The model-averaging property: a corner far below 1/T leaves no imprint."""
    _, f, df = _band()
    pl = np.asarray(s.powerlaw(f, df, -14.0, 4.33))

    for log10_fc, tol in ((-12.0, 1e-4), (-14.0, 1e-8)):
        q = np.asarray(s.turnover(f, df, -14.0, 4.33, log10_fc))
        assert np.max(np.abs(q / pl - 1)) < tol


def test_make_turnover_default_is_the_corner_form():
    _, f, df = _band()
    for gamma in (2.0, 4.33, 6.0):
        assert np.array_equal(np.asarray(s.make_turnover()(f, df, -14.0, gamma, -8.0)),
                              np.asarray(s.turnover(f, df, -14.0, gamma, -8.0)))


def test_make_turnover_samples_only_what_is_set_to_none():
    named = lambda fn: list(inspect.signature(fn).parameters)[2:]

    assert named(s.make_turnover()) == ['log10_A', 'gamma', 'log10_fc']
    assert named(s.make_turnover(kappa=None)) == ['log10_A', 'gamma', 'log10_fc', 'kappa']
    assert named(s.make_turnover(beta=None)) == ['log10_A', 'gamma', 'log10_fc', 'beta']
    assert named(s.make_turnover(kappa=None, beta=None)) == [
        'log10_A', 'gamma', 'log10_fc', 'kappa', 'beta']


def test_make_turnover_matches_the_enterprise_expression():
    _, f, df = _band()
    A, g, lfc = -14.0, 4.33, -8.0
    model = s.make_turnover(kappa=None, beta=None)

    for kappa, beta in ((2.0, 0.5), (4.33, 0.5), (1.0, 2.0)):
        hcf = 10**A * (f / const.fyr)**((3 - g) / 2) / (1 + (10**lfc / f)**kappa)**beta
        want = hcf**2 / 12 / np.pi**2 / f**3 * df
        got = np.asarray(model(f, df, A, g, lfc, kappa, beta))
        assert np.max(np.abs(got / want - 1)) < 1e-12


def test_make_turnover_flat_beta_keeps_the_low_frequency_slope_at_zero():
    f = np.logspace(-12, -6, 800)
    df = np.gradient(f)
    for kappa in (2.0, 4.0, 8.0):
        P = np.asarray(s.make_turnover(kappa=kappa)(f, df, -14.0, 4.33, -8.0)) / df
        slope = np.polyfit(np.log(f[:60]), np.log(P[:60]), 1)[0]
        assert abs(slope) < 0.01


def test_make_turnover_rejects_a_bad_beta():
    with pytest.raises(ValueError, match='flat'):
        s.make_turnover(beta='sharp')


def test_the_corner_prior_is_one_box_for_every_pulsar():
    """A hierarchical prior needs a single support, so the box must not track Tspan."""
    import discovery.models.mpta as mpta
    from discovery import prior

    mpta.update_priordict_standard_mpta()
    box = prior.getsupport('JXXXX+0000_red_noise_log10_fc')

    for par in ('J0437-4715_red_noise_log10_fc', 'J1909-3744_dm_gp_log10_fc',
                'J0030+0451_chrom_gp_log10_fc', 'B1937+21_red_noise2_log10_fc'):
        assert tuple(prior.getsupport(par)) == tuple(box)

    # and it is the range the derivation gives for the array span
    assert [float(v) for v in box] == [-11.5, -6.4]


def test_turnover_set_normalises_and_rejects_unknown_components():
    assert s.turnover_set(None) == frozenset()
    assert s.turnover_set('red') == frozenset({'red'})
    assert s.turnover_set(('red', 'dm')) == frozenset({'red', 'dm'})

    with pytest.raises(ValueError, match='unknown component'):
        s.turnover_set('spin')


def test_turnover_psd_picks_the_right_spectrum():
    assert s.turnover_psd('red', frozenset()) is s.powerlaw
    assert s.turnover_psd('red', frozenset({'red'})) is s.turnover
    assert s.turnover_psd('dm', frozenset({'red'})) is s.powerlaw
