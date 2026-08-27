"""Tests for the parallactic-angle-locked delay GP and the angle it is built on."""

import os
from pathlib import Path

import numpy as np
import pytest

import discovery as ds
from discovery import signals as s
from discovery import likelihood as dl


DATA = Path(__file__).resolve().parent.parent / "data"

# the bundled fixture carries no telescope column, so the site is named explicitly,
# which also exercises the site= path. Any site in OBSERVATORY_ITRF will do: these test
# the geometry, not this pulsar's observing history.
SITE = 'meerkat'


@pytest.fixture(scope="module")
def psr():
    f = DATA / "v1p1_de440_pint_bipm2019-J0030+0451.feather"
    if not f.exists():
        pytest.skip("pulsar data fixture missing")
    return ds.Pulsar.read_feather(f)


# --- the angle -------------------------------------------------------------------

def test_gmst_matches_the_defining_value_at_j2000():
    """GMST at J2000.0 is 18h 41m 50.5482s, i.e. 280.46061837 degrees."""
    got = np.rad2deg(s.greenwich_sidereal_angle(51544.5))
    assert abs(got - 280.46061837) < 1e-6


def test_gmst_advances_by_one_sidereal_day():
    """A mean sidereal day is 86164.0905 s, not 86400."""
    a = s.greenwich_sidereal_angle(58000.0)
    b = s.greenwich_sidereal_angle(58000.0 + 86164.0905 / 86400.0)
    assert abs((np.rad2deg(b - a) + 180.0) % 360.0 - 180.0) < 1e-4


def test_parallactic_angle_vanishes_at_transit(psr):
    """At zero hour angle the source, the zenith and the pole are on one great circle.

    The angle is then 0 or pi, according to whether the source transits south or north
    of the zenith, so it is sin(psi) that must vanish and not psi itself.
    """
    lat, lon = s.observatory_location(SITE)
    # solve for the MJDs at which H = 0 by stepping the sidereal rate
    mjd = 58000.0
    for _ in range(60):
        H = s.greenwich_sidereal_angle(mjd) + lon - psr.phi
        mjd -= float(np.arctan2(np.sin(H), np.cos(H))) / (2 * np.pi) * (86164.0905 / 86400.0)

    class AtTransit:
        name, phi, theta = psr.name, psr.phi, psr.theta
        stoas = np.array([mjd * 86400.0])

    assert abs(float(np.sin(s.parallactic_angle(AtTransit, site=SITE)[0]))) < 1e-5


def test_parallactic_angle_uses_site_arrival_times_not_barycentric(psr):
    """psr.toas carry the Roemer delay, which is degrees of hour angle."""
    psi = s.parallactic_angle(psr, site=SITE)

    class Barycentred:
        name, phi, theta = psr.name, psr.phi, psr.theta
        stoas = psr.toas

    other = s.parallactic_angle(Barycentred, site=SITE)
    shift = np.max(np.abs((np.rad2deg(psi - other) + 180.0) % 360.0 - 180.0))
    assert shift > 0.1, "swapping stoas for toas must change the angle appreciably"


def test_parallactic_angle_agrees_with_astropy(psr):
    """Against an independent apparent-sidereal-time implementation."""
    astropy = pytest.importorskip("astropy")
    from astropy.time import Time
    from astropy.coordinates import EarthLocation
    import astropy.units as u

    lat, lon = s.observatory_location(SITE)
    loc = EarthLocation(lat=np.rad2deg(lat) * u.deg, lon=np.rad2deg(lon) * u.deg)
    mjd = np.asarray(psr.stoas) / 86400.0
    t = Time(mjd, format="mjd", scale="utc", location=loc)
    H = (t.sidereal_time("apparent").radian + 0.0) - psr.phi
    dec = 0.5 * np.pi - psr.theta
    ref = np.arctan2(np.sin(H) * np.cos(lat),
                     np.sin(lat) * np.cos(dec) - np.cos(lat) * np.sin(dec) * np.cos(H))

    d = (np.rad2deg(s.parallactic_angle(psr, site=SITE) - ref) + 180.0) % 360.0 - 180.0
    assert np.max(np.abs(d)) < 0.05


def test_observatory_lookup_by_code_and_by_explicit_coordinates():
    assert np.allclose(np.rad2deg(s.observatory_location('pks')),
                       [-32.998406, 148.263510], atol=1e-5)
    assert np.allclose(np.rad2deg(s.observatory_location('meerkat')),
                       [-30.711056, 21.443889], atol=1e-5)
    assert np.allclose(np.rad2deg(s.observatory_location((-30.0, 21.0))), [-30.0, 21.0])


@pytest.mark.skipif(not os.path.exists(os.path.join(os.environ.get('TEMPO2', ''),
                                                    'observatory/observatories.dat')),
                    reason="needs $TEMPO2 for sites outside OBSERVATORY_ITRF")
def test_sites_outside_the_table_come_from_tempo2_by_code_or_name():
    assert np.allclose(s.observatory_location('PARKES'), s.observatory_location('pks'))
    assert np.allclose(np.rad2deg(s.observatory_location('gbt')),
                       [38.433130, -79.839843], atol=1e-5)


def test_unknown_site_raises_rather_than_guessing():
    with pytest.raises(KeyError, match="no coordinates"):
        s.observatory_location('not_a_telescope')


def test_no_site_and_no_telescope_column_raises(psr):
    with pytest.raises(ValueError, match="nowhere to stand"):
        s.parallactic_angle(psr)


def test_telescope_column_is_read_back_from_a_feather(psr, tmp_path):
    """Pulsar.optional_columns must survive a round trip, and stay optional."""
    out = tmp_path / "with.feather"
    psr.telescope = np.array([SITE] * len(psr.toas))
    psr.save_feather(str(out))
    assert np.all(ds.Pulsar.read_feather(str(out)).telescope == SITE)

    del psr.telescope
    plain = tmp_path / "without.feather"
    psr.save_feather(str(plain))          # must not raise on a Pulsar lacking it
    assert not hasattr(ds.Pulsar.read_feather(str(plain)), 'telescope')


# --- the basis -------------------------------------------------------------------

def test_column_count_and_disjoint_bin_support(psr):
    gp = s.makegp_pa_quadrature(psr, bin_flag='chan', site=SITE,
                                project_tm=False)
    F = np.asarray(gp.F)
    nbin = len(gp.pa_bins)
    assert F.shape[1] == 2 * nbin

    lab = np.asarray(psr.flags['chan']).astype(str)
    order = sorted(set(lab.tolist()), key=int)
    for j, value in enumerate(order):
        outside = lab != value
        assert np.all(F[outside, j] == 0.0)
        assert np.all(F[outside, j + nbin] == 0.0)


def test_one_scale_governs_the_whole_basis(psr):
    gp = s.makegp_pa_quadrature(psr, bin_flag='chan', site=SITE)
    assert gp.Phi.params == [f'{psr.name}_pa_gp_log10_sigma']
    assert gp.pa_harmonic == 2


def test_a_nonpositive_harmonic_raises(psr):
    with pytest.raises(ValueError, match="positive integer"):
        s.makegp_pa_quadrature(psr, harmonic=0, bin_flag='chan', site=SITE)


def test_a_missing_bin_flag_raises_rather_than_guessing_the_bins(psr):
    """The bins are the receiver channelisation; inferring them would be a guess."""
    with pytest.raises(ValueError, match="needs bin_flag"):
        s.makegp_pa_quadrature(psr, bin_flag=None, site=SITE)
    with pytest.raises(KeyError, match="has no flag"):
        s.makegp_pa_quadrature(psr, bin_flag='not_a_flag', site=SITE)


def test_the_implied_delay_covariance_is_coherent_in_pa_within_a_bin(psr):
    """sigma**2 F F^T must be sigma**2 cos(m (psi_i - psi_j)) inside a bin, zero across.

    That is the whole model: within a channel the delay is a sinusoid in the
    parallactic angle whose amplitude AND phase are free, and channels are independent.
    """
    gp = s.makegp_pa_quadrature(psr, bin_flag='chan', site=SITE,
                                project_tm=False)
    F = np.asarray(gp.F)
    psi = s.parallactic_angle(psr, site=SITE)
    lab = np.asarray(psr.flags['chan']).astype(str)

    idx = np.arange(len(psi))[:400]
    K = F[idx] @ F[idx].T
    same = lab[idx][:, None] == lab[idx][None, :]
    expect = np.where(same, np.cos(2 * (psi[idx][:, None] - psi[idx][None, :])), 0.0)
    assert np.allclose(K, expect, atol=1e-12)


def test_sigma_is_the_per_quadrature_delay(psr):
    """Every TOA gets variance sigma**2, so a channel amplitude has E[A**2] = 2 sigma**2."""
    gp = s.makegp_pa_quadrature(psr, bin_flag='chan', site=SITE,
                                project_tm=False)
    F = np.asarray(gp.F)
    assert np.allclose(np.sum(F**2, axis=1), 1.0, atol=1e-12)


def test_prior_box_is_registered_for_every_scale(psr):
    from discovery import prior
    s.makegp_pa_quadrature(psr, bin_flag='chan', site=SITE)
    import re
    key = f'{re.escape(psr.name)}_pa_gp_log10_sigma'
    assert prior.priordict_standard[key] == [-10.0, -6.0]


def test_projection_keeps_the_columns_rather_than_dropping_them(psr):
    """The column count must not depend on what the projection annihilates.

    An annihilated direction integrates back to its prior; dropping it would change
    the prior instead.
    """
    raw = s.makegp_pa_quadrature(psr, bin_flag='chan', site=SITE,
                                 project_tm=False)
    proj = s.makegp_pa_quadrature(psr, bin_flag='chan', site=SITE,
                                  project_tm=True)
    assert np.asarray(proj.F).shape == np.asarray(raw.F).shape


def test_projecting_an_improperly_marginalised_basis_is_a_no_op(psr):
    """Removing the fd span changes nothing when that span carries an improper prior."""
    fd = s.makegp_fd_piecewise(psr, spacing='flag', kind='constant', bin_flag='chan',
                               name='fd')

    def logl(project):
        pa = s.makegp_pa_quadrature(psr, bin_flag='chan', site=SITE,
                                    project=fd if project else None)
        m = dl.PulsarLikelihood([psr.residuals, s.makenoise_measurement(psr, {}),
                                 s.makegp_timing(psr, svd=True), fd, pa])
        p = {k: (1.0 if k.endswith('efac') else -7.5 if 'pa_gp' in k else -8.0)
             for k in m.logL.params}
        return float(m.logL(p))

    assert abs(logl(True) - logl(False)) < 1e-6


def test_injected_pa_delay_is_recovered_at_the_right_scale(psr):
    """Inject a known per-channel sinusoid in PA and scan the marginal likelihood."""
    rng = np.random.default_rng(20260827)
    psi = s.parallactic_angle(psr, site=SITE)
    lab = np.asarray(psr.flags['chan']).astype(str)
    order = sorted(set(lab.tolist()), key=int)

    # Scaled to the fixture rather than fixed, so the test cannot quietly become
    # untestable: two per-channel amplitudes are measured from ~n_chan TOAs each, so the
    # amplitude error is err / sqrt(n_chan / 2) and this injects twice that.
    err = np.asarray(psr.toaerrs)
    per_bin = np.median([int((lab == v).sum()) for v in order])
    sigma_true = float(2.0 * np.median(err) / np.sqrt(0.5 * per_bin))

    delay = np.zeros(len(psi))
    for value in order:
        m = lab == value
        a, b = rng.normal(scale=sigma_true, size=2)
        delay[m] = a * np.sin(2 * psi[m]) + b * np.cos(2 * psi[m])

    res = delay + rng.normal(scale=err)

    pa = s.makegp_pa_quadrature(psr, bin_flag='chan', site=SITE)
    m = dl.PulsarLikelihood([res, s.makenoise_measurement(psr, {}),
                             s.makegp_timing(psr, svd=True), pa])
    grid = np.linspace(-10.0, -6.0, 41)
    key = f'{psr.name}_pa_gp_log10_sigma'
    base = {k: (1.0 if k.endswith('efac') else -8.5) for k in m.logL.params}
    curve = np.array([float(m.logL({**base, key: g})) for g in grid])

    best = grid[int(np.argmax(curve))]
    assert abs(best - np.log10(sigma_true)) < 0.3, (best, np.log10(sigma_true))
    assert curve.max() - curve[0] > 20.0, "an injected term must beat the no-term end"


# --- threading through common_noise -----------------------------------------------

def _fake_chain(psr, pa=False):
    """Minimal stage-1 chain: red noise, per-backend efac, and optionally the PA scale."""
    import pandas as pd

    cols = [f'{psr.name}_red_noise_log10_A', f'{psr.name}_red_noise_gamma']
    cols += [f'{psr.name}_{be}_efac'
             for be in sorted(set(np.asarray(psr.backend_flags).tolist()))]
    if pa:
        cols += [f'{psr.name}_pa_gp_log10_sigma']

    df = pd.DataFrame({c: np.linspace(-8.0, -7.0, 8) for c in cols})
    df.attrs['noisedict'] = {}
    return df


@pytest.fixture(scope="module")
def two_psrs():
    """Two pulsars carrying a telescope column, which the mpta path requires.

    common_noise does not take a site: it relies on the column, which every MPTA and
    PPTA feather has and these bundled fixtures do not.
    """
    files = [DATA / "v1p1_de440_pint_bipm2019-J0030+0451.feather",
             DATA / "v1p1_de440_pint_bipm2019-B1855+09.feather"]
    if not all(f.exists() for f in files):
        pytest.skip("pulsar data fixtures missing")

    psrs = [ds.Pulsar.read_feather(f) for f in files]
    for psr in psrs:
        psr.telescope = np.array([SITE] * len(psr.toas))
    return psrs


def test_common_noise_switches_the_pa_gp_on_per_pulsar_from_the_chains(two_psrs):
    """Presence comes from the chain, as every other component does."""
    from discovery.models import mpta

    a, b = two_psrs
    m = mpta.common_noise(two_psrs, [_fake_chain(a, pa=True), _fake_chain(b)],
                          fd=False, pa_bin_flag='chan', noise_point='median')

    assert sorted(p for p in m.logL.params if 'pa_gp' in p) == \
        [f'{a.name}_pa_gp_log10_sigma']


def test_common_noise_leaves_the_pa_gp_out_where_no_chain_carries_it(two_psrs):
    from discovery.models import mpta

    a, b = two_psrs
    m = mpta.common_noise(two_psrs, [_fake_chain(a), _fake_chain(b)],
                          fd=False, pa_bin_flag='chan', noise_point='median')
    assert not [p for p in m.logL.params if 'pa_gp' in p]


def test_common_noise_warns_about_the_settings_the_chain_cannot_carry(two_psrs, capsys):
    """The bins and the projection change the basis, not the parameters."""
    from discovery.models import mpta

    a, b = two_psrs
    mpta.common_noise(two_psrs, [_fake_chain(a, pa=True), _fake_chain(b)],
                      fd=False, pa_bin_flag='chan', noise_point='median')

    out = capsys.readouterr().out
    assert 'a disagreement cannot be reported' in out
    assert "bin_flag='chan'" in out
    assert 'pa_project_fd=True' in out
    assert f'1 of {len(two_psrs)} pulsar(s) carry' in out


def test_commongp_falls_back_rather_than_stacking_a_variable_core(two_psrs, capsys):
    """The PA GP samples its scale, so it leaves the per-pulsar core variable."""
    from discovery.models import mpta

    a, b = two_psrs
    mpta.common_noise(two_psrs, [_fake_chain(a, pa=True), _fake_chain(b)],
                      fd=False, pa_bin_flag='chan',
                      use_commongp=True, fix_chrom_alpha=True, noise_point='median')

    out = capsys.readouterr().out
    assert 'parallactic-angle GP, which is not stackable' in out
    assert 'Falling back to the GlobalLikelihood path' in out
