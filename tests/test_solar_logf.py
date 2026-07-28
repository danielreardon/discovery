"""Tests for the log-spaced frequency option of the solar-wind DM Fourier basis."""

from pathlib import Path

import numpy as np
import pytest

import discovery as ds
from discovery import solar


DATA = Path(__file__).resolve().parent.parent / "data"
NCOMP = 20


@pytest.fixture(scope="module")
def psr():
    f = DATA / "v1p1_de440_pint_bipm2019-J0030+0451.feather"
    if not f.exists():
        pytest.skip("pulsar data fixture missing")
    return ds.Pulsar.read_feather(f)


def test_default_is_linear_and_unchanged(psr):
    """logf=False (the default) must reproduce the existing linear basis exactly."""
    a = solar.fourierbasis_solar_dm(psr, NCOMP)
    b = solar.fourierbasis_solar_dm(psr, NCOMP, logf=False)
    for x, y in zip(a, b):
        assert np.array_equal(np.asarray(x), np.asarray(y))
    # linear grid is f = n/T
    T = ds.getspan(psr)
    assert np.allclose(np.asarray(a[0])[::2] * T, np.arange(1, NCOMP + 1))


def test_logf_grid_is_log_spaced_over_the_same_band(psr):
    lin = solar.fourierbasis_solar_dm(psr, NCOMP)
    log = solar.fourierbasis_solar_dm(psr, NCOMP, logf=True)

    f_lin, f_log = np.asarray(lin[0])[::2], np.asarray(log[0])[::2]
    assert len(f_log) == NCOMP
    # same band endpoints
    assert np.isclose(f_log[0], f_lin[0])
    assert np.isclose(f_log[-1], f_lin[-1])
    # log-spaced => constant ratio between neighbours, and denser at low f
    ratios = f_log[1:] / f_log[:-1]
    assert np.allclose(ratios, ratios[0])
    assert f_log[1] - f_log[0] < f_lin[1] - f_lin[0]


def test_logf_shapes_and_df_tile_the_band(psr):
    f, df, fmat = solar.fourierbasis_solar_dm(psr, NCOMP, logf=True)
    assert fmat.shape == (len(psr.toas), 2 * NCOMP)
    assert len(f) == len(df) == 2 * NCOMP        # repeated per sin/cos column
    assert np.all(np.isfinite(fmat))
    # bins tile [0, f_max] contiguously
    assert np.isclose(np.sum(np.asarray(df)[::2]), np.asarray(f)[-1])


def test_factory_selects_the_grid(psr):
    """make_fourierbasis_solar_dm bakes logf in for makegp_fourier's (psr, components, T) call."""
    basis = solar.make_fourierbasis_solar_dm(logf=True)
    direct = solar.fourierbasis_solar_dm(psr, NCOMP, logf=True)
    for x, y in zip(basis(psr, NCOMP), direct):
        assert np.array_equal(np.asarray(x), np.asarray(y))


@pytest.mark.integration
def test_logf_gp_builds_and_logL_finite(psr):
    gp = ds.makegp_fourier(psr, ds.powerlaw, components=NCOMP,
                           fourierbasis=solar.make_fourierbasis_solar_dm(logf=True),
                           name="sw_gp")
    model = ds.PulsarLikelihood([psr.residuals,
                                 ds.makenoise_measurement(psr, psr.noisedict),
                                 ds.makegp_timing(psr, svd=True), gp])
    assert any("sw_gp_log10_A" in q for q in model.logL.params)
    p0 = ds.sample_uniform(model.logL.params)
    assert np.isfinite(float(model.logL(p0)))
