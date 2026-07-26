"""Tests for the band-limited (frequency-band) red-noise GPs, parametrised by
band centre (fcenter) and log10 bandwidth (log10_bw)."""

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


def _band_p0(params):
    p0 = {}
    for q in params:
        if "fcenter" in q:
            p0[q] = 1400.0
        elif "log10_bw" in q:
            p0[q] = 2.5
        elif "log10_A" in q:
            p0[q] = -14.0
        elif q.endswith("gamma"):
            p0[q] = 3.0
        elif q.endswith("alpha"):
            p0[q] = 2.0
        else:
            p0[q] = 0.0
    return p0


def test_band_envelope_unit_rms_and_nonnegative(psr):
    env = np.asarray(s._band_envelope(psr, fcenter=1400.0, log10_bw=2.5))
    assert env.shape == np.asarray(psr.freqs).shape
    assert np.all(env >= 0.0)
    assert np.isclose(np.sqrt(np.mean(env ** 2)), 1.0, atol=1e-3)


def test_fourierbasis_band_shape_and_finite(psr):
    _, _, fmatfunc = s.fourierbasis_band(psr, components=10)
    F = np.asarray(fmatfunc(1400.0, 2.5))
    assert F.shape == (len(psr.toas), 20)
    assert np.all(np.isfinite(F))


def test_fourierbasis_band_alpha_shape_and_finite(psr):
    _, _, fmatfunc = s.fourierbasis_band_alpha(psr, components=10)
    F = np.asarray(fmatfunc(1400.0, 2.5, 2.0))
    assert F.shape == (len(psr.toas), 20)
    assert np.all(np.isfinite(F))


@pytest.mark.integration
def test_fourier_band_gp_logL_finite(psr):
    gp = ds.makegp_fourier(psr, ds.powerlaw, components=10,
                           fourierbasis=s.fourierbasis_band, name="band_gp")
    model = ds.PulsarLikelihood([psr.residuals,
                                 ds.makenoise_measurement(psr, psr.noisedict),
                                 ds.makegp_timing(psr, svd=True), gp])
    assert any("band_gp_fcenter" in q for q in model.logL.params)
    assert np.isfinite(float(model.logL(_band_p0(model.logL.params))))


@pytest.mark.integration
def test_fftcov_band_gp_logL_finite(psr):
    gp = s.makegp_fftcov_band(psr, ds.powerlaw, components=21, name="band_gp")
    model = ds.PulsarLikelihood([psr.residuals,
                                 ds.makenoise_measurement(psr, psr.noisedict),
                                 ds.makegp_timing(psr, svd=True), gp])
    assert any("band_gp_log10_bw" in q for q in model.logL.params)
    assert np.isfinite(float(model.logL(_band_p0(model.logL.params))))
