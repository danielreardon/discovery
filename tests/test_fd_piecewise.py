"""Tests for the marginalised piecewise-linear frequency-dependent delay."""

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


def _qtm(psr):
    M = np.asarray(psr.Mmat, dtype=np.float64)
    return np.linalg.qr(M / np.sqrt(np.sum(M**2, axis=0)))[0]


def test_nodes_have_data_in_every_interval(psr):
    """Quantile placement must leave no empty interval between nodes."""
    gp = s.makegp_fd_piecewise(psr, nodes=16)
    counts, _ = np.histogram(np.asarray(psr.freqs), bins=gp.fd_nodes)
    assert np.all(counts > 0)
    # end nodes are the min/max observed frequency, up to the log/exp round-trip
    assert np.isclose(gp.fd_nodes[0], np.min(psr.freqs), rtol=1e-9)
    assert np.isclose(gp.fd_nodes[-1], np.max(psr.freqs), rtol=1e-9)


def test_basis_orthonormal_and_orthogonal_to_timing_model(psr):
    gp = s.makegp_fd_piecewise(psr, nodes=16, project_tm=True)
    F = np.asarray(gp.F)
    assert F.shape[0] == len(psr.toas)
    assert F.shape[1] <= 16                      # rank-deficient directions dropped
    assert np.allclose(F.T @ F, np.eye(F.shape[1]), atol=1e-8)
    assert np.abs(_qtm(psr).T @ F).max() < 1e-8


def test_projection_drops_degenerate_directions(psr):
    """With the TM projected out, the surviving basis is smaller than the node count."""
    kept = np.asarray(s.makegp_fd_piecewise(psr, nodes=16, project_tm=True).F).shape[1]
    raw = np.asarray(s.makegp_fd_piecewise(psr, nodes=16, project_tm=False).F).shape[1]
    assert kept < raw


def test_removes_time_constant_frequency_structure(psr):
    """An injected frequency-only signal is largely absorbed by the basis."""
    gp = s.makegp_fd_piecewise(psr, nodes=24)
    F, Qtm = np.asarray(gp.F), _qtm(psr)
    freqs = np.asarray(psr.freqs, dtype=np.float64)
    inject = 1e-6 * np.sin(3 * np.log(freqs / 1000.0)) + 4e-7 * (1400.0 / freqs) ** 4
    resid = inject - Qtm @ (Qtm.T @ inject)      # what the timing model leaves
    after = resid - F @ (F.T @ resid)
    assert after.std() < 0.2 * resid.std()


def test_linear_freq_option(psr):
    gp = s.makegp_fd_piecewise(psr, nodes=12, log_freq=False)
    assert np.asarray(gp.F).shape[0] == len(psr.toas)
    counts, _ = np.histogram(np.asarray(psr.freqs), bins=gp.fd_nodes)
    assert np.all(counts > 0)


def test_chrom_poly_project_removes_overlap(psr):
    """project= makes the chromatic-polynomial basis orthogonal to the fd basis."""
    fd = s.makegp_fd_piecewise(psr, nodes=16)
    F = np.asarray(fd.F)
    p = {f"{psr.name}_chrom_gp_alpha": 4.0}

    plain = np.asarray(s.makegp_chrom_poly_svd(psr, name="chrom_gp").F(p))
    projd = np.asarray(s.makegp_chrom_poly_svd(psr, name="chrom_gp", project=fd).F(p))

    assert np.abs(F.T @ plain).max() > 1e-3      # overlap exists without project=
    assert np.abs(F.T @ projd).max() < 1e-8      # and is removed with it
    assert np.allclose(projd.T @ projd, np.eye(projd.shape[1]), atol=1e-6)


@pytest.mark.integration
def test_fd_gp_logL_finite_and_adds_no_parameters(psr):
    gp = s.makegp_fd_piecewise(psr, nodes=16)
    base = ds.PulsarLikelihood([psr.residuals,
                                ds.makenoise_measurement(psr, psr.noisedict),
                                ds.makegp_timing(psr, svd=True)])
    model = ds.PulsarLikelihood([psr.residuals,
                                 ds.makenoise_measurement(psr, psr.noisedict),
                                 ds.makegp_timing(psr, svd=True), gp])
    # marginalised analytically: no new sampled parameters
    assert set(model.logL.params) == set(base.logL.params)
    p0 = ds.sample_uniform(model.logL.params)
    assert np.isfinite(float(model.logL(p0)))
