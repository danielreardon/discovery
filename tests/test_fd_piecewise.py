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


def test_quantile_spacing_has_data_in_every_interval(psr):
    """Quantile placement must leave no empty interval between nodes."""
    gp = s.makegp_fd_piecewise(psr, nodes=16, spacing='quantile')
    counts, _ = np.histogram(np.asarray(psr.freqs), bins=gp.fd_nodes)
    assert np.all(counts > 0)


def test_log_spacing_is_uniform_in_log_freq(psr):
    """Log spacing puts nodes evenly in log(freq) across the observed band."""
    gp = s.makegp_fd_piecewise(psr, nodes=16, spacing='log')
    lq = np.log(gp.fd_nodes)
    assert np.allclose(np.diff(lq), np.diff(lq)[0], rtol=1e-6)
    assert np.isclose(gp.fd_nodes[0], np.min(psr.freqs), rtol=1e-9)
    assert np.isclose(gp.fd_nodes[-1], np.max(psr.freqs), rtol=1e-9)


def test_log_spacing_survives_band_gaps(psr):
    """Nodes landing in a receiver gap give empty columns, which are dropped."""
    gp = s.makegp_fd_piecewise(psr, nodes=16, spacing='log')
    F = np.asarray(gp.F)
    assert F.shape[1] <= 16                      # gap nodes dropped, never singular
    assert np.allclose(F.T @ F, np.eye(F.shape[1]), atol=1e-8)


def test_selection_gives_one_basis_per_group(psr):
    """A selection builds a per-group basis, combined into one marginalised GP."""
    plain = s.makegp_fd_piecewise(psr, nodes=8)
    grouped = s.makegp_fd_piecewise(psr, nodes=8, selection=s.selection_backend_flags)

    ngroups = len(set(np.asarray(s.selection_backend_flags(psr)).tolist()))
    assert isinstance(grouped.fd_nodes, dict) and len(grouped.fd_nodes) == ngroups
    assert not isinstance(plain.fd_nodes, dict)

    F = np.asarray(grouped.F)
    assert F.shape[1] > np.asarray(plain.F).shape[1]     # more DOF than a single basis
    assert np.allclose(F.T @ F, np.eye(F.shape[1]), atol=1e-9)
    # projecting the timing model out of the STACK (not per group) keeps the
    # result orthogonal to it; doing it per group would not
    assert np.abs(_qtm(psr).T @ F).max() < 1e-9


def test_selection_blocks_have_disjoint_support(psr):
    """Per-group hat blocks are zero outside their own TOAs, hence orthogonal."""
    x = np.log(np.asarray(psr.freqs, dtype=np.float64))
    flags = np.asarray(s.selection_backend_flags(psr))
    blocks = [s._fd_piecewise_block(psr, x, flags == g, 8, 'quantile', str(g))[0]
              for g in sorted(set(flags.tolist()))]
    for i in range(len(blocks)):
        support = np.abs(blocks[i]).sum(axis=1) > 0
        assert support.sum() == int((flags == sorted(set(flags.tolist()))[i]).sum())
        for j in range(i + 1, len(blocks)):
            assert np.abs(blocks[i].T @ blocks[j]).max() < 1e-12


def test_unknown_spacing_raises(psr):
    with pytest.raises(ValueError, match="spacing"):
        s.makegp_fd_piecewise(psr, nodes=8, spacing='linear')


def test_basis_orthonormal_and_orthogonal_to_timing_model(psr):
    gp = s.makegp_fd_piecewise(psr, nodes=16, project_tm=True)
    F = np.asarray(gp.F)
    assert F.shape[0] == len(psr.toas)
    assert F.shape[1] <= 16                      # rank-deficient directions dropped
    assert np.allclose(F.T @ F, np.eye(F.shape[1]), atol=1e-8)
    assert np.abs(_qtm(psr).T @ F).max() < 1e-8


def test_projection_drops_degenerate_directions(psr):
    """With the TM projected out, the surviving basis is smaller than without it."""
    kept = np.asarray(s.makegp_fd_piecewise(psr, nodes=16, spacing='quantile', project_tm=True).F).shape[1]
    raw = np.asarray(s.makegp_fd_piecewise(psr, nodes=16, spacing='quantile', project_tm=False).F).shape[1]
    assert kept < raw


def test_removes_time_constant_frequency_structure(psr):
    """An injected frequency-only signal is largely absorbed by the basis."""
    gp = s.makegp_fd_piecewise(psr, nodes=24, spacing='quantile')
    F, Qtm = np.asarray(gp.F), _qtm(psr)
    freqs = np.asarray(psr.freqs, dtype=np.float64)
    inject = 1e-6 * np.sin(3 * np.log(freqs / 1000.0)) + 4e-7 * (1400.0 / freqs) ** 4
    resid = inject - Qtm @ (Qtm.T @ inject)      # what the timing model leaves
    after = resid - F @ (F.T @ resid)
    assert after.std() < 0.2 * resid.std()


@pytest.mark.parametrize("spacing", ["log", "quantile"])
def test_both_spacings_build_a_usable_basis(psr, spacing):
    gp = s.makegp_fd_piecewise(psr, nodes=12, spacing=spacing)
    F = np.asarray(gp.F)
    assert F.shape[0] == len(psr.toas) and F.shape[1] >= 2
    assert np.all(np.isfinite(F))


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
@pytest.mark.parametrize("nodes", [16, 32])
def test_mpta_wiring_auto_projects_chrom_poly(psr, nodes):
    """mpta.single_pulsar_noise(fd=True) adds the term and passes project= on."""
    import discovery.models.mpta as mpta

    kw = dict(fftint=False, noisedict=psr.noisedict, background=False,
              red=True, dm=True, chrom=True, sw=False)
    comps = mpta.single_pulsar_noise(psr, fd=True, fd_nodes=nodes, chrom_poly=True,
                                     return_components=True, **kw)[1]

    fd = [np.asarray(c.F) for c in comps if getattr(c, "gpname", None) == "fd"]
    assert len(fd) == 1 and fd[0].shape[1] <= nodes

    # the chromatic-polynomial GP is the one carrying .svd metadata
    poly = [c for c in comps if getattr(c, "svd", None) is not None]
    assert len(poly) == 1
    B = np.asarray(poly[0].F({f"{psr.name}_chrom_gp_alpha": 4.0}))
    assert np.abs(fd[0].T @ B).max() < 1e-8       # project= was auto-passed

    # and fd=False adds no such component
    off = mpta.single_pulsar_noise(psr, fd=False, return_components=True, **kw)[1]
    assert not any(getattr(c, "gpname", None) == "fd" for c in off)


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
