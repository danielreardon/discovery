"""Tests for the cross-correlation-only HD global GP.

Covers `hd_orf_crossonly` and the `cross_only=True` path of
`makeglobalgp_fourier` / `makeglobalgp_fftcov`, which build a decoupled
prior  Phi = D_auto(theta_auto)  +  C_cross(theta_cross), where the HD term
contributes only to the off-diagonal (cross-correlation) blocks and a
separate uncorrelated term carries the diagonal (auto) power.
"""

from pathlib import Path

import numpy as np
import pytest

import discovery as ds


DATA = Path(__file__).resolve().parent.parent / "data"
PSR_FILES = [
    "v1p1_de440_pint_bipm2019-B1855+09.feather",
    "v1p1_de440_pint_bipm2019-B1937+21.feather",
    "v1p1_de440_pint_bipm2019-J0030+0451.feather",
]


def test_hd_orf_crossonly_matches_hd_offdiagonal():
    """Zero on the diagonal, identical to hd_orf off the diagonal."""
    rng = np.random.default_rng(0)
    pos = rng.standard_normal((5, 3))
    pos /= np.linalg.norm(pos, axis=1, keepdims=True)

    for i in range(len(pos)):
        assert ds.hd_orf_crossonly(pos[i], pos[i]) == 0.0
        for j in range(len(pos)):
            if i != j:
                assert ds.hd_orf_crossonly(pos[i], pos[j]) == pytest.approx(ds.hd_orf(pos[i], pos[j]))


@pytest.fixture(scope="module")
def psrs():
    files = [DATA / f for f in PSR_FILES]
    if not all(f.exists() for f in files):
        pytest.skip("pulsar data fixtures not available")
    return [ds.Pulsar.read_feather(f) for f in files]


def _blocks(Phi, npsr):
    """Split the (npsr*nc, npsr*nc) prior into an npsr x npsr grid of nc x nc blocks."""
    Phi = np.asarray(Phi)
    nc = Phi.shape[0] // npsr
    return {(i, j): Phi[i * nc:(i + 1) * nc, j * nc:(j + 1) * nc]
            for i in range(npsr) for j in range(npsr)}, nc


@pytest.mark.integration
def test_crossonly_fourier_structure_and_decoupling(psrs):
    T = ds.getspan(psrs)
    npsr = len(psrs)

    gp = ds.makeglobalgp_fourier(psrs, ds.powerlaw, ds.hd_orf, components=6, T=T,
                                 cross_only=True, name="gw")

    # separate auto and cross parameters both present
    assert set(gp.Phi.params) == {"gw_log10_A", "gw_gamma", "gw_auto_log10_A", "gw_auto_gamma"}

    base = {"gw_log10_A": -14.5, "gw_gamma": 4.33,
            "gw_auto_log10_A": -14.0, "gw_auto_gamma": 4.33}
    Phi = gp.Phi.getN(base)
    blk, nc = _blocks(Phi, npsr)

    # off-diagonal blocks follow the HD cross ratios; diagonal blocks carry only auto power
    hd = np.array([[ds.hd_orf(p1.pos, p2.pos) for p2 in psrs] for p1 in psrs])
    for i in range(npsr):
        # diagonal block is diagonal (a per-frequency variance), i.e. uncorrelated auto power
        assert np.allclose(blk[(i, i)], np.diag(np.diag(blk[(i, i)])))
        for j in range(npsr):
            if i != j:
                # off-diagonal block == hd(i,j) * (a diagonal cross spectrum)
                d_ij = np.diag(blk[(i, j)])
                assert np.allclose(blk[(i, j)], np.diag(d_ij))
                # ratio of two off-diagonal blocks equals ratio of HD coefficients
                if abs(hd[i, j]) > 1e-3:
                    ref = np.diag(blk[(0, 1)])
                    scale = d_ij / ref
                    assert np.allclose(scale, hd[i, j] / hd[0, 1], rtol=1e-6, atol=1e-8)

    # DECOUPLING: changing the cross amplitude must not change the auto (diagonal) blocks
    hi_cross = dict(base, gw_log10_A=-12.0)
    Phi2 = gp.Phi.getN(hi_cross)
    blk2, _ = _blocks(Phi2, npsr)
    for i in range(npsr):
        assert np.allclose(blk[(i, i)], blk2[(i, i)]), "auto block changed with cross amplitude"
    # ...but the off-diagonal (cross) blocks DO change
    assert not np.allclose(blk[(0, 1)], blk2[(0, 1)])

    # NO HD AUTOCORRELATION: diagonal blocks equal a pure uncorrelated GP with the auto params
    auto_only = dict(base, gw_log10_A=-30.0)  # cross ~ 0
    Phi0 = gp.Phi.getN(auto_only)
    blk0, _ = _blocks(Phi0, npsr)
    for i in range(npsr):
        assert np.allclose(blk[(i, i)], blk0[(i, i)])
    # off-diagonal essentially vanishes when the cross amplitude is driven to zero
    assert np.max(np.abs(_blocks(Phi0, npsr)[0][(0, 1)])) < 1e-25


@pytest.mark.integration
def test_crossonly_positive_definite_when_auto_dominates(psrs):
    T = ds.getspan(psrs)
    gp = ds.makeglobalgp_fourier(psrs, ds.powerlaw, ds.hd_orf, components=6, T=T,
                                 cross_only=True, name="gw")

    # auto power dominates -> valid (PD) prior
    pd = {"gw_log10_A": -18.0, "gw_gamma": 4.33, "gw_auto_log10_A": -13.0, "gw_auto_gamma": 4.33}
    np.linalg.cholesky(np.asarray(gp.Phi.getN(pd)))  # must not raise

    # cross power dominates -> not PD (Cauchy-Schwarz violated)
    npd = {"gw_log10_A": -12.0, "gw_gamma": 4.33, "gw_auto_log10_A": -20.0, "gw_auto_gamma": 4.33}
    with pytest.raises(np.linalg.LinAlgError):
        np.linalg.cholesky(np.asarray(gp.Phi.getN(npd)))


@pytest.mark.integration
def test_crossonly_globallikelihood_logL_finite(psrs):
    T = ds.getspan(psrs)
    mdl = ds.GlobalLikelihood(
        [ds.PulsarLikelihood([psr.residuals,
                              ds.makenoise_measurement(psr, psr.noisedict),
                              ds.makegp_timing(psr, svd=True)]) for psr in psrs],
        globalgp=ds.makeglobalgp_fourier(psrs, ds.powerlaw, ds.hd_orf, components=6, T=T,
                                         cross_only=True, name="gw"))

    # auto-dominant point -> PD prior -> finite logL
    p0 = {"gw_log10_A": -16.0, "gw_gamma": 4.33, "gw_auto_log10_A": -13.0, "gw_auto_gamma": 4.33}
    val = float(mdl.logL(p0))
    assert np.isfinite(val)


@pytest.mark.integration
def test_crossonly_fftcov_smoke(psrs):
    T = ds.getspan(psrs)
    t0 = ds.getstart(psrs)
    npsr = len(psrs)

    gp = ds.makeglobalgp_fftcov(psrs, ds.powerlaw, ds.hd_orf, components=11, T=T, t0=t0,
                                order=1, cross_only=True, name="gw")
    assert set(gp.Phi.params) == {"gw_log10_A", "gw_gamma", "gw_auto_log10_A", "gw_auto_gamma"}

    pd = {"gw_log10_A": -18.0, "gw_gamma": 4.33, "gw_auto_log10_A": -13.0, "gw_auto_gamma": 4.33}
    Phi = np.asarray(gp.Phi.getN(pd))
    assert Phi.shape[0] == Phi.shape[1]
    assert Phi.shape[0] % npsr == 0
    np.linalg.cholesky(Phi)  # auto-dominant -> PD
