#!/usr/bin/env python3
"""Regression tests for the optimal statistic (discovery.optimal).

These pin the OS outputs on a small, fixed 3-pulsar dataset so that
performance refactors (e.g. rewriting ``trace(A @ B)`` as ``sum(A * B)``
in the pairwise normalization) provably do not change results.

The golden values were generated from the implementation *before* the
trace->sum refactor, with the fixed ``PARAMS`` below. ``rtol=1e-8`` is far
looser than the ~1e-12 floating-point reassociation the refactor introduces,
but tight enough to catch any genuine change in the computation.
"""

from pathlib import Path
import pytest

import numpy as np

import jax
jax.config.update('jax_enable_x64', True)

import discovery as ds
from discovery import optimal


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PSR_FILES = [
    "v1p1_de440_pint_bipm2019-B1855+09.feather",
    "v1p1_de440_pint_bipm2019-J0023+0923.feather",
    "v1p1_de440_pint_bipm2019-J0030+0451.feather",
]

# fixed parameters used to generate the golden values
PARAMS = {
    'B1855+09_rednoise_gamma': 3.2,   'B1855+09_rednoise_log10_A': -14.5,
    'J0023+0923_rednoise_gamma': 3.2, 'J0023+0923_rednoise_log10_A': -14.5,
    'J0030+0451_rednoise_gamma': 3.2, 'J0030+0451_rednoise_log10_A': -14.5,
    'gw_gamma': 3.2,                  'gw_log10_A': -14.5,
}

# golden values (generated pre-refactor; see module docstring)
GOLDEN_OS = {
    'hd':     dict(os=8.3961773831359343e-29, os_sigma=3.3828949412484901e-29, snr=2.4819503794691431e+00),
    'mono':   dict(os=1.7353193676835731e-30, os_sigma=9.2639350397105645e-30, snr=1.8731989810431465e-01),
    'dipole': dict(os=3.7929188575728078e-29, os_sigma=1.8191630514555593e-29, snr=2.0849801531193126e+00),
}
GOLDEN_Q = dict(eig_min=-1.7242147637816876e-01, eig_max=2.0158562767707627e-01)
GOLDEN_CDF_X = np.array([0.0, 1.0, 2.0])
GOLDEN_CDF = np.array([5.1039176222643856e-01, 8.5178948105146968e-01, 9.7186069481346693e-01])

RTOL = 1e-8


@pytest.fixture(scope="module")
def os_obj():
    psrs = [ds.Pulsar.read_feather(DATA_DIR / f) for f in PSR_FILES]
    T = ds.getspan(psrs)
    gbl = ds.GlobalLikelihood([
        ds.PulsarLikelihood([
            psr.residuals,
            ds.makenoise_measurement(psr, psr.noisedict),
            ds.makegp_ecorr(psr, psr.noisedict),
            ds.makegp_timing(psr, svd=True),
            ds.makegp_fourier(psr, ds.powerlaw, 30, T=T, name='rednoise'),
            ds.makegp_fourier(psr, ds.powerlaw, 14, T=T,
                              common=['gw_log10_A', 'gw_gamma'], name='gw'),
        ]) for psr in psrs])
    return ds.OS(gbl)


@pytest.mark.parametrize("orfname,orf", [
    ('hd', optimal.hd_orfa),
    ('mono', optimal.monopole_orfa),
    ('dipole', optimal.dipole_orfa),
])
def test_os_regression(os_obj, orfname, orf):
    """OS point estimate / sigma / snr must match pinned golden values."""
    result = os_obj.os(PARAMS, orf)
    golden = GOLDEN_OS[orfname]
    for key in ('os', 'os_sigma', 'snr'):
        np.testing.assert_allclose(float(result[key]), golden[key], rtol=RTOL,
                                   err_msg=f"{orfname} {key} changed")


def test_Q_eigenvalues_regression(os_obj):
    """Eigenvalues of the OS quadratic-form matrix Q must be unchanged."""
    eigs = np.linalg.eigvalsh(np.asarray(os_obj.Q(PARAMS)))
    np.testing.assert_allclose(eigs.min(), GOLDEN_Q['eig_min'], rtol=RTOL)
    np.testing.assert_allclose(eigs.max(), GOLDEN_Q['eig_max'], rtol=RTOL)


def test_gx2cdf_regression(os_obj):
    """quadax-based gx2cdf must match pinned golden values."""
    pytest.importorskip("quadax", reason="gx2cdf needs the optional quadax dependency")
    cdf = np.asarray(os_obj.gx2cdf(PARAMS, GOLDEN_CDF_X))
    np.testing.assert_allclose(cdf, GOLDEN_CDF, rtol=RTOL)


def test_imhof_u0_limit():
    """The Imhof integrand has a removable 0/0 at u=0; it must return the
    finite limit 1/2 (sum(eigs) - x), not nan (quadax may sample the endpoint)."""
    eigs = jax.numpy.array([0.20, 0.15, -0.17, -0.05, 0.02, -0.01, 1e-4, -1e-4])
    for x in (0.0, 1.0, 2.0):
        val = float(optimal.imhof(jax.numpy.array(0.0), x, eigs))
        assert np.isfinite(val)
        np.testing.assert_allclose(val, 0.5 * (float(np.sum(np.asarray(eigs))) - x),
                                   rtol=1e-12, atol=0)


def test_shift_zero_phase_matches_os(os_obj):
    """A zero phase shift must reproduce the unshifted OS exactly.

    OS.shift was dead code until the ``sN.ndim`` -> ``N.ndim`` fix in
    os_rhosigma_complex (it raised UnboundLocalError on every call), so this
    is the first test to exercise it. ``phases`` is per-pulsar per-frequency,
    matching the (npair, nfreq) shape of rhos_complex.
    """
    nf = 14   # gw components in the fixture
    npsr = len(PSR_FILES)
    shifted = os_obj.shift(PARAMS, np.zeros((npsr, nf)))
    plain = os_obj.os(PARAMS)
    for key in ('os', 'os_sigma', 'snr'):
        np.testing.assert_allclose(float(shifted[key]), float(plain[key]), rtol=1e-10)


def test_shift_random_phases_are_a_null(os_obj):
    """Random phase shifts must give a finite, roughly zero-mean null."""
    nf, npsr = 14, len(PSR_FILES)
    rng = np.random.default_rng(0)
    snrs = np.array([float(os_obj.shift(PARAMS, rng.uniform(0, 2 * np.pi, (npsr, nf)))['snr'])
                     for _ in range(50)])
    assert np.all(np.isfinite(snrs))
    assert abs(snrs.mean()) < 3.0 * snrs.std() / np.sqrt(len(snrs))


def test_inv_prior_matches_diag_for_1d():
    """For a diagonal (1-D) prior inv_prior must reproduce jnp.diag(1/P)."""
    rng = np.random.default_rng(0)
    P = 10.0 ** rng.uniform(-18, -14, size=12)
    np.testing.assert_allclose(np.asarray(optimal.inv_prior(P)), np.diag(1.0 / P), rtol=1e-12)


def test_inv_prior_keeps_2d_correlations():
    """For a 2-D prior inv_prior must invert the full matrix, not its diagonal.

    matrix.CompoundGP returns a 2-D block-diagonal prior whenever any intrinsic
    GP has a NoiseMatrix2D_var prior (the correlated Legendre ECORR). The old
    jnp.diag(1.0 / P) silently discarded the off-diagonal mode correlations.
    """
    rng = np.random.default_rng(1)
    A = rng.normal(size=(8, 8))
    P = A @ A.T + 8 * np.eye(8)                     # correlated, well conditioned

    got = np.asarray(optimal.inv_prior(P))
    np.testing.assert_allclose(got @ P, np.eye(8), atol=1e-10)
    # and it is genuinely different from the diagonal-only inverse
    assert np.abs(got - np.diag(1.0 / np.diag(P))).max() > 1e-3 * np.abs(got).max()


def test_inv_prior_rejects_higher_rank():
    with pytest.raises(ValueError, match="1-D or 2-D"):
        optimal.inv_prior(np.ones((2, 2, 2)))


@pytest.mark.integration
def test_correlated_ecorr_changes_Q_and_blocks_sampling():
    """End-to-end: a correlated-Legendre-ECORR model gives a 2-D P_var, whose
    off-diagonal terms measurably change Q, and sample_rhosigma refuses it."""
    psrs = [ds.Pulsar.read_feather(DATA_DIR / f) for f in PSR_FILES[:2]]
    T = ds.getspan(psrs)
    gbl = ds.GlobalLikelihood([
        ds.PulsarLikelihood([
            psr.residuals,
            ds.makenoise_measurement(psr, psr.noisedict),
            ds.makegp_ecorr_legendre_correlated(psr, psr.noisedict, variable=True),
            ds.makegp_timing(psr, svd=True),
            ds.makegp_fourier(psr, ds.powerlaw, 30, T=T, name='rednoise'),
            ds.makegp_fourier(psr, ds.powerlaw, 14, T=T,
                              common=['gw_log10_A', 'gw_gamma'], name='gw'),
        ]) for psr in psrs])
    os_obj = ds.OS(gbl)

    # every parameter is pinned: the size of the effect depends on how strongly
    # correlated the Legendre modes are, so a random draw would make the
    # assertion below flaky (it varies by two orders of magnitude across draws)
    params = {}
    for k in os_obj.params:
        if k.endswith('rednoise_gamma') or k == 'gw_gamma':
            params[k] = 3.2
        elif k.endswith('rednoise_log10_A') or k == 'gw_log10_A':
            params[k] = -14.5
        elif '_ecorr_corr_' in k:
            params[k] = 0.7            # strong off-diagonal mode correlation
        elif 'log10_ecorr' in k:
            params[k] = -6.5
        else:
            raise AssertionError(f"unpinned parameter {k!r}; update this test")

    # the prior really is 2-D here, otherwise this test proves nothing
    assert all(np.asarray(psl.N.P_var.getN(params)).ndim == 2 for psl in gbl.psls)

    Q_fixed = np.asarray(os_obj.Q(params))
    assert np.all(np.isfinite(Q_fixed))

    orig = optimal.inv_prior
    try:   # emulate the old diag-only inverse
        optimal.inv_prior = lambda P: (np.diag(1.0 / np.asarray(P)) if np.asarray(P).ndim == 1
                                       else np.diag(1.0 / np.diag(np.asarray(P))))
        Q_old = np.asarray(ds.OS(gbl).Q(params))
    finally:
        optimal.inv_prior = orig

    rel = np.abs(Q_old - Q_fixed).max() / np.abs(Q_fixed).max()
    assert rel > 1e-3, f"discarding the prior correlations changed Q by only {rel:.2e}"

    with pytest.raises(NotImplementedError, match="diagonal"):
        os_obj.sample_rhosigma(params)


def test_mcos_single_orf_reduces_to_os(os_obj):
    """The multi-component OS with one ORF must reproduce the scalar OS."""
    m = os_obj.mcos(PARAMS, orfs=(optimal.hd_orfa,))
    plain = os_obj.os(PARAMS)
    np.testing.assert_allclose(float(m['os'][0]), float(plain['os']), rtol=RTOL)
    np.testing.assert_allclose(float(m['os_sigma'][0]), float(plain['os_sigma']), rtol=RTOL)
    np.testing.assert_allclose(float(m['snr'][0]), float(plain['snr']), rtol=RTOL)


def test_trace_sum_identity():
    """Document the refactor's justification: trace(A @ B) == sum(A * B)
    for symmetric A, B (so the OS pairwise normalization is unchanged)."""
    rng = np.random.default_rng(0)
    for m in (4, 28):
        A = rng.normal(size=(m, m)); A = A + A.T
        B = rng.normal(size=(m, m)); B = B + B.T
        np.testing.assert_allclose(np.trace(A @ B), np.sum(A * B), rtol=1e-12)
