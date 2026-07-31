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
