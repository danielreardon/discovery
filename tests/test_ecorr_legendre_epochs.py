"""The Legendre ECORR variants use the same epochs as makegp_ecorr.

Single-backend pulsars keep their first epoch, and with enterprise=True a backend without
multi-TOA epochs gets an empty basis rather than an error.
"""
from pathlib import Path

import numpy as np
import pytest

import discovery as ds
from discovery import signals, likelihood

DATA = Path(__file__).resolve().parent / "data"
FIXTURES = ["single_backend_pulsar", "multi_backend_pulsar", "empty_epoch_pulsar"]


def _read(name):
    return ds.Pulsar.read_feather(DATA / f"{name}.feather")


def _partition(F):
    """TOAs grouped by the first column each is nonzero in, its epoch's mode-0 column."""
    F = np.asarray(F) != 0
    rows = np.nonzero(F.any(axis=1))[0]
    first = F[rows].argmax(axis=1)
    return sorted(tuple(rows[first == c]) for c in np.unique(first))


@pytest.mark.parametrize("name", FIXTURES)
@pytest.mark.parametrize("enterprise", [False, True])
@pytest.mark.parametrize("variant", ["legendre1", "legendre3", "correlated2"])
def test_legendre_epochs_are_the_ecorr_epochs(name, enterprise, variant):
    psr = _read(name)
    ref = signals.makegp_ecorr(psr, enterprise=enterprise).F
    gp = {"legendre1": lambda: signals.makegp_ecorr_legendre(psr, nmodes=1, enterprise=enterprise),
          "legendre3": lambda: signals.makegp_ecorr_legendre(psr, nmodes=3, enterprise=enterprise),
          "correlated2": lambda: signals.makegp_ecorr_legendre_correlated(psr, nmodes=2, enterprise=enterprise),
          }[variant]()
    assert _partition(gp.F) == _partition(ref)
    if not enterprise:
        covered = np.asarray(ref) != 0
        assert np.all(covered.sum(axis=1) == (np.asarray(psr.backend_flags) != '').astype(int))


@pytest.mark.parametrize("name", FIXTURES)
def test_one_mode_legendre_is_plain_ecorr(name):
    psr = _read(name)
    backends = sorted(set(np.asarray(psr.backend_flags).tolist()))
    nd = {f'{psr.name}_{b}_efac': 1.0 for b in backends}
    nd.update({f'{psr.name}_{b}_log10_t2equad': -8.0 for b in backends})

    def build(gp):
        return likelihood.PulsarLikelihood([psr.residuals, signals.makegp_timing(psr, svd=True),
                                            signals.makenoise_measurement(psr, noisedict=nd), gp])
    plain = build(signals.makegp_ecorr(psr))
    leg = build(signals.makegp_ecorr_legendre(psr, nmodes=1))
    assert plain.logL.params == leg.logL.params
    for v in (-7.5, -6.5, -5.5):
        x = {p: v for p in plain.logL.params}
        assert float(leg.logL(x)) == pytest.approx(float(plain.logL(x)), rel=1e-12, abs=0)
