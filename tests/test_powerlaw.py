"""Tests for the make_powerlaw PSD factory."""

import numpy as np
import inspect

import jax.numpy as jnp

from discovery import signals as s


F = jnp.asarray(np.linspace(1e-9, 3e-8, 20))
DF = jnp.full(20, 1e-9)
A, G = -14.5, 4.33


def test_both_free_matches_powerlaw():
    """make_powerlaw() with nothing fixed == the plain powerlaw."""
    m = s.make_powerlaw()
    assert list(inspect.signature(m).parameters) == ["f", "df", "log10_A", "gamma"]
    assert np.allclose(m(F, DF, A, G), s.powerlaw(F, DF, A, G))


def test_fixed_gamma_drops_gamma_param():
    m = s.make_powerlaw(gamma=G)
    assert list(inspect.signature(m).parameters) == ["f", "df", "log10_A"]
    assert np.allclose(m(F, DF, A), s.powerlaw(F, DF, A, G))


def test_fixed_log10A_drops_amplitude_param():
    m = s.make_powerlaw(log10_A=A)
    assert list(inspect.signature(m).parameters) == ["f", "df", "gamma"]
    assert np.allclose(m(F, DF, G), s.powerlaw(F, DF, A, G))


def test_both_fixed():
    m = s.make_powerlaw(gamma=G, log10_A=A)
    assert list(inspect.signature(m).parameters) == ["f", "df"]
    assert np.allclose(m(F, DF), s.powerlaw(F, DF, A, G))


def test_gwb_is_fixed_index_powerlaw():
    """Supplying gamma=13/3 gives the isotropic-GWB spectrum."""
    gwb = s.make_powerlaw(gamma=13.0 / 3.0)
    assert np.allclose(gwb(F, DF, A), s.powerlaw(F, DF, A, 13.0 / 3.0))


def test_powerlaw_gwb_sampled_amplitude():
    """powerlaw_gwb() samples log10_A at fixed gamma=13/3."""
    m = s.powerlaw_gwb()
    assert list(inspect.signature(m).parameters) == ["f", "df", "log10_A"]
    assert np.allclose(m(F, DF, A), s.powerlaw(F, DF, A, 13.0 / 3.0))


def test_powerlaw_gwb_fixed_amplitude():
    """powerlaw_gwb(log10_A=A) fixes both gamma and amplitude."""
    m = s.powerlaw_gwb(log10_A=A)
    assert list(inspect.signature(m).parameters) == ["f", "df"]
    assert np.allclose(m(F, DF), s.powerlaw(F, DF, A, 13.0 / 3.0))
