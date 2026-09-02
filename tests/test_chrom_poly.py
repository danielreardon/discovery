"""Tests for the chromatic polynomial GP and the varying-basis improper GP.

The polynomial models the part of a chromatic process below 1/Tspan, which the
Fourier basis cannot reach. Its basis depends on the chromatic index, and the two
operations that makes necessary -- removing the timing-model span and
orthonormalising -- are what these tests pin.
"""

import numpy as np
import pytest
from pathlib import Path

try:
    import discovery as ds
    from discovery import signals as s
    HAVE_DISCOVERY = True
except ImportError:
    HAVE_DISCOVERY = False

ALPHAS = [0.0, 2.0, 3.0, 6.0, 10.0, 14.0]


@pytest.fixture
def psr():
    """A real pulsar: the polynomial needs a real time span and timing model.

    tests/data/multi_backend_pulsar.feather spans hours and fits five parameters, so
    [1, t, t**2] does not collide with the timing model there and the behaviour under
    test does not arise.
    """
    if not HAVE_DISCOVERY:
        pytest.skip("discovery package not installed")
    f = Path(__file__).parent.parent / "data" / "v1p1_de440_pint_bipm2019-J0030+0451.feather"
    if not f.exists():
        pytest.skip("pulsar data fixture missing")
    return ds.Pulsar.read_feather(f)


def q_tm(psr):
    return np.linalg.qr(s.normalise_tm_basis(psr))[0]


def evaluate(gp, psr, alpha, name="chrom_gp"):
    """The design matrix at one chromatic index; gp.F takes a parameter dict."""
    return np.asarray(gp.F({f"{psr.name}_{name}_alpha": alpha}))


# --- the basis ---------------------------------------------------------------

def test_a_free_index_gives_a_callable_basis_and_a_fixed_one_a_constant(psr):
    free = s.makegp_chrom_poly_svd(psr, name="chrom_gp")
    assert callable(free.F)
    assert free.index == {f"{psr.name}_chrom_gp_coefficients(3)": slice(0, 3)}

    fixed = s.makegp_chrom_poly_svd(psr, name="chrom_gp",
                                    noisedict={f"{psr.name}_chrom_gp_alpha": 4.0})
    assert not callable(fixed.F)
    assert np.asarray(fixed.F).shape[1] == 3


@pytest.mark.parametrize("alpha", ALPHAS)
def test_the_basis_is_orthonormal_at_every_index(psr, alpha):
    """The prior is isotropic, so it puts the same signal power in at every alpha."""
    F = evaluate(s.makegp_chrom_poly_svd(psr, name="chrom_gp"), psr, alpha)
    assert np.allclose(F.T @ F, np.eye(F.shape[1]), atol=1e-10)


@pytest.mark.parametrize("alpha", [a for a in ALPHAS if a >= 3.0])
def test_the_basis_is_orthogonal_to_the_timing_model(psr, alpha):
    F = evaluate(s.makegp_chrom_poly_svd(psr, name="chrom_gp"), psr, alpha)
    assert np.abs(q_tm(psr).T @ F).max() < 1e-8


@pytest.mark.parametrize("alpha", [0.0, 2.0])
def test_below_the_usual_prior_floor_there_is_nothing_left_to_orthonormalise(psr, alpha):
    """At alpha 0 the raw basis is the spin terms and at alpha 2 it is DM and its
    derivatives, so the projection leaves singular values at machine epsilon and the
    orthonormalisation cannot deliver a basis orthogonal to the timing model. The
    chromatic index prior starts above this."""
    raw = np.asarray(s.chrom_poly_basis(psr)(alpha), dtype=float)
    Q = q_tm(psr)
    sv = np.linalg.svd(raw - Q @ (Q.T @ raw), compute_uv=False)
    assert sv[-1] / np.linalg.svd(raw, compute_uv=False)[0] < 1e-12


def test_the_raw_basis_really_does_collide_with_the_timing_model(psr):
    """The premise of the projection, measured rather than assumed."""
    raw = s.chrom_poly_basis(psr)
    Q = q_tm(psr)
    inside = {}
    for alpha in (0.0, 2.0, 3.0):
        F = np.asarray(raw(alpha), dtype=float)
        inside[alpha] = 1.0 - np.sum((F - Q @ (Q.T @ F))**2) / np.sum(F**2)

    assert inside[0.0] > 0.999    # exactly the spin terms
    assert inside[2.0] > 0.99     # DM and its derivatives
    assert inside[3.0] > 0.9      # still nearly degenerate at the usual prior floor


def test_the_orthonormalisation_flattens_the_index_dependent_volume_term(psr):
    """Under an improper prior the alpha-dependent term is -1/2 logdet(F^T N^-1 F),
    which is not invariant under F -> lambda F while alpha rescales the columns. Left
    raw it rewards the alphas where the basis collapses into the timing model."""
    Ninv = 1.0 / np.asarray(psr.toaerrs)**2
    Q, raw = q_tm(psr), s.chrom_poly_basis(psr)
    gp = s.makegp_chrom_poly_svd(psr, name="chrom_gp")

    def volume(F):
        return -0.5 * np.linalg.slogdet(F.T @ (F * Ninv[:, None]))[1]

    unshaped, orthonormal = [], []
    for alpha in ALPHAS[2:]:                      # over the usual prior support
        F = np.asarray(raw(alpha), dtype=float)
        unshaped.append(volume(F - Q @ (Q.T @ F)))
        orthonormal.append(volume(evaluate(gp, psr, alpha)))

    assert np.ptp(unshaped) > 10.0                # the pathology is real
    assert np.ptp(orthonormal) < 1.0              # and the orthonormalisation removes it


def test_the_svd_of_the_temporal_design_leaves_the_span_unchanged(psr):
    """chrom_poly_basis orthonormalises [1, t, t**2] before the chromatic scaling; that
    is a fixed right-multiplication, so it must not move the column space."""
    t = (psr.toas - np.mean(psr.toas)) / 3.15576e7
    plain = np.vstack([np.ones_like(t), t, t**2]).T
    for alpha in (3.0, 10.0):
        chrom = (1400.0 / np.asarray(psr.freqs))**alpha
        a = np.linalg.qr(plain * chrom[:, None])[0]
        b = np.linalg.qr(np.asarray(s.chrom_poly_basis(psr, fref=1400.0)(alpha)))[0]
        assert np.allclose(np.linalg.svd(a.T @ b, compute_uv=False), 1.0, atol=1e-8)


# --- the extra projection ----------------------------------------------------

def test_project_removes_a_further_basis(psr):
    """project= takes anything with a fixed design matrix, e.g. a time-constant
    frequency-dependent term overlapping the polynomial's constant-in-time part."""
    extra = np.asarray(s.chrom_poly_basis(psr)(9.0), dtype=float)
    gp = s.makegp_chrom_poly_svd(psr, name="chrom_gp", project=extra)
    F = evaluate(gp, psr, 6.0)

    Q = np.linalg.qr(extra - q_tm(psr) @ (q_tm(psr).T @ extra))[0]
    assert np.abs(Q.T @ F).max() < 1e-8


def test_project_refuses_a_basis_with_no_fixed_span(psr):
    other = s.makegp_chrom_poly_svd(psr, name="other")      # its F is callable
    with pytest.raises(ValueError, match="callable F"):
        s.makegp_chrom_poly_svd(psr, name="chrom_gp", project=other)


# --- the timing-model helper -------------------------------------------------

def test_normalise_tm_basis_unit_normalises_and_drops_empty_columns(psr):
    M = s.normalise_tm_basis(psr)
    assert np.allclose(np.sum(M**2, axis=0), 1.0)

    raw = np.asarray(psr.Mmat, dtype=np.float64)
    nonzero = int((np.sqrt(np.sum(raw**2, axis=0)) > 0).sum())
    assert M.shape[1] == nonzero
