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


def _labels(psr):
    return sorted(set(np.asarray(s.selection_backend_flags(psr)).tolist()))


def test_default_is_a_single_global_basis(psr):
    """No selection: one block over all TOAs, and fd_nodes is just the node array."""
    gp = s.makegp_fd_piecewise(psr, nodes=8)
    assert not isinstance(gp.fd_nodes, dict)
    assert np.asarray(gp.fd_nodes).ndim == 1


def test_all_groups_gets_no_separate_global(psr):
    """With every group selected the per-group bases already span common structure."""
    gp = s.makegp_fd_piecewise(psr, nodes=8, selection=s.selection_backend_flags)
    assert set(gp.fd_nodes) == set(_labels(psr))
    assert None not in gp.fd_nodes


def test_subset_of_groups_keeps_the_global(psr):
    """Selecting only some groups leaves the rest to the global basis."""
    sub = _labels(psr)[:2]
    gp = s.makegp_fd_piecewise(psr, nodes=8, selection=s.selection_backend_flags, groups=sub)
    assert set(gp.fd_nodes) == {None, *sub}

    # and it carries more freedom than the global alone
    alone = np.asarray(s.makegp_fd_piecewise(psr, nodes=8).F).shape[1]
    assert np.asarray(gp.F).shape[1] > alone


def test_unknown_group_warns_and_is_skipped(psr):
    gp = s.makegp_fd_piecewise(psr, nodes=8, selection=s.selection_backend_flags,
                               groups=['not-a-backend'])
    assert list(gp.fd_nodes) == [None]           # falls back to the global block


def test_user_supplied_list_of_selections(psr):
    """A list of selections lets the caller compose blocks explicitly."""
    def selection_global(p):
        return np.array(['global'] * len(p.toas))

    gp = s.makegp_fd_piecewise(psr, nodes=8,
                               selection=[selection_global, s.selection_backend_flags])
    assert set(gp.fd_nodes) == {'global', *_labels(psr)}


def test_selection_gives_one_basis_per_group(psr):
    """A selection builds a per-group basis, combined into one marginalised GP."""
    plain = s.makegp_fd_piecewise(psr, nodes=8)
    grouped = s.makegp_fd_piecewise(psr, nodes=8, selection=s.selection_backend_flags)

    ngroups = len(set(np.asarray(s.selection_backend_flags(psr)).tolist()))
    assert isinstance(grouped.fd_nodes, dict) and len(grouped.fd_nodes) == ngroups

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


# --- Matern-3/2 prior over the node amplitudes ------------------------------

NAME = 'fd_gp'


@pytest.fixture
def restore_priors():
    from discovery import prior
    saved = dict(prior.priordict_standard)
    yield prior
    prior.priordict_standard.clear()
    prior.priordict_standard.update(saved)


def _hypernames(psr, name=NAME):
    return f'{psr.name}_{name}_log10_sigma', f'{psr.name}_{name}_log10_ell'


def _edof(gp, params, Nvec):
    """tr[F (Ft Nm F + Phi^-1)^-1 Ft Nm F], the directions the GP actually absorbs."""
    F = np.asarray(gp.F)
    FtNmF = F.T @ (F / Nvec[:, None])
    Phi = np.asarray(gp.Phi.getN(params))
    Pinv = np.linalg.inv(Phi) if Phi.ndim == 2 else np.diag(1.0 / Phi)
    return float(np.trace(np.linalg.solve(FtNmF + Pinv, FtNmF)))


def _Nvec(psr):
    return np.asarray(psr.toaerrs, dtype=np.float64) ** 2


def test_matern_keeps_one_column_per_node(psr, restore_priors):
    """The basis is not orthonormalised, so amplitudes still map to nodes."""
    gp = s.makegp_fd_piecewise_matern(psr, nodes=16)

    assert gp.F.shape[1] == 16
    assert len(gp.fd_nodes) == 16


def test_matern_tolerates_a_rank_deficient_basis(psr, restore_priors):
    """The timing-model projection annihilates directions; Phi is inverted, not F."""
    gp = s.makegp_fd_piecewise_matern(psr, nodes=16)

    assert np.linalg.matrix_rank(np.asarray(gp.F)) < gp.F.shape[1]
    sig, ell = _hypernames(psr)
    assert np.all(np.isfinite(gp.Phi.make_inv()({sig: -7.0, ell: -0.5})[0]))


def test_matern_ell_prior_is_bounded_by_the_observed_band(psr, restore_priors):
    """log10_ell runs from a tenth of the node span to three times it."""
    gp = s.makegp_fd_piecewise_matern(psr, nodes=16)
    sig, ell = _hypernames(psr)

    lo, hi = restore_priors.getprior_uniform(ell)
    span = np.log(gp.fd_nodes.max()) - np.log(gp.fd_nodes.min())
    assert lo == pytest.approx(np.log10(0.1 * span))
    assert hi == pytest.approx(np.log10(3.0 * span))
    assert restore_priors.getprior_uniform(sig) == [-10.0, -4.0]


def test_matern_prior_is_positive_definite_across_its_range(psr, restore_priors):
    """Phi stays a valid covariance everywhere the sampler can go."""
    gp = s.makegp_fd_piecewise_matern(psr, nodes=32)
    sig, ell = _hypernames(psr)
    lo, hi = restore_priors.getprior_uniform(ell)

    for le in (lo, 0.5 * (lo + hi), hi):
        w = np.linalg.eigvalsh(np.asarray(gp.Phi.getN({sig: -7.0, ell: le})))
        assert w.min() > 0.0
        assert w.max() / w.min() < 1e10


def test_matern_jitter_bounds_the_condition_number(psr, restore_priors):
    """Without the jitter the kernel approaches rank one as ell passes the band."""
    sig, ell = _hypernames(psr)
    nodes = 32
    conds = {}
    for jitter in (1e-12, 1e-8):
        gp = s.makegp_fd_piecewise_matern(psr, nodes=nodes, jitter=jitter)
        hi = restore_priors.getprior_uniform(ell)[1]
        w = np.linalg.eigvalsh(np.asarray(gp.Phi.getN({sig: -7.0, ell: hi})))
        conds[jitter] = w.max() / w.min()

    # the largest eigenvalue of the flat-kernel limit grows with the node count,
    # so the jitter caps the ratio near nodes/jitter rather than 1/jitter
    assert conds[1e-8] < conds[1e-12]
    assert conds[1e-8] < 10.0 * nodes / 1e-8


def test_matern_inverse_and_gradients_are_finite_across_the_prior(psr, restore_priors):
    """Phi^-1, log|Phi| and their gradients survive any draw from the prior."""
    import jax
    import jax.numpy as jnp

    gp = s.makegp_fd_piecewise_matern(psr, nodes=16)
    sig, ell = _hypernames(psr)
    slo, shi = restore_priors.getprior_uniform(sig)
    elo, ehi = restore_priors.getprior_uniform(ell)
    inv = gp.Phi.make_inv()

    rng = np.random.default_rng(0)
    for _ in range(50):
        p = {sig: rng.uniform(slo, shi), ell: rng.uniform(elo, ehi)}
        Pinv, ld = inv(p)
        assert np.isfinite(np.asarray(Pinv)).all()
        assert np.isfinite(float(ld))

    grad = jax.grad(lambda q: jnp.sum(inv(q)[0]) + inv(q)[1])
    got = grad({sig: -7.0, ell: 0.5 * (elo + ehi)})
    assert all(np.isfinite(float(v)) for v in got.values())


def test_matern_blocks_are_independent_under_a_selection(psr, restore_priors):
    """Disjoint selections give a block-diagonal Phi with shared hyperparameters."""
    def two_bands(p):
        return np.where(np.asarray(p.freqs) < np.median(p.freqs), 'lo', 'hi')

    gp = s.makegp_fd_piecewise_matern(psr, nodes=8, selection=two_bands)
    sig, ell = _hypernames(psr)
    Phi = np.asarray(gp.Phi.getN({sig: -7.0, ell: -0.5}))

    n = gp.F.shape[1] // 2
    assert Phi.shape == (gp.F.shape[1], gp.F.shape[1])
    assert np.all(Phi[:n, n:] == 0.0)
    assert np.all(Phi[n:, :n] == 0.0)
    assert sorted(gp.Phi.params) == sorted([sig, ell])


def test_matern_effective_dof_rises_with_amplitude(psr, restore_priors):
    """A small sigma leaves the directions in the data; a large one absorbs them."""
    gp = s.makegp_fd_piecewise_matern(psr, nodes=16)
    sig, ell = _hypernames(psr)
    Nvec = _Nvec(psr)
    mid = float(np.mean(restore_priors.getprior_uniform(ell)))

    dofs = [_edof(gp, {sig: sv, ell: mid}, Nvec) for sv in (-10.0, -8.0, -6.0, -4.0)]

    assert dofs[0] < 0.5
    assert all(a <= b + 1e-8 for a, b in zip(dofs, dofs[1:]))
    assert dofs[-1] <= np.linalg.matrix_rank(np.asarray(gp.F)) + 1e-6


def test_the_improper_prior_spends_every_direction(psr, restore_priors):
    """The improper GP marginalises its whole basis whatever the data say."""
    gp = s.makegp_fd_piecewise(psr, nodes=16)
    Nvec = _Nvec(psr)

    F = np.asarray(gp.F)
    FtNmF = F.T @ (F / Nvec[:, None])
    Pinv = np.diag(1.0 / np.asarray(gp.Phi.N))
    dof = float(np.trace(np.linalg.solve(FtNmF + Pinv, FtNmF)))

    assert dof == pytest.approx(F.shape[1], abs=1e-6)


@pytest.mark.parametrize("nodes", [16, 32, 64])
def test_matern_effective_dof_is_stable_in_node_count(psr, restore_priors, nodes):
    """Raising the node count refines the basis without buying free parameters."""
    gp = s.makegp_fd_piecewise_matern(psr, nodes=nodes)
    sig, ell = _hypernames(psr)
    mid = float(np.mean(restore_priors.getprior_uniform(ell)))

    assert _edof(gp, {sig: -7.0, ell: mid}, _Nvec(psr)) < 0.5 * nodes


def test_matern_effective_dof_agrees_between_spacings(psr, restore_priors):
    """At high node count the two node placements describe the same model."""
    Nvec = _Nvec(psr)
    dofs = {}
    for spacing in ('quantile', 'log'):
        gp = s.makegp_fd_piecewise_matern(psr, nodes=64, spacing=spacing)
        sig, ell = _hypernames(psr)
        mid = float(np.mean(restore_priors.getprior_uniform(ell)))
        dofs[spacing] = _edof(gp, {sig: -7.0, ell: mid}, Nvec)

    assert dofs['quantile'] == pytest.approx(dofs['log'], rel=0.25)


def test_matern_warns_when_the_timing_model_fits_fd(psr, restore_priors, monkeypatch):
    """FD columns carry the smooth band structure the projection would strip."""
    monkeypatch.setattr(psr, 'fitpars', ['F0', 'F1', 'FD1', 'FD2'], raising=False)

    with pytest.warns(UserWarning, match='FD'):
        s.makegp_fd_piecewise_matern(psr, nodes=16)


def test_matern_does_not_warn_without_fd(psr, restore_priors, monkeypatch):
    """A par file with no FD terms needs no warning."""
    import warnings as w

    monkeypatch.setattr(psr, 'fitpars', ['F0', 'F1', 'DM1'], raising=False)

    with w.catch_warnings():
        w.simplefilter('error')
        s.makegp_fd_piecewise_matern(psr, nodes=16)


def test_matern_unknown_spacing_raises(psr, restore_priors):
    """Node placement must be one of the two supported rules."""
    with pytest.raises(ValueError, match='spacing'):
        s.makegp_fd_piecewise_matern(psr, nodes=16, spacing='linear')


@pytest.mark.integration
def test_matern_drives_a_real_likelihood(psr, restore_priors):
    """The GP composes into a pulsar likelihood with a finite gradient."""
    import jax

    from discovery import prior as _p

    gp = s.makegp_fd_piecewise_matern(psr, nodes=16)
    psl = ds.PulsarLikelihood([psr.residuals, ds.makenoise_measurement(psr), gp])

    t = _p.makelogtransform_uniform(psl.logL)
    ys = np.zeros(len(t.params))

    assert np.isfinite(float(t(ys)))
    assert np.isfinite(np.asarray(jax.grad(t)(ys))).all()
