"""Tests for the robust (fcenter, log10_bw) band-noise models.

Covers the normalised band envelope, the Fourier- and FFT-covariance band bases
parametrised by band centre and log10 bandwidth, and the back-compatibility (with
DeprecationWarning) of the legacy (flow, fhigh) ``*_band_range*`` functions.
"""

import numpy as np
import jax.numpy as jnp
import pytest
from pathlib import Path

try:
    import discovery as ds
    from discovery import signals
    HAVE_DISCOVERY = True
except ImportError:
    HAVE_DISCOVERY = False


@pytest.fixture(autouse=True)
def _restore_priordict():
    """Snapshot/restore the global prior dict.

    Importing discovery.models.mpta (done by the per-pulsar prior test) runs
    update_priordict_standard_mpta(), which mutates the shared prior.priordict_standard
    in place. Restore it after each test so this module never leaks PTA-specific or
    per-pulsar prior overrides into other test modules.
    """
    if not HAVE_DISCOVERY:
        yield
        return
    from discovery import prior
    snapshot = dict(prior.priordict_standard)
    yield
    prior.priordict_standard.clear()
    prior.priordict_standard.update(snapshot)


@pytest.fixture
def data_dir():
    return Path(__file__).parent / "data"


@pytest.fixture
def psr(data_dir):
    if not HAVE_DISCOVERY:
        pytest.skip("discovery package not installed")
    return ds.Pulsar.read_feather(data_dir / "multi_backend_pulsar.feather")


@pytest.fixture
def fcenter(psr):
    return 0.5 * (float(psr.freqs.min()) + float(psr.freqs.max()))


# ---------------------------------------------------------------------------
# Envelope: normalisation, finiteness, no dead zone
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize("log10_bw", [1.3, 1.7, 2.0, 2.5, 2.9])
def test_band_envelope_unit_rms(psr, fcenter, log10_bw):
    """The envelope is RMS-normalised to ~1 at any bandwidth (amplitude-width decoupling)."""
    env = signals._band_envelope(psr, fcenter, log10_bw)
    rms = float(jnp.sqrt(jnp.mean(env ** 2)))
    assert np.isclose(rms, 1.0, atol=1e-3)


@pytest.mark.unit
@pytest.mark.parametrize("log10_bw", [1.0, 2.0, 3.0])
def test_band_envelope_finite_and_positive(psr, fcenter, log10_bw):
    """Envelope is finite and non-negative everywhere, and never identically zero."""
    env = np.asarray(signals._band_envelope(psr, fcenter, log10_bw))
    assert np.all(np.isfinite(env))
    assert np.all(env >= 0.0)
    assert env.sum() > 0.0


@pytest.mark.unit
def test_band_envelope_centred(psr, fcenter):
    """In-band TOAs carry more weight than out-of-band ones."""
    freqs = np.asarray(psr.freqs)
    env = np.asarray(signals._band_envelope(psr, fcenter, np.log10(100.0)))  # 100 MHz band
    inband = np.abs(freqs - fcenter) < 50.0
    if inband.any() and (~inband).any():
        assert env[inband].mean() > env[~inband].mean()


# ---------------------------------------------------------------------------
# Fourier-domain band-width bases
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_fourierbasis_band_width(psr, fcenter):
    """fourierbasis_band_width returns a callable basis with the expected shape."""
    f, df, fmatfunc = signals.fourierbasis_band_width(psr, 5)
    assert callable(fmatfunc)
    F = np.asarray(fmatfunc(fcenter, 2.0))
    assert F.shape == (len(psr.toas), 10)
    assert np.all(np.isfinite(F))


@pytest.mark.unit
def test_fourierbasis_band_width_alpha_scales(psr, fcenter):
    """The alpha variant adds (fref/freqs)**alpha on top of the band envelope."""
    fref = 1400.0
    _, _, base = signals.fourierbasis_band_width(psr, 5)
    _, _, achr = signals.fourierbasis_band_width_alpha(psr, 5, fref=fref)
    F0 = np.asarray(base(fcenter, 2.0))
    Fa = np.asarray(achr(fcenter, 2.0, 1.5))
    expected = F0 * ((fref / np.asarray(psr.freqs)) ** 1.5)[:, None]
    assert np.allclose(Fa, expected)


# ---------------------------------------------------------------------------
# FFT-covariance band-width GPs
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_makegp_fftcov_band_width(psr):
    """makegp_fftcov_band_width builds a band GP parametrised by (fcenter, log10_bw)."""
    gp = signals.makegp_fftcov_band_width(psr, signals.powerlaw, components=5)
    assert gp.gpname == "band_gp"
    assert callable(gp.F)
    assert gp.F.params == [f"{psr.name}_band_gp_fcenter", f"{psr.name}_band_gp_log10_bw"]
    assert sorted(gp.Phi.params) == sorted(
        [f"{psr.name}_band_gp_log10_A", f"{psr.name}_band_gp_gamma"]
    )


@pytest.mark.unit
def test_makegp_fftcov_band_width_alpha(psr):
    """The alpha variant additionally exposes a free chromatic index."""
    gp = signals.makegp_fftcov_band_width_alpha(psr, signals.powerlaw, components=5)
    assert gp.gpname == "bandalpha_gp"
    assert gp.F.params == [
        f"{psr.name}_bandalpha_gp_fcenter",
        f"{psr.name}_bandalpha_gp_log10_bw",
        f"{psr.name}_bandalpha_gp_alpha",
    ]


@pytest.mark.unit
def test_band_width_uses_centre_width_not_edges(psr):
    """The robust model exposes fcenter/log10_bw, never the legacy flow/fhigh."""
    gp = signals.makegp_fftcov_band_width(psr, signals.powerlaw, components=5)
    joined = " ".join(gp.F.params)
    assert "fcenter" in joined and "log10_bw" in joined
    assert "flow" not in joined and "fhigh" not in joined


# ---------------------------------------------------------------------------
# Per-pulsar, data-bounded priors
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_set_band_priors_bounds_to_data(psr):
    """_set_band_priors bounds fcenter to the pulsar's coverage and overrides any generic fallback."""
    from discovery import prior
    from discovery.models import mpta
    fmin, fmax = float(psr.freqs.min()), float(psr.freqs.max())
    mpta._set_band_priors(psr, band=True, band_alpha=True)

    fcenter_prior = prior.getprior_uniform(f"{psr.name}_band_gp_fcenter")
    assert np.isclose(fcenter_prior[0], fmin) and np.isclose(fcenter_prior[1], fmax)
    # bandwidth runs from the floor up to the full coverage span
    bw_prior = prior.getprior_uniform(f"{psr.name}_band_gp_log10_bw")
    assert np.isclose(bw_prior[1], np.log10(fmax - fmin))
    # amplitude still resolves to the generic GP prior
    assert prior.getprior_uniform(f"{psr.name}_band_gp_log10_A") == [-18, -11]


# ---------------------------------------------------------------------------
# Deprecated (flow, fhigh) band_range functions
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_fourierbasis_band_range_deprecated(psr):
    with pytest.warns(DeprecationWarning, match="fourierbasis_band_width"):
        signals.fourierbasis_band_range(psr, 5)


@pytest.mark.unit
def test_fourierbasis_band_range_alpha_deprecated(psr):
    with pytest.warns(DeprecationWarning, match="fourierbasis_band_width_alpha"):
        signals.fourierbasis_band_range_alpha(psr, 5)


@pytest.mark.unit
def test_make_timeinterpbasis_band_range_deprecated():
    with pytest.warns(DeprecationWarning, match="make_timeinterpbasis_band_width"):
        signals.make_timeinterpbasis_band_range()


@pytest.mark.unit
def test_make_timeinterpbasis_band_range_alpha_deprecated():
    with pytest.warns(DeprecationWarning, match="make_timeinterpbasis_band_width_alpha"):
        signals.make_timeinterpbasis_band_range_alpha()


@pytest.mark.unit
def test_makegp_fftcov_band_range_deprecated(psr):
    with pytest.warns(DeprecationWarning, match="makegp_fftcov_band_width"):
        signals.makegp_fftcov_band_range(psr, signals.powerlaw, components=5)


@pytest.mark.unit
def test_makegp_fftcov_band_range_alpha_deprecated(psr):
    with pytest.warns(DeprecationWarning, match="makegp_fftcov_band_width_alpha"):
        signals.makegp_fftcov_band_range_alpha(psr, signals.powerlaw, components=5)
