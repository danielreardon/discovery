"""Deterministic physical-ephemeris (BayesEphem 2.0) signal for Discovery.

This is the JAX backend of the broadened, multi-ephemeris BayesEphem model
(Jupiter + Saturn orbital blocks, 3-axis frame rotation rate, outer-planet
masses, optional main-belt term). It mirrors enterprise's
``physical_ephem_delay`` math but, because the delay is *linear* in the global
correction coefficients ``c`` and the partials are fixed, the entire per-pulsar
design matrix ``G_alpha`` is precomputed once at build time as a static
``jnp.ndarray``. The returned delay is then simply ``G_alpha @ assemble_c(params)``
-- exactly differentiable and negligible cost.

The correction coefficients are **global/common across all pulsars** (the same
parameter names appear in every ``PulsarLikelihood``, like ``crn_*``); only
``G_alpha`` differs per pulsar (its TOAs and sky position).

Parameterization (all uniform priors on [-1, 1]): the artifact's physical prior
half-widths are folded into ``G_alpha``, so each sampled coefficient is a
dimensionless fraction of its prior range. Physical perturbation = width x
coefficient. This keeps every BayesEphem parameter uniform (Discovery
convention) while reproducing the artifact's inter-ephemeris prior envelope.

Artifact produced by the ``bayesephem2`` pipeline (see
BayesEphem2.0/src/bayesephem2). Consumed identically here and by the enterprise
backend.
"""

import json
import os

import numpy as np
import jax.numpy as jnp

from . import const

# Default shipped artifact (multi-ephemeris: DE440 ref, INPOP21a, EPM2021).
DEFAULT_PARTIALS = "/fred/oz002/dreardon/BayesEphem2.0/src/BayesEphem2.0/bayesephem2_artifact.npz"

# planetssb body-index order (matches enterprise / libstempo).
_PLANET_IDX = {"jupiter": 4, "saturn": 5, "uranus": 6, "neptune": 7}

# Outer-planet / Sun mass ratios (m_planet / M_sun), matching
# enterprise.signals.utils.physical_ephem_delay.
_MASS_RATIO = {
    "jupiter": 0.0009547918983127075,
    "saturn": 0.00028588567008942334,
}

# Frame-rate reference offset: MJD 2010/01/01 (enterprise ss_framerotate).
_FRAME_T0 = 55197.0

# Ecliptic-frame rotation generators d/dtheta of the Euler rotations used by
# enterprise.euler_vec (about ecliptic x, y, z).
_GEN = {
    "x": np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]], float),
    "y": np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]], float),
    "z": np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]], float),
}

# Solar GM in light-seconds^3 / s^2 (for the crude main-belt model).
_MU_SUN_LS = const.GMsun / const.c**3
_AU_LS = const.AU / const.c
_MJD_J2000 = 51544.5


def _ecl2eq(x):
    """(...,3) ecliptic -> equatorial (ICRF), same matrix as enterprise."""
    return np.einsum("jk,...k->...j", const.M_ecl, x)


def _eq2ecl(x):
    return np.einsum("kj,...k->...j", const.M_ecl, x)


def _load_artifact(path):
    npz = dict(np.load(path, allow_pickle=False))
    prior_block = json.loads(str(npz["prior_block_json"]))
    return npz, prior_block


def _interp_partial(grid_mjd, Q_planet, toas_mjd):
    """Interpolate orthonormal partials (6,N,3) onto TOAs -> (6, ntoa, 3)."""
    n_mode = Q_planet.shape[0]
    out = np.zeros((n_mode, toas_mjd.size, 3))
    for n in range(n_mode):
        for k in range(3):
            out[n, :, k] = np.interp(toas_mjd, grid_mjd, Q_planet[n, :, k])
    return out


def physical_ephem_design_matrix(psr, partials_file=DEFAULT_PARTIALS,
                                 inc_saturn=True, frame_drift_3axis=True,
                                 inc_mainbelt=False):
    """Build the static per-pulsar design matrix and the parameter spec.

    Returns
    -------
    G : (ntoa, ncol) ndarray
        Design matrix with the artifact's prior half-widths folded in, so the
        sampled coefficients are uniform on [-1, 1]. ``delay = G @ c``.
    params : list[str]
        Global (common) parameter names, with vector parameters tagged
        ``name(size)``, in the column order of ``G``.
    """
    npz, prior_block = _load_artifact(partials_file)
    grid = npz["grid_mjd"]
    toas_mjd = np.asarray(psr.toas) / 86400.0
    pos = np.asarray(psr.pos_t)                      # (ntoa, 3) unit vectors
    ntoa = toas_mjd.size

    cols, params = [], []

    def add_block(vectors, scales, name, size):
        """vectors: (size, ntoa, 3) per-unit Earth-SSB shifts; scales: (size,)."""
        block = np.empty((ntoa, size))
        for i in range(size):
            dearth = scales[i] * vectors[i]          # (ntoa, 3) light-seconds
            block[:, i] = -np.einsum("ij,ij->i", dearth, pos)   # -dearth . phat
        cols.append(block)
        params.append(f"{name}({size})" if size > 1 else name)

    # --- orbit blocks (Jupiter, then optionally Saturn) -----------------
    for planet, flag in [("jupiter", True), ("saturn", inc_saturn)]:
        if not flag:
            continue
        Q = _interp_partial(grid, npz[f"{planet}_Q"], toas_mjd)   # (6, ntoa, 3)
        widths = npz[f"{planet}_orbit_widths"]                    # (6,)
        scales = _MASS_RATIO[planet] * widths
        add_block(Q, scales, f"bayesephem_{planet}_orbit", 6)

    # --- outer-planet masses -------------------------------------------
    mass_w = prior_block["mass_uniform_width"]
    mass_order = ["d_jupiter_mass", "d_saturn_mass", "d_uranus_mass", "d_neptune_mass"]
    body = {"d_jupiter_mass": "jupiter", "d_saturn_mass": "saturn",
            "d_uranus_mass": "uranus", "d_neptune_mass": "neptune"}
    mvecs = np.stack([psr.planetssb[:, _PLANET_IDX[body[m]], :3] for m in mass_order])
    add_block(mvecs, np.array([mass_w[m] for m in mass_order]),
              "bayesephem_mass", 4)

    # --- frame rotation rate (3-axis or z-only) ------------------------
    earth = np.asarray(psr.planetssb[:, 2, :3])      # (ntoa, 3) light-seconds
    earth_ecl = _eq2ecl(earth)
    yrfrac = (toas_mjd - _FRAME_T0) / 365.25
    axes = ["x", "y", "z"] if frame_drift_3axis else ["z"]
    fwidth = np.atleast_1d(np.asarray(prior_block["frame_rate_width"], float))
    if fwidth.size != len(axes):
        fwidth = np.full(len(axes), fwidth.flat[0])
    fvecs = np.stack([
        _ecl2eq(yrfrac[:, None] * np.einsum("jk,ik->ij", _GEN[a], earth_ecl))
        for a in axes
    ])
    add_block(fvecs, fwidth, "bayesephem_frame_rate", len(axes))

    # --- optional main-belt total-mass term ----------------------------
    if inc_mainbelt:
        bw = prior_block.get("mainbelt_width")
        if bw is None or not np.isfinite(bw):
            raise ValueError("inc_mainbelt=True but artifact has no mainbelt_width")
        belt = _mainbelt_vector(toas_mjd)            # (ntoa, 3) light-seconds
        add_block(belt[None], np.array([bw]), "bayesephem_mainbelt", 1)

    G = np.concatenate(cols, axis=1)
    return G, params


def _mainbelt_vector(toas_mjd):
    """Crude single-body circular-orbit model of the main-belt centre of mass.

    NOTE: this is a deliberately simple placeholder (a point mass on a fixed
    circular ecliptic orbit at R = 2.77 AU) so the main-belt mass parameter can
    be exercised and its effect on GW posteriors assessed (per the brief). It is
    NOT a faithful belt model and should be refined (e.g. Ceres/Vesta/Pallas
    individually, or a precomputed belt basis in the artifact).
    """
    R = 2.77 * _AU_LS
    nmean = np.sqrt(_MU_SUN_LS / R**3)               # rad/s
    phase = nmean * (toas_mjd * 86400.0 - _MJD_J2000 * 86400.0)
    x_ecl = np.stack([R * np.cos(phase), R * np.sin(phase), np.zeros_like(phase)], axis=1)
    return _ecl2eq(x_ecl)


def makedelay_bayesephem(psr, partials_file=DEFAULT_PARTIALS, *, inc_saturn=True,
                         frame_drift_3axis=True, inc_mainbelt=False):
    """Factory for the BayesEphem 2.0 deterministic delay component.

    Returns a callable ``delay(params) -> jnp.ndarray`` (length ntoa) with a
    ``.params`` attribute listing the global/common parameter names. The design
    matrix is static; only the global coefficients ``c`` are sampled.
    """
    G_np, names = physical_ephem_design_matrix(
        psr, partials_file, inc_saturn=inc_saturn,
        frame_drift_3axis=frame_drift_3axis, inc_mainbelt=inc_mainbelt)
    G = jnp.asarray(G_np)

    def assemble_c(params):
        # Gather the named global coefficients (vectors stay vectors) in column
        # order and flatten to the c-vector.
        parts = [jnp.atleast_1d(params[name]) for name in names]
        return jnp.concatenate(parts)

    def delay_bayesephem(params):
        return G @ assemble_c(params)

    delay_bayesephem.params = names
    return delay_bayesephem


def bayesephem_priordict():
    """Uniform [-1, 1] priors for every BayesEphem 2.0 coefficient.

    Widths are folded into the design matrix, so the sampled coefficients are
    dimensionless fractions of their (inter-ephemeris-derived) prior ranges.
    """
    return {
        "bayesephem_jupiter_orbit": [-1.0, 1.0],
        "bayesephem_saturn_orbit": [-1.0, 1.0],
        "bayesephem_mass": [-1.0, 1.0],
        "bayesephem_frame_rate": [-1.0, 1.0],
        "bayesephem_mainbelt": [-1.0, 1.0],
    }
