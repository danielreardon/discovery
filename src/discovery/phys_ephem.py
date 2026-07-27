"""Deterministic physical-ephemeris (PEBBLE) signal for Discovery.

This is the JAX backend of the broadened, multi-ephemeris PEBBLE model
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
coefficient. This keeps every PEBBLE parameter uniform (Discovery
convention) while reproducing the artifact's inter-ephemeris prior envelope.

Artifact produced by the ``PEBBLE`` pipeline (see
PEBBLE/src/pebble). Consumed identically here and by the enterprise
backend.
"""

import json
import os

import numpy as np
import jax.numpy as jnp

from . import const

# Default shipped artifact (multi-ephemeris: DE440 ref, INPOP21a, EPM2021).
# Set PEBBLE_PARTIALS to point at your PEBBLE checkout's src/pebble/pebble.npz;
# read at import time, so export it before importing discovery. Individual calls
# can still override with partials_file=... . The literal below is the historical
# OzSTAR location, kept as the fallback so existing setups keep working -- it
# will not resolve on any other machine, hence the env var.
_FALLBACK_PARTIALS = "/fred/oz002/dreardon/PEBBLE/src/pebble/pebble.npz"
DEFAULT_PARTIALS = os.environ.get("PEBBLE_PARTIALS", _FALLBACK_PARTIALS)

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


def _ecl2eq(x):
    """(...,3) ecliptic -> equatorial (ICRF), same matrix as enterprise."""
    return np.einsum("jk,...k->...j", const.M_ecl, x)


def _eq2ecl(x):
    return np.einsum("kj,...k->...j", const.M_ecl, x)


def _load_artifact(path):
    if not os.path.exists(path):
        hint = ("Set PEBBLE_PARTIALS to your PEBBLE checkout's "
                "src/pebble/pebble.npz, or pass partials_file=... .")
        if path == _FALLBACK_PARTIALS and "PEBBLE_PARTIALS" not in os.environ:
            hint = ("This is the built-in OzSTAR fallback and PEBBLE_PARTIALS is "
                    "unset, so it will not resolve off OzSTAR. " + hint)
        raise FileNotFoundError(f"PEBBLE artifact not found: {path}\n{hint}")

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


def _col_ranges(params):
    """Map each param name to its column index range in the assembled G."""
    ranges, j = {}, 0
    for nm in params:
        sz = int(nm[nm.index("(") + 1:-1]) if nm.endswith(")") else 1
        ranges[nm] = list(range(j, j + sz))
        j += sz
    return ranges


def physical_ephem_design_matrix(psr, partials_file=DEFAULT_PARTIALS,
                                 inc_jupiter=True, inc_saturn=True, inc_masses=True,
                                 frame_drift_3axis=True, inc_frame_drift=True, inc_mainbelt=False,
                                 inc_minorbody=True, orthogonalize_minorbody=True,
                                 inc_jerk=False, mainbelt_prior_scale=1.0,
                                 mass_bodies=("jupiter", "saturn", "uranus", "neptune")):
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
    for planet, flag in [("jupiter", inc_jupiter), ("saturn", inc_saturn)]:
        if not flag:
            continue
        Q = _interp_partial(grid, npz[f"{planet}_Q"], toas_mjd)   # (6, ntoa, 3)
        widths = npz[f"{planet}_orbit_widths"]                    # (6,)
        scales = _MASS_RATIO[planet] * widths
        add_block(Q, scales, f"phys_ephem_{planet}_orbit", 6)

    # --- outer-planet masses -------------------------------------------
    if inc_masses:
        mass_w = prior_block["mass_uniform_width"]
        # Select which outer-planet masses to free. Default all four, but over a
        # short (~6-yr) baseline Uranus/Neptune produce <1 ns of *residual* (their
        # >80-yr arcs are absorbed by F0/F1) and only serve as leakage channels
        # for eta, so they can be dropped via mass_bodies=("jupiter","saturn").
        _all = [("jupiter", "d_jupiter_mass"), ("saturn", "d_saturn_mass"),
                ("uranus", "d_uranus_mass"), ("neptune", "d_neptune_mass")]
        sel = [(b, m) for (b, m) in _all if b in mass_bodies]
        mvecs = np.stack([psr.planetssb[:, _PLANET_IDX[b], :3] for (b, m) in sel])
        add_block(mvecs, np.array([mass_w[m] for (b, m) in sel]),
                  "phys_ephem_mass", len(sel))

    # --- frame rotation rate (3-axis, z-only, or off) ------------------
    if inc_frame_drift:
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
        add_block(fvecs, fwidth, "phys_ephem_frame_rate", len(axes))

    # --- optional main-belt asteroid block (Ceres/Pallas/Vesta) --------
    # Mass perturbations d_mu_a = d(m_a / M_sun); basis = each asteroid's real
    # barycentric trajectory (from EPM2021) tabulated in the artifact. Periods
    # 3.6-4.6 yr sit inside the MPTA band; priors come from the inter-ephemeris
    # GM spread (Pallas dominates).
    if inc_mainbelt:
        if "mainbelt_basis" not in npz:
            raise ValueError("inc_mainbelt=True but artifact has no main-belt block")
        basis = _interp_partial(grid, npz["mainbelt_basis"], toas_mjd)  # (n_ast, ntoa, 3)
        # mainbelt_prior_scale multiplies every mode's physical half-width by a
        # common factor (coefficients stay uniform on [-1, 1], so the SVD's
        # relative mode scaling is preserved). Use >1 to test whether a mode that
        # leans on the fiducial prior edge localises at a finite amplitude when
        # allowed more room, or simply tracks the boundary (= unconstrained
        # direction, prior doing the work).
        widths = np.asarray(npz["mainbelt_widths"], float) * float(mainbelt_prior_scale)
        add_block(basis, widths, "phys_ephem_mainbelt", basis.shape[0])

    # --- minor-body (TNO-dominated) barycentre normalisation eta -------
    # The Kuiper belt is a quasi-symmetric ring centred on the SSB: it has no
    # net dipole, so its mass enters only the *denominator* M_tot of
    # R_SSB = sum(m_i r_i)/sum(m_i). Changing the adopted total minor-body mass
    # therefore rescales the entire (Jupiter-dominated) Sun->SSB wobble by
    # eta = -dM/M_tot, coherently: dx_Earth->SSB(t) = eta * r_Sun->SSB(t)|_ref.
    # This is the physical origin of the otherwise "effective, unphysical
    # Jupiter-mass" term seen when bridging ephemerides; its prior is the
    # inter-ephemeris TNO+asteroid total-mass spread (~1-1.5e-7), NOT the
    # Juno-tight GM_Jupiter prior. Degenerate with d_jupiter_mass at the ~85%
    # level (they differ only by the Saturn-and-beyond fraction of the wobble);
    # the correct, very different priors break the degeneracy.
    if inc_minorbody:
        # r_Sun->SSB = R_SSB - r_Sun = -(Sun relative to SSB) = -sunssb.
        r_sun_ssb = -np.asarray(psr.sunssb)[:, :3]      # (ntoa, 3) light-seconds
        eta_w = float(prior_block.get("minorbody_width", 1.5e-7))
        add_block(r_sun_ssb[None], np.array([eta_w]), "phys_ephem_minorbody", 1)

    # --- free-direction SSB jerk (3-axis) ------------------------------
    # A distant unmodelled mass (e.g. Planet Nine) or any common dipolar
    # perturber: the *constant* SSB acceleration it produces is degenerate with
    # per-pulsar F1 (spin-down) and unobservable, but the *jerk* (cubic in time)
    # survives an F0+F1 fit. dx_Earth->SSB(t) = (1/6) jerk * (t-tref)^3 (free 3D
    # direction); delay = -(1/6) jerk.p_hat (t-tref)^3. Generalises the
    # fixed-direction outer-planet mass dipoles to a free direction; the fitted
    # vector maps to (mass, distance) of a perturber via |jerk| ~ m r^-3.5.
    if inc_jerk:
        tsec = np.asarray(psr.toas)
        tc3 = ((tsec - tsec.mean()) ** 3) / 6.0          # (ntoa,), seconds^3
        jvecs = np.zeros((3, ntoa, 3))
        for a in range(3):
            jvecs[a, :, a] = tc3                          # dx_Earth->SSB along axis a
        jw = float(prior_block.get("jerk_width", 1.0e-31))
        add_block(jvecs, np.array([jw, jw, jw]), "phys_ephem_ssb_jerk", 3)

    G = np.concatenate(cols, axis=1)

    # Orthogonalise the planet mass/orbit columns against eta so eta carries the
    # full (Jupiter-dominated) normalisation wobble and the planet terms only
    # span its orthogonal complement -- removing the ~85% eta<->Jupiter
    # degeneracy at the design-matrix level (cleaner sampling; eta is "pinned"
    # to the normalisation, not double-counted by Jupiter mass/orbit).
    if inc_minorbody and orthogonalize_minorbody:
        rng = _col_ranges(params)            # keys carry the (size) suffix
        base = {nm.split("(")[0]: cols for nm, cols in rng.items()}
        e = G[:, base["phys_ephem_minorbody"][0]]
        ee = float(e @ e)
        if ee > 0:
            for nm in ("phys_ephem_jupiter_orbit", "phys_ephem_saturn_orbit",
                       "phys_ephem_mass"):
                for j in base.get(nm, []):
                    G[:, j] -= (G[:, j] @ e / ee) * e
    return G, params


def makedelay_phys_ephem(psr, partials_file=DEFAULT_PARTIALS, *, inc_jupiter=True,
                         inc_saturn=True, inc_masses=True, frame_drift_3axis=True,
                         inc_frame_drift=True, inc_mainbelt=False, inc_minorbody=True,
                         orthogonalize_minorbody=True, inc_jerk=False,
                         mainbelt_prior_scale=1.0,
                         mass_bodies=("jupiter", "saturn", "uranus", "neptune")):
    """Factory for the PEBBLE deterministic delay component.

    Block toggles (``inc_jupiter``, ``inc_saturn``, ``inc_masses``,
    ``frame_drift_3axis``, ``inc_mainbelt``) select which perturbation blocks
    enter the model, e.g. frame + main belt only with everything else off.

    Returns a callable ``delay(params) -> jnp.ndarray`` (length ntoa) with a
    ``.params`` attribute listing the global/common parameter names. The design
    matrix is static; only the global coefficients ``c`` are sampled.
    """
    G_np, names = physical_ephem_design_matrix(
        psr, partials_file, inc_jupiter=inc_jupiter, inc_saturn=inc_saturn,
        inc_masses=inc_masses, frame_drift_3axis=frame_drift_3axis,
        inc_frame_drift=inc_frame_drift,
        inc_mainbelt=inc_mainbelt, inc_minorbody=inc_minorbody,
        orthogonalize_minorbody=orthogonalize_minorbody, inc_jerk=inc_jerk,
        mainbelt_prior_scale=mainbelt_prior_scale, mass_bodies=mass_bodies)
    G = jnp.asarray(G_np)

    def assemble_c(params):
        # Gather the named global coefficients (vectors stay vectors) in column
        # order and flatten to the c-vector.
        parts = [jnp.atleast_1d(params[name]) for name in names]
        return jnp.concatenate(parts)

    def delay_phys_ephem(params):
        return G @ assemble_c(params)

    delay_phys_ephem.params = names
    return delay_phys_ephem


def phys_ephem_priordict():
    """Uniform [-1, 1] priors for every PEBBLE coefficient.

    Widths are folded into the design matrix, so the sampled coefficients are
    dimensionless fractions of their (inter-ephemeris-derived) prior ranges.
    """
    return {
        "phys_ephem_jupiter_orbit": [-1.0, 1.0],
        "phys_ephem_saturn_orbit": [-1.0, 1.0],
        "phys_ephem_mass": [-1.0, 1.0],
        "phys_ephem_frame_rate": [-1.0, 1.0],
        "phys_ephem_mainbelt": [-1.0, 1.0],
        "phys_ephem_minorbody": [-1.0, 1.0],
        "phys_ephem_ssb_jerk": [-1.0, 1.0],
    }
