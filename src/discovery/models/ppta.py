"""PPTA-DR4 single-pulsar noise model.

Built on :mod:`discovery.models.mpta`, with PPTA-specific per-backend group
noise, band noise, deterministic chromatic events and ECORR architecture.

Entry point: :func:`single_pulsar_noise`. Per-pulsar content is driven by
:data:`PPTA_CONFIG`.

Default model:

* FFT-covariance power-law GPs for the per-pulsar background, red noise, DM and
  chromatic noise at a maximum cadence of 1/30 days, with a free chromatic index
* global two-mode correlated-Legendre ECORR plus per-backend ECORR, on top of
  per-backend EFAC and TempoNest EQUAD
* quasi-periodic time-domain solar-wind DM GP for pulsars within
  :data:`SW_ELAT_MAX` degrees of the ecliptic
* group noise, band noise and chromatic events for the pulsars listed in
  :data:`PPTA_CONFIG`

The mean solar-wind density (``n_earth``) is off by default; it is expected to
be fitted in the par file and marginalised with the timing model.
"""

import numpy as np
import re

from .. import signals
from .. import prior
from .. import solar
from .. import likelihood
from .. import deterministic
from .. import matrix
from .. import const
from discovery.models import mpta

jnp = matrix.jnp


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

PPTA_CONFIG = {

    # Prior ranges, registered by update_priordict_standard_ppta(). Per-pulsar
    # event epochs are registered at model-build time by make_psr_delays().
    # The white-noise key is '(.*_)?tnequad', not '(.*_)?log10_tnequad':
    # priordict_standard is scanned in insertion order with re.match and the
    # first hit wins, so the stock '(.*_)?tnequad' would shadow the longer key.
    "priors": {
        '(.*_)?efac': [0.1, 5.0],
        '(.*_)?tnequad': [-10, -5],
        '(.*_)?log10_ecorr': [-10, -5],
        '(.*_)?bkgrnd_log10_A': [-18, -11],
        '(.*_)?red_noise_log10_A': [-18, -11],
        '(.*_)?red_noise_gamma': [0, 7],
        '(.*_)?red_noise2_log10_A': [-18, -11],
        '(.*_)?red_noise2_gamma': [0, 7],
        '(.*_)?dm_gp_log10_A': [-18, -11],
        '(.*_)?dm_gp_gamma': [0, 7],
        '(.*_)?chrom_gp_log10_A': [-18, -11],
        '(.*_)?chrom_gp_gamma': [0, 7],
        '(.*_)?chrom_gp_alpha': [2.5, 14],
        # band GP centre and bandwidth are bounded per-pulsar from psr.freqs by
        # mpta._set_band_priors at model-build time.
        '(.*_)?band_gp_log10_A': [-18, -11],
        '(.*_)?band_gp_gamma': [0, 7],
        '(.*_)?bandalpha_gp_log10_A': [-18, -11],
        '(.*_)?bandalpha_gp_gamma': [0, 7],
        '(.*_)?bandalpha_gp_alpha': [0, 10],
        r'(.*_)?group_noise_.*_log10_A': [-18, -11],
        r'(.*_)?group_noise_.*_gamma': [0, 7],
        # solar wind, quasi-periodic time-domain kernel. sigma is the rms
        # electron-density variability (cm^-3); ell and p are in days.
        '(.*_)?sw_gp_log10_sigma': [-2, 1.3],
        '(.*_)?sw_gp_log10_ell': [1, 4],
        '(.*_)?sw_gp_log10_Gamma': [-3, 2],
        '(.*_)?sw_gp_log10_p': [1.5, 4.5],
        # Fourier power-law solar-wind GP, used when sw_powerlaw=True.
        '(.*_)?sw_gp_log10_A': [-10, 1],
        '(.*_)?sw_gp_gamma': [-4, 4],
        '(.*_)?n_earth': [0, 20],
        # deterministic chromatic events. Event epochs t0 are per-pulsar.
        r'(.*_)?chrom_exp_\d+_log10_Amp': [-10, -2],
        r'(.*_)?chrom_exp_\d+_log10_tau': [0, 4],
        r'(.*_)?chrom_exp_\d+_sign_param': [-1, 1],
        r'(.*_)?chrom_exp_\d+_alpha': [0, 7],
        '(.*_)?chrom_1yr_log10_Amp': [-10, -2],
        '(.*_)?chrom_1yr_phase': [0, 2 * np.pi],
        '(.*_)?chrom_1yr_alpha': [0, 7],
        '(.*_)?chrom_gauss_log10_Amp': [-10, -2],
        '(.*_)?chrom_gauss_log10_sigma': [0, 3],
        '(.*_)?chrom_gauss_sign_param': [-1, 1],
        '(.*_)?chrom_gauss_alpha': [0, 7],
        '(.*_)?gauss_20cm_log10_Amp': [-10, -2],
        '(.*_)?gauss_20cm_log10_sigma': [0, 3],
        '(.*_)?gauss_20cm_t0': [57385, 57785],
        'curn_log10_A': [-18, -11],
        'curn_gamma': [0, 7],
    },

    # Which pulsars carry each optional component.
    "models_dict": {

        # Band noise: one GP per pulsar, centre and bandwidth sampled.
        "band_noise": ['J0437-4715', 'J0613-0200', 'J1017-7156'],

        # Exponential dips. 't0' is the epoch prior in MJD; 'alpha' the
        # chromatic index prior. An event is included only if its epoch window
        # overlaps the data.
        "chrom_exp": {
            'J1713+0747': [{'t0': (54650.0, 54850.0), 'alpha': (1.0, 3.0)},
                           {'t0': (57400.0, 57600.0), 'alpha': (0.0, 2.0)}],
            'J0437-4715': [{'t0': (57000.0, 57200.0), 'alpha': (-1.0, 2.0)}],
            'J1643-1224': [{'t0': (57000.0, 57200.0), 'alpha': (-2.0, 0.0)}],
            'J2145-0750': [{'t0': (56250.0, 56450.0), 'alpha': (-2.0, 2.0)}],
        },

        # Annual chromatic sinusoid.
        "chrom_annual": ['J0613-0200'],

        # Chromatic Gaussian event, with per-pulsar epoch and width priors.
        "chrom_gauss": {
            'J1603-7202': {'t0': (53800.0, 54000.0), 'log10_sigma': (0.0, 3.0)},
        },

        # Gaussian event confined to a 20 cm top-hat in observing frequency.
        "chrom_gauss_20cm": ['J1600-3053'],

        # Groups given their own piecewise frequency-dependent delay basis, in
        # addition to the global one. Pulsars absent from this dict get the
        # global basis only when fd=True.
        "fd_groups": {},
    },

    # Per-backend group noise.
    "group_dict": {
        'J0437-4715': ['CASPSR_40CM', 'PDFB_20CM', 'UWL_PDFB4_20CM', 'UWL_sbD', 'UWL_sbE'],
        'J0613-0200': ['UWL_sbA', 'UWL_sbD', 'UWL_sbF', 'UWL_sbH'],
        'J1017-7156': ['CASPSR_40CM', 'UWL_sbC', 'UWL_sbD', 'UWL_sbH'],
        'J1022+1001': ['CPSR2_50CM', 'PDFB_10CM', 'UWL_sbF', 'UWL_sbH'],
        'J1045-4509': ['UWL_sbF'],
        'J1125-6014': ['UWL_sbE'],
        'J1603-7202': ['CPSR2_50CM', 'UWL_sbF'],
        'J1643-1224': ['PDFB_40CM', 'UWL_sbB', 'UWL_sbD', 'UWL_sbF'],
        'J1713+0747': ['CPSR2_50CM', 'PDFB1_early_10CM', 'WBCORR_10CM'],
        'J1730-2304': ['UWL_sbD'],
        'J1744-1134': ['CASPSR_40CM', 'PDFB_20CM', 'UWL_sbA', 'UWL_sbC', 'UWL_sbD'],
        'J1909-3744': ['CASPSR_40CM', 'UWL_PDFB4_10CM', 'UWL_sbA'],
        'J1939+2134': ['CPSR2_50CM', 'UWL_sbA', 'UWL_sbD', 'UWL_sbH'],
        'J2129-5721': ['UWL_sbG'],
        'J2145-0750': ['CASPSR_40CM', 'CPSR2_50CM', 'PDFB_20CM', 'UWL_sbA'],
        'J2241-5236': ['UWL_sbF', 'UWL_sbH'],
    },

    # Backends given their own ECORR, in addition to the global term.
    "ecorr_dict": {
        'J0437-4715': ['PDFB_10CM', 'PDFB_20CM', 'PDFB_40CM', 'UWL_sbC',
                       'UWL_sbD', 'UWL_sbE', 'UWL_sbF', 'UWL_sbG'],
        'J1022+1001': ['UWL_sbH'],
        'J1603-7202': ['CPSR2_50CM'],
        'J1744-1134': ['CPSR2_50CM'],
        'J1824-2452A': ['UWL_sbB'],
        'J1909-3744': ['CASPSR_40CM', 'UWL_PDFB4_10CM', 'UWL_sbA'],
        'J2051-0827': ['UWL_sbE'],
        'J2129-5721': ['PDFB1_20CM'],
        'J2241-5236': ['UWL_sbB'],
    },
}

# Obliquity of the ecliptic (deg), J2000.
_OBLIQUITY_DEG = 23.4392911

# Maximum |ecliptic latitude| (deg) for which the solar-wind GP is included.
SW_ELAT_MAX = 41.0

# Epoch prior (MJD) for the 20 cm Gaussian bump.
GAUSS_20CM_T0 = (57385.0, 57785.0)

# Group-noise components are capped at ntoa_group // GROUP_TOA_PER_MODE.
GROUP_TOA_PER_MODE = 4


# ---------------------------------------------------------------------------
# priors
# ---------------------------------------------------------------------------

def register_priors(updates):
    """Insert ``updates`` into ``prior.priordict_standard`` ahead of existing keys.

    ``prior.getprior_uniform`` scans the dict in insertion order and returns the
    first ``re.match``, so entries must precede any more general pattern that
    also matches them.
    """
    merged = {**updates,
              **{k: v for k, v in prior.priordict_standard.items()
                 if k not in updates}}
    prior.priordict_standard.clear()
    prior.priordict_standard.update(merged)


def update_priordict_standard_ppta(config=PPTA_CONFIG):
    """Register the PPTA prior ranges."""
    register_priors(config["priors"])

    return


update_priordict_standard_ppta()


# ---------------------------------------------------------------------------
# selections and helpers
# ---------------------------------------------------------------------------

def ecliptic_latitude(psr):
    """Ecliptic latitude in degrees, from the equatorial unit vector ``psr.pos``."""
    x, y, z = np.asarray(psr.pos, dtype=np.float64)
    eps = np.radians(_OBLIQUITY_DEG)

    return float(np.degrees(np.arcsin(np.clip(z * np.cos(eps) - y * np.sin(eps),
                                              -1.0, 1.0))))


def use_solar_wind(psr, elat_max=SW_ELAT_MAX):
    """Whether the solar-wind GP is included for this pulsar."""
    return abs(ecliptic_latitude(psr)) <= elat_max


def group_masks(psr, groups):
    """``{group name: boolean mask}`` for backend-flag groups, matched exactly.

    Groups selecting no TOAs are dropped with a warning.
    """
    flags = np.asarray(psr.backend_flags)

    masks = {}
    for g in sorted(set(groups)):
        m = np.asarray(flags == g)
        if m.any():
            masks[g] = m
        else:
            print(f'Warning: group {g!r} selects no TOAs for {psr.name}; dropped.')

    return masks


def _make_selection(labels):
    """Wrap a per-TOA label array as a selection function."""
    labels = np.asarray(labels, dtype=str)

    def selection(psr):
        return labels

    return selection


def _make_mask_selection(mask):
    """Wrap a boolean mask as a selection function."""
    mask = np.asarray(mask, dtype=bool)

    def selection(psr):
        return mask

    return selection


def selection_global(psr):
    """Single-group selection placing every TOA in one block."""
    return np.full(len(psr.toas), 'global', dtype='<U6')


def masked_fourierbasis(selection, base=signals.fourierbasis):
    """Fourier basis restricted to ``selection``, with rows outside it zeroed.

    When ``T`` is not supplied the frequency grid uses the span of the selected
    TOAs.
    """
    def basis(psr, components, T=None):
        m = np.asarray(selection(psr), dtype=bool)
        if T is None:
            sel = psr.toas[m]
            T = float(sel.max() - sel.min())
        f, df, fmat = base(psr, components, T)

        return f, df, fmat * m[:, None]

    return basis

def masked_timeinterpbasis(selection, start_time=None, order=1):
    """Time-interpolation basis restricted to ``selection``."""

    base = signals.make_timeinterpbasis(start_time=start_time, order=order)

    def basis(psr, components, T=None):
        t_coarse, dt_coarse, Bmat = base(psr, components, T)
        m = np.asarray(selection(psr), dtype=bool)

        return t_coarse, dt_coarse, Bmat * m[:, None]

    return basis


def chromatic_gaussian_20cm(psr, nu1=1000.0, nu2=2000.0):
    """Gaussian event confined to observing frequencies in ``[nu1, nu2)`` MHz.

    Returns a delay with signature ``(t0, log10_Amp, log10_sigma)``, where
    ``t0`` is in MJD and ``log10_sigma`` in log10 days.
    """
    toas = matrix.jnparray(psr.toas / const.day)
    freqs = matrix.jnparray(psr.freqs)
    tophat = ((freqs >= nu1) & (freqs < nu2)).astype(jnp.float64)

    def delay(t0, log10_Amp, log10_sigma):
        return 10**log10_Amp * jnp.exp(
            -(toas - t0)**2 / (2 * (10**log10_sigma)**2)) * tophat

    return delay


# ---------------------------------------------------------------------------
# ECORR
# ---------------------------------------------------------------------------

def makegp_ecorr_ppta(psr, noisedict={}, nmodes=2, correlated=True,
                      per_backend=True, ecorr_dict=None):
    """Global Legendre ECORR block plus per-backend white ECORR.

    Parameters
    ----------
    nmodes : int or None
        Legendre modes in the global block. ``None`` omits the global term.
    correlated : bool
        Use the full mode-covariance variant of the Legendre ECORR.
    per_backend : bool
        Add ordinary white ECORR. Restricted to the backends listed in
        ``ecorr_dict`` for this pulsar when one is supplied, otherwise applied
        to every backend.
    ecorr_dict : dict, optional
        ``{pulsar name: [backend flags]}``.
    """
    gps = []

    if nmodes is not None:
        maker = (signals.makegp_ecorr_legendre_correlated if correlated
                 else signals.makegp_ecorr_legendre)
        gps += [maker(psr, noisedict=noisedict, nmodes=nmodes,
                      selection=selection_global, name='ecorrleg_global')]

    if per_backend:
        if ecorr_dict is None:
            gps += [signals.makegp_ecorr(psr, noisedict=noisedict,
                                         name='ecorr_backend')]
        elif psr.name in ecorr_dict:
            masks = group_masks(psr, ecorr_dict[psr.name])
            if masks:
                labels = np.full(len(psr.toas), '', dtype=object)
                for b, m in masks.items():
                    labels = np.where(m, b, labels)
                gps += [signals.makegp_ecorr(
                    psr, noisedict=noisedict,
                    selection=_make_selection(labels), name='ecorr_backend')]

    return gps


# ---------------------------------------------------------------------------
# group noise and solar wind
# ---------------------------------------------------------------------------

def makegp_group_noise_four(psr, group_dict=None, max_cadence_days=30.0, Tspan=None,
                       group_tspan='backend'):
    """Per-backend-group achromatic power-law GPs.

    Parameters
    ----------
    group_dict : dict, optional
        ``{pulsar name: [backend flags]}``. Defaults to the module config.
    group_tspan : {'backend', 'psr'}
        Frequency-grid span: the group's own TOAs, or the full pulsar span.
    """
    group_dict = (PPTA_CONFIG["group_dict"] if group_dict is None
                  else group_dict)
    groups = group_dict.get(psr.name, [])
    if not groups:
        return []

    psr_Tspan = signals.getspan(psr) if Tspan is None else Tspan
    use_psr_span = (group_tspan == 'psr')

    gps = []
    for g, mask in group_masks(psr, groups).items():
        ntoa = int(mask.sum())
        T = psr_Tspan if use_psr_span else float(psr.toas[mask].max()
                                                 - psr.toas[mask].min())
        ncomp = min(int(T / (max_cadence_days * 86400.0)),
                    max(1, ntoa // GROUP_TOA_PER_MODE))
        if ncomp < 1:
            print(f'Warning: group {g!r} for {psr.name} has {ntoa} TOAs over '
                  f'{T / 86400.0:.0f} d; skipped.')
            continue

        gps += [signals.makegp_fourier(
            psr, signals.powerlaw, components=ncomp,
            T=(psr_Tspan if use_psr_span else None),
            fourierbasis=masked_fourierbasis(_make_mask_selection(mask)),
            name=f'group_noise_{g}')]

    return gps


def makegp_group_noise_fftcov(psr, group_dict=None, max_cadence_days=30.0,
        Tspan=None, group_tspan="backend", order=1, name="group_noise"):
    """Per-backend-group FFT-covariance GPs.

    Parameters
    ----------
    group_dict : dict, optional {pulsar name: [backend flags]}.
    group_tspan : {'backend', 'psr'} Basis span determined from the backend TOAs or the full pulsar span.
    """

    group_dict = (PPTA_CONFIG["group_dict"] if group_dict is None else group_dict)

    groups = group_dict.get(psr.name, [])

    if not groups:
        return []

    psr_Tspan = signals.getspan(psr) if Tspan is None else Tspan
    use_psr_span = (group_tspan == "psr")

    gps = []

    for g, mask in group_masks(psr, groups).items():

        ntoa = int(mask.sum())

        if ntoa < 2:
            continue

        group_toas = psr.toas[mask]
        start_time = float(group_toas.min())

        T = (psr_Tspan if use_psr_span else float(group_toas.max() - group_toas.min()))

        ncomp = min(int(T / (max_cadence_days * 86400.0)), max(1, ntoa // GROUP_TOA_PER_MODE))

        if ncomp < 1:
            print(
                f"Warning: group {g!r} for {psr.name} has "
                f"{ntoa} TOAs over {T/86400.0:.0f} d; skipped."
            )
            continue

        fb = masked_timeinterpbasis(_make_mask_selection(mask), start_time=start_time, order=order)

        gps.append(signals.makegp_fftcov(psr, signals.powerlaw, components=ncomp, T=(psr_Tspan if use_psr_span else T),
                fourierbasis=fb, name=f"{name}_{g}"))

    return gps



def makegp_solar_wind(psr, max_cadence_days=30.0, kernel='qp'):
    """Time-domain solar-wind DM GP.

    Parameters
    ----------
    kernel : {'qp', 'se'}
        Quasi-periodic or squared-exponential covariance.
    """
    covariance = {'qp': signals.quasi_periodic,
                  'se': signals.squared_exponential}[kernel]

    return [solar.makegp_timedomain_solar_dm(
        psr, covariance=covariance, dt=max_cadence_days * 86400.0,
        name='sw_gp')]


# ---------------------------------------------------------------------------
# frequency-dependent delay
# ---------------------------------------------------------------------------

def makegp_fd(psr, nodes=16, spacing='quantile', selection=None, groups=None,
              fd_groups_dict=None):
    """Marginalised piecewise-linear frequency-dependent delay.

    Returns a single GP, or ``None`` if the basis would be empty.

    Parameters
    ----------
    nodes : int
        Number of frequency nodes.
    spacing : {'quantile', 'log'}
        Node placement across the observed band.
    selection : callable, optional
        Maps ``psr`` to per-TOA group labels. Defaults to
        ``signals.selection_backend_flags`` when per-group bases are requested.
    groups : list of str, optional
        Groups given their own basis in addition to the global one. Overrides
        the per-pulsar entry in ``fd_groups_dict``.
    fd_groups_dict : dict, optional
        ``{pulsar name: [backend flags]}``. Defaults to the module config.
    """
    if groups is None:
        fd_groups_dict = (PPTA_CONFIG["models_dict"]["fd_groups"]
                          if fd_groups_dict is None else fd_groups_dict)
        groups = fd_groups_dict.get(psr.name, None)

    if groups and selection is None:
        selection = signals.selection_backend_flags

    return signals.makegp_fd_piecewise(psr, nodes=nodes, spacing=spacing,
                                       selection=selection, groups=groups,
                                       name='fd')


# ---------------------------------------------------------------------------
# deterministic delays
# ---------------------------------------------------------------------------

def make_psr_delays(psr, config=PPTA_CONFIG, mean_sw=False, chrom_exp=None,
                    chrom_annual=None, chrom_gauss=None, chrom_gauss_20cm=None):
    """Deterministic delays for one pulsar, registering per-pulsar epoch priors.

    Each per-pulsar switch defaults to ``None``, meaning "use the configuration
    for this pulsar". Events whose epoch prior does not overlap the data are
    skipped.

    Parameters
    ----------
    mean_sw : bool
        Include the mean solar-wind density ``n_earth`` as a delay. Off by
        default; ``NE_SW`` is normally fitted in the par file instead.
    """
    md = config["models_dict"]

    chrom_exp = (psr.name in md["chrom_exp"]) if chrom_exp is None else chrom_exp
    chrom_annual = ((psr.name in md["chrom_annual"]) if chrom_annual is None
                    else chrom_annual)
    chrom_gauss = ((psr.name in md["chrom_gauss"]) if chrom_gauss is None
                   else chrom_gauss)
    chrom_gauss_20cm = ((psr.name in md["chrom_gauss_20cm"])
                        if chrom_gauss_20cm is None else chrom_gauss_20cm)

    tmin, tmax = psr.toas.min() / 86400.0, psr.toas.max() / 86400.0

    delays, updates = [], {}
    key = re.escape(psr.name)

    def overlaps(t0):
        return tmin < t0[1] and tmax > t0[0]

    if mean_sw:
        delays += [signals.makedelay(psr, solar.make_solardm(psr), name='sw')]

    if chrom_exp and psr.name in md["chrom_exp"]:
        kept = 0
        for i, ev in enumerate(md["chrom_exp"][psr.name], start=1):
            if not overlaps(ev['t0']):
                print(f'{psr.name}: dip {i} epoch prior {ev["t0"]} does not '
                      f'overlap [{tmin:.0f}, {tmax:.0f}]; skipped.')
                continue
            nm = f'chrom_exp_{i}'
            updates[f'{key}_{nm}_t0'] = list(ev['t0'])
            if 'alpha' in ev:
                updates[f'{key}_{nm}_alpha'] = list(ev['alpha'])
            delays += [signals.makedelay(
                psr, deterministic.chromatic_exponential(psr), name=nm)]
            kept += 1
        if kept == 0:
            print(f'{psr.name}: no exponential dips overlap the data.')

    if chrom_annual:
        delays += [signals.makedelay(psr, deterministic.chromatic_annual(psr),
                                     name='chrom_1yr')]

    if chrom_gauss and psr.name in md["chrom_gauss"]:
        ev = md["chrom_gauss"][psr.name]
        if overlaps(ev['t0']):
            updates[f'{key}_chrom_gauss_t0'] = list(ev['t0'])
            if 'log10_sigma' in ev:
                updates[f'{key}_chrom_gauss_log10_sigma'] = list(ev['log10_sigma'])
            delays += [signals.makedelay(
                psr, deterministic.chromatic_gaussian(psr), name='chrom_gauss')]
        else:
            print(f'{psr.name}: Gaussian-event epoch prior {ev["t0"]} does not '
                  f'overlap [{tmin:.0f}, {tmax:.0f}]; skipped.')

    if chrom_gauss_20cm:
        if overlaps(GAUSS_20CM_T0):
            delays += [signals.makedelay(psr, chromatic_gaussian_20cm(psr),
                                         name='gauss_20cm')]
        else:
            print(f'{psr.name}: 20 cm bump epoch prior {GAUSS_20CM_T0} does not '
                  f'overlap [{tmin:.0f}, {tmax:.0f}]; skipped.')

    if updates:
        register_priors(updates)

    return delays

# ---------------------------------------------------------------------------
# stochastic delays
# ---------------------------------------------------------------------------

def make_psr_gps_fourier(psr, max_cadence_days=14, bkgrnd_log10_A=None, Tspan=None, background=True, red=True, dm=True, chrom=True, chrom_alpha=None, chrom_poly=False, sw=True, sw_powerlaw=False, sw_logf=False, band=False, band_alpha=False, fd_gp=None):
    psr_Tspan = signals.getspan(psr) if Tspan is None else Tspan
    psr_components = int(psr_Tspan / (max_cadence_days * 86400))
    mpta._set_band_priors(psr, band=band, band_alpha=band_alpha)

    gp_signals = []

    if background:
        gp_signals.append(signals.makegp_fourier(psr, signals.powerlaw_gwb(log10_A=bkgrnd_log10_A), components=psr_components, T=psr_Tspan, name='bkgrnd'))
    if red:
        gp_signals.append(signals.makegp_fourier(psr, signals.powerlaw, components=psr_components, T=psr_Tspan, name='red_noise'))
    
    if dm:
        gp_signals.append(signals.makegp_fourier(psr, signals.powerlaw, components=psr_components, T=psr_Tspan, fourierbasis=signals.fourierbasis_dm, name='dm_gp'))
    if chrom:
        gp_signals.append(signals.makegp_fourier(psr, signals.powerlaw, components=psr_components, T=psr_Tspan, fourierbasis=signals.fourierbasis_chrom, name='chrom_gp', alpha=chrom_alpha))
    if chrom and chrom_poly:
        gp_signals.append(signals.makegp_chrom_poly_svd(psr, name='chrom_gp', project=fd_gp))
    if sw:
        if not sw_powerlaw:
            gp_signals.append(solar.makegp_timedomain_solar_dm(psr, covariance=signals.squared_exponential, dt=max_cadence_days*86400.0, name='sw_gp'))
        else:
            gp_signals.append(signals.makegp_fourier(psr, signals.powerlaw, components=psr_components, T=psr_Tspan, fourierbasis=solar.make_fourierbasis_solar_dm(logf=sw_logf), name='sw_gp'))
    if band:
        gp_signals.append(signals.makegp_fourier(psr, signals.powerlaw, components=psr_components, T=psr_Tspan, fourierbasis=signals.fourierbasis_band, name='band_gp'))
    if band_alpha:
        gp_signals.append(signals.makegp_fourier(psr, signals.powerlaw, components=psr_components, T=psr_Tspan, fourierbasis=signals.fourierbasis_band_alpha, name='bandalpha_gp')) 

    return gp_signals


def make_psr_gps_fftint(psr, max_cadence_days=14, bkgrnd_log10_A=None, Tspan=None, background=True, red=True, red2=False, dm=True, chrom=True, chrom_alpha=None, chrom_poly=False, sw=True, sw_powerlaw=False, band=False, band_alpha=False, fd_gp=None):
    psr_Tspan = signals.getspan(psr) if Tspan is None else Tspan
    psr_components = int(psr_Tspan / (max_cadence_days * 86400))
    psr_knots = 2 * psr_components + 1
    mpta._set_band_priors(psr, band=band, band_alpha=band_alpha)

    gp_signals = []

    if background:
        gp_signals.append(signals.makegp_fftcov(psr, signals.powerlaw_gwb(log10_A=bkgrnd_log10_A), components=psr_knots, T=psr_Tspan, name='bkgrnd'))
    if red:
        gp_signals.append(signals.makegp_fftcov(psr, signals.powerlaw, components=psr_knots, T=psr_Tspan, name='red_noise'))
    if red2:
        gp_signals.append(signals.makegp_fftcov(psr, signals.powerlaw, components=psr_knots, T=psr_Tspan, name='red_noise2'))
    if dm:
        gp_signals.append(signals.makegp_fftcov_dm(psr, signals.powerlaw, components=psr_knots, T=psr_Tspan, name='dm_gp'))
    if chrom:
        gp_signals.append(signals.makegp_fftcov_chrom(psr, signals.powerlaw, components=psr_knots, T=psr_Tspan, name='chrom_gp', alpha=chrom_alpha))
    if chrom and chrom_poly:
        gp_signals.append(signals.makegp_chrom_poly_svd(psr, name='chrom_gp', project=fd_gp))
    if sw:
        if not sw_powerlaw:
            gp_signals.append(solar.makegp_timedomain_solar_dm(psr, covariance=signals.squared_exponential, dt=max_cadence_days*86400.0, name='sw_gp'))
        else:
            gp_signals.append(signals.makegp_fftcov_solar(psr, signals.powerlaw, components=psr_knots, T=psr_Tspan, name='sw_gp'))
    if band:
        gp_signals.append(signals.makegp_fftcov_band(psr, signals.powerlaw, components=psr_knots, T=psr_Tspan, name='band_gp'))
    if band_alpha:
        gp_signals.append(signals.makegp_fftcov_band_alpha(psr, signals.powerlaw, components=psr_knots, T=psr_Tspan, name='bandalpha_gp'))

    return gp_signals


def single_pulsar_noise(psr, fftint=True, max_cadence_days=14, Tspan=None, noisedict={},
                        ecorr=True, quadratic=False, ecorr_nmodes=None, ecorr_correlated=False, global_ecorr=False, # ecorr options. ecorr_nmodes=N selects an N-mode Legendre ECORR (log-frequency basis; nmodes=1 is standard ECORR); ecorr_correlated=True uses the full-M (correlated-mode) variant that can also model a frequency-asymmetric jitter amplitude
                        background=True, bkgrnd_log10_A=None, red=True, red2=False, dm=True, chrom=True, chrom_alpha=None, chrom_poly=False, sw=True, sw_powerlaw=False, sw_logf=False, # Base model: gwb, red, dm, chromatic, solar wind (sw_powerlaw=True selects the legacy power-law solar-wind GP instead of the time-domain one; sw_logf=True log-spaces its frequencies -- Fourier path only)
                        band=False, band_alpha=False, fd=False, fd_nodes=16, fd_spacing='quantile', fd_selection=None, # Additional GP models (fd=True marginalises an arbitrary time-constant frequency-dependent delay over fd_nodes frequency nodes; fd_selection splits it per TOA group)
                        chrom_annual=False, chrom_exponential=False, chrom_gaussian=False, chrom_sphere=False, chrom_step=False, # Deterministic chromatic models
                        shapiro=False, orbital_dm=False, orbital_dm_fourier=False, extra_gps=None, # Shapiro delay and orbital DM, and extra GPs
                        return_components=False): # Whether to return the list of model components in addition to the likelihood object (useful for adding additional components)
    # Set up per-backend white noise (efac and tnequad)
    measurement_noise = signals.makenoise_measurement(psr, tnequad=True, noisedict=noisedict)
    # Set up model components
    model_components = [psr.residuals]
    model_components += [signals.makegp_timing(psr, svd=True)] # Set up timing model (analytically marginalised)
    model_components += [measurement_noise]
    if ecorr:
        if ecorr_nmodes is not None:
            if ecorr_correlated:
                model_components += [signals.makegp_ecorr_legendre_correlated(psr, noisedict=noisedict, nmodes=ecorr_nmodes)]
            else:
                model_components += [signals.makegp_ecorr_legendre(psr, noisedict=noisedict, nmodes=ecorr_nmodes)]
        elif quadratic:
            model_components += [signals.makegp_quadratic_ecorr_legendre(psr, noisedict=noisedict)]
        else:
            model_components += [signals.makegp_ecorr(psr, noisedict=noisedict)]
    if global_ecorr: # add an additional global ECORR term
        model_components += [signals.makegp_ecorr_simple(psr, noisedict=noisedict)]
    # Add deterministic chromatic components
    if chrom_annual:
        model_components += [signals.makedelay(psr, deterministic.chromatic_annual(psr), name='chrom_1yr')]
    if chrom_exponential:
        mpta._set_chrom_exp_priors(psr, chrom_exponential=True)
        model_components += [signals.makedelay(psr, deterministic.chromatic_exponential(psr), name='chrom_exp')]
    if chrom_gaussian:
        model_components += [signals.makedelay(psr, deterministic.chromatic_gaussian(psr), name='chrom_gauss')]
    if chrom_sphere:
        model_components += [signals.makedelay(psr, deterministic.chromatic_sphere(psr), name='chrom_sphere')]
    if chrom_step:
        model_components += [signals.makedelay(psr, deterministic.chromatic_step(psr), name='chrom_step')]
    
    # Models that require orbital phase information    
    if shapiro or orbital_dm or orbital_dm_fourier:
        if psr.tasc is None or psr.pb is None:
            raise ValueError("Error: You must set psr.tasc and psr.pb to use the deterministic Shapiro delay function")
        print("Warning: Binary phase calculation assumes constant orbital period and CIRCULAR ORBIT. Ensure this is a valid approximation for your pulsar and model choice.")
        binphase = (2 * np.pi / psr.pb) * (psr.toas - psr.tasc)
        if shapiro:
            model_components += [signals.makedelay(psr, deterministic.shapiro_cosi(psr, binphase), name='shapiro')]
        if orbital_dm:
            model_components += [signals.makedelay(psr, deterministic.orbital_DM_gaussian(psr, binphase), name='orbital_dm')]
        if orbital_dm_fourier:
            model_components += [signals.makedelay(psr, deterministic.orbital_DM_fourier(psr, binphase), name='orbital_dm_fourier')]

    # Marginalised time-constant frequency-dependent delay (residual DM, mean
    # scattering, uncorrected profile evolution). Built first so it can be
    # projected out of the chromatic polynomial, whose constant-in-time column
    # would otherwise be degenerate with it.
    # fd_selection (e.g. signals.selection_backend_flags) gives each TOA group its
    # own frequency basis, combined into the one marginalised GP.
    fd_gp = signals.makegp_fd_piecewise(psr, nodes=fd_nodes, spacing=fd_spacing,
                                        selection=fd_selection, name='fd') if fd else None
    if fd_gp is not None:
        model_components += [fd_gp]

    # Add GP components
    if fftint:
        if sw_logf:
            # the fftint solar GP uses a time-interpolation basis, so there is no
            # Fourier frequency grid to log-space; sw_logf needs fftint=False.
            print("Warning: sw_logf=True is ignored with fftint=True (the FFT-covariance "
                  "solar GP uses a time-interpolation basis). Use fftint=False.")
        model_components += make_psr_gps_fftint(psr, max_cadence_days=max_cadence_days, bkgrnd_log10_A=bkgrnd_log10_A, Tspan=Tspan, background=background, red=red, dm=dm, chrom=chrom, chrom_alpha=chrom_alpha, chrom_poly=chrom_poly, sw=sw, sw_powerlaw=sw_powerlaw, band=band, band_alpha=band_alpha, fd_gp=fd_gp)
    else:
        model_components += make_psr_gps_fourier(psr, max_cadence_days=max_cadence_days, bkgrnd_log10_A=bkgrnd_log10_A, Tspan=Tspan, background=background, red=red, dm=dm, chrom=chrom, chrom_alpha=chrom_alpha, chrom_poly=chrom_poly, sw=sw, sw_powerlaw=sw_powerlaw, sw_logf=sw_logf, band=band, band_alpha=band_alpha, fd_gp=fd_gp)


    if extra_gps is not None:
        model_components += extra_gps

    m = likelihood.PulsarLikelihood(model_components)

    if return_components:
        return m, model_components
    
    return m


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def ppta_single_pulsar_noise(psr, fftint=True, max_cadence_days=30, Tspan=None, noisedict={},
                        ecorr=True, ecorr_nmodes=2, ecorr_correlated=True, ecorr_per_backend=True, ecorr_dict=None,
                        background=True, bkgrnd_log10_A=None, red=True, red2=False, dm=True,
                        chrom=True, chrom_alpha=None, chrom_poly=False,
                        fd=False, fd_nodes=16, fd_spacing='quantile', fd_selection=None, fd_groups=None,
                        sw=None, sw_elat_max=SW_ELAT_MAX, sw_kernel='qp', sw_powerlaw=False, sw_logf=False, mean_sw=False,
                        band=None, band_alpha=False,
                        group=None, group_dict=None, group_tspan='backend',
                        chrom_exp=None, chrom_annual=None, chrom_gauss=None, chrom_gauss_20cm=None,
                        chrom_sphere=False, chrom_step=False,
                        config=PPTA_CONFIG, extra_gps=None, return_components=False):
    """Build the PPTA-DR4 single-pulsar noise likelihood.

    The core model comes from :func:`mpta.single_pulsar_noise`; the PPTA ECORR
    stack, group noise, solar-wind GP, frequency-dependent delay and chromatic
    events are added here.

    Per-pulsar switches (``sw``, ``band``, ``group``, ``chrom_exp``,
    ``chrom_annual``, ``chrom_gauss``, ``chrom_gauss_20cm``) default to ``None``,
    meaning "use the configuration for this pulsar". An explicit value overrides
    the configuration.

    Parameters
    ----------
    fftint : bool
        Use FFT-covariance GP bases rather than Fourier.
    max_cadence_days : float
        Sets the component count of each power-law GP as ``int(T / cadence)``.
    ecorr_nmodes : int or None
        Legendre modes in the global ECORR block. ``None`` leaves per-backend
        ECORR only.
    ecorr_dict : dict, optional
        Backends given per-backend ECORR, ``{pulsar: [backends]}``. Defaults to
        the module config; pass ``{}`` to apply ECORR to every backend.
    chrom_poly : bool
        Add the marginalised chromatic polynomial, sharing ``alpha`` with the
        chromatic GP.
    fd : bool
        Add the marginalised piecewise-linear frequency-dependent delay.
    fd_nodes : int
        Number of frequency nodes in the fd basis.
    fd_spacing : {'quantile', 'log'}
        Node placement across the observed band.
    fd_selection : callable, optional
        Per-TOA group labels for per-group fd bases.
    fd_groups : list of str, optional
        Groups given their own fd basis in addition to the global one. Defaults
        to the per-pulsar entry in ``models_dict['fd_groups']``.
    sw : bool, optional
        ``None`` includes the solar-wind GP when
        ``|ecliptic latitude| <= sw_elat_max``.
    sw_kernel : {'qp', 'se'}
        Time-domain solar-wind covariance. Ignored when ``sw_powerlaw=True``.
    mean_sw : bool
        Include ``n_earth`` as a deterministic delay.
    group_tspan : {'backend', 'psr'}
        Frequency-grid span for group noise.
    return_components : bool
        Also return the list of model components.

    Returns
    -------
    discovery.likelihood.PulsarLikelihood
    """
    update_priordict_standard_ppta(config)

    if background and red2:
        print('Warning: background and red2 are both enabled.')

    md = config["models_dict"]
    ecorr_dict = config["ecorr_dict"] if ecorr_dict is None else ecorr_dict
    group_dict = config["group_dict"] if group_dict is None else group_dict

    sw = use_solar_wind(psr, sw_elat_max) if sw is None else sw
    band = (psr.name in md["band_noise"]) if band is None else band
    group = (psr.name in group_dict) if group is None else group

    print(f'{psr.name}: |ecliptic latitude| = '
          f'{abs(ecliptic_latitude(psr)):.1f} deg -> solar wind '
          f'{"ENABLED" if sw else "DISABLED"}')

    ppta_gps = []

    if ecorr:
        ppta_gps += makegp_ecorr_ppta(psr, noisedict=noisedict,
                                      nmodes=ecorr_nmodes,
                                      correlated=ecorr_correlated,
                                      per_backend=ecorr_per_backend,
                                      ecorr_dict=(ecorr_dict or None))

    if sw and not sw_powerlaw:
        ppta_gps += makegp_solar_wind(psr, max_cadence_days=max_cadence_days,
                                      kernel=sw_kernel)

    if group and not fftint:
        ppta_gps += makegp_group_noise_four(psr, group_dict=group_dict,
                                       max_cadence_days=max_cadence_days,
                                       Tspan=Tspan, group_tspan=group_tspan)
    elif group and fftint:
        ppta_gps += makegp_group_noise_fftcov(psr, group_dict=group_dict,
                                       max_cadence_days=max_cadence_days,
                                       Tspan=Tspan, group_tspan=group_tspan)

    # The fd basis is built here so that its groups can be set per pulsar, and
    # is projected out of the chromatic polynomial where both are present.
    fd_gp = None
    if fd:
        fd_gp = makegp_fd(psr, nodes=fd_nodes, spacing=fd_spacing,
                          selection=fd_selection, groups=fd_groups,
                          fd_groups_dict=md.get("fd_groups"))
        if fd_gp is not None:
            ppta_gps += [fd_gp]

    if chrom and chrom_poly:
        ppta_gps += [signals.makegp_chrom_poly_svd(psr, name='chrom_gp',
                                                   project=fd_gp)]

    ppta_gps += make_psr_delays(psr, config=config, mean_sw=mean_sw,
                                chrom_exp=chrom_exp,
                                chrom_annual=chrom_annual,
                                chrom_gauss=chrom_gauss,
                                chrom_gauss_20cm=chrom_gauss_20cm)

    if extra_gps is not None:
        ppta_gps += extra_gps

    model, components = single_pulsar_noise(
        psr, fftint=fftint, max_cadence_days=max_cadence_days, Tspan=Tspan,
        noisedict=noisedict,
        ecorr=False, global_ecorr=False,
        background=background, bkgrnd_log10_A=bkgrnd_log10_A,
        red=red, red2=red2, dm=dm,
        chrom=chrom, chrom_alpha=chrom_alpha, chrom_poly=False,
        sw=(sw and sw_powerlaw), sw_powerlaw=sw_powerlaw, sw_logf=sw_logf,
        band=band, band_alpha=band_alpha,
        chrom_annual=False, chrom_exponential=False, chrom_gaussian=False,
        chrom_sphere=chrom_sphere, chrom_step=chrom_step,
        extra_gps=ppta_gps, return_components=True)

    update_priordict_standard_ppta(config)

    if return_components:
        return model, components

    return model


def common_noise(psrs, chain_dfs, **kwargs):
    """PPTA common-noise likelihood.

    Wraps :func:`mpta.common_noise`, which rebuilds per-pulsar models from the
    chain columns. The PPTA group noise and ECORR stack are not reinstated.
    """
    update_priordict_standard_ppta()

    return mpta.common_noise(psrs, chain_dfs, **kwargs)
