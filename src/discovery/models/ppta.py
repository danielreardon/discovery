"""PPTA-DR4 single-pulsar noise model.

Built on :mod:`discovery.models.mpta`, with PPTA-specific per-backend group
noise, band noise, deterministic chromatic events and ECORR architecture.

Entry point: :func:`single_pulsar_noise`. Per-pulsar content is driven by
:data:`PPTA_CONFIG`.

Default model:

* FFT-covariance power-law GPs for the per-pulsar background, red noise, DM and
  chromatic noise at a maximum cadence of 1/30 days, with a free chromatic index
* marginalised chromatic polynomial, sharing the chromatic GP's index
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
import scipy.interpolate as si

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
        '(.*_)?chrom_gp_log10_A': [-20, -11],
        '(.*_)?chrom_gp_gamma': [0, 7],
        '(.*_)?chrom_gp_alpha': [3, 10],
        # Corner frequency of the optional low-frequency turnover, one box for every
        # pulsar and every component so a hierarchical prior has a single support to
        # work with. The lower bound is set so the prior carries even odds on the
        # turnover: a corner below f_low/5 suppresses the lowest sampled bin by a few
        # per cent, which is undetectable, so that region is the "no turnover"
        # hypothesis and its share of the box is the prior probability of it. At the
        # 20.96 yr PPTA array span, f_low/5 is 10^-9.52 and a lower bound of -12.6
        # puts 0.498 of the box below it. The upper bound is 1/(30 d). Recompute both
        # for a different array span, and quote the f_low/5 threshold with any odds
        # read off this box.
        '(.*_)?log10_fc': [-12.6, -6.4],
        # band GP centre and bandwidth are bounded per-pulsar from psr.freqs by
        # _set_band_priors at model-build time.
        '(.*_)?band_gp_log10_A': [-18, -11],
        '(.*_)?band_gp_gamma': [0, 7],
        # floored two decades lower than band_gp: the free chromatic index
        # makes the band GP sensitive to smaller amplitudes.
        '(.*_)?bandalpha_gp_log10_A': [-20, -11],
        '(.*_)?bandalpha_gp_gamma': [0, 7],
        '(.*_)?bandalpha_gp_alpha': [0, 10],
        r'(.*_)?group_noise_.*_log10_A': [-18, -11],
        r'(.*_)?group_noise_.*_gamma': [0, 7],
        # solar wind, quasi-periodic time-domain kernel. sigma is the rms
        # electron-density variability (cm^-3); ell and p are in days.
        '(.*_)?sw_gp_log10_sigma': [-2, 1.3],
        '(.*_)?sw_gp_log10_ell': [1, 4],
        '(.*_)?sw_gp_log10_Gamma': [-3, 2],
        '(.*_)?sw_gp_log10_p': [-2, 1.3],   # years (0.01 - 20 yr), as scaled in signals.quasi_periodic
        # Fourier power-law solar-wind GP, used when sw_powerlaw=True.
        '(.*_)?sw_gp_log10_A': [-10, 1],
        '(.*_)?sw_gp_gamma': [-4, 4],
        '(.*_)?n_earth': [0, 20],
        # deterministic chromatic events. Event epochs t0 are per-pulsar.
        r'(.*_)?chrom_exp_\d+_log10_Amp': [-10, -2],
        # tau floored at 10**1.5 = 31.6 d. PPTA-DR4 sessions are ~18 d apart, so a shorter
        # recovery is unresolvable and lets the event collapse onto a single outlier epoch.
        # It also bounds the exp(-dt/tau) dynamic range that the where-branch has to carry.
        r'(.*_)?chrom_exp_\d+_log10_tau': [1.5, 4],
        r'(.*_)?chrom_exp_\d+_sign_param': [-1, 1],
        r'(.*_)?chrom_exp_\d+_alpha': [-4, 4],
        '(.*_)?chrom_1yr_log10_Amp': [-10, -3],
        '(.*_)?chrom_1yr_phase': [0, 2 * np.pi],
        '(.*_)?chrom_1yr_alpha': [0, 7],
        '(.*_)?chrom_gauss_log10_Amp': [-10, -3],
        # sigma floored at 10**1.2 = 15.8 d, just under the ~18 d session spacing.
        '(.*_)?chrom_gauss_log10_sigma': [1.2, 3],
        '(.*_)?chrom_gauss_sign_param': [-1, 1],
        '(.*_)?chrom_gauss_alpha': [0, 7],
        '(.*_)?gauss_20cm_log10_Amp': [-10, -3],
        '(.*_)?gauss_20cm_log10_sigma': [0, 3],
        '(.*_)?gauss_20cm_t0': [57385, 57785],
        'curn_log10_A': [-18, -11],
        'curn_gamma': [0, 7],
    },

    # Which pulsars carry each optional component.
    "models_dict": {

        # Band noise: one chromatic GP per pulsar, with centre, bandwidth
        # and chromatic index sampled. Drives band_alpha; the achromatic
        # band GP is opt-in via band=True.
        "band_noise": ['J0125-2327', 'J0437-4715', 'J0613-0200', 'J1017-7156',
                       'J1045-4509', 'J1600-3053', 'J1643-1224', 'J1705-1903',
                       'J1713+0747', 'J1824-2452A', 'J1909-3744', 'J1939+2134'],

        # Exponential dips. 't0' is the epoch prior in MJD; 'alpha' the
        # chromatic index prior. An event is included only if its epoch window
        # overlaps the data.
        #
        # 'sign' fixes the sign of a known event, which removes sign_param from the sampled
        # parameters. Sampling it makes the log-posterior piecewise constant with a step at
        # zero (measured jumps of 12-68 nats) and zero gradient either side, which held NUTS
        # at a step size of 1e-4 to 2e-3 on every pulsar carrying an event. Set it only where
        # the sign is established; where it is not, leave it out, run both branches and
        # compare evidence rather than sampling across the discontinuity.
        "chrom_exp": {
            'J1713+0747': [{'t0': (54650.0, 54850.0), 'alpha': (1.0, 3.0), 'sign': -1},
                           {'t0': (57400.0, 57600.0), 'alpha': (0.0, 2.0), 'sign': -1},
                           {'t0': (59318.0, 59330.0), 'alpha': (-4.0, 4.0), 'sign': -1}],
            'J0437-4715': [{'t0': (57000.0, 57200.0), 'alpha': (-1.0, 2.0), 'sign': -1},
                           {'t0': (58032.0, 58132.0), 'alpha': (-4.0, 4.0), 'sign': -1},
                           {'t0': (59630.0, 59690.0), 'alpha': (-4.0, 4.0), 'sign': -1}],
            'J1643-1224': [{'t0': (57000.0, 57200.0), 'alpha': (-2.0, 0.0), 'sign': -1}],
            'J2145-0750': [{'t0': (56250.0, 56450.0), 'alpha': (-2.0, 2.0), 'sign': -1}],
        },

        # Annual chromatic sinusoid.
        "chrom_annual": [],

        # Chromatic Gaussian event, with per-pulsar epoch and width priors. 'sign' as above.
        "chrom_gauss": {
            'J1603-7202': {'t0': (53800.0, 54000.0), 'log10_sigma': (1.2, 3.0),
                           'sign': +1},
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
        'J1017-7156': ["PDFB_10CM", "UWL_sbH"],
        'J1603-7202': ["CASPSR_40CM", "CPSR2_50CM", "PDFB1_20CM"],
        'J1643-1224': ["CASPSR_40CM"],
        'J1902-5105': ["UWL_sbD"],
    },

    # Backends given their own ECORR, in addition to the global term.
    "ecorr_dict": {
        'J0437-4715': ["PDFB_10CM", "PDFB_20CM", "PDFB_40CM", "UWL_sbC",
                       "UWL_sbD", "UWL_sbE", "UWL_sbF", "UWL_sbG"],
        'J1022+1001': ["UWL_sbH"],
        'J1603-7202': ["CPSR2_50CM"],
        'J1705-1903': ["UWL_sbH"],
        'J1744-1134': ["CPSR2_50CM"],
        'J1824-2452A': ["UWL_sbB"],
        'J1902-5105': ["UWL_sbD"],
        'J1909-3744': ["PDFB1_1433", "UWL_PDFB4_20CM", "UWL_sbG"],
        'J2051-0827': ["UWL_sbE"],
        'J2129-5721': ["PDFB1_20CM"],
        'J2241-5236': ["UWL_sbB"]
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
    """Time-interpolation basis restricted to ``selection``, rows outside zeroed.

    Counterpart of :func:`masked_fourierbasis` for the FFT-covariance GPs. When
    ``T`` is not supplied the coarse grid spans the selected TOAs; an explicit
    ``T`` is always honoured, and must bracket them.
    """
    def basis(psr, components, T=None):
        m = np.asarray(selection(psr), dtype=bool)
        t_fine = psr.toas[m]
        tmin = float(t_fine.min())

        t0 = tmin if start_time is None else float(start_time)
        if t0 > tmin:
            raise ValueError('Coarse time basis start must be earlier than '
                             'earliest TOA.')

        if T is None:
            T = float(t_fine.max() - tmin)

        t_coarse = np.linspace(t0, t0 + T, components)
        dt_coarse = t_coarse[1] - t_coarse[0]

        if t_fine.max() > t_coarse[-1]:
            raise ValueError(
                f'{psr.name}: selected TOAs extend '
                f'{(t_fine.max() - t_coarse[-1]) / 86400.0:.1f} d beyond the '
                f'coarse grid; T is too short.')

        # fill_value=0 supplies the zero rows outside the selection; the check
        # above ensures no selected TOA is silently zeroed.
        Bsub = si.interp1d(t_coarse, np.identity(components), kind=order,
                           bounds_error=False, fill_value=0.0,
                           axis=0)(t_fine)

        Bmat = np.zeros((len(psr.toas), components))
        Bmat[m, :] = Bsub

        return t_coarse, dt_coarse, Bmat

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

# TOA quantisation bin width for the time-domain solar-wind GP; see mpta.SW_DT. Half a
# day, which puts one observing session in each bin. The period of the quasi-periodic
# kernel is NOT fixed here: the PPTA span exceeds a solar cycle, so the data can
# measure it.
SW_DT = 43200.0


def _set_band_priors(psr, band=False, band_alpha=False, bw_min_mhz=20.0):
    """Set data-bounded per-pulsar priors for the (fcenter, log10_bw) band GPs.

    The band centre is bounded by the pulsar's actual frequency coverage and the
    bandwidth runs from ``bw_min_mhz`` to the full coverage span, so the band
    always overlaps data and can never collapse to an empty envelope.
    """
    freqs = np.asarray(psr.freqs)
    fmin, fmax = float(freqs.min()), float(freqs.max())
    span = max(fmax - fmin, 2.0 * bw_min_mhz)
    names = (['band_gp'] if band else []) + (['bandalpha_gp'] if band_alpha else [])
    updates = {}
    for n in names:
        psr_key = re.escape(psr.name)
        updates[f'{psr_key}_{n}_fcenter'] = [fmin, fmax]
        updates[f'{psr_key}_{n}_log10_bw'] = [float(np.log10(bw_min_mhz)),
                                              float(np.log10(span))]
    prior.priordict_standard.update(updates)


def _set_chrom_exp_priors(psr, chrom_exponential=False, tau_min_days=10.0,
                          log10_amp_max=-5.0):
    """Set data-bounded per-pulsar priors for the generic chromatic exponential.

    Applies to the single unlabelled ``chrom_exp`` event enabled by the
    ``chrom_exponential`` switch, not to the per-event windows in
    ``models_dict['chrom_exp']``, which :func:`make_psr_delays` registers.

    The two are kept apart by name: labelled events are ``chrom_exp_<i>_*``, so
    the unnumbered ``chrom_exp_*`` patterns written here cannot match them.

    The epoch is bounded by the pulsar's observing span and the decay timescale
    runs from ``tau_min_days`` to that span, so the event always overlaps data.
    The amplitude is capped at the residual peak-to-peak, or at
    ``log10_amp_max`` where that is tighter.

    Keys are inserted AHEAD of the existing entries: ``getprior_uniform`` takes
    the first regex match in insertion order and the generic
    ``(.*_)?chrom_exp_*`` patterns are already present, so a key appended after
    them would never be reached.
    """
    if not chrom_exponential:
        return {}

    mjd = np.asarray(psr.toas) / 86400.0
    tmin, tmax = float(mjd.min()), float(mjd.max())
    span = max(tmax - tmin, 2.0 * tau_min_days)

    resid = np.asarray(psr.residuals)
    ptp = float(resid.max() - resid.min())
    amp_hi = min(float(log10_amp_max), float(np.log10(ptp)))
    try:
        amp_lo = float(prior.getprior_uniform(f'{psr.name}_chrom_exp_log10_Amp')[0])
    except (KeyError, ValueError):
        amp_lo = -10.0

    psr_key = re.escape(psr.name)
    updates = {
        f'{psr_key}_chrom_exp_t0': [tmin, tmax],
        f'{psr_key}_chrom_exp_log10_tau': [float(np.log10(tau_min_days)),
                                           float(np.log10(span))],
        f'{psr_key}_chrom_exp_log10_Amp': [amp_lo, amp_hi],
    }
    rest = {k: v for k, v in prior.priordict_standard.items() if k not in updates}
    prior.priordict_standard.clear()
    prior.priordict_standard.update({**updates, **rest})

    return updates


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

def makegp_group_noise(psr, group_dict=None, max_cadence_days=30.0, Tspan=None,
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
                              Tspan=None, group_tspan='backend', order=1):
    """Per-backend-group achromatic power-law GPs, FFT-covariance basis.

    FFT-covariance counterpart of :func:`makegp_group_noise`, sharing its
    component count and span conventions and the same power-law
    parametrisation, so priors are interchangeable between the two.

    Parameters
    ----------
    group_dict : dict, optional
        ``{pulsar name: [backend flags]}``. Defaults to the module config.
    group_tspan : {'backend', 'psr'}
        Coarse-grid span: the group's own TOAs, or the full pulsar span.
    order : int
        Interpolation order of the time-domain basis.
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
        if ntoa < 2:
            print(f'Warning: group {g!r} for {psr.name} has {ntoa} TOAs; '
                  f'skipped.')
            continue

        group_toas = psr.toas[mask]
        T = psr_Tspan if use_psr_span else float(group_toas.max()
                                                 - group_toas.min())
        ncomp = min(int(T / (max_cadence_days * 86400.0)),
                    max(1, ntoa // GROUP_TOA_PER_MODE))
        if ncomp < 1:
            print(f'Warning: group {g!r} for {psr.name} has {ntoa} TOAs over '
                  f'{T / 86400.0:.0f} d; skipped.')
            continue

        # psd2cov requires an odd number of knots.
        knots = 2 * ncomp + 1
        t0 = float(group_toas.min())

        # T is passed explicitly: makegp_fftcov otherwise defaults it to the
        # full pulsar span, which would size the prior for a span the coarse
        # grid does not cover.
        gps += [signals.makegp_fftcov(
            psr, signals.powerlaw, components=knots, T=T,
            fourierbasis=masked_timeinterpbasis(
                _make_mask_selection(mask), start_time=t0, order=order),
            name=f'group_noise_{g}')]

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
        psr, covariance=covariance, dt=SW_DT,
        name='sw_gp')]


# ---------------------------------------------------------------------------
# frequency-dependent delay
# ---------------------------------------------------------------------------

def makegp_fd(psr, nodes=16, spacing='quantile', selection=None, groups=None,
              fd_groups_dict=None, prior='improper'):
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
    prior : {'improper', 'matern'}
        ``'improper'`` marginalises the node amplitudes with an improper prior,
        removing those directions unconditionally. ``'matern'`` gives them a
        Matern-3/2 prior in log-frequency and samples its scale and correlation
        length, so the data set how much is absorbed.
    """
    if groups is None:
        fd_groups_dict = (PPTA_CONFIG["models_dict"]["fd_groups"]
                          if fd_groups_dict is None else fd_groups_dict)
        groups = fd_groups_dict.get(psr.name, None)

    if groups and selection is None:
        selection = signals.selection_backend_flags

    if prior == 'matern':
        return signals.makegp_fd_piecewise_matern(psr, nodes=nodes, spacing=spacing,
                                                  selection=selection, groups=groups,
                                                  name='fd_gp')
    if prior != 'improper':
        raise ValueError(f"makegp_fd: prior must be 'improper' or 'matern', got {prior!r}.")

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
                psr, deterministic.chromatic_exponential(psr, sign=ev.get('sign')),
                name=nm)]
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
                psr, deterministic.chromatic_gaussian(psr, sign=ev.get('sign')),
                name='chrom_gauss')]
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
# entry point
# ---------------------------------------------------------------------------

def make_psr_gps_fourier(
    psr,
    max_cadence_days=30,
    bkgrnd_log10_A=None,
    Tspan=None,
    background=True,
    red=True,
    red2=False,
    dm=True,
    chrom=True,
    chrom_alpha=None,
    chrom_poly=False,
    sw=True,
    sw_powerlaw=False,
    sw_logf=False,
    band=False,
    band_alpha=False,
    turnover=None,
    fd_gp=None,
):
    """Per-pulsar power-law GPs on a Fourier basis.

    Component count is ``int(T / max_cadence_days)``. ``sw_powerlaw`` selects the
    Fourier solar-wind GP over the time-domain one; ``sw_logf`` log-spaces its
    frequencies.
    """
    psr_Tspan = signals.getspan(psr) if Tspan is None else Tspan
    psr_components = int(psr_Tspan / (max_cadence_days * 86400))
    _set_band_priors(psr, band=band, band_alpha=band_alpha)
    turnover = signals.turnover_set(turnover)

    gp_signals = []

    if background:
        gp_signals.append(signals.makegp_fourier(psr, signals.powerlaw_gwb(log10_A=bkgrnd_log10_A), components=psr_components, T=psr_Tspan, name='bkgrnd'))
    if red:
        gp_signals.append(signals.makegp_fourier(psr, signals.turnover_psd('red', turnover), components=psr_components, T=psr_Tspan, name='red_noise'))
    if red2:
        gp_signals.append(signals.makegp_fourier(psr, signals.turnover_psd('red2', turnover), components=psr_components, T=psr_Tspan, name='red_noise2'))
    if dm:
        gp_signals.append(signals.makegp_fourier(psr, signals.turnover_psd('dm', turnover), components=psr_components, T=psr_Tspan, fourierbasis=signals.fourierbasis_dm, name='dm_gp'))
    if chrom:
        gp_signals.append(signals.makegp_fourier(psr, signals.turnover_psd('chrom', turnover), components=psr_components, T=psr_Tspan, fourierbasis=signals.fourierbasis_chrom, name='chrom_gp', alpha=chrom_alpha))
    if chrom and chrom_poly:
        gp_signals.append(signals.makegp_chrom_poly_svd(psr, name='chrom_gp', project=fd_gp))
    if sw and not sw_powerlaw:
        gp_signals.append(solar.makegp_timedomain_solar_dm(psr, covariance=signals.squared_exponential, dt=SW_DT, name='sw_gp'))
    if sw and sw_powerlaw:
        gp_signals.append(signals.makegp_fourier(psr, signals.powerlaw, components=psr_components, T=psr_Tspan, fourierbasis=solar.make_fourierbasis_solar_dm(logf=sw_logf), name='sw_gp'))
    if band:
        gp_signals.append(signals.makegp_fourier(psr, signals.powerlaw, components=psr_components, T=psr_Tspan, fourierbasis=signals.fourierbasis_band, name='band_gp'))
    if band_alpha:
        gp_signals.append(signals.makegp_fourier(psr, signals.powerlaw, components=psr_components, T=psr_Tspan, fourierbasis=signals.fourierbasis_band_alpha, name='bandalpha_gp'))

    return gp_signals


def make_psr_gps_fftint(
    psr,
    max_cadence_days=30,
    bkgrnd_log10_A=None,
    Tspan=None,
    background=True,
    red=True,
    red2=False,
    dm=True,
    chrom=True,
    chrom_alpha=None,
    chrom_poly=False,
    sw=True,
    sw_powerlaw=False,
    band=False,
    band_alpha=False,
    turnover=None,
    fd_gp=None,
):
    """Per-pulsar power-law GPs on an FFT-covariance basis.

    FFT-covariance counterpart of :func:`make_psr_gps_fourier`, using
    ``2 * components + 1`` knots as psd2cov requires an odd count.
    """
    psr_Tspan = signals.getspan(psr) if Tspan is None else Tspan
    psr_components = int(psr_Tspan / (max_cadence_days * 86400))
    psr_knots = 2 * psr_components + 1
    _set_band_priors(psr, band=band, band_alpha=band_alpha)
    turnover = signals.turnover_set(turnover)

    gp_signals = []

    if background:
        gp_signals.append(signals.makegp_fftcov(psr, signals.powerlaw_gwb(log10_A=bkgrnd_log10_A), components=psr_knots, T=psr_Tspan, name='bkgrnd'))
    if red:
        gp_signals.append(signals.makegp_fftcov(psr, signals.turnover_psd('red', turnover), components=psr_knots, T=psr_Tspan, name='red_noise'))
    if red2:
        gp_signals.append(signals.makegp_fftcov(psr, signals.turnover_psd('red2', turnover), components=psr_knots, T=psr_Tspan, name='red_noise2'))
    if dm:
        gp_signals.append(signals.makegp_fftcov_dm(psr, signals.turnover_psd('dm', turnover), components=psr_knots, T=psr_Tspan, name='dm_gp'))
    if chrom:
        gp_signals.append(signals.makegp_fftcov_chrom(psr, signals.turnover_psd('chrom', turnover), components=psr_knots, T=psr_Tspan, name='chrom_gp', alpha=chrom_alpha))
    if chrom and chrom_poly:
        gp_signals.append(signals.makegp_chrom_poly_svd(psr, name='chrom_gp', project=fd_gp))
    if sw and not sw_powerlaw:
        gp_signals.append(solar.makegp_timedomain_solar_dm(psr, covariance=signals.squared_exponential, dt=SW_DT, name='sw_gp'))
    if sw and sw_powerlaw:
        gp_signals.append(signals.makegp_fftcov_solar(psr, signals.powerlaw, components=psr_knots, T=psr_Tspan, name='sw_gp'))
    if band:
        gp_signals.append(signals.makegp_fftcov_band(psr, signals.powerlaw, components=psr_knots, T=psr_Tspan, name='band_gp'))
    if band_alpha:
        gp_signals.append(signals.makegp_fftcov_band_alpha(psr, signals.powerlaw, components=psr_knots, T=psr_Tspan, name='bandalpha_gp'))

    return gp_signals


def single_pulsar_noise(
    psr,
    fftint=True,
    max_cadence_days=30,
    Tspan=None,
    noisedict={},
    white_selection=None,  # per-TOA flag whose values split efac and tnequad; ECORR is not split
    ecorr=True,
    ecorr_nmodes=3,
    ecorr_correlated=True,
    ecorr_per_backend=True,
    ecorr_dict=None,
    background=True,
    bkgrnd_log10_A=None,
    red=True,
    red2=False,
    dm=True,
    chrom=True,
    chrom_alpha=None,
    chrom_poly=True,
    fd=False,
    fd_nodes=16,
    fd_spacing='quantile',
    fd_selection=None,
    fd_groups=None,
    fd_prior='improper',
    sw=None,
    sw_elat_max=SW_ELAT_MAX,
    sw_kernel='qp',
    sw_powerlaw=False,
    sw_logf=False,
    mean_sw=False,
    band=False,
    band_alpha=None,
    turnover=None,  # components ('red', 'red2', 'dm', 'chrom') whose power law takes a low-frequency turnover
    group=None,
    group_dict=None,
    group_tspan='backend',
    chrom_exp=None,
    chrom_annual=None,
    chrom_gauss=None,
    chrom_gauss_20cm=None,
    chrom_exponential=False,
    chrom_sphere=False,
    chrom_step=False,
    config=PPTA_CONFIG,
    extra_gps=None,
    return_components=False,
):
    """Build the PPTA-DR4 single-pulsar noise likelihood.

    The base GPs come from :func:`make_psr_gps_fftint` or
    :func:`make_psr_gps_fourier`; the PPTA ECORR stack, group noise, solar-wind
    GP, frequency-dependent delay and chromatic events are added here.

    Per-pulsar switches (``sw``, ``band_alpha``, ``group``, ``chrom_exp``,
    ``chrom_annual``, ``chrom_gauss``, ``chrom_gauss_20cm``) default to ``None``,
    meaning "use the configuration for this pulsar". An explicit value overrides
    the configuration. ``band`` is not among them: the configured band model is
    the chromatic one, so the achromatic GP is off unless asked for.

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
        chromatic GP. On by default.
    band : bool
        Add the achromatic band GP. Off by default; not driven by the
        configuration.
    band_alpha : bool, optional
        Add the chromatic band GP. ``None`` uses ``models_dict['band_noise']``.
    chrom_exponential : bool
        Search for one additional unlabelled chromatic exponential event, over
        the full span with data-bounded priors. The labelled events in
        ``models_dict['chrom_exp']`` are always included regardless.
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
    band_alpha = ((psr.name in md["band_noise"]) if band_alpha is None
                  else band_alpha)
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

    if group:
        make_group = (makegp_group_noise_fftcov if fftint
                      else makegp_group_noise)
        ppta_gps += make_group(psr, group_dict=group_dict,
                               max_cadence_days=max_cadence_days,
                               Tspan=Tspan, group_tspan=group_tspan)

    # The fd basis is built here so that its groups can be set per pulsar, and
    # is projected out of the chromatic polynomial where both are present.
    fd_gp = None
    if fd:
        fd_gp = makegp_fd(psr, nodes=fd_nodes, spacing=fd_spacing,
                          selection=fd_selection, groups=fd_groups,
                          fd_groups_dict=md.get("fd_groups"), prior=fd_prior)
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

    # Base model. The solar-wind GP is passed through only in its power-law
    # form; the time-domain one is already in ppta_gps.
    components = [psr.residuals]
    components += [signals.makegp_timing(psr, svd=True)]
    _white_sel = (signals.selection_flags(white_selection) if isinstance(white_selection, str)
                  else white_selection)
    components += [signals.makenoise_measurement(
        psr, tnequad=True, noisedict=noisedict,
        **({'selection': _white_sel} if _white_sel is not None else {}))]

    # An additional unlabelled event, searched over the full span. The labelled
    # events in models_dict['chrom_exp'] are always in the model, added by
    # make_psr_delays above and unaffected by this switch.
    if chrom_exponential:
        if psr.name in md["chrom_exp"]:
            print(f'{psr.name}: searching for 1 unlabelled chromatic '
                  f'exponential event, in addition to the '
                  f'{len(md["chrom_exp"][psr.name])} labelled event(s).')
        _set_chrom_exp_priors(psr, chrom_exponential=True)
        components += [signals.makedelay(
            psr, deterministic.chromatic_exponential(psr), name='chrom_exp')]
    if chrom_sphere:
        components += [signals.makedelay(
            psr, deterministic.chromatic_sphere(psr), name='chrom_sphere')]
    if chrom_step:
        components += [signals.makedelay(
            psr, deterministic.chromatic_step(psr), name='chrom_step')]

    make_gps = make_psr_gps_fftint if fftint else make_psr_gps_fourier
    gp_kwargs = dict(max_cadence_days=max_cadence_days,
                     bkgrnd_log10_A=bkgrnd_log10_A, Tspan=Tspan,
                     background=background, red=red, red2=red2, dm=dm,
                     chrom=chrom, chrom_alpha=chrom_alpha, chrom_poly=False,
                     sw=(sw and sw_powerlaw), sw_powerlaw=sw_powerlaw,
                     band=band, band_alpha=band_alpha, turnover=turnover, fd_gp=None)
    if fftint:
        if sw_logf:
            print('Warning: sw_logf=True is ignored with fftint=True (the '
                  'FFT-covariance solar GP uses a time-interpolation basis). '
                  'Use fftint=False.')
    else:
        gp_kwargs['sw_logf'] = sw_logf
    components += make_gps(psr, **gp_kwargs)

    components += ppta_gps

    model = likelihood.PulsarLikelihood(components)

    # No second update_priordict_standard_ppta(config) here. It is already called at the top
    # of this function, before the model is built. Calling it again after make_psr_delays has
    # registered the per-pulsar event windows re-inserted the generic patterns AHEAD of them
    # (register_priors prepends, and getprior_uniform takes the first re.match), silently
    # discarding every per-event override whose name a generic pattern also matches. The
    # visible symptom was chrom_exp_<i>_alpha reverting from its per-event window to the
    # generic [0, 7] between make_psr_delays and the return; t0 escaped only because no
    # generic chrom_exp_\d+_t0 pattern exists.

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
