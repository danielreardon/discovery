"""Single-pulsar noise model for the IPTA data release.

This is a trimmed-down sibling of :mod:`discovery.models.mpta`. The IPTA base
single-pulsar model only requires the standard white noise (efac, equad, ecorr),
red noise, DM, chromatic noise and a solar-wind term. Some pulsars will get
additional chromatic processes, but those are added manually on top of this base
model (via the ``extra_gps`` argument or by appending model components).

The complication relative to MPTA is that the IPTA combines several PTAs, each of
which defines its backend/frontend in a different TOA flag. The
``get_flag_groups_by_PTA`` helper and the ``CustomSelections`` class (ported from
the platypus DR3 noise-modelling utilities) decide which TOA flag acts as the
backend flag for each PTA, and turn that into the per-TOA backend labels that
Discovery's noise functions consume through their ``selection`` argument.

By default efac and equad are split per backend (using the ported selection), and
a single global *decorrelating* ECORR is applied across all TOAs (no backend
selection). Set ``ecorr_per_backend=True`` to additionally split ECORR by backend.
"""

import numpy as np
import jax.numpy as jnp
import re

from .. import matrix
from .. import signals
from .. import prior
from .. import solar
from .. import likelihood
from .. import deterministic
from .. import const


# ---------------------------------------------------------------------------
# Backend-flag selection utilities (ported from platypus selections.py and
# model_utils.py, adapted to Discovery's per-TOA selection-function interface).
# ---------------------------------------------------------------------------

# LOFAR appears as a set of -group values rather than a -pta value, so it is
# handled as a special case when assigning ECORR backends.
LOFAR_groups = ['DE601.150', 'DE602.150', 'DE603.150', 'DE604.150',
                'DE605.150', 'DE609.150', 'LOFAR.150', 'NENUFAR.50']


def _pta_flags(psr):
    """Return the per-TOA -pta flag array for a Discovery Pulsar.

    Falls back to labelling everything 'MPTA' when there is no -pta flag but the
    -f flag identifies MeerKAT (KAT) data, matching the platypus behaviour.
    """
    flags = psr.flags
    if 'pta' in flags:
        return flags['pta']
    if 'f' in flags and len(flags['f']) and 'KAT' in flags['f'][0]:
        return np.array(['MPTA'] * len(flags['f']))
    raise ValueError("Pulsar has no -pta flag and could not be identified as "
                     "MPTA from the -f flag; cannot assign backend flags.")


# These flag mappings are currently hard-coded for IPTA DR3 vanilla noise modelling
def get_flag_groups_by_PTA(psr):
    """Decide which TOA flag is the backend flag for each PTA in this pulsar.

    Returns two dictionaries, ``{pta_name: flag_name}``, telling
    :class:`CustomSelections` which TOA flag to use for the efac/equad backends
    and for the ECORR backends respectively. These selections are hard-coded for
    IPTA DR3 vanilla noise modelling.
    """
    ptas = _pta_flags(psr)
    groups = psr.flags['group'] if 'group' in psr.flags else np.array([])

    efeq_groups_by_PTA = {}
    ecorr_groups_by_PTA = {}
    for pta_name in np.unique(ptas):
        # MPTA, NANOGrav, CHIME, CPTA use the -f flag
        if pta_name in ['MPTA', 'NANOGrav', 'CHIME', 'CPTA']:
            efeq_groups_by_PTA[pta_name] = 'f'
            ecorr_groups_by_PTA[pta_name] = 'f'
        # everything else uses the -group flag
        else:
            efeq_groups_by_PTA[pta_name] = 'group'
            # Use -b or -B for PPTA with PINT or T2 files, respectively.
            # PINT tim flags are lower-cased on load; T2 tim flags keep their case.
            if pta_name == 'PPTA' and 'b' in psr.flags:
                ecorr_groups_by_PTA[pta_name] = 'b'
            elif pta_name == 'PPTA' and 'B' in psr.flags:
                ecorr_groups_by_PTA[pta_name] = 'B'
            # do not apply per-backend ECORR to EPTA data
            elif pta_name != 'EPTA':
                ecorr_groups_by_PTA[pta_name] = 'group'

    # LOFAR is not in the -pta flags; detect it from -group and add it explicitly.
    if 'group' in psr.flags and np.any([gr in groups for gr in LOFAR_groups]):
        ecorr_groups_by_PTA['LOFAR'] = 'group'

    return efeq_groups_by_PTA, ecorr_groups_by_PTA


class CustomSelections(object):
    """Per-TOA backend-label builder for a multi-PTA pulsar.

    ``efeq_groups``/``ecorr_groups`` are ``{pta_name: flag_name}`` dictionaries
    (typically from :func:`get_flag_groups_by_PTA`) specifying which TOA flag to
    use as the backend flag for each PTA's efac/equad and ECORR respectively.

    Unlike the enterprise version (which returns ``{flagval: mask}`` dictionaries),
    these methods return a single per-TOA array of backend labels, which is what
    Discovery's noise functions expect from a ``selection(psr)`` callable. TOAs
    belonging to a PTA that is not in the relevant group dictionary are labelled
    with the empty string and are skipped by the noise builders.
    """

    def __init__(self, efeq_groups, ecorr_groups):
        self.efeq_groups = efeq_groups
        self.ecorr_groups = ecorr_groups

    def by_backend(self, psr):
        """Per-TOA efac/equad backend labels."""
        flags = psr.flags
        pta_flags = _pta_flags(psr)
        labels = np.array([''] * len(pta_flags), dtype=object)
        for pta, flag in self.efeq_groups.items():
            mask = (pta_flags == pta)
            labels[mask] = flags[flag][mask]
        return labels.astype(str)

    def by_sb_backend(self, psr):
        """Per-TOA ECORR (sub-band) backend labels, with LOFAR handled by -group."""
        flags = psr.flags
        pta_flags = _pta_flags(psr)
        labels = np.array([''] * len(pta_flags), dtype=object)
        for pta, flag in self.ecorr_groups.items():
            if pta == 'LOFAR':
                # LOFAR is identified by its -group values, not by -pta.
                mask = np.zeros(len(pta_flags), dtype=bool)
                for gr in LOFAR_groups:
                    mask |= (flags['group'] == gr)
            else:
                mask = (pta_flags == pta)
            labels[mask] = flags[flag][mask]
        return labels.astype(str)


def selection_ipta_efeq(psr):
    """Discovery selection function: per-TOA efac/equad backend labels."""
    efeq_groups, _ = get_flag_groups_by_PTA(psr)
    return CustomSelections(efeq_groups, {}).by_backend(psr)


def selection_ipta_ecorr(psr):
    """Discovery selection function: per-TOA ECORR backend labels."""
    _, ecorr_groups = get_flag_groups_by_PTA(psr)
    return CustomSelections({}, ecorr_groups).by_sb_backend(psr)


def selection_global(psr):
    """Discovery selection function: a single backend covering all TOAs.

    This is the "no backend selection" case. It is needed to build a *global*
    ECORR with :func:`signals.makegp_ecorr_legendre`, whose default selection
    (``selection_backend_flags``) would otherwise split ECORR by backend.
    """
    return np.array(['global'] * len(psr.toas))


# ---------------------------------------------------------------------------
# Priors
# ---------------------------------------------------------------------------

def update_priordict_standard_ipta():
    """Update the standard prior dictionary with the IPTA base-model parameters."""
    prior.priordict_standard.update({
        # White noise parameters
        '(.*_)?efac':               [0.5, 2],
        '(.*_)?log10_tnequad':      [-10, -5],
        '(.*_)?log10_ecorr_k.*':    [-10, -5],
        '(.*_)?log10_ecorr':        [-10, -5],
        # GP parameters
        '(.*_)?red_noise_log10_A.*':  [-18, -11],
        '(.*_)?red_noise_gamma.*':    [0, 7],
        '(.*_)?dm_gp_log10_A':      [-18, -11],
        '(.*_)?dm_gp_gamma':        [0, 7],
        '(.*_)?chrom_gp_log10_A':   [-18, -11],
        '(.*_)?chrom_gp_gamma':     [0, 7],
        '(.*_)?chrom_gp_alpha':     [3.0, 14],  # start at 3 to avoid confusion with DM
        # Solar wind GP
        '(.*_)?sw_gp_log10_A':      [-10, -2],
        '(.*_)?sw_gp_gamma':        [0, 4],
        # SE kernel for time-domain SW GP
        # sigma : rms electron density variability (cm^-3)
        # ell   : correlation timescale (days)
        '(.*_)?sw_gp_log10_sigma':  [-2, 1.3],   # 0.01 - 20 cm^-3
        '(.*_)?sw_gp_log10_ell':    [1, 4],     # 10 days - ~30 yr
    })

    return


# Ensure priordict_standard is updated on import, and again at model-build time
# to catch any changes during likelihood/prior initialisation.
update_priordict_standard_ipta()


# ---------------------------------------------------------------------------
# GP components
# ---------------------------------------------------------------------------

def make_psr_gps_fourier(psr, max_cadence_days=14, Tspan=None,
                         red=True, dm=True, chrom=True, chrom_poly=False,
                         sw=True, sw_powerlaw=False):
    """Build the Fourier-basis GP components (red, DM, chromatic, solar wind)."""
    psr_Tspan = signals.getspan(psr) if Tspan is None else Tspan
    psr_components = int(psr_Tspan / (max_cadence_days * 86400))

    return (([signals.makegp_fourier(psr, signals.powerlaw, components=psr_components, name='red_noise')] if red else []) + \
            ([signals.makegp_fourier(psr, signals.powerlaw, components=psr_components, fourierbasis=signals.fourierbasis_dm, name='dm_gp')] if dm else []) + \
            ([signals.makegp_fourier(psr, signals.powerlaw, components=psr_components, fourierbasis=signals.fourierbasis_chrom, name='chrom_gp')] if chrom else []) + \
            ([signals.makegp_chrom_poly_svd(psr, name='chrom_gp')] if (chrom and chrom_poly) else []) + \
            # Solar wind: time-domain squared-exponential GP by default, or the power-law (Fourier) GP when sw_powerlaw=True (legacy treatment).
            ([solar.makegp_timedomain_solar_dm(psr, covariance=signals.squared_exponential, dt=max_cadence_days*86400.0, name='sw_gp')] if (sw and not sw_powerlaw) else []) + \
            ([signals.makegp_fourier(psr, signals.powerlaw, components=psr_components, fourierbasis=solar.fourierbasis_solar, name='sw_gp')] if (sw and sw_powerlaw) else []))


def make_psr_gps_fftint(psr, max_cadence_days=14, Tspan=None,
                        red=True, dm=True, chrom=True, chrom_poly=False,
                        sw=True, sw_powerlaw=False):
    """Build the FFT-covariance GP components (red, DM, chromatic, solar wind)."""
    psr_Tspan = signals.getspan(psr) if Tspan is None else Tspan
    psr_components = int(psr_Tspan / (max_cadence_days * 86400))
    psr_knots = 2 * psr_components + 1

    return (([signals.makegp_fftcov(psr, signals.powerlaw, components=psr_knots, name='red_noise')] if red else []) + \
            ([signals.makegp_fftcov_dm(psr, signals.powerlaw, components=psr_knots, name='dm_gp')] if dm else []) + \
            ([signals.makegp_fftcov_chrom(psr, signals.powerlaw, components=psr_knots, name='chrom_gp')] if chrom else []) + \
            ([signals.makegp_chrom_poly_svd(psr, name='chrom_gp')] if (chrom and chrom_poly) else []) + \
            # Solar wind: time-domain squared-exponential GP by default, or the power-law (FFT-covariance) GP when sw_powerlaw=True (legacy treatment).
            ([solar.makegp_timedomain_solar_dm(psr, covariance=signals.squared_exponential, dt=max_cadence_days*86400.0, name='sw_gp')] if (sw and not sw_powerlaw) else []) + \
            ([signals.makegp_fftcov_solar(psr, signals.powerlaw, components=psr_knots, name='sw_gp')] if (sw and sw_powerlaw) else []))


# ---------------------------------------------------------------------------
# Single-pulsar model
# ---------------------------------------------------------------------------

def single_pulsar_noise(psr, fftint=True, max_cadence_days=14, Tspan=None, noisedict={},
                        ecorr=True, ecorr_nmodes=None, ecorr_per_backend=False,  # ECORR options (see below)
                        red=True, dm=True, chrom=True, chrom_poly=False, sw=True, sw_powerlaw=False,  # Base GP model: red, dm, chromatic, solar wind (sw_powerlaw=True selects the legacy power-law solar-wind GP instead of the time-domain one)
                        extra_gps=None,  # Extra GPs (e.g. additional chromatic processes added per pulsar)
                        return_components=False):  # Whether to return the list of model components in addition to the likelihood object (useful for adding additional components)
    """Build the IPTA base single-pulsar noise model.

    White noise: per-backend efac and equad, with the backend flag chosen per PTA
    by :func:`selection_ipta_efeq`. ECORR: a single global ECORR across all TOAs by
    default (``ecorr=True``, no backend selection). Set ``ecorr_nmodes=N`` to make
    that global ECORR the decorrelating Legendre-mode model used in MPTA (mode 0 is
    standard ECORR; higher modes add chromatic decorrelation across the band) -- the
    only difference from :func:`signals.makegp_ecorr_legendre` is that it is applied
    globally rather than per backend (its default). Set ``ecorr_per_backend=True`` to
    additionally include a per-backend ECORR (using :func:`selection_ipta_ecorr`,
    Legendre as well when ``ecorr_nmodes`` is set). GP processes: red noise, DM,
    chromatic and a solar-wind term. Additional chromatic processes for specific
    pulsars can be supplied through ``extra_gps``.
    """
    # Ensure the prior dictionary is up to date at model-build time.
    update_priordict_standard_ipta()

    # Per-backend white noise (efac and tnequad), using the IPTA backend selection.
    measurement_noise = signals.makenoise_measurement(psr, tnequad=True, noisedict=noisedict,
                                                       selection=selection_ipta_efeq)
    # Set up model components
    model_components = [psr.residuals]
    model_components += [signals.makegp_timing(psr, svd=True)]  # Timing model (analytically marginalised)
    model_components += [measurement_noise]

    # Global ECORR across all TOAs (no backend selection). With ecorr_nmodes set,
    # use the decorrelating Legendre-mode ECORR (as in MPTA) but applied globally.
    if ecorr:
        if ecorr_nmodes is not None:
            model_components += [signals.makegp_ecorr_legendre(psr, noisedict=noisedict,
                                                               nmodes=ecorr_nmodes, selection=selection_global)]
        else:
            model_components += [signals.makegp_ecorr_simple(psr, noisedict=noisedict)]
    # Optional additional per-backend ECORR, using the IPTA ECORR backend selection.
    if ecorr_per_backend:
        if ecorr_nmodes is not None:
            model_components += [signals.makegp_ecorr_legendre(psr, noisedict=noisedict,
                                                               nmodes=ecorr_nmodes, selection=selection_ipta_ecorr)]
        else:
            model_components += [signals.makegp_ecorr(psr, noisedict=noisedict, selection=selection_ipta_ecorr)]

    # Add GP components
    if fftint:
        model_components += make_psr_gps_fftint(psr, max_cadence_days=max_cadence_days, Tspan=Tspan,
                                                red=red, dm=dm, chrom=chrom, chrom_poly=chrom_poly, sw=sw, sw_powerlaw=sw_powerlaw)
    else:
        model_components += make_psr_gps_fourier(psr, max_cadence_days=max_cadence_days, Tspan=Tspan,
                                                 red=red, dm=dm, chrom=chrom, chrom_poly=chrom_poly, sw=sw, sw_powerlaw=sw_powerlaw)

    if extra_gps is not None:
        model_components += extra_gps

    m = likelihood.PulsarLikelihood(model_components)

    if return_components:
        return m, model_components

    return m
