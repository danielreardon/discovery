"""PPTA-DR3 single-pulsar noise model.

Reproduces the PPTA-DR3 single-pulsar noise model of
``PPTA_DR3/analysis_codes/singlePsrNoise.py`` (``dir='all'``).

Entry point: :func:`single_pulsar_noise`. Per-pulsar content is driven by
:data:`PPTA_CONFIG`.

Model conventions:

* chromatic GP index fixed at alpha = 4, DM GP at alpha = 2
* exponential dips have ``sign_param`` fixed at -1, the DM Gaussian at +1;
  neither is sampled
* solar-wind GP frequencies are log spaced
* band and group Fourier grids use the span of the selected TOAs
* group-noise frequencies are ``linspace(1/T_sel, 1/(30 d), n)``, evenly spaced
  up to a fixed fmax rather than harmonic ``i/T``
* band-noise component count comes from the band's own TOA span, so the highest
  mode sits at the nominal 60-day cadence
* ECORR is a basis GP with 1-second epoch quantisation and legacy
  degree-of-freedom accounting, applied as three simultaneous terms:
  band-split, global DR2/UWL, and per-group

Parameter names match enterprise for the white noise, ECORR and power-law GPs.
Deterministic events use the discovery argument names, so a DR3 noisedict needs
these keys renamed:

    enterprise                     this module
    ---------------------------    ------------------------------
    {psr}_dmexp_{i}_idx            {psr}_chrom_exp_{i}_alpha
    {psr}_dmgauss_epoch            {psr}_dmgauss_t0
    n_earth                        {psr}_sw_n_earth

See :mod:`discovery.models.ppta` for the PPTA-DR4 model.
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

jnp = matrix.jnp


# ---------------------------------------------------------------------------
# module options
# ---------------------------------------------------------------------------

# Backend-group matching. 'substring' selects any flag containing the group
# name, so CASPSR_40CM also selects UWL_CASPSR_40CM; 'exact' requires equality.
GROUP_MATCH = 'substring'

# Group-noise frequency grid. 'linspace' spaces n modes evenly from 1/T_sel to
# 1/(30 d); 'harmonic' uses f = i/T_sel.
GROUP_GRID = 'linspace'


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

PPTA_CONFIG = {

    # Prior ranges, registered by update_priordict_standard_ppta(). Per-pulsar
    # event epochs and indices are registered at model-build time by
    # make_psr_delays(). The white-noise key is '(.*_)?tnequad', not
    # '(.*_)?log10_tnequad': priordict_standard is scanned in insertion order
    # with re.match and the first hit wins.
    "priors": {
        '(.*_)?efac': [0.01, 10.0],
        '(.*_)?tnequad': [-10, -5],
        '(.*_)?log10_ecorr': [-10, -5],
        # power-law GPs
        '(.*_)?red_noise_log10_A': [-20, -11],
        '(.*_)?red_noise_gamma': [0, 7],
        '(.*_)?hf_noise_log10_A': [-20, -11],
        '(.*_)?hf_noise_gamma': [0, 7],
        '(.*_)?dm_gp_log10_A': [-20, -11],
        '(.*_)?dm_gp_gamma': [0, 7],
        '(.*_)?chrom_gp_log10_A': [-20, -11],
        '(.*_)?chrom_gp_gamma': [0, 7],
        r'(.*_)?band_noise_gp_(low|mid|high)_log10_A': [-20, -11],
        r'(.*_)?band_noise_gp_(low|mid|high)_gamma': [0, 7],
        r'(.*_)?group_noise_.*_gp_log10_A': [-20, -11],
        r'(.*_)?group_noise_.*_gp_gamma': [0, 7],
        # solar wind
        '(.*_)?sw_gp_log10_A': [-10, 1],
        '(.*_)?sw_gp_gamma': [-4, 4],
        '(.*_)?n_earth': [0, 20],
        # deterministic chromatic events; event epochs t0 are per-pulsar
        r'(.*_)?chrom_exp_\d+_log10_Amp': [-10, -2],
        r'(.*_)?chrom_exp_\d+_log10_tau': [0, 4],
        '(.*_)?dm1yr_log10_Amp': [-10, -2],
        '(.*_)?dm1yr_phase': [0, 2 * np.pi],
        '(.*_)?dmgauss_log10_Amp': [-10, -2],
        '(.*_)?gauss_20cm_log10_Amp': [-10, -2],
        '(.*_)?gauss_20cm_log10_sigma': [0, 3],
        '(.*_)?gauss_20cm_t0': [57385, 57785],
    },

    # Which pulsars carry each optional component.
    "models_dict": {
        "chrom": ['J0437-4715', 'J0613-0200', 'J1017-7156', 'J1045-4509',
                  'J1600-3053', 'J1643-1224', 'J1939+2134'],

        "hf_noise": ['J0437-4715', 'J1017-7156', 'J1022+1001', 'J1600-3053',
                     'J1713+0747', 'J1744-1134', 'J1909-3744', 'J2241-5236'],

        # Band noise, per band.
        "band_noise": {
            'low':  ['J0437-4715', 'J0613-0200', 'J1017-7156', 'J1045-4509',
                     'J1600-3053', 'J1643-1224', 'J1713+0747', 'J1909-3744',
                     'J1939+2134'],
            'mid':  ['J0437-4715'],
            'high': ['J0437-4715'],
        },

        # Exponential dips. 't0' is the epoch prior in MJD, 'alpha' the
        # chromatic index prior; 'gate' is the TOA-coverage requirement
        # (min(toas) < gate[0] and max(toas) > gate[1]), in MJD.
        "chrom_exp": {
            'J1713+0747': {'gate': (57500.0, 54650.0),
                           'dips': [{'t0': (54650.0, 54850.0), 'alpha': (1.0, 3.0)},
                                    {'t0': (57400.0, 57600.0), 'alpha': (0.0, 2.0)}]},
            'J0437-4715': {'gate': (57100.0, 57000.0),
                           'dips': [{'t0': (57000.0, 57200.0), 'alpha': (-1.0, 2.0)}]},
            'J1643-1224': {'gate': (57100.0, 57000.0),
                           'dips': [{'t0': (57000.0, 57200.0), 'alpha': (-2.0, 0.0)}]},
            'J2145-0750': {'gate': (56450.0, 56300.0),
                           'dips': [{'t0': (56250.0, 56450.0), 'alpha': (-2.0, 2.0)}]},
        },

        # Annual DM sinusoid, alpha fixed at 2.
        "chrom_annual": ['J0613-0200'],

        # DM Gaussian event, alpha fixed at 2, amplitude strictly positive.
        # 'gate' requires min(toas) < gate, in MJD.
        "chrom_gauss": {
            'J1603-7202': {'gate': 57500.0, 't0': (53800.0, 54000.0),
                           'log10_sigma': (0.0, 3.0)},
        },

        # Gaussian event confined to a 20 cm top-hat in observing frequency.
        "chrom_gauss_20cm": ['J1600-3053'],
        "chrom_gauss_20cm_gate": 57585.0,   # include only if min(toas) < this MJD, in MJD
    },

    # Per-backend group noise.
    "group_dict": {
        'J0437-4715': ['UWL_PDFB4_20CM', 'UWL_sbA', 'UWL_sbG', 'CASPSR_40CM'],
        'J1017-7156': ['UWL_sbA', 'UWL_sbD'],
        'J1022+1001': ['UWL_sbE', 'UWL_sbH'],
        'J1713+0747': ['UWL_sbA', 'UWL_sbE', 'UWL_sbF', 'WBCORR_10CM'],
        'J1909-3744': ['CPSR2_50CM'],
    },

    # Backends given per-group ECORR. Not the same set as group_dict.
    "ecorr_dict": {
        'J0437-4715': ['UWL_PDFB4_20CM', 'UWL_sbA', 'UWL_sbG', 'CASPSR_40CM',
                       'PDFB_20CM'],
        'J1017-7156': ['UWL_sbA', 'UWL_sbD'],
        'J1022+1001': ['UWL_sbE', 'UWL_sbH'],
        'J1713+0747': ['UWL_sbA', 'UWL_sbE', 'UWL_sbF', 'WBCORR_10CM',
                       'CPSR2_20CM'],
        'J1909-3744': ['CPSR2_50CM', 'CASPSR_40CM', 'PDFB1_1433',
                       'PDFB1_early_20CM'],
    },
}

# Cadences in days, setting each GP's component count as
# int(Tspan / (cadence * 86400)).
CADENCE_RED_CHROM = 240.0
CADENCE_DM_BAND_SW = 60.0
CADENCE_HF_GROUP = 30.0

# Band-noise edges in MHz. 'low' is inclusive at 960, 'high' inclusive at 2048
# with no upper cut.
LOW_FREQ = [960.0]
MID_FREQ = [960.0, 2048.0]
HIGH_FREQ = [2048.0]

# Band-split ECORR edges in MHz. Not the same as the band-noise edges above.
BAND_SPLIT_EDGES = (960.0, 2048.0, 4032.0)


# ---------------------------------------------------------------------------
# priors
# ---------------------------------------------------------------------------

def _register_priors(updates):
    """Insert ``updates`` into ``prior.priordict_standard`` AHEAD of existing keys.

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
    """Register the dataset-wide PPTA-DR3 priors."""
    _register_priors(config["priors"])

    return


update_priordict_standard_ppta()  # on import, and again on model creation


# ---------------------------------------------------------------------------
# selections
# ---------------------------------------------------------------------------

def selection_backend_flags(psr):
    """EFAC / EQUAD / group backends: the ``-group`` flag, matched exactly."""
    return psr.backend_flags


def band_split(psr, selection=selection_backend_flags):
    """Six-way band x (DR2, UWL) ECORR selection.

    Labels are prefixed ``basis_ecorr_`` so parameter names come out as
    ``{psr}_basis_ecorr_40CM_uwl_log10_ecorr``. Edges are inclusive at the lower
    bound and exclusive at the upper; TOAs at or above 4032 MHz are unlabelled.
    """
    backend_flags = selection(psr)
    freqs = np.asarray(psr.freqs)
    flo, fmid, fhi = BAND_SPLIT_EDGES
    uwl = np.array(['UWL' in val for val in backend_flags])

    labels = np.full(len(freqs), '', dtype='U30')
    for band, m in (('40CM', freqs < flo),
                    ('20CM', (flo <= freqs) & (freqs < fmid)),
                    ('10CM', (fmid <= freqs) & (freqs < fhi))):
        labels[m & ~uwl] = f'basis_ecorr_{band}'
        labels[m & uwl] = f'basis_ecorr_{band}_uwl'

    return labels


def global_ecorr(psr, selection=selection_backend_flags):
    """Two-way DR2 / UWL ECORR selection."""
    backend_flags = selection(psr)
    uwl = np.array(['UWL' in val for val in backend_flags])

    labels = np.full(len(backend_flags), '', dtype='U30')
    labels[~uwl] = 'basis_ecorr_all_dr2'
    labels[uwl] = 'basis_ecorr_all_uwl'

    return labels


def group_masks(psr, groups, match=None):
    """Masks for a list of backend-group names.

    ``match='substring'`` selects any flag containing the name, so
    ``CASPSR_40CM`` also picks up ``UWL_CASPSR_40CM``; ``match='exact'``
    requires equality. Defaults to :data:`GROUP_MATCH`. Groups selecting no TOAs
    are dropped with a warning.
    """
    match = GROUP_MATCH if match is None else match
    flags = np.asarray(psr.backend_flags)

    masks = {}
    for g in sorted(set(groups)):
        m = (np.array([g in f for f in flags]) if match == 'substring'
             else np.asarray(flags == g))
        if m.any():
            masks[g] = m
        else:
            print(f'Warning: group {g!r} selects no TOAs for {psr.name}; dropped.')

    return masks


def _make_selection(labels):
    """Wrap a precomputed label array as a discovery selection function."""
    labels = np.asarray(labels, dtype=str)

    def selection(psr):
        return labels

    return selection


def _make_mask_selection(mask):
    """Wrap a precomputed boolean mask as a selection function."""
    mask = np.asarray(mask, dtype=bool)

    def selection(psr):
        return mask

    return selection


def _drop_ecorr_backends_without_epochs(psr, labels):
    """Blank out backends with no multi-TOA epoch.

    Such a backend yields a zero-column ECORR block, so its ``log10_ecorr``
    parameter cannot affect the likelihood.
    """
    labels = np.asarray(labels, dtype=str)
    keep = set()
    for b in sorted(set(labels) - {''}):
        bins = signals.quantize(psr.toas * (labels == b))
        uniques, counts = np.unique(bins, return_counts=True)
        if any(c > 1 for c in counts[1:]):
            keep.add(b)

    dropped = sorted((set(labels) - {''}) - keep)
    if dropped:
        print(f'{psr.name}: ECORR backends with no repeated epoch, dropped: {dropped}')

    return np.array([b if b in keep else '' for b in labels])


def make_band_selection(band_range):
    """Low / mid / high band-noise selection."""
    def selection(psr):
        freqs = np.asarray(psr.freqs)
        if band_range == LOW_FREQ:
            return freqs <= band_range[0]
        if band_range == HIGH_FREQ:
            return freqs >= band_range[0]
        if len(band_range) == 2:
            return (freqs > band_range[0]) & (freqs < band_range[1])
        raise ValueError(f'Invalid band range {band_range!r}')

    return selection


# ---------------------------------------------------------------------------
# ECORR
# ---------------------------------------------------------------------------

def makegp_band_ecorr(psr, noisedict={}):
    """Six-way band x (DR2, UWL) ECORR."""
    labels = _drop_ecorr_backends_without_epochs(psr, band_split(psr))

    return [signals.makegp_ecorr(psr, noisedict=noisedict, enterprise=True,
                                 selection=_make_selection(labels),
                                 name='ecorr_band')]


def makegp_global_ecorr(psr, noisedict={}):
    """DR2 / UWL global ECORR. Returns a list, empty if no group qualifies."""
    labels = _drop_ecorr_backends_without_epochs(psr, global_ecorr(psr))
    if not set(labels) - {''}:
        print(f'{psr.name}: no global-ECORR group has a repeated epoch; skipped.')
        return []

    return [signals.makegp_ecorr(psr, noisedict=noisedict, enterprise=True,
                                 selection=_make_selection(labels),
                                 name='ecorr_all')]


def makegp_group_ecorr(psr, groups, noisedict={}):
    """Per-group ECORR for the listed backend groups."""
    masks = group_masks(psr, groups)
    if not masks:
        return []

    labels = np.full(len(psr.toas), '', dtype=object)
    for g, m in masks.items():
        labels = np.where(m, f'basis_ecorr_group_{g}', labels)
    labels = _drop_ecorr_backends_without_epochs(psr, labels)
    if not set(labels) - {''}:
        return []

    return [signals.makegp_ecorr(psr, noisedict=noisedict, enterprise=True,
                                 selection=_make_selection(labels),
                                 name='ecorr_group')]


# ---------------------------------------------------------------------------
# masked Fourier bases
# ---------------------------------------------------------------------------

def masked_fourierbasis(selection, base=signals.fourierbasis):
    """Fourier basis restricted to ``selection``, with T from the selected TOAs.

    Rows outside the mask are zero. When ``T`` is not supplied the frequency
    grid uses the span of the selected TOAs.
    """
    def basis(psr, components, T=None):
        m = np.asarray(selection(psr), dtype=bool)
        if T is None:
            sel = psr.toas[m]
            T = float(sel.max() - sel.min())
        f, df, fmat = base(psr, components, T)

        return f, df, fmat * m[:, None]

    return basis


def make_group_fourierbasis(selection, fmax, grid=None):
    """Masked group-noise basis over the group's own TOA span.

    ``grid='linspace'`` spaces ``components`` modes evenly from ``1/T`` to
    ``fmax``; ``'harmonic'`` uses ``f = i/T``. Defaults to :data:`GROUP_GRID`.
    ``df`` follows ``diff([0, f_1, f_2, ...])``, so the first bin width is
    ``f_1``.
    """
    grid = GROUP_GRID if grid is None else grid

    def basis(psr, components, T=None):
        m = np.asarray(selection(psr), dtype=bool)
        sel = psr.toas[m]
        Tsel = float(sel.max() - sel.min()) if T is None else float(T)

        n = int(components)
        if grid == 'linspace':
            f = np.linspace(1.0 / Tsel, fmax, n)
        else:
            f = np.arange(1, n + 1, dtype=np.float64) / Tsel
        df = np.diff(np.concatenate((np.array([0.0]), f)))

        fmat = np.zeros((psr.toas.shape[0], 2 * n), dtype=np.float64)
        for i in range(n):
            fmat[:, 2 * i] = np.sin(2.0 * np.pi * f[i] * psr.toas)
            fmat[:, 2 * i + 1] = np.cos(2.0 * np.pi * f[i] * psr.toas)

        return np.repeat(f, 2), np.repeat(df, 2), fmat * m[:, None]

    return basis


# ---------------------------------------------------------------------------
# group and band GPs
# ---------------------------------------------------------------------------

def make_group_gps_fourier(psr, group_cadence=CADENCE_HF_GROUP, name='group_noise',
                           group_dict=None):
    """Per-backend-group achromatic power-law GPs."""
    group_dict = PPTA_CONFIG["group_dict"] if group_dict is None else group_dict
    fmax = 1.0 / (group_cadence * 86400.0)

    gps = []
    for g, m in group_masks(psr, group_dict.get(psr.name, [])).items():
        sel = psr.toas[m]
        Tsel = float(sel.max() - sel.min())
        n = int(Tsel / (group_cadence * 86400.0))
        if n < 1:
            print(f'Warning: group {g!r} for {psr.name} spans '
                  f'{Tsel / 86400.0:.0f} d (< {group_cadence:.0f} d); skipped.')
            continue

        gps += [signals.makegp_fourier(
            psr, signals.powerlaw, components=n,
            fourierbasis=make_group_fourierbasis(_make_mask_selection(m), fmax),
            name=f'{name}_{g}_gp')]

    return gps


def make_band_gps_fourier(psr, band_cadence=CADENCE_DM_BAND_SW,
                          name='band_noise', band_dict=None):
    """Low / mid / high band-noise GPs, per the DR3 per-band pulsar lists.

    Both the component count and the frequency grid come from the span of the
    band's own TOAs, so the highest mode sits at the nominal ``band_cadence``.
    """
    band_dict = (PPTA_CONFIG["models_dict"]["band_noise"] if band_dict is None
                 else band_dict)

    gps = []
    for band_name, band_range in (('low', LOW_FREQ), ('mid', MID_FREQ),
                                  ('high', HIGH_FREQ)):
        if psr.name not in band_dict.get(band_name, []):
            continue

        sel = make_band_selection(band_range)
        m = np.asarray(sel(psr), dtype=bool)
        if not m.any():
            print(f'Warning: band {band_name!r} selects no TOAs for {psr.name}; skipped.')
            continue

        tb = psr.toas[m]
        n = int((tb.max() - tb.min()) / (band_cadence * 86400.0))
        if n < 1:
            print(f'Warning: band {band_name!r} for {psr.name} spans '
                  f'{(tb.max() - tb.min()) / 86400.0:.0f} d '
                  f'(< {band_cadence:.0f} d); skipped.')
            continue

        gps += [signals.makegp_fourier(
            psr, signals.powerlaw, components=n,
            fourierbasis=masked_fourierbasis(sel),
            name=f'{name}_gp_{band_name}')]

    return gps


# ---------------------------------------------------------------------------
# deterministic delays, with the enterprise sign conventions
# ---------------------------------------------------------------------------

def chromatic_exponential_dip(psr, alpha=None):
    """Exponential dip with ``sign_param`` fixed at -1, so it is not sampled."""
    base = deterministic.chromatic_exponential(psr, alpha=alpha)

    if alpha is None:
        def delay(t0, log10_Amp, log10_tau, alpha):
            return base(t0, log10_Amp, log10_tau, -1.0, alpha)
    else:
        def delay(t0, log10_Amp, log10_tau):
            return base(t0, log10_Amp, log10_tau, -1.0)

    delay.__name__ = 'chromatic_exponential_dip_delay'

    return delay


def chromatic_gaussian_positive(psr, alpha=None):
    """Gaussian event with ``sign_param`` fixed at +1, so the amplitude is
    strictly positive and the sign is not sampled."""
    base = deterministic.chromatic_gaussian(psr, alpha=alpha)

    if alpha is None:
        def delay(t0, log10_Amp, log10_sigma, alpha):
            return base(t0, log10_Amp, log10_sigma, 1.0, alpha)
    else:
        def delay(t0, log10_Amp, log10_sigma):
            return base(t0, log10_Amp, log10_sigma, 1.0)

    delay.__name__ = 'chromatic_gaussian_positive_delay'

    return delay


def chromatic_gaussian_20cm(psr, nu1=1000.0, nu2=2000.0):
    """Gaussian event confined to a 20 cm top-hat in observing frequency.

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


def _dip_gate_ok(psr, gate):
    """TOA-coverage gate: ``min(toas) < gate[0]`` and ``max(toas) > gate[1]``."""
    tmin, tmax = psr.toas.min() / 86400.0, psr.toas.max() / 86400.0

    return (tmin < gate[0]) and (tmax > gate[1])


def make_psr_delays(psr, config=PPTA_CONFIG, mean_sw=True, chrom_exp=None,
                    chrom_annual=None, chrom_gauss=None, chrom_gauss_20cm=None):
    """Deterministic PPTA-DR3 delays, registering the per-pulsar priors they need.

    Each per-pulsar switch defaults to ``None``, meaning "use the configuration
    for this pulsar"; an explicit value overrides it.
    """
    md = config["models_dict"]

    chrom_exp = (psr.name in md["chrom_exp"]) if chrom_exp is None else chrom_exp
    chrom_annual = ((psr.name in md["chrom_annual"]) if chrom_annual is None
                    else chrom_annual)
    chrom_gauss = ((psr.name in md["chrom_gauss"]) if chrom_gauss is None
                   else chrom_gauss)
    chrom_gauss_20cm = ((psr.name in md["chrom_gauss_20cm"])
                        if chrom_gauss_20cm is None else chrom_gauss_20cm)

    delays, updates = [], {}
    key = re.escape(psr.name)

    if mean_sw:
        delays += [signals.makedelay(psr, solar.make_solardm(psr), name='sw')]

    if chrom_exp and psr.name in md["chrom_exp"]:
        entry = md["chrom_exp"][psr.name]
        if _dip_gate_ok(psr, entry['gate']):
            for i, dip in enumerate(entry['dips'], start=1):
                nm = f'chrom_exp_{i}'
                updates[f'{key}_{nm}_t0'] = list(dip['t0'])
                updates[f'{key}_{nm}_alpha'] = list(dip['alpha'])
                delays += [signals.makedelay(psr, chromatic_exponential_dip(psr),
                                             name=nm)]
        else:
            print(f'{psr.name}: TOA span fails the dip gate {entry["gate"]}; '
                  'no dips added.')

    if chrom_annual:
        # alpha FIXED at 2 (enterprise idx=2): a dispersive annual term.
        delays += [signals.makedelay(
            psr, deterministic.chromatic_annual(psr, alpha=2.0), name='dm1yr')]

    if chrom_gauss and psr.name in md["chrom_gauss"]:
        entry = md["chrom_gauss"][psr.name]
        if psr.toas.min() / 86400.0 < entry['gate']:
            updates[f'{key}_dmgauss_t0'] = list(entry['t0'])
            updates[f'{key}_dmgauss_log10_sigma'] = list(entry['log10_sigma'])
            # alpha FIXED at 2, amplitude strictly positive.
            delays += [signals.makedelay(
                psr, chromatic_gaussian_positive(psr, alpha=2.0), name='dmgauss')]
        else:
            print(f'{psr.name}: earliest TOA is after MJD {entry["gate"]}; '
                  'no DM Gaussian added.')

    if chrom_gauss_20cm:
        gate = md.get("chrom_gauss_20cm_gate", None)
        if gate is None or psr.toas.min() / 86400.0 < gate:
            delays += [signals.makedelay(psr, chromatic_gaussian_20cm(psr),
                                         name='gauss_20cm')]
        else:
            print(f'{psr.name}: earliest TOA is after MJD {gate}; '
                  'no 20 cm Gaussian bump added.')

    if updates:
        _register_priors(updates)

    return delays


# ---------------------------------------------------------------------------
# GP block
# ---------------------------------------------------------------------------

def make_psr_gps_fourier(psr, red_chrom_cadence_days=CADENCE_RED_CHROM,
                         dm_band_sw_cadence_days=CADENCE_DM_BAND_SW,
                         hf_group_cadence_days=CADENCE_HF_GROUP,
                         Tspan=None, config=PPTA_CONFIG,
                         red=True, hf=None, dm=True, chrom=None, chrom_alpha=4.0,
                         sw=True, sw_gp=True, sw_logf=True,
                         band=None, group=None, group_dict=None):
    """Build the DR3 Fourier GPs for one pulsar.

    ``hf``, ``chrom``, ``band`` and ``group`` default to ``None``, meaning "use
    the configuration for this pulsar"; an explicit value overrides it.
    """
    md = config["models_dict"]
    group_dict = config["group_dict"] if group_dict is None else group_dict

    psr_Tspan = signals.getspan(psr) if Tspan is None else Tspan
    n_red_chrom = int(psr_Tspan / (red_chrom_cadence_days * 86400.0))
    n_dm_band_sw = int(psr_Tspan / (dm_band_sw_cadence_days * 86400.0))
    n_hf = int(psr_Tspan / (hf_group_cadence_days * 86400.0))

    hf = (psr.name in md["hf_noise"]) if hf is None else hf
    chrom = (psr.name in md["chrom"]) if chrom is None else chrom
    band = (any(psr.name in v for v in md["band_noise"].values())
            if band is None else band)
    group = (psr.name in group_dict) if group is None else group

    gps = []

    if red:
        gps += [signals.makegp_fourier(psr, signals.powerlaw,
                                       components=n_red_chrom, T=psr_Tspan,
                                       name='red_noise')]
    if hf:
        # Second achromatic power law on a denser grid. The hf and red
        # cadences must differ, or the two GPs share a design matrix.
        gps += [signals.makegp_fourier(psr, signals.powerlaw, components=n_hf,
                                       T=psr_Tspan, name='hf_noise')]
    if dm:
        gps += [signals.makegp_fourier(psr, signals.powerlaw,
                                       components=n_dm_band_sw, T=psr_Tspan,
                                       fourierbasis=signals.fourierbasis_dm,
                                       name='dm_gp')]
    if chrom:
        # A float alpha makes fourierbasis_chrom return a constant matrix, so
        # no chrom_gp_alpha parameter is created.
        gps += [signals.makegp_fourier(psr, signals.powerlaw,
                                       components=n_red_chrom, T=psr_Tspan,
                                       fourierbasis=signals.fourierbasis_chrom,
                                       name='chrom_gp', alpha=chrom_alpha)]
    if sw and sw_gp:
        # Log-spaced frequency grid.
        gps += [signals.makegp_fourier(
            psr, signals.powerlaw, components=n_dm_band_sw, T=psr_Tspan,
            fourierbasis=solar.make_fourierbasis_solar_dm(logf=sw_logf),
            name='sw_gp')]
    if band:
        gps += make_band_gps_fourier(psr, band_cadence=dm_band_sw_cadence_days,
                                     band_dict=md["band_noise"])
    if group:
        gps += make_group_gps_fourier(psr, group_cadence=hf_group_cadence_days,
                                      group_dict=group_dict)

    return gps


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def single_pulsar_noise(psr,
                        red_chrom_cadence_days=CADENCE_RED_CHROM,
                        dm_band_sw_cadence_days=CADENCE_DM_BAND_SW,
                        hf_group_cadence_days=CADENCE_HF_GROUP,
                        Tspan=None, noisedict={}, outliers=False,
                        PPTA_band_ecorr=True, PPTA_global_ecorr=True,  # the three simultaneous DR3 ECORR terms
                        PPTA_ecorr_backend=True, ecorr_dict=None,
                        red=True, dm=True, chrom=None, chrom_alpha=4.0, hf=None,  # power-law GPs
                        mean_sw=True, sw_gp=True, sw_logf=True,
                        band=None, group=None, group_dict=None,
                        chrom_exp=None, chrom_annual=None, chrom_gauss=None,  # deterministic events
                        chrom_gauss_20cm=None,
                        config=PPTA_CONFIG, extra_gps=None,
                        return_components=False):
    """Build the PPTA-DR3 single-pulsar noise likelihood.

    Every per-pulsar switch (``chrom``, ``hf``, ``band``, ``group``,
    ``chrom_exp``, ``chrom_annual``, ``chrom_gauss``, ``chrom_gauss_20cm``)
    defaults to ``None``, meaning "use the configuration for this pulsar". An
    explicit value overrides it.

    Parameters
    ----------
    psr : discovery.Pulsar
    Tspan : float, optional
        Span in seconds setting component counts and frequency grids. Defaults
        to the pulsar's own span.
    noisedict : dict, optional
        Fixed values for white-noise and ECORR parameters.
    PPTA_band_ecorr, PPTA_global_ecorr, PPTA_ecorr_backend : bool, optional
        The three simultaneous ECORR terms.
    return_components : bool, optional
        Also return the list of model components.

    Returns
    -------
    discovery.likelihood.PulsarLikelihood
    """
    update_priordict_standard_ppta(config)

    ecorr_dict = config["ecorr_dict"] if ecorr_dict is None else ecorr_dict

    model_components = [psr.residuals]
    model_components += [signals.makegp_timing(psr, svd=True)]
    # EFAC + TempoNest EQUAD split by the 'group' backend flag:
    #   N_ii = efac^2 sigma_i^2 + 10^(2 log10_tnequad)
    # The three ECORR terms are added below as separate basis GPs.
    model_components += [signals.makenoise_measurement(
        psr, tnequad=True, noisedict=noisedict, outliers=outliers)]

    if PPTA_band_ecorr:
        model_components += makegp_band_ecorr(psr, noisedict=noisedict)
    if PPTA_global_ecorr:
        model_components += makegp_global_ecorr(psr, noisedict=noisedict)
    if PPTA_ecorr_backend and psr.name in ecorr_dict:
        model_components += makegp_group_ecorr(psr, ecorr_dict[psr.name],
                                               noisedict=noisedict)

    model_components += make_psr_delays(psr, config=config, mean_sw=mean_sw,
                                        chrom_exp=chrom_exp,
                                        chrom_annual=chrom_annual,
                                        chrom_gauss=chrom_gauss,
                                        chrom_gauss_20cm=chrom_gauss_20cm)

    model_components += make_psr_gps_fourier(
        psr, red_chrom_cadence_days=red_chrom_cadence_days,
        dm_band_sw_cadence_days=dm_band_sw_cadence_days,
        hf_group_cadence_days=hf_group_cadence_days,
        Tspan=Tspan, config=config, red=red, hf=hf, dm=dm, chrom=chrom,
        chrom_alpha=chrom_alpha, sw=True, sw_gp=sw_gp, sw_logf=sw_logf,
        band=band, group=group, group_dict=group_dict)

    if extra_gps is not None:
        model_components += extra_gps

    m = likelihood.PulsarLikelihood(model_components)

    if return_components:
        return m, model_components

    return m
