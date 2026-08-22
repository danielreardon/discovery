import numpy as np
import jax.numpy as jnp
import re

from .. import matrix
from .. import signals
from .. import prior
from .. import solar
from .. import likelihood
from .. import deterministic
from .. import phys_ephem as phys_ephem_mod
from .. import const

def write_ml_json(df, savename):
    import json
    ml_idx = df['logl'].idxmax()
    ml_params = df.loc[ml_idx].to_dict()
    with open(savename, 'w') as f:
        json.dump(ml_params, f, indent=2)
    return

def update_priordict_standard_mpta():
    # Update the standard prior dictionary with PTA-specific parameters
    prior.priordict_standard.update({
        # White noise parameters
        '(.*_)?efac':               [0.5, 2],
        '(.*_)?log10_tnequad':      [-10, -5],
        '(.*_)?log10_ecorr_q.*':    [-10, -5],
        '(.*_)?log10_ecorr':        [-10, -5],
        # Per-pulsar GW background parameters
        '(.*_)?bkgrnd_log10_A':     [-18, -11],
        # GP parameters
        '(.*_)?red_noise_log10_A.*':  [-18, -11],
        '(.*_)?red_noise_gamma.*':    [0, 7],
        '(.*_)?red_noise2_log10_A.*':  [-18, -11],
        '(.*_)?red_noise2_gamma.*':    [0, 7],
        '(.*_)?dm_gp_log10_A':      [-18, -11],
        '(.*_)?dm_gp_gamma':        [0, 7],
        '(.*_)?chrom_gp_log10_A':   [-20, -11], # -20 minimum is "effectively zero" at alpha=10
        '(.*_)?chrom_gp_gamma':     [0, 7],
        '(.*_)?chrom_gp_alpha':     [3.0, 10], # start at 3 to avoid confusion with DM.
        '(.*_)?sw_gp_log10_A':      [-10, -2],
        '(.*_)?sw_gp_gamma':        [0, 4],
        # SE kernel for time-domain SW GP
        # sigma : rms electron density variability (cm^-3)
        # ell   : correlation timescale (days)
        '(.*_)?sw_gp_log10_sigma':  [ -2,  1.3],   # 0.01 - 20 cm^-3
        '(.*_)?sw_gp_log10_ell':    [  1,  4],   # 10 days - ~30 yr
        # QP kernel adds:
        '(.*_)?sw_gp_log10_Gamma':  [ -3,  2],   # dimensionless
        '(.*_)?sw_gp_log10_p':      [ -2,  1.3],   # years (0.01 - 20 yr)
        '(.*_)?band_gp_log10_A':    [-18, -11],
        '(.*_)?band_gp_gamma':      [0, 7],
        '(.*_)?bandalpha_gp_log10_A':    [-18, -11],
        '(.*_)?bandalpha_gp_gamma':      [0, 7],
        '(.*_)?bandalpha_gp_alpha':      [0, 10],
        # The band GP centre/bandwidth (fcenter, log10_bw) priors are NOT set here:
        # they are bounded per-pulsar from psr.freqs by _set_band_priors() at model-build
        # time. A generic regex fallback would shadow those per-pulsar entries
        # (getprior_uniform returns the first matching pattern), so it is omitted. The old
        # (flow, fhigh, fcutoff) edge parametrisation is gone -- band/band_alpha now use the
        # robust fcenter+log10_bw band models.
        # common noise parameters
        'curn_log10_A':             [-18, -11],
        'curn_gamma':               [0, 7],
        # per-pulsar forms, used only when common_noise(curn_per_pulsar=True)
        '(.*_)?curn_log10_A':       [-18, -11],
        '(.*_)?curn_gamma':         [0, 7],
        'gw_log10_A':             [-18, -11],
        'gw_gamma':               [0, 7],
        # deterministic parameters. chrom_exp_t0, _log10_tau and _log10_Amp are set
        # per-pulsar from the data by _set_chrom_exp_priors and have no entry here.
        '(.*_)?chrom_exp_sign_param': [-1, 1],
        '(.*_)?chrom_exp_alpha': [0, 7],
        '(.*_)?chrom_1yr_log10_Amp': [-10, -4],
        '(.*_)?chrom_1yr_phase': [0, 2 * np.pi],
        '(.*_)?chrom_1yr_alpha': [0, 7],
        '(.*_)?chrom_gauss_t0': [58525, 60900], # MPTA 6-yr range
        '(.*_)?chrom_gauss_log10_Amp': [-10, -4],
        '(.*_)?chrom_gauss_log10_sigma': [0.5, 4],
        '(.*_)?chrom_gauss_sign_param': [-1, 1],
        '(.*_)?chrom_gauss_alpha': [0, 7],
        r'(.*_)?chrom_sphere_t0': [58525, 60900],
        r'(.*_)?chrom_sphere_log10_Amp': [-10, -4],
        r'(.*_)?chrom_sphere_log10_tau': [1.0, 4.0],
        r'(.*_)?chrom_sphere_sign_param': [-1, 1],
        r'(.*_)?chrom_sphere_alpha': [0, 10],
        r'(.*_)?chrom_sphere_smooth': [10, 200],
        r'(.*_)?chrom_step_t0': [58525, 60900],
        r'(.*_)?chrom_step_log10_Amp': [-10, -4],
        r'(.*_)?chrom_step_log10_span': [1.0, 4.0],
        r'(.*_)?chrom_step_sign_param': [-1, 1],
        r'(.*_)?chrom_step_alpha': [0, 10],
        r'(.*_)?chrom_step_smooth': [10, 200], 
        r'(.*_)?timingmodel_coefficients\(\d+\)': [-20.0, 20.0],
        r'(.*_)?alpha_scaling\(\d+\)': [0.0, 100.0],
        r'(.*_)?h3': [1e-10, 1e-5],
        r'(.*_)?stig': [1e-6, 1.0 - 1e-6], # clip to avoid singularities
        r'(.*_)?cosi': [1e-6, 1.0 - 1e-6], # clip to avoid singularities
        r'(.*_)?orbital_dm_amp': [0.0, 1e-3],        # pc cm^-3
        r'(.*_)?orbital_dm_phi0': [-1.0, 1.0],           # radians
        r'(.*_)?orbital_dm_sigma_phi': [0.0, 0.5],      # radians
        r'(.*_)?orbital_dm_fourier_cos\d+': [-1e-4, 1e-4],
        r'(.*_)?orbital_dm_fourier_sin\d+': [-1e-4, 1e-4],
        r'(.*_)?orbital_dm_gp_log10_A': [-12, -4],
        r'(.*_)?orbital_dm_gp_gamma': [0, 7],
        r"(.*_)?chrom_gp_CM0": [-1e-4, 1e-4], # <100us at 1.4 GHz
        r"(.*_)?chrom_gp_CM1": [-1e-5, 1e-5], # <10us/yr at 1.4 GHz
        r"(.*_)?chrom_gp_CM2": [-1e-6, 1e-6], # <1us/yr^2 at 1.4 GHz
        r"(.*_)?chrom_gp_c0": [-1e-4, 1e-4],
        r"(.*_)?chrom_gp_c1": [-1e-4, 1e-4],
        r"(.*_)?chrom_gp_c2": [-1e-4, 1e-4],
    })

    return

update_priordict_standard_mpta() # Ensure priordict_standard is updated on import, but also update when a model is created to catch any changes during likelihood/prior initialisation

def gps2commongp(gps):
    """Pad/stack a list of per-pulsar GPs (each a ``matrix.CompoundGP`` of that
    pulsar's sampled GPs, or a single ``matrix.VariableGP``) into one common GP
    for ``ArrayLikelihood(commongp=...)``.

    Handles BOTH the 1D (diagonal ``Phi``, from ``makegp_fourier`` power-law GPs)
    and the 2D (dense Toeplitz ``Phi``, from ``makegp_fftcov`` GPs) cases. The
    per-pulsar ``CompoundGP`` result is exactly one of ``NoiseMatrix1D_var`` or
    ``NoiseMatrix2D_var``, so we dispatch on that. Pulsars are assumed homogeneous
    (all 1D or all 2D); this is asserted.

    In both cases the padded (unused) basis dimensions are given a tiny prior
    variance (1e-40) and zero design columns, which is exactly neutral to the
    likelihood: |P| |P^-1 + F^T N^-1 F| contributions from the padded block cancel
    and the zero F columns contribute nothing to the quadratic form.
    """
    is2d = [isinstance(gp.Phi, matrix.NoiseMatrix2D_var) for gp in gps]
    if any(is2d) and not all(is2d):
        raise ValueError(
            "gps2commongp: mixed 1D/2D per-pulsar GP priors are not supported; all "
            "pulsars must be homogeneous (all Fourier/1D or all fftcov/2D). Got "
            f"{sum(is2d)}/{len(is2d)} pulsars with 2D (dense) priors.")

    priors = [gp.Phi.getN for gp in gps]
    pmax = len(gps)
    ns = [gp.F.shape[1] for gp in gps]  # requires non-callable gp.F (fix_chrom_alpha=True)
    nmax = max(ns)

    prior_params = sorted(set([par for p in priors for par in p.params]))
    Fs = [np.pad(gp.F, [(0, 0), (0, nmax - gp.F.shape[1])]) for gp in gps]

    if not any(is2d):
        # 1D (diagonal) branch -- identical to the original behaviour
        def prior(params):
            yp = matrix.jnp.full((pmax, nmax), 1e-40)
            for i, p in enumerate(priors):
                yp = yp.at[i, :ns[i]].set(p(params))
            return yp

        prior.params = prior_params
        return matrix.VariableGP(matrix.VectorNoiseMatrix1D_var(prior), Fs)

    # 2D (dense) branch: build a (pmax, nmax, nmax) batched covariance, placing each
    # pulsar's (ns_i, ns_i) dense block top-left and padding the remaining diagonal
    # with 1e-40 so the matrix stays positive-definite / invertible.
    diag = matrix.jnp.arange(nmax)

    def prior(params):
        yp = matrix.jnp.zeros((pmax, nmax, nmax))
        yp = yp.at[:, diag, diag].set(1e-40)
        for i, p in enumerate(priors):
            yp = yp.at[i, :ns[i], :ns[i]].set(p(params))
        return yp

    prior.params = prior_params
    return matrix.VariableGP(matrix.VectorNoiseMatrix2D_var(prior), Fs)


def _set_band_priors(psr, band=False, band_alpha=False, bw_min_mhz=20.0):
    """Set data-bounded per-pulsar priors for the (fcenter, log10_bw) band GPs.

    The band centre is bounded by the pulsar's actual frequency coverage and the
    bandwidth runs from ``bw_min_mhz`` to the full coverage span, so the band always
    overlaps data and can never collapse to an empty envelope.
    """
    freqs = np.asarray(psr.freqs)
    fmin, fmax = float(freqs.min()), float(freqs.max())
    span = max(fmax - fmin, 2.0 * bw_min_mhz)
    names = ([] + (['band_gp'] if band else []) + (['bandalpha_gp'] if band_alpha else []))
    updates = {}
    for n in names:
        psr_key = re.escape(psr.name)
        updates[f'{psr_key}_{n}_fcenter'] = [fmin, fmax]
        updates[f'{psr_key}_{n}_log10_bw'] = [float(np.log10(bw_min_mhz)), float(np.log10(span))]
    prior.priordict_standard.update(updates)


def _set_chrom_exp_priors(psr, chrom_exponential=False, tau_min_days=10.0,
                          log10_amp_max=-5.0):
    """Set data-bounded per-pulsar priors for the chromatic exponential event.

    The event epoch is bounded by the pulsar's own observing span and the decay
    timescale runs from ``tau_min_days`` to that span, so the event always overlaps
    data and cannot decay over an interval the data do not constrain.

    The amplitude is capped at the peak-to-peak of the pulsar's residuals, an event
    larger than which would be visible in the raw timing, or at ``log10_amp_max``
    where that is tighter. An event far above the residual scale drives the
    log-likelihood to ~1e8, which float32 resolves only to ~10 and which sends a
    gradient-based variational fit to nan.

    These three parameters have no entry in the MPTA prior dictionary; this helper
    is their MPTA source. ``prior.priordict_standard`` still carries wide generic
    fallbacks, which apply only if a model is built without calling this.

    The per-pulsar keys are inserted AHEAD of the existing entries: ``_matchprior``
    returns the first regex match in insertion order, and the generic
    ``(.*_)?chrom_exp_*`` defaults are already present, so a key appended after them
    would never be reached. (``_set_band_priors`` can append because the band
    parameters have no generic default.)
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


def _chrom_poly_noisedict(psr, chrom_alpha):
    """Noisedict fixing the chromatic polynomial's alpha, or empty if it is sampled.

    The polynomial GP shares {psr}_chrom_gp_alpha with the chromatic Fourier GP, so it
    takes the same value: a fixed alpha makes its basis constant, which turns it into a
    ConstantGP folded into the precomputed part of the kernel rather than a callable
    design matrix rebuilt on every likelihood call.
    """
    if chrom_alpha is None:
        return {}
    return {f'{psr.name}_chrom_gp_alpha': float(chrom_alpha)}


def make_psr_gps_fourier(psr, max_cadence_days=14, bkgrnd_log10_A=None, Tspan=None, background=True, red=True, red2=False, dm=True, chrom=True, chrom_alpha=None, chrom_poly=True, sw=True, sw_powerlaw=False, sw_qp=False, sw_logf=False, band=False, band_alpha=False, band_bw_min=20.0, fd_gp=None):
    psr_Tspan = signals.getspan(psr) if Tspan is None else Tspan
    psr_components = int(psr_Tspan / (max_cadence_days * 86400))
    _set_band_priors(psr, band=band, band_alpha=band_alpha, bw_min_mhz=band_bw_min)

    return (([signals.makegp_fourier(psr, signals.powerlaw_gwb(log10_A=bkgrnd_log10_A), components=psr_components, T=psr_Tspan, name='bkgrnd')] if background else []) + \
            ([signals.makegp_fourier(psr, signals.powerlaw, components=psr_components, T=psr_Tspan, name='red_noise')] if red else []) + \
            ([signals.makegp_fourier(psr, signals.powerlaw, components=psr_components, T=psr_Tspan, name='red_noise2')] if red2 else []) + \
            ([signals.makegp_fourier(psr, signals.powerlaw, components=psr_components, T=psr_Tspan, fourierbasis=signals.fourierbasis_dm, name='dm_gp')] if dm else [])+ \
            ([signals.makegp_fourier(psr, signals.powerlaw, components=psr_components, T=psr_Tspan, fourierbasis=signals.fourierbasis_chrom, name='chrom_gp', alpha=chrom_alpha)] if chrom else [])+ \
            ([signals.makegp_chrom_poly_svd(psr, name='chrom_gp', project=fd_gp, noisedict=_chrom_poly_noisedict(psr, chrom_alpha))] if chrom_poly else []) + \
            # Solar wind: time-domain GP by default, quasi-periodic when sw_qp=True and squared-exponential otherwise, or the power-law (Fourier) GP when sw_powerlaw=True (legacy treatment).
            ([solar.makegp_timedomain_solar_dm(psr, covariance=(signals.quasi_periodic if sw_qp else signals.squared_exponential), dt=max_cadence_days*86400.0, name='sw_gp')] if (sw and not sw_powerlaw) else []) + \
            ([signals.makegp_fourier(psr, signals.powerlaw, components=psr_components, T=psr_Tspan, fourierbasis=solar.make_fourierbasis_solar_dm(logf=sw_logf), name='sw_gp')] if (sw and sw_powerlaw) else []) + \
            ([signals.makegp_fourier(psr, signals.powerlaw, components=psr_components, T=psr_Tspan, fourierbasis=signals.fourierbasis_band, name='band_gp')] if band else []) + \
            ([signals.makegp_fourier(psr, signals.powerlaw, components=psr_components, T=psr_Tspan, fourierbasis=signals.fourierbasis_band_alpha, name='bandalpha_gp')] if band_alpha else []))


def make_psr_gps_fftint(psr, max_cadence_days=14, bkgrnd_log10_A=None, Tspan=None, background=True, red=True, red2=False, dm=True, chrom=True, chrom_alpha=None, chrom_poly=True, sw=True, sw_powerlaw=False, sw_qp=False, band=False, band_alpha=False, band_bw_min=20.0, fd_gp=None):
    psr_Tspan = signals.getspan(psr) if Tspan is None else Tspan
    psr_components = int(psr_Tspan / (max_cadence_days * 86400))
    psr_knots = 2 * psr_components + 1
    _set_band_priors(psr, band=band, band_alpha=band_alpha, bw_min_mhz=band_bw_min)

    return (([signals.makegp_fftcov(psr, signals.powerlaw_gwb(log10_A=bkgrnd_log10_A), components=psr_knots, T=psr_Tspan, name='bkgrnd')] if background else []) + \
            ([signals.makegp_fftcov(psr, signals.powerlaw, components=psr_knots, T=psr_Tspan, name='red_noise')] if red else []) + \
            ([signals.makegp_fftcov(psr, signals.powerlaw, components=psr_knots, T=psr_Tspan, name='red_noise2')] if red2 else []) + \
            ([signals.makegp_fftcov_dm(psr, signals.powerlaw, components=psr_knots, T=psr_Tspan, name='dm_gp')] if dm else [])+ \
            ([signals.makegp_fftcov_chrom(psr, signals.powerlaw, components=psr_knots, T=psr_Tspan, name='chrom_gp', alpha=chrom_alpha)] if chrom else [])+ \
            ([signals.makegp_chrom_poly_svd(psr, name='chrom_gp', project=fd_gp, noisedict=_chrom_poly_noisedict(psr, chrom_alpha))] if chrom_poly else []) + \
            # Solar wind: time-domain GP by default, quasi-periodic when sw_qp=True and squared-exponential otherwise, or the power-law (FFT-covariance) GP when sw_powerlaw=True (legacy treatment).
            ([solar.makegp_timedomain_solar_dm(psr, covariance=(signals.quasi_periodic if sw_qp else signals.squared_exponential), dt=max_cadence_days*86400.0, name='sw_gp')] if (sw and not sw_powerlaw) else []) + \
            ([signals.makegp_fftcov_solar(psr, signals.powerlaw, components=psr_knots, T=psr_Tspan, name='sw_gp')] if (sw and sw_powerlaw) else []) + \
            ([signals.makegp_fftcov_band(psr, signals.powerlaw, components=psr_knots, T=psr_Tspan, name='band_gp')] if band else []) + \
            ([signals.makegp_fftcov_band_alpha(psr, signals.powerlaw, components=psr_knots, T=psr_Tspan, name='bandalpha_gp')] if band_alpha else []))


def single_pulsar_noise(psr, fftint=True, max_cadence_days=14, Tspan=None, noisedict={},
                        ecorr=True, quadratic=False, ecorr_nmodes=None, ecorr_correlated=False, global_ecorr=False, # ecorr options. ecorr_nmodes=N selects an N-mode Legendre ECORR (log-frequency basis; nmodes=1 is standard ECORR); ecorr_correlated=True uses the full-M (correlated-mode) variant that can also model a frequency-asymmetric jitter amplitude
                        background=True, bkgrnd_log10_A=None, red=True, red2=False, dm=True, chrom=True, chrom_alpha=None, chrom_poly=True, sw=True, sw_powerlaw=False, sw_qp=False, sw_logf=False, # Base model: gwb, red, dm, chromatic, solar wind (sw_powerlaw=True selects the legacy power-law solar-wind GP instead of the time-domain one; sw_logf=True log-spaces its frequencies -- Fourier path only)
                        band=False, band_alpha=False, band_bw_min=20.0, fd=False, fd_nodes=16, fd_spacing='quantile', fd_selection=None, fd_prior='improper', # Additional GP models (fd=True marginalises an arbitrary time-constant frequency-dependent delay over fd_nodes frequency nodes; fd_selection splits it per TOA group; fd_prior selects the improper or the Matern-3/2 prior over the node amplitudes)
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
        _set_chrom_exp_priors(psr, chrom_exponential=True)
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
    # fd_prior='improper' removes the basis directions unconditionally; 'matern'
    # gives the node amplitudes a Matern-3/2 prior in log-frequency and samples its
    # scale and correlation length, so the data set how much is absorbed.
    if not fd:
        fd_gp = None
    elif fd_prior == 'matern':
        fd_gp = signals.makegp_fd_piecewise_matern(psr, nodes=fd_nodes, spacing=fd_spacing,
                                                   selection=fd_selection, name='fd_gp')
    elif fd_prior == 'improper':
        fd_gp = signals.makegp_fd_piecewise(psr, nodes=fd_nodes, spacing=fd_spacing,
                                            selection=fd_selection, name='fd')
    else:
        raise ValueError(f"single_pulsar_noise: fd_prior must be 'improper' or 'matern', "
                         f"got {fd_prior!r}.")
    if fd_gp is not None:
        model_components += [fd_gp]

    # Add GP components
    if fftint:
        if sw_logf:
            # the fftint solar GP uses a time-interpolation basis, so there is no
            # Fourier frequency grid to log-space; sw_logf needs fftint=False.
            print("Warning: sw_logf=True is ignored with fftint=True (the FFT-covariance "
                  "solar GP uses a time-interpolation basis). Use fftint=False.")
        model_components += make_psr_gps_fftint(psr, max_cadence_days=max_cadence_days, bkgrnd_log10_A=bkgrnd_log10_A, Tspan=Tspan, background=background, red=red, red2=red2, dm=dm, chrom=chrom, chrom_alpha=chrom_alpha, chrom_poly=chrom_poly, sw=sw, sw_powerlaw=sw_powerlaw, sw_qp=sw_qp, band=band, band_alpha=band_alpha, band_bw_min=band_bw_min, fd_gp=fd_gp)
    else:
        model_components += make_psr_gps_fourier(psr, max_cadence_days=max_cadence_days, bkgrnd_log10_A=bkgrnd_log10_A, Tspan=Tspan, background=background, red=red, red2=red2, dm=dm, chrom=chrom, chrom_alpha=chrom_alpha, chrom_poly=chrom_poly, sw=sw, sw_powerlaw=sw_powerlaw, sw_qp=sw_qp, sw_logf=sw_logf, band=band, band_alpha=band_alpha, band_bw_min=band_bw_min, fd_gp=fd_gp)


    if extra_gps is not None:
        model_components += extra_gps

    m = likelihood.PulsarLikelihood(model_components)

    if return_components:
        return m, model_components
    
    return m

def common_noise(psrs, chain_dfs, fftInt=True, max_cadence_days=14, Tspan=None,
                 chrom_poly=True, fix_chrom_alpha=True, hd=False, hd_fixed_gamma=False,
                 hd_components=None,  # HD Fourier bins; None -> common_components (i.e. tied to max_cadence_days)
                 os_analysis=False,  # put the HD spectrum (gw_log10_A/gw_gamma) into a PER-PULSAR GP instead of a globalgp, so discovery.optimal.OS can see it. For OS runs only -- NOT for Bayesian sampling, which wants the correlated globalgp.
                 fd=False, fd_nodes=16, fd_spacing='quantile', fd_selection=None, fd_prior='improper',  # piecewise-linear frequency-dependent delay; nodes/spacing/selection MUST match the stage-1 runs, as they cannot be auto-detected (see below)
                 use_commongp=False,
                 curn_per_pulsar=False,  # give the common process PER-PULSAR (log10_A, gamma), or per-bin log10_rho under freespec, instead of parameters shared across the array
                 red2=False,  # force a SECOND per-pulsar red power law (name='red_noise2') in every pulsar, regardless of whether the stage-1 chains carry one; without it, red2 is enabled only where has_param(df, "red_noise2") finds it
                 freespec=False, freespec_components=30,  # free-spectrum CURN (per-bin log10_rho) instead of the power law; ~30 components keeps the parameter space manageable for a steep process
                 red_fixed_dict=None,  # {psrname: (log10_A, gamma)}: FIX each pulsar's red noise at these values (e.g. the power-law common-run posteriors) so the free-spectrum bins test excess over the same null the band power was defined against, rather than competing with co-sampled red noise for the same variance
                 use_phys_ephem=False, phys_ephem_partials=phys_ephem_mod.DEFAULT_PARTIALS,
                 phys_ephem_inc_jupiter=True, phys_ephem_inc_saturn=False, phys_ephem_inc_masses=True,
                 phys_ephem_frame_3axis=True, phys_ephem_inc_frame_drift=True, phys_ephem_inc_mainbelt=True,
                 phys_ephem_inc_minorbody=True, phys_ephem_orthogonalize_minorbody=False,
                 phys_ephem_inc_jerk=True, phys_ephem_mainbelt_prior_scale=1.0,
                 phys_ephem_mainbelt_block="mass",
                 phys_ephem_belt_eta_convention="none",
                 phys_ephem_prior_units="edge", phys_ephem_minorbody_sigma=None,
                 phys_ephem_mass_bodies=("jupiter", "saturn", "uranus", "neptune")):
    # Accepts a list of pulsars and their corresponding chain dataframes and constructs a GlobalLikelihood
    def has_param(df, param_string):
        return any(param_string in col for col in df.columns)

    if chrom_poly:
        print("Note: chrom_poly=True (the chromatic polynomial is marginalised analytically).")

    # The commongp/ArrayLikelihood path moves every SAMPLED Fourier/fftcov GP out of
    # the per-pulsar likelihoods into a single stacked common GP and uses the
    # vectorised ArrayLikelihood. It requires a non-callable chromatic basis
    # (fix_chrom_alpha=True) and supports neither the marginalised chromatic
    # polynomial nor os_analysis: the OS reads psl.gw, psl.N.F and psl.N.P_var from
    # each per-pulsar likelihood, and the commongp path stacks those GPs out of them.
    if os_analysis and use_commongp:
        print("Warning: os_analysis=True is incompatible with use_commongp=True (the OS "
              "needs the per-pulsar 'gw' GP inside each pulsar likelihood, but the commongp "
              "path stacks it out). Falling back to the GlobalLikelihood path.")
        use_commongp = False
    if os_analysis and not hd:
        print("Warning: os_analysis=True has no effect with hd=False -- there is no HD "
              "process to carry. Ignoring it.")
        os_analysis = False

    # fd (marginalised piecewise-linear frequency-dependent delay) is the ONE
    # per-pulsar model component that cannot be auto-detected from chain_dfs. Every
    # other component is inferred with has_param(), but the fd GP is a ConstantGP
    # with a NoiseMatrix1D_novar prior -- fully marginalised, ZERO sampled
    # hyperparameters -- so an fd-enabled stage-1 chain contains no fd columns to
    # match on. It must therefore be passed explicitly, set to whatever the stage-1
    # runs used. Getting it wrong is silent: the common model would marginalise over
    # a different basis than the single-pulsar fits did, with no missing-parameter
    # error to catch it (there are no parameters to miss).
    if fd:
        print(f"fd=True ({fd_prior} prior): a piecewise-linear frequency-dependent delay "
              f"over {fd_nodes} nodes, {fd_spacing} spacing"
              f"{'' if fd_selection is None else ', per-group selection'}. The node layout must "
              f"match the stage-1 single-pulsar runs -- it is not auto-detected, because the "
              f"amplitudes are marginalised and leave no parameters in the chains.")

    # the Matern prior samples log10_sigma/log10_ell, so unlike the improper prior its
    # presence IS visible in the stage-1 chains; disagreement means the common model
    # is not the model the single-pulsar runs used
    _fd_in_chains = any(has_param(df, "fd_gp_log10_sigma") for df in chain_dfs)
    if _fd_in_chains and not (fd and fd_prior == 'matern'):
        print("Warning: the single-pulsar chains carry fd_gp hyperparameters, but this common "
              "model is being built with "
              f"{'fd=False' if not fd else f'fd_prior={fd_prior!r}'}. Pass fd=True and "
              "fd_prior='matern' to match them.")
    elif fd and fd_prior == 'matern' and not _fd_in_chains:
        print("Warning: fd_prior='matern' was requested, but no chain carries fd_gp "
              "hyperparameters -- the stage-1 runs did not use this GP.")

    commongp_path = use_commongp and fix_chrom_alpha
    if use_commongp and not fix_chrom_alpha:
        print("Warning: use_commongp=True requires fix_chrom_alpha=True (the stacked "
              "commongp path needs a non-callable chromatic basis). Falling back to the "
              "GlobalLikelihood path.")
    # The vectorised kernel product calls solve_2d on each pulsar's core kernel, which
    # only a constant kernel provides. The time-domain solar-wind GP is sampled and is
    # kept per-pulsar rather than stacked, so a pulsar carrying one leaves its core a
    # WoodburyKernel_varP and the commongp path fails inside matrix.py.
    if commongp_path:
        sw_psrs = [psr.name for psr, df in zip(psrs, chain_dfs) if has_param(df, "sw_gp")]
        if sw_psrs:
            print(f"Warning: use_commongp=True needs a constant per-pulsar core, but "
                  f"{len(sw_psrs)} pulsar(s) carry the sampled time-domain solar-wind GP, "
                  f"which is not stackable. Falling back to the GlobalLikelihood path.")
            commongp_path = False
    if freespec:
        prior.priordict_standard.update({r'curn_log10_rho\(([0-9]*)\)': [-9, -4],
                                         'curn_log10_rho': [-9, -4]})
    if use_phys_ephem:
        # Register uniform [-1, 1] priors for the global PEBBLE (physical-ephemeris)
        # coefficients (the inter-ephemeris prior half-widths are folded into the design matrix).
        prior.priordict_standard.update(phys_ephem_mod.phys_ephem_priordict())

    if Tspan is None:
        Tspan = signals.getspan(psrs)
    common_components = int(Tspan / (max_cadence_days * 86400))
    common_knots = 2 * common_components + 1
    # Bin count for the HD process, shared by the globalgp and the os_analysis
    # per-pulsar GP so the two carry the same spectral parametrisation.
    hd_nc = common_components if hd_components is None else int(hd_components)

    if os_analysis:
        print(f"os_analysis=True: the HD spectrum (gw_log10_A"
              f"{'' if hd_fixed_gamma else '/gw_gamma'}, {hd_nc} Fourier bins) is carried by a "
              f"PER-PULSAR GP named 'gw' instead of a cross-pulsar globalgp.\n"
              f"  FOR OPTIMAL-STATISTIC RUNS ONLY -- this is NOT the Bayesian model. The "
              f"inter-pulsar HD correlations are deliberately absent, because the OS is what "
              f"measures them; the GP exists so the HD amplitude and spectral index enter "
              f"psl.N and psl.gw and can normalise the OS covariance.\n"
              f"  The parameter names are unchanged, so a chain from a standard hd=True "
              f"Bayesian run supplies them directly.")

    psls = []
    per_psr_stack_gps = []  # commongp path: per-pulsar stacked sampled GPs
    for psr, df in zip(psrs, chain_dfs):
        if not any(psr.name in col for col in df.columns):
            raise ValueError("Chain data frames do not match pulsar names")
        # per-pulsar noise from the max-likelihood row of the chain
        ml_idx = df['logl'].idxmax()
        noisedict = {col: df.loc[ml_idx, col] for col in df.columns if col.startswith(psr.name)}
        # Fix chromatic alpha, or leave it sampled with a callable basis
        chrom_alpha = None
        if fix_chrom_alpha:
            chrom_alpha = noisedict.get(f"{psr.name}_chrom_gp_alpha", None)
            print(f"Using chromatic alpha={chrom_alpha} for pulsar {psr.name}")
 
        # Detect a multi-mode Legendre ecorr from the higher-mode amplitudes
        # ..._log10_ecorr_k{m} (mode 0 is the unsuffixed ..._log10_ecorr, identical
        # to standard ECORR). nmodes = highest k index + 1; if no _k params are
        # present it is either standard ECORR or nmodes=1 Legendre (the same model),
        # so ecorr_nmodes stays None and the standard makegp_ecorr branch is used.
        kidxs = [int(col.rsplit('_k', 1)[-1]) for col in df.columns
                 if 'log10_ecorr_k' in col and col.rsplit('_k', 1)[-1].isdigit()]
        ecorr_nmodes = max(kidxs) + 1 if kidxs else None
        # Detect the full-M (correlated-mode) variant from its correlation params
        ecorr_correlated = has_param(df, "ecorr_corr_k")
        if ecorr_correlated:
            print(f'detected {ecorr_nmodes} ecorr nmodes for {psr.name}')
        # Detect the solar-wind GP variant: the legacy power-law GP uses
        # sw_gp_log10_A / sw_gp_gamma; the time-domain GP uses
        # sw_gp_log10_ell / sw_gp_log10_sigma. See single_pulsar_noise.
        sw_powerlaw = has_param(df, "sw_gp_log10_A") or has_param(df, "sw_gp_gamma")
        # the quasi-periodic kernel adds sw_gp_log10_Gamma / sw_gp_log10_p on top of
        # the squared-exponential sw_gp_log10_sigma / sw_gp_log10_ell
        sw_qp = has_param(df, "sw_gp_log10_Gamma") or has_param(df, "sw_gp_log10_p")
        if sw_qp:
            print(f"detected quasi-periodic solar-wind kernel for {psr.name}")

        if freespec:
            # Free-spectrum CURN: one common log10_rho per frequency bin
            # (Fourier basis; use with fftInt=False). Motivated by the
            # non-power-law common band power at 1.5-2.1 yr (Gate-2).
            curn = signals.makegp_fourier(psr, signals.freespectrum, freespec_components, Tspan, common=([] if curn_per_pulsar else ['curn_log10_rho']), name='curn')
        elif not fftInt:
            curn = signals.makegp_fourier(psr, signals.powerlaw, common_components, Tspan, common=([] if curn_per_pulsar else ['curn_log10_A', 'curn_gamma']), name='curn')
        else:
            curn = signals.makegp_fftcov(psr, signals.powerlaw, common_knots, Tspan, common=([] if curn_per_pulsar else ['curn_log10_A', 'curn_gamma']), name='curn')
        # Sampled common GPs that are STACKABLE into the commongp (curn, red_fixed).
        common_gps = curn if isinstance(curn, list) else [curn]

        if os_analysis:
            # The HD process as a PER-PULSAR GP named 'gw', keeping the same
            # parameter names (gw_log10_A, and gw_gamma unless fixed) as the
            # globalgp it stands in for -- so a chain from a standard hd=True
            # Bayesian run feeds it directly, with no column renaming anywhere.
            #
            # It is UNCORRELATED between pulsars by design: the OS estimates the
            # cross-correlations itself and only needs the spectrum to normalise its
            # covariance. What this buys is that the HD power now enters psl.N (the
            # OS noise model) and psl.gw (its template) -- with the globalgp neither
            # happens, so the OS silently normalised by CURN and left the HD power
            # out of the covariance entirely.
            #
            # Always a FOURIER powerlaw GP even when fftInt=True: optimal.OS does
            # sPhi = sqrt(psl.gw.Phi.getN(params)) and forms sPhi[:,None]*S*sPhi[None,:],
            # which requires a 1-D (diagonal) PSD. makegp_fftcov's prior is a dense
            # 2-D covariance over time-domain knots and would break that elementwise
            # scaling silently.
            hd_prior = signals.powerlaw_gwb() if hd_fixed_gamma else signals.powerlaw
            hd_common = ['gw_log10_A'] if hd_fixed_gamma else ['gw_log10_A', 'gw_gamma']
            common_gps = common_gps + [
                signals.makegp_fourier(psr, hd_prior, hd_nc, Tspan,
                                       common=hd_common, name='gw')]

        red_flag = has_param(df, "red_noise")
        if red_fixed_dict is not None and psr.name in red_fixed_dict:
            _la, _ga = red_fixed_dict[psr.name]
            def _make_fixed_red(_la=_la, _ga=_ga):
                def powerlaw_fixed(f, df):
                    return signals.powerlaw(f, df, log10_A=_la, gamma=_ga)
                return powerlaw_fixed
            common_gps = common_gps + [signals.makegp_fourier(psr, _make_fixed_red(), common_components, Tspan, name='red_noise_fixed')]
            red_flag = False

        # PEBBLE (physical-ephemeris) is a DETERMINISTIC (callable) delay, not a sampled
        # GP: it flows through psl.y as a CompoundDelay and stays in the per-pulsar
        # likelihood in both paths (it is not stacked into the commongp).
        pe_delays = []
        if use_phys_ephem:
            # Global (common) deterministic physical-ephemeris delay; the same
            # coefficient names appear in every pulsar, so they are shared.
            pe_delays = [phys_ephem_mod.makedelay_phys_ephem(
                psr, phys_ephem_partials, inc_jupiter=phys_ephem_inc_jupiter,
                inc_saturn=phys_ephem_inc_saturn, inc_masses=phys_ephem_inc_masses,
                frame_drift_3axis=phys_ephem_frame_3axis,
                inc_frame_drift=phys_ephem_inc_frame_drift,
                inc_mainbelt=phys_ephem_inc_mainbelt,
                inc_minorbody=phys_ephem_inc_minorbody,
                orthogonalize_minorbody=phys_ephem_orthogonalize_minorbody,
                inc_jerk=phys_ephem_inc_jerk,
                mainbelt_prior_scale=phys_ephem_mainbelt_prior_scale,
                mainbelt_block=phys_ephem_mainbelt_block,
                belt_eta_convention=phys_ephem_belt_eta_convention,
                prior_units=phys_ephem_prior_units,
                minorbody_sigma=phys_ephem_minorbody_sigma,
                mass_bodies=phys_ephem_mass_bodies)]

        if commongp_path:
            # Build the STACKABLE sampled Fourier/fftcov GPs with the same makegp_*
            # calls as the per-pulsar path; the time-domain solar-wind GP (gpname
            # 'sw_gp') is filtered out and kept per-pulsar (dense, not stackable).
            gp_builder = make_psr_gps_fftint if fftInt else make_psr_gps_fourier
            psr_gps = gp_builder(psr, max_cadence_days=max_cadence_days, Tspan=Tspan, background=False,
                                 red=red_flag, red2=(red2 or has_param(df, "red_noise2")),
                                 dm=has_param(df, "dm_gp"), chrom=has_param(df, "chrom_gp"),
                                 chrom_alpha=chrom_alpha, chrom_poly=False,
                                 sw=has_param(df, "sw_gp"), sw_powerlaw=sw_powerlaw, sw_qp=sw_qp,
                                 band=has_param(df, "band_gp"), band_alpha=has_param(df, "bandalpha_gp"))
            sw_gps    = [g for g in psr_gps if getattr(g, 'gpname', None) == 'sw_gp']
            stack_gps = [g for g in psr_gps if getattr(g, 'gpname', None) != 'sw_gp']

            # Core per-pulsar likelihood: residuals + timing + white noise + ecorr +
            # deterministic delays + BayesEphem + time-domain solar-wind GP. NO sampled
            # Fourier/fftcov GPs (those go to the commongp).
            core = single_pulsar_noise(psr, fftint=fftInt, max_cadence_days=max_cadence_days, Tspan=Tspan, background=False, noisedict=noisedict,
                                       ecorr_nmodes=ecorr_nmodes, ecorr_correlated=ecorr_correlated, global_ecorr=has_param(df, f"{psr.name}_ecorr"),
                                       red=False, red2=False, dm=False, chrom=False, chrom_alpha=chrom_alpha, chrom_poly=(chrom_poly and has_param(df, "chrom_gp")), sw=False, sw_powerlaw=sw_powerlaw, sw_qp=sw_qp,
                                       band=False, band_alpha=False,
                                       chrom_annual=has_param(df, "chrom_1yr"), chrom_exponential=has_param(df, "chrom_exp"), chrom_gaussian=has_param(df, "chrom_gauss"), chrom_sphere=has_param(df, "chrom_sphere"), chrom_step=has_param(df, "chrom_step"),
                                       fd=fd, fd_nodes=fd_nodes, fd_spacing=fd_spacing, fd_selection=fd_selection, fd_prior=fd_prior,
                                       extra_gps=(sw_gps + pe_delays))

            per_psr_stack_gps.append(matrix.CompoundGP(stack_gps + common_gps))
            print("Including pulsar", psr.name, "(commongp) with model parameters:\n", core.logL.params)
            psls.append(core)
        else:
            # background=False, as we are including a common red noise process
            m = single_pulsar_noise(psr, fftint=fftInt, max_cadence_days=max_cadence_days, Tspan=Tspan, background=False, noisedict=noisedict,
                                    ecorr_nmodes=ecorr_nmodes, ecorr_correlated=ecorr_correlated, global_ecorr=has_param(df, f"{psr.name}_ecorr"),
                                    red=red_flag, red2=(red2 or has_param(df, "red_noise2")),
                                    dm=has_param(df, "dm_gp"), chrom=has_param(df, "chrom_gp"), chrom_alpha=chrom_alpha, chrom_poly=(chrom_poly and has_param(df, "chrom_gp")), sw=has_param(df, "sw_gp"), sw_powerlaw=sw_powerlaw, sw_qp=sw_qp,
                                    band=has_param(df, "band_gp"), band_alpha=has_param(df, "bandalpha_gp"),
                                    chrom_annual=has_param(df, "chrom_1yr"), chrom_exponential=has_param(df, "chrom_exp"), chrom_gaussian=has_param(df, "chrom_gauss"), chrom_sphere=has_param(df, "chrom_sphere"), chrom_step=has_param(df, "chrom_step"),
                                    fd=fd, fd_nodes=fd_nodes, fd_spacing=fd_spacing, fd_selection=fd_selection, fd_prior=fd_prior,
                                    extra_gps=(common_gps + pe_delays))

            print("Including pulsar", psr.name, "with model parameters:\n", m.logL.params)
            psls.append(m)

    # Optional Hellings-Downs (quadrupole) correlated common process, fit
    # *simultaneously* with the per-pulsar `curn` (common uncorrelated red noise)
    # and the dipolar BayesEphem terms -- so quadrupolar GW power and dipolar
    # (ephemeris / Planet-Nine) power are separated by angular correlation rather
    # than absorbed into one another. Parameters: gw_log10_A, gw_gamma.
    globalgp = None
    if hd and os_analysis:
        # The HD process is already present, per-pulsar and uncorrelated, as the
        # 'gw' GP built in the loop above. Building the globalgp as well would
        # double-count its power.
        pass
    elif hd:
        # hd_fixed_gamma: fix the HD spectral index to 13/3 (signals.powerlaw_gwb)
        # so only gw_log10_A is sampled -- isolates the HD amplitude<->PEBBLE
        # covariance from the A-gamma degeneracy. Default: free gw_gamma.
        hd_spectrum = signals.powerlaw_gwb() if hd_fixed_gamma else signals.powerlaw
        # hd_components sets the HD Fourier bin count independently of
        # max_cadence_days, which fixes common_components for the per-pulsar
        # DM/chromatic bases. The global term scales as (n_psr*2*n_bins)^2.
        # None ties the HD bin count to common_components. (hd_nc is computed
        # above, shared with the os_analysis per-pulsar GP.)
        globalgp = signals.makeglobalgp_fourier(
            psrs, hd_spectrum, signals.hd_orf, hd_nc, Tspan, name='gw')

    if commongp_path:
        # Stack the per-pulsar sampled GPs into a single (padded) common GP and use
        # the vectorised ArrayLikelihood, which only reads psl.N and psl.y.
        cgp = gps2commongp(per_psr_stack_gps)
        return likelihood.ArrayLikelihood(psls, commongp=cgp, globalgp=globalgp)

    return likelihood.GlobalLikelihood(psls, globalgp=globalgp)
