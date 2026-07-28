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
        '(.*_)?chrom_gp_log10_A':   [-18, -11],
        '(.*_)?chrom_gp_gamma':     [0, 7],
        '(.*_)?chrom_gp_alpha':     [3.0, 14], # start at 3 to avoid confusion with DM
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
        'gw_log10_A':             [-18, -11],
        'gw_gamma':               [0, 7],
        # deterministic parameters
        '(.*_)?chrom_exp_t0': [58525, 60900], # MPTA 6-yr range
        '(.*_)?chrom_exp_log10_Amp': [-10, -4],
        '(.*_)?chrom_exp_log10_tau': [0, 4],
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
        r'(.*_)?chrom_sphere_alpha': [0, 14],
        r'(.*_)?chrom_sphere_smooth': [10, 200],
        r'(.*_)?chrom_step_t0': [58525, 60900],
        r'(.*_)?chrom_step_log10_Amp': [-10, -4],
        r'(.*_)?chrom_step_log10_span': [1.0, 4.0],
        r'(.*_)?chrom_step_sign_param': [-1, 1],
        r'(.*_)?chrom_step_alpha': [0, 14],
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


def make_psr_gps_fourier(psr, max_cadence_days=14, bkgrnd_log10_A=None, Tspan=None, background=True, red=True, red2=False, dm=True, chrom=True, chrom_alpha=None, chrom_poly=False, sw=True, sw_powerlaw=False, sw_logf=False, band=False, band_alpha=False):
    psr_Tspan = signals.getspan(psr) if Tspan is None else Tspan
    psr_components = int(psr_Tspan / (max_cadence_days * 86400))
    _set_band_priors(psr, band=band, band_alpha=band_alpha)

    return (([signals.makegp_fourier(psr, signals.powerlaw_gwb(log10_A=bkgrnd_log10_A), components=psr_components, T=psr_Tspan, name='bkgrnd')] if background else []) + \
            ([signals.makegp_fourier(psr, signals.powerlaw, components=psr_components, T=psr_Tspan, name='red_noise')] if red else []) + \
            ([signals.makegp_fourier(psr, signals.powerlaw, components=psr_components, T=psr_Tspan, name='red_noise2')] if red2 else []) + \
            ([signals.makegp_fourier(psr, signals.powerlaw, components=psr_components, T=psr_Tspan, fourierbasis=signals.fourierbasis_dm, name='dm_gp')] if dm else [])+ \
            ([signals.makegp_fourier(psr, signals.powerlaw, components=psr_components, T=psr_Tspan, fourierbasis=signals.fourierbasis_chrom, name='chrom_gp', alpha=chrom_alpha)] if chrom else [])+ \
            ([signals.makegp_chrom_poly_svd(psr, name='chrom_gp')] if (chrom and chrom_poly) else []) + \
            # Solar wind: time-domain squared-exponential GP by default, or the power-law (Fourier) GP when sw_powerlaw=True (legacy treatment).
            ([solar.makegp_timedomain_solar_dm(psr, covariance=signals.squared_exponential, dt=max_cadence_days*86400.0, name='sw_gp')] if (sw and not sw_powerlaw) else []) + \
            ([signals.makegp_fourier(psr, signals.powerlaw, components=psr_components, T=psr_Tspan, fourierbasis=solar.make_fourierbasis_solar_dm(logf=sw_logf), name='sw_gp')] if (sw and sw_powerlaw) else []) + \
            ([signals.makegp_fourier(psr, signals.powerlaw, components=psr_components, T=psr_Tspan, fourierbasis=signals.fourierbasis_band, name='band_gp')] if band else []) + \
            ([signals.makegp_fourier(psr, signals.powerlaw, components=psr_components, T=psr_Tspan, fourierbasis=signals.fourierbasis_band_alpha, name='bandalpha_gp')] if band_alpha else []))


def make_psr_gps_fftint(psr, max_cadence_days=14, bkgrnd_log10_A=None, Tspan=None, background=True, red=True, red2=False, dm=True, chrom=True, chrom_alpha=None, chrom_poly=False, sw=True, sw_powerlaw=False, band=False, band_alpha=False):
    psr_Tspan = signals.getspan(psr) if Tspan is None else Tspan
    psr_components = int(psr_Tspan / (max_cadence_days * 86400))
    psr_knots = 2 * psr_components + 1
    _set_band_priors(psr, band=band, band_alpha=band_alpha)

    return (([signals.makegp_fftcov(psr, signals.powerlaw_gwb(log10_A=bkgrnd_log10_A), components=psr_knots, T=psr_Tspan, name='bkgrnd')] if background else []) + \
            ([signals.makegp_fftcov(psr, signals.powerlaw, components=psr_knots, T=psr_Tspan, name='red_noise')] if red else []) + \
            ([signals.makegp_fftcov(psr, signals.powerlaw, components=psr_knots, T=psr_Tspan, name='red_noise2')] if red2 else []) + \
            ([signals.makegp_fftcov_dm(psr, signals.powerlaw, components=psr_knots, T=psr_Tspan, name='dm_gp')] if dm else [])+ \
            ([signals.makegp_fftcov_chrom(psr, signals.powerlaw, components=psr_knots, T=psr_Tspan, name='chrom_gp', alpha=chrom_alpha)] if chrom else [])+ \
            ([signals.makegp_chrom_poly_svd(psr, name='chrom_gp')] if (chrom and chrom_poly) else []) + \
            # Solar wind: time-domain squared-exponential GP by default, or the power-law (FFT-covariance) GP when sw_powerlaw=True (legacy treatment).
            ([solar.makegp_timedomain_solar_dm(psr, covariance=signals.squared_exponential, dt=max_cadence_days*86400.0, name='sw_gp')] if (sw and not sw_powerlaw) else []) + \
            ([signals.makegp_fftcov_solar(psr, signals.powerlaw, components=psr_knots, T=psr_Tspan, name='sw_gp')] if (sw and sw_powerlaw) else []) + \
            ([signals.makegp_fftcov_band(psr, signals.powerlaw, components=psr_knots, T=psr_Tspan, name='band_gp')] if band else []) + \
            ([signals.makegp_fftcov_band_alpha(psr, signals.powerlaw, components=psr_knots, T=psr_Tspan, name='bandalpha_gp')] if band_alpha else []))


def single_pulsar_noise(psr, fftint=True, max_cadence_days=14, Tspan=None, noisedict={},
                        ecorr=True, quadratic=False, ecorr_nmodes=None, ecorr_correlated=False, global_ecorr=False, # ecorr options. ecorr_nmodes=N selects an N-mode Legendre ECORR (log-frequency basis; nmodes=1 is standard ECORR); ecorr_correlated=True uses the full-M (correlated-mode) variant that can also model a frequency-asymmetric jitter amplitude
                        background=True, bkgrnd_log10_A=None, red=True, red2=False, dm=True, chrom=True, chrom_alpha=None, chrom_poly=False, sw=True, sw_powerlaw=False, sw_logf=False, # Base model: gwb, red, dm, chromatic, solar wind (sw_powerlaw=True selects the legacy power-law solar-wind GP instead of the time-domain one; sw_logf=True log-spaces its frequencies -- Fourier path only)
                        band=False, band_alpha=False, # Additional GP models
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

    # Add GP components
    if fftint:
        if sw_logf:
            # the fftint solar GP uses a time-interpolation basis, so there is no
            # Fourier frequency grid to log-space; sw_logf needs fftint=False.
            print("Warning: sw_logf=True is ignored with fftint=True (the FFT-covariance "
                  "solar GP uses a time-interpolation basis). Use fftint=False.")
        model_components += make_psr_gps_fftint(psr, max_cadence_days=max_cadence_days, bkgrnd_log10_A=bkgrnd_log10_A, Tspan=Tspan, background=background, red=red, red2=red2, dm=dm, chrom=chrom, chrom_alpha=chrom_alpha, chrom_poly=chrom_poly, sw=sw, sw_powerlaw=sw_powerlaw, band=band, band_alpha=band_alpha)
    else:
        model_components += make_psr_gps_fourier(psr, max_cadence_days=max_cadence_days, bkgrnd_log10_A=bkgrnd_log10_A, Tspan=Tspan, background=background, red=red, red2=red2, dm=dm, chrom=chrom, chrom_alpha=chrom_alpha, chrom_poly=chrom_poly, sw=sw, sw_powerlaw=sw_powerlaw, sw_logf=sw_logf, band=band, band_alpha=band_alpha)


    if extra_gps is not None:
        model_components += extra_gps

    m = likelihood.PulsarLikelihood(model_components)

    if return_components:
        return m, model_components
    
    return m

def common_noise(psrs, chain_dfs, fftInt=True, max_cadence_days=14, Tspan=None,
                 chrom_poly=False, fix_chrom_alpha=True, hd=False, hd_fixed_gamma=False,
                 hd_components=None,  # HD Fourier bins; None -> common_components (i.e. tied to max_cadence_days)
                 use_commongp=False,
                 freespec=False, freespec_components=30,  # free-spectrum CURN (per-bin log10_rho) instead of the power law; ~30 components keeps the parameter space manageable for a steep process
                 red_fixed_dict=None,  # {psrname: (log10_A, gamma)}: FIX each pulsar's red noise at these values (e.g. the power-law common-run posteriors) so the free-spectrum bins test excess over the same null the band power was defined against, rather than competing with co-sampled red noise for the same variance
                 use_phys_ephem=False, phys_ephem_partials=phys_ephem_mod.DEFAULT_PARTIALS,
                 phys_ephem_inc_jupiter=True, phys_ephem_inc_saturn=False, phys_ephem_inc_masses=True,
                 phys_ephem_frame_3axis=True, phys_ephem_inc_frame_drift=True, phys_ephem_inc_mainbelt=True,
                 phys_ephem_inc_minorbody=True, phys_ephem_orthogonalize_minorbody=False,
                 phys_ephem_inc_jerk=True, phys_ephem_mainbelt_prior_scale=1.0,
                 phys_ephem_mass_bodies=("jupiter", "saturn", "uranus", "neptune")):
    # Accepts a list of pulsars and their corresponding chain dataframes and constructs a GlobalLikelihood
    def has_param(df, param_string):
        return any(param_string in col for col in df.columns)

    if chrom_poly:
        print("Note: chrom_poly=True (chromatic polynomial is marginalised by default). Set chrom_poly=False to disable.")

    # The commongp/ArrayLikelihood path moves every SAMPLED Fourier/fftcov GP out of
    # the per-pulsar likelihoods into a single stacked common GP and uses the
    # vectorised ArrayLikelihood. Opt-in (default False): it is FASTER on GPU
    # (~1.3-2x for logL/grad, verified) but its .logL uses EQUAL-OR-MORE peak memory
    # than GlobalLikelihood (the stacked/padded bases enlarge compile-time constant
    # folding). The genuine memory win is ArrayLikelihood.cglogL (matrix-free), but
    # that is numerically fragile (NaN on ~37% of prior draws at npsr=20 with default
    # cgmaxiter/clip) and not yet wired into common_search -- validate before relying
    # on it. Requires a non-callable chromatic basis (fix_chrom_alpha=True) and does
    # not support the marginalised chromatic polynomial.
    commongp_path = use_commongp and fix_chrom_alpha
    if use_commongp and not fix_chrom_alpha:
        print("Warning: use_commongp=True requires fix_chrom_alpha=True (the stacked "
              "commongp path needs a non-callable chromatic basis). Falling back to the "
              "GlobalLikelihood path.")
    if commongp_path and chrom_poly:
        print("Warning: use_commongp=True is not supported together with chrom_poly=True; "
              "falling back to the GlobalLikelihood path.")
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

    psls = []
    per_psr_stack_gps = []  # commongp path: per-pulsar stacked sampled GPs
    for psr, df in zip(psrs, chain_dfs):
        if not any(psr.name in col for col in df.columns):
            raise ValueError("Chain data frames do not match pulsar names")
        # noisedict set to median of each column
        noisedict = {col: np.median(df[col]) for col in df.columns if col.startswith(psr.name)}
        # Fix chromatic alpha
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

        if freespec:
            # Free-spectrum CURN: one common log10_rho per frequency bin
            # (Fourier basis; use with fftInt=False). Motivated by the
            # non-power-law common band power at 1.5-2.1 yr (Gate-2).
            curn = signals.makegp_fourier(psr, signals.freespectrum, freespec_components, Tspan, common=['curn_log10_rho'], name='curn')
        elif not fftInt:
            curn = signals.makegp_fourier(psr, signals.powerlaw, common_components, Tspan, common=['curn_log10_A', 'curn_gamma'], name='curn')
        else:
            curn = signals.makegp_fftcov(psr, signals.powerlaw, common_knots, Tspan, common=['curn_log10_A', 'curn_gamma'], name='curn')
        # Sampled common GPs that are STACKABLE into the commongp (curn, red_fixed).
        common_gps = curn if isinstance(curn, list) else [curn]

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
                mass_bodies=phys_ephem_mass_bodies)]

        if commongp_path:
            # Build the STACKABLE sampled Fourier/fftcov GPs with the SAME makegp_*
            # calls used today; the time-domain solar-wind GP (gpname 'sw_gp') is
            # filtered out and kept per-pulsar (dense, not stackable).
            gp_builder = make_psr_gps_fftint if fftInt else make_psr_gps_fourier
            psr_gps = gp_builder(psr, max_cadence_days=max_cadence_days, Tspan=Tspan, background=False,
                                 red=red_flag, red2=has_param(df, "red_noise2"),
                                 dm=has_param(df, "dm_gp"), chrom=has_param(df, "chrom_gp"),
                                 chrom_alpha=chrom_alpha, chrom_poly=False,
                                 sw=has_param(df, "sw_gp"), sw_powerlaw=sw_powerlaw,
                                 band=has_param(df, "band_gp"), band_alpha=has_param(df, "bandalpha_gp"))
            sw_gps    = [g for g in psr_gps if getattr(g, 'gpname', None) == 'sw_gp']
            stack_gps = [g for g in psr_gps if getattr(g, 'gpname', None) != 'sw_gp']

            # Core per-pulsar likelihood: residuals + timing + white noise + ecorr +
            # deterministic delays + BayesEphem + time-domain solar-wind GP. NO sampled
            # Fourier/fftcov GPs (those go to the commongp).
            core = single_pulsar_noise(psr, fftint=fftInt, max_cadence_days=max_cadence_days, Tspan=Tspan, background=False, noisedict=noisedict,
                                       ecorr_nmodes=ecorr_nmodes, ecorr_correlated=ecorr_correlated, global_ecorr=has_param(df, f"{psr.name}_ecorr"),
                                       red=False, red2=False, dm=False, chrom=False, chrom_alpha=chrom_alpha, chrom_poly=False, sw=False, sw_powerlaw=sw_powerlaw,
                                       band=False, band_alpha=False,
                                       chrom_annual=has_param(df, "chrom_1yr"), chrom_exponential=has_param(df, "chrom_exp"), chrom_gaussian=has_param(df, "chrom_gauss"), chrom_sphere=has_param(df, "chrom_sphere"), chrom_step=has_param(df, "chrom_step"),
                                       extra_gps=(sw_gps + pe_delays))

            per_psr_stack_gps.append(matrix.CompoundGP(stack_gps + common_gps))
            print("Including pulsar", psr.name, "(commongp) with model parameters:\n", core.logL.params)
            psls.append(core)
        else:
            # background=False, as we are including a common red noise process
            m = single_pulsar_noise(psr, fftint=fftInt, max_cadence_days=max_cadence_days, Tspan=Tspan, background=False, noisedict=noisedict,
                                    ecorr_nmodes=ecorr_nmodes, ecorr_correlated=ecorr_correlated, global_ecorr=has_param(df, f"{psr.name}_ecorr"),
                                    red=red_flag, red2=has_param(df, "red_noise2"),
                                    dm=has_param(df, "dm_gp"), chrom=has_param(df, "chrom_gp"), chrom_alpha=chrom_alpha, chrom_poly=chrom_poly, sw=has_param(df, "sw_gp"), sw_powerlaw=sw_powerlaw,
                                    band=has_param(df, "band_gp"), band_alpha=has_param(df, "bandalpha_gp"),
                                    chrom_annual=has_param(df, "chrom_1yr"), chrom_exponential=has_param(df, "chrom_exp"), chrom_gaussian=has_param(df, "chrom_gauss"), chrom_sphere=has_param(df, "chrom_sphere"), chrom_step=has_param(df, "chrom_step"),
                                    extra_gps=(common_gps + pe_delays))

            print("Including pulsar", psr.name, "with model parameters:\n", m.logL.params)
            psls.append(m)

    # Optional Hellings-Downs (quadrupole) correlated common process, fit
    # *simultaneously* with the per-pulsar `curn` (common uncorrelated red noise)
    # and the dipolar BayesEphem terms -- so quadrupolar GW power and dipolar
    # (ephemeris / Planet-Nine) power are separated by angular correlation rather
    # than absorbed into one another. Parameters: gw_log10_A, gw_gamma.
    globalgp = None
    if hd:
        # hd_fixed_gamma: fix the HD spectral index to 13/3 (signals.powerlaw_gwb)
        # so only gw_log10_A is sampled -- isolates the HD amplitude<->PEBBLE
        # covariance from the A-gamma degeneracy. Default: free gw_gamma.
        hd_spectrum = signals.powerlaw_gwb() if hd_fixed_gamma else signals.powerlaw
        # hd_components decouples the HD Fourier bin count from max_cadence_days.
        # common_components = int(Tspan/max_cadence_days) is set by what the
        # per-pulsar DM/chromatic bases need; a gamma~13/3 GWB has essentially no
        # support in the high bins, so tying the two makes the global term
        # (n_psr*2*n_bins)^2 far larger than the physics requires -- and that dense
        # Cholesky is ~57% of the per-step cost at 14-day cadence. Fewer GWB
        # frequencies than intrinsic red-noise frequencies is standard NANOGrav /
        # EPTA practice; validate with a bin-count ladder on (gw_log10_A, gw_gamma).
        # None reproduces the previous behaviour exactly.
        hd_nc = common_components if hd_components is None else int(hd_components)
        globalgp = signals.makeglobalgp_fourier(
            psrs, hd_spectrum, signals.hd_orf, hd_nc, Tspan, name='gw')

    if commongp_path:
        # Stack the per-pulsar sampled GPs into a single (padded) common GP and use
        # the vectorised ArrayLikelihood, which only reads psl.N and psl.y.
        cgp = gps2commongp(per_psr_stack_gps)
        return likelihood.ArrayLikelihood(psls, commongp=cgp, globalgp=globalgp)

    return likelihood.GlobalLikelihood(psls, globalgp=globalgp)
