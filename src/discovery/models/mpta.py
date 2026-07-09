import numpy as np
import jax.numpy as jnp
import re

from .. import matrix
from .. import signals
from .. import prior
from .. import solar
from .. import likelihood
from .. import deterministic
from .. import bayesephem as bayesephem_mod
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
        # robust fcenter+log10_bw band_width models.
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
    priors = [gp.Phi.getN for gp in gps]
    pmax = len(gps)
    ns = [gp.F.shape[1] for gp in gps]  # Does not work for callable gp.F (e.g. chromatic GP)
    nmax = max(ns)

    def prior(params):
        yp = matrix.jnp.full((pmax, nmax), 1e-40)
        for i,p in enumerate(priors):
            yp = yp.at[i, :ns[i]].set(p(params))

        return yp

    prior.params = sorted(set([par for p in priors for par in p.params]))
    Fs = [np.pad(gp.F, [(0,0), (0,nmax - gp.F.shape[1])]) for gp in gps]

    return matrix.VariableGP(matrix.VectorNoiseMatrix1D_var(prior), Fs)


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


def make_psr_gps_fourier(psr, max_cadence_days=14, bkgrnd_log10_A=None, Tspan=None, background=True, red=True, red2=False, dm=True, chrom=True, chrom_alpha=None, chrom_poly=False, sw=True, sw_powerlaw=False, band=False, band_alpha=False):
    psr_Tspan = signals.getspan(psr) if Tspan is None else Tspan
    psr_components = int(psr_Tspan / (max_cadence_days * 86400))
    _set_band_priors(psr, band=band, band_alpha=band_alpha)

    return (([signals.makegp_fourier(psr, signals.powerlaw_gwb(log10_A=bkgrnd_log10_A), components=psr_components, name='bkgrnd')] if background else []) + \
            ([signals.makegp_fourier(psr, signals.powerlaw, components=psr_components, name='red_noise')] if red else []) + \
            ([signals.makegp_fourier(psr, signals.powerlaw, components=psr_components, name='red_noise2')] if red2 else []) + \
            ([signals.makegp_fourier(psr, signals.powerlaw, components=psr_components, fourierbasis=signals.fourierbasis_dm, name='dm_gp')] if dm else [])+ \
            ([signals.makegp_fourier(psr, signals.powerlaw, components=psr_components, fourierbasis=signals.fourierbasis_chrom, name='chrom_gp', alpha=chrom_alpha)] if chrom else [])+ \
            ([signals.makegp_chrom_poly_svd(psr, name='chrom_gp')] if (chrom and chrom_poly) else []) + \
            # Solar wind: time-domain squared-exponential GP by default, or the power-law (Fourier) GP when sw_powerlaw=True (legacy treatment).
            ([solar.makegp_timedomain_solar_dm(psr, covariance=signals.squared_exponential, dt=max_cadence_days*86400.0, name='sw_gp')] if (sw and not sw_powerlaw) else []) + \
            ([signals.makegp_fourier(psr, signals.powerlaw, components=psr_components, fourierbasis=solar.fourierbasis_solar, name='sw_gp')] if (sw and sw_powerlaw) else []) + \
            ([signals.makegp_fourier(psr, signals.powerlaw, components=psr_components, fourierbasis=signals.fourierbasis_band_width, name='band_gp')] if band else []) + \
            ([signals.makegp_fourier(psr, signals.powerlaw, components=psr_components, fourierbasis=signals.fourierbasis_band_width_alpha, name='bandalpha_gp')] if band_alpha else []))


def make_psr_gps_fftint(psr, max_cadence_days=14, bkgrnd_log10_A=None, Tspan=None, background=True, red=True, red2=False, dm=True, chrom=True, chrom_alpha=None, chrom_poly=False, sw=True, sw_powerlaw=False, band=False, band_alpha=False):
    psr_Tspan = signals.getspan(psr) if Tspan is None else Tspan
    psr_components = int(psr_Tspan / (max_cadence_days * 86400))
    psr_knots = 2 * psr_components + 1
    _set_band_priors(psr, band=band, band_alpha=band_alpha)

    return (([signals.makegp_fftcov(psr, signals.powerlaw_gwb(log10_A=bkgrnd_log10_A), components=psr_knots, name='bkgrnd')] if background else []) + \
            ([signals.makegp_fftcov(psr, signals.powerlaw, components=psr_knots, name='red_noise')] if red else []) + \
            ([signals.makegp_fftcov(psr, signals.powerlaw, components=psr_knots, name='red_noise2')] if red2 else []) + \
            ([signals.makegp_fftcov_dm(psr, signals.powerlaw, components=psr_knots, name='dm_gp')] if dm else [])+ \
            ([signals.makegp_fftcov_chrom(psr, signals.powerlaw, components=psr_knots, name='chrom_gp', alpha=chrom_alpha)] if chrom else [])+ \
            ([signals.makegp_chrom_poly_svd(psr, name='chrom_gp')] if (chrom and chrom_poly) else []) + \
            # Solar wind: time-domain squared-exponential GP by default, or the power-law (FFT-covariance) GP when sw_powerlaw=True (legacy treatment).
            ([solar.makegp_timedomain_solar_dm(psr, covariance=signals.squared_exponential, dt=max_cadence_days*86400.0, name='sw_gp')] if (sw and not sw_powerlaw) else []) + \
            ([signals.makegp_fftcov_solar(psr, signals.powerlaw, components=psr_knots, name='sw_gp')] if (sw and sw_powerlaw) else []) + \
            ([signals.makegp_fftcov_band_width(psr, signals.powerlaw, components=psr_knots, name='band_gp')] if band else []) + \
            ([signals.makegp_fftcov_band_width_alpha(psr, signals.powerlaw, components=psr_knots, name='bandalpha_gp')] if band_alpha else []))


def single_pulsar_noise(psr, fftint=True, max_cadence_days=14, Tspan=None, noisedict={},
                        ecorr=True, quadratic=False, ecorr_nmodes=None, ecorr_correlated=False, global_ecorr=False, # ecorr options. ecorr_nmodes=N selects an N-mode Legendre ECORR (log-frequency basis; nmodes=1 is standard ECORR); ecorr_correlated=True uses the full-M (correlated-mode) variant that can also model a frequency-asymmetric jitter amplitude
                        background=True, bkgrnd_log10_A=None, red=True, red2=False, dm=True, chrom=True, chrom_alpha=None, chrom_poly=False, sw=True, sw_powerlaw=False, # Base model: gwb, red, dm, chromatic, solar wind (sw_powerlaw=True selects the legacy power-law solar-wind GP instead of the time-domain one)
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
        model_components += make_psr_gps_fftint(psr, max_cadence_days=max_cadence_days, bkgrnd_log10_A=bkgrnd_log10_A, Tspan=Tspan, background=background, red=red, red2=red2, dm=dm, chrom=chrom, chrom_alpha=chrom_alpha, chrom_poly=chrom_poly, sw=sw, sw_powerlaw=sw_powerlaw, band=band, band_alpha=band_alpha)
    else:
        model_components += make_psr_gps_fourier(psr, max_cadence_days=max_cadence_days, bkgrnd_log10_A=bkgrnd_log10_A, Tspan=Tspan, background=background, red=red, red2=red2, dm=dm, chrom=chrom, chrom_alpha=chrom_alpha, chrom_poly=chrom_poly, sw=sw, sw_powerlaw=sw_powerlaw, band=band, band_alpha=band_alpha)

    if extra_gps is not None:
        model_components += extra_gps

    m = likelihood.PulsarLikelihood(model_components)

    if return_components:
        return m, model_components
    
    return m

def common_noise(psrs, chain_dfs, fftInt=True, max_cadence_days=14, name="gw_crn", Tspan=None,
                 chrom_poly=False, fix_chrom_alpha=True, hd=False,
                 freespec=False, freespec_components=30,  # free-spectrum CURN (per-bin log10_rho) instead of the power law; ~30 components keeps the parameter space manageable for a steep process
                 red_fixed_dict=None,  # {psrname: (log10_A, gamma)}: FIX each pulsar's red noise at these values (e.g. the power-law common-run posteriors) so the free-spectrum bins test excess over the same null the band power was defined against, rather than competing with co-sampled red noise for the same variance
                 bayesephem=False, bayesephem_partials=bayesephem_mod.DEFAULT_PARTIALS,
                 bayesephem_inc_jupiter=True, bayesephem_inc_saturn=False, bayesephem_inc_masses=True,
                 bayesephem_frame_3axis=True, bayesephem_inc_mainbelt=True,
                 bayesephem_inc_minorbody=True, bayesephem_orthogonalize_minorbody=False,
                 bayesephem_inc_jerk=True,
                 bayesephem_mass_bodies=("jupiter", "saturn", "uranus", "neptune")):
    # Accepts a list of pulsars and their corresponding chain dataframes and constructs a GlobalLikelihood
    def has_param(df, param_string):
        return any(param_string in col for col in df.columns)

    if chrom_poly:
        print("Note: chrom_poly=True (chromatic polynomial is marginalised by default). Set chrom_poly=False to disable.")

    if freespec:
        prior.priordict_standard.update({r'curn_log10_rho\(([0-9]*)\)': [-9, -4],
                                         'curn_log10_rho': [-9, -4]})
    if bayesephem:
        # Register uniform [-1, 1] priors for the global BayesEphem coefficients
        # (the inter-ephemeris prior half-widths are folded into the design matrix).
        prior.priordict_standard.update(bayesephem_mod.bayesephem_priordict())

    if Tspan is None:
        Tspan = signals.getspan(psrs)
    common_components = int(Tspan / (max_cadence_days * 86400))
    common_knots = 2 * common_components + 1

    psls = []
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
        extra_gps = curn if isinstance(curn, list) else [curn]

        red_flag = has_param(df, "red_noise")
        if red_fixed_dict is not None and psr.name in red_fixed_dict:
            _la, _ga = red_fixed_dict[psr.name]
            def _make_fixed_red(_la=_la, _ga=_ga):
                def powerlaw_fixed(f, df):
                    return signals.powerlaw(f, df, log10_A=_la, gamma=_ga)
                return powerlaw_fixed
            extra_gps = extra_gps + [signals.makegp_fourier(psr, _make_fixed_red(), common_components, Tspan, name='red_noise_fixed')]
            red_flag = False

        if bayesephem:
            # Global (common) deterministic physical-ephemeris delay; the same
            # coefficient names appear in every pulsar, so they are shared.
            extra_gps = extra_gps + [bayesephem_mod.makedelay_bayesephem(
                psr, bayesephem_partials, inc_jupiter=bayesephem_inc_jupiter,
                inc_saturn=bayesephem_inc_saturn, inc_masses=bayesephem_inc_masses,
                frame_drift_3axis=bayesephem_frame_3axis,
                inc_mainbelt=bayesephem_inc_mainbelt,
                inc_minorbody=bayesephem_inc_minorbody,
                orthogonalize_minorbody=bayesephem_orthogonalize_minorbody,
                inc_jerk=bayesephem_inc_jerk, mass_bodies=bayesephem_mass_bodies)]

        # background=False, as we are including a common red noise process
        m = single_pulsar_noise(psr, fftint=fftInt, max_cadence_days=max_cadence_days, Tspan=Tspan, background=False, noisedict=noisedict, 
                                ecorr_nmodes=ecorr_nmodes, ecorr_correlated=ecorr_correlated, global_ecorr=has_param(df, f"{psr.name}_ecorr"),
                                red=red_flag, red2=has_param(df, "red_noise2"),
                                dm=has_param(df, "dm_gp"), chrom=has_param(df, "chrom_gp"), chrom_alpha=chrom_alpha, chrom_poly=chrom_poly, sw=has_param(df, "sw_gp"), sw_powerlaw=sw_powerlaw,
                                band=has_param(df, "band_gp"), band_alpha=has_param(df, "bandalpha_gp"),
                                chrom_annual=has_param(df, "chrom_1yr"), chrom_exponential=has_param(df, "chrom_exp"), chrom_gaussian=has_param(df, "chrom_gauss"), chrom_sphere=has_param(df, "chrom_sphere"), chrom_step=has_param(df, "chrom_step"),
                                extra_gps=extra_gps)
 
        print("Including pulsar", psr.name, "with model parameters:\n", m.logL.params)
        psls.append(m)

    # Optional Hellings-Downs (quadrupole) correlated common process, fit
    # *simultaneously* with the per-pulsar `curn` (common uncorrelated red noise)
    # and the dipolar BayesEphem terms -- so quadrupolar GW power and dipolar
    # (ephemeris / Planet-Nine) power are separated by angular correlation rather
    # than absorbed into one another. Parameters: gw_log10_A, gw_gamma.
    globalgp = None
    if hd:
        globalgp = signals.makeglobalgp_fourier(
            psrs, signals.powerlaw, signals.hd_orf, common_components, Tspan, name='gw')

    return likelihood.GlobalLikelihood(psls, globalgp=globalgp)
    # return likelihood.ArrayLikelihood(psls)