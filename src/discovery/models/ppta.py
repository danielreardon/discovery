import numpy as np
from .. import prior
from discovery.models import mpta

def update_priordict_standard_ppta():
    # Update the standard prior dictionary with PPTA-specific parameters
    prior.priordict_standard.update({
        # White noise parameters
        '(.*_)?efac':               [0.0, 5],
        '(.*_)?log10_tnequad':      [-10, -5],
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
        '(.*_)?chrom_gp_alpha':     [2.5, 14],
        '(.*_)?sw_gp_log10_A':      [-10, -2],
        '(.*_)?sw_gp_gamma':        [0, 4],
        '(.*_)?band_gp_log10_A':    [-18, -11],
        '(.*_)?band_gp_gamma':      [0, 7],
        '(.*_)?bandalpha_gp_log10_A':    [-18, -11],
        '(.*_)?bandalpha_gp_gamma':      [0, 7],
        '(.*_)?bandalpha_gp_alpha':      [0, 10],
        # common noise parameters
        'curn_log10_A':             [-18, -11],
        'curn_gamma':               [0, 7],
        # deterministic parameters
        '(.*_)?chrom_exp_t0': [53000, 60700], # PPTA-DR4 range
        '(.*_)?chrom_exp_log10_Amp': [-10, -4],
        '(.*_)?chrom_exp_log10_tau': [0, 4],
        '(.*_)?chrom_exp_sign_param': [-1, 1],
        '(.*_)?chrom_exp_alpha': [0, 7],
        '(.*_)?chrom_1yr_log10_Amp': [-10, -4],
        '(.*_)?chrom_1yr_phase': [0, 2 * np.pi],
        '(.*_)?chrom_1yr_alpha': [0, 7],
        '(.*_)?chrom_gauss_t0': [53000, 60700], # PPTA-DR4 range
        '(.*_)?chrom_gauss_log10_Amp': [-10, -4],
        '(.*_)?chrom_gauss_log10_sigma': [0, 4],
        '(.*_)?chrom_gauss_sign_param': [-1, 1],
        '(.*_)?chrom_gauss_alpha': [0, 7],
        r'(.*_)?timingmodel_coefficients\(\d+\)': [-20.0, 20.0],
        r'(.*_)?dm_sw_log10_rho\(\d+\)': [-10, 4],
        r'(.*_)?alpha_scaling\(\d+\)': [0.0, 100.0],
    })
    return

def single_pulsar_noise(psr, fftint=True, max_cadence_days=30, Tspan=None, noisedict={}, global_ecorr=True,
                        background=True, red=True, red2=False, dm=True, chrom=True, sw=True, band=False, band_alpha=False, # GP models
                        chrom_annual=False, chrom_exponential=False, chrom_gaussian=False, extra_gps=None): # Deterministic chromatic models
    """ Different defaults for PPTA single pulsar analyses. max_cadence_days=30 and global_ecorr=True.
    See mpta.single_pulsar_noise for details of the parameters."""

    model = mpta.single_pulsar_noise(psr, fftint=fftint, max_cadence_days=max_cadence_days, Tspan=Tspan, noisedict=noisedict, global_ecorr=global_ecorr,
                        background=background, red=red, red2=red2, dm=dm, chrom=chrom, sw=sw, band=band, band_alpha=band_alpha, # GP models
                        chrom_annual=chrom_annual, chrom_exponential=chrom_exponential, chrom_gaussian=chrom_gaussian, extra_gps=extra_gps) # Deterministic chromatic models

    update_priordict_standard_ppta()

    return model