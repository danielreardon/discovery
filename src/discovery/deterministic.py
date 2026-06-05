from cmath import tau
import functools

import numpy as np
import jax
import jax.numpy as jnp

from . import matrix
from . import const


def fpc_fast(pos, gwtheta, gwphi):
    x, y, z = pos

    sin_phi = jnp.sin(gwphi)
    cos_phi = jnp.cos(gwphi)
    sin_theta = jnp.sin(gwtheta)
    cos_theta = jnp.cos(gwtheta)

    m_dot_pos = sin_phi * x - cos_phi * y
    n_dot_pos = -cos_theta * cos_phi * x - cos_theta * sin_phi * y + sin_theta * z
    omhat_dot_pos = -sin_theta * cos_phi * x - sin_theta * sin_phi * y - cos_theta * z

    denom = 1.0 + omhat_dot_pos

    fplus = 0.5 * (m_dot_pos**2 - n_dot_pos**2) / denom
    fcross = (m_dot_pos * n_dot_pos) / denom

    return fplus, fcross


def makedelay_binary(pulsarterm=True):
    def delay_binary(toas, pos, log10_h0, log10_f0, ra, sindec, cosinc, psi, phi_earth, phi_psr):
        """BBH residuals from Ellis et. al 2012, 2013"""

        h0 = 10**log10_h0
        f0 = 10**log10_f0

        dec, inc = jnp.arcsin(sindec), jnp.arccos(cosinc)

        # calculate antenna pattern (note: pos is pulsar sky position unit vector)
        fplus, fcross = fpc_fast(pos, 0.5 * jnp.pi - dec, ra)  # careful with dec -> gwtheta conversion

        if pulsarterm:
            phi_avg = 0.5 * (phi_earth + phi_psr)
        else:
            phi_avg = phi_earth

        tref = 86400.0 * 51544.5  # MJD J2000 in seconds

        phase = phi_avg + 2.0 * jnp.pi * f0 * (toas - tref)
        cphase, sphase = jnp.cos(phase), jnp.sin(phase)

        # fix this for no pulsarterm

        if pulsarterm:
            phi_diff = 0.5 * (phi_earth - phi_psr)
            sin_diff = jnp.sin(phi_diff)

            delta_sin =  2.0 * cphase * sin_diff
            delta_cos = -2.0 * sphase * sin_diff
        else:
            delta_sin = sphase
            delta_cos = cphase

        At = -1.0 * (1.0 + jnp.cos(inc)**2) * delta_sin
        Bt =  2.0 * jnp.cos(inc) * delta_cos

        alpha = h0 / (2 * jnp.pi * f0)

        # calculate rplus and rcross
        rplus  = alpha * (-At * jnp.cos(2 * psi) + Bt * jnp.sin(2 * psi))
        rcross = alpha * ( At * jnp.sin(2 * psi) + Bt * jnp.cos(2 * psi))

        # calculate residuals
        res = -fplus * rplus - fcross * rcross

        return res

    if not pulsarterm:
        delay_binary = functools.partial(delay_binary, phi_psr=jnp.nan)

    return delay_binary


def cos2comp(f, df, A, f0, phi, t0):
    """Project signal A * cos(2pi f t + phi) onto Fourier basis
    cos(2pi k t/T), sin(2pi k t/T) for t in [t0, t0+T]."""

    T = 1.0 / df[0]

    Delta_omega = 2.0 * jnp.pi * (f0 - f[::2])
    Sigma_omega = 2.0 * jnp.pi * (f0 + f[::2])

    phase_Delta_start = phi + Delta_omega * t0
    phase_Delta_end   = phi + Delta_omega * (t0 + T)

    phase_Sigma_start = phi + Sigma_omega * t0
    phase_Sigma_end   = phi + Sigma_omega * (t0 + T)

    ck = (A / T) * (
        (jnp.sin(phase_Delta_end) - jnp.sin(phase_Delta_start)) / Delta_omega +
        (jnp.sin(phase_Sigma_end) - jnp.sin(phase_Sigma_start)) / Sigma_omega
    )

    sk = (A / T) * (
        (jnp.cos(phase_Delta_end) - jnp.cos(phase_Delta_start)) / Delta_omega -
        (jnp.cos(phase_Sigma_end) - jnp.cos(phase_Sigma_start)) / Sigma_omega
    )

    return jnp.stack((sk, ck), axis=1).reshape(-1)


def makefourier_binary(pulsarterm=True):
    def fourier_binary(f, df, mintoa, pos, log10_h0, log10_f0, ra, sindec, cosinc, psi, phi_earth, phi_psr):
        """BBH residuals from Ellis et. al 2012, 2013"""

        h0 = 10**log10_h0
        f0 = 10**log10_f0

        dec, inc = jnp.arcsin(sindec), jnp.arccos(cosinc)

        # calculate antenna pattern (note: pos is pulsar sky position unit vector)
        fplus, fcross = fpc_fast(pos, 0.5 * jnp.pi - dec, ra)  # careful with dec -> gwtheta conversion

        if pulsarterm:
            phi_avg  = 0.5 * (phi_earth + phi_psr)
        else:
            phi_avg = phi_earth

        tref = 86400.0 * 51544.5  # MJD J2000 in seconds

        cphase = cos2comp(f, df, 1.0, f0, phi_avg - 2.0 * jnp.pi * f0 * tref, mintoa)
        sphase = cos2comp(f, df, 1.0, f0, phi_avg - 2.0 * jnp.pi * f0 * tref - 0.5*jnp.pi, mintoa)

        # fix this for no pulsarterm

        if pulsarterm:
            phi_diff = 0.5 * (phi_earth - phi_psr)
            sin_diff = jnp.sin(phi_diff)

            delta_sin =  2.0 * cphase * sin_diff
            delta_cos = -2.0 * sphase * sin_diff
        else:
            delta_sin = sphase
            delta_cos = cphase

        At = -1.0 * (1.0 + jnp.cos(inc)**2) * delta_sin
        Bt =  2.0 * jnp.cos(inc) * delta_cos

        alpha = h0 / (2 * jnp.pi * f0)

        # calculate rplus and rcross
        rplus  = alpha * (-At * jnp.cos(2 * psi) + Bt * jnp.sin(2 * psi))
        rcross = alpha * ( At * jnp.sin(2 * psi) + Bt * jnp.cos(2 * psi))

        # calculate residuals
        res = -fplus * rplus - fcross * rcross

        return res

    if not pulsarterm:
        fourier_binary = functools.partial(fourier_binary, phi_psr=jnp.nan)

    return fourier_binary


def chromatic_exponential(psr, fref=1400.0):
    """Chromatic exponential delay model."""
    toas, fnorm = matrix.jnparray(psr.toas / const.day), matrix.jnparray(fref / psr.freqs)

    def delay(t0, log10_Amp, log10_tau, sign_param, alpha):
        dt = toas - t0
        tau = 10**log10_tau
        amp = 10**log10_Amp
        return jnp.sign(sign_param) * amp * fnorm**alpha * jnp.where(dt >= 0, jnp.exp(-dt / tau), 0.0 )
    
    return delay


def chromatic_annual(psr, fref=1400.0):
    """Chromatic annual delay model."""
    toas, fnorm = matrix.jnparray(psr.toas), matrix.jnparray(fref / psr.freqs)

    def delay(log10_Amp, phase, alpha):
        return 10**log10_Amp * jnp.sin(2*jnp.pi * const.fyr * toas + phase) * fnorm**alpha

    return delay


def chromatic_gaussian(psr, fref=1400.0):
    """Chromatic Gaussian delay model."""
    toas, fnorm = matrix.jnparray(psr.toas / const.day), matrix.jnparray(fref / psr.freqs)

    def delay(t0, log10_Amp, log10_sigma, sign_param, alpha):
        return jnp.sign(sign_param) * 10**log10_Amp * jnp.exp(-(toas - t0)**2 / (2 * (10**log10_sigma)**2)) * fnorm**alpha

    return delay


def chromatic_sphere(psr, fref=1400.0):
    """Chromatic delay from a uniform-density sphere crossing the line of sight."""
    toas, fnorm = matrix.jnparray(psr.toas / const.day), matrix.jnparray(fref / psr.freqs)

    def delay(t0, log10_Amp, log10_tau, sign_param, alpha, smooth):
        tau = 10**log10_tau
        x = (toas - t0) / tau
        # Scale factor k controls the sharpness of the edge transition
        chord = jnp.sqrt(jnp.logaddexp(0.0, smooth * (1.0 - x**2)) / smooth)
        return jnp.sign(sign_param) * 10**log10_Amp * chord * fnorm**alpha

    return delay


def chromatic_step(psr, fref=1400.0):
    """Chromatic delay from a flat-bottomed rise or drop with smooth edges."""
    toas, fnorm = matrix.jnparray(psr.toas / const.day), matrix.jnparray(fref / psr.freqs)

    def delay(t0, log10_Amp, log10_span, sign_param, alpha, smooth):
        span = 10**log10_span
        t_start = t0 - 0.5 * span
        t_end = t0 + 0.5 * span

        sigmoid_on = jnp.reciprocal(1.0 + jnp.exp(-(toas - t_start) * smooth / span))
        sigmoid_off = jnp.reciprocal(1.0 + jnp.exp((toas - t_end) * smooth / span))

        profile = sigmoid_on * sigmoid_off

        return jnp.sign(sign_param) * 10**log10_Amp * profile * fnorm**alpha

    return delay


def orthometric_shapiro(psr, binphase, eps_stig=1e-6, eps_log=1e-10):
    """Orthometric Shapiro delay model from Freire & Wex (2010)."""
    toas, binphase = matrix.jnparray(psr.toas / const.day), matrix.jnparray(binphase)
    if not np.shape(binphase) == np.shape(toas):
        raise ValueError("Input binphase must have the same shape as toas")

    def delay(h3, stig):
        stig_clipped = jnp.clip(stig, eps_stig, 1.0 - eps_stig)
        log_arg = 1 + stig_clipped**2 - 2 * stig_clipped * jnp.sin(binphase)
        log_arg = jnp.maximum(log_arg, eps_log)
        return -(2.0 * h3 / stig_clipped**3) * jnp.log(log_arg)

    return delay

def shapiro_cosi(psr, binphase, eps_cosi=1e-6, eps_log=1e-10):
    """Orthometric Shapiro delay model from Freire & Wex (2010), 
    modified for a uniform cos(i) prior."""
    toas, binphase = matrix.jnparray(psr.toas / const.day), matrix.jnparray(binphase)
    if not np.shape(binphase) == np.shape(toas):
        raise ValueError("Input binphase must have the same shape as toas")

    def delay(h3, cosi):
        cosi_clipped = jnp.clip(cosi, -1.0 + eps_cosi, 1.0 - eps_cosi)
        sin_i = jnp.sqrt(1.0 - cosi_clipped**2)
        stig  = sin_i / (1.0 + cosi_clipped)
        log_arg = 1 + stig**2 - 2 * stig * jnp.sin(binphase)
        log_arg = jnp.maximum(log_arg, eps_log)
        return -(2.0 * h3 / stig**3) * jnp.log(log_arg)

    return delay

def chromatic_polynomial(psr, fref=None, name='chrom_gp'):
    """SVD-orthogonalised chromatic polynomial as a sampled deterministic delay.

    Models a (constant + linear + quadratic)-in-time chromatic delay
    referenced to the mean TOA, scaled by ``(fref / freq)**alpha``.
    The three SVD-orthonormalised coefficients (c0, c1, c2) are sampled
    directly with uniform priors set in the priordict.

    The chromatic index ``alpha`` is shared with the companion Fourier (or
    FFTint) chromatic GP via the parameter name ``{psr}_{name}_alpha``, so
    one consistent chromatic index governs both the short-period Fourier
    modes and the long-period polynomial drift.

    Because the polynomial columns are SVD-orthonormalised over the TOAs,
    a uniform prior ``[-A, A]`` on each c_k bounds that mode's rms
    contribution to the delay (in seconds at ``fref``) to A.

    Sampled parameters
    ------------------
    {psr}_{name}_c0, {psr}_{name}_c1, {psr}_{name}_c2  : polynomial amplitudes
    {psr}_{name}_alpha                                  : chromatic index
                                                          (shared with chrom GP)
    """
    t0_sec  = float(np.mean(psr.toas))
    toas_yr = (psr.toas - t0_sec) / const.yr
    if fref is None:
        fref = float(np.exp(np.mean(np.log(np.asarray(psr.freqs)))))

    M = np.vstack([np.ones_like(toas_yr), toas_yr, toas_yr**2]).T
    U, _, _ = np.linalg.svd(M, full_matrices=False)

    Mmat = np.asarray(psr.Mmat, dtype=np.float64)
    M_norm = Mmat / np.sqrt(np.sum(Mmat**2, axis=0))
    Q_tm, _ = np.linalg.qr(M_norm)

    U_j     = matrix.jnparray(U)
    Q_tm_j  = matrix.jnparray(Q_tm)
    fnorm_j = matrix.jnparray(fref / np.asarray(psr.freqs))

    c0_p    = f'{psr.name}_{name}_c0'
    c1_p    = f'{psr.name}_{name}_c1'
    c2_p    = f'{psr.name}_{name}_c2'
    alpha_p = f'{psr.name}_{name}_alpha'

    def delay(params):
        alpha = params[alpha_p]
        F = U_j * fnorm_j[:, None] ** alpha
        F = F - Q_tm_j @ (Q_tm_j.T @ F)        # project out timing-model subspace
        F, _ = jnp.linalg.qr(F)                # orthonormalise -> |F^T F| = 1 for all alpha
        c = jnp.array([params[c0_p], params[c1_p], params[c2_p]])
        return F @ c

    delay.params = [c0_p, c1_p, c2_p, alpha_p]
    return delay


def orbital_DM_gaussian(psr, binphase):
    """
    An excess DM term centred near superior conjunction
    (binphase = pi/2), with a phase offset phi0 and angular width sigma_phi.
    """
    toas, binphase = matrix.jnparray(psr.toas / const.day), matrix.jnparray(binphase)
    if not np.shape(binphase) == np.shape(toas):
        raise ValueError("Input binphase must have the same shape as toas")

    freqs = matrix.jnparray(psr.freqs)
    K_DM = 4.148808e3  # MHz^2 cm^3 pc^-1 s

    def delay(dm_orb_amp, phi0, sigma_phi):
        # Gaussian DM excess near superior conjunction
        delta_phi = jnp.arctan2(jnp.cos(binphase - phi0), jnp.sin(binphase - phi0))
        excess_dm = dm_orb_amp * jnp.exp(-0.5 * (delta_phi / sigma_phi) ** 2)

        dm_delay = K_DM * excess_dm / freqs**2

        return dm_delay

    return delay


def orbital_DM_fourier(psr, binphase, n_harmonics=16):
    """
    Excess DM as an arbitrary function of orbital phase, constructed
    from a truncated Fourier series.
    """
    binphase = matrix.jnparray(binphase)
    freqs = matrix.jnparray(psr.freqs)
    K_DM = 4.148808e3  # MHz^2 cm^3 pc^-1 s

    cos_harmonics = [matrix.jnparray(jnp.cos(k * binphase)) for k in range(1, n_harmonics + 1)]
    sin_harmonics = [matrix.jnparray(jnp.sin(k * binphase)) for k in range(1, n_harmonics + 1)]

    if n_harmonics == 4:
        def delay(cos1, sin1, cos2, sin2, cos3, sin3, cos4, sin4):
            series = (cos1 * cos_harmonics[0] + sin1 * sin_harmonics[0] +
                        cos2 * cos_harmonics[1] + sin2 * sin_harmonics[1] +
                        cos3 * cos_harmonics[2] + sin3 * sin_harmonics[2] +
                        cos4 * cos_harmonics[3] + sin4 * sin_harmonics[3])
            return K_DM * series / freqs**2
    elif n_harmonics == 8:
        def delay(cos1, sin1, cos2, sin2, cos3, sin3, cos4, sin4,
                  cos5, sin5, cos6, sin6, cos7, sin7, cos8, sin8):
            series = (cos1 * cos_harmonics[0] + sin1 * sin_harmonics[0] +
                        cos2 * cos_harmonics[1] + sin2 * sin_harmonics[1] +
                        cos3 * cos_harmonics[2] + sin3 * sin_harmonics[2] +
                        cos4 * cos_harmonics[3] + sin4 * sin_harmonics[3] +
                        cos5 * cos_harmonics[4] + sin5 * sin_harmonics[4] +
                        cos6 * cos_harmonics[5] + sin6 * sin_harmonics[5] +
                        cos7 * cos_harmonics[6] + sin7 * sin_harmonics[6] +
                        cos8 * cos_harmonics[7] + sin8 * sin_harmonics[7])
            return K_DM * series / freqs**2
    elif n_harmonics == 16:
        def delay(cos1, sin1, cos2, sin2, cos3, sin3, cos4, sin4,
                  cos5, sin5, cos6, sin6, cos7, sin7, cos8, sin8,
                  cos9, sin9, cos10, sin10, cos11, sin11, cos12, sin12,
                  cos13, sin13, cos14, sin14, cos15, sin15, cos16, sin16):
            series = (cos1 * cos_harmonics[0] + sin1 * sin_harmonics[0] +
                      cos2 * cos_harmonics[1] + sin2 * sin_harmonics[1] +
                      cos3 * cos_harmonics[2] + sin3 * sin_harmonics[2] +
                      cos4 * cos_harmonics[3] + sin4 * sin_harmonics[3] +
                      cos5 * cos_harmonics[4] + sin5 * sin_harmonics[4] +
                      cos6 * cos_harmonics[5] + sin6 * sin_harmonics[5] +
                      cos7 * cos_harmonics[6] + sin7 * sin_harmonics[6] +
                      cos8 * cos_harmonics[7] + sin8 * sin_harmonics[7] +
                      cos9 * cos_harmonics[8] + sin9 * sin_harmonics[8] +
                      cos10 * cos_harmonics[9] + sin10 * sin_harmonics[9] +
                      cos11 * cos_harmonics[10] + sin11 * sin_harmonics[10] +
                      cos12 * cos_harmonics[11] + sin12 * sin_harmonics[11] +
                      cos13 * cos_harmonics[12] + sin13 * sin_harmonics[12] +
                      cos14 * cos_harmonics[13] + sin14 * sin_harmonics[13] +
                      cos15 * cos_harmonics[14] + sin15 * sin_harmonics[14] +
                      cos16 * cos_harmonics[15] + sin16 * sin_harmonics[15])
            return K_DM * series / freqs**2     
    else:
        raise ValueError(f"n_harmonics={n_harmonics} not supported; use 4 or 8 or 16")

    return delay