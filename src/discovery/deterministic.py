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


def makedelay_binary_phases(pulsarterm=True):
    """Factory for computing cphase and sphase vectors from binary parameters."""
    def delay_binary_phases(toas, log10_f0):
        """Compute cosine and sine phase vectors.

        Returns:
            cphase: cosine phase vector (toas,)
            sphase: sine phase vector (toas,)
        """
        f0 = 10**log10_f0
        tref = 86400.0 * 51544.5  # MJD J2000 in seconds

        # Compute base phase vectors from frequency evolution only
        phase_base = 2.0 * jnp.pi * f0 * (toas - tref)
        cphase = jnp.cos(phase_base)
        sphase = jnp.sin(phase_base)

        return jnp.array([cphase, sphase])

    return delay_binary_phases


def makedelay_binary_coefficients(pulsarterm=True):
    """Factory for computing coefficients to multiply phase vectors."""
    def delay_binary_coefficients(pos, log10_h0, log10_f0, ra, sindec, cosinc, psi, phi_earth, phi_psr):
        """Compute antenna pattern factors and phase coefficients.

        Returns:
            coeffs: dictionary with keys:
                - 'fplus', 'fcross': antenna pattern factors
                - 'rplus_coeff_c', 'rplus_coeff_s': coefficients for cphase/sphase in rplus
                - 'rcross_coeff_c', 'rcross_coeff_s': coefficients for cphase/sphase in rcross

            Full response is reconstructed as:
            res = -fplus * (rplus_coeff_c * cphase + rplus_coeff_s * sphase)
                  -fcross * (rcross_coeff_c * cphase + rcross_coeff_s * sphase)
        """
        h0 = 10**log10_h0
        f0 = 10**log10_f0

        dec, inc = jnp.arcsin(sindec), jnp.arccos(cosinc)

        # calculate antenna pattern (note: pos is pulsar sky position unit vector)
        fplus, fcross = fpc_fast(pos, 0.5 * jnp.pi - dec, ra)  # careful with dec -> gwtheta conversion

        # Calculate coefficients that multiply cphase and sphase
        # Apply addition theorem: cos(phi_avg + phase_base) = cos(phi_avg)*cos(phase_base) - sin(phi_avg)*sin(phase_base)
        # Original cphase_orig = cos(phi_avg + phase_base)
        # Original sphase_orig = sin(phi_avg + phase_base)
        if pulsarterm:
            phi_avg = 0.5 * (phi_earth + phi_psr)
            phi_diff = 0.5 * (phi_earth - phi_psr)

            cos_avg = jnp.cos(phi_avg)
            sin_avg = jnp.sin(phi_avg)
            sin_diff = jnp.sin(phi_diff)

            # cphase_orig = cos_avg * cphase - sin_avg * sphase
            # sphase_orig = sin_avg * cphase + cos_avg * sphase
            # delta_sin =  2.0 * cphase_orig * sin_diff
            # delta_cos = -2.0 * sphase_orig * sin_diff

            c_coeff_sin = 2.0 * cos_avg * sin_diff    # coefficient for cphase in delta_sin
            s_coeff_sin = -2.0 * sin_avg * sin_diff   # coefficient for sphase in delta_sin
            c_coeff_cos = -2.0 * sin_avg * sin_diff   # coefficient for cphase in delta_cos
            s_coeff_cos = -2.0 * cos_avg * sin_diff   # coefficient for sphase in delta_cos
        else:
            # cphase_orig = cos(phi_earth + phase_base) = cos(phi_earth)*cphase - sin(phi_earth)*sphase
            # sphase_orig = sin(phi_earth + phase_base) = sin(phi_earth)*cphase + cos(phi_earth)*sphase
            # delta_sin = sphase_orig
            # delta_cos = cphase_orig
            cos_earth = jnp.cos(phi_earth)
            sin_earth = jnp.sin(phi_earth)

            c_coeff_sin = sin_earth    # coefficient for cphase in delta_sin
            s_coeff_sin = cos_earth    # coefficient for sphase in delta_sin
            c_coeff_cos = cos_earth    # coefficient for cphase in delta_cos
            s_coeff_cos = -sin_earth   # coefficient for sphase in delta_cos

        # At = -1.0 * (1.0 + cos(inc)^2) * delta_sin
        # Bt = 2.0 * cos(inc) * delta_cos
        cos_inc = jnp.cos(inc)
        At_coeff_c = -1.0 * (1.0 + cos_inc**2) * c_coeff_sin
        At_coeff_s = -1.0 * (1.0 + cos_inc**2) * s_coeff_sin
        Bt_coeff_c = 2.0 * cos_inc * c_coeff_cos
        Bt_coeff_s = 2.0 * cos_inc * s_coeff_cos

        alpha = h0 / (2 * jnp.pi * f0)
        cos_2psi = jnp.cos(2 * psi)
        sin_2psi = jnp.sin(2 * psi)

        # rplus = alpha * (-At * cos(2*psi) + Bt * sin(2*psi))
        # rcross = alpha * (At * sin(2*psi) + Bt * cos(2*psi))

        # Coefficient for cphase in rplus: alpha * (-At_coeff_c * cos(2*psi) + Bt_coeff_c * sin(2*psi))
        rplus_coeff_c = alpha * (-At_coeff_c * cos_2psi + Bt_coeff_c * sin_2psi)
        # Coefficient for sphase in rplus: alpha * (-At_coeff_s * cos(2*psi) + Bt_coeff_s * sin(2*psi))
        rplus_coeff_s = alpha * (-At_coeff_s * cos_2psi + Bt_coeff_s * sin_2psi)

        # Coefficient for cphase in rcross: alpha * (At_coeff_c * sin(2*psi) + Bt_coeff_c * cos(2*psi))
        rcross_coeff_c = alpha * (At_coeff_c * sin_2psi + Bt_coeff_c * cos_2psi)
        # Coefficient for sphase in rcross: alpha * (At_coeff_s * sin(2*psi) + Bt_coeff_s * cos(2*psi))
        rcross_coeff_s = alpha * (At_coeff_s * sin_2psi + Bt_coeff_s * cos_2psi)

        Ac = -fplus * rplus_coeff_c - fcross * rcross_coeff_c
        As = -fplus * rplus_coeff_s - fcross * rcross_coeff_s

        return jnp.array([Ac, As])

    if not pulsarterm:
        delay_binary_coefficients = functools.partial(delay_binary_coefficients, phi_psr=jnp.nan)

    return delay_binary_coefficients


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
    r"""
    Factory function for chromatic exponential delay model.

    Creates a delay function that models chromatic exponential events (e.g., profile
    state changes) with frequency-dependent amplitude scaling.

    Parameters
    ----------
    psr : Pulsar
        Pulsar object containing toas and freqs attributes
    fref : float, optional
        Reference frequency in MHz for normalization (default: 1400.0)

    Returns
    -------
    delay : callable
        Function with signature (t0, log10_Amp, log10_tau, sign_param, alpha) -> ndarray
        Computes chromatic exponential delay:

        .. math::

            \Delta(t) = \pm A_0 \exp\left(-\frac{t - t_0}{\tau}\right) \left(\frac{f_{\text{ref}}}{f}\right)^\alpha H(t - t_0)

        where :math:`H(t - t_0)` is the Heaviside step function.
    """
    toas, fnorm = matrix.jnparray(psr.toas / const.day), matrix.jnparray(fref / psr.freqs)

    def delay(t0, log10_Amp, log10_tau, sign_param, alpha):
        dt = toas - t0
        tau = 10**log10_tau
        amp = 10**log10_Amp
        return jnp.sign(sign_param) * amp * fnorm**alpha * jnp.where(dt >= 0, jnp.exp(-dt / tau), 0.0 )

    delay.__name__ = "chromatic_exponential_delay"
    return delay


def chromatic_annual(psr, fref=1400.0):
    r"""
    Factory function for chromatic annual delay model.

    Creates a delay function that models chromatic annual sinusoidal variations
    (e.g., annual DM or scattering variations) with frequency-dependent amplitude scaling.

    Parameters
    ----------
    psr : Pulsar
        Pulsar object containing toas and freqs attributes
    fref : float, optional
        Reference frequency in MHz for normalization (default: 1400.0)

    Returns
    -------
    delay : callable
        Function with signature (log10_Amp, phase, alpha) -> ndarray
        Computes chromatic annual delay:

        .. math::

            \Delta(t) = A_0 \sin(2\pi f_{\text{yr}} t + \phi) \left(\frac{f_{\text{ref}}}{f}\right)^\alpha

        where :math:`f_{\text{yr}}` is the annual frequency (1/year).
    """
    toas, fnorm = matrix.jnparray(psr.toas), matrix.jnparray(fref / psr.freqs)

    def delay(log10_Amp, phase, alpha):
        return 10**log10_Amp * jnp.sin(2*jnp.pi * const.fyr * toas + phase) * fnorm**alpha

    delay.__name__ = "chromatic_annual_delay"
    return delay


def chromatic_gaussian(psr, fref=1400.0):
    r"""
    Factory function for chromatic Gaussian delay model.

    Creates a delay function that models chromatic Gaussian events (e.g., transient
    DM variations, localized profile events) with frequency-dependent amplitude scaling.

    Parameters
    ----------
    psr : Pulsar
        Pulsar object containing toas and freqs attributes
    fref : float, optional
        Reference frequency in MHz for normalization (default: 1400.0)

    Returns
    -------
    delay : callable
        Function with signature (t0, log10_Amp, log10_sigma, sign_param, alpha) -> ndarray
        Computes chromatic Gaussian delay:

        .. math::

            \Delta(t) = \pm A_0 \exp\left(-\frac{(t - t_0)^2}{2\sigma^2}\right) \left(\frac{f_{\text{ref}}}{f}\right)^\alpha
    """
    toas, fnorm = matrix.jnparray(psr.toas / const.day), matrix.jnparray(fref / psr.freqs)

    def delay(t0, log10_Amp, log10_sigma, sign_param, alpha):
        return jnp.sign(sign_param) * 10**log10_Amp * jnp.exp(-(toas - t0)**2 / (2 * (10**log10_sigma)**2)) * fnorm**alpha

    delay.__name__ = "chromatic_gaussian_delay"
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
    r"""
    Factory function for orthometric Shapiro delay model.

    Creates a delay function that models Shapiro delay in binary pulsars using
    the orthometric parameterization from Freire & Wex (2010).

    Parameters
    ----------
    psr : Pulsar
        Pulsar object containing toas attribute
    binphase : array-like
        Binary orbital phase :math:`\Phi` at each TOA (same shape as psr.toas)
    eps_stig : float, optional
        Lower/upper clipping bound for ``stig`` for numerical stability (default: 1e-6)
    eps_log : float, optional
        Floor applied to the log argument for numerical stability (default: 1e-10)

    Returns
    -------
    delay : callable
        Function with signature (h3, stig) -> ndarray
        Computes orthometric Shapiro delay (Equation 29 in Freire & Wex 2010):

        .. math::

            \Delta_s = -\frac{2 h_3}{\zeta^3} \log(1 + \zeta^2 - 2 \zeta \sin\Phi)

    Raises
    ------
    ValueError
        If binphase shape does not match psr.toas shape

    References
    ----------
    Freire, P. C. C., & Wex, N. (2010). The orthometric parametrization of the
    Shapiro delay and an improved test of general relativity with binary pulsars.
    MNRAS, 409(1), 199-212.
    """
    toas, binphase = matrix.jnparray(psr.toas / const.day), matrix.jnparray(binphase)
    if not np.shape(binphase) == np.shape(toas):
        raise ValueError("Input binphase must have the same shape as toas")

    def delay(h3, stig):
        stig_clipped = jnp.clip(stig, eps_stig, 1.0 - eps_stig)
        log_arg = 1 + stig_clipped**2 - 2 * stig_clipped * jnp.sin(binphase)
        log_arg = jnp.maximum(log_arg, eps_log)
        return -(2.0 * h3 / stig_clipped**3) * jnp.log(log_arg)

    delay.__name__ = "orthometric_shapiro_delay"
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