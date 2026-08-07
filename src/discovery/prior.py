import re
import warnings
from collections import namedtuple

import jax
import numpy as np
import pandas as pd
import scipy.linalg
import scipy.special
import scipy.stats

from . import matrix
jnp = matrix.jnp

LOG2PI = float(np.log(2.0 * np.pi))


def uniform(par, a, b):
    def logpriorfunc(params):
        x = params[par]
        # jnp.sum collapses an array-valued parameter (e.g. log10_rho(30)) to a
        # scalar contribution; for a scalar parameter it is a no-op.
        return matrix.jnp.sum(matrix.jnp.where(matrix.jnp.logical_and(x >= a, x <= b), 0.0, -matrix.jnp.inf))

    return logpriorfunc


priordict_standard = {
    "(.*_)?efac": [0.1, 10],
    "(.*_)?t2equad": [-8.5, -5],
    "(.*_)?tnequad": [-8.5, -5],
    "(.*_)?log10_ecorr": [-10, -5],  # also matches the Legendre mode amplitudes ..._log10_ecorr_k{m}
    r"(.*_)?ecorr_corr_k[0-9]+k[0-9]+": [-1, 1],  # (partial) correlations of the Legendre mode covariance (makegp_ecorr_legendre_correlated); C-vine parametrisation, valid PSD for any values in (-1, 1)
    "(.*_)?rednoise_log10_A.*": [-20, -11],
    "(.*_)?rednoise_gamma.*": [0, 7],
    "(.*_)?rednoise_log10_fb": [-9, -6],
    "(.*_)?red_noise_log10_A.*": [-20, -11],  # deprecated
    "(.*_)?red_noise_gamma.*": [0, 7],  # deprecated
    "(.*_)?red_noise_log10_fb": [-9, -6],
    "(.*_)?sw_gp_log10_A": [-10, -2],
    "(.*_)?sw_gp_gamma": [0, 4],
    "crn_log10_A.*": [-18, -11],
    "crn_gamma.*": [0, 7],
    "crn_log10_fb": [-9, -6],
    "gw_(.*_)?log10_A": [-18, -11],
    "gw_(.*_)?gamma": [0, 7],
    "gw_log10_fb": [-9, -6],
    "(.*_)?dmgp_log10_A": [-20, -11],
    "(.*_)?dmgp_gamma": [0, 7],
    "(.*_)?dmgp_alpha": [1, 3],
    "(.*_)?chromgp_log10_A": [-20, -11],
    "(.*_)?chromgp_gamma": [0, 7],
    "(.*_)?chromgp_alpha": [1, 7],
    "(.*_)?dm_gp_log10_A": [-20, -11],
    "(.*_)?dm_gp_gamma": [0, 7],
    "(.*_)?dm_gp_alpha": [1, 3],
    "(.*_)?chrom_gp_log10_A": [-20, -11],
    "(.*_)?chrom_gp_gamma": [0, 7],
    "(.*_)?chrom_gp_alpha": [2.5, 14],  # scattering noise. Should be steeper than DM
    "crn_log10_rho": [-9, -4],
    "gw_(.*_)?log10_rho": [-9, -4],
    r"(.*_)?red_noise_log10_rho\(([0-9]*)\)": [-9, -4],
    r"(.*_)?red_noise_crn_log10_rho\(([0-9]*)\)": [-9, -4],
    "cw_ra": [0, 2*np.pi],
    "cw_dec": [-0.5*np.pi, 0.5*np.pi],
    "cw_inc": [0, np.pi],
    "cw_sindec": [-1.0, 1.0],
    "cw_cosinc": [-1.0, 1.0],
    "cw_psi": [0, np.pi],
    "cw_log10_f0": [-9.0, -7.0],
    "cw_log10_h0": [-18.0, -11.0],
    "cw_phi_earth": [0., 2*np.pi],
    "(.*_)?cw_phi_psr": [0., 2*np.pi],
    "(.*_)?chrom_exp_t0": [50000, 65000],
    "(.*_)?chrom_exp_log10_Amp": [-10, -4],
    "(.*_)?chrom_exp_log10_tau": [0, 4],
    "(.*_)?chrom_exp_sign_param": [-1, 1],
    "(.*_)?chrom_exp_alpha": [0, 7],
    "(.*_)?chrom_1yr_log10_Amp": [-10, -4],
    "(.*_)?chrom_1yr_phase": [0, 2 * np.pi],
    "(.*_)?chrom_1yr_alpha": [0, 7],
    "(.*_)?chrom_gauss_t0": [50000, 65000],
    "(.*_)?chrom_gauss_log10_Amp": [-10, -4],
    "(.*_)?chrom_gauss_log10_sigma": [0, 4],
    "(.*_)?chrom_gauss_sign_param": [-1, 1],
    "(.*_)?chrom_gauss_alpha": [0, 7],
    "(.*_)?h3": [0.0, 10**-5],
    "(.*_)?stig": [0.0, 1.0]
}


# joint priors keyed by a tuple of parameter regexes, merged into the jointpriors
# argument of makelogtransform the way priordict_standard is merged into priordict
jointpriors_standard = {}


# --- prior families ---------------------------------------------------------

# name:    family name used as the trailing tag of a prior specification
# nargs:   number of numbers the specification carries before the tag
# support: (lo, hi) arrays bounding the density; the unconstrained link is chosen
#          from these, so it must cover every component of a mixture
# prepare: build-time constants derived from args, passed to logpdf as `extra`
# logpdf:  log density in the physical variable x
# loc:     location mapped to y = 0
# scale:   characteristic width of the density, used to scale the link
_Family = namedtuple('_Family', 'name nargs support prepare logpdf loc scale')

_FAMILIES = {}


def register_family(name, nargs, support, prepare, logpdf, loc, scale):
    """Add a prior family to the registry consulted by parse_spec.

    name:    trailing string tag of a prior specification using this family
    nargs:   count of leading numbers in the specification
    support: args -> (lo, hi) arrays
    prepare: args -> tuple of build-time constants passed to logpdf
    logpdf:  (x, args, extra) -> log density in the physical variable
    loc:     args -> location placed at y = 0
    scale:   args -> characteristic width
    """
    _FAMILIES[name] = _Family(name, nargs, support, prepare, logpdf, loc, scale)


def _log_ndtr_diff_np(al, be):
    """log(Phi(be) - Phi(al)) for be > al, evaluated in numpy.

    Mirrors the inputs so that both log_ndtr calls are made in the lower tail,
    where log_ndtr is accurate; the naive difference of two Phi cancels.
    """
    al, be = np.asarray(al, dtype=float), np.asarray(be, dtype=float)
    flip = (al + be) > 0.0
    u = np.where(flip, -be, al)
    v = np.where(flip, -al, be)
    lu, lv = scipy.special.log_ndtr(u), scipy.special.log_ndtr(v)
    return lv + np.log(-np.expm1(lu - lv))


def _log_ndtr_diff(al, be):
    """log(Phi(be) - Phi(al)) for be > al, differentiable in JAX.

    Mirrors the inputs rather than selecting between two computed branches, so
    that no -inf from an unselected branch reaches the reverse-mode gradient.
    """
    import jax.scipy.special as jsp_special

    flip = (al + be) > 0.0
    u = jnp.where(flip, -be, al)
    v = jnp.where(flip, -al, be)
    lu, lv = jsp_special.log_ndtr(u), jsp_special.log_ndtr(v)
    return lv + jnp.log(-jnp.expm1(lu - lv))


def _mix_logweights(chi):
    """Return (log(1 - chi), log(chi)) for an outlier weight chi in [0, 1]."""
    chi = np.asarray(chi, dtype=float)
    if np.any(chi < 0.0) or np.any(chi > 1.0):
        raise ValueError(f'Outlier weight chi must lie in [0, 1], got {chi}.')
    tiny = np.finfo(float).tiny
    chi = np.clip(chi, tiny, 1.0 - np.finfo(float).eps)
    return np.log1p(-chi), np.log(chi)


def _inbox(x, lo, hi):
    return (x >= lo) & (x <= hi)


def _warn_mixture_step(label, logcore, logout):
    """Warn when a mixture's density steps at the edge of its core component.

    logcore, logout: weighted log densities of the two components, evaluated on
    the core boundary where the core density is largest
    """
    jump = float(np.log1p(np.exp(np.max(logcore) - logout)))
    if jump > 1.0:
        warnings.warn(f'{label}: the outlier mixture steps by {jump:.2f} nats at the core '
                      f'boundary, which Hamiltonian samplers cross with acceptance '
                      f'{np.exp(-jump):.3f}.')


def _mixture_step(name, args, extra):
    """Return (logcore, logout) on the core boundary, or None if the core fills the support.

    Only families whose core is bounded and whose outlier component is strictly
    wider can step; elsewhere the density falls to zero at the support edge, as
    it does for any truncated prior.
    """
    if name == 'UniformWithOutliers':
        a, b, chi, A, B = args
        edges = np.stack([a, b])
        logcore = np.broadcast_to(extra[0], edges.shape)
    elif name == 'TruncatedNormalWithOutliers':
        mu, sd, a, b, chi, A, B = args
        edges = np.stack([a, b])
        z = (edges - mu) / sd
        logcore = -0.5 * z * z + extra[0]
    else:
        return None

    if not (np.any(A < a) or np.any(B > b)):
        return None

    return logcore, np.min(extra[1])


def _uniform_prepare(args):
    a, b = args
    # a zero-width range fixes the parameter at its midpoint; its coordinate keeps
    # the density of a unit-width range, so it stays bounded under sampling
    width = np.where(b == a, 2.0, b - a)
    return (-np.log(width),)


# Uniform: [a, b] or [a, b, 'Uniform'].
register_family(
    'Uniform', 2,
    support=lambda args: (args[0], args[1]),
    prepare=_uniform_prepare,
    logpdf=lambda x, args, extra: jnp.broadcast_to(extra[0], jnp.shape(x)),
    loc=lambda args: 0.5 * (args[0] + args[1]),
    scale=lambda args: 0.5 * (args[1] - args[0]),
)


def _normal_logpdf(x, args, extra):
    mu, sd = args
    z = (x - mu) / sd
    return -0.5 * z * z + extra[0]


# Normal: [mean, std, 'Normal']. Support is unbounded, so the link is affine and
# the transformed density is exactly N(0, 1) whatever the mean and width.
register_family(
    'Normal', 2,
    support=lambda args: (np.full(np.shape(args[0]), -np.inf), np.full(np.shape(args[0]), np.inf)),
    prepare=lambda args: (-np.log(args[1]) - 0.5 * LOG2PI,),
    logpdf=_normal_logpdf,
    loc=lambda args: args[0],
    scale=lambda args: args[1],
)


def _truncnorm_prepare(args):
    mu, sd, a, b = args
    logZ = _log_ndtr_diff_np((a - mu) / sd, (b - mu) / sd)
    return (-np.log(sd) - 0.5 * LOG2PI - logZ,)


def _truncnorm_logpdf(x, args, extra):
    mu, sd, a, b = args
    z = (x - mu) / sd
    return -0.5 * z * z + extra[0]


# TruncatedNormal: [mean, std, minval, maxval, 'TruncatedNormal']. The link maps
# onto [minval, maxval], so the normalising constant is a build-time constant.
register_family(
    'TruncatedNormal', 4,
    support=lambda args: (args[2], args[3]),
    prepare=_truncnorm_prepare,
    logpdf=_truncnorm_logpdf,
    loc=lambda args: np.clip(args[0], args[2], args[3]),
    scale=lambda args: args[1],
)


def _unif_out_prepare(args):
    a, b, chi, A, B = args
    lfg, lout = _mix_logweights(chi)
    return (lfg - np.log(b - a), lout - np.log(B - A))


def _unif_out_logpdf(x, args, extra):
    a, b, chi, A, B = args
    core = jnp.where(_inbox(x, a, b), extra[0], -jnp.inf)
    out = jnp.where(_inbox(x, A, B), extra[1], -jnp.inf)
    return jnp.logaddexp(core, out)


# UniformWithOutliers: [minval, maxval, chi, outmin, outmax, 'UniformWithOutliers'].
# chi is the outlier weight. The link covers the union of both boxes.
register_family(
    'UniformWithOutliers', 5,
    support=lambda args: (np.minimum(args[0], args[3]), np.maximum(args[1], args[4])),
    prepare=_unif_out_prepare,
    logpdf=_unif_out_logpdf,
    loc=lambda args: 0.5 * (args[0] + args[1]),
    scale=lambda args: 0.5 * (args[1] - args[0]),
)


def _norm_out_prepare(args):
    mu, sd, chi, A, B = args
    lfg, lout = _mix_logweights(chi)
    return (lfg - np.log(sd) - 0.5 * LOG2PI, lout - np.log(B - A))


def _norm_out_logpdf(x, args, extra):
    mu, sd, chi, A, B = args
    z = (x - mu) / sd
    core = -0.5 * z * z + extra[0]
    out = jnp.where(_inbox(x, A, B), extra[1], -jnp.inf)
    return jnp.logaddexp(core, out)


# NormalWithOutliers: [mean, std, chi, outmin, outmax, 'NormalWithOutliers'].
# The Gaussian core is unbounded, so the union support is the whole line.
register_family(
    'NormalWithOutliers', 5,
    support=lambda args: (np.full(np.shape(args[0]), -np.inf), np.full(np.shape(args[0]), np.inf)),
    prepare=_norm_out_prepare,
    logpdf=_norm_out_logpdf,
    loc=lambda args: args[0],
    scale=lambda args: args[1],
)


def _tnorm_out_prepare(args):
    mu, sd, a, b, chi, A, B = args
    logZ = _log_ndtr_diff_np((a - mu) / sd, (b - mu) / sd)
    lfg, lout = _mix_logweights(chi)
    return (lfg - np.log(sd) - 0.5 * LOG2PI - logZ, lout - np.log(B - A))


def _tnorm_out_logpdf(x, args, extra):
    mu, sd, a, b, chi, A, B = args
    z = (x - mu) / sd
    core = jnp.where(_inbox(x, a, b), -0.5 * z * z + extra[0], -jnp.inf)
    out = jnp.where(_inbox(x, A, B), extra[1], -jnp.inf)
    return jnp.logaddexp(core, out)


# TruncatedNormalWithOutliers:
# [mean, std, minval, maxval, chi, outmin, outmax, 'TruncatedNormalWithOutliers'].
register_family(
    'TruncatedNormalWithOutliers', 7,
    support=lambda args: (np.minimum(args[2], args[5]), np.maximum(args[3], args[6])),
    prepare=_tnorm_out_prepare,
    logpdf=_tnorm_out_logpdf,
    loc=lambda args: np.clip(args[0], args[2], args[3]),
    scale=lambda args: args[1],
)


def _norm_norm_out_prepare(args):
    mu, sd, chi, sd_out = args
    lfg, lout = _mix_logweights(chi)
    return (lfg - np.log(sd) - 0.5 * LOG2PI, lout - np.log(sd_out) - 0.5 * LOG2PI)


def _norm_norm_out_logpdf(x, args, extra):
    mu, sd, chi, sd_out = args
    z, zo = (x - mu) / sd, (x - mu) / sd_out
    return jnp.logaddexp(-0.5 * z * z + extra[0], -0.5 * zo * zo + extra[1])


# NormalWithNormalOutliers: [mean, std, chi, outlierstd, 'NormalWithNormalOutliers'].
# Both components are smooth, so the density has no step for the sampler to cross.
register_family(
    'NormalWithNormalOutliers', 4,
    support=lambda args: (np.full(np.shape(args[0]), -np.inf), np.full(np.shape(args[0]), np.inf)),
    prepare=_norm_norm_out_prepare,
    logpdf=_norm_norm_out_logpdf,
    loc=lambda args: args[0],
    scale=lambda args: args[1],
)


# --- prior specifications ---------------------------------------------------

def parse_spec(spec):
    """Split a prior specification into a family name and its numeric arguments.

    spec: [minval, maxval], or a list whose last element is a family name and
          whose leading elements are that family's numeric arguments
    """
    if not isinstance(spec, (list, tuple)):
        raise TypeError(f'Prior specification must be a list or tuple, got {spec!r}.')

    if spec and isinstance(spec[-1], str):
        name, args = spec[-1], tuple(spec[:-1])
        if name not in _FAMILIES:
            raise KeyError(f'Unknown prior family {name!r}; known families are '
                           f'{sorted(_FAMILIES)}.')
    elif len(spec) == 2:
        name, args = 'Uniform', tuple(spec)
    else:
        raise ValueError(f'Prior specification {spec!r} is neither [minval, maxval] nor '
                         f'a list ending in a family name.')

    family = _FAMILIES[name]
    if len(args) != family.nargs:
        raise ValueError(f"Prior family {name!r} takes {family.nargs} numbers, "
                         f"got {len(args)} in {spec!r}.")
    if any(isinstance(arg, str) for arg in args):
        raise TypeError(f'Prior specification {spec!r} has a string where a number is expected.')

    return name, args


def _matchprior(par, priordict):
    """Return the prior specification for par, matching keys as regexes in order."""
    for pname, spec in priordict.items():
        if re.match(pname, par):
            return spec

    raise KeyError(f'No known prior for {par}.')


def _uniform_range(par, priordict):
    """Return [minval, maxval] for par, refusing any family that is not Uniform."""
    spec = _matchprior(par, priordict)
    name, args = parse_spec(spec)
    if name != 'Uniform':
        raise ValueError(f"Prior for {par} is a {name} prior, which has no uniform range. "
                         f"Use getsupport() for its support, or sample_prior() to draw from it.")

    return list(args)


def getprior_uniform(par, priordict={}):
    """Return [minval, maxval] for par. Raises if the prior is not Uniform."""
    return _uniform_range(par, {**priordict_standard, **priordict})


def getsupport(par, priordict={}):
    """Return (lo, hi) bounding the prior for par, for any family."""
    priordict = {**priordict_standard, **priordict}
    name, args = parse_spec(_matchprior(par, priordict))
    lo, hi = _FAMILIES[name].support(tuple(np.asarray(a, dtype=float) for a in args))

    return float(lo), float(hi)


def makelogprior_uniform(params, priordict={}):
    """Build a log-prior over a dictionary of physical parameter values.

    params:    parameter names to include
    priordict: prior specifications, overriding priordict_standard
    """
    priordict = {**priordict_standard, **priordict}

    priors = []
    for par in params:
        for parname, spec in priordict.items():
            if re.match(parname, par):
                name, args = parse_spec(spec)
                if name == 'Uniform':
                    priors.append(uniform(par, *args))
                else:
                    priors.append(_makelogprior_family(par, name, args))
                break

    def logprior(params):
        return sum(prior(params) for prior in priors)

    return logprior


makelogprior = makelogprior_uniform


def _makelogprior_family(par, name, args):
    family = _FAMILIES[name]
    args = tuple(np.asarray(a, dtype=float) for a in args)
    extra = family.prepare(args)

    def logpriorfunc(params):
        return jnp.sum(family.logpdf(params[par], args, extra))

    return logpriorfunc


# --- unconstrained links ----------------------------------------------------

def _tanh_link_params(lo, hi, loc, scale, whiten):
    """Return (m, h, y0, kappa, hj) for x = m + h*tanh(kappa*y + y0).

    y0 places the family's location at y = 0; kappa matches the local width of
    the map at that point to the family's own scale. A zero-width support fixes
    the parameter at m, and hj carries a unit width so that the map contributes
    nothing to the log-Jacobian.
    """
    m, h = 0.5 * (lo + hi), 0.5 * (hi - lo)
    fixed = (h == 0.0)
    hj = np.where(fixed, 1.0, h)

    t = np.clip((loc - m) / hj, -1.0 + np.finfo(float).eps, 1.0 - np.finfo(float).eps)
    t = np.where(fixed, 0.0, t)
    y0 = np.arctanh(t)

    if whiten:
        kappa = np.clip(scale / (hj * (1.0 - t * t)), 1e-3, 1e3)
    else:
        kappa = np.ones_like(hj)
    kappa = np.where(fixed | (scale == 0.0), 1.0, kappa)

    return m, h, y0, kappa, hj


def _tanh_forward(y, m, h, y0, kappa, hj):
    u = kappa * y + y0
    x = m + h * jnp.tanh(u)
    logjac = jnp.log(hj * kappa) + jnp.log(4.0) - 2.0 * jnp.logaddexp(u, -u)

    return x, logjac


def _tanh_inverse(x, m, h, y0, kappa, hj):
    t = (x - m) / hj
    # arctanh(+/-1) is infinite; pull |t| == 1 back to the largest representable
    # magnitude below one, leaving |t| > 1 to produce NaN as an out-of-prior signal.
    t = jnp.where(jnp.abs(t) == 1.0, jnp.sign(t) * (1.0 - 0.5 * jnp.finfo(t.dtype).eps), t)

    return (jnp.arctanh(t) - y0) / kappa


def _affine_forward(y, loc, scale):
    return loc + scale * y, jnp.broadcast_to(jnp.log(scale), jnp.shape(y))


def _affine_inverse(x, loc, scale):
    return (x - loc) / scale


_Block = namedtuple('_Block', 'family idx args extra link linkargs')


def _make_block(name, idx, specs, whiten, labels=None):
    family = _FAMILIES[name]
    args = tuple(np.asarray([spec[i] for spec in specs], dtype=float) for i in range(family.nargs))
    extra = family.prepare(args)

    step = _mixture_step(name, args, extra)
    if step is not None:
        _warn_mixture_step(f'Prior {name}' if labels is None else f'Prior for {labels[0]}', *step)

    lo, hi = family.support(args)
    lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    loc = np.asarray(family.loc(args), dtype=float)
    scale = np.asarray(family.scale(args), dtype=float)

    if np.all(np.isfinite(lo)) and np.all(np.isfinite(hi)):
        linkargs = tuple(matrix.jnparray(v) for v in _tanh_link_params(lo, hi, loc, scale, whiten))
        link = 'tanh'
    elif np.all(np.isinf(lo)) and np.all(np.isinf(hi)):
        linkargs = (matrix.jnparray(loc), matrix.jnparray(scale))
        link = 'affine'
    else:
        raise ValueError(f'Prior family {name!r} has a half-infinite support, which has no link.')

    return _Block(family, np.asarray(idx, dtype=int), tuple(matrix.jnparray(a) for a in args),
                  tuple(matrix.jnparray(np.asarray(e, dtype=float)) for e in extra), link, linkargs)


def _block_forward(blk, y):
    if blk.link == 'tanh':
        return _tanh_forward(y, *blk.linkargs)

    return _affine_forward(y, *blk.linkargs)


def _block_inverse(blk, x):
    if blk.link == 'tanh':
        return _tanh_inverse(x, *blk.linkargs)

    return _affine_inverse(x, *blk.linkargs)


# --- joint priors -----------------------------------------------------------

def _chol_rho(rho, name):
    rho = np.asarray(rho, dtype=float)
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError(f'Joint prior {name}: correlation matrix must be square.')
    if not np.allclose(np.diag(rho), 1.0):
        raise ValueError(f'Joint prior {name}: correlation matrix must have unit diagonal.')
    if not np.allclose(rho, rho.T):
        raise ValueError(f'Joint prior {name}: correlation matrix must be symmetric.')
    if np.min(np.linalg.eigvalsh(rho)) <= 0.0:
        raise ValueError(f'Joint prior {name}: correlation matrix is not positive definite.')

    return np.linalg.cholesky(rho)


def _mvn_box_logmass(mu, sigma, rho, bounds):
    """log of the Gaussian mass inside an axis-aligned box, by inclusion-exclusion.

    mu, sigma: mean and marginal widths
    rho:       correlation matrix
    bounds:    (d, 2) array of per-coordinate limits
    """
    d = len(mu)
    cov = np.outer(sigma, sigma) * rho
    mvn = scipy.stats.multivariate_normal(mean=np.asarray(mu, dtype=float), cov=cov,
                                          allow_singular=False)

    total = 0.0
    for corner in range(2 ** d):
        pick = [(corner >> j) & 1 for j in range(d)]
        sign = (-1.0) ** (d - sum(pick))
        total += sign * mvn.cdf(np.array([bounds[j][pick[j]] for j in range(d)]))

    return total


# name:    label used in messages
# idx:     (n_inst, d) flat offsets of the group's members in the sampled vector
# forward: (ys, xs) -> (physical values, log-prior), with xs supplying the
#          physical values of any hyperparameters the group refers to
_Group = namedtuple('_Group', 'name idx forward base_scale')


def _walk_hypernames(value, found):
    """Collect the hyperparameter names naming numbers in a joint prior specification."""
    if isinstance(value, str):
        found.append(value)
    elif isinstance(value, dict):
        for v in value.values():
            _walk_hypernames(v, found)
    elif isinstance(value, (list, tuple)):
        for v in value:
            _walk_hypernames(v, found)

    return found


def _group_hypernames(spec):
    """Return the hyperparameter names a joint prior samples, in specification order."""
    found = []
    for field in ('mu', 'sigma', 'rho', 'chol', 'outlier', 'covariates'):
        if field in spec:
            _walk_hypernames(spec[field], found)

    seen = set()
    return [n for n in found if not (n in seen or seen.add(n))]


def _resolve_vec(entries, hyperoffsets):
    """Build xs -> vector, reading any entry that names a hyperparameter out of xs."""
    consts = np.array([0.0 if isinstance(e, str) else float(e) for e in entries])
    idxs = [hyperoffsets[e] if isinstance(e, str) else -1 for e in entries]

    if all(i < 0 for i in idxs):
        fixed = matrix.jnparray(consts)
        return lambda xs: fixed

    def resolve(xs):
        # constants must carry the batch shape of xs to stack against the entries
        # read out of it, so that to_df can map a whole chain at once
        batch = jnp.shape(xs)[:-1]
        return jnp.stack([xs[..., i] if i >= 0 else jnp.broadcast_to(jnp.asarray(c), batch)
                          for i, c in zip(idxs, consts)], axis=-1)

    return resolve


def _resolve_mat(rows, hyperoffsets):
    """Build xs -> matrix from a nested list of numbers and hyperparameter names."""
    rowfuncs = [_resolve_vec(row, hyperoffsets) for row in rows]

    return lambda xs: jnp.stack([f(xs) for f in rowfuncs], axis=-2)


def _make_link(spec):
    """Return (forward, inverse) for one coordinate of a latent-space joint prior.

    spec: 'identity', or ('logistic', minval, maxval)
    """
    if spec == 'identity':
        return (lambda z: z), (lambda x: x)

    kind, lo, hi = spec
    if kind != 'logistic':
        raise KeyError(f'Unknown joint prior link {kind!r}; known links are '
                       f"['identity', 'logistic'].")
    lo, hi = float(lo), float(hi)

    return (lambda z: lo + (hi - lo) * jax.nn.sigmoid(z),
            lambda x: jax.scipy.special.logit((x - lo) / (hi - lo)))


def _group_instances(key, params, name):
    """Match a tuple of parameter regexes against params and bucket them by instance.

    An instance is the text captured by a named group (?P<inst>...), or by the
    single capturing group if there is one, or the empty string otherwise.
    """
    slots = []
    for rx in key:
        compiled = re.compile(rx)
        found = {}
        for par in params:
            m = compiled.fullmatch(par)
            if m is None:
                continue
            inst = m.groupdict().get('inst')
            if inst is None and compiled.groups == 1:
                inst = m.group(1)
            inst = inst or ''
            if inst in found:
                raise ValueError(f'Joint prior {name}: pattern {rx!r} matches both '
                                 f'{found[inst]!r} and {par!r} for instance {inst!r}.')
            found[inst] = par
        slots.append(found)

    instances = sorted(set().union(*[set(s) for s in slots])) if slots else []
    for inst in instances:
        missing = [key[j] for j, s in enumerate(slots) if inst not in s]
        if missing:
            raise ValueError(f'Joint prior {name}: instance {inst!r} has no parameter '
                             f'matching {missing}.')

    return [[slots[j][inst] for j in range(len(key))] for inst in instances], instances


def _build_group(key, spec, params, offsets, hyperoffsets, whiten):
    """Build a joint prior over the parameters matching a tuple of regexes."""
    name = spec.get('family', 'TruncatedMultivariateNormal')
    label = '/'.join(key)

    members, instances = _group_instances(key, params, label)
    if not members:
        return None

    idx = np.array([[offsets[par] for par in row] for row in members], dtype=int)

    if name == 'TruncatedMultivariateNormal':
        return _build_tmvn_group(label, key, spec, idx, whiten), instances
    if name == 'MultivariateNormal':
        return _build_mvn_group(label, key, spec, idx, instances, hyperoffsets), instances

    raise KeyError(f'Unknown joint prior family {name!r}; known families are '
                   f"['TruncatedMultivariateNormal', 'MultivariateNormal'].")


def _build_tmvn_group(label, key, spec, idx, whiten):
    """Joint prior on the physical parameters, truncated to a box.

    The population hyperparameters are fixed, so the box's Gaussian mass is a
    build-time constant and the per-coordinate link Jacobians are part of the
    density.
    """
    if _group_hypernames(spec):
        raise ValueError(f'Joint prior {label}: a TruncatedMultivariateNormal cannot sample its '
                         f'hyperparameters, because its normalisation is the Gaussian mass of '
                         f'the truncation box. Use a MultivariateNormal for a hierarchical fit.')

    d = len(key)
    mu = np.asarray(spec['mu'], dtype=float)
    sigma = np.asarray(spec['sigma'], dtype=float)
    rho = np.asarray(spec.get('rho', np.eye(d)), dtype=float)
    bounds = np.asarray(spec['bounds'], dtype=float)

    if mu.shape != (d,) or sigma.shape != (d,) or bounds.shape != (d, 2):
        raise ValueError(f'Joint prior {label}: mu, sigma and bounds must have '
                         f'{d} entries to match the {d} parameter patterns.')
    if np.any(sigma <= 0.0):
        raise ValueError(f'Joint prior {label}: sigma must be positive.')

    chol = _chol_rho(rho, label)
    siginv = np.linalg.inv(chol) / sigma[None, :]

    mass = _mvn_box_logmass(mu, sigma, rho, bounds)
    if mass <= 1e-12:
        raise ValueError(f'Joint prior {label}: the truncation box holds no Gaussian mass '
                         f'({mass:.3e}); widen the bounds or move the mean inside them.')
    logdens_const = (-np.sum(np.log(sigma)) - np.sum(np.log(np.diag(chol)))
                     - 0.5 * d * LOG2PI - np.log(mass))

    outlier = spec.get('outlier')
    linkbounds = bounds.copy()
    if outlier is not None:
        obounds = np.asarray(outlier['bounds'], dtype=float)
        if obounds.shape != (d, 2):
            raise ValueError(f'Joint prior {label}: outlier bounds must have {d} entries.')
        lfg, lout = _mix_logweights(outlier['chi'])
        outlier = (float(lfg), float(lout) - float(np.sum(np.log(obounds[:, 1] - obounds[:, 0]))),
                   matrix.jnparray(obounds[:, 0]), matrix.jnparray(obounds[:, 1]))
        linkbounds[:, 0] = np.minimum(linkbounds[:, 0], obounds[:, 0])
        linkbounds[:, 1] = np.maximum(linkbounds[:, 1], obounds[:, 1])

        if np.any(obounds[:, 0] < bounds[:, 0]) or np.any(obounds[:, 1] > bounds[:, 1]):
            # evaluate the step on the face of the core box nearest the mean, where
            # the core density along the boundary is largest
            zface = np.concatenate([(mu - bounds[:, 0]) / sigma, (bounds[:, 1] - mu) / sigma])
            axis = int(np.argmin(np.abs(zface))) % d
            edge = bounds[axis, 0] if np.argmin(np.abs(zface)) < d else bounds[axis, 1]
            xface = np.clip(mu, bounds[:, 0], bounds[:, 1])
            xface[axis] = edge
            w = siginv @ (xface - mu)
            _warn_mixture_step(f'Joint prior {label}',
                               np.array([lfg - 0.5 * float(w @ w) + logdens_const]), outlier[1])

    linkargs = tuple(matrix.jnparray(v) for v in
                     _tanh_link_params(linkbounds[:, 0], linkbounds[:, 1],
                                       np.clip(mu, linkbounds[:, 0], linkbounds[:, 1]),
                                       sigma, whiten))

    jmu, jsiginv = matrix.jnparray(mu), matrix.jnparray(siginv)
    jlo, jhi = matrix.jnparray(bounds[:, 0]), matrix.jnparray(bounds[:, 1])
    logdens_const = float(logdens_const)

    def forward(ys, xs):
        x, logjac = _tanh_forward(ys[..., idx], *linkargs)

        z = (x - jmu) @ jsiginv.T
        core = -0.5 * jnp.sum(z * z, axis=-1) + logdens_const
        core = jnp.where(jnp.all(_inbox(x, jlo, jhi), axis=-1), core, -jnp.inf)

        if outlier is not None:
            lfg, lout, olo, ohi = outlier
            out = jnp.where(jnp.all(_inbox(x, olo, ohi), axis=-1), lout, -jnp.inf)
            core = jnp.logaddexp(lfg + core, out)

        return x, jnp.sum(core) + jnp.sum(logjac)

    def inverse(x, xs):
        return _tanh_inverse(x, *linkargs)

    forward.inverse = inverse

    return _Group(label, idx, forward, 10.0)


def _build_mvn_group(label, key, spec, idx, instances, hyperoffsets):
    """Joint prior on a latent Gaussian vector, with the parameters as deterministic links.

    The prior is specified on the latent vector, so the links carry no Jacobian.
    Any entry of mu, chol, chi or a covariate coefficient may name a
    hyperparameter, which is then sampled alongside the population.
    """
    d = len(key)
    n_inst = len(instances)

    links = spec.get('link', ['identity'] * d)
    if len(links) != d:
        raise ValueError(f'Joint prior {label}: link must have {d} entries to match the '
                         f'{d} parameter patterns.')
    linkpairs = [_make_link(l) for l in links]

    mu_of = _resolve_vec(spec['mu'], hyperoffsets)
    chol_of = _resolve_mat(spec['chol'], hyperoffsets)
    for i, row in enumerate(spec['chol']):
        if len(row) != d or any(row[j] != 0 for j in range(i + 1, d)):
            raise ValueError(f'Joint prior {label}: chol must be a {d}x{d} lower-triangular '
                             f'array with zeros above the diagonal.')

    parametrisation = spec.get('parametrisation', 'noncentered')
    if parametrisation not in ('noncentered', 'centered'):
        raise ValueError(f"Joint prior {label}: parametrisation must be 'noncentered' or "
                         f"'centered', got {parametrisation!r}.")

    # per-instance additive shifts of the population mean, from a pulsar covariate
    shifts = []
    for cov in spec.get('covariates', []):
        coord = int(cov['coord'])
        values = np.array([float(cov['values'][inst]) for inst in instances])
        coeff_of = _resolve_vec([cov['coefficient']], hyperoffsets)
        onehot = matrix.jnparray(np.eye(d)[coord])
        jvalues = matrix.jnparray(values)
        shifts.append(lambda xs, c=coeff_of, v=jvalues, o=onehot:
                      (c(xs)[..., 0, None] * v)[..., None] * o)

    outlier = spec.get('outlier')
    if outlier is not None:
        ochol_of = _resolve_mat(outlier['chol'], hyperoffsets)
        chi_of = _resolve_vec([outlier['chi']], hyperoffsets)

    def _population(xs):
        # the instance axis is opened so the mean broadcasts against the (n_inst, d)
        # latent vectors, and so a covariate can make it per-instance
        mu = mu_of(xs)[..., None, :]
        for shift in shifts:
            mu = mu + shift(xs)

        return mu, chol_of(xs)

    def _logdet(chol):
        # trailing axis kept so this broadcasts against the per-instance terms
        return jnp.sum(jnp.log(jnp.abs(jnp.diagonal(chol, axis1=-2, axis2=-1))),
                       axis=-1)[..., None]

    def forward(ys, xs):
        zt = ys[..., idx]
        mu, chol = _population(xs)

        if parametrisation == 'noncentered':
            z = mu + zt @ jnp.swapaxes(chol, -1, -2)
            # (1 - chi) * N(z; mu, LL^T) * |L| is exactly (1 - chi) * N(zt; 0, I)
            logfg = -0.5 * jnp.sum(zt * zt, axis=-1) - 0.5 * d * LOG2PI
        else:
            z = zt
            w = (z - mu) @ jnp.swapaxes(jnp.linalg.inv(chol), -1, -2)
            logfg = -0.5 * jnp.sum(w * w, axis=-1) - _logdet(chol) - 0.5 * d * LOG2PI

        lp = logfg
        if outlier is not None:
            ochol = ochol_of(xs)
            wo = (z - mu) @ jnp.swapaxes(jnp.linalg.inv(ochol), -1, -2)
            logout = -0.5 * jnp.sum(wo * wo, axis=-1) - _logdet(ochol) - 0.5 * d * LOG2PI
            if parametrisation == 'noncentered':
                logout = logout + _logdet(chol)
            chi = chi_of(xs)[..., 0, None]
            lp = jnp.logaddexp(jnp.log1p(-chi) + logfg, jnp.log(chi) + logout)

        x = jnp.stack([linkpairs[j][0](z[..., j]) for j in range(d)], axis=-1)

        return x, jnp.sum(lp)

    def inverse(x, xs):
        z = jnp.stack([linkpairs[j][1](x[..., j]) for j in range(d)], axis=-1)
        mu, chol = _population(xs)

        if parametrisation == 'noncentered':
            return (z - mu) @ jnp.swapaxes(jnp.linalg.inv(chol), -1, -2)

        return z

    forward.inverse = inverse

    # the latent coordinates carry a unit-scale prior of their own
    return _Group(label, idx, forward, 1e3)


# --- transforms -------------------------------------------------------------

def _layout(params):
    """Return per-parameter slices into the flat vector, the flat length, and DF columns."""
    slices, columns, offset = [], [], 0
    for par in params:
        if '(' in par:
            root = par[:par.index('(')]
            l = int(par[par.index('(')+1:par.index(')')])
            slices.append(slice(offset, offset+l))
            columns.extend(f'{root}[{i}]' for i in range(l))
            offset = offset + l
        else:
            slices.append(offset)
            columns.append(par)
            offset = offset + 1

    return slices, offset, columns


def _attach(transformed, func, to_dict, to_vec, to_df, logprior, base_scale, params=None):
    def logL(ys):
        return func(to_dict(ys))

    transformed.params = func.params if params is None else params
    transformed.logprior = logprior
    transformed.logL = logL
    transformed.to_dict = to_dict
    transformed.to_vec = to_vec
    transformed.to_df = to_df
    transformed.base_scale = base_scale

    return transformed


def makelogtransform(func, priordict={}, jointpriors={}, prior_whiten=True):
    """Reparametrise a log-likelihood onto an unconstrained vector.

    Returns a callable ys -> logL + logprior carrying to_dict, to_vec, to_df,
    logL, logprior, params and base_scale.

    func:         log-likelihood with a .params list of parameter names
    priordict:    per-parameter prior specifications, overriding priordict_standard
    jointpriors:  joint priors keyed by a tuple of parameter regexes
    prior_whiten: scale each unconstrained coordinate so its prior is order unity
    """
    priordict = {**priordict_standard, **priordict}
    jointpriors = {**jointpriors_standard, **jointpriors}

    # hyperparameters named by a joint prior are sampled alongside the model's own
    # parameters, so they extend the unconstrained vector
    hypernames = []
    for key, spec in jointpriors.items():
        if _group_instances(tuple(key), func.params, '/'.join(key))[0]:
            hypernames.extend(n for n in _group_hypernames(spec) if n not in hypernames)
    clash = [n for n in hypernames if n in func.params]
    if clash:
        raise ValueError(f'Joint prior hyperparameters {clash} are already model parameters.')

    allparams = list(func.params) + hypernames
    slices, parlen, columns = _layout(allparams)
    hasvector = any('(' in par for par in func.params)

    offsets = {}
    for par, slice_ in zip(allparams, slices):
        if isinstance(slice_, slice):
            offsets[par] = slice_.start
        else:
            offsets[par] = slice_
    hyperoffsets = {n: offsets[n] for n in hypernames}

    groups, claimed = [], set()
    for key, spec in jointpriors.items():
        built = _build_group(tuple(key), spec, func.params, offsets, hyperoffsets, prior_whiten)
        if built is None:
            continue
        grp, instances = built
        for par in np.asarray(grp.idx).ravel():
            claimed.add(int(par))
        for key_ in key:
            for par in func.params:
                if re.fullmatch(key_, par) and '(' in par:
                    raise ValueError(f'Joint prior {grp.name}: member {par} is vector-valued.')
        groups.append(grp)

    byfamily = {}
    for par, slice_ in zip(allparams, slices):
        if isinstance(slice_, slice):
            flat = list(range(slice_.start, slice_.stop))
        else:
            flat = [slice_]
        # a parameter carried by a joint prior needs no scalar prior of its own
        flat = [i for i in flat if i not in claimed]
        if flat:
            name, args = parse_spec(_matchprior(par, priordict))
            byfamily.setdefault(name, ([], [], []))
            byfamily[name][0].extend(flat)
            byfamily[name][1].extend([args] * len(flat))
            byfamily[name][2].extend([par] * len(flat))

    if not groups and set(byfamily) <= {'Uniform'}:
        return _makelogtransform_uniform(func, priordict, slices, parlen, columns, hasvector)

    blocks = [_make_block(name, idx, specs, prior_whiten, labels)
              for name, (idx, specs, labels) in sorted(byfamily.items())]

    base_scale = np.full(parlen, 10.0)
    for blk in blocks:
        if blk.link == 'affine':
            base_scale[blk.idx] = 1e3
    for grp in groups:
        base_scale[np.asarray(grp.idx).ravel()] = grp.base_scale
    base_scale = matrix.jnparray(base_scale)

    def _forward(ys):
        xs = jnp.zeros(jnp.shape(ys)[:-1] + (parlen,))
        lp = 0.0

        # scalar blocks first: a joint prior may read a hyperparameter's physical value
        for blk in blocks:
            x, logjac = _block_forward(blk, ys[..., blk.idx])
            xs = xs.at[..., blk.idx].set(x)
            lp = lp + jnp.sum(blk.family.logpdf(x, blk.args, blk.extra) + logjac)

        for grp in groups:
            x, lpg = grp.forward(ys, xs)
            xs = xs.at[..., grp.idx].set(x)
            lp = lp + lpg

        return xs, lp

    def _split(xs):
        if hasvector:
            return {par: xs[..., slice_] for par, slice_ in zip(allparams, slices)}
        else:
            return dict(zip(allparams, jnp.moveaxis(xs, -1, 0)))

    def to_dict(ys):
        return _split(_forward(ys)[0])

    def to_vec(params):
        # parameters the caller omits, hyperparameters in particular, keep the
        # physical values that ys = 0 maps to
        xs = _forward(jnp.zeros(parlen))[0]
        for par, slice_ in zip(allparams, slices):
            if par in params:
                xs = xs.at[slice_].set(params[par])

        ys = jnp.zeros(parlen)
        for blk in blocks:
            ys = ys.at[blk.idx].set(_block_inverse(blk, xs[blk.idx]))
        for grp in groups:
            ys = ys.at[grp.idx].set(grp.forward.inverse(xs[grp.idx], xs))

        return ys

    def to_df(ys, psrs=None):
        return _make_df(_forward(ys)[0], columns, psrs)

    def logprior(ys):
        return _forward(ys)[1]

    def transformed(ys):
        xs, lp = _forward(ys)
        return func(_split(xs)) + lp

    return _attach(transformed, func, to_dict, to_vec, to_df, logprior, base_scale, allparams)


def _make_df(xs, columns, psrs=None):
    if psrs is None:
        return pd.DataFrame(np.array(xs), columns=columns)
    else:
        # rename columns from psr number to psr name
        psrdict = {f'{i}]': psr.name for i, psr in enumerate(psrs)}
        psrcols = [psrdict[par.split('[')[1]] + '_' + par.split('[')[0] if '[' in par else par for par in columns]
        return pd.DataFrame(np.array(xs), columns=psrcols).sort_index(axis=1)


def _makelogtransform_uniform(func, priordict, slices, parlen, columns, hasvector):
    """Uniform-only transform, kept as a single vectorised tanh over the whole vector."""
    a, b = [], []
    for par, slice_ in zip(func.params, slices):
        therange = _uniform_range(par, priordict)

        if isinstance(slice_, slice):
            for i in range(slice_.stop - slice_.start):
                a.append(therange[0])
                b.append(therange[1])
        else:
            a.append(therange[0])
            b.append(therange[1])

    a, b = matrix.jnparray(a), matrix.jnparray(b)

    def to_dict(ys):
        xs = 0.5 * (b + a + (b - a) * jnp.tanh(ys))

        if hasvector:
            return {par: xs[slice_] for par, slice_ in zip(func.params, slices)}
        else:
            return dict(zip(func.params, xs))

    def to_vec(params):
        xs = jnp.zeros_like(a)
        for par, slice_ in zip(func.params, slices):
            xs = xs.at[slice_].set(params[par])

        u = (a + b - 2*xs)/(a - b)
        # arctanh(+/-1) is infinite; pull |u| == 1 back to the largest representable
        # magnitude below one, leaving |u| > 1 to produce NaN as an out-of-prior signal.
        u = jnp.where(jnp.abs(u) == 1.0, jnp.sign(u) * (1.0 - 0.5 * jnp.finfo(u.dtype).eps), u)

        return jnp.arctanh(u)

    def to_df(ys, psrs=None):
        xs = 0.5 * (b + a + (b - a) * jnp.tanh(ys))
        return _make_df(xs, columns, psrs)

    def logprior(ys):
        return jnp.sum(jnp.log(2.0) - 2.0 * jnp.logaddexp(ys, -ys))

    def transformed(ys):
        return func(to_dict(ys)) + logprior(ys)

    return _attach(transformed, func, to_dict, to_vec, to_df, logprior,
                   matrix.jnparray(np.full(parlen, 10.0)))


makelogtransform_uniform = makelogtransform


def makelogtransform_classic(func, priordict={}):
    """Uniform-only transform for models whose parameters are all scalar."""
    priordict = {**priordict_standard, **priordict}

    a, b = [], []
    for par in func.params:
        therange = _uniform_range(par, priordict)
        a.append(therange[0])
        b.append(therange[1])

    a, b = matrix.jnparray(a), matrix.jnparray(b)

    def to_dict(ys):
        xs = 0.5 * (b + a + (b - a) * jnp.tanh(ys))
        return dict(zip(func.params, xs))

    def to_vec(params):
        xs = matrix.jnparray([params[pname] for pname in func.params])

        u = (a + b - 2*xs)/(a - b)
        # arctanh(+/-1) is infinite; pull |u| == 1 back to the largest representable
        # magnitude below one, leaving |u| > 1 to produce NaN as an out-of-prior signal.
        u = jnp.where(jnp.abs(u) == 1.0, jnp.sign(u) * (1.0 - 0.5 * jnp.finfo(u.dtype).eps), u)

        return jnp.arctanh(u)

    def to_df(ys):
        xs = 0.5 * (b + a + (b - a) * jnp.tanh(ys))
        return pd.DataFrame(np.array(xs), columns=func.params)

    def logprior(ys):
        return jnp.sum(jnp.log(2.0) - 2.0 * jnp.logaddexp(ys, -ys))

        # return jnp.sum(jnp.log(0.5) - 2.0 * jnp.log(jnp.cosh(ys)))
        # but   log(0.5) - 2 * log(cosh(y))
        #     = log(0.5) - 2 * log((exp(x) + exp(-x))/2)
        #     = log(0.5) - 2 * (log(exp(x) - exp(-x)) - log(2.0))
        #     = log(2.0) - 2 * logaddexp(x, -x)

    def transformed(ys):
        return func(to_dict(ys)) + logprior(ys)

    return _attach(transformed, func, to_dict, to_vec, to_df, logprior,
                   matrix.jnparray(np.full(len(func.params), 10.0)))


# --- drawing from the prior -------------------------------------------------

def _rvs(name, args, size):
    if name == 'Uniform':
        a, b = args
        return np.random.uniform(a, b, size=size)
    elif name == 'Normal':
        mu, sd = args
        return np.random.normal(mu, sd, size=size)
    elif name == 'TruncatedNormal':
        mu, sd, a, b = args
        return scipy.stats.truncnorm.rvs((a - mu)/sd, (b - mu)/sd, loc=mu, scale=sd, size=size)
    elif name in ('UniformWithOutliers', 'NormalWithOutliers',
                  'TruncatedNormalWithOutliers', 'NormalWithNormalOutliers'):
        if name == 'UniformWithOutliers':
            core, chi, out = ('Uniform', args[:2]), args[2], ('Uniform', args[3:5])
        elif name == 'NormalWithOutliers':
            core, chi, out = ('Normal', args[:2]), args[2], ('Uniform', args[3:5])
        elif name == 'TruncatedNormalWithOutliers':
            core, chi, out = ('TruncatedNormal', args[:4]), args[4], ('Uniform', args[5:7])
        else:
            core, chi, out = ('Normal', args[:2]), args[2], ('Normal', (args[0], args[3]))

        pick = np.random.uniform(size=size) < chi
        return np.where(pick, _rvs(out[0], out[1], size), _rvs(core[0], core[1], size))

    raise KeyError(f'No sampler for prior family {name!r}.')


def sample_prior(params, priordict={}, n=1, fail=True):
    """Draw parameter values from their priors.

    params:    parameter names to draw
    priordict: prior specifications, overriding priordict_standard
    n:         number of draws; n == 1 returns scalars rather than length-1 arrays
    fail:      raise if a parameter has no matching prior, rather than skipping it
    """
    priordict = {**priordict_standard, **priordict}

    sample = {}
    for par in params:
        for parname, spec in priordict.items():
            if parname == par or re.match(parname, par):
                name, args = parse_spec(spec)
                if par.endswith(")"):
                    size = int(par[par.index("(") + 1 : -1])
                    sample[par] = _rvs(name, args, size if n == 1 else (n, size))
                else:
                    sample[par] = (float(_rvs(name, args, 1)[0]) if n == 1
                                   else _rvs(name, args, n))
                break
        else:
            if fail:
                raise KeyError(f"No known prior for {par}.")

    return sample


def sample_uniform(params, priordict={}, n=1, fail=True):
    """Draw parameter values from uniform priors. Raises if a prior is not Uniform."""
    priordict = {**priordict_standard, **priordict}

    sample = {}
    for par in params:
        for parname, spec in priordict.items():
            if parname == par or re.match(parname, par):
                range = _uniform_range(par, priordict)
                if par.endswith(")"):
                    size = int(par[par.index("(") + 1 : -1])
                    sample[par] = (
                        np.random.uniform(*range, size=size)
                        if n == 1
                        else np.random.uniform(*range, size=(n, size))
                    )
                else:
                    sample[par] = np.random.uniform(*range) if n == 1 else np.random.uniform(*range, size=n)
                break
        else:
            if fail:
                raise KeyError(f"No known prior for {par}.")

    return sample
