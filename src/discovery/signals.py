import os
import re
import inspect
import typing
import warnings
from collections.abc import Iterable

import numpy as np
import scipy.interpolate as si
import jax
import jax.numpy as jnp

from . import matrix
from . import const

# residuals
def residuals(psr):
    return psr.residuals


# EFAC/EQUAD/ECORR noise

# no backends
def makenoise_measurement_simple(psr, noisedict={}, add_equad=True, tnequad=False):
    """Single-EFAC (optionally single-EQUAD) white-noise model for a pulsar.

    Builds a diagonal measurement-noise matrix using one ``efac`` for the whole
    pulsar. When ``add_equad`` is True an EQUAD term is included: ``tnequad=True``
    uses the TempoNest convention (EQUAD added outside the EFAC scaling), while the
    default (``tnequad=False``) uses the tempo2/``t2equad`` convention (EQUAD added
    in quadrature with the TOA errors, inside the EFAC scaling). Set
    ``add_equad=False`` for an EFAC-only model. If all required parameters are present
    in ``noisedict`` a constant matrix is returned, otherwise a variable one.
    """
    efac = f'{psr.name}_efac'
    if tnequad and add_equad:
        log10_tnequad = f'{psr.name}_log10_tnequad'
        params = [efac, log10_tnequad]
    elif add_equad:
        log10_t2equad = f'{psr.name}_log10_t2equad'
        params = [efac, log10_t2equad]
    else:
        params = [efac]

    if all(par in noisedict for par in params):
        if tnequad and add_equad:
            noise = noisedict[efac]**2 * psr.toaerrs**2 + (10.0**(2.0 * noisedict[log10_tnequad]))
        elif add_equad:
            noise = noisedict[efac]**2 * (psr.toaerrs**2 + 10.0**(2.0 * noisedict[log10_t2equad]))
        else:
            noise = noisedict[efac]**2 * psr.toaerrs**2
        return matrix.NoiseMatrix1D_novar(noise)
    else:
        toaerrs = matrix.jnparray(psr.toaerrs)
        def getnoise(params, tnequad=tnequad):
            if tnequad and add_equad:
                return params[efac]**2 * toaerrs**2 + 10.0**(2.0 * params[log10_tnequad])
            elif add_equad:
                return params[efac]**2 * (toaerrs**2 + 10.0**(2.0 * params[log10_t2equad]))
            else:
                return params[efac]**2 * toaerrs**2
        getnoise.params = params

        return matrix.NoiseMatrix1D_var(getnoise)


# nanograv backends
def selection_backend_flags(psr):
    return psr.backend_flags


def selection_flags(flags, sep='_', warn_below=10):
    """Return a selection splitting the TOAs on one or more per-TOA flags.

    Any function taking a selection -- :func:`makenoise_measurement`,
    :func:`makegp_ecorr` and its Legendre variants, :func:`makegp_fd_piecewise` --
    splits on the backend by default. This generalises that to any entry of
    ``psr.flags``, so a per-channel efac and equad is ``selection_flags('chan')``.

    Labels carry the flag name, giving parameters like
    ``J0437-4715_chan15_efac`` rather than ``J0437-4715_15_efac``, so a chain stays
    readable and two flags cannot collide.

    Raises if a named flag is absent, and if any TOA carries an empty value for one:
    :func:`makenoise_measurement` drops the empty label, which would leave those TOAs
    with zero measurement noise rather than an error.

    flags:      flag name, or a sequence of names whose values are combined into one
                label
    sep:        separator between the parts of a combined label
    warn_below: warn about groups holding fewer than this many TOAs, which cannot
                constrain a white-noise parameter of their own
    """
    names = [flags] if isinstance(flags, str) else list(flags)

    def selection(psr):
        missing = [n for n in names if n not in psr.flags]
        if missing:
            raise KeyError(f'selection_flags: {psr.name} has no flag(s) {missing}; '
                           f'available flags are {sorted(psr.flags)}.')

        cols = []
        for n in names:
            v = np.asarray(psr.flags[n]).astype(str)
            nempty = int((v == '').sum())
            if nempty:
                raise ValueError(
                    f'selection_flags: {psr.name} has {nempty} TOAs with an empty {n!r} '
                    f'flag. Those TOAs would be dropped from the selection and left with '
                    f'zero measurement noise, so fix the flag rather than proceeding.')
            cols.append(np.char.add(n, v))

        labels = cols[0] if len(cols) == 1 else np.array(
            [sep.join(parts) for parts in zip(*cols)])

        counts = {lab: int((labels == lab).sum()) for lab in set(labels.tolist())}
        thin = {k: v for k, v in counts.items() if v < warn_below}
        if thin:
            print(f'Warning: selection_flags on {names} for {psr.name} gives '
                  f'{len(counts)} groups, of which {len(thin)} hold fewer than '
                  f'{warn_below} TOAs: {dict(sorted(thin.items())[:6])}'
                  f'{" ..." if len(thin) > 6 else ""}')

        return labels

    selection.__name__ = 'selection_' + '_'.join(names)
    selection.flags = names

    return selection


def selection_flag_values(flag, values, rest='rest', warn_below=10):
    """Selection giving NAMED values of a flag their own group, pooling the rest.

    :func:`selection_flags` gives every distinct value its own group, which on an MPTA
    pulsar is 32 EFAC/EQUAD pairs and takes the sampled dimension from 19 to 81. When the
    question is about a few known channels, ``k`` named values cost ``2(k+1)`` white
    parameters instead of ``2 n_values``.

    Labels follow the same convention, so a TOA whose ``chan`` is ``'26'`` with
    ``values=['7', '26']`` is labelled ``'chan26'`` and everything else ``'chanrest'``,
    giving ``{psr}_chan26_efac`` and ``{psr}_chanrest_efac``.

    Values compare as strings, matching :func:`selection_flags`, which casts through
    ``astype(str)``; ints are accepted and cast, so ``values=[7]`` and ``values=['7']``
    behave identically.

    Raises rather than degrading on every failure mode below: each of them otherwise
    yields a model that runs and quietly differs from the one asked for.

    flag:       per-TOA flag name
    values:     the values to give their own group
    rest:       label suffix for the pooled remainder
    warn_below: warn about named groups holding fewer TOAs than this
    """
    names = [str(v) for v in ([values] if isinstance(values, (str, int)) else values)]
    if not names:
        raise ValueError('selection_flag_values: no values given; use selection_flags to '
                         'split on every value of a flag.')
    if len(set(names)) != len(names):
        raise ValueError(f'selection_flag_values: duplicate values in {names}.')

    def selection(psr):
        if flag not in psr.flags:
            raise KeyError(f'selection_flag_values: {psr.name} has no flag {flag!r}; '
                           f'available flags are {sorted(psr.flags)}.')
        v = np.asarray(psr.flags[flag]).astype(str)
        nempty = int((v == '').sum())
        if nempty:
            raise ValueError(
                f'selection_flag_values: {psr.name} has {nempty} TOAs with an empty '
                f'{flag!r} flag. makenoise_measurement drops the empty label, which would '
                f'leave those TOAs with zero measurement noise, so fix the flag.')

        present = set(v.tolist())
        missing = [n for n in names if n not in present]
        if missing:
            raise ValueError(
                f'selection_flag_values: {psr.name} has no TOA with {flag}={missing}. '
                f'Present values are {sorted(present)}. A value that selects nothing '
                f'would silently give one fewer free group than asked for.')
        if rest in present:
            raise ValueError(
                f'selection_flag_values: {flag}={rest!r} is a real value for {psr.name}, '
                f'so it would collide with the pooled-remainder label. Pass a different '
                f'rest=.')
        if not (present - set(names)):
            raise ValueError(
                f'selection_flag_values: the named values cover every value of {flag!r} '
                f'for {psr.name}, so the pooled group would be empty and its parameters '
                f'would sample their prior. Use selection_flags instead.')

        keep = np.isin(v, names)
        labels = np.where(keep, np.char.add(flag, v), flag + rest)

        counts = {n: int((v == n).sum()) for n in names}
        thin = {k: c for k, c in counts.items() if c < warn_below}
        if thin:
            print(f'Warning: selection_flag_values on {flag!r} for {psr.name}: named '
                  f'group(s) {thin} hold fewer than {warn_below} TOAs and cannot '
                  f'constrain a white-noise parameter of their own.')
        print(f'selection_flag_values: {psr.name} {flag}: '
              f'{len(names)} named group(s) {counts}, pooled remainder '
              f'{int((~keep).sum())} TOAs -> {2 * (len(names) + 1)} white parameters.')
        return labels

    selection.__name__ = f'selection_{flag}_' + '_'.join(names)
    selection.flags = [flag]
    selection.values = names
    selection.rest = rest

    return selection


def makenoise_measurement(psr, noisedict={}, scale=1.0, tnequad=False, ecorr=False, selection=selection_backend_flags, vectorize=True,
                          outliers=False, enterprise=False):
    backend_flags = selection(psr)
    backends = [b for b in sorted(set(backend_flags)) if b != '']

    efacs = [f'{psr.name}_{backend}_efac' for backend in backends]
    if tnequad:
        log10_tnequads = [f'{psr.name}_{backend}_log10_tnequad' for backend in backends]
        params = efacs + log10_tnequads
    else:
        log10_t2equads = [f'{psr.name}_{backend}_log10_t2equad' for backend in backends]
        params = efacs + log10_t2equads

    masks = [(backend_flags == backend) for backend in backends]
    logscale = np.log10(scale)

    # scale each toa individually. register scales as a parameter
    if outliers:
        toaerr_scaling = f'{psr.name}_alpha_scaling({psr.toas.size})'
        params.append(toaerr_scaling)

    if all(par in noisedict for par in params):
        if outliers:
            raise ValueError("No outlier scaling if white noise is fixed.")
        if tnequad:
            noise = sum(mask * (noisedict[efac]**2 * (scale * psr.toaerrs)**2 + 10.0**(2 * (logscale + noisedict[log10_tnequad])))
                        for mask, efac, log10_tnequad in zip(masks, efacs, log10_tnequads))
        else:
            noise = sum(mask * noisedict[efac]**2 * ((scale * psr.toaerrs)**2 + 10.0**(2 * (logscale + noisedict[log10_t2equad])))
                        for mask, efac, log10_t2equad in zip(masks, efacs, log10_t2equads))

        if ecorr:
            egp = makegp_ecorr(psr, noisedict=noisedict, enterprise=enterprise, scale=scale, selection=selection)
            return matrix.NoiseMatrixSM_novar(noise, egp.F, egp.Phi.N)
        else:
            return matrix.NoiseMatrix1D_novar(noise)
    else:
        if vectorize:
            toaerrs2, masks = matrix.jnparray(scale**2 * psr.toaerrs**2), matrix.jnparray([mask for mask in masks])

            if tnequad:
                def getnoise(params):
                    if outliers:
                        alpha_scaling = params[toaerr_scaling]
                    else:
                        alpha_scaling = 1.0
                    efac2  = matrix.jnparray([params[efac]**2 for efac in efacs])
                    equad2 = matrix.jnparray([10.0**(2 * (logscale + params[log10_tnequad])) for log10_tnequad in log10_tnequads])

                    return (masks * (efac2[:,jnp.newaxis] * (alpha_scaling*toaerrs2)[jnp.newaxis,:] + equad2[:,jnp.newaxis])).sum(axis=0)
            else:

                def getnoise(params):
                    if outliers:
                        alpha_scaling = params[toaerr_scaling]
                    else:
                        alpha_scaling = 1.0
                    efac2  = matrix.jnparray([params[efac]**2 for efac in efacs])
                    equad2 = matrix.jnparray([10.0**(2 * (logscale + params[log10_t2equad])) for log10_t2equad in log10_t2equads])

                    return (masks * efac2[:,jnp.newaxis] * ((alpha_scaling*toaerrs2)[jnp.newaxis,:] + equad2[:,jnp.newaxis])).sum(axis=0)
        else:
            toaerrs, masks = matrix.jnparray(scale * psr.toaerrs), [matrix.jnparray(mask) for mask in masks]
            if tnequad:
                def getnoise(params):
                    if outliers:
                        alpha_scaling = params[toaerr_scaling]
                    else:
                        alpha_scaling = 1.0

                    return sum(mask * (params[efac]**2 * (alpha_scaling * toaerrs**2) + 10.0**(2 * (logscale + params[log10_tnequad])))
                               for mask, efac, log10_tnequad in zip(masks, efacs, log10_tnequads))
            else:
                def getnoise(params):
                    if outliers:
                        alpha_scaling = params[toaerr_scaling]
                    else:
                        alpha_scaling = 1.0
                    return sum(mask * params[efac]**2 * (alpha_scaling * toaerrs**2 + 10.0**(2 * (logscale + params[log10_t2equad])))
                               for mask, efac, log10_t2equad in zip(masks, efacs, log10_t2equads))

        getnoise.params = params

        if ecorr:
            egp = makegp_ecorr(psr, noisedict={}, enterprise=enterprise, scale=scale, selection=selection)
            return matrix.NoiseMatrixSM_var(getnoise, egp.F, egp.Phi.getN)
        else:
            return matrix.NoiseMatrix1D_var(getnoise)


# ECORR

# quantization
# note the resulting ecorr degrees of freedom are slightly different than in enterprise
# (and of course I forgot about it)

# bins = (psr.toas + 0.5).astype(np.int64)
# uniques, counts = np.unique(bins, return_counts=True)
# Umat = jnp.array(np.vstack([bins == unique for unique, count in zip(uniques, counts) if count > 1]).astype(jnp.float64).T)

def quantize(toas, dt=1.0):
    isort = np.argsort(toas)
    bins = np.zeros_like(toas, np.int64)

    b, v = 0, toas.min()
    for j in isort:
        if toas[j] - v > dt:
            v = toas[j]
            b = b + 1

        bins[j] = b

    return bins

# no backends
def ecorr_tobs_weights(psr, tobs_ref=1800.0):
    """Per-TOA jitter weights ``sqrt(tobs_ref / tobs)``, for scaling an ECORR basis.

    Jitter averages down over the pulses in an integration, so its amplitude goes as
    1/sqrt(tobs) and a single ECORR over TOAs of unequal length is an average over
    that. Weighting the basis rows makes ``log10_ecorr`` the jitter at ``tobs_ref``.

    Raises if the flag is missing or non-positive, and warns if any integration is
    shorter than a tenth of the median, which would carry more than three times the
    typical weight.

    psr:      Discovery Pulsar object, carrying a per-TOA 'tobs' flag in seconds
    tobs_ref: reference integration time in seconds
    """
    if 'tobs' not in psr.flags:
        raise KeyError(
            f"ecorr_tobs_weights: {psr.name} has no 'tobs' flag, so the ECORR basis "
            f"cannot be scaled by integration time; available flags are "
            f"{sorted(psr.flags)}.")

    tobs = np.asarray(psr.flags['tobs'])
    try:
        tobs = tobs.astype(np.float64)
    except ValueError:
        raise ValueError(f"ecorr_tobs_weights: {psr.name} has non-numeric 'tobs' values.")

    bad = ~np.isfinite(tobs) | (tobs <= 0.0)
    if bad.any():
        raise ValueError(
            f"ecorr_tobs_weights: {psr.name} has {int(bad.sum())} TOA(s) with a "
            f"non-positive or non-finite 'tobs'. Scaling by integration time would "
            f"give them an infinite weight, so fix the flag.")

    if tobs_ref <= 0.0:
        raise ValueError(f'ecorr_tobs_weights: tobs_ref must be positive, got {tobs_ref}.')

    median = float(np.median(tobs))
    short = tobs < 0.1 * median
    if short.any():
        print(f"Warning: ecorr_tobs_weights: {psr.name} has {int(short.sum())} TOA(s) "
              f"with tobs below a tenth of the median {median:.0f} s (shortest "
              f"{tobs.min():.0f} s), which take {np.sqrt(median / tobs.min()):.1f}x the "
              f"typical weight.")

    return np.sqrt(tobs_ref / tobs)


def makegp_ecorr_simple(psr, noisedict={}):
    log10_ecorr = f'{psr.name}_log10_ecorr'
    params = [log10_ecorr]

    bins = quantize(psr.toas)
    Umat = np.vstack([bins == i for i in range(bins.max() + 1)]).T
    ones = np.ones(Umat.shape[1], dtype=np.float64)

    if all(par in noisedict for par in params):
        phi = (10.0**(2.0 * noisedict[log10_ecorr])) * ones

        return matrix.ConstantGP(matrix.NoiseMatrix1D_novar(phi), Umat)
    else:
        ones = matrix.jnparray(ones)
        def getphi(params):
            return (10.0**(2.0 * params[log10_ecorr])) * ones
        getphi.params = params

        return matrix.VariableGP(matrix.NoiseMatrix1D_var(getphi), Umat)

# nanograv backends
def makegp_ecorr(psr, noisedict={}, enterprise=False, scale=1.0, selection=selection_backend_flags, variable=False, name='ecorrGP',
                 tobs_scale=False, tobs_ref=1800.0):
    log10_ecorrs, Umats = [], []

    backend_flags = selection(psr)
    backends = [b for b in sorted(set(backend_flags)) if b != '']
    masks = [np.array(backend_flags == backend) for backend in backends]
    for backend, mask in zip(backends, masks):
        log10_ecorrs.append(f'{psr.name}_{backend}_log10_ecorr')


        # For handling the single backend case
        if len(np.unique(masks)) == 1:
            # for those pulsar with only one backend
            first_valid_bin = 0
        else:
            # if the mask contains zeros
            # the zeros in quantize below end up in the
            # first entry, which we skip later.
            first_valid_bin = 1

        bins = quantize(psr.toas * mask)

        if enterprise:
            # legacy accounting of degrees of freedom
            uniques, counts = np.unique(bins, return_counts=True)
            epoch_masks = [bins == i for i, cnt in zip(
                uniques[first_valid_bin:],
                counts[first_valid_bin:]) if cnt > 1]

            if epoch_masks:
                U_backend = np.vstack(epoch_masks).T
            else:
                # if there is no ToAs observed at the same time
                U_backend = np.zeros((len(bins), 0))

            Umats.append(U_backend)
        else:
            Umats.append(np.vstack([bins == i for i in range(first_valid_bin, bins.max() + 1)]).T)
    Umatall = np.hstack(Umats)
    if tobs_scale:
        Umatall = Umatall * ecorr_tobs_weights(psr, tobs_ref)[:, None]
    params = log10_ecorrs

    pmasks, cnt = [], 0
    for Umat in Umats:
        z = np.zeros(Umatall.shape[1], dtype=np.float64)
        z[cnt:cnt+Umat.shape[1]] = 1.0
        pmasks.append(z)
        cnt = cnt + Umat.shape[1]
    logscale = np.log10(scale)

    if all(par in noisedict for par in params):
        phi = sum(10.0**(2 * (logscale + noisedict[log10_ecorr])) * pmask for (log10_ecorr, pmask) in zip(log10_ecorrs, pmasks))

        if variable:
            def getphi(params):
                return phi
            getphi.params = []

            gp = matrix.VariableGP(matrix.NoiseMatrix1D_var(getphi), Umatall)
            gp.index = {f'{psr.name}_{name}_coefficients({Umatall.shape[1]})': slice(0,Umatall.shape[1])} # better for cosine
            gp.name, gp.pos = psr.name, psr.pos
            gp.gpname, gp.gpcommon = name, []

            return gp
        else:
            return matrix.ConstantGP(matrix.NoiseMatrix1D_novar(phi), Umatall)
    else:
        pmasks = [matrix.jnparray(pmask) for pmask in pmasks]
        def getphi(params):
            return sum(10.0**(2 * (logscale + params[log10_ecorr])) * pmask for (log10_ecorr, pmask) in zip(log10_ecorrs, pmasks))
        getphi.params = params

        gp = matrix.VariableGP(matrix.NoiseMatrix1D_var(getphi), Umatall)
        gp.index = {f'{psr.name}_{name}_coefficients({Umatall.shape[1]})': slice(0,Umatall.shape[1])} # better for cosine
        gp.name, gp.pos = psr.name, psr.pos
        gp.gpname, gp.gpcommon = name, []

        return gp
    
def makegp_ecorr_legendre(psr, noisedict={}, enterprise=False, scale=1.0,
                          selection=selection_backend_flags, variable=False,
                          nmodes=3, fref=None, name='ecorrGPleg',
                          tobs_scale=False, tobs_ref=1800.0):
    """ECORR GP with a Legendre-polynomial frequency basis (``nmodes`` modes).

    Within each backend, TOA frequencies are mapped to a normalised log-frequency
    coordinate ``y in [-1, 1]`` (``y = (log f - mid) / half``, with ``mid`` and
    ``half`` set by the backend's band edges), and the first ``nmodes`` Legendre
    polynomials P_0(y), ..., P_{nmodes-1}(y) form the within-epoch frequency basis.
    Because these polynomials are orthogonal on [-1, 1], the ``nmodes`` per-backend
    amplitudes (``..._log10_ecorr`` for mode 0, then ``..._log10_ecorr_k1`` ...
    ``_k{nmodes-1}``) act as nearly-independent jitter modes and the posteriors
    stay well conditioned:

        nmodes=1  ->  standard ECORR (band-mean amplitude only; the mode-0
                      parameter name ``..._log10_ecorr`` is identical to that of
                      :func:`makegp_ecorr`, so noisedicts are interchangeable)
        nmodes=2  ->  + linear chromatic decorrelation across the band
        nmodes=3  ->  + quadratic curvature
        nmodes=N  ->  Legendre modes up to degree N-1

    Varying ``nmodes`` lets the number of significant jitter modes (cf. the SVD
    analysis of Kulkarni et al.) be part of per-pulsar model selection.

    The mode amplitudes are independent (diagonal prior), so the reconstructed
    jitter variance is symmetric about band centre. For a frequency-asymmetric
    amplitude (e.g. the low-frequency excess in J0437), use the correlated-mode
    variant :func:`makegp_ecorr_legendre_correlated`.

    If ``fref`` is None (default) it is set per-backend to the mean ``psr.freqs``;
    it only centres ``x`` and cancels out of ``y``, so it merely keeps the basis
    well conditioned. The log-frequency coordinate matches the observed
    stationarity of jitter decorrelation in log(fa/fb) (Kulkarni et al.).
    """

    if not isinstance(nmodes, int) or nmodes < 1:
        raise ValueError('makegp_ecorr_legendre: nmodes must be an integer >= 1')

    freqs = np.asarray(psr.freqs)

    backend_flags = selection(psr)
    backends = [b for b in sorted(set(backend_flags)) if b != '']
    masks = [np.array(backend_flags == backend) for backend in backends]

    log10_ecorrs_per_mode = [[] for _ in range(nmodes)]
    U_blocks_per_mode = [[] for _ in range(nmodes)]

    for backend, mask in zip(backends, masks):
        for m in range(nmodes):
            suffix = '' if m == 0 else f'_k{m}'
            log10_ecorrs_per_mode[m].append(f'{psr.name}_{backend}_log10_ecorr{suffix}')

        # per-backend reference frequency (only centres x; cancels out of y)
        fref_b = float(np.mean(freqs[mask])) if fref is None else float(fref)
        x = np.log(freqs / fref_b)

        # rescale to y in [-1, 1] across this backend's band so the Legendre
        # polynomials are evaluated on their natural support
        x_backend = x[mask]
        x_min, x_max = float(x_backend.min()), float(x_backend.max())
        half_range = 0.5 * (x_max - x_min)
        if half_range == 0.0:
            # only one observing frequency in this backend: fall back to the
            # centred coordinate, where only P_0 (the constant mode) is meaningful
            y = np.zeros_like(freqs)
        else:
            y = (x - 0.5 * (x_max + x_min)) / half_range

        # columns of legvander are P_0(y) ... P_{nmodes-1}(y)
        legendre_modes = np.polynomial.legendre.legvander(y, nmodes - 1)

        # For handling the single backend case
        if len(np.unique(masks)) == 1:
            # for those pulsar with only one backend
            first_valid_bin = 0
        else:
            # if the mask contains zeros
            # the zeros in quantize below end up in the
            # first entry, which we skip later.
            first_valid_bin = 1

        # quantize TOAs that belong to this backend
        bins = quantize(psr.toas * mask)

        if enterprise:
            uniques, counts = np.unique(bins, return_counts=True)
            epoch_masks = [bins == i for i, cnt in zip(
                uniques[first_valid_bin:],
                counts[first_valid_bin:]) if cnt > 1]

            if epoch_masks:
                U0 = np.vstack(epoch_masks).T
            else:
                # if there is no ToAs observed at the same time
                U0 = np.zeros((len(bins), 0))
        else:
            U0 = np.vstack([bins == i for i in range(first_valid_bin, bins.max() + 1)]).T

        U0 = U0.astype(np.float64)
        # multiply the per-epoch indicator columns by each Legendre mode
        for m in range(nmodes):
            U_blocks_per_mode[m].append(U0 * legendre_modes[:, m][:, None])

    # full basis, grouped by mode: [ mode0 columns | mode1 columns | ... ]
    U_per_mode = [np.hstack(blocks) for blocks in U_blocks_per_mode]
    Umatall = np.hstack(U_per_mode)
    if tobs_scale:
        Umatall = Umatall * ecorr_tobs_weights(psr, tobs_ref)[:, None]

    # build per-parameter masks selecting the corresponding columns of Umatall
    pmasks, params = [], []
    cnt = 0
    for m in range(nmodes):
        for U_block, log10_ecorr in zip(U_blocks_per_mode[m], log10_ecorrs_per_mode[m]):
            n = U_block.shape[1]
            z = np.zeros(Umatall.shape[1], dtype=np.float64)
            z[cnt:cnt + n] = 1.0
            pmasks.append(z)
            params.append(log10_ecorr)
            cnt += n

    logscale = np.log10(scale)

    if all(par in noisedict for par in params):
        phi = sum(10.0**(2 * (logscale + noisedict[log10_ecorr])) * pmask
                  for (log10_ecorr, pmask) in zip(params, pmasks))

        if variable:
            def getphi(params):
                return phi
            getphi.params = []

            gp = matrix.VariableGP(matrix.NoiseMatrix1D_var(getphi), Umatall)
            gp.index = {f'{psr.name}_{name}_coefficients({Umatall.shape[1]})': slice(0, Umatall.shape[1])}
            gp.name, gp.pos = psr.name, psr.pos
            gp.gpname, gp.gpcommon = name, []

            return gp
        else:
            return matrix.ConstantGP(matrix.NoiseMatrix1D_novar(phi), Umatall)
    else:
        pmasks = [matrix.jnparray(pmask) for pmask in pmasks]
        ecorr_names = params
        def getphi(params):
            return sum(10.0**(2 * (logscale + params[log10_ecorr])) * pmask
                       for (log10_ecorr, pmask) in zip(ecorr_names, pmasks))
        getphi.params = ecorr_names

        gp = matrix.VariableGP(matrix.NoiseMatrix1D_var(getphi), Umatall)
        gp.index = {f'{psr.name}_{name}_coefficients({Umatall.shape[1]})': slice(0, Umatall.shape[1])}
        gp.name, gp.pos = psr.name, psr.pos
        gp.gpname, gp.gpcommon = name, []

        return gp

def makegp_ecorr_legendre_modulated(psr, noisedict={}, enterprise=False, scale=1.0,
                                    selection=selection_backend_flags, variable=False,
                                    nmodes=3, modulation=2, fref=None, name='modecorrGPleg',
                                    tobs_scale=False, tobs_ref=1800.0):
    """Legendre ECORR GP whose jitter is modulated by a low-order profile across the band.

    Each epoch's jitter is ``j(y) = m(y) u(y)``. ``u`` is the diagonal Legendre process of
    :func:`makegp_ecorr_legendre` (same epochs, ``y`` mapping, per-backend handling, amplitude
    names and priors), and

        m(y) = 1 + sum_{d=1..modulation} b_d P_d(y)

    is a per-backend profile with parameters ``{psr.name}_{backend}_ecorr_mod_b{d}``. The
    within-epoch covariance is ``K(y, y') = m(y) m(y') sum_k sigma_k^2 P_k(y) P_k(y')``: the b_d
    set the variance profile ``V(y) = m(y)^2 V_u(y)`` (b_1 tilts it, negative b_2 flattens the
    band edges) and leave the correlation coefficient across the band to the sigma_k. All
    b_d = 0 is exactly :func:`makegp_ecorr_legendre`.

    The basis is the diagonal model's, with each TOA's row multiplied by m at that TOA, so it
    depends on the b_d when they are sampled; the prior on the mode amplitudes stays diagonal.

    The standard prior, each b_d uniform in [-0.4, 0.4], keeps m >= 0.6 on [-1, 1] for D = 1 and
    m >= 0.2 for D = 2. For D = 3 it allows m down to -0.2 at the band edges (|y| >= 0.948)
    near the corners of the box; a sign change there is a degenerate model, not an error.

    psr:        Discovery Pulsar object
    noisedict:  fixed parameter values; amplitudes are fixed only if all are given, as in
                :func:`makegp_ecorr_legendre`, and each b_d given is fixed
    nmodes:     number of Legendre modes in u (integer >= 1)
    modulation: degree D of the profile m (integer >= 1)
    name:       GP name; the coefficients are indexed as ``{psr.name}_{name}_coefficients(n)``
    Other arguments as :func:`makegp_ecorr_legendre`.

    Raises ValueError if nmodes or modulation is not an integer >= 1, or if a b_d fixed in
    noisedict lies outside its prior.
    """
    if not isinstance(modulation, (int, np.integer)) or isinstance(modulation, bool) or modulation < 1:
        raise ValueError('makegp_ecorr_legendre_modulated: modulation must be an integer >= 1')
    if not isinstance(nmodes, (int, np.integer)) or isinstance(nmodes, bool) or nmodes < 1:
        raise ValueError('makegp_ecorr_legendre_modulated: nmodes must be an integer >= 1')
    modulation, nmodes = int(modulation), int(nmodes)

    base = makegp_ecorr_legendre(psr, noisedict=noisedict, enterprise=enterprise, scale=scale,
                                 selection=selection, variable=True, nmodes=nmodes, fref=fref,
                                 name=name, tobs_scale=tobs_scale, tobs_ref=tobs_ref)
    Umatall = np.asarray(base.F)

    # per-TOA band coordinate y and backend membership, mapped as in makegp_ecorr_legendre
    freqs = np.asarray(psr.freqs)
    backend_flags = selection(psr)
    backends = [b for b in sorted(set(backend_flags)) if b != '']
    y = np.zeros(len(freqs))
    membership = np.zeros((len(freqs), len(backends)))
    for j, backend in enumerate(backends):
        mask = np.array(backend_flags == backend)
        fref_b = float(np.mean(freqs[mask])) if fref is None else float(fref)
        x = np.log(freqs / fref_b)
        x_min, x_max = float(x[mask].min()), float(x[mask].max())
        half_range = 0.5 * (x_max - x_min)
        if half_range != 0.0:
            y[mask] = ((x - 0.5 * (x_max + x_min)) / half_range)[mask]
        membership[mask, j] = 1.0
    profile = np.polynomial.legendre.legvander(y, modulation)[:, 1:]

    bnames = [[f'{psr.name}_{backend}_ecorr_mod_b{d}' for d in range(1, modulation + 1)]
              for backend in backends]
    fixed = {b: float(noisedict[b]) for row in bnames for b in row if b in noisedict}
    if fixed:
        from . import prior as _prior
        outside = {b: v for b, v in fixed.items()
                   if not _prior.getprior_uniform(b)[0] <= v <= _prior.getprior_uniform(b)[1]}
        if outside:
            raise ValueError(f'makegp_ecorr_legendre_modulated: {psr.name}: fixed modulation '
                             f'value(s) outside the prior: {outside}.')
    sampled = [b for row in bnames for b in row if b not in fixed]

    def modulation_factor(params, xp):
        bmat = xp.array([[fixed[b] if b in fixed else params[b] for b in row] for row in bnames])
        return 1.0 + xp.sum((xp.asarray(membership) @ bmat) * xp.asarray(profile), axis=1)

    if sampled:
        Uj = matrix.jnparray(Umatall)

        def F(params):
            return modulation_factor(params, jnp)[:, None] * Uj
        F.params = sampled
    else:
        F = modulation_factor({}, np)[:, None] * Umatall

    if not sampled and not base.Phi.params and not variable:
        gp = matrix.ConstantGP(matrix.NoiseMatrix1D_novar(np.asarray(base.Phi.getN({}))), F)
    else:
        gp = matrix.VariableGP(base.Phi, F)
        gp.index = {f'{psr.name}_{name}_coefficients({Umatall.shape[1]})': slice(0, Umatall.shape[1])}
        gp.name, gp.pos = psr.name, psr.pos
        gp.gpname, gp.gpcommon = name, []
    gp.modulation, gp.modulation_params = modulation, [b for row in bnames for b in row]

    return gp

def makegp_quadratic_ecorr_legendre(psr, noisedict={}, enterprise=False, scale=1.0,
                                    selection=selection_backend_flags, variable=False,
                                    fref=None, name='quadecorrGPleg',
                                    tobs_scale=False, tobs_ref=1800.0):
    """Three-mode ECORR GP using a Legendre-polynomial frequency basis.

    Thin wrapper around :func:`makegp_ecorr_legendre` with ``nmodes=3`` (Legendre
    degrees 0-2), preserved for backward compatibility (parameters
    ``..._log10_ecorr``, ``_k1``, ``_k2`` per backend).
    """
    return makegp_ecorr_legendre(psr, noisedict=noisedict, enterprise=enterprise, scale=scale,
                                 selection=selection, variable=variable,
                                 nmodes=3, fref=fref, name=name,
                                 tobs_scale=tobs_scale, tobs_ref=tobs_ref)

def make_withinday_powerlaw(f_lo=1.0 / 86400.0, f_hi=None, nf=64):
    """Band-limited power-law PSD as a covariance in lag, for :func:`makegp_withinday`.

    ``S(f) = f**-gamma`` over ``nf`` log-spaced frequencies in ``[f_lo, f_hi]``, summed to
    ``k(tau) = sum_k S(f_k) df_k cos(2 pi f_k tau)`` and normalised to ``k(0) = sigma**2``,
    so ``log10_sigma`` is the within-day rms in seconds and ``gamma`` only sets the shape.
    Written this way ``k`` is positive semi-definite by construction -- it is
    ``C diag(w) C^T + S diag(w) S^T`` for the cosine and sine bases -- and differentiable
    in both parameters without special functions.

    ``f_lo`` is the outer scale, beyond which the process decorrelates. The default of one
    day matches the blocking and assumes nothing about a pulsar; set it per pulsar.

    ``f_hi`` left as None builds an unbound kernel, which :func:`makegp_withinday` binds to
    the highest frequency the epochs resolve; calling it directly raises. Pass a number
    to evaluate the kernel outside a GP.

    A structure function quoted as ``sqrt(D/2) = sqrt(k(0) - k(tau))`` is an rms per point,
    not ``sqrt(D)``, the rms of the difference; fitting the second form to the first
    returns a ``sigma`` low by ``sqrt(2)``.

    The kernel loses rank once a day holds more epochs than it can span: its condition
    number runs 8.6 at 4 epochs and 25 at 16, then 1.6e288 at 64 and beyond, where the
    Cholesky fails. The ECORR diagonal of :func:`makegp_withinday` is the nugget that
    regularises it, holding the condition number under 300 at 256 epochs.

    f_lo, f_hi: frequency bounds in Hz; f_hi None binds it to the epoch sampling
    nf:         number of log-spaced modes
    """
    if f_hi is None:
        return _withinday_unbound(make_withinday_powerlaw, f_lo=f_lo, nf=nf)

    if not f_hi > f_lo:
        raise ValueError(f'make_withinday_powerlaw: f_hi = 1/{1 / f_hi:.0f} s is not above f_lo = '
                         f'1/{1 / f_lo:.0f} s, so the kernel has no band. For a kernel bound '
                         f'to the epoch sampling this means the epochs resolve nothing above '
                         f'the outer scale.')

    f = np.exp(np.linspace(np.log(f_lo), np.log(f_hi), int(nf)))
    dlnf = np.log(f_hi / f_lo) / (int(nf) - 1)
    fj, wj = matrix.jnparray(f), matrix.jnparray(f * dlnf)

    def covariance(tau, log10_sigma, gamma):
        w = wj * (fj / fj[0]) ** (-gamma)
        w = w / jnp.sum(w) * 10.0 ** (2.0 * log10_sigma)
        return jnp.sum(w[:, None, None] * jnp.cos(2.0 * np.pi * fj[:, None, None] * tau[None]), axis=0)

    covariance.f_hi = float(f_hi)
    return covariance


def make_withinday_brokenpowerlaw(f_lo=1.0 / 86400.0, f_hi=None, nf=96):
    """Broken power-law PSD as a covariance in lag, for :func:`makegp_withinday`.

    ``S(f) = (f/f_b)**-gamma_lo`` below the break and ``(f/f_b)**-gamma_hi`` above it,
    summed over log-spaced modes as :func:`make_withinday_powerlaw` does and normalised to
    ``k(0) = sigma**2``; equal slopes give that kernel exactly. ``gamma_hi`` prices the
    power between the break and the band edge. It cannot price power above the edge,
    which is why the edge is set from the epoch sampling rather than chosen: ``f_hi``
    None binds it there, as for :func:`make_withinday_powerlaw`.

    f_lo, f_hi: frequency bounds in Hz; f_hi None binds it to the epoch sampling
    nf:         number of log-spaced modes
    """
    if f_hi is None:
        return _withinday_unbound(make_withinday_brokenpowerlaw, f_lo=f_lo, nf=nf)

    if not f_hi > f_lo:
        raise ValueError(f'make_withinday_brokenpowerlaw: f_hi = 1/{1 / f_hi:.0f} s is not above f_lo = '
                         f'1/{1 / f_lo:.0f} s, so the kernel has no band. For a kernel bound '
                         f'to the epoch sampling this means the epochs resolve nothing above '
                         f'the outer scale.')

    f = np.exp(np.linspace(np.log(f_lo), np.log(f_hi), int(nf)))
    dlnf = np.log(f_hi / f_lo) / (int(nf) - 1)
    fj, wj = matrix.jnparray(f), matrix.jnparray(f * dlnf)

    def covariance(tau, log10_sigma, gamma_lo, gamma_hi, log10_fbreak):
        fb = 10.0 ** log10_fbreak
        x = fj / fb
        w = wj * jnp.where(fj < fb, x ** (-gamma_lo), x ** (-gamma_hi))
        w = w / jnp.sum(w) * 10.0 ** (2.0 * log10_sigma)
        return jnp.sum(w[:, None, None] * jnp.cos(2.0 * np.pi * fj[:, None, None] * tau[None]), axis=0)

    # the break must stay inside the modelled band, or one slope stops being identified
    covariance.prior_boxes = {'log10_fbreak': [float(np.log10(f_lo)) + 0.2,
                                               float(np.log10(f_hi)) - 0.2]}
    covariance.f_hi = float(f_hi)
    return covariance


def _withinday_unbound(factory, **kw):
    """Placeholder for a within-day kernel whose ``f_hi`` is set from the data."""
    def covariance(*args, **kwargs):
        raise ValueError(f'{factory.__name__}: f_hi is None, so this kernel is bound to the '
                         f'epoch sampling by makegp_withinday; pass f_hi to evaluate it '
                         f'directly.')
    covariance.unbound = True
    covariance.bind = lambda f_hi: factory(f_hi=f_hi, **kw)
    return covariance


def _withinday_epochs(psr, selection, dt, tobs_scale, tobs_ref, what):
    """ECORR's per-backend epoch basis and the epoch times, for the within-day GP."""
    backend_flags = selection(psr)
    backends = [b for b in sorted(set(backend_flags)) if b != '']

    Umats = []
    for b in backends:
        mask = np.array(backend_flags == b)
        first = 0 if len(backends) == 1 else 1
        bins = quantize(psr.toas * mask, dt)
        Umats.append(np.vstack([bins == i for i in range(first, bins.max() + 1)]).T)
    Umatall = np.hstack(Umats).astype(np.float64)
    if tobs_scale:
        Umatall = Umatall * ecorr_tobs_weights(psr, tobs_ref)[:, None]

    counts = Umatall.sum(axis=0)
    if np.any(counts <= 0):
        raise ValueError(f'{what}: {psr.name} has an empty epoch column; the '
                         f'selection and dt must partition the TOAs.')
    return backends, Umats, Umatall, (np.asarray(psr.toas) @ Umatall) / counts


def _withinday_fhi(psr, epoch_t, day, what):
    dayidx = np.floor(epoch_t / day).astype(np.int64)
    gaps = [np.diff(np.sort(epoch_t[dayidx == d])) for d in np.unique(dayidx)]
    gaps = [g for g in gaps if g.size]
    if not gaps:
        raise ValueError(f'{what}: {psr.name} has no day holding more than one epoch, so no '
                         f'within-day correlation is measurable and the component is ECORR.')
    return 1.0 / (2.0 * float(np.median(np.concatenate(gaps))))


def withinday_resolution(psr, selection=selection_backend_flags, dt=1.0, day=86400.0,
                         tobs_scale=False, tobs_ref=1800.0):
    """Highest frequency the within-day epochs resolve, in Hz.

    ``1 / (2 x median gap between consecutive epochs of one day)``, over the epochs
    :func:`makegp_withinday` builds with the same arguments.
    """
    *_, epoch_t = _withinday_epochs(psr, selection, dt, tobs_scale, tobs_ref,
                                    'withinday_resolution')
    return _withinday_fhi(psr, epoch_t, day, 'withinday_resolution')


def makegp_withinday(psr, covariance=None, noisedict={}, scale=1.0,
                     selection=selection_backend_flags, dt=1.0, day=86400.0,
                     ecorr=True, tobs_scale=False, tobs_ref=1800.0,
                     common=[], name='withinday_gp', resolution_day=None):
    """Achromatic noise correlated within an observing day and independent between days.

    Fills the gap between a Fourier GP, whose highest frequency is set by the array
    cadence, and ECORR, which is independent per epoch. The basis is ECORR's own per-epoch
    indicator matrix, so this is ECORR with a correlated prior rather than a diagonal one:
    the prior is block diagonal by day, with block ``g``

        Phi_g = covariance(|t_i - t_j|, ...) + diag(ecorr**2)

    over the epochs of that day. It therefore CARRIES the ECORR term and replaces
    :func:`makegp_ecorr` rather than sitting beside it -- two GPs on one basis would
    duplicate every column and leave the diagonal split between them unidentified. The
    ``{psr}_{backend}_log10_ecorr`` names are unchanged, so a chain from
    :func:`makegp_ecorr` supplies them directly.

    The diagonal is not optional wherever a day is densely sampled. The bare kernel goes
    singular once a day holds more epochs than it spans -- its condition number is 25 at
    16 epochs and 1e288 by 64 -- and J1902-5105 reaches 174 epochs in a day, where the
    Cholesky fails outright; the ECORR variance regularises it to a condition number of
    40 at the fitted 93 ns. A pulsar observed a few times a night never reaches that
    regime. Days holding a single epoch carry only the diagonal, so there the component
    and ECORR are exactly degenerate and only their sum is identified.

    Blocking by day is a modelling choice, not a measurement: it asserts that the
    day-to-day correlation is already carried by the red, DM and chromatic GPs, so what
    remains is the within-day residual and nothing is counted twice. It holds while the
    correlation length stays well inside a day.

    The kernel is purely temporal and the basis carries no frequency weight, so the
    component is achromatic by construction and has no chromatic index.

    The kernel band ends at :func:`withinday_resolution`, the highest frequency the epochs
    resolve. Above it no pair of epochs is close enough to see a correlation and the kernel
    is degenerate with ECORR, so variance placed there is split between the two by the
    kernel's prior rather than the data. A kernel built without ``f_hi`` is bound to that
    edge here, and one built with a higher ``f_hi`` gets a warning. The edge follows the
    subintegration length, so fits at different subintegrations are not directly
    comparable, and the band from ``f_lo`` to the edge carries all the information.

    psr:        Discovery Pulsar object
    covariance: callable ``covariance(tau, ...)`` in seconds, its arguments after ``tau``
                sampled as ``{psr}_{name}_{arg}``, or as ``{psr}_{stem}_{arg}`` for an
                argument its ``stems`` attribute maps to ``stem``; defaults to
                :func:`make_withinday_powerlaw` bound to the epoch sampling
    noisedict:  fixed ``{psr}_{backend}_log10_ecorr`` values; those present are held
                fixed rather than sampled
    dt:         epoch quantisation in seconds, as :func:`quantize` uses
    day:        block length in seconds; ``np.inf`` gives one block over all epochs
    ecorr:      add the per-backend ECORR diagonal; without it the kernel is singular, so
                False is for tests only
    selection, scale, tobs_scale, tobs_ref: as :func:`makegp_ecorr`
    name:       parameter-name stem for the kernel
    resolution_day: grouping in seconds for :func:`withinday_resolution`; defaults to
                ``day``
    """

    covariance = make_withinday_powerlaw() if covariance is None else covariance
    backends, Umats, Umatall, epoch_t = _withinday_epochs(psr, selection, dt, tobs_scale,
                                                          tobs_ref, 'makegp_withinday')
    log10_ecorrs = [f'{psr.name}_{b}_log10_ecorr' for b in backends]

    f_res = _withinday_fhi(psr, epoch_t, day if resolution_day is None else resolution_day,
                           'makegp_withinday')
    if getattr(covariance, 'unbound', False):
        covariance = covariance.bind(f_res)
        print(f'makegp_withinday: {psr.name}: kernel band up to 1/{1 / f_res:.0f} s, the '
              f'highest frequency the within-day epochs resolve.')
    elif getattr(covariance, 'f_hi', None) is not None and covariance.f_hi > f_res * (1 + 1e-9):
        print(f'Warning: makegp_withinday: {psr.name}: the kernel band reaches '
              f'1/{1 / covariance.f_hi:.0f} s, {np.log2(covariance.f_hi / f_res):.1f} octaves '
              f'above the 1/{1 / f_res:.0f} s the epochs resolve. There it is degenerate '
              f'with ECORR, so its prior rather than the data sets the split.')

    stems = getattr(covariance, 'stems', {})
    kargs = [arg for arg in inspect.getfullargspec(covariance).args if arg not in ('tau', 'tau_t')]
    argmap = [(arg if arg in common
               else f'{name}_{arg}' if f'{name}_{arg}' in common
               else f'{psr.name}_{stems.get(arg, name)}_{arg}')
              for arg in kargs]

    # the days holding a single epoch become one diagonal chunk rather than that many
    # one-by-one factorisations, since the kernel is diagonal on a single epoch
    dayidx = np.floor(epoch_t / day).astype(np.int64)
    uday, inv = np.unique(dayidx, return_inverse=True)
    multi = np.bincount(inv, minlength=len(uday))[inv] > 1
    order = np.lexsort((inv, multi))

    Umatall, epoch_t, inv, multi = Umatall[:, order], epoch_t[order], inv[order], multi[order]
    pmasks, cnt = [], 0
    for Umat in Umats:
        z = np.zeros(Umatall.shape[1], dtype=np.float64)
        z[cnt:cnt + Umat.shape[1]] = 1.0
        pmasks.append(z[order])
        cnt += Umat.shape[1]

    n_single = int(np.sum(~multi))
    spans, taus = [], []
    if n_single:
        spans.append((0, n_single, True))
    i0 = n_single
    for d in dict.fromkeys(inv[n_single:].tolist()):
        tt = epoch_t[inv == d]
        taus.append(matrix.jnparray(np.abs(tt[:, None] - tt[None, :])))
        spans.append((i0, len(tt), False))
        i0 += len(tt)

    logscale = np.log10(scale)
    pmasks_j = [matrix.jnparray(pm) for pm in pmasks]
    total = Umatall.shape[1]
    zero = matrix.jnparray(np.zeros((1, 1)))

    def _diag(params):
        if not ecorr:
            return jnp.zeros(total)
        return sum(10.0 ** (2 * (logscale + (noisedict[p] if p in noisedict else params[p]))) * pm
                   for p, pm in zip(log10_ecorrs, pmasks_j))

    blockfuncs, k = [], 0
    for (start, n, isdiag) in spans:
        if isdiag:
            def getb(params, start=start, n=n):
                return (jax.lax.dynamic_slice(_diag(params), (start,), (n,))
                        + covariance(zero, *[params[a] for a in argmap])[0, 0])
            blockfuncs.append((True, getb))
        else:
            def getb(params, start=start, n=n, tau=taus[k]):
                return (covariance(tau, *[params[a] for a in argmap])
                        + jnp.diag(jax.lax.dynamic_slice(_diag(params), (start,), (n,))))
            blockfuncs.append((False, getb))
            k += 1

    def getphi(params):
        return matrix.jsp.linalg.block_diag(
            *[(jnp.diag(f(params)) if isdiag else f(params)) for isdiag, f in blockfuncs])
    getphi.params = ([p for p in log10_ecorrs if p not in noisedict] if ecorr else []) + argmap

    Phi = matrix.BlockDiagNoiseMatrix2D_var(getphi, blockfuncs)
    Phi.params = getphi.params
    gp = matrix.VariableGP(Phi, Umatall)
    gp.index = {f'{psr.name}_{name}_coefficients({total})': slice(0, total)}
    gp.name, gp.pos, gp.gpname, gp.gpcommon = psr.name, psr.pos, name, common
    gp.wd_epochs, gp.wd_blocks, gp.wd_nsingle = epoch_t, spans, n_single
    gp.wd_fhi = f_res
    boxes = getattr(covariance, 'prior_boxes', {})
    gp.prior_boxes = {p: boxes[a] for a, p in zip(
        [a for a in inspect.getfullargspec(covariance).args if a not in ('tau', 'tau_t')], argmap)
        if a in boxes}
    return gp


def _step_kernel(tau, w, f, df):
    """``sum_k w_k cos(2 pi f_k tau) sinc(df_k tau)``, accumulated one mode at a time and
    recomputed in the gradient, so memory is one lag matrix rather than one per mode."""
    def step(acc, mode):
        wk, fk, dk = mode
        return acc + wk * jnp.cos(2.0 * np.pi * fk * tau) * jnp.sinc(dk * tau), None

    acc, _ = jax.lax.scan(jax.checkpoint(step), jnp.zeros(jnp.shape(tau)), (w, f, df))
    return acc


def make_red_tail(psd, f_lo, f_hi=None, nf=128, broken=False, fbreak_max=None,
                  stem='red_noise'):
    """A Fourier red-noise PSD continued above its highest bin, as a covariance in lag.

    ``psd`` is the callable the Fourier GP uses -- ``psd(f, df, *params)`` returning
    ``P(f) df`` with its absolute normalisation -- integrated over ``nf`` log-spaced bins
    spanning ``[f_lo, f_hi]``. Bin ``k`` has variance ``w_k``, the Simpson integral of
    ``P`` across it, and is taken flat across its width ``df_k`` about its midpoint
    ``f_k``, whose covariance is exactly

        k(tau) = sum_k w_k cos(2 pi f_k tau) sinc(df_k tau)

    so correlations decay across the span instead of recurring, and ``k`` is positive
    semi-definite. The PSD's own parameters are sampled under ``stem``, the Fourier GP's
    name, so the two share one spectrum. ``broken`` steepens it above a break,
    ``P(f) = P(f_b) (f / f_b)**-gamma_hi`` for ``f > f_b``, sampling ``gamma_hi`` and
    ``log10_fbreak`` under the covariance's own name, with ``f_b`` bounded to
    ``[f_lo, fbreak_max]``.

    ``covariance.spectrum(f, *params)`` is the one-sided PSD ``P(f)`` in s^2/Hz, the
    Fourier GP's below ``f_lo`` and the tail's above it, so ``sqrt(f P(f))`` is the rms per
    e-fold of frequency. ``covariance.weights(*params)`` are the bin variances ``w_k``; the
    bin holding the break is split there and each part integrated on its own slope, so they
    are continuous in ``log10_fbreak``.

    psd:        PSD callable ``psd(f, df, *params)``
    f_lo:       lower band edge in Hz, the upper edge of the Fourier GP's last bin
    f_hi:       upper band edge in Hz; None binds it to the epoch resolution in
                :func:`makegp_withinday`
    nf:         number of log-spaced bins
    broken:     add the break
    fbreak_max: upper bound of the break frequency in Hz; defaults to ``f_hi``
    stem:       parameter stem the PSD's parameters share, the Fourier GP's name
    """
    if f_hi is None:
        return _withinday_unbound(make_red_tail, psd=psd, f_lo=f_lo, nf=nf, broken=broken,
                                  fbreak_max=fbreak_max, stem=stem)
    if not f_hi > f_lo:
        raise ValueError(f'make_red_tail: f_hi = 1/{1 / f_hi:.0f} s is not above f_lo = '
                         f'1/{1 / f_lo:.0f} s, so the tail has no band.')
    fb_max = float(f_hi) if fbreak_max is None else float(fbreak_max)
    if not f_lo < fb_max <= f_hi:
        raise ValueError(f'make_red_tail: fbreak_max = 1/{1 / fb_max:.0f} s is not inside the '
                         f'tail band (1/{1 / f_lo:.0f} s, 1/{1 / f_hi:.0f} s].')

    edges = np.exp(np.linspace(np.log(f_lo), np.log(f_hi), int(nf) + 1))
    fm, dfk = 0.5 * (edges[1:] + edges[:-1]), np.diff(edges)
    fm_j, df_j, e_j = matrix.jnparray(fm), matrix.jnparray(dfk), matrix.jnparray(edges[:-1])
    pnames = list(inspect.signature(psd).parameters)[2:]
    extra = ['gamma_hi', 'log10_fbreak'] if broken else []

    def spectrum(f, *args):
        f, pars = jnp.asarray(f), args[:len(pnames)]
        P = psd(f, 1.0, *pars)
        if broken:
            gamma_hi, log10_fbreak = args[len(pnames):]
            fb = 10.0 ** log10_fbreak
            P = jnp.where(f > fb, psd(fb, 1.0, *pars) * (f / fb) ** (-gamma_hi), P)
        return P

    def simpson(a, d, P):
        return d / 6.0 * (P(a) + 4.0 * P(a + 0.5 * d) + P(a + d))

    def weights(*args):
        pars = args[:len(pnames)]
        P_lo = lambda f: psd(f, 1.0, *pars)
        if not broken:
            return simpson(e_j, df_j, P_lo)
        # the bin holding the break is split there, each part integrated on its own slope
        gamma_hi, log10_fbreak = args[len(pnames):]
        fb = 10.0 ** log10_fbreak
        P_b = psd(fb, 1.0, *pars)
        P_hi = lambda f: P_b * (f / fb) ** (-gamma_hi)
        below = jnp.clip(fb - e_j, 0.0, df_j)
        return simpson(e_j, below, P_lo) + simpson(e_j + below, df_j - below, P_hi)

    def covariance(tau, *args):
        return _step_kernel(tau, weights(*args), fm_j, df_j)

    kind = inspect.Parameter.POSITIONAL_OR_KEYWORD
    sig = inspect.Signature([inspect.Parameter(n, kind) for n in ['tau'] + pnames + extra])
    covariance.__signature__ = sig
    spectrum.__signature__ = sig.replace(parameters=[inspect.Parameter('f', kind)]
                                         + list(sig.parameters.values())[1:])
    covariance.spectrum, covariance.weights = spectrum, weights
    covariance.stems = {n: stem for n in pnames}
    covariance.prior_boxes = ({'log10_fbreak': [float(np.log10(f_lo)), float(np.log10(fb_max))]}
                              if broken else {})
    covariance.f_hi, covariance.f_lo = float(f_hi), float(f_lo)
    covariance.modes = (fm, dfk)
    return covariance


def makegp_red_tail(psr, psd, components, T=None, nf=128, broken=False, fbreak_max=None,
                    dt=1.0, resolution_day=86400.0, noisedict={}, scale=1.0,
                    selection=selection_backend_flags, tobs_scale=False, tobs_ref=1800.0,
                    stem='red_noise', name='red_tail'):
    """Red noise continued past a Fourier GP's last bin up to the epoch resolution.

    A Fourier GP of ``components`` bins centred on ``k/T`` with width ``1/T`` covers
    frequencies up to ``(components + 1/2)/T``; this carries the same PSD from there to
    :func:`withinday_resolution`, so no frequency is counted twice and none is skipped.
    It is :func:`makegp_withinday` with :func:`make_red_tail` as the kernel and one block
    over all epochs, so every epoch is correlated with every other and ECORR is the
    nugget. Like that function it carries the ECORR term and replaces
    :func:`makegp_ecorr`.

    With ``broken`` False it adds no parameters: the spectrum is the Fourier GP's, so a
    chain carries only the red-noise and ECORR names and cannot be told from a
    standard one by its columns. A later stage rebuilding the model has to be told the
    tail was used.

    The prior is dense over all epochs, so its factorisation grows as the cube of the
    epoch count and dominates the cost once that count runs to thousands.

    psr:            Discovery Pulsar object
    psd:            the Fourier GP's PSD callable
    components, T:  the Fourier GP's bin count and span; T defaults to the pulsar span
    nf, broken, fbreak_max: as :func:`make_red_tail`
    dt:             epoch quantisation in seconds
    resolution_day: grouping in seconds for :func:`withinday_resolution`
    stem:           the Fourier GP's name, under which the PSD parameters are shared
    name:           name of this GP, and stem for the break parameters
    noisedict, scale, selection, tobs_scale, tobs_ref: as :func:`makegp_withinday`
    """
    T = getspan(psr) if T is None else T
    f_lo = (int(components) + 0.5) / T
    f_res = withinday_resolution(psr, selection, dt, resolution_day, tobs_scale, tobs_ref)
    if not f_res > f_lo:
        raise ValueError(f'makegp_red_tail: {psr.name}: the epochs resolve only up to '
                         f'1/{1 / f_res:.0f} s, below the Fourier GP\'s last bin edge at '
                         f'1/{1 / f_lo:.0f} s, so there is no band left for the tail.')

    gp = makegp_withinday(psr, covariance=make_red_tail(psd, f_lo, f_hi=f_res, nf=nf,
                                                        broken=broken, fbreak_max=fbreak_max,
                                                        stem=stem),
                          noisedict=noisedict, scale=scale, selection=selection, dt=dt,
                          day=np.inf, resolution_day=resolution_day, ecorr=True,
                          tobs_scale=tobs_scale, tobs_ref=tobs_ref, name=name)
    gp.tail_band = (f_lo, f_res)
    return gp


def make_band_powerlaw(flo_range, f_hi=None, nf=256):
    """Power law above a sampled outer scale, as a covariance in lag.

    ``S(f) ~ f**-gamma`` on ``[f_lo, f_hi]`` and zero below ``f_lo``, normalised to
    ``k(0) = sigma**2``. The band is split into ``nf`` log-spaced bins whose edges move with
    ``f_lo``, each taken flat across its width, so

        k(tau) = sum_k w_k cos(2 pi f_k tau) sinc(df_k tau)

    is the exact covariance of that step spectrum: positive semi-definite, decaying across
    the span rather than recurring, and differentiable in ``log10_flo`` as well as in
    ``log10_sigma`` and ``gamma``. ``covariance.spectrum(f, log10_sigma, gamma, log10_flo)``
    is the one-sided PSD in s^2/Hz, so ``sqrt(f S(f))`` is the rms per e-fold of frequency.

    flo_range: ``(lo, hi)`` in Hz, the prior box of ``f_lo``
    f_hi:      upper band edge in Hz; None binds it to the epoch resolution in
               :func:`makegp_withinday`
    nf:        number of log-spaced bins
    """
    if f_hi is None:
        return _withinday_unbound(make_band_powerlaw, flo_range=flo_range, nf=nf)
    lo, hi = float(flo_range[0]), float(flo_range[1])
    if not 0.0 < lo < hi < f_hi:
        raise ValueError(f'make_band_powerlaw: the f_lo range (1/{1 / lo:.0f} s, '
                         f'1/{1 / hi:.0f} s) must lie below f_hi = 1/{1 / f_hi:.0f} s.')

    u = matrix.jnparray(np.arange(int(nf) + 1) / int(nf))
    ln_fhi = float(np.log(f_hi))

    def modes(log10_flo):
        ln_flo = log10_flo * np.log(10.0)
        edges = jnp.exp(ln_flo + (ln_fhi - ln_flo) * u)
        return 0.5 * (edges[1:] + edges[:-1]), jnp.diff(edges)

    def norm(log10_sigma, gamma, fm, dfk):
        return 10.0 ** (2.0 * log10_sigma) / jnp.sum((fm / f_hi) ** (-gamma) * dfk)

    def covariance(tau, log10_sigma, gamma, log10_flo):
        fm, dfk = modes(log10_flo)
        w = norm(log10_sigma, gamma, fm, dfk) * (fm / f_hi) ** (-gamma) * dfk
        return _step_kernel(tau, w, fm, dfk)

    def spectrum(f, log10_sigma, gamma, log10_flo):
        f = jnp.asarray(f)
        fm, dfk = modes(log10_flo)
        inband = (f >= 10.0 ** log10_flo) & (f <= f_hi)
        return jnp.where(inband, norm(log10_sigma, gamma, fm, dfk) * (f / f_hi) ** (-gamma), 0.0)

    covariance.spectrum = spectrum
    covariance.modes_at = modes
    covariance.prior_boxes = {'log10_flo': [float(np.log10(lo)), float(np.log10(hi))]}
    covariance.f_hi = float(f_hi)
    return covariance


def makegp_red_hf(psr, flo_range, nf=256, dt=1.0, resolution_day=86400.0, noisedict={},
                  scale=1.0, selection=selection_backend_flags, tobs_scale=False,
                  tobs_ref=1800.0, name='red_hf'):
    """An achromatic process above a sampled outer scale, correlated across all epochs.

    :func:`makegp_withinday` with :func:`make_band_powerlaw` as the kernel and one block
    over all epochs, so every epoch is correlated with every other and ECORR is the
    nugget; like that function it carries the ECORR term and replaces
    :func:`makegp_ecorr`. Its amplitude and slope are its own, independent of any
    Fourier GP, sampled as ``{psr}_{name}_log10_sigma``, ``_gamma`` and ``_log10_flo``.
    With ``flo_range`` starting at a Fourier GP's last bin edge the two cover disjoint
    bands.

    The prior is dense over all epochs, so its factorisation grows as the cube of the
    epoch count.

    psr:            Discovery Pulsar object
    flo_range:      ``(lo, hi)`` in Hz, the prior box of the outer scale
    nf:             number of log-spaced bins
    dt:             epoch quantisation in seconds
    resolution_day: grouping in seconds for :func:`withinday_resolution`, which sets the
                    upper band edge
    name:           name of this GP and stem of its parameters
    noisedict, scale, selection, tobs_scale, tobs_ref: as :func:`makegp_withinday`
    """
    f_res = withinday_resolution(psr, selection, dt, resolution_day, tobs_scale, tobs_ref)
    if not f_res > flo_range[1]:
        raise ValueError(f'makegp_red_hf: {psr.name}: the epochs resolve only up to '
                         f'1/{1 / f_res:.0f} s, not above the top of the outer-scale range at '
                         f'1/{1 / flo_range[1]:.0f} s.')

    gp = makegp_withinday(psr, covariance=make_band_powerlaw(flo_range, f_hi=f_res, nf=nf),
                          noisedict=noisedict, scale=scale, selection=selection, dt=dt,
                          day=np.inf, resolution_day=resolution_day, ecorr=True,
                          tobs_scale=tobs_scale, tobs_ref=tobs_ref, name=name)
    gp.flo_range, gp.f_hi = (float(flo_range[0]), float(flo_range[1])), f_res
    return gp


def makegp_ecorr_legendre_correlated(psr, noisedict={}, enterprise=False, scale=1.0,
                                     selection=selection_backend_flags, variable=False,
                                     nmodes=3, fref=None,
                                     name='correcorrGPleg',
                                     tobs_scale=False, tobs_ref=1800.0):
    """ECORR GP with a *correlated* Legendre frequency basis (full mode covariance).

    Identical within-epoch basis to :func:`makegp_ecorr_legendre`, but the
    ``nmodes`` Legendre mode amplitudes share a full positive-definite covariance
    matrix ``M`` per backend rather than being independent. Writing the
    within-epoch frequency covariance as

        K(y_i, y_j) = sum_{a,b} M_ab P_a(y_i) P_b(y_j),

    the off-diagonal (a != b) cross terms break the centrosymmetry the diagonal
    model is forced into. In particular the odd-parity term M_01 (P_0 P_1 ~ y)
    *tilts* the variance across the band, so this model CAN represent jitter
    whose amplitude is higher at one end of the band -- e.g. the low-frequency
    excess seen in the J0437 jitter covariance -- which the diagonal
    :func:`makegp_ecorr_legendre` cannot (there K(y, y) = sum_m a_m^2 P_m(y)^2
    is even in y and hence symmetric about band centre for any parameters).

    M is parametrised as ``M = D R D`` with ``D`` the diagonal of per-mode
    amplitudes (times ``scale``), so the amplitudes (``..._log10_ecorr`` for mode
    0, ``..._log10_ecorr_k{m}`` above) keep exactly the same meaning and prior as
    in the diagonal model (R has unit diagonal, so each amplitude is the marginal
    std of its mode), and R is a unit-diagonal correlation matrix built from the
    parameters ``..._ecorr_corr_k{a}k{b}`` (a > b), each in (-1, 1), via the
    partial-correlation (C-vine) Cholesky construction: ``ecorr_corr_k{a}k0`` is
    the correlation of mode a with mode 0, and ``ecorr_corr_k{a}k{b}`` (b > 0) is
    the partial correlation of modes a and b given modes 0..b-1. Any combination
    of values in (-1, 1) yields a valid positive-semidefinite R (hence M), so the
    parameters can be given a flat [-1, 1] prior. Setting every correlation
    parameter to zero gives R = I and recovers :func:`makegp_ecorr_legendre`
    exactly, so the two models are nested for model comparison.

    This costs ``nmodes(nmodes+1)/2`` parameters per backend (6 at nmodes=3)
    versus ``nmodes`` for the diagonal model. The prior covariance Phi is block
    diagonal (one ``nmodes x nmodes`` block per epoch) but is currently assembled
    and inverted densely via NoiseMatrix2D, so it is best reserved for the bright,
    high-S/N pulsars where the extra parameters can be constrained.
    """

    if not isinstance(nmodes, int) or nmodes < 1:
        raise ValueError('makegp_ecorr_legendre_correlated: nmodes must be an integer >= 1')

    freqs = np.asarray(psr.freqs)

    backend_flags = selection(psr)
    backends = [b for b in sorted(set(backend_flags)) if b != '']
    masks = [np.array(backend_flags == backend) for backend in backends]

    k = nmodes  # dimension of the per-epoch mode covariance M

    U_backend = []               # per-backend basis blocks (columns epoch-major, mode-minor)
    nepochs_per_backend = []
    amp_params_per_backend = []   # [log10_ecorr, _k1 .. _k{nmodes-1}] amplitude param names per backend
    corr_params_per_backend = []  # {(a, b): name} off-diagonal correlation params per backend

    for backend, mask in zip(backends, masks):
        # per-backend reference frequency (controls the centring of x)
        fref_b = float(np.mean(freqs[mask])) if fref is None else float(fref)
        x = np.log(freqs / fref_b)

        # rescale x linearly to y in [-1, 1] across this backend's TOAs
        x_backend = x[mask]
        x_min, x_max = float(x_backend.min()), float(x_backend.max())
        half_range = 0.5 * (x_max - x_min)
        if half_range == 0.0:
            y = np.zeros_like(freqs)
        else:
            y = (x - 0.5 * (x_max + x_min)) / half_range

        # columns of legvander are P_0(y) ... P_{nmodes-1}(y)
        P = np.polynomial.legendre.legvander(y, nmodes - 1)

        # For handling the single backend case
        if len(np.unique(masks)) == 1:
            # for those pulsar with only one backend
            first_valid_bin = 0
        else:
            # if the mask contains zeros
            # the zeros in quantize below end up in the
            # first entry, which we skip later.
            first_valid_bin = 1

        bins = quantize(psr.toas * mask)
        if enterprise:
            uniques, counts = np.unique(bins, return_counts=True)
            epoch_masks = [bins == i for i, cnt in zip(
                uniques[first_valid_bin:],
                counts[first_valid_bin:]) if cnt > 1]

            if epoch_masks:
                U0 = np.vstack(epoch_masks).T
            else:
                # if there is no ToAs observed at the same time
                U0 = np.zeros((len(bins), 0))
        else:
            U0 = np.vstack([bins == i for i in range(first_valid_bin, bins.max() + 1)]).T
        U0 = U0.astype(np.float64)

        nep = U0.shape[1]
        if nep == 0:
            continue

        # columns ordered (epoch e outer, mode o inner) so the per-backend Phi
        # block is exactly kron(I_nep, M) below
        cols = np.empty((U0.shape[0], nep * k), dtype=np.float64)
        for e in range(nep):
            for o in range(k):
                cols[:, e * k + o] = U0[:, e] * P[:, o]
        U_backend.append(cols)
        nepochs_per_backend.append(nep)

        amp_params_per_backend.append([f'{psr.name}_{backend}_log10_ecorr' + ('' if m == 0 else f'_k{m}')
                                       for m in range(k)])
        corr_params_per_backend.append({(a, b): f'{psr.name}_{backend}_ecorr_corr_k{a}k{b}'
                                        for a in range(1, k) for b in range(a)})

    Umatall = np.hstack(U_backend)
    if tobs_scale:
        Umatall = Umatall * ecorr_tobs_weights(psr, tobs_ref)[:, None]

    params = []
    for amps in amp_params_per_backend:
        params.extend(amps)
    for corr in corr_params_per_backend:
        params.extend(corr.values())

    logscale = np.log10(scale)
    jnp_ = matrix.jnp

    def build_phi(p):
        mblocks = []
        for amps, corr, nep in zip(amp_params_per_backend, corr_params_per_backend, nepochs_per_backend):
            d = jnp_.stack([10.0 ** (logscale + p[a]) for a in amps])  # (k,) marginal std per mode
            # partial-correlation (C-vine) Cholesky factor L of the mode
            # correlation matrix R = L L^T. The parameters p[corr[(i, j)]] in
            # (-1, 1) are correlations (j == 0) / partial correlations (j > 0),
            # so R -- and hence M -- is PSD with unit diagonal for any values.
            L_rows = []
            for i in range(k):
                row = [0.0] * k
                acc = 0.0  # sum of squares of the entries already placed in this row
                for j in range(i):
                    Lij = p[corr[(i, j)]] * jnp_.sqrt(jnp_.clip(1.0 - acc, 1e-12, None))
                    row[j] = Lij
                    acc = acc + Lij ** 2
                row[i] = jnp_.sqrt(jnp_.clip(1.0 - acc, 1e-12, None))  # unit row norm
                L_rows.append(jnp_.stack([jnp_.asarray(v, dtype=jnp_.float64) for v in row]))
            L = jnp_.stack(L_rows)                                 # (k, k) lower-triangular
            R = L @ L.T
            M = (d[:, None] * d[None, :]) * R                      # (k, k) PSD covariance
            mblocks.append(jnp_.kron(jnp_.eye(nep), M))
        return matrix.jsp.linalg.block_diag(*mblocks)

    if all(par in noisedict for par in params):
        phi = np.asarray(build_phi(noisedict), dtype=np.float64)

        if variable:
            def getphi(params):
                return phi
            getphi.params = []

            gp = matrix.VariableGP(matrix.NoiseMatrix2D_var(getphi), Umatall)
            gp.index = {f'{psr.name}_{name}_coefficients({Umatall.shape[1]})': slice(0, Umatall.shape[1])}
            gp.name, gp.pos = psr.name, psr.pos
            gp.gpname, gp.gpcommon = name, []

            return gp
        else:
            return matrix.ConstantGP(matrix.NoiseMatrix2D_novar(phi), Umatall)
    else:
        def getphi(params):
            return build_phi(params)
        getphi.params = params

        gp = matrix.VariableGP(matrix.NoiseMatrix2D_var(getphi), Umatall)
        gp.index = {f'{psr.name}_{name}_coefficients({Umatall.shape[1]})': slice(0, Umatall.shape[1])}
        gp.name, gp.pos = psr.name, psr.pos
        gp.gpname, gp.gpcommon = name, []

        return gp

# timing model
def makegp_improper(psr, fmat, constant=1.0e40, name='improperGP', variable=False):
    if variable:
        phi = matrix.jnparray(constant * np.ones(fmat.shape[1]))

        def getphi(params):
            return phi
        getphi.params = []

        gp = matrix.VariableGP(matrix.NoiseMatrix1D_var(getphi), fmat)
        gp.index = {f'{psr.name}_{name}_coefficients({fmat.shape[1]})': slice(0, fmat.shape[1])}
    else:
        gp = matrix.ConstantGP(matrix.NoiseMatrix1D_novar(constant * np.ones(fmat.shape[1])), fmat)

    gp.name = psr.name
    gp.gpname = name

    return gp

def makegp_improper_varF(psr, fmat, constant=1.0e40, name='improperGP_varF',
                         param_names=[], noisedict={}, project=None,
                         project_tm=True, orthonormalize=True):
    """Improper GP with a parameter-dependent design matrix.

    Like :func:`makegp_improper`, but the design matrix comes from a callable basis
    whose columns depend on fit parameters -- for example :func:`chrom_poly_basis`,
    whose columns depend on the chromatic index. The varying parameter is named
    ``{psr.name}_{name}_{param}``, so it is shared with any other signal carrying the
    same name, such as a chromatic Fourier GP.

    ``project_tm`` and ``orthonormalize`` default on because a basis that overlaps the
    timing model is otherwise not marginalisable: the timing model already carries an
    improper prior over those directions, so the joint model is singular wherever the
    two spans coincide, and the marginal likelihood is arbitrary. Turn them off for a
    basis known to be orthogonal to the timing model and fixed in normalisation.

    psr:            Discovery Pulsar object
    fmat:           basis factory ``fmat(*param_values) -> (N_toa, N_col)`` array. May
                    carry an ``ncol`` attribute giving its column count; if absent the
                    width is found by evaluating it once
    constant:       diagonal of the flat improper prior over the coefficients
    name:           base name for the GP parameters
    param_names:    names of the parameters passed positionally to fmat
    noisedict:      fixed parameter values; if every entry of param_names is present
                    the basis is evaluated once and a ConstantGP returned, otherwise a
                    VariableGP whose design matrix varies with the free parameters
    project:        further bases to remove alongside the timing model, each an array
                    or a GP with a non-callable ``F``
    project_tm:     remove the timing-model column span from the basis
    orthonormalize: replace the basis by an orthonormal one spanning the same space, so
                    the marginal likelihood does not depend on how the basis was scaled
    """
    Q_null = None
    if project_tm:
        Q_null, _ = np.linalg.qr(normalise_tm_basis(psr))

    if project is not None:
        parts = project if isinstance(project, (list, tuple)) else [project]
        mats = []
        for p in parts:
            F_p = getattr(p, 'F', p)
            if callable(F_p):
                raise ValueError(
                    f'makegp_improper_varF: {psr.name}: a basis passed to project has a '
                    f'callable F, so it has no fixed column span to remove. Only bases '
                    f'with a constant design matrix can be projected out.')
            mats.append(np.asarray(F_p, dtype=np.float64))
        P = np.hstack(mats)
        if Q_null is not None:
            P = P - Q_null @ (Q_null.T @ P)
        Up, Sp, _ = np.linalg.svd(P, full_matrices=False)
        keep = Up[:, Sp > 1e-10 * Sp[0]]
        Q_null = keep if Q_null is None else np.hstack([Q_null, keep])

    Q_j = None if Q_null is None else matrix.jnparray(Q_null)

    def shape_np(F):
        if Q_null is not None:
            F = F - Q_null @ (Q_null.T @ F)
        return np.linalg.qr(F)[0] if orthonormalize else F

    def shape_jnp(F):
        if Q_j is not None:
            F = F - Q_j @ (Q_j.T @ F)
        return jnp.linalg.qr(F)[0] if orthonormalize else F

    ncol = getattr(fmat, 'ncol', None)
    if ncol is None:
        ncol = np.asarray(fmat(*[1.0 for _ in param_names])).shape[1]

    # noisedict keys are the full parameter names, as everywhere else in this module,
    # so that a dict taken straight from a single-pulsar chain fixes the basis
    argmap = [f'{psr.name}_{name}_{param}' for param in param_names]

    if all(arg in noisedict for arg in argmap):
        F_const = shape_np(np.asarray(fmat(*[noisedict[arg] for arg in argmap]),
                                      dtype=np.float64))
        gp = matrix.ConstantGP(matrix.NoiseMatrix1D_novar(constant * np.ones(ncol)),
                               F_const)
    else:
        phi = matrix.jnparray(constant * np.ones(ncol))

        def getphi(params):
            return phi
        getphi.params = []

        def get_fmat(params):
            return shape_jnp(fmat(*[params[arg] for arg in argmap]))
        get_fmat.params = argmap

        gp = matrix.VariableGP(matrix.NoiseMatrix1D_var(getphi), get_fmat)
        gp.index = {f'{psr.name}_{name}_coefficients({ncol})': slice(0, ncol)}

    gp.name, gp.pos, gp.gpname, gp.gpcommon = psr.name, psr.pos, name, []

    return gp

def normalise_tm_basis(psr, scale=1.0):
    """Timing-model design matrix with unit-norm columns.

    All-zero columns, which arise when a fitted par-file parameter has no TOAs
    behind it, are dropped and reported. Dividing by their zero norm would give
    NaNs, and they span nothing, so removing them leaves the column space
    unchanged.
    """
    Mmat = np.asarray(scale * psr.Mmat, dtype=np.float64)
    norms = np.sqrt(np.sum(Mmat**2, axis=0))
    keep = norms > 0

    ndrop = int((~keep).sum())
    if ndrop:
        idx = np.where(~keep)[0]
        names = (list(np.asarray(psr.fitpars)[idx]) if hasattr(psr, 'fitpars')
                 else list(idx))
        print(f'Warning: {psr.name} has {ndrop} all-zero timing-model column(s), '
              f'dropped: {names}')

    return Mmat[:, keep] / norms[keep]


def makegp_timing(psr, constant=None, variance=None, svd=False, scale=1.0, variable=False):
    if svd:
        fmat, _, _ = np.linalg.svd(scale * psr.Mmat, full_matrices=False)
    else:
        fmat = normalise_tm_basis(psr, scale=scale)

    if variance is None:
        if constant is None:
            constant = 1.0e40
        # else constant can stay what it is
    else:
        if constant is None:
            constant = variance * psr.Mmat.shape[0] / psr.Mmat.shape[1]
            return makegp_improper(psr, fmat, constant=constant, name='timingmodel', variable=variable)
        else:
            raise ValueError("signals.makegp_timing() can take a specification of _either_ `constant` or `variance`.")

    return makegp_improper(psr, fmat, constant=constant, name='timingmodel', variable=variable)

# Analytically-marginalised SVD chromatic polynomial GP.
def makegp_fd_piecewise(psr, nodes=16, spacing='quantile', selection=None, groups=None,
                        project_tm=True, constant=1.0e40, name='fd',
                        kind='linear', bin_flag=None):
    """Piecewise-linear frequency-dependent delay, constant in time, marginalised.

    Absorbs arbitrary time-constant structure across the observing band -- residual
    DM, a mean scattering delay, and profile evolution the template does not capture
    -- as a non-parametric generalisation of the timing model's ``FDx`` parameters.

    The basis is a set of hat (linear B-spline) functions, linear in ``log(freq)``,
    over ``nodes`` frequency nodes placed according to ``spacing``:

    - ``'quantile'`` (default): at quantiles of the observed frequency
      distribution, so each interval holds roughly ``ntoa/nodes`` TOAs and every
      node earns its place. Adapts to gapped coverage: on a band with a large
      receiver gap this delivers the full ``nodes`` where ``'log'`` would waste
      them (J0030, 439-1174 MHz empty: 14 surviving directions vs 7).
    - ``'log'``: uniform in ``log(freq)`` across the observed band, giving even
      resolution in the coordinate the ``FDx`` convention uses. Nodes landing in
      a gap produce empty columns which are dropped, so ``nodes`` is a requested
      rather than a delivered count.
    - ``'edge'``: at the boundaries of the ``bin_flag`` bins -- the geometric
      midpoint of the gap between consecutive bins, plus the two band ends -- so a
      segment spans one bin and the delay is linear within it with a free slope.
      ``nodes`` is ignored; the count follows from the flag, one more than the
      number of bins. The bins must be disjoint in frequency, which a per-receiver
      channel index reused across receivers is not. Adjacent segments share a node,
      so this does not span the
      ``kind='constant'`` indicator basis: holding one bin constant fixes a node
      its neighbour also uses, and independent per-bin constants would need two
      unshared nodes per bin.

    (Quantile placement is invariant under monotone transforms, so it gives the
    same nodes whether computed in frequency or log-frequency.)

    ``selection`` optionally gives TOA groups their own basis, for cases where
    profile evolution differs between receivers whose bands overlap -- which a
    single frequency-only basis cannot separate. It follows the usual convention:
    a callable mapping ``psr`` to an array of per-TOA labels (e.g.
    :func:`selection_backend_flags`), as the white-noise parameters use.

    - ``selection=None`` (default): one global basis over all TOAs.
    - ``selection=callable``: one basis per group, as the white-noise parameters
      are split per backend. No separate global term -- the per-group bases
      already span any structure common to all of them.
    - ``selection=callable, groups=[...]``: only the listed groups get their own
      basis, as band noise is switched on for selected pulsars, PLUS the global
      basis, which covers the remaining groups and any common structure.
    - ``selection=[callable, ...]``: a user-defined set of selections; every
      label of every callable gets a block, and it is the caller's job to
      include a global one if wanted (e.g. a selection labelling all TOAs alike).

    All blocks are combined into the ONE marginalised GP, so a selection changes
    the number of basis columns, not the number of GPs. Groups too sparse in
    frequency to support a basis are skipped with a warning, and any redundancy
    between blocks is removed by the rank-revealing step.

    The amplitudes are marginalised analytically with an improper (very broad)
    prior, exactly like the timing model, so this removes the corresponding
    directions from the data rather than measuring them.

    ``project_tm=True`` (default) projects the timing-model column subspace out of
    the basis. This is REQUIRED, not merely tidy: a flat-in-frequency column is
    exactly the timing-model phase offset, a constant-in-time nu^-2 term is exactly
    DM, and any ``FDx`` in the par file are literally columns of ``psr.Mmat`` -- so
    without the projection the joint model is singular. Directions annihilated by
    the projection are dropped, so the surviving basis is full rank.

    ``kind='constant'`` replaces the hat functions with indicator (boxcar) columns,
    one per frequency bin with disjoint support. A hat basis interpolates between its
    nodes and so cannot represent a step; an indicator basis represents a per-channel
    offset exactly, in one column, and ``F^T F`` is diagonal at any bin count. For
    hats a node is a centre; for indicators it is a bin, so the placement differs:

    - ``spacing='flag'`` with ``bin_flag='chan'``: one column per distinct value of a
      named per-TOA flag, using the instrument's own channelisation. Exact, and the
      only mode that needs no estimate of where the edges are.
    - ``spacing='gap'``: edges at the midpoints of the ``nodes - 1`` largest gaps in
      the sorted unique frequencies, recovering the channelisation from the
      frequencies alone where no flag exists. A fallback, not exact: the largest
      within-channel gap can come within a factor of 1.3 of the smallest
      between-channel gap.
    - ``spacing='log'`` or ``'quantile'``: allowed, but these place edges without
      regard to the channel structure and split channels across two columns, which is
      reported. Balanced TOA counts do not reveal it, since quantile placement
      balances them by construction.

    Indicator columns are not normalised, so a coefficient reads directly as a delay
    in seconds. They sum to the all-ones vector, which is exactly the timing model's
    phase offset, so ``project_tm`` is required here too.

    Note this term is degenerate with the constant-in-time part of
    :func:`makegp_chrom_poly_svd`; pass this GP to that function's ``project``
    argument to remove the overlap.
    """
    x = np.log(np.asarray(psr.freqs, dtype=np.float64))
    blocks = _fd_piecewise_selection_blocks(psr, len(x), selection, groups)

    mats, group_nodes = [], {}
    for label, sel in blocks:
        block = _fd_piecewise_block(psr, x, sel, nodes, spacing,
                                    name if label is None else f'{name}_{label}',
                                    kind=kind, bin_flag=bin_flag)
        if block is not None:
            fmat_g, q = block
            mats.append(fmat_g)
            group_nodes[label] = np.exp(q)

    if not mats:
        raise ValueError(f"makegp_fd_piecewise: no usable frequency basis for {psr.name}.")

    # Stack the per-group blocks (disjoint TOA support, hence mutually orthogonal)
    # and project the timing model out of the COMBINED basis exactly once, which
    # keeps the blocks disjoint.
    fmat = np.hstack(mats)

    if project_tm:
        Q_tm, _ = np.linalg.qr(normalise_tm_basis(psr))
        fmat = fmat - Q_tm @ (Q_tm.T @ fmat)

    # Rank-revealing orthonormalisation: the projection above can annihilate
    # directions (e.g. the flat and nu^-2 ones), which are dropped at a relative
    # singular-value cut of 1e-8.
    U, S, _ = np.linalg.svd(fmat, full_matrices=False)
    fmat = U[:, S > 1e-8 * S[0]]
    if fmat.shape[1] == 0:
        raise ValueError(f"makegp_fd_piecewise: basis for {psr.name} is entirely "
                         f"degenerate with the timing model.")

    gp = makegp_improper(psr, fmat, constant=constant, name=name)
    gp.fd_nodes = group_nodes[None] if selection is None else group_nodes
    return gp

def _fd_piecewise_selection_blocks(psr, ntoa, selection, groups):
    """Split the TOAs into ``(label, mask)`` blocks for a piecewise-frequency basis.

    psr:       pulsar, passed to the selection callables
    ntoa:      number of TOAs, for the global mask
    selection: callable, or list of callables, returning a per-TOA label
    groups:    labels that get their own block alongside a global one
    """
    everything = np.ones(ntoa, dtype=bool)

    if selection is None:
        return [(None, everything)]

    if isinstance(selection, (list, tuple)):
        # user-defined set of selections: every label of every callable gets a
        # block, and it is the caller's job to include a global one if wanted
        # (e.g. a selection returning the same label for all TOAs).
        blocks = []
        for sel_fn in selection:
            flags = np.asarray(sel_fn(psr))
            blocks += [(str(g), flags == g) for g in sorted(set(flags.tolist()))]
        return blocks

    flags = np.asarray(selection(psr))
    present = sorted(set(flags.tolist()))

    if groups is None:
        # every group gets its own basis: a global term would add nothing
        # conceptually, since per-group terms already span any common
        # structure (raise `nodes` for more resolution instead).
        return [(str(g), flags == g) for g in present]

    # only some groups get their own basis, so the rest -- and any
    # structure common to all -- still need the global term.
    for g in groups:
        if g not in present:
            print(f"Warning: fd_piecewise group {g!r} not found among {psr.name}'s "
                  f"selection labels; skipped.")

    return [(None, everything)] + [(str(g), flags == g) for g in groups if g in present]


_SQRT3 = float(np.sqrt(3.0))


def makegp_fd_piecewise_matern(psr, nodes=16, spacing='quantile', selection=None, groups=None,
                               project_tm=True, jitter=1e-8, name='fd_gp',
                               kind='linear', bin_flag=None, normalise=False, project_nodes=False):
    """Piecewise-linear frequency-dependent delay under a Matern-3/2 prior.

    Same hat-function basis as :func:`makegp_fd_piecewise`, but the node
    amplitudes carry a proper Gaussian prior rather than being marginalised with
    an improper one, so the data set how much band structure is absorbed instead
    of those directions being removed unconditionally.

    The prior covariance is a Matern-3/2 kernel in node log-frequency
    ``q = log(nu)``::

        Phi_ij = sigma**2 (1 + sqrt(3)|q_i - q_j|/ell) exp(-sqrt(3)|q_i - q_j|/ell)

    evaluated at the node positions, so the basis must keep its
    amplitude-to-node correspondence: unlike :func:`makegp_fd_piecewise` this
    function does not orthonormalise. Rank deficiency left by the timing-model
    projection is harmless, since the Woodbury form inverts Phi rather than F.
    Amplitudes stay analytically marginalised.

    Samples ``{psr}_{name}_log10_sigma`` (delay units) and
    ``{psr}_{name}_log10_ell``, whose priors the model sets (``mpta._set_fd_priors``
    bounds log10_ell by the node span). With ``selection`` active the blocks are
    disjoint and Phi is block diagonal, with the hyperparameters shared across blocks.

    nodes:      frequency nodes per block
    spacing:    'quantile' (equal TOA counts), 'log' (equal in log-frequency), or
                'edge' (at the bin_flag bin boundaries, one segment per bin)
    selection:  callable, or list of callables, splitting the TOAs into blocks
    groups:     labels that get their own block alongside a global one
    project_tm: project the timing-model column subspace out of the basis
    normalise:  scale Phi so log10_sigma is the rms delay per TOA that the prior
                admits after the timing-model projection, rather than the node
                amplitude. Without it sigma absorbs the ell dependence of that
                projection and the two are degenerate
    jitter:     relative diagonal added to Phi, capping its condition number near
                nodes/jitter
    name:       parameter-name stem
    kind:       'linear' for hat functions, 'constant' for indicator columns; see
                :func:`makegp_fd_piecewise` for the placement modes each admits. Under
                'constant' the kernel positions are the TOA-weighted mean log(freq)
                within each bin, so the kernel distance follows the data rather than
                the bin edges.
    bin_flag:   per-TOA flag whose distinct values define the bins, for
                kind='constant' with spacing='flag'
    project_nodes: also remove from the prior the node-space directions whose delay
                patterns the timing model spans, as :func:`makegp_fd_piecewise_powerlaw`
                does, so the prior is ``B^T Phi B`` on the remaining node space and
                ``normalise`` refers to it; ``jitter`` is then relative to the mean
                diagonal of the projected prior
    """
    x = np.log(np.asarray(psr.freqs, dtype=np.float64))
    blocks = _fd_piecewise_selection_blocks(psr, len(x), selection, groups)

    mats, qs, group_nodes = [], [], {}
    for label, sel in blocks:
        block = _fd_piecewise_block(psr, x, sel, nodes, spacing,
                                    name if label is None else f'{name}_{label}',
                                    kind=kind, bin_flag=bin_flag)
        if block is not None:
            fmat_g, q = block
            mats.append(fmat_g)
            qs.append(q)
            group_nodes[label] = np.exp(q)

    if not mats:
        raise ValueError(f"makegp_fd_piecewise_matern: no usable frequency basis for {psr.name}.")

    fmat = np.hstack(mats)
    fmat_raw = fmat

    if project_tm:
        fdpars = [p for p in getattr(psr, 'fitpars', []) if re.fullmatch(r'FD\d+', str(p).upper())]
        if fdpars:
            warnings.warn(
                f"{psr.name}: the timing model still fits {fdpars}, whose columns carry the "
                f"low-order smooth part of the band structure. Projecting them out leaves that "
                f"part improperly marginalised through the timing model, so the Matern prior "
                f"only shapes what is left. Remove FD from the par file to measure it.")

        Mmat = np.asarray(psr.Mmat, dtype=np.float64)
        M_norm = Mmat / np.sqrt(np.sum(Mmat**2, axis=0))
        Q_tm, _ = np.linalg.qr(M_norm)
        fmat = fmat - Q_tm @ (Q_tm.T @ fmat)

    signame, ellname = f'{psr.name}_{name}_log10_sigma', f'{psr.name}_{name}_log10_ell'
    sizes = [len(q) for q in qs]

    offsets = np.cumsum([0] + sizes)
    total = int(offsets[-1])

    dists = [matrix.jnparray(np.abs(q[:, None] - q[None, :])) for q in qs]
    eyes = [matrix.jnparray(np.eye(n)) for n in sizes]

    def _corr(dist, eye, params):
        r = _SQRT3 * dist / (10.0 ** params[ellname])
        return (1.0 + r) * jnp.exp(-r) + jitter * eye

    if normalise:
        # trace(F K F^T) = sum(G * K), with G the Gram of the basis the GP ships,
        # so the per-TOA mean square costs one nodes x nodes product per call
        grams = [matrix.jnparray(fmat[:, i0:i0+n].T @ fmat[:, i0:i0+n])
                 for i0, n in zip(offsets[:-1], sizes)]
        ntoa = fmat.shape[0]

        def _amp(params):
            g = sum(jnp.sum(G * _corr(dist, eye, params))
                    for G, dist, eye in zip(grams, dists, eyes)) / ntoa
            return 10.0 ** (2.0 * params[signame]) / g
    else:
        def _amp(params):
            return 10.0 ** (2.0 * params[signame])

    def _makeblock(dist, eye):
        def getblock(params):
            return _amp(params) * _corr(dist, eye, params)
        getblock.params = [signame, ellname]

        return getblock

    blockfuncs = [_makeblock(dist, eye) for dist, eye in zip(dists, eyes)]

    def getphi(params):
        phi = jnp.zeros((total, total))
        for getblock, i0, n in zip(blockfuncs, offsets[:-1], sizes):
            phi = phi.at[i0:i0+n, i0:i0+n].set(getblock(params))
        return phi
    getphi.params = [signame, ellname]

    if len(blockfuncs) > 1:
        Phi = matrix.BlockDiagNoiseMatrix2D_var(getphi, [(False, f) for f in blockfuncs])
    else:
        Phi = matrix.NoiseMatrix2D_var(getphi)

    if project_nodes:
        B, removed = _fd_node_projection(psr, fmat_raw, fmat, np.concatenate(qs),
                                         'makegp_fd_piecewise_matern')
        project_basis = fmat
        fmat = fmat @ B
        Bj = matrix.jnparray(B)
        gram_b = matrix.jnparray(fmat.T @ fmat)
        ntoa_b = fmat.shape[0]

        eye_b = matrix.jnparray(np.eye(B.shape[1]))

        def getphi_nodes(params):
            Ks = Bj.T @ jax.scipy.linalg.block_diag(
                *[_corr(dist, eye, params) - jitter * eye for dist, eye in zip(dists, eyes)]) @ Bj
            Ks = Ks + jitter * jnp.mean(jnp.diag(Ks)) * eye_b
            g = jnp.sum(gram_b * Ks) / ntoa_b if normalise else 1.0
            return 10.0 ** (2.0 * params[signame]) / g * Ks
        getphi_nodes.params = [signame, ellname]

        Phi = matrix.NoiseMatrix2D_var(getphi_nodes)

    gp = matrix.VariableGP(Phi, fmat)
    gp.index = {f'{psr.name}_{name}_coefficients({fmat.shape[1]})': slice(0, fmat.shape[1])}
    gp.name = psr.name
    gp.gpname = name
    gp.fd_nodes = group_nodes[None] if selection is None else group_nodes
    if project_nodes:
        gp.node_basis, gp.node_removed, gp.project_basis = B, removed, project_basis

    return gp


def _fd_frequency_only_patterns(psr, fmat_raw, q, tol=1e-9, degree=12):
    """Node patterns of the timing-model columns that depend on observing frequency alone.

    A column of ``psr.Mmat`` counts when its per-TOA values are reproduced to relative
    residual ``tol`` either by a function of frequency in the span of Legendre
    polynomials in ``log(freq)`` up to ``degree`` and ``nu**-2`` -- the phase offset,
    every ``FDk`` and a constant ``DM`` -- or by the basis itself, like a jump between
    groups the blocks separate. The first kind is evaluated at the node positions ``q``,
    the second is its least-squares node representation. Columns that vary in time
    reproduce neither and are left to the TOA-space projection.

    Returns ``(patterns, labels)``: one node pattern per counted column, and its name.
    """
    M = np.asarray(psr.Mmat, dtype=np.float64)
    x = np.log(np.asarray(psr.freqs, dtype=np.float64))
    lo, hi = float(x.min()), float(x.max())

    def functions(y):
        u = (2.0 * y - lo - hi) / (hi - lo)
        return np.column_stack([np.polynomial.legendre.legvander(u, degree), np.exp(-2.0 * (y - lo))])

    at_toas, at_nodes = functions(x), functions(np.asarray(q, dtype=np.float64))
    names = [str(n) for n in getattr(psr, 'fitpars', [])]
    if len(names) != M.shape[1]:
        names = [f'Mmat column {j}' for j in range(M.shape[1])]

    patterns, labels = [], []
    for j in range(M.shape[1]):
        m = M[:, j]
        scale = np.linalg.norm(m)
        if scale == 0.0:
            continue
        coef, *_ = np.linalg.lstsq(at_toas, m, rcond=None)
        if np.linalg.norm(at_toas @ coef - m) < tol * scale:
            patterns.append(at_nodes @ coef)
            labels.append(names[j])
            continue
        g, *_ = np.linalg.lstsq(fmat_raw, m, rcond=None)
        if np.linalg.norm(fmat_raw @ g - m) < tol * scale:
            patterns.append(g)
            labels.append(names[j])

    return patterns, labels


def _fd_node_projection(psr, fmat_raw, fmat, q, what, tol=1e-10):
    """Orthonormal basis of the node space left once the timing model's frequency-only
    directions are removed, and a one-line report of what it retains.

    The removed directions are the node patterns of :func:`_fd_frequency_only_patterns`,
    orthonormalised with singular values below ``tol`` of the largest dropped, so a prior
    ``B^T Phi B`` on the returned ``B`` puts exactly no power in them.

    fmat_raw: basis before any projection, used to recognise the columns
    fmat:     basis the GP ships, used to report the power retained
    q:        node positions in log(freq), concatenated over blocks
    """
    patterns, labels = _fd_frequency_only_patterns(psr, fmat_raw, q)
    n = fmat.shape[1]
    if not patterns:
        print(f'{what}: {psr.name}: no timing-model column depends on frequency alone; '
              f'node space left whole ({n} directions).')
        return np.eye(n), labels

    U, S, _ = np.linalg.svd(np.column_stack(patterns), full_matrices=True)
    r = int(np.sum(S > tol * S[0]))
    B = U[:, r:]
    kept = np.linalg.norm(fmat @ B) ** 2 / np.linalg.norm(fmat) ** 2
    print(f'{what}: {psr.name}: node-space projection removes {r} direction(s) spanned by '
          f'{len(labels)} frequency-only timing-model column(s) ({", ".join(labels)}); '
          f'retains rank {n - r} of {n} and {kept:.4f} of the basis power.')
    if n - r < 0.5 * n:
        warnings.warn(f'{what}: {psr.name}: the node-space projection leaves {n - r} of {n} node '
                      f'directions, since the timing model fits {r} frequency-only terms; the prior '
                      f'then resolves little. Use more nodes.')
    return B, labels


def _fd_powerlaw_correlation(q, f_lo, f_hi, nf):
    """Band-limited power law in log-frequency as a correlation matrix over nodes ``q``.

    ``S(f) = f**-gamma`` over ``nf`` log-spaced bins of ``[f_lo, f_hi]``, each bin's weight
    ``w_j`` its Simpson integral and the bin taken flat across its width ``df_j``, so

        k(dq) = sum_j w_j cos(2 pi f_j dq) sinc(df_j dq),

    normalised to ``k(0) = 1``, is the exact covariance of that step spectrum. Returns
    ``corr(gamma)``.
    """
    edges = np.exp(np.linspace(np.log(f_lo), np.log(f_hi), int(nf) + 1))
    a, b = matrix.jnparray(edges[:-1] / f_hi), matrix.jnparray(edges[1:] / f_hi)
    m = 0.5 * (a + b)
    dq = np.asarray(q)[None, :, None] - np.asarray(q)[None, None, :]
    cosines = matrix.jnparray(np.cos(2.0 * np.pi * f_hi * np.asarray(m)[:, None, None] * dq)
                              * np.sinc(f_hi * (np.asarray(b) - np.asarray(a))[:, None, None] * dq))

    def corr(gamma):
        w = (b - a) / 6.0 * (a ** -gamma + 4.0 * m ** -gamma + b ** -gamma)
        return jnp.tensordot(w / jnp.sum(w), cosines, axes=1)

    return corr


def makegp_fd_piecewise_powerlaw(psr, nodes=16, spacing='quantile', selection=None, groups=None,
                                 project_tm=True, project_nodes=True, independent=False,
                                 powerlaw=True, flo_pad=2.0, nf=128, jitter=1e-8, name='fd_gp',
                                 kind='linear', bin_flag=None, normalise=True):
    """Piecewise frequency-dependent delay under a power-law prior in log-frequency.

    The basis, placements and selections of :func:`makegp_fd_piecewise_matern`, with the
    Matern kernel replaced by a band-limited power law in ``q = log(nu)``: over "frequencies"
    ``f`` in cycles per unit ``log(nu)``,

        S(f) = f**-gamma on [f_lo, f_hi],   k(dq) = sum_j w_j cos(2 pi f_j dq) sinc(df_j dq),

    with ``w_j`` the integral of ``S`` over log-spaced bin ``j``, normalised to ``k(0) = 1``
    and evaluated at the node positions; it is the exact covariance of that step spectrum. ``f_hi`` is the Nyquist
    frequency of the node grid, ``1/(2 x median node spacing)``, and ``f_lo`` the outer
    scale, ``1/(flo_pad x node span)``, per block. The kernel is positive semi-definite and
    differentiable in ``gamma``, and there is no length scale.

    ``gamma`` sets the smoothness. ``gamma = 0`` is a flat spectrum up to the node Nyquist,
    close to offsets independent from node to node, and only approximately so where the node
    spacing departs from its median. A Matern kernel of order ``nu`` has a high-frequency
    tail ``f**-(2 nu + 1)``, so ``gamma = 2, 4, 6`` correspond to ``nu = 1/2, 3/2, 5/2``.

    ``independent`` adds offsets drawn independently per node, ``N(0, sigma_ind**2)``,
    constant in time, so the prior is ``sigma**2 K_gamma + sigma_ind**2 I`` in node space;
    at ``gamma = 0`` the two terms are nearly degenerate.

    ``project_nodes`` removes from the prior the node-space directions whose delay patterns
    the timing model spans (:func:`_fd_node_projection`): the node amplitudes are
    ``c = B a`` with ``B`` an orthonormal basis of what is left, so the prior is
    ``B^T Phi B`` and puts exactly no power along the phase offset, the ``FDk`` polynomials
    or a constant ``DM`` evaluated at the nodes. The TOA-space projection (``project_tm``)
    still removes the rest of the timing model from the basis. The removed directions are
    polynomials at the nodes, whose delays a basis reproduces only up to its resolution --
    within-bin frequency spread for indicators, interpolation between hat nodes -- so they
    carry that residue with them, and each fitted ``FDk`` costs one node direction.

    ``normalise`` scales each term so its ``log10_sigma`` is the rms delay per TOA that the
    prior admits after both projections, so ``sigma`` and ``gamma`` are not degenerate
    through them.

    ``gp.project_basis`` is the basis before the node-space reduction, the whole
    frequency-dependent span, for other components to project out whatever the prior on it.

    Samples ``{psr}_{name}_log10_sigma`` and ``{psr}_{name}_gamma``, and with
    ``independent`` ``{psr}_{name}_log10_sigma_ind``, whose priors the model sets; with
    ``selection`` active the blocks share them.

    nodes, spacing, selection, groups, kind, bin_flag: as :func:`makegp_fd_piecewise_matern`
    project_tm:    project the timing-model column subspace out of the basis
    project_nodes: remove the frequency-only timing-model directions from the prior
    independent:   add the independent per-node term
    powerlaw:      include the power-law term; False with ``independent`` leaves the
                   independent term alone, sampling only ``log10_sigma_ind``
    flo_pad:       outer scale as a multiple of the node span
    nf:            log-spaced bins in the power-law sum
    jitter:        diagonal added to the prior after the node projection, relative to its
                   mean diagonal, so the floor scales with the power the projection leaves
                   rather than with the raw kernel
    normalise:     scale so log10_sigma is the rms delay per TOA after projection
    name:          parameter-name stem
    """
    if not (powerlaw or independent):
        raise ValueError("makegp_fd_piecewise_powerlaw: powerlaw=False needs independent=True, "
                         "or the prior has no term.")

    x = np.log(np.asarray(psr.freqs, dtype=np.float64))
    blocks = _fd_piecewise_selection_blocks(psr, len(x), selection, groups)

    mats, qs, group_nodes = [], [], {}
    for label, sel in blocks:
        block = _fd_piecewise_block(psr, x, sel, nodes, spacing,
                                    name if label is None else f'{name}_{label}',
                                    kind=kind, bin_flag=bin_flag)
        if block is not None:
            fmat_g, q = block
            mats.append(fmat_g)
            qs.append(q)
            group_nodes[label] = np.exp(q)

    if not mats:
        raise ValueError(f"makegp_fd_piecewise_powerlaw: no usable frequency basis for {psr.name}.")

    fmat_raw = np.hstack(mats)
    fmat = fmat_raw
    if project_tm:
        Mmat = np.asarray(psr.Mmat, dtype=np.float64)
        Q_tm, _ = np.linalg.qr(Mmat / np.sqrt(np.sum(Mmat**2, axis=0)))
        fmat = fmat_raw - Q_tm @ (Q_tm.T @ fmat_raw)

    q_all = np.concatenate(qs)
    if project_nodes:
        B, removed = _fd_node_projection(psr, fmat_raw, fmat, q_all, 'makegp_fd_piecewise_powerlaw')
    else:
        B, removed = np.eye(len(q_all)), []
    fmat_out = fmat @ B
    if fmat_out.shape[1] == 0:
        raise ValueError(f"makegp_fd_piecewise_powerlaw: {psr.name}: the timing model spans "
                         f"every node direction.")

    bands, corrs = [], []
    for q in (qs if powerlaw else []):
        dq = np.diff(np.sort(q))
        f_hi = 1.0 / (2.0 * float(np.median(dq[dq > 0])))
        f_lo = 1.0 / (float(flo_pad) * float(q.max() - q.min()))
        if not f_hi > f_lo:
            raise ValueError(f"makegp_fd_piecewise_powerlaw: {psr.name}: the node Nyquist "
                             f"1/{1 / f_hi:.3g} is not above the outer scale 1/{1 / f_lo:.3g} "
                             f"(per unit log freq); lower flo_pad or use more nodes.")
        bands.append((f_lo, f_hi))
        corrs.append(_fd_powerlaw_correlation(q, f_lo, f_hi, nf))

    sizes = [len(q) for q in qs]
    offsets = np.cumsum([0] + sizes)
    Bj = matrix.jnparray(B)
    eye_out = matrix.jnparray(np.eye(B.shape[1]))

    signame, gamname = f'{psr.name}_{name}_log10_sigma', f'{psr.name}_{name}_gamma'
    indname = f'{psr.name}_{name}_log10_sigma_ind'
    pnames = ([signame, gamname] if powerlaw else []) + ([indname] if independent else [])

    def shape(params):
        Ks = Bj.T @ jax.scipy.linalg.block_diag(*[c(params[gamname]) for c in corrs]) @ Bj
        return Ks + jitter * jnp.mean(jnp.diag(Ks)) * eye_out

    ntoa = fmat_out.shape[0]
    gram = matrix.jnparray(fmat_out.T @ fmat_out)
    g_ind = float(np.trace(fmat_out.T @ fmat_out)) / ntoa if normalise else 1.0

    def getphi(params):
        phi = 0.0
        if powerlaw:
            Ks = shape(params)
            g = jnp.sum(gram * Ks) / ntoa if normalise else 1.0
            phi = 10.0 ** (2.0 * params[signame]) / g * Ks
        if independent:
            phi = phi + 10.0 ** (2.0 * params[indname]) / g_ind * eye_out
        return phi
    getphi.params = pnames

    Phi = matrix.NoiseMatrix2D_var(getphi)

    gp = matrix.VariableGP(Phi, fmat_out)
    gp.index = {f'{psr.name}_{name}_coefficients({fmat_out.shape[1]})': slice(0, fmat_out.shape[1])}
    gp.name = psr.name
    gp.gpname = name
    gp.fd_nodes = group_nodes[None] if selection is None else group_nodes
    gp.node_basis, gp.node_removed, gp.fd_bands = B, removed, bands
    gp.project_basis = fmat

    return gp


def _fd_piecewise_block(psr, x, sel, nodes, spacing, name, kind='linear', bin_flag=None):
    """Basis block for one TOA selection, zero outside it.

    Returns ``(fmat, q)`` with ``q`` one position in ``log(freq)`` per column, or None
    (with a warning) if the selection cannot support a basis. Blocks for different
    selections have disjoint support and so are mutually orthogonal.

    kind='linear' builds hat functions, where a node is a centre and the basis
    interpolates between centres. kind='constant' builds indicator columns, where a
    node is a bin and the columns are disjoint, so a per-bin step is represented
    exactly in one column and ``F^T F`` is diagonal.

    Under kind='linear' with spacing='edge' the nodes are the bin boundaries rather
    than bin centres, so each hat straddles two bins and each segment covers one.
    """
    xs = x[sel]

    if len(np.unique(xs)) < 2:
        print(f"Warning: fd_piecewise selection {name!r} for {psr.name} spans "
              f"{len(np.unique(xs))} distinct frequencies over {int(sel.sum())} TOAs; skipped.")
        return None

    if kind == 'constant':
        return _fd_constant_block(psr, x, sel, nodes, spacing, name, bin_flag)
    if kind != 'linear':
        raise ValueError(f"makegp_fd_piecewise: kind must be 'linear' or 'constant', got {kind!r}.")

    if spacing == 'log':
        q = np.linspace(xs.min(), xs.max(), nodes)
    elif spacing == 'quantile':
        q = np.quantile(xs, np.linspace(0.0, 1.0, nodes))
    elif spacing == 'edge':
        pairs = [(lab, m) for lab, m in
                 _fd_flag_masks(psr, sel, name, bin_flag, spacing='edge') if m.any()]
        if len(pairs) < 2:
            print(f"Warning: fd_piecewise selection {name!r} for {psr.name} has "
                  f"{len(pairs)} non-empty {bin_flag!r} bin(s); skipped.")
            return None

        stats = np.array(sorted((x[m].mean(), x[m].min(), x[m].max()) for _, m in pairs))
        centres, los, his = stats[:, 0], stats[:, 1], stats[:, 2]
        sep = np.diff(centres)
        # a boundary is only a boundary if consecutive bins do not share frequencies
        over = np.divide(his[:-1] - los[1:], sep, out=np.full(len(sep), np.inf), where=sep > 0)
        if np.any(over > 1.0):
            raise ValueError(
                f"makegp_fd_piecewise: spacing='edge' needs the {bin_flag!r} bins to be "
                f"disjoint in frequency, but {int((over > 1.0).sum())} of {len(sep)} "
                f"neighbouring pairs overlap by more than their separation, so their "
                f"midpoints are not boundaries. A per-receiver channel index reused across "
                f"receivers does this; combine it with the receiver flag, or pick a flag "
                f"whose values partition the band.")
        if np.any(over > 0.0):
            print(f"Warning: fd_piecewise {name!r} for {psr.name}: "
                  f"{int((over > 0.0).sum())} neighbouring {bin_flag!r} bin pair(s) overlap "
                  f"in frequency; the node between them sits inside both.")

        # the boundary is the middle of the empty gap between consecutive bins, which
        # keeps each segment on one bin when the bins differ in width; where they touch
        # there is no gap and the midpoint of their means is the only definition left
        interior = np.where(his[:-1] < los[1:],
                            0.5 * (his[:-1] + los[1:]),
                            0.5 * (centres[:-1] + centres[1:]))
        q = np.concatenate(([xs.min()], interior, [xs.max()]))
    else:
        raise ValueError(f"makegp_fd_piecewise: for kind='linear', spacing must be "
                         f"'log', 'quantile' or 'edge', got {spacing!r}.")

    q = np.unique(q)
    if len(q) < 2:
        print(f"Warning: fd_piecewise selection {name!r} for {psr.name} has too little "
              f"frequency coverage for a piecewise basis; skipped.")
        return None

    # hat functions: unity at their own node, falling linearly to zero at the neighbours
    fmat = np.zeros((len(x), len(q)), dtype=np.float64)
    for i, c in enumerate(q):
        lo = q[i-1] if i > 0 else c - (q[1] - q[0])
        hi = q[i+1] if i < len(q) - 1 else c + (q[-1] - q[-2])
        left = sel & (x >= lo) & (x <= c)
        right = sel & (x > c) & (x <= hi)
        fmat[left, i] = (x[left] - lo) / (c - lo)
        fmat[right, i] = (hi - x[right]) / (hi - c)

    return fmat, q


def _fd_flag_masks(psr, sel, name, bin_flag, what='fd_piecewise', spacing='flag'):
    """Per-TOA masks for each distinct value of a named flag, in numeric order if possible."""
    if bin_flag is None:
        raise ValueError(f"{what}: spacing={spacing!r} needs bin_flag, the name of "
                         f"the per-TOA flag whose distinct values define the bins.")
    if bin_flag not in psr.flags:
        raise KeyError(f"{what}: {psr.name} has no flag {bin_flag!r}; available "
                       f"flags are {sorted(psr.flags)}. A silently skipped basis is worse "
                       f"than a failure here.")

    labels = np.asarray(psr.flags[bin_flag]).astype(str)
    present = set(labels[sel].tolist())
    if '' in present:
        print(f"Warning: {what} {name!r} for {psr.name}: {int((labels[sel] == '').sum())} "
              f"TOAs carry an empty {bin_flag!r} flag and get no column.")
        present.discard('')

    try:
        order = sorted(present, key=int)
    except ValueError:
        # not every label is an integer, so plain string order is the only option
        order = sorted(present)

    return [(v, sel & (labels == v)) for v in order]


def _fd_gap_edges(freqs, nbins):
    """Interior bin edges at the midpoints of the ``nbins - 1`` largest frequency gaps.

    Recovers an instrument's channelisation from the frequencies alone, for data whose
    per-TOA frequencies are channel centroids rather than exact channel centres. A
    threshold on the gap size is not used: the largest within-channel gap can come
    within a factor of 1.6 of the smallest between-channel gap, so any fixed multiple
    of the median gap needs tuning per pulsar.
    """
    uf = np.unique(freqs)
    if nbins < 2 or len(uf) < 2:
        return np.array([])
    gaps = np.diff(uf)
    take = min(int(nbins) - 1, len(gaps))
    idx = np.argsort(gaps)[-take:]

    return np.sort(0.5 * (uf[:-1][idx] + uf[1:][idx]))


def _fd_constant_block(psr, x, sel, nodes, spacing, name, bin_flag, what='fd_piecewise'):
    """Indicator (boxcar) block for one TOA selection, one column per frequency bin.

    Columns are 1 inside their bin and 0 outside, and are not normalised, so a fitted
    coefficient reads directly as a delay in seconds. ``q`` is the TOA-weighted mean of
    ``log(freq)`` within each bin, so the Matern kernel distance reflects where the
    data are rather than where the bin edges are.
    """
    if spacing == 'flag':
        pairs = _fd_flag_masks(psr, sel, name, bin_flag, what)
    else:
        xs = x[sel]
        if spacing == 'gap':
            edges = np.log(_fd_gap_edges(np.exp(xs), nodes))
        elif spacing == 'log':
            edges = np.linspace(xs.min(), xs.max(), int(nodes) + 1)[1:-1]
        elif spacing == 'quantile':
            edges = np.quantile(xs, np.linspace(0.0, 1.0, int(nodes) + 1)[1:-1])
        else:
            raise ValueError(f"{what}: for kind='constant', spacing must be "
                             f"'flag', 'gap', 'log' or 'quantile', got {spacing!r}.")
        which = np.searchsorted(np.asarray(edges, dtype=np.float64), x)
        pairs = [(str(b), sel & (which == b)) for b in range(len(edges) + 1)]

    counts = np.array([int(m.sum()) for _, m in pairs])
    empty = [lab for (lab, _), n in zip(pairs, counts) if n == 0]
    if empty:
        print(f"Warning: {what} {name!r} for {psr.name}: {len(empty)} empty "
              f"{spacing} bin(s) {empty} dropped.")
    pairs = [(lab, m) for (lab, m), n in zip(pairs, counts) if n > 0]

    if len(pairs) < 2:
        print(f"Warning: {what} selection {name!r} for {psr.name} yields "
              f"{len(pairs)} non-empty bin(s); skipped.")
        return None

    if spacing in ('log', 'quantile'):
        # These place edges without regard to the channel structure. Balanced TOA counts
        # do not show it -- quantile placement balances them by construction while still
        # cutting channels in half -- so count the edges that land inside a populated
        # block rather than in a gap between blocks.
        uf = np.unique(np.exp(x[sel]))
        gaps = np.diff(uf)
        nb = len(pairs)
        if len(gaps) >= nb > 1:
            boundary = float(np.sort(gaps)[-(nb - 1)])
            j = np.clip(np.searchsorted(uf, np.exp(edges)), 1, len(uf) - 1)
            split = int(np.sum((uf[j] - uf[j - 1]) < 0.5 * boundary))
            if split:
                print(f"Warning: {what} {name!r} for {psr.name}: spacing={spacing!r} "
                      f"with kind='constant' puts {split} of {len(edges)} edges inside a "
                      f"populated frequency block, splitting it across two columns; "
                      f"'flag' or 'gap' place edges between blocks.")
        live = np.array([int(m.sum()) for _, m in pairs])
        thin = int(np.sum(live < 0.5 * np.median(live)))
        if thin:
            print(f"Warning: {what} {name!r} for {psr.name}: {thin} of {len(live)} "
                  f"bins hold below half the median TOA count ({int(np.median(live))}).")

    fmat = np.zeros((len(x), len(pairs)), dtype=np.float64)
    q = np.empty(len(pairs), dtype=np.float64)
    for i, (_, mask) in enumerate(pairs):
        fmat[mask, i] = 1.0
        q[i] = float(np.mean(x[mask]))

    return fmat, q

# Parallactic angle

# ITRF geocentric coordinates in metres, verbatim from tempo2's observatories.dat
OBSERVATORY_ITRF = {
    'pks':     (-4554231.500,  2816759.100, -3454036.300),   # PARKES / MURRIYANG
    'meerkat': ( 5109360.133,  2006852.586, -3238948.127),   # MEERKAT
}

_WGS84_A = 6378137.0
_WGS84_E2 = (1.0 / 298.257223563) * (2.0 - 1.0 / 298.257223563)


def _itrf_to_geodetic(x, y, z):
    """Geodetic latitude and longitude, in radians, from ITRF metres on WGS84.

    The latitude is geodetic, i.e. measured from the local vertical, which is the
    convention :func:`parallactic_angle` needs.
    """
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1.0 - _WGS84_E2))
    for _ in range(8):
        N = _WGS84_A / np.sqrt(1.0 - _WGS84_E2 * np.sin(lat)**2)
        h = p / np.cos(lat) - N
        lat = np.arctan2(z, p * (1.0 - _WGS84_E2 * N / (N + h)))

    return lat, lon


def _observatories_dat():
    """tempo2's observatory table as ``{key: (x, y, z)}``, or ``{}`` if unreachable.

    Keys are the lowercased code and name columns, so either identifies a site.
    """
    root = os.environ.get('TEMPO2')
    if not root:
        return {}
    path = os.path.join(root, 'observatory', 'observatories.dat')
    if not os.path.exists(path):
        return {}

    table = {}
    for line in open(path):
        fields = line.split()
        if len(fields) < 5 or line.lstrip().startswith('#'):
            continue
        try:
            xyz = tuple(float(v) for v in fields[:3])
        except ValueError:
            continue
        for key in (fields[4], fields[3]):
            table.setdefault(key.lower(), xyz)

    return table


def observatory_location(site):
    """Geodetic latitude and longitude in radians for an observing site.

    site: tempo2 observatory code or name, matched case-insensitively against
          :data:`OBSERVATORY_ITRF` and then against
          ``$TEMPO2/observatory/observatories.dat``; or an explicit
          ``(latitude, longitude)`` pair in degrees
    """
    if not isinstance(site, str):
        lat, lon = site
        return np.deg2rad(float(lat)), np.deg2rad(float(lon))

    key = site.strip().lower()
    xyz = OBSERVATORY_ITRF.get(key) or _observatories_dat().get(key)
    if xyz is None:
        raise KeyError(
            f'observatory_location: no coordinates for site {site!r}. Known sites are '
            f'{sorted(OBSERVATORY_ITRF)}; set $TEMPO2 to reach the full tempo2 table, or '
            f'pass site=(latitude_deg, longitude_deg).')

    return _itrf_to_geodetic(*xyz)


def greenwich_sidereal_angle(mjd_utc):
    """Greenwich mean sidereal time in radians, from a UTC MJD.

    UT1 is approximated by UTC and the equation of the equinoxes is dropped, together
    worth under 0.01 degrees of hour angle.
    """
    d = np.asarray(mjd_utc, dtype=np.float64) + (2400000.5 - 2451545.0)
    T = d / 36525.0
    deg = 280.46061837 + 360.98564736629 * d + 0.000387933 * T**2 - T**3 / 38710000.0

    return np.deg2rad(np.mod(deg, 360.0))


def parallactic_angle(psr, site=None):
    """Parallactic angle at each TOA, in radians, in ``(-pi, pi]``.

    The angle between the great circle from the source to the zenith and the great
    circle from the source to the north celestial pole,

        tan(psi) = sin(H) cos(phi) / [sin(phi) cos(dec) - cos(phi) sin(dec) cos(H)]

    with ``H`` the local hour angle and ``phi`` the geodetic latitude of the site.

    Computed from ``psr.stoas``, the site arrival times, and NOT from ``psr.toas``,
    which are barycentric: the two differ by the Roemer delay plus TDB - UTC, which is
    degrees of hour angle.

    psr:  Discovery Pulsar object, carrying ``stoas`` and the source position in
          ``phi`` and ``theta``
    site: observing site, if ``psr.telescope`` is absent or is to be overridden --
          a tempo2 code or name, or a ``(latitude, longitude)`` pair in degrees.
          With ``psr.telescope`` present each TOA is placed at its own site, so
          combined data spanning several telescopes needs no argument here.
    """
    if not hasattr(psr, 'stoas'):
        raise AttributeError(
            f'parallactic_angle: {psr.name} has no stoas, the site arrival times. The '
            f'barycentric toas are not a substitute -- they carry the Roemer delay, '
            f'which is degrees of hour angle. Rebuild the feather with the stoas column.')

    mjd = np.asarray(psr.stoas, dtype=np.float64) / 86400.0
    ra, dec = float(psr.phi), 0.5 * np.pi - float(psr.theta)

    telescope = getattr(psr, 'telescope', None)
    if site is not None:
        if telescope is not None:
            named = sorted(set(np.asarray(telescope).astype(str).tolist()))
            if len(named) > 1 or (len(named) == 1 and named[0].lower() != str(site).lower()):
                print(f'Warning: parallactic_angle: {psr.name} carries telescope(s) {named} '
                      f'but site={site!r} was given, and overrides them for every TOA.')
        lat, lon = observatory_location(site)
        H = greenwich_sidereal_angle(mjd) + lon - ra
    elif telescope is not None:
        labels = np.asarray(telescope).astype(str)
        gst = greenwich_sidereal_angle(mjd)
        H = np.empty(len(mjd), dtype=np.float64)
        lat = np.empty(len(mjd), dtype=np.float64)
        for value in sorted(set(labels.tolist())):
            mask = labels == value
            lat_s, lon_s = observatory_location(value)
            lat[mask] = lat_s
            H[mask] = gst[mask] + lon_s - ra
    else:
        raise ValueError(
            f'parallactic_angle: {psr.name} has no telescope column and no site= was '
            f'given, so there is nowhere to stand. Feathers written by enterprise carry '
            f'the column; otherwise pass e.g. site="meerkat".')

    return np.arctan2(np.sin(H) * np.cos(lat),
                      np.sin(lat) * np.cos(dec) - np.cos(lat) * np.sin(dec) * np.cos(H))


def _pa_basis_parts(psr, bin_flag, harmonic, site, project, project_tm, name, what):
    """Ingredients shared by the parallactic-angle GPs.

    Returns ``(ind, q, psi, Q_null)``: the channel-indicator block and its bin positions
    from :func:`_fd_constant_block`, the angle from :func:`parallactic_angle`, and an
    orthonormal basis for the span to remove (timing model, plus any ``project`` bases),
    or None when nothing is projected.
    """
    if int(harmonic) < 1:
        raise ValueError(f'{what}: harmonic must be a positive integer, '
                         f'got {harmonic}.')
    if not bin_flag:
        raise ValueError(
            f'{what}: {psr.name} needs bin_flag, the per-TOA flag whose '
            f'distinct values define the frequency bins. Bins guessed from the '
            f'frequencies would not be the receiver channelisation the leakage follows.')

    psi = parallactic_angle(psr, site=site)

    x = np.log(np.asarray(psr.freqs, dtype=np.float64))
    block = _fd_constant_block(psr, x, np.ones(len(x), dtype=bool), None, 'flag',
                               name, bin_flag, what=what)
    if block is None:
        raise ValueError(f'{what}: no usable frequency bins for {psr.name}.')
    ind, q = block

    Q_null = np.linalg.qr(normalise_tm_basis(psr))[0] if project_tm else None

    if project is not None:
        parts = project if isinstance(project, (list, tuple)) else [project]
        mats = []
        for part in parts:
            F_p = getattr(part, 'F', part)
            if callable(F_p):
                raise ValueError(
                    f'{what}: {psr.name}: a basis passed to project has a '
                    f'callable F, so it has no fixed column span to remove. Only bases '
                    f'with a constant design matrix can be projected out.')
            mats.append(np.asarray(F_p, dtype=np.float64))
        P = np.hstack(mats)
        if Q_null is not None:
            P = P - Q_null @ (Q_null.T @ P)
        Up, Sp, _ = np.linalg.svd(P, full_matrices=False)
        keep = Up[:, Sp > 1e-10 * Sp[0]]
        Q_null = keep if Q_null is None else np.hstack([Q_null, keep])

    return ind, q, psi, Q_null


def makegp_pa_free_phase(psr, bin_flag='chan', harmonic=2, site=None,
                         project=None, project_tm=True, variant='iso', name='pa_gp'):
    """Delay locked to a harmonic of the parallactic angle, per frequency channel.

    For a receiver with fixed feeds the sky rotates through the parallactic angle
    ``psi``, so leakage between the polarisation products enters the total-intensity
    profile modulated by ``sin(m psi)`` and ``cos(m psi)`` and shifts the fitted arrival
    time. The shift is constant in time at fixed ``psi`` and differs between frequency
    channels.

    ``harmonic`` is 2 because a rotation of the feed mixes Q and U through twice the
    angle while leaving V invariant, so a Jones rotation produces no first harmonic at
    all. A term at ``m = 1`` would need a different mechanism, such as an
    elevation-dependent deformation, and belongs in its own component rather than here.

    The basis is the channel-indicator basis of :func:`makegp_fd_piecewise` with
    ``kind='constant'`` and ``spacing='flag'``, multiplied row-wise by the sine and the
    cosine of the angle::

        column (c, sin):   F[i] = 1[chan_i = c] sin(m psi_i)
        column (c, cos):   F[i] = 1[chan_i = c] cos(m psi_i)

    giving ``2 x n_chan`` columns, the sin block then the cos block, so column j of one
    matches column j of the other and both match bin j. Each channel therefore carries a
    free amplitude and a free phase. The bins come from ``bin_flag`` and nowhere else:
    the leakage is a property of the receiver's channelisation, so bins inferred from
    the frequencies would be a guess at something the instrument already records.

    The coefficients carry a proper prior over the whole basis, in seconds, on
    ``[-10, -6]``. ``variant='iso'`` is ``Phi = sigma**2 I``, sampling one
    ``{psr}_{name}_log10_sigma``, so a channel of amplitude A has ``E[A**2] = 2 sigma**2``
    and its phase is uniform. ``'diag'`` gives the sin and cos blocks their own scales
    ``_log10_sigma_X`` and ``_log10_sigma_Y``, and ``'full'`` adds their correlation
    ``_rho_XY`` on ``[-1, 1]``; both give ``E[A**2] = sigma_X**2 + sigma_Y**2`` and a
    phase drawn from an ellipse. Shifting the origin of ``psi`` rotates the coefficient
    pair, so only ``'iso'`` and ``'full'`` are closed under it: ``'diag'`` pins the
    ellipse to the sine and cosine axes, and ``rho -> +-1`` is
    :func:`makegp_pa_common_phase`.

    The basis is not orthonormalised, and the projections are applied to the design
    matrix while Phi is left alone, which leaves the columns unequal in norm:
    ``sigma**2 F F^T`` with the projected ``F`` is the prior the unprojected model
    induces on the subspace that remains, so rescaling the columns would substitute a
    different one. Rank deficiency is harmless, since the Woodbury form inverts Phi
    rather than F, and an annihilated direction integrates back to its prior. The
    delivered rank is reported.

    psr:        Discovery Pulsar object
    bin_flag:   per-TOA flag whose distinct values define the frequency bins
    harmonic:   multiple of the parallactic angle to model
    site:       observing site, passed to :func:`parallactic_angle`
    project:    further bases to remove alongside the timing model, each an array or a
                GP with a non-callable ``F``. Both columns of a channel have a non-zero
                mean within it, so each overlaps the corresponding channel
                indicator of a piecewise-constant FD basis; pass that GP here to remove
                the overlap
    project_tm: remove the timing-model column span
    variant:    ``'iso'``, ``'diag'`` or ``'full'``
    name:       base name for the GP parameters
    """
    from . import prior as _prior

    ind, q, psi, Q_null = _pa_basis_parts(psr, bin_flag, harmonic, site, project,
                                          project_tm, name, 'makegp_pa_free_phase')
    harmonic = int(harmonic)

    fmat = np.hstack([ind * np.sin(harmonic * psi)[:, None],
                      ind * np.cos(harmonic * psi)[:, None]])

    if Q_null is not None:
        before = np.linalg.svd(fmat, compute_uv=False)
        fmat = fmat - Q_null @ (Q_null.T @ fmat)
        after = np.linalg.svd(fmat, compute_uv=False)
        rank = int(np.sum(after > 1e-8 * after[0])) if after[0] > 0 else 0
        print(f'makegp_pa_free_phase: {psr.name} {name}: {fmat.shape[1]} columns over '
              f'{ind.shape[1]} bin(s) at {harmonic} x PA; projection leaves rank {rank}, '
              f'retaining {np.sum(after**2) / np.sum(before**2):.4f} of the basis power.')
        if rank < fmat.shape[1]:
            print(f'Warning: makegp_pa_free_phase: {psr.name} {name}: the projection '
                  f'annihilates {fmat.shape[1] - rank} direction(s), which stay in the '
                  f'basis and integrate back to their prior rather than being dropped, '
                  f'so the marginal likelihood is unaffected.')

    if variant not in ('iso', 'diag', 'full'):
        raise ValueError(f"makegp_pa_free_phase: variant must be 'iso', 'diag' or 'full', "
                         f"got {variant!r}.")
    nbin = ind.shape[1]

    if variant == 'iso':
        suffixes = [('log10_sigma', [-10.0, -6.0])]
        signame = f'{psr.name}_{name}_log10_sigma'
        ones = matrix.jnparray(np.ones(fmat.shape[1]))

        def getphi(params):
            return 10.0**(2.0 * params[signame]) * ones
    else:
        suffixes = [('log10_sigma_X', [-10.0, -6.0]), ('log10_sigma_Y', [-10.0, -6.0])]
        sxname = f'{psr.name}_{name}_log10_sigma_X'
        syname = f'{psr.name}_{name}_log10_sigma_Y'
        rhoname = f'{psr.name}_{name}_rho_XY'
        blk = matrix.jnparray(np.ones(nbin))

        if variant == 'diag':
            def getphi(params):
                return jnp.concatenate([10.0**(2.0 * params[sxname]) * blk,
                                        10.0**(2.0 * params[syname]) * blk])
        else:
            suffixes.append(('rho_XY', [-1.0, 1.0]))
            eye = matrix.jnparray(np.eye(nbin))

            def getphi(params):
                sX, sY = 10.0**params[sxname], 10.0**params[syname]
                r = params[rhoname]
                M = jnp.array([[sX * sX, r * sX * sY], [r * sX * sY, sY * sY]])
                return jnp.kron(M, eye)

    getphi.params = [f'{psr.name}_{name}_{suf}' for suf, _ in suffixes]

    _prior.priordict_standard.update(
        {f'{re.escape(psr.name)}_{name}_{suf}': box for suf, box in suffixes})

    Phi = (matrix.NoiseMatrix2D_var(getphi) if variant == 'full'
           else matrix.NoiseMatrix1D_var(getphi))
    gp = matrix.VariableGP(Phi, fmat)
    gp.index = {f'{psr.name}_{name}_coefficients({fmat.shape[1]})': slice(0, fmat.shape[1])}
    gp.name, gp.pos, gp.gpname, gp.gpcommon = psr.name, psr.pos, name, []
    gp.pa_harmonic = harmonic
    gp.pa_bins = np.exp(q)
    gp.pa_angle = psi
    gp.pa_variant = variant

    return gp


makegp_pa_quadrature = makegp_pa_free_phase


def makegp_pa_common_phase(psr, bin_flag='chan', harmonic=2, site=None, project=None,
                           project_tm=True, phase=None, name='pa_gp'):
    """Delay locked to a harmonic of the parallactic angle, one phase across channels.

    The restriction of :func:`makegp_pa_free_phase` to a phase shared by every channel:
    each channel carries one signed amplitude on the single sinusoid::

        column c:   F[i] = 1[chan_i = c] sin(m psi_i + theta)

    giving ``n_chan`` columns whose coefficients are the channel amplitudes in seconds.
    ``psi`` is :func:`parallactic_angle`: ``tan(psi) = sin(H) cos(phi) / [sin(phi) cos(dec)
    - cos(phi) sin(dec) cos(H)]`` in ``(-pi, pi]``, so ``theta`` flips sign with ``psi``.
    A signed amplitude makes ``theta`` and ``theta + pi`` the same model, so ``theta``
    is sampled on ``[-pi/2, pi/2)`` as ``{psr}_{name}_phase``, in radians; ``F`` is
    callable unless ``phase`` fixes it.

    The coefficients carry ``Phi = sigma**2 I``, sampling ``{psr}_{name}_log10_sigma``
    in seconds on ``[-10, -6]``; a channel of amplitude a has ``E[a**2] = sigma**2``,
    against ``2 sigma**2`` for the free-phase model. The basis is not
    orthonormalised, as for :func:`makegp_pa_free_phase`.

    psr:        Discovery Pulsar object
    bin_flag:   per-TOA flag whose distinct values define the frequency bins
    harmonic:   multiple of the parallactic angle to model
    site:       observing site, passed to :func:`parallactic_angle`
    project:    further bases to remove alongside the timing model, each an array or a
                GP with a non-callable ``F``
    project_tm: remove the timing-model column span
    phase:      fixed value of ``theta`` in radians; None samples it
    name:       base name for the GP parameters
    """
    from . import prior as _prior

    ind, q, psi, Q_null = _pa_basis_parts(psr, bin_flag, harmonic, site, project,
                                          project_tm, name, 'makegp_pa_common_phase')
    harmonic = int(harmonic)
    ncol = ind.shape[1]

    sinm = ind * np.sin(harmonic * psi)[:, None]
    cosm = ind * np.cos(harmonic * psi)[:, None]
    if Q_null is not None:
        S = sinm - Q_null @ (Q_null.T @ sinm)
        C = cosm - Q_null @ (Q_null.T @ cosm)
        ranks, kept = [], []
        for theta in (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4):
            F = S * np.cos(theta) + C * np.sin(theta)
            raw = sinm * np.cos(theta) + cosm * np.sin(theta)
            sv = np.linalg.svd(F, compute_uv=False)
            ranks.append(int(np.sum(sv > 1e-8 * sv[0])) if sv[0] > 0 else 0)
            kept.append(np.sum(F**2) / np.sum(raw**2))
        print(f'makegp_pa_common_phase: {psr.name} {name}: {ncol} columns over {ncol} '
              f'bin(s) at {harmonic} x PA with one phase; over four phases the projection '
              f'leaves rank {min(ranks)}-{max(ranks)}, retaining {min(kept):.4f}-{max(kept):.4f} '
              f'of the basis power.')
        if min(ranks) < ncol:
            print(f'Warning: makegp_pa_common_phase: {psr.name} {name}: the projection '
                  f'annihilates up to {ncol - min(ranks)} direction(s), which stay in the '
                  f'basis and integrate back to their prior rather than being dropped, '
                  f'so the marginal likelihood is unaffected.')
    else:
        S, C = sinm, cosm

    signame = f'{psr.name}_{name}_log10_sigma'
    phname = f'{psr.name}_{name}_phase'
    ones = matrix.jnparray(np.ones(ncol))

    def getphi(params):
        return 10.0**(2.0 * params[signame]) * ones
    getphi.params = [signame]

    _prior.priordict_standard.update(
        {f'{re.escape(psr.name)}_{name}_log10_sigma': [-10.0, -6.0]})

    if phase is not None:
        fmat = S * np.cos(float(phase)) + C * np.sin(float(phase))
    else:
        _prior.priordict_standard.update(
            {f'{re.escape(psr.name)}_{name}_phase': [-0.5 * float(np.pi), 0.5 * float(np.pi)]})
        S_j, C_j = matrix.jnparray(S), matrix.jnparray(C)

        def fmat(params):
            theta = params[phname]
            return S_j * jnp.cos(theta) + C_j * jnp.sin(theta)
        fmat.params = [phname]

    gp = matrix.VariableGP(matrix.NoiseMatrix1D_var(getphi), fmat)
    gp.index = {f'{psr.name}_{name}_coefficients({ncol})': slice(0, ncol)}
    gp.name, gp.pos, gp.gpname, gp.gpcommon = psr.name, psr.pos, name, []
    gp.pa_harmonic = harmonic
    gp.pa_bins = np.exp(q)
    gp.pa_angle = psi
    gp.pa_phase = phase

    return gp

def makegp_pa_boost(psr, z, bin_flag='chan', hand=1, chi0=0.0, variant='full', era_split=None,
                    site=None, project=None, project_tm=True, name='pa_boost'):
    """Polarimetric boost delay, with coefficients in dimensionless boost units.

    A Hermitian instrumental distortion with boost vector ``b = (bX, bY, bZ)`` in the
    antenna frame changes the total intensity to first order as ``S0' = S0 + 2 b . S_ant``
    (van Straten 2013, eq. 3, |b| << 1), so a differential gain error of 1 per cent is
    ``bX = 0.005``; the coefficients here are ``b``, not ``2b``. Antenna-frame Stokes are
    the celestial-frame (IAU, Britton 2000) ones rotated by twice the parallactic angle,
    with handedness ``h`` (Zahraoui et al., PIB eq. 5)::

        Q_ant =  Q cos 2h psi + U sin 2h psi
        U_ant = -Q sin 2h psi + U cos 2h psi
        V_ant =  V

    The ToA responds through the susceptibilities ``z = (zQ, zU, zV)`` of each channel,
    in seconds per unit boost, positive when a positive boost delays the ToA: fixed
    inputs computed from the Stokes template, not here. For TOA i in channel k::

        dX_i =  zQ[k] cos 2h psi_i + zU[k] sin 2h psi_i
        dY_i = -zQ[k] sin 2h psi_i + zU[k] cos 2h psi_i
        delay_i = bX[k] dX_i + bY[k] dY_i

    The basis is ``[F_FD * dX | F_FD * dY]``, ``F_FD`` the channel indicators, ``2 n_chan``
    columns. ``dZ = zV[k]`` is constant in time within a channel and shares its column
    space with the FD basis, so no Z block is built: an FD coefficient ``c_k`` contains
    ``bZ[k] zV[k]``.

    ``psi`` is :func:`parallactic_angle`: ``atan2(sin H cos phi_site, sin phi_site cos dec
    - cos phi_site sin dec cos H)`` in ``(-pi, pi]``, positive east of the meridian and
    zero at upper culmination, per TOA. ``chi0`` offsets the receptor basis,
    ``psi -> psi + chi0``; ``chi0='sample'`` samples it on ``[-pi/4, pi/4)`` as
    ``{psr}_{name}_chi0``, and the basis is then callable.

    Prior ``Phi = I (x) M`` over the channels, ``M = D R D``, ``D = diag(sigmaX, sigmaY)``,
    ``R = [[1, rho], [rho, 1]]`` (the n = 2 partial-correlation construction of the
    correlated Legendre ECORR), in boost units with no normalisation after projection.
    ``variant='full'`` samples ``{psr}_{name}_log10_sigma_X`` and ``_log10_sigma_Y`` on
    ``[-6, -1.3]`` and ``_rho_XY`` on ``[-1, 1]``; ``'diag'`` fixes ``rho = 0``; ``'iso'``
    sets ``sigmaX = sigmaY`` as ``{psr}_{name}_log10_sigma`` with ``rho = 0``.

    ``era_split`` (an MJD in site time) gives each era its own coefficient set under the
    shared prior, doubling the blocks. Projections are those of
    :func:`makegp_pa_free_phase`, applied once to the sine and cosine blocks, of which
    ``dX`` and ``dY`` are per-channel combinations.

    z:          mapping ``{bin label: (zQ, zU[, zV])}``, or an array ``(n_chan, 2 or 3)`` in
                the order of the sorted ``bin_flag`` values; seconds per unit boost
    bin_flag:   per-TOA flag whose distinct values define the channels
    hand:       ``+1`` or ``-1``
    chi0:       receptor-basis offset in radians, or ``'sample'``
    variant:    ``'full'``, ``'diag'`` or ``'iso'``
    era_split:  MJD splitting the coefficient sets, or None
    site, project, project_tm, name: as for :func:`makegp_pa_free_phase`
    """
    from . import prior as _prior

    if variant not in ('full', 'diag', 'iso'):
        raise ValueError(f"makegp_pa_boost: variant must be 'full', 'diag' or 'iso', got {variant!r}.")
    hand = int(hand)
    if hand not in (1, -1):
        raise ValueError(f'makegp_pa_boost: hand must be +1 or -1, got {hand}.')
    sample_chi0 = isinstance(chi0, str)
    if sample_chi0 and chi0 != 'sample':
        raise ValueError(f"makegp_pa_boost: chi0 must be a number or 'sample', got {chi0!r}.")

    ind, q, psi, Q_null = _pa_basis_parts(psr, bin_flag, 2, site, project, project_tm, name,
                                          'makegp_pa_boost')
    ncol = ind.shape[1]
    labels = [lab for lab, m in _fd_flag_masks(psr, np.ones(len(psi), dtype=bool), name,
                                               bin_flag, 'makegp_pa_boost') if m.any()]
    if len(labels) != ncol:
        raise ValueError(f'makegp_pa_boost: {psr.name}: {len(labels)} bin labels for '
                         f'{ncol} basis columns.')

    if isinstance(z, dict):
        missing = [lab for lab in labels if lab not in z]
        if missing:
            raise KeyError(f'makegp_pa_boost: {psr.name}: no susceptibility for '
                           f'{bin_flag!r} bin(s) {missing}.')
        zarr = np.array([np.asarray(z[lab], dtype=np.float64)[:3] for lab in labels])
    else:
        zarr = np.asarray(z, dtype=np.float64)
    if zarr.ndim != 2 or zarr.shape[0] != ncol or zarr.shape[1] not in (2, 3):
        raise ValueError(f'makegp_pa_boost: {psr.name}: z must give (zQ, zU[, zV]) for each '
                         f'of the {ncol} bins, got shape {zarr.shape}.')
    if not np.all(np.isfinite(zarr[:, :2])):
        raise ValueError(f'makegp_pa_boost: {psr.name}: zQ and zU must be finite.')
    zQ, zU = zarr[:, 0], zarr[:, 1]
    zV = zarr[:, 2] if zarr.shape[1] == 3 else np.full(ncol, np.nan)

    S = ind * np.sin(2.0 * psi)[:, None]
    C = ind * np.cos(2.0 * psi)[:, None]
    if era_split is not None:
        t = np.asarray(getattr(psr, 'stoas', psr.toas), dtype=np.float64) / 86400.0
        early = t < float(era_split)
        if early.all() or not early.any():
            raise ValueError(f'makegp_pa_boost: {psr.name}: era_split={era_split} leaves an '
                             f'era with no TOAs.')
        eras = [(S * early[:, None], C * early[:, None]),
                (S * ~early[:, None], C * ~early[:, None])]
    else:
        eras = [(S, C)]

    def proj(F):
        return F if Q_null is None else F - Q_null @ (Q_null.T @ F)
    eras = [(proj(S_), proj(C_)) for S_, C_ in eras]
    nera = len(eras)
    ntot = 2 * ncol * nera

    def combine(S_, C_, cos_a, sin_a, xp):
        Sh = hand * S_
        cp = C_ * cos_a - Sh * sin_a
        sp = Sh * cos_a + C_ * sin_a
        return xp.hstack([cp * zQ + sp * zU, -sp * zQ + cp * zU])

    def basis_np(offset):
        a = 2.0 * hand * offset
        return np.hstack([combine(S_, C_, np.cos(a), np.sin(a), np) for S_, C_ in eras])

    F0 = basis_np(0.0 if sample_chi0 else float(chi0))
    if Q_null is not None:
        raw = np.hstack([combine(S_, C_, 1.0, 0.0, np) for S_, C_ in
                         ([(S, C)] if era_split is None else
                          [(S * early[:, None], C * early[:, None]),
                           (S * ~early[:, None], C * ~early[:, None])])])
        sv = np.linalg.svd(F0, compute_uv=False)
        rank = int(np.sum(sv > 1e-8 * sv[0])) if sv[0] > 0 else 0
        print(f'makegp_pa_boost: {psr.name} {name}: {ntot} columns over {ncol} bin(s)'
              f'{f" and {nera} eras" if nera > 1 else ""}, hand {hand:+d}; projection leaves '
              f'rank {rank}, retaining {np.sum(F0**2) / np.sum(raw**2):.4f} of the basis power.')
        if rank < ntot:
            print(f'Warning: makegp_pa_boost: {psr.name} {name}: the projection annihilates '
                  f'{ntot - rank} direction(s), which stay in the basis and integrate back '
                  f'to their prior rather than being dropped.')

    if variant == 'iso':
        sxname = syname = f'{psr.name}_{name}_log10_sigma'
        names = [sxname]
    else:
        sxname, syname = f'{psr.name}_{name}_log10_sigma_X', f'{psr.name}_{name}_log10_sigma_Y'
        names = [sxname, syname]
    rhoname = f'{psr.name}_{name}_rho_XY'
    if variant == 'full':
        names.append(rhoname)
    eye_c, eye_e = matrix.jnparray(np.eye(ncol)), matrix.jnparray(np.eye(nera))

    def getphi(params):
        sX = 10.0 ** params[sxname]
        sY = 10.0 ** params[syname]
        r = params[rhoname] if variant == 'full' else 0.0
        M = jnp.array([[sX * sX, r * sX * sY], [r * sX * sY, sY * sY]])
        return jnp.kron(eye_e, jnp.kron(M, eye_c))
    getphi.params = list(names)

    boxes = {f'{re.escape(n)}': [-6.0, -1.3] for n in dict.fromkeys([sxname, syname])}
    if variant == 'full':
        boxes[re.escape(rhoname)] = [-1.0, 1.0]
    chiname = f'{psr.name}_{name}_chi0'
    if sample_chi0:
        boxes[re.escape(chiname)] = [-0.25 * float(np.pi), 0.25 * float(np.pi)]
    _prior.priordict_standard.update(boxes)

    if sample_chi0:
        eras_j = [(matrix.jnparray(S_), matrix.jnparray(C_)) for S_, C_ in eras]
        zQ_j, zU_j = matrix.jnparray(zQ), matrix.jnparray(zU)

        def combine_j(S_, C_, a):
            Sh = hand * S_
            cp = C_ * jnp.cos(a) - Sh * jnp.sin(a)
            sp = Sh * jnp.cos(a) + C_ * jnp.sin(a)
            return jnp.hstack([cp * zQ_j + sp * zU_j, -sp * zQ_j + cp * zU_j])

        def fmat(params):
            a = 2.0 * hand * params[chiname]
            return jnp.hstack([combine_j(S_, C_, a) for S_, C_ in eras_j])
        fmat.params = [chiname]
    else:
        fmat = F0

    gp = matrix.VariableGP(matrix.NoiseMatrix2D_var(getphi), fmat)
    gp.index = {f'{psr.name}_{name}_coefficients({ntot})': slice(0, ntot)}
    gp.name, gp.pos, gp.gpname, gp.gpcommon = psr.name, psr.pos, name, []
    gp.pa_harmonic = 2
    gp.pa_bins = np.exp(q)
    gp.pa_labels = labels
    gp.pa_angle = psi
    gp.pa_hand = hand
    gp.pa_chi0 = None if sample_chi0 else float(chi0)
    gp.pa_variant = variant
    gp.pa_z = np.column_stack([zQ, zU, zV])
    gp.pa_era_split = era_split
    gp.pa_delta = None if sample_chi0 else (F0[:, :ncol].sum(axis=1), F0[:, ncol:2 * ncol].sum(axis=1))

    return gp

def chrom_poly_basis(psr, fref=None):
    """Callable chromatic polynomial basis ``U * (fref/freq)**alpha``.

    ``U`` is the SVD-orthonormalised [1, t, t**2] temporal design matrix. The SVD is a
    fixed right-multiplication of the raw polynomial, so it leaves the column span, and
    hence the marginal likelihood under an orthonormalising GP, unchanged.

    Returns ``fmat(alpha) -> (N_toa, 3)``, carrying ``ncol``, the reference frequency
    ``fref`` and the temporal ``svd`` factors, for use with
    :func:`makegp_improper_varF`.

    psr:  Discovery Pulsar object
    fref: reference frequency; defaults to the geometric mean of the TOA frequencies
    """
    t0_sec  = float(np.mean(psr.toas))
    toas_yr = (psr.toas - t0_sec) / const.yr

    if fref is None:
        # Geometric mean of observing frequencies
        fref = float(np.exp(np.mean(np.log(np.asarray(psr.freqs)))))

    M_poly = np.vstack([np.ones_like(toas_yr), toas_yr, toas_yr**2]).T
    U, S, Vt = np.linalg.svd(M_poly, full_matrices=False)

    U_j     = matrix.jnparray(U)
    fnorm_j = matrix.jnparray(fref / np.asarray(psr.freqs))

    def fmat(alpha):
        return U_j * fnorm_j[:, None] ** alpha
    fmat.ncol = 3
    fmat.fref = fref
    fmat.svd = {'S': S, 'Vt': Vt}

    return fmat


def makegp_chrom_poly_svd(psr, fref=None, constant=1e40, name='chrom_gp', project=None,
                          noisedict={}):
    """SVD-orthogonalised chromatic polynomial GP, marginalised analytically.

    A :func:`chrom_poly_basis` carried by :func:`makegp_improper_varF` with the timing
    model projected out and the basis orthonormalised at every alpha. Both are needed:
    at alpha = 3 the raw basis lies 99.95% inside the timing-model span, so without the
    projection the marginal likelihood is singular, and without the orthonormalisation
    the volume term alone swings some 19 nats across the alpha prior.

    Shares ``alpha`` with a companion chromatic Fourier (or FFTint) GP via the
    parameter name ``{psr}_{name}_alpha``.

    psr:       Discovery Pulsar object
    fref:      reference frequency; defaults to the geometric mean of the TOA frequencies
    constant:  diagonal of the flat improper prior over the coefficients
    name:      base name for the GP parameters
    project:   further bases to remove alongside the timing model -- an array or a GP
               with a non-callable ``F``, e.g. the result of
               :func:`makegp_fd_piecewise`, whose time-constant frequency structure
               overlaps the constant-in-time part of this basis
    noisedict: fixed value for ``{psr}_{name}_alpha``; if present the basis is
               evaluated once and a ConstantGP returned
    """
    fmat = chrom_poly_basis(psr, fref=fref)

    gp = makegp_improper_varF(psr, fmat, constant=constant, name=name,
                              param_names=['alpha'], noisedict=noisedict,
                              project=project, project_tm=True, orthonormalize=True)
    gp.svd = fmat.svd

    return gp

# Fourier GP

def psr_components(toas, dt=7.0, Tspan=None):
    """Fourier bins a TOA series can resolve, from its block-to-block cadence.

    TOAs within dt days count as one observing block; the median gap between
    blocks sets the Nyquist period 2*gap, and the span divides by it.

    toas : TOA times in seconds
    Tspan : span in seconds the basis is built on; the TOA span if None
    """
    day = np.asarray(toas) / 86400.0
    blocks = np.unique(np.floor((day - day.min()) / dt))
    gaps = np.diff(blocks)
    span = (day.max() - day.min()) if Tspan is None else Tspan / 86400.0
    return max(1, round(span / (2 * (np.median(gaps) if gaps.size else 1.0) * dt)))


def getspan(psrs):
    if isinstance(psrs, Iterable):
        return max(psr.toas.max() for psr in psrs) - min(psr.toas.min() for psr in psrs)
    else:
        return psrs.toas.max() - psrs.toas.min()

def getstart(psrs):
    if isinstance(psrs, Iterable):
        return min(psr.toas.min() for psr in psrs)
    else:
        return psrs.toas.min()


def fourierbasis(psr, components, T=None):
    if T is None:
        T = getspan(psr)

    f  = np.arange(1, components + 1, dtype=np.float64) / T
    df = np.diff(np.concatenate((np.array([0]), f)))

    fmat = np.zeros((psr.toas.shape[0], 2*components), dtype=np.float64)
    for i in range(components):
        fmat[:, 2*i  ] = np.sin(2.0 * np.pi * f[i] * psr.toas)
        fmat[:, 2*i+1] = np.cos(2.0 * np.pi * f[i] * psr.toas)

    return np.repeat(f, 2), np.repeat(df, 2), fmat

def fourierbasis_dm(psr, components, T=None, fref=1400.0):
    """Fourier design matrix for a DM (dispersion measure) Gaussian process.

    Identical to :func:`fourierbasis`, but each row is scaled by the cold-plasma
    dispersion factor ``(fref / psr.freqs) ** 2``, i.e. a fixed chromatic index
    alpha = 2. Use :func:`fourierbasis_chrom` when the chromatic index is a free
    parameter, in which case the process is general chromatic noise rather than DM.
    """
    f, df, fmat = fourierbasis(psr, components, T)

    Dm = (fref / psr.freqs)**2

    return f, df, fmat * Dm[:, None]

def fourierbasis_chrom(psr, components, T=None, fref=1400.0, alpha=None):
    """Fourier design matrix for a chromatic Gaussian process with variable index.

    Returns a callable design-matrix factory ``fmatfunc(alpha)`` that scales the
    achromatic :func:`fourierbasis` columns by ``(fref / psr.freqs) ** alpha``,
    where the chromatic index ``alpha`` is a free parameter. Because alpha is not
    fixed to 2 the resulting process is general chromatic noise, not DM; use
    :func:`fourierbasis_dm` for the alpha = 2 (DM) case.
    """
    f, df, fmat = fourierbasis(psr, components, T)

    fmat, fnorm = matrix.jnparray(fmat), matrix.jnparray(fref / psr.freqs)
    if alpha is None:
        def fmatfunc(alpha):
            return fmat * fnorm[:, None]**alpha
    else:
        return f, df, fmat * fnorm[:, None]**alpha

    return f, df, fmatfunc

def _band_envelope(psr, fcenter, log10_bw, scale=None):
    """Smooth, RMS-normalised band-pass envelope over ``psr.freqs``.

    Parametrised by band centre ``fcenter`` (MHz) and ``log10`` of the bandwidth, so
    the band is well-defined for every parameter value (there is no inverted
    ``fhigh < flow`` region that collapses the envelope to zero). The sigmoid edges
    roll off over ``scale`` MHz, defaulting to 10% of the bandwidth so the gradient is
    informative at any width. The envelope is normalised to unit RMS across the TOAs,
    which decouples the GP amplitude from the bandwidth and removes the
    amplitude-width funnel that makes the (flow, fhigh) model hard to sample.
    """
    freqs = matrix.jnparray(psr.freqs)
    bw = 10.0 ** log10_bw
    flow, fhigh = fcenter - 0.5 * bw, fcenter + 0.5 * bw
    s = 0.1 * bw if scale is None else scale
    env = jnp.reciprocal(1.0 + jnp.exp((flow - freqs) / s)) * jnp.reciprocal(1.0 + jnp.exp((freqs - fhigh) / s))
    return env / jnp.sqrt(jnp.mean(env ** 2) + 1e-12)

def fourierbasis_band(psr, components, T=None, scale=None):
    """Band-limited Fourier basis parametrised by ``(fcenter, log10_bw)``.

    The band is specified by its centre and log10 bandwidth (guaranteeing
    ``fhigh > flow``), with the smooth, RMS-normalised envelope of
    :func:`_band_envelope`. The returned ``fmatfunc(fcenter, log10_bw)`` scales the
    achromatic :func:`fourierbasis` columns by that envelope.
    """
    f, df, fmat = fourierbasis(psr, components, T)

    fmat = matrix.jnparray(fmat)
    def fmatfunc(fcenter, log10_bw):
        return fmat * _band_envelope(psr, fcenter, log10_bw, scale)[:, None]

    return f, df, fmatfunc

def fourierbasis_band_alpha(psr, components, T=None, fref=1400.0, scale=None):
    """Chromatic band-limited Fourier basis parametrised by ``(fcenter, log10_bw, alpha)``.

    As :func:`fourierbasis_band`, but additionally scales by
    ``(fref / psr.freqs) ** alpha`` for a variable chromatic index ``alpha``.
    """
    f, df, fmat = fourierbasis(psr, components, T)

    fmat, fnorm = matrix.jnparray(fmat), matrix.jnparray(fref / psr.freqs)
    def fmatfunc(fcenter, log10_bw, alpha):
        return fmat * fnorm[:, None]**alpha * _band_envelope(psr, fcenter, log10_bw, scale)[:, None]

    return f, df, fmatfunc

def make_fourierbasis_dm(alpha=2.0, tndm=False):
    """Build a DM Fourier-basis function with a fixed chromatic index ``alpha``.

    Returns a ``basis(psr, components, T, fref)`` callable whose columns are the
    achromatic :func:`fourierbasis` scaled by ``(fref / psr.freqs) ** alpha``. With
    ``tndm=True`` the tempo2/TempoNest DM normalisation is also applied. A genuine
    DM basis should keep ``alpha = 2``; for a variable chromatic index use
    :func:`fourierbasis_chrom`.
    """
    def basis(psr, components, T=None, fref=1400.0):
        f, df, fmat = fourierbasis(psr, components, T)

        if tndm:
            Dm = (fref / psr.freqs) ** alpha * np.sqrt(12.0) * np.pi / 1400.0 / 1400.0 / 2.41e-4
        else:
            Dm = (fref / psr.freqs) ** alpha

        return f, df, fmat * Dm[:, None]

    return basis

def make_fourierbasis_chrom(alpha=4.0, tndm=False):
    """Build a chromatic Fourier-basis function with a fixed chromatic index ``alpha``.

    Thin wrapper around :func:`make_fourierbasis_dm` with a default ``alpha = 4`` (a
    common scattering-like index). The returned basis scales the achromatic
    :func:`fourierbasis` columns by ``(fref / psr.freqs) ** alpha``. Use this for a
    fixed-index chromatic process; for DM use :func:`make_fourierbasis_dm` (alpha = 2).
    """
    return make_fourierbasis_dm(alpha=alpha, tndm=tndm)

def dmfourierbasis(psr, components, T=None, fref=1400.0):
    warnings.warn("dmfourierbasis is deprecated; use fourierbasis_dm instead.",
                  DeprecationWarning, stacklevel=2)
    return fourierbasis_dm(psr, components, T=T, fref=fref)

def dmfourierbasis_alpha(psr, components, T=None, fref=1400.0):
    warnings.warn("dmfourierbasis_alpha is deprecated; use fourierbasis_chrom instead.",
                  DeprecationWarning, stacklevel=2)
    return fourierbasis_chrom(psr, components, T=T, fref=fref)

def make_dmfourierbasis(alpha=2.0, tndm=False):
    warnings.warn("make_dmfourierbasis is deprecated; use make_fourierbasis_dm instead.",
                  DeprecationWarning, stacklevel=2)
    return make_fourierbasis_dm(alpha=alpha, tndm=tndm)

def makegp_fourier(psr, prior, components, T=None, mean=None, fourierbasis=fourierbasis, common=[], exclude=['f', 'df'], name='fourierGP', **kwargs):
    argspec = inspect.getfullargspec(prior)
    argmap = [(arg if arg in common else f'{name}_{arg}' if f'{name}_{arg}' in common else f'{psr.name}_{name}_{arg}') +
              (f'({components[arg] if isinstance(components, dict) else components})' if argspec.annotations.get(arg) == typing.Sequence else '')
              for arg in argspec.args if arg not in exclude]

    # we'll create frequency bases using the longest vector parameter (e.g., for makefreespectrum_crn)
    if isinstance(components, dict):
        components = max(components.values())

    f, df, fmat = fourierbasis(psr, components, T, **kwargs)

    # f, df = matrix.jnparray(f), matrix.jnparray(df)
    def priorfunc(params):
        return prior(f, df, *[params[arg] for arg in argmap])
    priorfunc.params = argmap
    priorfunc.type = getattr(prior, 'type', None)

    if callable(fmat):
        argspec = inspect.getfullargspec(fmat)
        fargmap = [(arg if arg in common else f'{name}_{arg}' if f'{name}_{arg}' in common else f'{psr.name}_{name}_{arg}') +
                   (f'({components})' if argspec.annotations.get(arg) == typing.Sequence else '')
                   for arg in argspec.args if arg not in ['f', 'df']]

        def fmatfunc(params):
            return fmat(*[params[arg] for arg in fargmap])
        fmatfunc.params = fargmap

    gp = matrix.VariableGP(matrix.NoiseMatrix12D_var(priorfunc), fmatfunc if callable(fmat) else fmat)
    gp.index = {f'{psr.name}_{name}_coefficients({len(f)})': slice(0,len(f))} # better for cosine
    gp.name, gp.pos = psr.name, psr.pos
    gp.gpname, gp.gpcommon = name, common

    if mean is not None:
        margspec = inspect.getfullargspec(mean)
        margs = margspec.args + [arg for arg in margspec.kwonlyargs if arg not in margspec.kwonlydefaults]
        margmap = {arg: (arg if arg in common else f'{name}_{arg}' if f'{name}_{arg}' in common else f'{psr.name}_{name}_{arg}')
                   for arg in margs if not hasattr(psr, arg) and arg not in exclude}

        psrpars = {arg: getattr(psr, arg) for arg in margspec.args if hasattr(psr, arg)}

        def meanfunc(params):
            return mean(f, df, *psrpars.values(), **{arg: params[argname] for arg, argname in margmap.items()})
        meanfunc.params = sorted(margmap.values())

        gp.mean = meanfunc

    return gp


# for use in ArrayLikelihood. Same process for all pulsars.
def makecommongp_fourier(psrs, prior, components, T, fourierbasis=fourierbasis, means=None, common=[], exclude=['f', 'df'], vector=False,
                         name='fourierCommonGP', meansname='meanFourierCommonGP'):
    argspec = inspect.getfullargspec(prior)

    if vector:
        argmap = [arg if arg in common else f'{name}_{arg}' if f'{name}_{arg}' in common else
                  f'{name}_{arg}({len(psrs)})' for arg in argspec.args if arg not in exclude]
    else:
        argmaps = [[(arg if arg in common else f'{name}_{arg}' if f'{name}_{arg}' in common else f'{psr.name}_{name}_{arg}') +
                    (f'({components[arg] if isinstance(components, dict) else components})' if argspec.annotations.get(arg) == typing.Sequence else '') for psr in psrs]
                   for arg in argspec.args if arg not in exclude]

    # we'll create frequency bases using the longest vector parameter (e.g., for makefreespectrum_crn)
    if isinstance(components, dict):
        components = max(components.values())

    fs, dfs, fmats = zip(*[fourierbasis(psr, components, T) for psr in psrs])
    f, df = fs[0], dfs[0]

    if vector:
        vprior = jax.vmap(prior, in_axes=[None, None] +
                                         [0 if f'({len(psrs)})' in arg else None for arg in argmap])

        def priorfunc(params):
            return vprior(f, df, *[params[arg] for arg in argmap])

        priorfunc.params = sorted(argmap)
        priorfunc.type = getattr(prior, 'type', None)
    else:
        vprior = jax.vmap(prior, in_axes=[None, None] +
                                         [0 if isinstance(argmap, list) else None for argmap in argmaps])

        def priorfunc(params):
            vpars = [matrix.jnparray([params[arg] for arg in argmap]) if isinstance(argmap, list) else params[argmap]
                    for argmap in argmaps]
            return vprior(f, df, *vpars)

        priorfunc.params = sorted(set(sum([argmap if isinstance(argmap, list) else [argmap] for argmap in argmaps], [])))
        priorfunc.type = getattr(prior, 'type', None)

    gp = matrix.VariableGP(matrix.VectorNoiseMatrix12D_var(priorfunc), fmats)
    gp.index = {f'{psr.name}_{name}_coefficients({len(f)})': slice(len(f)*i,len(f)*(i+1))
                for i, psr in enumerate(psrs)}

    if means is not None:
        margspec = inspect.getfullargspec(means)
        margs = margspec.args + [arg for arg in margspec.kwonlyargs if arg not in margspec.kwonlydefaults]

        # parameters carried by the pulsar objects (e.g., pos), should be at the beginning of function
        psrpars = [{arg: getattr(psr, arg) for arg in margspec.args if hasattr(psrs[0], arg) and arg not in exclude}
                   for psr in psrs]

        # other means parameters, either common or pulsar-specific
        margmaps = [{arg: f'{meansname}_{arg}' if (f'{meansname}_{arg}' in common or arg in common) else f'{psr.name}_{meansname}_{arg}'
                     for arg in margs if not hasattr(psr, arg) and arg not in exclude} for psr in psrs]

        def meanfunc(params):
            return matrix.jnparray([means(f, df, *psrpar.values(), **{arg: params[argname] for arg, argname in margmap.items()})
                                    for psrpar, margmap in zip(psrpars, margmaps)])
        meanfunc.params = sorted(set.union(*[set(margmap.values()) for margmap in margmaps]))

        gp.means = meanfunc

    return gp


# these support leave-one-out PPC

def makegp_fourier_delay(psr, components, T=None, name='fourierGP'):
    argname = f'{psr.name}_{name}_mean({components*2})'

    _, _, fmat = fourierbasis(psr, components, T)
    Fmat = matrix.jnparray(fmat)

    def delayfunc(params):
        return matrix.jnp.dot(Fmat, params[argname])
    delayfunc.params = [argname]

    return delayfunc

def makegp_fourier_variance(psr, components, T=None, name='fourierGP', noisedict={}):
    argname = f'{psr.name}_{name}_variance({components*2},{components*2})'

    _, _, fmat = fourierbasis(psr, components, T)

    if argname in noisedict:
        return matrix.ConstantGP(matrix.NoiseMatrix2D_novar(noisedict[argname]), fmat)
    else:
        def priorfunc(params):
            return params[argname]
        priorfunc.params = [argname]

        return matrix.VariableGP(matrix.NoiseMatrix2D_var(priorfunc), fmat)

# Global Fourier GP

# makes a block-diagonal GP over all pulsars; returns a GlobalVariableGP object in which
# the prior is the concatenation of single-pulsar priors; with common variables, it can be used
# to implement CURN as a globalgp object, or to set up the optimal statistic
def makegp_fourier_allpsr(psrs, prior, components, T=None, fourierbasis=fourierbasis, common=[], name='allpsrFourierGP'):
    argspec = inspect.getfullargspec(prior)
    argmaps = [[(arg if arg in common else f'{name}_{arg}' if f'{name}_{arg}' in common else f'{psr.name}_{name}_{arg}') +
                (f'({components})' if argspec.annotations.get(arg) == typing.Sequence else '')
                for arg in argspec.args if arg not in ['f', 'df']] for psr in psrs]

    fs, dfs, fmats = zip(*[fourierbasis(psr, components, T) for psr in psrs])
    f, df = matrix.jnparray(fs[0]), matrix.jnparray(dfs[0])

    def priorfunc(params):
        return jnp.concatenate([prior(f, df, *[params[arg] for arg in argmap]) for argmap in argmaps])
    priorfunc.params = sorted(set(sum(argmaps, [])))

    def invprior(params):
        p = priorfunc(params)
        return 1.0 / p, jnp.sum(jnp.log(p))
    invprior.params = priorfunc.params

    gp = matrix.GlobalVariableGP(matrix.NoiseMatrix1D_var(priorfunc), fmats)
    gp.Phi_inv = invprior

    gp.index = {f'{psr.name}_{name}_coefficients({2*components})':
                slice((2*components)*i, (2*components)*(i+1)) for i, psr in enumerate(psrs)}
    gp.pos = [psr.pos for psr in psrs]
    gp.name = [psr.name for psr in psrs]

    return gp


def makeglobalgp_fourier(psrs, priors, orfs, components, T, fourierbasis=fourierbasis, means=None, common=[], exclude=['f', 'df'],
                         name='fourierGlobalGP', meansname='meanFourierGlobalGP'):
    priors = priors if isinstance(priors, list) else [priors]
    orfs   = orfs   if isinstance(orfs, list)   else [orfs]

    argmaps = []
    for prior, orf in zip(priors, orfs):
        argspec = inspect.getfullargspec(prior)
        priorname = f'{name}' if len(priors) == 1 else f'{name}_{re.sub("_", "", orf.__name__)}'
        argmaps.append([f'{priorname}_{arg}' + (f'({components})' if argspec.annotations.get(arg) == typing.Sequence else '')
                        for arg in argspec.args if arg not in exclude])

    fs, dfs, fmats = zip(*[fourierbasis(psr, components, T) for psr in psrs])
    f, df = matrix.jnparray(fs[0]), matrix.jnparray(dfs[0])

    orfmats = [matrix.jnparray([[orf(p1.pos, p2.pos) for p1 in psrs] for p2 in psrs]) for orf in orfs]

    if len(priors) == 1 and len(orfs) == 1:
        prior, orfmat, argmap = priors[0], orfmats[0], argmaps[0]

        def priorfunc(params):
            phi = prior(f, df, *[params[arg] for arg in argmap])

            # the jnp.dot handles the "pixel basis" case where the elements of orfmat are n-vectors
            # and phidiag is an (m x n)-matrix; here n is the number of pixels and m of Fourier components
            return jnp.block([[jnp.make2d(jnp.dot(phi, val)) for val in row] for row in orfmat])
        priorfunc.params = argmap
        priorfunc.type = jax.Array

        # if we're not in the pixel-basis case we can take a shortcut in making the inverse
        if orfmat.ndim == 2:
            invorf, orflogdet = matrix.jnparray(np.linalg.inv(orfmat)), np.linalg.slogdet(orfmat)[1]
            def invprior(params):
                phi = prior(f, df, *[params[arg] for arg in argmap])
                invphi = 1.0 / phi if phi.ndim == 1 else jnp.linalg.inv(phi)
                logdetphi = jnp.sum(jnp.log(phi)) if phi.ndim == 1 else jnp.linalg.slogdet(phi)[1]

                # |S_ij Gamma_ab| = prod_i (|S_i Gamma_ab|) = prod_i (S_i^npsr |Gamma_ab|)
                # log |S_ij Gamma_ab| = log (prod_i S_i^npsr) + log prod_i |Gamma_ab|
                #                     = npsr * sum_i log S_i + nfreqs |Gamma_ab|
                return (jnp.block([[jnp.make2d(val * invphi) for val in row] for row in invorf]),
                        phi.shape[0] * orflogdet + orfmat.shape[0] * logdetphi)
                        # was -orfmat.shape[0] * jnp.sum(jnp.log(invphidiag)))
            invprior.params = argmap
            invprior.type = jax.Array

            orfcf = matrix.jsp.linalg.cho_factor(orfmat)
            def factors(params):
                phi = prior(f, df, *[params[arg] for arg in argmap])
                # phi is the 1D diagonal PSD for a power-law (diagonal) HD spectrum;
                # cglogL consumes phicf as an (ngp x ngp) Cholesky factor tuple, so
                # densify the diagonal before factoring (ngp x ngp is small -- the
                # matrix-free path never forms the full (npsr*ngp)^2 matrix).
                phimat = phi if phi.ndim == 2 else jnp.diag(phi)
                phicf = matrix.jsp.linalg.cho_factor(phimat)

                return orfcf, phicf
            factors.params = argmap
        else:
            invprior, factors = None, None
    else:
        def priorfunc(params):
            phis = [prior(f, df, *[params[arg] for arg in argmap]) for prior, argmap in zip(priors, argmaps)]

            return sum(jnp.block([[jnp.make2d(val * phi) for val in row] for row in orfmat])
                       for phi, orfmat in zip(phis, orfmats))
        priorfunc.params = sorted(set.union(*[set(argmap) for argmap in argmaps]))
        priorfunc.type = jax.Array

        invprior, factors = None, None

    gp = matrix.GlobalVariableGP(matrix.NoiseMatrix12D_var(priorfunc), fmats)
    gp.Phi_inv, gp.factors = invprior, factors

    gp.index = {f'{psr.name}_{name}_coefficients({len(f)})':
                slice(len(f)*i, len(f)*(i+1)) for i, psr in enumerate(psrs)}
    gp.pos = [psr.pos for psr in psrs]
    gp.name = [psr.name for psr in psrs]

    if means is not None:
        margspec = inspect.getfullargspec(means)
        margs = margspec.args + [arg for arg in margspec.kwonlyargs if arg not in margspec.kwonlydefaults]

        # parameters carried by the pulsar objects (e.g., pos), should be at the beginning of function
        psrpars = [{arg: getattr(psr, arg) for arg in margspec.args if hasattr(psrs[0], arg) and arg not in exclude}
                   for psr in psrs]

        # other means parameters, either common or pulsar-specific
        margmaps = [{arg: f'{meansname}_{arg}' if (f'{meansname}_{arg}' in common or arg in common) else f'{psr.name}_{meansname}_{arg}'
                     for arg in margs if not hasattr(psr, arg) and arg not in exclude} for psr in psrs]

        def meanfunc(params):
            return jnp.concatenate([means(f, df, *psrpar.values(), **{arg: params[argname] for arg, argname in margmap.items()})
                                    for psrpar, margmap in zip(psrpars, margmaps)])
        meanfunc.params = sorted(set.union(*[set(margmap.values()) for margmap in margmaps]))

        gp.means = meanfunc

    return gp

makegp_fourier_global = makeglobalgp_fourier


# epoch-averaged covariance matrix from covfunc(t1, t2, *args)

def epochavgbasis(psr, components, T=None, dt=1.0):
    bins = quantize(psr.toas, dt)
    Umat = np.vstack([bins == i for i in range(bins.max() + 1)]).T.astype('d')
    t_avg = psr.toas @ Umat / Umat.sum(axis=0)

    return t_avg, None, Umat

def cov2cov(covfunc):
    argspec = inspect.getfullargspec(covfunc)
    arglist = argspec.args

    if arglist[0] == 't1' and arglist[1] == 't2':
        def covmat(f, df, *args):
            return covfunc(f, f, *args)
    elif arglist[0] == 'tau':
        def covmat(f, df, *args):
            return covfunc(jnp.abs(f[:, jnp.newaxis] - f[jnp.newaxis, :]), *args)
    else:
        raise ValueError('cov2avg() must take a covariance function with arguments t1, t2 or tau.')

    covmat.__signature__ = inspect.signature(covfunc)
    covmat.type = jax.Array

    return covmat

def makegp_avgcov(psr, prior, epochavgbasis=epochavgbasis, common=[], name='avgcovGP'):
    # assume prior(t1, t2, *args) or prior(tau, *args) returns a covariance matrix
    return makegp_fourier(psr, cov2cov(prior), components=0, T=1.0, fourierbasis=epochavgbasis,
                          common=common, exclude=['t1', 't2', 'tau'], name=name)

def makecommongp_avgcov(psrs, prior, epochavgbasis=epochavgbasis, common=[], vector=False, name='avgcovCommonGP'):
    return makecommongp_fourier(psr, cov2cov(prior), components=0, T=1.0, fourierbasis=epochavgbasis,
                                common=common, exclude=['t1', 't2', 'tau'], name=name)

def makeglobalgp_avgcov(psrs, prior, epochavgbasis=epochavgbasis, common=[], vector=False, name='avgcovCommonGP'):
    return makeglobalgp_fourier(psr, cov2cov(prior), components=0, T=1.0, fourierbasis=epochavgbasis,
                                exclude=['t1', 't2', 'tau'], name=name)


# time-interpolated covariance matrix from FFT

def timeinterpbasis(psr, components, T=None, start_time=None):
    if start_time is None:
        start_time = np.min(psr.toas)
    else:
        if start_time > np.min(psr.toas):
            raise ValueError('Coarse time basis start must be earlier than earliest TOA.')

    if T is None:
        T = getspan(psr)

    t_fine = psr.toas
    t_coarse = np.linspace(start_time, start_time + T, components)
    dt_coarse = t_coarse[1] - t_coarse[0]

    idx = np.arange(len(t_fine))
    idy = np.searchsorted(t_coarse, t_fine)
    idy[idy == 0] = 1

    Bmat = np.zeros((len(t_fine), len(t_coarse)), 'd')

    Bmat[idx, idy] = (t_fine - t_coarse[idy - 1]) / dt_coarse
    Bmat[idx, idy - 1] = (t_coarse[idy] - t_fine) / dt_coarse

    return t_coarse, dt_coarse, Bmat

def make_timeinterpbasis(start_time=None, order=1):
    def timeinterpbasis(psr, components, T=None):
        t0 = start_time if start_time is not None else np.min(psr.toas)
        if t0 > np.min(psr.toas):
            raise ValueError('Coarse time basis start must be earlier than earliest TOA.')

        if T is None:
            T = getspan(psr)

        t_fine = psr.toas
        t_coarse = np.linspace(t0, t0 + T, components)
        dt_coarse = t_coarse[1] - t_coarse[0]

        Bmat = si.interp1d(t_coarse, np.identity(components), kind=order)(t_fine).T

        return t_coarse, dt_coarse, Bmat

    return timeinterpbasis

def make_timeinterpbasis_chromatic(start_time=None, order=1, fref=1400.0, alpha=None):
    """Build a chromatic time-interpolation basis with a variable chromatic index.

    Time-domain analogue of :func:`fourierbasis_chrom` used by the FFT-covariance
    GPs. The returned basis yields a callable ``Bmat_func(alpha)`` that scales the
    achromatic :func:`make_timeinterpbasis` basis by ``(fref / psr.freqs) ** alpha``,
    with ``alpha`` a free parameter. Used by :func:`makegp_fftcov_chrom`.
    """
    timeinterpbasis_achrom = make_timeinterpbasis(start_time=start_time, order=order)

    def timeinterpbasis_chrom(psr, nmodes, T):
        t_coarse, dt_coarse, Bmat = timeinterpbasis_achrom(psr, nmodes, T)
        scale = (fref / psr.freqs)
        if alpha is None:
            def Bmat_func(alpha):
                return (scale[:, None]**alpha) * Bmat
        else:
            Bmat_func = (scale[:, None]**alpha) * Bmat
        return t_coarse, dt_coarse, Bmat_func

    return timeinterpbasis_chrom

def make_timeinterpbasis_band(start_time=None, order=1, fref=1400.0, scale=None):
    """Time-interpolation band basis parametrised by ``(fcenter, log10_bw)``.

    Used by the FFT-covariance GPs: it applies the smooth, RMS-normalised
    :func:`_band_envelope` to the achromatic :func:`make_timeinterpbasis` basis. Used
    by :func:`makegp_fftcov_band`.
    """
    timeinterpbasis_achrom = make_timeinterpbasis(start_time=start_time, order=order)

    def timeinterpbasis_band(psr, nmodes, T):
        t_coarse, dt_coarse, Bmat = timeinterpbasis_achrom(psr, nmodes, T)
        def Bmat_func(fcenter, log10_bw):
            return Bmat * _band_envelope(psr, fcenter, log10_bw, scale)[:, None]
        return t_coarse, dt_coarse, Bmat_func

    return timeinterpbasis_band

def make_timeinterpbasis_band_alpha(start_time=None, order=1, fref=1400.0, scale=None):
    """Chromatic time-interpolation band basis parametrised by ``(fcenter, log10_bw, alpha)``.

    As :func:`make_timeinterpbasis_band`, additionally scaling by
    ``(fref / psr.freqs) ** alpha`` for a variable chromatic index ``alpha``. Used by
    :func:`makegp_fftcov_band_alpha`.
    """
    timeinterpbasis_achrom = make_timeinterpbasis(start_time=start_time, order=order)

    def timeinterpbasis_band_alpha(psr, nmodes, T):
        t_coarse, dt_coarse, Bmat = timeinterpbasis_achrom(psr, nmodes, T)
        fnorm = (fref / psr.freqs)
        def Bmat_func(fcenter, log10_bw, alpha):
            return fnorm[:, None]**alpha * Bmat * _band_envelope(psr, fcenter, log10_bw, scale)[:, None]
        return t_coarse, dt_coarse, Bmat_func

    return timeinterpbasis_band_alpha

def make_timeinterpbasis_dm(start_time=None, order=1, fref=1400.0):
    """Build a DM time-interpolation basis (fixed chromatic index alpha = 2).

    Time-domain analogue of :func:`make_fourierbasis_dm` used by the FFT-covariance
    GPs: it scales the achromatic :func:`make_timeinterpbasis` basis by the
    cold-plasma dispersion factor ``(fref / psr.freqs) ** 2``. Used by
    :func:`makegp_fftcov_dm`.
    """
    timeinterpbasis_achrom = make_timeinterpbasis(start_time=start_time, order=order)

    def timeinterpbasis_dm(psr, nmodes, T):
        t_coarse, dt_coarse, Bmat = timeinterpbasis_achrom(psr, nmodes, T)
        scale = (fref / psr.freqs) ** 2
        return t_coarse, dt_coarse, scale[:, None] * Bmat

    return timeinterpbasis_dm

def make_dmtimeinterpbasis(alpha=2.0, tndm=False, start_time=None, order=1):
    warnings.warn("make_dmtimeinterpbasis is deprecated; use make_timeinterpbasis_dm "
                  "(alpha=2 DM) or make_timeinterpbasis_chromatic (variable alpha) instead.",
                  DeprecationWarning, stacklevel=2)
    basis = make_timeinterpbasis(start_time, order)

    def dmbasis(psr, components, T=None, fref=1400.0):
        t_coarse, dt_coarse, Bmat = basis(psr, components, T)

        if tndm:
            Dm = (fref / psr.freqs) ** alpha * np.sqrt(12.0) * np.pi / 1400.0 / 1400.0 / 2.41e-4
        else:
            Dm = (fref / psr.freqs) ** alpha

        return t_coarse, dt_coarse, Bmat * Dm[:, None]

    return dmbasis

def make_timeinterpbasis_solar(start_time=None, order=1):
    timeinterpbasis_achrom = make_timeinterpbasis(start_time=start_time, order=order)
    from .solar import theta_impact, dm_solar

    def timeinterpbasis_solar(psr, nmodes, T):
        t_coarse, dt_coarse, Bmat = timeinterpbasis_achrom(psr, nmodes, T)
        theta, R_earth, _, _ = theta_impact(psr)
        dm_sol_wind = dm_solar(1.0, theta, R_earth)
        dt_DM = dm_sol_wind * 4.148808e3 / (psr.freqs**2)
        return t_coarse, dt_coarse, dt_DM[:, None] * Bmat
    return timeinterpbasis_solar

# Relative white floor added to every sampled PSD bin in psd2cov, as a fraction of the
# PSD peak. The Toeplitz covariance psd2cov builds has cond(Phi) ~ (components/2)**gamma,
# and NoiseMatrix2D_var.make_inv forms Phi^-1 explicitly, so for the ~500-knot bases a
# 20-yr baseline produces at 30-day cadence the condition number crosses 1/eps64 near
# gamma = 6.5: Phi loses positive definiteness by gamma = 7 and its inverse returns NaN.
# At 1e-10 the floor holds cond(Phi) at ~4e9 independently of gamma while perturbing Phi
# by 4e-9 in Frobenius norm, i.e. ~1e-7 of the integrated variance -- ten decades below
# anything the data constrain. Override per call with nugget=, or set to 0 to disable.
PSD_NUGGET = 1e-10

def psd2cov(psdfunc, components, T, oversample=4, fmax_factor=1, cutoff=4, nugget=None):
    if not (isinstance(oversample, int) and isinstance(fmax_factor, int) and isinstance(cutoff, int)):
        raise ValueError('psd2cov: oversample, fmax_factor and cutoff must be integers.')

    if components % 2 == 0:
        raise ValueError('psd2cov: number of components must be odd.')

    nugget = PSD_NUGGET if nugget is None else nugget

    scaled_components = (components - 1) * fmax_factor + 1
    n_freqs = int((scaled_components - 1) / 2 * oversample + 1)
    fmax = (scaled_components - 1) / T / 2
    freqs = np.linspace(0, fmax, n_freqs)
    df = 1 / T / oversample

    if cutoff is not None:
        i_cutoff = int(np.ceil(oversample / cutoff))
        fs, zs = matrix.jnparray(freqs[i_cutoff:]), jnp.zeros(i_cutoff)
    else:
        fs = matrix.jnparray(freqs)

    def covmat(*args):
        psd = psdfunc(fs, 1.0, *args[2:])

        if nugget:
            # Additive, not jnp.maximum: a hard floor changes how many bins are clipped as
            # gamma varies, which puts a kink in the gradient. Applied before the cutoff
            # bins are prepended so those stay exactly zero. jnp.max is smooth here because
            # a decreasing power law always peaks in the lowest retained bin.
            psd = psd + nugget * jnp.max(psd)

        if cutoff is not None:
            psd = jnp.concatenate([zs, psd])

        fullpsd = jnp.concatenate((psd, psd[-2:0:-1]))
        Cfreq = jnp.fft.ifft(fullpsd, norm='backward')
        Ctau = Cfreq.real * len(fullpsd) * df / 2

        return matrix.jsp.linalg.toeplitz(Ctau[:scaled_components:fmax_factor])
    covmat.__signature__ = inspect.signature(psdfunc)
    covmat.type = jax.Array

    return covmat

def makegp_fftcov(psr, prior, components, T=None, t0=None, order=1, oversample=4, fmax_factor=1, cutoff=4, fourierbasis=None, common=[], name='fftcovGP'):
    T = getspan(psr) if T is None else T
    return makegp_fourier(psr, psd2cov(prior, components, T, oversample, fmax_factor, cutoff), components, T=T,
                          fourierbasis=(make_timeinterpbasis(start_time=t0, order=order) if fourierbasis is None else fourierbasis),
                          common=common, name=name)

def makegp_fftcov_dm(psr, prior, components, T=None, t0=None, order=1, oversample=4, fmax_factor=1, cutoff=4, common=[], name='dm_gp', fref=1400.0):
    """FFT-covariance (time-domain) GP for DM noise (fixed chromatic index alpha = 2).

    DM counterpart of :func:`makegp_fftcov`: the achromatic time-interpolation basis
    is replaced by :func:`make_timeinterpbasis_dm`, scaling each row by the cold-plasma
    dispersion factor ``(fref / psr.freqs) ** 2``. ``prior`` is a power-spectral-density
    function (e.g. :func:`powerlaw`) that is converted to a time-domain covariance via
    :func:`psd2cov`. For a free chromatic index use :func:`makegp_fftcov_chrom`.
    """
    T = getspan(psr) if T is None else T
    return makegp_fourier(psr, psd2cov(prior, components, T, oversample, fmax_factor, cutoff),
                          components, T=T, fourierbasis=make_timeinterpbasis_dm(start_time=t0, order=order, fref=fref), common=common, name=name)

def makegp_fftcov_chrom(psr, prior, components, T=None, t0=None, order=1, oversample=4, fmax_factor=1, cutoff=4, common=[], name='chrom_gp', fref=1400.0, alpha=None):
    """FFT-covariance (time-domain) GP for chromatic noise with a variable index.

    Chromatic counterpart of :func:`makegp_fftcov`: the achromatic time-interpolation
    basis is replaced by :func:`make_timeinterpbasis_chromatic`, scaling each row by
    ``(fref / psr.freqs) ** alpha`` with the chromatic index ``alpha`` a free parameter.
    ``prior`` is a power-spectral-density function (e.g. :func:`powerlaw`) converted to a
    time-domain covariance via :func:`psd2cov`. For the alpha = 2 (DM) case use
    :func:`makegp_fftcov_dm`.
    """
    T = getspan(psr) if T is None else T
    return makegp_fourier(psr, psd2cov(prior, components, T, oversample, fmax_factor, cutoff),
                          components, T=T, fourierbasis=make_timeinterpbasis_chromatic(start_time=t0, order=order, fref=fref, alpha=alpha), common=common, name=name)

def makegp_fftcov_band(psr, prior, components, T=None, t0=None, order=1, oversample=4, fmax_factor=1, cutoff=4, common=[], name='band_gp', fref=1400.0, scale=None):
    """FFT-covariance (time-domain) band-limited GP parametrised by ``(fcenter, log10_bw)``.

    Uses the smooth, RMS-normalised band basis :func:`make_timeinterpbasis_band`, so
    the band edges cannot invert (no ``fhigh < flow`` dead zone) and the GP amplitude is
    decoupled from the bandwidth. ``scale`` sets the sigmoid roll-off in MHz (default:
    10% of bandwidth).
    """
    T = getspan(psr) if T is None else T
    return makegp_fourier(psr, psd2cov(prior, components, T, oversample, fmax_factor, cutoff),
                          components, T=T, fourierbasis=make_timeinterpbasis_band(start_time=t0, order=order, fref=fref, scale=scale), common=common, name=name)

def makegp_fftcov_band_alpha(psr, prior, components, T=None, t0=None, order=1, oversample=4, fmax_factor=1, cutoff=4, common=[], name='bandalpha_gp', fref=1400.0, scale=None):
    """FFT-covariance (time-domain) chromatic band-limited GP parametrised by ``(fcenter, log10_bw, alpha)``.

    As :func:`makegp_fftcov_band`, additionally fitting a variable chromatic index
    ``alpha`` via :func:`make_timeinterpbasis_band_alpha`.
    """
    T = getspan(psr) if T is None else T
    return makegp_fourier(psr, psd2cov(prior, components, T, oversample, fmax_factor, cutoff),
                          components, T=T, fourierbasis=make_timeinterpbasis_band_alpha(start_time=t0, order=order, fref=fref, scale=scale), common=common, name=name)

def makegp_fftcov_solar(psr, prior, components, T=None, t0=None, order=1, oversample=4, fmax_factor=1, cutoff=4, common=[], name='fftcovGP_solar'):
    T = getspan(psr) if T is None else T
    return makegp_fourier(psr, psd2cov(prior, components, T, oversample, fmax_factor, cutoff),
                          components, T=T, fourierbasis=make_timeinterpbasis_solar(start_time=t0, order=order), common=common, name=name)

def makecommongp_fftcov(psrs, prior, components, T, t0=None, order=1, oversample=4, fmax_factor=1, cutoff=4, fourierbasis=None, common=[], vector=False, name='fftcovCommonGP'):
    return makecommongp_fourier(psrs, psd2cov(prior, components, T, oversample, fmax_factor, cutoff), components, T,
                                fourierbasis=(make_timeinterpbasis(start_time=t0, order=order) if fourierbasis is None else fourierbasis),
                                common=common, vector=vector, name=name)

def makeglobalgp_fftcov(psrs, prior, orf, components, T, t0, order=1, oversample=4, fmax_factor=1, cutoff=4, fourierbasis=None, name='fftcovGlobalGP'):
    return makeglobalgp_fourier(psrs, psd2cov(prior, components, T, oversample, fmax_factor, cutoff), orf, components, T,
                                fourierbasis=(make_timeinterpbasis(start_time=t0, order=order) if fourierbasis is None else fourierbasis),
                                name=name)


# time-interpolated covariance matrix from time-domain

def makegp_intcov(psr, prior, components, T=None, timeinterpbasis=timeinterpbasis, common=[], name='intcovGP'):
    T = getspan(psr) if T is None else T
    return makegp_fourier(psr, cov2cov(prior),
                          components, T, fourierbasis=timeinterpbasis, common=common, exclude=['t1', 't2', 'tau'], name=name)

def makecommongp_intcov(psr, prior, components, T, timeinterpbasis=timeinterpbasis, common=[], name='intcovCommonGP'):
    return makecommongp_fourier(psr, cov2cov(prior),
                                components, T, fourierbasis=timeinterpbasis, common=common, exclude=['t1', 't2', 'tau'], name=name)

def makeglobalgp_intcov(psr, prior, orf, components, T, timeinterpbasis=timeinterpbasis, common=[], name='intcovGlobalGP'):
    return makeglobalgp_fourier(psr, cov2cov(prior), orf,
                                components, T, fourierbasis=timeinterpbasis, exclude=['t1', 't2', 'tau'], name=name)


# single powerlaws
def powerlaw(f, df, log10_A, gamma):
    return (10.0**(2.0 * log10_A)) / 12.0 / jnp.pi**2 * const.fyr ** (gamma - 3.0) * f ** (-gamma) * df

def powerlaw_gwb(log10_A=None):
    """Return a powerlaw spectral model with gamma fixed to 13/3.

    If ``log10_A`` is None, the returned model takes ``log10_A`` as a sampled
    parameter. Otherwise ``log10_A`` is used in as a fixed constant.
    """
    gamma = 13.0 / 3.0   # 4.33 isotropic GWB

    if log10_A is None:
        def powerlaw_model(f, df, log10_A):
            return 10.0 ** (2.0 * log10_A) / 12.0 / jnp.pi**2* const.fyr ** (gamma - 3.0) * f ** (-gamma) * df
    else:
        def powerlaw_model(f, df):
            return 10.0 ** (2.0 * float(log10_A)) / 12.0 / jnp.pi**2 * const.fyr ** (gamma - 3.0) * f ** (-gamma) * df

    return powerlaw_model

def brokenpowerlaw(f, df, log10_A, gamma, log10_fb):
    kappa = 0.1 # smoothness of transition

    return (10.0**(2.0 * log10_A)) / 12.0 / jnp.pi**2 * const.fyr ** (gamma - 3.0) * f ** (-gamma) * df * \
        (1.0 + (f / 10.0**log10_fb) ** (1.0 / kappa)) ** (kappa * gamma)

def turnover(f, df, log10_A, gamma, log10_fc):
    """Power law whose spectrum flattens below a corner frequency.

    P(f) = A^2 / (12 pi^2) fyr^(gamma-3) f^-gamma df / [1 + (fc/f)^2]^(gamma/2)

    log10_A and gamma describe the high-frequency power law, exactly as in
    :func:`powerlaw`, and the spectrum flattens to a constant below fc. Equivalently
    P(f) = P0 / [1 + (f/fc)^2]^(gamma/2) with a plateau
    P0 = A^2 / (12 pi^2) fyr^(gamma-3) fc^-gamma df, which is the same model written
    with the plateau as the amplitude rather than the power law. The flat
    low-frequency limit leaves the integrated variance finite.

    f:        Fourier frequencies, Hz
    df:       frequency spacing, Hz
    log10_A:  log10 amplitude of the high-frequency power law at 1/yr
    gamma:    high-frequency spectral index
    log10_fc: log10 corner frequency, Hz
    """
    return powerlaw(f, df, log10_A, gamma) / (1.0 + (10.0**log10_fc / f)**2.0)**(gamma / 2.0)

# Components whose power-law GP can take a low-frequency turnover, and the GP name each
# carries in the mpta and ppta models. Common processes are not offered here.
TURNOVER_COMPONENTS = {'red': 'red_noise', 'red2': 'red_noise2',
                       'dm': 'dm_gp', 'chrom': 'chrom_gp'}


def turnover_set(turnover):
    """Normalise a turnover argument -- None, a name, or a sequence -- to a set."""
    if not turnover:
        return frozenset()
    names = (turnover,) if isinstance(turnover, str) else tuple(turnover)
    unknown = set(names) - set(TURNOVER_COMPONENTS)
    if unknown:
        raise ValueError(f"turnover: unknown component(s) {sorted(unknown)}; known are "
                         f"{sorted(TURNOVER_COMPONENTS)}.")
    return frozenset(names)


def turnover_psd(component, components):
    """:func:`turnover` for a component named in components, else :func:`powerlaw`."""
    return turnover if component in components else powerlaw


def make_turnover(kappa=2.0, beta='flat'):
    """Return a power law with a low-frequency turnover.

    P(f) = A^2 / (12 pi^2) fyr^(gamma-3) f^-gamma df / [1 + (fc/f)^kappa]^(2 beta)

    The spectral index is -gamma above fc and -gamma + 2 kappa beta below it. The
    defaults tie beta to gamma so the index below the corner is zero, which is the
    corner-frequency form :func:`turnover` implements; kappa = 2 then makes the
    turnover factor [1 + (f/fc)^2]^(-gamma/2). Setting kappa or beta to None samples
    it instead, giving the general form, whose integrated variance is finite only for
    2 kappa beta > gamma - 1.

    kappa: turnover sharpness. A number fixes it, None samples it.
    beta:  turnover depth. 'flat' ties it to gamma / (2 kappa), which flattens the
           spectrum below the corner whatever kappa is; a number fixes it; None
           samples it.
    """
    if isinstance(beta, str) and beta != 'flat':
        raise ValueError(f"make_turnover: beta must be 'flat', a number or None, "
                         f"got {beta!r}.")
    flat = isinstance(beta, str)
    kappa_c = None if kappa is None else float(kappa)
    beta_c = None if (flat or beta is None) else float(beta)

    def psd(f, df, log10_A, gamma, log10_fc, kappa, beta):
        twobeta = gamma / kappa if flat else 2.0 * beta
        return (powerlaw(f, df, log10_A, gamma) /
                (1.0 + (10.0**log10_fc / f)**kappa)**twobeta)

    if kappa is None and beta is None:
        def turnover_model(f, df, log10_A, gamma, log10_fc, kappa, beta):
            return psd(f, df, log10_A, gamma, log10_fc, kappa, beta)
    elif kappa is None:
        def turnover_model(f, df, log10_A, gamma, log10_fc, kappa):
            return psd(f, df, log10_A, gamma, log10_fc, kappa, beta_c)
    elif beta is None:
        def turnover_model(f, df, log10_A, gamma, log10_fc, beta):
            return psd(f, df, log10_A, gamma, log10_fc, kappa_c, beta)
    else:
        def turnover_model(f, df, log10_A, gamma, log10_fc):
            return psd(f, df, log10_A, gamma, log10_fc, kappa_c, beta_c)

    return turnover_model

def freespectrum(f, df, log10_rho: typing.Sequence):
    return jnp.repeat(10.0**(2.0 * log10_rho), 2)


def make_combined_crn(components, irn_psd, crn_psd, crn_prefix: typing.Optional[str] = 'crn_'):
    """
    Combine an intrinsic red noise PSD and a common red noise PSD into a
    single PSD function that shares the same Fourier basis.

    The intrinsic red noise PSD is evaluated over the full frequency basis,
    while the common red noise PSD is added only to the first
    ``2 * components`` frequency bins (sine and cosine for each component).

    Parameters
    ----------
    components : int
        Number of shared Fourier frequency components used by the CRN model.
        This determines how many low-frequency bins of the intrinsic basis
        receive the CRN contribution (specifically, the first
        ``2 * components`` entries, corresponding to sine/cosine pairs).
        This is *not* the same as the ``components`` argument passed to
        ``makegp_fourier`` — that controls the total number of Fourier
        components in the basis for the GP (and may be larger, since the
        intrinsic noise can extend to higher frequencies than the CRN).
    irn_psd : callable
        PSD function for the intrinsic red noise. Must accept ``(f, df, ...)``
        and return a PSD array over the full basis.
    crn_psd : callable
        PSD function for the common red noise. Must accept ``(f, df, ...)``
        and return a PSD array. Will only be called on the first
        ``2 * components`` frequency bins.
    crn_prefix : str or None
        Prefix applied to CRN parameter names that overlap with IRN names.
        For example, if both PSDs have ``log10_A`` and ``crn_prefix='crn_'``,
        the combined function will have ``log10_A`` (IRN) and
        ``crn_log10_A`` (CRN) as separate parameters.
        If None, overlapping names are shared (both PSDs receive the same
        value), which is valid when you intentionally want tied parameters.

    Returns
    -------
    combined : callable
        A PSD function whose signature is the union of ``irn_psd`` and
        ``crn_psd`` signatures (with CRN overlaps prefixed). Compatible
        with ``makegp_fourier``: argument names are inspectable via
        ``getfullargspec``, and ``typing.Sequence`` annotations are
        preserved for parameter expansion.
    crn_params : list of str
        The parameter names (as they appear in ``combined``'s signature)
        that belong to the CRN PSD. Pass these directly as the ``common``
        argument to ``makegp_fourier`` or ``makecommongp_fourier`` so that
        the CRN parameters are shared across pulsars rather than given
        per-pulsar names.

        Example::

            combined, crn_params = make_combined_crn(14, ds.powerlaw, ds.powerlaw)
            gp = makegp_fourier(psr, combined, components=30, common=crn_params)
    """
    from discovery import matrix
    irn_spec = inspect.getfullargspec(irn_psd)
    crn_spec = inspect.getfullargspec(crn_psd)

    shared = {'f', 'df'}
    irn_names = [a for a in irn_spec.args if a not in shared]
    crn_names = [a for a in crn_spec.args if a not in shared]

    # Rename overlapping CRN params
    irn_set = set(irn_names)
    crn_rename = {}  # original_name -> merged_name
    for a in crn_names:
        if a in irn_set and crn_prefix is not None:
            crn_rename[a] = crn_prefix + a
        else:
            crn_rename[a] = a

    # Build merged argument list: f, df, irn params, then (renamed) crn params
    merged_args = ['f', 'df']
    seen = set(shared)
    for arg in irn_names:
        if arg not in seen:
            merged_args.append(arg)
            seen.add(arg)
    for arg in crn_names:
        renamed = crn_rename[arg]
        if renamed not in seen:
            merged_args.append(renamed)
            seen.add(renamed)

    # Merge annotations (applying rename to CRN annotations)
    annotations = {}
    if irn_spec.annotations:
        annotations.update({k: v for k, v in irn_spec.annotations.items()
                            if k not in shared})
    if crn_spec.annotations:
        for k, v in crn_spec.annotations.items():
            if k not in shared:
                annotations[crn_rename.get(k, k)] = v

    def _impl(f, df, kw):
        irn_kw = {k: kw[k] for k in irn_names}
        crn_kw = {k: kw[crn_rename[k]] for k in crn_names}
        if matrix.jnp == jnp:
            phi = irn_psd(f, df, **irn_kw)
            phi = phi.at[:2 * components].add(
                crn_psd(f[:2 * components], df[:2 * components], **crn_kw)
            )
        else:
            phi = irn_psd(f, df, **irn_kw)
            phi[:2 * components] += crn_psd(
                f[:2 * components], df[:2 * components], **crn_kw
            )
        return phi

    # Dynamically build a function with the correct inspectable signature
    param_args = merged_args[2:]
    args_str = ', '.join(merged_args)
    kwargs_dict = '{' + ', '.join(f"'{a}': {a}" for a in param_args) + '}'
    func_code = f"def combined({args_str}): return _impl(f, df, {kwargs_dict})"
    ns = {'_impl': _impl}
    exec(func_code, ns)
    combined = ns['combined']
    combined.__annotations__ = annotations

    # Deduplicated list of CRN param names as they appear in the combined signature
    crn_params = list(dict.fromkeys(crn_rename[k] for k in crn_names))

    return combined, crn_params



# combined red_noise + crn

# this is a factory because it needs to specify a different number of components for the CRN
# note that the preferred way to fix gamma is for the user to use matrix.partial directly
def makepowerlaw_crn(components, crn_gamma='variable'):
    if matrix.jnp == jnp:
        def powerlaw_crn(f, df, log10_A, gamma, crn_log10_A, crn_gamma):
            phi = (10.0**(2.0 * log10_A)) / 12.0 / jnp.pi**2 * const.fyr ** (gamma - 3.0) * f ** (-gamma) * df
            phi = phi.at[:2*components].add((10.0**(2.0 * crn_log10_A)) / 12.0 / jnp.pi**2 *
                                            const.fyr ** (crn_gamma - 3.0) * f[:2*components] ** (-crn_gamma) * df[:2*components])
            return phi
    elif matrix.jnp == np:
        def powerlaw_crn(f, df, log10_A, gamma, crn_log10_A, crn_gamma):
            phi = (10.0**(2.0 * log10_A)) / 12.0 / np.pi**2 * const.fyr ** (gamma - 3.0) * f ** (-gamma) * df
            phi[:2*components] += ((10.0**(2.0 * crn_log10_A)) / 12.0 / np.pi**2 *
                                   const.fyr ** (crn_gamma - 3.0) * f[:2*components] ** (-crn_gamma) * df[:2*components])
            return phi

    if crn_gamma != 'variable':
        return matrix.partial(powerlaw_crn, crn_gamma=crn_gamma)
    else:
        return powerlaw_crn

def powerlaw_brokencrn(f, df, log10_A, gamma, crn_log10_A, crn_gamma, crn_log10_fb):
    kappa = 0.1 # smoothness of transition

    phi = (10.0**(2.0 * log10_A)) / 12.0 / jnp.pi**2 * const.fyr ** (gamma - 3.0) * f ** (-gamma) * df
    return phi + (10.0**(2.0 * crn_log10_A)) / 12.0 / jnp.pi**2 * const.fyr ** (crn_gamma - 3.0) * f ** (-crn_gamma) * df * \
        (1 + (f / 10**crn_log10_fb) ** (1 / kappa)) ** (kappa * crn_gamma)

def brokenpowerlaw_brokencrn(f, df, log10_A, gamma, log10_fb, crn_log10_A, crn_gamma, crn_log10_fb):
    kappa = 0.1 # smoothness of transition

    phi = (10.0**(2.0 * log10_A)) / 12.0 / jnp.pi**2 * const.fyr ** (gamma - 3.0) * f ** (-gamma) * df * \
        (1 + (f / 10**log10_fb) ** (1 / kappa)) ** (kappa * gamma)
    return phi + (10.0**(2.0 * crn_log10_A)) / 12.0 / jnp.pi**2 * const.fyr ** (crn_gamma - 3.0) * f ** (-crn_gamma) * df * \
        (1 + (f / 10**crn_log10_fb) ** (1 / kappa)) ** (kappa * crn_gamma)

def makefreespectrum_crn(components):
    if matrix.jnp == jnp:
        def freespectrum_crn(f, df, log10_rho: typing.Sequence, crn_log10_rho: typing.Sequence):
            phi = jnp.repeat(10.0**(2.0 * log10_rho), 2)
            phi = phi.at[:2*components].add(jnp.repeat(10.0**(2.0 * crn_log10_rho), 2))
            return phi
    elif matrix.jnp == np:
        def freespectrum_crn(f, df, log10_rho: typing.Sequence, crn_log10_rho: typing.Sequence):
            phi = jnp.repeat(10.0**(2.0 * log10_rho), 2)
            phi[:2*components] += jnp.repeat(10.0**(2.0 * crn_log10_rho), 2)
            return phi

    return freespectrum_crn


# ORFs: OK as numpy functions

def uncorrelated_orf(pos1, pos2):
    return 1.0 if np.all(pos1 == pos2) else 0.0

def hd_orf(pos1, pos2):
    if np.all(pos1 == pos2):
        return 1.0
    else:
        omc2 = (1.0 - np.dot(pos1, pos2)) / 2.0
        return 1.5 * omc2 * np.log(omc2) - 0.25 * omc2 + 0.5

def monopole_orf(pos1, pos2):
    if np.all(pos1 == pos2):
        # conditioning trick from enterprise
        return 1.0 + 1.0e-6
    else:
        return 1.0

def dipole_orf(pos1, pos2):
    if np.all(pos1 == pos2):
        return 1.0 + 1.0e-6
    else:
        return np.dot(pos1, pos2)


def makedelay(psr, delay, components=None, common=[], name='delay'):
    argspec = inspect.getfullargspec(delay)
    args = argspec.args + [arg for arg in argspec.kwonlyargs if arg not in argspec.kwonlydefaults]

    argmap = {arg: (arg if arg in common else f'{name}_{arg}' if f'{name}_{arg}' in common else f'{psr.name}_{name}_{arg}') +
                   (f'({components})' if (argspec.annotations.get(arg) == typing.Sequence and components is not None) else '')
              for arg in args if not hasattr(psr, arg)}

    psrpars = {arg: matrix.jnparray(getattr(psr, arg)) for arg in args if hasattr(psr, arg)}

    def delayfunc(params):
        return delay(**psrpars, **{arg: params[argname] for arg,argname in argmap.items()})
    delayfunc.params = sorted(argmap.values())

    return delayfunc

# use with makedelay to set residuals dynamically from arrays
def getresiduals(y):
    return -y

def make_phaseinterpbasis_orbital_dm(fref=1400.0, order=1):
    """Phase-interpolation basis for orbital DM GP.

    Coarse grid is evenly spaced in orbital phase [0, 2*pi).
    Periodic boundary handling by padding knots on each side.
    """

    def phaseinterpbasis_orbital_dm(psr, components, T=None):
        binphase = (2 * np.pi / psr.pb) * (psr.toas - psr.tasc)
        binphase = binphase % (2 * np.pi)

        # Coarse grid: components knots over [0, 2*pi), excluding endpoint
        phi_coarse = np.linspace(0, 2 * np.pi, components, endpoint=False)
        dphi_coarse = phi_coarse[1] - phi_coarse[0]

        # Periodic padding: copy a few knots on each side for interpolation
        n_pad = max(2, order + 1)
        phi_padded = np.concatenate([
            phi_coarse[-n_pad:] - 2 * np.pi,
            phi_coarse,
            phi_coarse[:n_pad] + 2 * np.pi
        ])

        identity = np.eye(components)
        identity_padded = np.vstack([
            identity[-n_pad:, :],
            identity,
            identity[:n_pad, :]
        ])

        Bmat = si.interp1d(phi_padded, identity_padded, kind=order, axis=0)(binphase)

        # Dispersive frequency scaling
        scale = (fref / psr.freqs) ** 2

        return phi_coarse, dphi_coarse, scale[:, None] * Bmat

    return phaseinterpbasis_orbital_dm


def makegp_fftcov_orbital_dm(psr, prior, components, order=1,
                              oversample=3, fmax_factor=1, cutoff=1,
                              common=[], name='fftcovGP_orbital_dm', fref=1400.0):
    """Orbital-phase GP with FFT covariance for DM variations."""
    # Use T=1 so that frequencies from psd2cov are simply k = 1, 2, 3, ...
    # (harmonic numbers). The powerlaw then evaluates as A^2 * k^(-gamma),
    # which is a natural spectral model for orbital harmonics.
    T_phase = 1.0
    if components % 2 == 0:
        components += 1

    return makegp_fourier(psr,
                          psd2cov(prior, components, T_phase, oversample, fmax_factor, cutoff),
                          components, T=T_phase,
                          fourierbasis=make_phaseinterpbasis_orbital_dm(fref=fref, order=order),
                          common=common, name=name)


"""Time-domain models and kernels implemented below. Under testing"""

#time-domain covariance functions
# Time-domain kernels matching NANOGrav 12.5yr CNM paper (Hazboun+ 2025).
#
# Both kernels operate on tau given in SECONDS (matching psr.toas units),
# while the kernel hyperparameters are in physically natural log10 units:
#   log10_sigma  -- log10 of amplitude, in the units of the GP coefficients: for
#                   solar.makegp_timedomain_solar_dm the basis carries the delay per
#                   unit density, so sigma is an electron density in cm^-3
#   log10_ell    -- log10 of correlation timescale in DAYS
#   log10_Gamma  -- log10 of periodicity weight (dimensionless, QP only)
#   log10_p      -- log10 of period in YEARS (QP only)
# These match the priors in Table A1 of the paper.
def squared_exponential(tau, log10_sigma, log10_ell):
    """Squared-exponential kernel (Eq. 4 of Hazboun+ 2025).

    k_SE(tau) = sigma^2 * exp(-tau^2 / (2*ell^2))
              + (sigma/500)^2 * delta(tau)

    The (sigma/500)^2 diagonal regulariser stabilises the inversion of phi
    for large ell, exactly as in the paper.
    """
    sigma  = 10.0 ** log10_sigma
    ell_s  = (10.0 ** log10_ell) * 86400.0   # days -> seconds

    smooth = (sigma ** 2) * jnp.exp(-0.5 * (tau / ell_s) ** 2)

    # Kronecker delta on tau == 0 (tau is |t_k - t_l| so only diagonal hits 0).
    diag   = (sigma / 500.0) ** 2 * (tau == 0.0)

    return smooth + diag


def quasi_periodic(tau, log10_sigma, log10_ell, log10_Gamma, log10_p):
    """Quasi-periodic kernel (Eq. 6 of Hazboun+ 2025).

    k_QP(tau) = (sigma/500)^2 * delta(tau)
              + sigma^2 * exp(-tau^2 / (2*ell^2))
                       * exp(-Gamma * sin^2(pi*tau/p))

    Reduces to k_SE in the limit Gamma -> 0.
    """
    sigma  = 10.0 ** log10_sigma
    ell_s  = (10.0 ** log10_ell) * 86400.0   # days -> seconds
    Gamma  = 10.0 ** log10_Gamma
    p_s    = (10.0 ** log10_p) * const.yr     # years -> seconds

    se_part  = jnp.exp(-0.5 * (tau / ell_s) ** 2)
    per_part = jnp.exp(-Gamma * jnp.sin(jnp.pi * tau / p_s) ** 2)

    smooth = (sigma ** 2) * se_part * per_part
    diag   = (sigma / 500.0) ** 2 * (tau == 0.0)

    return smooth + diag

def make_quasi_periodic(log10_p=None):
    """Return a :func:`quasi_periodic` kernel, optionally with the period held fixed.

    log10_p: log10 period in years. None samples it; a number fixes it, and the
             returned kernel no longer takes log10_p, so the GP does not carry that
             parameter at all.
    """
    if log10_p is None:
        return quasi_periodic

    lp = float(log10_p)

    def kernel(tau, log10_sigma, log10_ell, log10_Gamma):
        return quasi_periodic(tau, log10_sigma, log10_ell, log10_Gamma, lp)

    return kernel


# Generic time-domain GP  (achromatic / DM / chromatic; NO freq covariance)
def makegp_timedomain(psr, covariance, dt=14 * 86400.0, fref=1400.0,
                      chromatic_index=0, common=[], name='timedomain_gp'):
    """Time-domain GP with linear-interpolation temporal basis.

    The kernel is purely temporal: ``k = covariance(tau_t, ...)``. Frequency
    enters only as a multiplicative basis weight, *not* as a kernel coordinate.

    chromatic_index : 0   -> achromatic (red-noise-like in time)
                      2   -> DM        (basis weighted by K_DM / nu^2)
                      >0  -> chromatic (basis weighted by (fref/nu)^alpha)
    """
    argspec = inspect.getfullargspec(covariance)
    argmap = [(arg if arg in common
               else f'{name}_{arg}' if f'{name}_{arg}' in common
               else f'{psr.name}_{name}_{arg}')
              for arg in argspec.args if arg not in ('tau', 'tau_t')]

    if chromatic_index == 0:
        weight = np.ones_like(psr.freqs)
    elif chromatic_index == 2:
        weight = 4.148808e3 / (psr.freqs ** 2)
    else:
        weight = (fref / np.asarray(psr.freqs)) ** chromatic_index

    bins = quantize(psr.toas, dt)
    Umat = np.vstack([bins == i for i in range(bins.max() + 1)]).T.astype('d')
    Umat = Umat * weight[:, None]
    toas = psr.toas @ Umat / np.maximum(Umat.sum(axis=0), 1e-30)

    tau = jnp.abs(toas[:, None] - toas[None, :])

    def getphi(params):
        return covariance(tau, *[params[arg] for arg in argmap])
    getphi.params = argmap

    gp = matrix.VariableGP(matrix.NoiseMatrix2D_var(getphi), Umat)
    n = Umat.shape[1]
    gp.index = {f'{psr.name}_{name}_coefficients({n})': slice(0, n)}
    gp.name, gp.pos, gp.gpname, gp.gpcommon = psr.name, psr.pos, name, common
    return gp


# Finite-scintle-effect kernel (time x frequency)
def kernel_finite_scintle(tau_t, tau_nu,
                          log10_sigma, log10_t_d, log10_nu_d,
                          t_max=86400.0):
    """Diffractive / finite-scintle-effect kernel on (tau_t, tau_nu).

    k = sigma^2 * exp(-(tau_t / t_d)^(5/3)) * exp(-|tau_nu| / nu_d)

    Multiplied by a hard window ``tau_t < t_max`` so that cross-epoch
    correlation is identically zero (a single ionised cloud refreshes between
    observing days; FSE does not persist beyond ~1 day).

    sigma  : rms delay amplitude (seconds)
    t_d    : diffractive timescale (seconds; typically minutes)
    nu_d   : decorrelation bandwidth (MHz; typically <10 MHz at L-band)
    t_max  : hard cross-epoch cutoff (seconds; default 1 day)
    """
    sigma = 10.0 ** log10_sigma
    t_d   = 10.0 ** log10_t_d
    nu_d  = 10.0 ** log10_nu_d

    # 5/3 exponent: tiny epsilon prevents 0**(5/3) gradient pathology
    time_part = jnp.exp(-((tau_t + 1e-30) / t_d) ** (5.0 / 3.0))
    freq_part = jnp.exp(-jnp.abs(tau_nu) / nu_d)
    window    = (tau_t < t_max).astype(tau_t.dtype)

    diag = (sigma / 500.0) ** 2 * (tau_t == 0.0) * (tau_nu == 0.0)
    return sigma ** 2 * time_part * freq_part * window + diag


# Chromatic time-domain GP with joint (time, freq) bins
def kernel_qp_time_se_freq(tau_t, tau_nu,
                           log10_sigma, log10_ell_t, log10_p,
                           log10_Gamma, log10_ell_nu):
    """Default kernel for refractive chromatic GP: QP in time, SE in freq.

    Time:   squared-exponential * sin^2-periodic
            (months-scale correlation, ~1 yr period)
    Freq:   squared-exponential
            (slow ISM decorrelation across the band)
    """
    sigma   = 10.0 ** log10_sigma
    ell_t   = (10.0 ** log10_ell_t) * 86400.0     # days  -> seconds
    p_t     = (10.0 ** log10_p)     * const.yr    # years -> seconds
    Gamma   = 10.0 ** log10_Gamma
    ell_nu  = 10.0 ** log10_ell_nu                # MHz

    se_t  = jnp.exp(-0.5 * (tau_t  / ell_t) ** 2)
    per_t = jnp.exp(-Gamma * jnp.sin(jnp.pi * tau_t / p_t) ** 2)
    se_nu = jnp.exp(-0.5 * (tau_nu / ell_nu) ** 2)

    diag = (sigma / 500.0) ** 2 * (tau_t == 0.0) * (tau_nu == 0.0)
    return sigma ** 2 * se_t * per_t * se_nu + diag


def makegp_timedomain_chromatic(psr, covariance=kernel_qp_time_se_freq,
                                dt=14 * 86400.0, dnu=200.0, fref=1400.0,
                                chromatic_index=4, common=[],
                                name='chrom_timedomain_gp'):
    """Chromatic time-domain GP with *joint* (time, frequency) bins.

    Bins TOAs by (time, frequency) and evaluates a 2D kernel
    ``covariance(tau_t, tau_nu, ...)``. Default kernel is QP-in-time x SE-in-freq
    (refractive scintillation; months-to-year temporal scale, slow chromatic
    decorrelation across the band).

    Pass ``covariance=kernel_finite_scintle`` and small ``dt`` (e.g. 3600) to
    use the same builder for the diffractive / FSE component.

    dt   : time-bin width (seconds)
    dnu  : frequency-bin width (MHz)
    """
    argspec = inspect.getfullargspec(covariance)
    skip = {'tau_t', 'tau_nu', 't_max'}
    argmap = [(arg if arg in common
               else f'{name}_{arg}' if f'{name}_{arg}' in common
               else f'{psr.name}_{name}_{arg}')
              for arg in argspec.args
              if arg not in skip and argspec.defaults is None
              or arg not in skip]
    # (drop kw-only arguments with defaults from argmap; keep only sampled ones)
    argmap = [a for a in argmap if a not in skip]

    weight = (fref / np.asarray(psr.freqs)) ** chromatic_index

    # Joint (time, freq) binning
    bins_t  = quantize(psr.toas, dt)
    bins_nu = ((np.asarray(psr.freqs) - np.min(psr.freqs)) / dnu).astype(int)
    n_nu    = bins_nu.max() + 1
    joint   = bins_t * n_nu + bins_nu

    # Drop empty bins (sparse coverage in (epoch, sub-band) is the norm)
    used    = np.unique(joint)
    remap   = {b: i for i, b in enumerate(used)}
    joint_r = np.array([remap[b] for b in joint])
    n_bins  = len(used)

    Umat = np.zeros((len(psr.toas), n_bins))
    Umat[np.arange(len(psr.toas)), joint_r] = weight

    norms    = np.maximum(np.abs(Umat).sum(axis=0), 1e-30)
    toa_bin  = (psr.toas  @ np.abs(Umat)) / norms
    freq_bin = (psr.freqs @ np.abs(Umat)) / norms

    tau_t_j  = jnp.abs(toa_bin[:, None]  - toa_bin[None, :])
    tau_nu_j = jnp.abs(freq_bin[:, None] - freq_bin[None, :])

    def getphi(params):
        return covariance(tau_t_j, tau_nu_j,
                          *[params[arg] for arg in argmap])
    getphi.params = argmap

    gp = matrix.VariableGP(matrix.NoiseMatrix2D_var(getphi), Umat)
    gp.index = {f'{psr.name}_{name}_coefficients({n_bins})': slice(0, n_bins)}
    gp.name, gp.pos, gp.gpname, gp.gpcommon = psr.name, psr.pos, name, common
    return gp