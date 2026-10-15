"""Tests for WoodburyKernel_varFP, the pulsar kernel with fixed white noise and a GP basis that depends on parameters.

Each test builds the same model twice: with a chromatic Fourier GP whose index ``alpha`` is a parameter
(:func:`signals.fourierbasis_chrom`, giving WoodburyKernel_varFP) and with the index fixed in the basis
(:func:`signals.make_fourierbasis_chrom`, giving WoodburyKernel_varP), and checks that GlobalLikelihood, OS.os and
PulsarLikelihood's conditional, sample and clogL agree.
"""

import functools
import types

import numpy as np
import pytest

import jax

import discovery as ds
from discovery import matrix, signals


YR = 365.25 * 86400.0
BANDS = np.array([900.0, 1100.0, 1300.0, 1600.0])
POS = [np.array([0.6, 0.0, 0.8]), np.array([0.0, 0.6, 0.8]), np.array([-0.6, 0.0, -0.8])]
ALPHA = 4.0


@functools.lru_cache(maxsize=None)
def _pulsar(i):
    """Synthetic pulsar with four sub-band TOAs per 14-day epoch over 6 yr, a quadratic spin and DM timing model,
    and white-noise residuals."""
    rng = np.random.default_rng(i)
    tep = np.linspace(0.0, 6.0 * YR, 157)
    toas = 58000.0 * 86400.0 + np.repeat(tep, len(BANDS))
    freqs = np.tile(BANDS, len(tep))
    n = len(toas)
    x = (toas - toas.mean()) / (6.0 * YR)
    nu = (freqs / 1400.0) ** -2
    return types.SimpleNamespace(name=f'SYN{i}', toas=toas, freqs=freqs, toaerrs=np.full(n, 1e-7),
                                 Mmat=np.column_stack([np.ones(n), x, x ** 2, nu]), backend_flags=np.full(n, 'BE'),
                                 pos=POS[i], pos_t=np.tile(POS[i], (n, 1)), residuals=1e-7 * rng.standard_normal(n))


def _noise(psr, white):
    fixed = {f'{psr.name}_BE_efac': 1.0, f'{psr.name}_BE_log10_t2equad': -8.0}
    return ds.makenoise_measurement(psr, fixed if white == 'fixed' else {})


def _chrom(psr, T, variable, name='chrom_gp', common=()):
    basis = signals.fourierbasis_chrom if variable else signals.make_fourierbasis_chrom(alpha=ALPHA)
    return ds.makegp_fourier(psr, ds.powerlaw, 9, T=T, fourierbasis=basis, common=list(common), name=name)


def _values(names):
    values = {'efac': 1.0, 't2equad': -8.0, 'log10_A': -13.5, 'gamma': 3.0, 'alpha': ALPHA}
    p = {}
    for name in names:
        suffix = [s for s in values if name.endswith(s)]
        assert len(suffix) == 1, name
        p[name] = values[suffix[0]]
    return p


def _psls(white, variable, gw=False):
    psrs = [_pulsar(i) for i in range(len(POS))]
    T = signals.getspan(psrs)
    psls = []
    for psr in psrs:
        gps = [_chrom(psr, T, variable)]
        if gw:
            gps.append(_chrom(psr, T, variable, name='gw', common=('gw_log10_A', 'gw_gamma', 'gw_alpha')))
        psls.append(ds.PulsarLikelihood([psr.residuals, _noise(psr, white), ds.makegp_timing(psr, svd=True)] + gps))
    return psrs, T, psls


def test_variable_basis_with_fixed_white_noise_builds_varfp():
    """The variable-index model uses WoodburyKernel_varFP and the fixed-index model WoodburyKernel_varP."""
    _, _, var = _psls('fixed', True)
    _, _, fix = _psls('fixed', False)
    assert isinstance(var[0].N, matrix.WoodburyKernel_varFP)
    assert isinstance(fix[0].N, matrix.WoodburyKernel_varP)


def test_global_likelihood_matches_fixed_basis():
    """GlobalLikelihood (make_kernelterms) agrees between the variable and fixed bases."""
    out = []
    for variable in (True, False):
        psrs, T, psls = _psls('fixed', variable)
        logl = ds.GlobalLikelihood(psls, globalgp=ds.makeglobalgp_fourier(psrs, ds.powerlaw, ds.hd_orf, 5, T=T,
                                                                          name='gw')).logL
        out.append(float(logl(_values(logl.params))))
    assert np.isfinite(out[0]) and abs(out[0] - out[1]) < 1e-8 * abs(out[1]), out


@pytest.mark.parametrize('white', ['fixed', 'sampled'])
def test_optimal_statistic_matches_fixed_basis(white):
    """OS.os (make_kernelsolve, with the GW basis itself variable) agrees between the variable and fixed bases."""
    out = []
    for variable in (True, False):
        get_os = ds.OS(ds.GlobalLikelihood(_psls(white, variable, gw=True)[2])).os
        res = get_os(_values(get_os.params))
        out.append(np.array([float(res[k]) for k in ('os', 'os_sigma', 'snr')]))
    assert np.all(np.isfinite(out[0])) and np.allclose(out[0], out[1], rtol=1e-8, atol=0.0), out


@pytest.mark.parametrize('white, timing', [('fixed', True), ('fixed', False), ('sampled', True)],
                         ids=['fixed-timing', 'fixed-notiming', 'sampled-timing'])
def test_pulsar_likelihood_methods_match_fixed_basis(white, timing):
    """PulsarLikelihood logL, conditional and sample_conditional (make_kernelsolve_simple without a timing model),
    sample and clogL agree between the variable and fixed bases (same keys), with the white noise fixed or sampled."""
    psr = _pulsar(0)
    T = signals.getspan(psr)
    c = np.random.default_rng(1).standard_normal(9 * 2) * 1e-7
    res = []
    for variable in (True, False):
        m = ds.PulsarLikelihood([psr.residuals, _noise(psr, white)] +
                                ([ds.makegp_timing(psr, svd=True)] if timing else []) + [_chrom(psr, T, variable)])
        if white == 'fixed':
            assert isinstance(m.N, matrix.WoodburyKernel_varFP if variable else matrix.WoodburyKernel_varP)
        p = _values(m.logL.params)
        mu, _ = m.conditional(p)
        _, draw = m.sample(jax.random.PRNGKey(0), p)
        _, cdraw = m.sample_conditional(jax.random.PRNGKey(1), p)
        cdraw = np.concatenate([np.asarray(cdraw[k]) for k in sorted(cdraw)])
        cname = [k for k in m.N.index if 'chrom_gp' in k][0]
        cp = {**p, **{k: (c if k == cname else np.zeros(sl.stop - sl.start)) for k, sl in m.N.index.items()}}
        res.append((float(m.logL(p)), np.asarray(mu), np.asarray(draw), float(m.clogL(cp)), cdraw))
    (l1, mu1, d1, c1, s1), (l2, mu2, d2, c2, s2) = res
    assert abs(l1 - l2) < 1e-8 * abs(l2), (l1, l2)
    assert np.allclose(mu1, mu2, rtol=1e-8, atol=1e-8 * np.max(np.abs(mu2)))
    assert np.allclose(d1, d2, rtol=1e-10, atol=0.0)
    assert abs(c1 - c2) < 1e-8 * abs(c2), (c1, c2)
    assert np.allclose(s1, s2, rtol=1e-6, atol=1e-6 * np.max(np.abs(s2)))
