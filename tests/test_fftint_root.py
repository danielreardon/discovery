"""Tests for the square-root representation of the fftint GP prior (``signals.FFTCOV_METHOD = 'root'``).

The root path builds each single-pulsar fftint GP as the basis ``Bmat @ R.T`` with a unit prior, where ``R``
(K x K) is the triangular QR factor of ``(C * sqrt(psd)).T`` and ``f, w, C`` come from :func:`signals.psd2factor`.

Covers:

- exactness of ``R.T @ R`` against psd2factor's ``C diag(psd) C.T`` and against ``psd2cov(nugget=0)``;
- the default likelihood against a stable float64 reference at posterior-like and steep rows, on the pulsar's own
  span and on a common span twice as long, and against the ``'factor'`` path;
- the root basis itself, evaluated with the reference's stable algebra;
- gradients against central differences of the reference;
- the legacy ``'inverse'`` path, method validation and parameter names;
- the J0437 new-mode row under the default, and the ``'inverse'`` path against the 45e6adb build, when the
  platypus capture is present.

The reference whitens by the white noise, projects the timing model out with an orthonormal QR, multiplies by the
full psd2factor factor (no QR compression), and takes the log-determinant and quadratic form from one Householder
QR of ``[[B, P y], [I, 0]]``. Tests marked ``integration`` are deselected by the default addopts.
"""

import functools
import hashlib
import pickle
import types
from pathlib import Path

import numpy as np
import pytest

import jax
jax.config.update('jax_enable_x64', True)

import discovery as ds
from discovery import signals


YR = 365.25 * 86400.0
AU_LIGHT_SEC = 499.004783836
BANDS = np.array([900.0, 1100.0, 1300.0, 1600.0])
TOAERR = 1.0e-7
LOG10_T2EQUAD = -8.0
LOGL_TOL = 1.0e-4
POSTERIOR_TOL = 1.0e-6

# (log10_A, gamma) of the GP under test
ROWS = {'posterior': (-13.3, 2.0),
        'steep65': (-12.0, 6.5),
        'steep7': (-12.0, 7.0),
        'corner': (-11.0, 7.0)}

# kind, knots, pulsar span (yr), cadence (days), knots of the simulated DM signal, GP span / pulsar span, and the
# knot-grid start before the first TOA as a fraction of the pulsar span (None: the first TOA)
CASES = {'dm145': dict(kind='dm', K=145, span=5.94, cadence=14.0, data_knots=145, Tfactor=1.0, shift=None),
         'red145': dict(kind='red', K=145, span=5.94, cadence=14.0, data_knots=145, Tfactor=1.0, shift=None),
         'dm331': dict(kind='dm', K=331, span=5.94, cadence=6.57, data_knots=331, Tfactor=1.0, shift=None),
         'dm509': dict(kind='dm', K=509, span=20.91, cadence=30.0, data_knots=509, Tfactor=1.0, shift=None),
         'dm549': dict(kind='dm', K=549, span=20.91, cadence=14.0, data_knots=549, Tfactor=1.0, shift=None),
         'dm1001': dict(kind='dm', K=1001, span=20.91, cadence=30.0, data_knots=509, Tfactor=1.0, shift=None),
         'dm289_Tc2Tp': dict(kind='dm', K=289, span=5.94, cadence=14.0, data_knots=145, Tfactor=2.0, shift=None),
         'dm289_Tc2Tp_centred': dict(kind='dm', K=289, span=5.94, cadence=14.0, data_knots=145, Tfactor=2.0,
                                     shift=0.5)}

HEAVY = {'dm509', 'dm549', 'dm1001'}

# (test, case, row) -> reason, for targets these tests miss on this tree
_WOODBURY = 'joint Woodbury removes the 1e40 timing model by subtraction'
XFAIL = {
    ('root_vs_reference', 'dm145', 'steep65'): _WOODBURY,
    ('root_vs_reference', 'dm145', 'steep7'): _WOODBURY,
    ('root_vs_reference', 'dm145', 'corner'): _WOODBURY,
    ('root_vs_reference', 'red145', 'steep65'): _WOODBURY,
    ('root_vs_reference', 'red145', 'steep7'): _WOODBURY,
    ('root_vs_reference', 'red145', 'corner'): _WOODBURY,
    ('root_vs_reference', 'dm289_Tc2Tp', 'steep65'): _WOODBURY,
    ('root_vs_reference', 'dm289_Tc2Tp', 'steep7'): _WOODBURY,
    ('root_vs_reference', 'dm289_Tc2Tp', 'corner'): _WOODBURY,
    ('root_vs_reference', 'dm289_Tc2Tp_centred', 'steep65'): _WOODBURY,
    ('root_vs_reference', 'dm289_Tc2Tp_centred', 'steep7'): _WOODBURY,
    ('root_vs_reference', 'dm289_Tc2Tp_centred', 'corner'): _WOODBURY,
    ('root_vs_factor', 'dm145', 'steep65'): _WOODBURY,
    ('root_vs_factor', 'dm145', 'steep7'): _WOODBURY,
    ('root_vs_factor', 'dm145', 'corner'): _WOODBURY,
    ('root_vs_factor', 'red145', 'steep65'): _WOODBURY,
    ('root_vs_factor', 'red145', 'steep7'): _WOODBURY,
    ('root_vs_factor', 'red145', 'corner'): _WOODBURY,
    ('root_vs_reference', 'dm509', 'steep65'): _WOODBURY,
    ('root_vs_reference', 'dm509', 'steep7'): _WOODBURY,
    ('root_vs_reference', 'dm509', 'corner'): _WOODBURY,
    ('root_vs_reference', 'dm1001', 'steep65'): _WOODBURY,
    ('root_vs_reference', 'dm1001', 'steep7'): _WOODBURY,
    ('root_vs_reference', 'dm1001', 'corner'): _WOODBURY,
    ('root_vs_factor', 'dm509', 'steep65'): _WOODBURY,
    ('root_vs_factor', 'dm509', 'steep7'): _WOODBURY,
    ('root_vs_factor', 'dm509', 'corner'): _WOODBURY,
    ('root_vs_factor', 'dm1001', 'steep65'): _WOODBURY,
    ('root_vs_factor', 'dm1001', 'steep7'): _WOODBURY,
    ('root_vs_factor', 'dm1001', 'corner'): _WOODBURY,
    ('grad', 'dm509', 'steep65'): _WOODBURY,
    ('grad', 'dm145', 'steep65'): _WOODBURY,
}

GPNAME = {'dm': 'dm_gp', 'red': 'fftcovGP', 'chrom': 'chrom_gp'}


def _param(case, row, test):
    marks = []
    if case in HEAVY:
        marks += [pytest.mark.integration, pytest.mark.slow]
    if (test, case, row) in XFAIL:
        marks.append(pytest.mark.xfail(strict=True, reason=XFAIL[(test, case, row)]))
    return pytest.param(case, row, id=f'{case}-{row}', marks=marks)


def _grid(test, cases, rows=tuple(ROWS)):
    return [_param(case, row, test) for case in cases for row in rows]


# --- synthetic pulsar, models and reference -------------------------------------------------------------------

def _factor(K, T, grid, log10_A, gamma):
    """psd2factor's full real factor ``C diag(sqrt(psd))`` (K x M) for a powerlaw."""
    f, w, C = signals.psd2factor(K, T, grid[0], 1, grid[1])
    return C * np.sqrt(np.asarray(signals.powerlaw(f, w, log10_A, gamma)))


@functools.lru_cache(maxsize=None)
def _pulsar(span_yr, cadence_days, data_knots, seed=0):
    """Pulsar with four sub-band TOAs per epoch over ``span_yr``, a quadratic spin, quadratic DM and annual
    timing model, a circular 1 AU Earth orbit for the solar-wind basis, and residuals of white noise plus one
    DM-GP draw at the posterior row."""
    rng = np.random.default_rng(seed)
    span = span_yr * YR
    nep = int(round(span / (cadence_days * 86400.0))) + 1
    tep = np.linspace(0.0, span, nep)
    tep[1:-1] += rng.uniform(-86400.0, 86400.0, nep - 2)
    toas = 58000.0 * 86400.0 + np.repeat(tep, len(BANDS))
    freqs = np.tile(BANDS, nep)
    n = len(toas)

    x = (toas - toas.mean()) / span
    nu = (freqs / 1400.0) ** -2
    phase = 2.0 * np.pi * (toas - toas[0]) / YR
    Mmat = np.column_stack([np.ones(n), x, x ** 2, nu, x * nu, x ** 2 * nu, np.sin(phase), np.cos(phase)])

    pos = np.array([0.6, 0.0, 0.8])
    planetssb = np.zeros((n, 3, 3))
    planetssb[:, 2, 0], planetssb[:, 2, 1] = AU_LIGHT_SEC * np.cos(phase), AU_LIGHT_SEC * np.sin(phase)
    psr = types.SimpleNamespace(name='SYN', toas=toas, freqs=freqs, toaerrs=np.full(n, TOAERR), Mmat=Mmat,
                                backend_flags=np.full(n, 'BE'), pos=pos, pos_t=np.tile(pos, (n, 1)),
                                planetssb=planetssb, sunssb=np.zeros((n, 3)), residuals=np.zeros(n))

    Bmat = np.asarray(signals.make_timeinterpbasis_dm()(psr, data_knots, span)[2])
    L = _factor(data_knots, span, (4, 4), *ROWS['posterior'])
    white = np.sqrt(TOAERR ** 2 + 10.0 ** (2 * LOG10_T2EQUAD)) * rng.standard_normal(n)
    psr.residuals = white + Bmat @ (L @ rng.standard_normal(L.shape[1]))
    return psr


def _noisedict(psr):
    return {f'{psr.name}_BE_efac': 1.0, f'{psr.name}_BE_log10_t2equad': LOG10_T2EQUAD}


def _span(case, psr):
    c = CASES[case]
    Tp = signals.getspan(psr)
    t0 = None if c['shift'] is None else psr.toas.min() - c['shift'] * Tp
    return c['Tfactor'] * Tp, t0


def _gp(psr, kind, K, T, t0, method, grid=(4, 4)):
    make = {'dm': signals.makegp_fftcov_dm, 'red': signals.makegp_fftcov}[kind]
    return make(psr, signals.powerlaw, K, T=T, t0=t0, oversample=grid[0], cutoff=grid[1], method=method)


def _basis(psr, kind, K, T, t0):
    make = {'dm': signals.make_timeinterpbasis_dm, 'red': signals.make_timeinterpbasis}[kind]
    return np.asarray(make(start_time=t0)(psr, K, T)[2])


@functools.lru_cache(maxsize=None)
def _model(case, method):
    """Production likelihood (white noise, SVD timing model, one fftint GP) for ``case`` under ``method``."""
    c = CASES[case]
    psr = _pulsar(c['span'], c['cadence'], c['data_knots'])
    T, t0 = _span(case, psr)
    noise = ds.makenoise_measurement(psr, _noisedict(psr))
    tm = ds.makegp_timing(psr, svd=True)
    gp = _gp(psr, c['kind'], c['K'], T, t0, method)
    m = ds.PulsarLikelihood([psr.residuals, noise, tm, gp])
    return types.SimpleNamespace(psr=psr, noise=noise, tm=tm, gp=gp, T=T, t0=t0, params=list(m.logL.params),
                                 logL=jax.jit(m.logL), grad=jax.jit(jax.grad(m.logL)))


def _params(case, log10_A, gamma):
    name = GPNAME[CASES[case]['kind']]
    return {f'SYN_{name}_log10_A': log10_A, f'SYN_{name}_gamma': gamma}


def _production(case, method, log10_A, gamma):
    mod = _model(case, method)
    p = _params(case, log10_A, gamma)
    assert sorted(mod.params) == sorted(p)
    return float(mod.logL(p))


def _stable_logl(y, N, T, Tprior, F, L=None):
    """Log-likelihood of ``y`` under ``N + T diag(Tprior) T.T + F L L.T F.T`` (``L`` = identity if None) for the large
    ``Tprior`` of an improper timing model, without the constant ``n/2 log 2 pi``.

    Whitens by ``N``, projects the timing model out with an orthonormal QR of ``N^-1/2 T``, and reads the
    log-determinant of ``I + B.T B`` and the quadratic form from the diagonal of the R factor of
    ``[[B, P y], [I, 0]]``, with ``B = P N^-1/2 F L``.
    """
    s = np.sqrt(N)
    Tw, Fw, yw = T / s[:, None], F / s[:, None], y / s
    Qt, Rt = np.linalg.qr(Tw)
    PFw = Fw - Qt @ (Qt.T @ Fw)
    Pyw = yw - Qt @ (Qt.T @ yw)
    B = PFw if L is None else PFw @ L
    k = B.shape[1]
    A = np.block([[B, Pyw[:, None]], [np.eye(k), np.zeros((k, 1))]])
    d = np.abs(np.diag(np.linalg.qr(A, mode='r')))
    logdet = (np.sum(np.log(N)) + np.sum(np.log(Tprior)) + 2.0 * np.sum(np.log(np.abs(np.diag(Rt))))
              + 2.0 * np.sum(np.log(d[:k])))
    return -0.5 * d[k] ** 2 - 0.5 * logdet


def _ref_inputs(case):
    mod = _model(case, 'root')
    return (mod.psr.residuals, np.asarray(mod.noise.N, dtype=np.float64), np.asarray(mod.tm.F, dtype=np.float64),
            np.asarray(mod.tm.Phi.N, dtype=np.float64))


@functools.lru_cache(maxsize=None)
def _reference(case, log10_A, gamma):
    """Stable reference logL for ``case`` with the full psd2factor factor."""
    c = CASES[case]
    mod = _model(case, 'root')
    F = _basis(mod.psr, c['kind'], c['K'], mod.T, mod.t0)
    return _stable_logl(*_ref_inputs(case), F, _factor(c['K'], mod.T, (4, 4), log10_A, gamma))


# --- 1. exactness of R.T R --------------------------------------------------------------------------------------

def _identity_basis(psr, components, T):
    return np.zeros(components), 0.0, np.eye(components)


def _root_R(K, T, grid, log10_A, gamma):
    """``R`` from make_rootbasis on an identity time basis, so that its basis is ``R.T``."""
    _, _, fmat = signals.make_rootbasis(_identity_basis, signals.powerlaw, grid[0], 1, grid[1])(None, K, T)
    return np.asarray(fmat(log10_A, gamma)).T


@pytest.mark.parametrize('grid', [(4, 4), (4, 1), (3, 1), (8, 8)], ids=lambda g: f'grid{g[0]}{g[1]}')
@pytest.mark.parametrize('span_yr', [5.94, 20.91])
@pytest.mark.parametrize('K', [145, 509, pytest.param(1001, marks=[pytest.mark.integration, pytest.mark.slow])])
def test_root_factor_reproduces_the_toeplitz_prior(K, span_yr, grid):
    """R.T @ R equals psd2factor's C diag(psd) C.T, and psd2cov's nugget-free Toeplitz prior entrywise."""
    T = span_yr * YR
    f, w, C = signals.psd2factor(K, T, grid[0], 1, grid[1])
    cov = signals.psd2cov(signals.powerlaw, K, T, grid[0], 1, grid[1], nugget=0)
    for gamma in (1.0, 3.0, 5.0, 6.5, 7.0):
        R = _root_R(K, T, grid, -14.0, gamma)
        assert R.shape == (K, K) and np.allclose(R, np.triu(R), rtol=0.0, atol=0.0)

        RtR = R.T @ R
        phi = (C * np.asarray(signals.powerlaw(f, w, -14.0, gamma))) @ C.T
        toeplitz = np.asarray(cov(None, None, -14.0, gamma))

        scale = np.max(np.abs(phi))
        assert np.max(np.abs(RtR - phi)) / scale <= 1e-12, gamma
        assert np.max(np.abs(RtR - toeplitz)) / scale <= 1e-12, gamma


# --- 2. likelihood accuracy -------------------------------------------------------------------------------------

LOGL_CASES = ['dm145', 'red145', 'dm509', 'dm1001']


@pytest.mark.parametrize('case, row', _grid('root_vs_reference', LOGL_CASES))
def test_root_logl_matches_reference(case, row):
    """The default (root) production logL agrees with the stable reference to LOGL_TOL."""
    assert signals.FFTCOV_METHOD == 'root'
    prod = _production(case, None, *ROWS[row])
    ref = _reference(case, *ROWS[row])
    assert np.isfinite(ref)
    assert abs(prod - ref) < LOGL_TOL, (prod, ref, prod - ref)


POSTERIOR_CASES = ['dm145', 'red145', 'dm331', 'dm509', 'dm549', 'dm1001']


@pytest.mark.parametrize('case, row', _grid('root_vs_reference', POSTERIOR_CASES, rows=('posterior',)))
def test_root_logl_matches_reference_at_the_posterior(case, row):
    """At the posterior-like row the default (root) logL agrees with the stable reference to POSTERIOR_TOL."""
    assert signals.FFTCOV_METHOD == 'root'
    prod = _production(case, None, *ROWS[row])
    ref = _reference(case, *ROWS[row])
    assert abs(prod - ref) < POSTERIOR_TOL, (prod, ref, prod - ref)


@pytest.mark.parametrize('case, row', _grid('root_basis_vs_reference', LOGL_CASES))
def test_root_basis_matches_reference(case, row):
    """The production root basis Bmat @ R.T, evaluated with the reference's stable algebra, agrees with the
    reference built from the full factor."""
    mod = _model(case, 'root')
    F = np.asarray(mod.gp.F(_params(case, *ROWS[row])), dtype=np.float64)
    assert F.shape[1] == CASES[case]['K']
    root = _stable_logl(*_ref_inputs(case), F)
    ref = _reference(case, *ROWS[row])
    assert abs(root - ref) < LOGL_TOL, (root, ref, root - ref)


@pytest.mark.parametrize('case, row', _grid('root_vs_factor', LOGL_CASES))
def test_root_logl_matches_factor(case, row):
    """The root and factor production logLs agree to LOGL_TOL."""
    root = _production(case, 'root', *ROWS[row])
    factor = _production(case, 'factor', *ROWS[row])
    assert abs(root - factor) < LOGL_TOL, (root, factor, root - factor)


# --- 3. gradients -----------------------------------------------------------------------------------------------

GRAD_STEP = 1.0e-4
GRAD_TOL = 1.0e-4


@pytest.mark.parametrize('case, row', [_param('dm145', 'posterior', 'grad'), _param('dm145', 'steep65', 'grad'),
                                       _param('dm509', 'steep65', 'grad')])
def test_root_gradient_matches_reference_differences(case, row):
    """jax.grad of the root logL agrees with central differences of the reference.

    Tolerance: max_i |g_i - g_fd_i| <= GRAD_TOL * max(||g_fd||_inf, 1). The difference error is estimated as
    |g_fd(h) - g_fd(2h)| and must be below a tenth of that tolerance.
    """
    log10_A, gamma = ROWS[row]
    mod = _model(case, 'root')
    p = _params(case, log10_A, gamma)
    g = mod.grad(p)
    keys = list(p)
    gj = np.array([float(g[k]) for k in keys])

    def fd(h):
        return np.array([(_reference(case, log10_A + h, gamma) - _reference(case, log10_A - h, gamma)) / (2 * h),
                         (_reference(case, log10_A, gamma + h) - _reference(case, log10_A, gamma - h)) / (2 * h)])

    g1, g2 = fd(GRAD_STEP), fd(2 * GRAD_STEP)
    tol = GRAD_TOL * max(np.max(np.abs(g1)), 1.0)
    assert np.max(np.abs(g1 - g2)) < tol / 10, ('finite differences unresolved', g1, g2)
    assert np.all(np.isfinite(gj)) and np.max(np.abs(gj - g1)) <= tol, (gj, g1, gj - g1)


def _stable_logl_jnp(y, N, T, Tprior, F):
    """:func:`_stable_logl` with ``L`` = identity, in jax.numpy so that it can be differentiated."""
    jnp = jax.numpy
    s = jnp.sqrt(N)
    Tw, Fw, yw = T / s[:, None], F / s[:, None], y / s
    Qt, Rt = jnp.linalg.qr(Tw)
    B = Fw - Qt @ (Qt.T @ Fw)
    Pyw = yw - Qt @ (Qt.T @ yw)
    k = B.shape[1]
    A = jnp.block([[B, Pyw[:, None]], [jnp.eye(k), jnp.zeros((k, 1))]])
    d = jnp.abs(jnp.diag(jnp.linalg.qr(A, mode='r')))
    logdet = (jnp.sum(jnp.log(N)) + jnp.sum(jnp.log(Tprior)) + 2.0 * jnp.sum(jnp.log(jnp.abs(jnp.diag(Rt))))
              + 2.0 * jnp.sum(jnp.log(d[:k])))
    return -0.5 * d[k] ** 2 - 0.5 * logdet


@pytest.mark.parametrize('case, row', [_param('dm145', 'posterior', 'grad_basis'),
                                       _param('dm145', 'steep65', 'grad_basis'),
                                       _param('dm509', 'steep65', 'grad_basis')])
def test_root_basis_gradient_matches_reference_differences(case, row):
    """jax.grad through the production root basis Bmat @ R.T, evaluated with the reference's stable algebra, agrees
    with central differences of the reference (tolerance as in the production-gradient test)."""
    log10_A, gamma = ROWS[row]
    mod = _model(case, 'root')
    y, N, T, Tprior = _ref_inputs(case)
    p = _params(case, log10_A, gamma)
    g = jax.grad(lambda q: _stable_logl_jnp(y, N, T, Tprior, mod.gp.F(q)))(p)
    gj = np.array([float(g[k]) for k in p])

    def fd(h):
        return np.array([(_reference(case, log10_A + h, gamma) - _reference(case, log10_A - h, gamma)) / (2 * h),
                         (_reference(case, log10_A, gamma + h) - _reference(case, log10_A, gamma - h)) / (2 * h)])

    g1, g2 = fd(GRAD_STEP), fd(2 * GRAD_STEP)
    tol = GRAD_TOL * max(np.max(np.abs(g1)), 1.0)
    assert np.max(np.abs(g1 - g2)) < tol / 10, ('finite differences unresolved', g1, g2)
    assert np.all(np.isfinite(gj)) and np.max(np.abs(gj - g1)) <= tol, (gj, g1, gj - g1)


# --- 4. common span ---------------------------------------------------------------------------------------------

SPAN_CASES = ['dm289_Tc2Tp', 'dm289_Tc2Tp_centred']


@pytest.mark.parametrize('case, row', _grid('root_vs_reference', SPAN_CASES))
def test_root_logl_matches_reference_on_common_span(case, row):
    """On a GP span twice the pulsar's, the default logL agrees with the stable reference to LOGL_TOL."""
    mod = _model(case, 'root')
    assert np.isclose(mod.T, 2.0 * signals.getspan(mod.psr), rtol=1e-15)
    prod = _production(case, None, *ROWS[row])
    ref = _reference(case, *ROWS[row])
    assert np.isfinite(ref)
    assert abs(prod - ref) < LOGL_TOL, (prod, ref, prod - ref)


def test_inverse_floor_lift_on_common_span(record_property):
    """Reports logL('inverse') - reference on the pulsar span and on twice it (numbers only)."""
    lines = []
    for case in ['dm145'] + SPAN_CASES:
        for row, (log10_A, gamma) in ROWS.items():
            ref = _reference(case, log10_A, gamma)
            assert np.isfinite(ref)
            lift = _production(case, 'inverse', log10_A, gamma) - ref
            record_property(f'lift_{case}_{row}', lift)
            lines.append(f'{case:22s} {row:10s} inverse - reference = {lift:+.6e} nat')
    print('\n' + '\n'.join(lines))


# --- 5. legacy path, validation and names -----------------------------------------------------------------------

def _legacy_pair(psr, kind, K):
    T = signals.getspan(psr)
    new = {'red': signals.makegp_fftcov, 'dm': signals.makegp_fftcov_dm,
           'chrom': signals.makegp_fftcov_chrom}[kind](psr, signals.powerlaw, K, method='inverse')
    basis = {'red': signals.make_timeinterpbasis, 'dm': signals.make_timeinterpbasis_dm,
             'chrom': signals.make_timeinterpbasis_chromatic}[kind]()
    old = signals.makegp_fourier(psr, signals.psd2cov(signals.powerlaw, K, T, 4, 1, 4), K, T=T,
                                 fourierbasis=basis, name=GPNAME[kind])
    return new, old


@pytest.mark.parametrize('kind', ['red', 'dm', 'chrom'])
def test_inverse_reproduces_psd2cov_bit_for_bit(kind):
    """method='inverse' builds the psd2cov GP exactly: same prior, basis and logL, bit for bit."""
    psr = _pulsar(5.94, 14.0, 145)
    new, old = _legacy_pair(psr, kind, 145)
    models = [ds.PulsarLikelihood([psr.residuals, ds.makenoise_measurement(psr, _noisedict(psr)),
                                   ds.makegp_timing(psr, svd=True), gp]) for gp in (new, old)]
    assert models[0].logL.params == models[1].logL.params
    logls = [jax.jit(m.logL) for m in models]

    for row in ('posterior', 'steep7'):
        p = {f'SYN_{GPNAME[kind]}_log10_A': ROWS[row][0], f'SYN_{GPNAME[kind]}_gamma': ROWS[row][1]}
        if kind == 'chrom':
            p[f'SYN_{GPNAME[kind]}_alpha'] = 4.0
        assert np.array_equal(np.asarray(new.Phi.getN(p)), np.asarray(old.Phi.getN(p)))
        Fn, Fo = ((new.F(p), old.F(p)) if callable(new.F) else (new.F, old.F))
        assert np.array_equal(np.asarray(Fn), np.asarray(Fo))
        a, b = (float(f(p)) for f in logls)
        assert a == b, (a, b)


def test_module_default_is_root_and_read_at_build(monkeypatch):
    """FFTCOV_METHOD defaults to 'root', and method=None reads it when the GP is built."""
    assert signals.FFTCOV_METHOD == 'root'
    psr = _pulsar(5.94, 14.0, 145)
    assert type(signals.makegp_fftcov_dm(psr, signals.powerlaw, 145).Phi).__name__ == 'NoiseMatrix1D_var'

    monkeypatch.setattr(signals, 'FFTCOV_METHOD', 'inverse')
    gp = signals.makegp_fftcov_dm(psr, signals.powerlaw, 145)
    assert type(gp.Phi).__name__ == 'NoiseMatrix2D_var'
    p = _params('dm145', *ROWS['posterior'])
    assert np.array_equal(np.asarray(gp.Phi.getN(p)),
                          np.asarray(signals.makegp_fftcov_dm(psr, signals.powerlaw, 145, method='inverse').Phi.getN(p)))


def test_unknown_method_raises(monkeypatch):
    """An unknown method string raises, whether passed per GP or set as the module default."""
    psr = _pulsar(5.94, 14.0, 145)
    with pytest.raises(ValueError, match='fftint method'):
        signals.makegp_fftcov(psr, signals.powerlaw, 145, method='qr')
    monkeypatch.setattr(signals, 'FFTCOV_METHOD', 'Root')
    with pytest.raises(ValueError, match='fftint method'):
        signals.makegp_fftcov_dm(psr, signals.powerlaw, 145)


VARIANTS = {'fftcov': signals.makegp_fftcov, 'dm': signals.makegp_fftcov_dm, 'chrom': signals.makegp_fftcov_chrom,
            'band': signals.makegp_fftcov_band, 'band_alpha': signals.makegp_fftcov_band_alpha,
            'solar': signals.makegp_fftcov_solar}

VARIANT_VALUES = {'log10_A': -13.3, 'gamma': 3.0, 'alpha': 4.0, 'fcenter': 1200.0, 'log10_bw': 2.5}
VARIANT_OVERRIDES = {'solar': {'log10_A': -4.0}}


@pytest.mark.parametrize('variant', [
    pytest.param(v, marks=[pytest.mark.xfail(strict=True, reason=(
        'root - factor = 4.3e-8 nat: factor is 4.3e-8 off the stable reference, root 4.1e-11 (log10_A = -4); '
        + _WOODBURY))] if v == 'solar' else []) for v in VARIANTS])
def test_methods_share_parameter_names_and_values(variant):
    """Every fftint variant has the same likelihood parameters under root, factor and inverse, and root and factor
    give the same logL at a moderate point (the basis and PSD arguments reach the right slots)."""
    psr = _pulsar(5.94, 14.0, 145)
    noise = ds.makenoise_measurement(psr, _noisedict(psr))
    values = {**VARIANT_VALUES, **VARIANT_OVERRIDES.get(variant, {})}
    models = {m: ds.PulsarLikelihood([psr.residuals, noise, ds.makegp_timing(psr, svd=True),
                                      VARIANTS[variant](psr, signals.powerlaw, 9, method=m)])
              for m in ('root', 'factor', 'inverse')}
    names = {m: sorted(model.logL.params) for m, model in models.items()}
    assert names['root'] == names['factor'] == names['inverse'] and names['root']

    p = {}
    for name in names['root']:
        suffix = [s for s in values if name.endswith('_' + s)]
        assert len(suffix) == 1, name
        p[name] = values[suffix[0]]
    root, factor, inverse = (float(models[m].logL(p)) for m in ('root', 'factor', 'inverse'))
    null = float(ds.PulsarLikelihood([psr.residuals, noise, ds.makegp_timing(psr, svd=True)]).logL({}))
    assert np.isfinite(root) and abs(root - null) > 1.0, (root, null)
    assert abs(root - factor) < 1e-8, (root, factor)
    assert abs(root - inverse) < 1e-4, (root, inverse)


# --- 6. J0437 integration ---------------------------------------------------------------------------------------

J0437_CAPTURE = Path('/fred/oz002/dreardon/platypus/outputs/reweight_j0437/runs/'
                     'J0437-4715-newport_ns_freeA_ecorr1modes_nosw/capture.pkl')
J0437_CAPTURE_SHA256 = 'c5bb83e0ba60823fca9cedef8a95752d62fc6c31d904a104f5e00f86d5a8a22c'

J0437_OLD_ROW = {'J0437-4715_KAT_MKBF_efac': 1.4336240500348427,
                 'J0437-4715_KAT_MKBF_log10_ecorr': -6.671811669222361,
                 'J0437-4715_KAT_MKBF_log10_tnequad': -7.578110032434317,
                 'J0437-4715_bkgrnd_log10_A': -17.505434297957443,
                 'J0437-4715_chrom_gp_alpha': 6.653297439406475,
                 'J0437-4715_chrom_gp_gamma': 0.1742542531865614,
                 'J0437-4715_chrom_gp_log10_A': -14.985540600163468,
                 'J0437-4715_dm_gp_gamma': 0.8904158693634958,
                 'J0437-4715_dm_gp_log10_A': -13.262246657917053,
                 'J0437-4715_red_noise_gamma': 2.469793225600816,
                 'J0437-4715_red_noise_log10_A': -16.583043118261017}

J0437_NEW_ROW = {'J0437-4715_KAT_MKBF_efac': 1.437258213412012,
                 'J0437-4715_KAT_MKBF_log10_ecorr': -6.640061363031213,
                 'J0437-4715_KAT_MKBF_log10_tnequad': -7.599528168054915,
                 'J0437-4715_bkgrnd_log10_A': -17.15338886450012,
                 'J0437-4715_chrom_gp_alpha': 7.4479145235999065,
                 'J0437-4715_chrom_gp_gamma': 0.28568614167123885,
                 'J0437-4715_chrom_gp_log10_A': -15.101550224988864,
                 'J0437-4715_dm_gp_gamma': 6.758718709151903,
                 'J0437-4715_dm_gp_log10_A': -13.39178062734969,
                 'J0437-4715_red_noise_gamma': 1.6211331060901144,
                 'J0437-4715_red_noise_log10_A': -17.881004474229787}


# logL of the 45e6adb build at J0437_OLD_ROW and J0437_NEW_ROW
J0437_45E6ADB = (79340.76444697368, 79341.4494835078)


def _j0437_logl():
    if not J0437_CAPTURE.exists():
        pytest.skip('platypus J0437 capture not present')
    blob = J0437_CAPTURE.read_bytes()
    assert hashlib.sha256(blob).hexdigest() == J0437_CAPTURE_SHA256
    cap = pickle.loads(blob)

    from discovery.models import mpta
    model = mpta.single_pulsar_noise(cap['psr'], **cap['kwargs'])
    assert sorted(model.logL.params) == sorted(J0437_OLD_ROW) == sorted(J0437_NEW_ROW)
    logl = jax.jit(model.logL)
    old, new = float(logl(J0437_OLD_ROW)), float(logl(J0437_NEW_ROW))
    assert np.isfinite(old) and np.isfinite(new)
    return old, new


@pytest.mark.integration
@pytest.mark.slow
def test_j0437_new_mode_row_falls_below_the_old_row():
    """With the default method, the J0437 1-mode new-mode row sits more than 600 nat below the old posterior row."""
    assert signals.FFTCOV_METHOD == 'root'
    old, new = _j0437_logl()
    delta = new - old
    print(f'\nJ0437 root: logL(new) - logL(old) = {delta:+.6f} nat')
    assert delta < -600.0, delta


@pytest.mark.integration
@pytest.mark.slow
def test_j0437_inverse_path_reproduces_45e6adb(monkeypatch):
    """With FFTCOV_METHOD = 'inverse' and the default nugget_ref = 'max', the J0437 rows reproduce the 45e6adb build."""
    monkeypatch.setattr(signals, 'FFTCOV_METHOD', 'inverse')
    assert signals.PSD_NUGGET == 1e-10 and signals.PSD_NUGGET_REF == 'max'
    old, new = _j0437_logl()
    assert abs(old - J0437_45E6ADB[0]) < POSTERIOR_TOL, old - J0437_45E6ADB[0]
    assert abs(new - J0437_45E6ADB[1]) < POSTERIOR_TOL, new - J0437_45E6ADB[1]
