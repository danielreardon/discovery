"""Tests for the square-root representation of the fftint GP prior (``signals.FFTCOV_METHOD = 'root'``).

The root path builds each single-pulsar fftint GP as the basis ``Bmat @ R.T`` with a unit prior, where ``R``
(K x K) is the triangular QR factor of ``(C * sqrt(w * psd)).T`` and ``f, w, C`` come from :func:`signals.psd2factor`.

Covers:

- exactness of ``R.T @ R`` against psd2factor's ``C diag(w * psd) C.T`` and against psd2cov, including the clipped
  range of the PSD;
- the default likelihood against a stable float64 reference at posterior-like and steep rows, on the pulsar's own
  span and on a common span twice as long;
- the root basis itself, evaluated with the reference's stable algebra;
- gradients against central differences of the reference;
- the ``'inverse'`` path, method validation, parameter names and fixed hyperparameters in ``noisedict``.

The reference whitens by the white noise, projects the timing model out with an orthonormal QR, multiplies by the
full psd2factor factor (no QR compression), and takes the log-determinant and quadratic form from one Householder
QR of ``[[B, P y], [I, 0]]``. Tests marked ``integration`` are deselected by the default addopts.
"""

import functools
import types

import numpy as np
import pytest

import jax
jax.config.update('jax_enable_x64', True)

import discovery as ds
from discovery import matrix, signals


YR = 365.25 * 86400.0
AU_LIGHT_SEC = 499.004783836
BANDS = np.array([900.0, 1100.0, 1300.0, 1600.0])
TOAERR = 1.0e-7
LOG10_T2EQUAD = -8.0
# |logL - reference| bounds (nat): production logL at the posterior-like row and at every row, and the root basis
# evaluated with the reference's algebra
POSTERIOR_TOL = 1.0e-6
STEEP_TOL = 1.0e-2
LOGL_TOL = 1.0e-4

# (log10_A, gamma) of the GP under test
ROWS = {'posterior': (-13.3, 2.0),
        'steep65': (-12.0, 6.5),
        'steep7': (-12.0, 7.0),
        'corner': (-11.0, 7.0),
        'quiet65': (-13.0, 6.5),
        'quiet7': (-13.0, 7.0)}

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

GPNAME = {'dm': 'dm_gp', 'red': 'fftcovGP', 'chrom': 'chrom_gp'}


def _param(case, row):
    marks = [pytest.mark.integration, pytest.mark.slow] if case in HEAVY else []
    return pytest.param(case, row, id=f'{case}-{row}', marks=marks)


def _grid(cases, rows=tuple(ROWS)):
    return [_param(case, row) for case in cases for row in rows]


# --- synthetic pulsar, models and reference -------------------------------------------------------------------

def _psd(f, w, log10_A, gamma):
    """``w * psd`` for a powerlaw, with the PSD evaluated and clipped as psd2cov does."""
    with np.errstate(divide='ignore'):
        return w * np.clip(np.asarray(signals.powerlaw(f, 1.0, log10_A, gamma)), *signals.PSD2COV_CLIP)


def _factor(K, T, grid, log10_A, gamma):
    """psd2factor's full real factor ``C diag(sqrt(w * psd))`` (K x M) for a powerlaw."""
    f, w, C = signals.psd2factor(K, T, grid[0], 1, grid[1])
    return C * np.sqrt(_psd(f, w, log10_A, gamma))


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
    _, _, fmat = signals.make_rootbasis(_identity_basis, signals.powerlaw, *grid)(None, K, T)
    return np.asarray(fmat(log10_A, gamma)).T


# (log10_A, gamma): the last two reach the upper and lower ends of PSD2COV_CLIP
EXACT_ROWS = [(-14.0, 1.0), (-14.0, 3.0), (-14.0, 5.0), (-14.0, 6.5), (-14.0, 7.0), (-11.0, 7.0), (-20.0, 1.0)]


# (oversample, fmax_factor, cutoff)
@pytest.mark.parametrize('grid', [(4, 1, 4), (4, 1, 1), (3, 1, 1), (8, 1, 8), (4, 1, None), (2, 2, 1)],
                         ids=lambda g: 'grid{}{}{}'.format(*g))
@pytest.mark.parametrize('span_yr', [5.94, 20.91])
@pytest.mark.parametrize('K', [145, 509, pytest.param(1001, marks=[pytest.mark.integration, pytest.mark.slow])])
def test_root_factor_reproduces_the_toeplitz_prior(K, span_yr, grid):
    """R.T @ R equals psd2factor's C diag(w * psd) C.T, and psd2cov's Toeplitz prior entrywise."""
    T = span_yr * YR
    f, w, C = signals.psd2factor(K, T, *grid)
    cov = signals.psd2cov(signals.powerlaw, K, T, *grid)
    for log10_A, gamma in EXACT_ROWS:
        R = _root_R(K, T, grid, log10_A, gamma)
        assert R.shape == (K, K) and np.allclose(R, np.triu(R), rtol=0.0, atol=0.0)

        RtR = R.T @ R
        phi = (C * _psd(f, w, log10_A, gamma)) @ C.T
        toeplitz = np.asarray(cov(None, None, log10_A, gamma))

        scale = np.max(np.abs(phi))
        assert np.max(np.abs(RtR - phi)) / scale <= 1e-12, (log10_A, gamma)
        assert np.max(np.abs(RtR - toeplitz)) / scale <= 1e-12, (log10_A, gamma)


def test_exact_rows_reach_both_ends_of_the_clip():
    """The last two EXACT_ROWS clip the PSD from above and below on the (4, 1, 4) grid."""
    f, _, _ = signals.psd2factor(145, 5.94 * YR, 4, 1, 4)
    lo, hi = signals.PSD2COV_CLIP
    assert np.max(np.asarray(signals.powerlaw(f, 1.0, *EXACT_ROWS[-2]))) > hi
    assert np.min(np.asarray(signals.powerlaw(f, 1.0, *EXACT_ROWS[-1]))) < lo


@pytest.mark.parametrize('oversample', [1, 4.0])
def test_psd2factor_rejects_bad_oversample(oversample):
    """psd2factor needs an integer oversample of at least 2."""
    with pytest.raises(ValueError, match='oversample'):
        signals.psd2factor(145, 5.94 * YR, oversample, 1, 1)


# --- 2. likelihood accuracy -------------------------------------------------------------------------------------

LOGL_CASES = ['dm145', 'red145', 'dm509', 'dm1001']


@pytest.mark.parametrize('case, row', _grid(LOGL_CASES))
def test_root_logl_matches_reference(case, row):
    """The default (root) production logL is finite and agrees with the stable reference to STEEP_TOL."""
    assert signals.FFTCOV_METHOD == 'root'
    prod = _production(case, None, *ROWS[row])
    ref = _reference(case, *ROWS[row])
    assert np.isfinite(ref) and np.isfinite(prod)
    assert abs(prod - ref) < STEEP_TOL, (prod, ref, prod - ref)


POSTERIOR_CASES = ['dm145', 'red145', 'dm331', 'dm509', 'dm549', 'dm1001']


@pytest.mark.parametrize('case, row', _grid(POSTERIOR_CASES, rows=('posterior',)))
def test_root_logl_matches_reference_at_the_posterior(case, row):
    """At the posterior-like row the default (root) logL agrees with the stable reference to POSTERIOR_TOL."""
    assert signals.FFTCOV_METHOD == 'root'
    prod = _production(case, None, *ROWS[row])
    ref = _reference(case, *ROWS[row])
    assert abs(prod - ref) < POSTERIOR_TOL, (prod, ref, prod - ref)


@pytest.mark.parametrize('case, row', _grid(LOGL_CASES))
def test_root_basis_matches_reference(case, row):
    """The production root basis Bmat @ R.T, evaluated with the reference's stable algebra, agrees with the
    reference built from the full factor."""
    mod = _model(case, 'root')
    F = np.asarray(mod.gp.F(_params(case, *ROWS[row])), dtype=np.float64)
    assert F.shape[1] == CASES[case]['K']
    root = _stable_logl(*_ref_inputs(case), F)
    ref = _reference(case, *ROWS[row])
    assert abs(root - ref) < LOGL_TOL, (root, ref, root - ref)


# --- 3. gradients -----------------------------------------------------------------------------------------------

GRAD_STEP = 1.0e-4
GRAD_TOL = 1.0e-4


@pytest.mark.parametrize('case, row', [_param('dm145', 'posterior'), _param('dm145', 'steep65'), _param('dm509', 'steep65')])
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


@pytest.mark.parametrize('case, row', [_param('dm145', 'posterior'), _param('dm145', 'steep65'), _param('dm509', 'steep65')])
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


@pytest.mark.parametrize('case, row', _grid(SPAN_CASES))
def test_root_logl_matches_reference_on_common_span(case, row):
    """On a GP span twice the pulsar's, the default logL is finite and agrees with the stable reference to STEEP_TOL,
    and to POSTERIOR_TOL at the posterior row."""
    mod = _model(case, 'root')
    assert np.isclose(mod.T, 2.0 * signals.getspan(mod.psr), rtol=1e-15)
    prod = _production(case, None, *ROWS[row])
    ref = _reference(case, *ROWS[row])
    assert np.isfinite(ref) and np.isfinite(prod)
    assert abs(prod - ref) < (POSTERIOR_TOL if row == 'posterior' else STEEP_TOL), (prod, ref, prod - ref)


# --- 5. the inverse method, validation and names ---------------------------------------------------------------

def _inverse_pair(psr, kind, K):
    T = signals.getspan(psr)
    built = {'red': signals.makegp_fftcov, 'dm': signals.makegp_fftcov_dm,
             'chrom': signals.makegp_fftcov_chrom}[kind](psr, signals.powerlaw, K, method='inverse')
    basis = {'red': signals.make_timeinterpbasis, 'dm': signals.make_timeinterpbasis_dm,
             'chrom': signals.make_timeinterpbasis_chromatic}[kind]()
    direct = signals.makegp_fourier(psr, signals.psd2cov(signals.powerlaw, K, T, 4, 1, 4), K, T=T,
                                    fourierbasis=basis, name=GPNAME[kind])
    return built, direct


@pytest.mark.parametrize('kind', ['red', 'dm', 'chrom'])
def test_inverse_reproduces_psd2cov_bit_for_bit(kind):
    """method='inverse' builds the psd2cov GP of makegp_fourier exactly: same prior, basis and logL, bit for bit."""
    psr = _pulsar(5.94, 14.0, 145)
    built, direct = _inverse_pair(psr, kind, 145)
    models = [ds.PulsarLikelihood([psr.residuals, ds.makenoise_measurement(psr, _noisedict(psr)),
                                   ds.makegp_timing(psr, svd=True), gp]) for gp in (built, direct)]
    assert models[0].logL.params == models[1].logL.params
    logls = [jax.jit(m.logL) for m in models]

    for row in ('posterior', 'steep7'):
        p = {f'SYN_{GPNAME[kind]}_log10_A': ROWS[row][0], f'SYN_{GPNAME[kind]}_gamma': ROWS[row][1]}
        if kind == 'chrom':
            p[f'SYN_{GPNAME[kind]}_alpha'] = 4.0
        assert np.array_equal(np.asarray(built.Phi.getN(p)), np.asarray(direct.Phi.getN(p)))
        Fn, Fo = ((built.F(p), direct.F(p)) if callable(built.F) else (built.F, direct.F))
        assert np.array_equal(np.asarray(Fn), np.asarray(Fo))
        a, b = (float(f(p)) for f in logls)
        assert np.array_equal([a], [b], equal_nan=True), (a, b)


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


VARIANTS = {'fftcov': signals.makegp_fftcov, 'dm': signals.makegp_fftcov_dm, 'chrom': signals.makegp_fftcov_chrom}

VARIANT_VALUES = {'log10_A': -13.3, 'gamma': 3.0, 'alpha': 4.0}


def _variant_values(names):
    """VARIANT_VALUES keyed by full parameter name, matched on the name's suffix."""
    p = {}
    for name in names:
        suffix = [s for s in VARIANT_VALUES if name.endswith('_' + s)]
        assert len(suffix) == 1, name
        p[name] = VARIANT_VALUES[suffix[0]]
    return p


@pytest.mark.parametrize('variant', list(VARIANTS))
def test_methods_share_parameter_names_and_values(variant):
    """Every fftint variant has the same likelihood parameters under root and inverse, and they give the same logL
    at a moderate point (the basis and PSD arguments reach the right slots)."""
    psr = _pulsar(5.94, 14.0, 145)
    noise = ds.makenoise_measurement(psr, _noisedict(psr))
    models = {m: ds.PulsarLikelihood([psr.residuals, noise, ds.makegp_timing(psr, svd=True),
                                      VARIANTS[variant](psr, signals.powerlaw, 9, method=m)])
              for m in ('root', 'inverse')}
    names = {m: sorted(model.logL.params) for m, model in models.items()}
    assert names['root'] == names['inverse'] and names['root']

    p = _variant_values(names['root'])
    root, inverse = (float(models[m].logL(p)) for m in ('root', 'inverse'))
    null = float(ds.PulsarLikelihood([psr.residuals, noise, ds.makegp_timing(psr, svd=True)]).logL({}))
    assert np.isfinite(root) and abs(root - null) > 1.0, (root, null)
    assert abs(root - inverse) < 1e-6, (root, inverse)


@pytest.mark.parametrize('timing', [True, False], ids=['timing', 'notiming'])
def test_conditional_process_is_the_same_under_root_and_inverse(timing):
    """At the posterior row, the conditional mean of the GP at the TOAs (basis times the conditional mean of the
    coefficients) is the same under root and inverse, and sample_conditional draws are finite."""
    psr = _pulsar(5.94, 14.0, 145)
    p = _params('dm145', *ROWS['posterior'])
    process = {}
    for method in ('root', 'inverse'):
        gp = signals.makegp_fftcov_dm(psr, signals.powerlaw, 145, method=method)
        m = ds.PulsarLikelihood([psr.residuals, ds.makenoise_measurement(psr, _noisedict(psr))] +
                                ([ds.makegp_timing(psr, svd=True)] if timing else []) + [gp])
        mu, _ = m.conditional(p)
        F = gp.F(p) if callable(gp.F) else gp.F
        process[method] = np.asarray(F) @ np.asarray(mu[m.N.index['SYN_dm_gp_coefficients(145)']])
        _, draw = m.sample_conditional(jax.random.PRNGKey(0), p)
        assert all(np.all(np.isfinite(np.asarray(v))) for v in draw.values())
    scale = np.max(np.abs(process['inverse']))
    assert np.max(np.abs(process['root'] - process['inverse'])) / scale < 1e-8


@pytest.mark.parametrize('variant', list(VARIANTS))
def test_noisedict_with_every_hyperparameter_gives_a_constant_gp(variant):
    """With every hyperparameter in noisedict the root GP is a ConstantGP whose logL equals the variable GP's at the
    same values; with one missing it stays variable."""
    psr = _pulsar(5.94, 14.0, 145)
    noise = ds.makenoise_measurement(psr, _noisedict(psr))
    gp = VARIANTS[variant](psr, signals.powerlaw, 145)
    fixed = _variant_values(gp.F.params)
    const = VARIANTS[variant](psr, signals.powerlaw, 145, noisedict=fixed)
    assert isinstance(const, matrix.ConstantGP)
    assert not isinstance(VARIANTS[variant](psr, signals.powerlaw, 145, noisedict=dict(list(fixed.items())[1:])),
                          matrix.ConstantGP)

    logls = [ds.PulsarLikelihood([psr.residuals, noise, ds.makegp_timing(psr, svd=True), g]).logL for g in (gp, const)]
    assert logls[1].params == []
    a, b = float(logls[0](fixed)), float(logls[1]({}))
    assert np.isfinite(a) and abs(a - b) < 1e-8, (a, b)


def test_clogl_differs_from_inverse_by_the_jacobian_of_the_root():
    """With fixed white noise, clogL under root at whitened coefficients z equals clogL under inverse at the
    coefficients R.T z plus log|det R|, and PulsarLikelihood.sample returns finite residuals."""
    psr = _pulsar(5.94, 14.0, 145)
    p = _params('dm145', *ROWS['posterior'])
    R = _root_R(145, signals.getspan(psr), (4, 1, 4), *ROWS['posterior'])
    z = np.random.default_rng(1).standard_normal(145)
    clogl = {}
    for method, c in (('root', z), ('inverse', R.T @ z)):
        m = ds.PulsarLikelihood([psr.residuals, ds.makenoise_measurement(psr, _noisedict(psr)),
                                 signals.makegp_fftcov_dm(psr, signals.powerlaw, 145, method=method)])
        clogl[method] = float(m.clogL({**p, 'SYN_dm_gp_coefficients(145)': c}))
        _, sample = m.sample(jax.random.PRNGKey(0), p)
        assert np.asarray(sample).shape == psr.toas.shape and np.all(np.isfinite(np.asarray(sample)))
    jacobian = np.sum(np.log(np.abs(np.diag(R))))
    assert abs(clogl['root'] - clogl['inverse'] - jacobian) < 1e-6, (clogl, jacobian)


# --- 6. array likelihoods and the optimal statistic ----------------------------------------------------------------

ARRAY_POS = [np.array([0.6, 0.0, 0.8]), np.array([0.0, 0.6, 0.8]), np.array([-0.6, 0.0, -0.8])]
ARRAY_K = 31


@functools.lru_cache(maxsize=None)
def _array():
    """Three synthetic pulsars with distinct names, positions and noise draws."""
    psrs = []
    for i, pos in enumerate(ARRAY_POS):
        psr = types.SimpleNamespace(**vars(_pulsar(5.94, 14.0, 145, seed=i)))
        psr.name, psr.pos, psr.pos_t = f'SYN{i}', pos, np.tile(pos, (len(psr.toas), 1))
        psrs.append(psr)
    return psrs


def _array_gp(psr, kind, T, t0, method, name, common=()):
    if kind == 'fourier':
        return ds.makegp_fourier(psr, ds.powerlaw, 7, T=T, common=list(common), name=name)
    return signals.makegp_fftcov(psr, ds.powerlaw, ARRAY_K, T=T, t0=t0, common=list(common), name=name, method=method)


def _array_psls(white, red, gw, method):
    psrs = _array()
    T, t0 = signals.getspan(psrs), min(psr.toas.min() for psr in psrs)
    psls = []
    for psr in psrs:
        noise = ds.makenoise_measurement(psr, {f'{psr.name}_BE_efac': 1.0, f'{psr.name}_BE_log10_t2equad': LOG10_T2EQUAD}
                                         if white == 'fixed' else {})
        gps = [_array_gp(psr, red, T, t0, method, 'red_noise')]
        if gw is not None:
            gps.append(_array_gp(psr, gw, T, t0, method, 'gw', common=('gw_log10_A', 'gw_gamma')))
        psls.append(ds.PulsarLikelihood([psr.residuals, noise, ds.makegp_timing(psr, svd=True)] + gps))
    return psrs, T, psls


def _array_values(names):
    values = {'efac': 1.0, 't2equad': LOG10_T2EQUAD, 'log10_A': -14.0, 'gamma': 3.0}
    p = {}
    for name in names:
        suffix = [s for s in values if name.endswith(s)]
        assert len(suffix) == 1, name
        p[name] = {'log10_A': -14.5, 'gamma': 13 / 3}[suffix[0]] if name.startswith('gw_') else values[suffix[0]]
    return p


def _consumer(consumer, white, red, gw, method):
    """(function of params returning an array, its parameter names) for one array consumer."""
    if consumer == 'global':
        psrs, T, psls = _array_psls(white, red, None, method)
        logl = ds.GlobalLikelihood(psls, globalgp=ds.makeglobalgp_fourier(psrs, ds.powerlaw, ds.hd_orf, 7, T=T,
                                                                          name='gw')).logL
        return (lambda p: np.atleast_1d(float(logl(p)))), logl.params
    if consumer == 'array':
        logl = ds.ArrayLikelihood(_array_psls(white, red, gw, method)[2]).logL
        return (lambda p: np.atleast_1d(float(logl(p)))), logl.params
    get_os = ds.OS(ds.GlobalLikelihood(_array_psls(white, red, gw, method)[2])).os
    return (lambda p: np.array([float(v) for k, v in get_os(p).items() if k in ('os', 'os_sigma', 'snr')])), get_os.params


@pytest.mark.parametrize('white', ['fixed', 'sampled'])
@pytest.mark.parametrize('consumer, red, gw', [('global', 'fftcov', None), ('array', 'fftcov', 'fftcov'),
                                               ('os', 'fftcov', 'fftcov'), ('os', 'fourier', 'fftcov'),
                                               ('os', 'fftcov', 'fourier')],
                         ids=['global', 'array', 'os', 'os-fftcovgw', 'os-fftcovred'])
def test_array_consumers_are_the_same_under_root_and_inverse(consumer, red, gw, white):
    """GlobalLikelihood, ArrayLikelihood and OS.os give the same values whether the fftint GPs (red noise, and the GW
    as a per-pulsar common GP) are built with root or inverse, with the white noise fixed or sampled."""
    out = {}
    for method in ('root', 'inverse'):
        fn, names = _consumer(consumer, white, red, gw, method)
        out[method] = fn(_array_values(names))
    assert np.all(np.isfinite(out['root']))
    assert np.allclose(out['root'], out['inverse'], rtol=1e-8, atol=0.0), out


def test_keyword_only_psd_arguments_keep_their_values():
    """A PSD with a keyword-only argument fixed by partial (gamma = 13/3) has the same parameters and logL under
    root and inverse."""
    psr = _pulsar(5.94, 14.0, 145)
    out = {}
    for method in ('root', 'inverse'):
        gp = signals.makegp_fftcov(psr, matrix.partial(signals.powerlaw, gamma=13 / 3), 31, method=method)
        m = ds.PulsarLikelihood([psr.residuals, ds.makenoise_measurement(psr, _noisedict(psr)),
                                 ds.makegp_timing(psr, svd=True), gp])
        assert m.logL.params == ['SYN_fftcovGP_log10_A']
        out[method] = float(m.logL({'SYN_fftcovGP_log10_A': -13.3}))
    assert np.isfinite(out['root']) and abs(out['root'] - out['inverse']) < 1e-6, out


def test_vector_psd_arguments_are_rejected():
    """make_rootbasis rejects a PSD with a vector (typing.Sequence) parameter."""
    psr = _pulsar(5.94, 14.0, 145)
    with pytest.raises(ValueError, match='elementwise'):
        signals.makegp_fftcov(psr, signals.freespectrum, 31)


def test_root_factor_is_continuous_in_the_hyperparameters():
    """R has a positive diagonal, so R.T z at fixed z changes smoothly as gamma moves."""
    T = 5.94 * YR
    z = np.random.default_rng(2).standard_normal(31)
    gammas = np.arange(1.5, 6.0, 1e-3)
    Rs = [_root_R(31, T, (4, 1, 4), -13.3, g) for g in gammas]
    assert all(np.all(np.diag(R) > 0) for R in Rs)
    steps = [np.max(np.abs((R2 - R1).T @ z)) / np.max(np.abs(R1.T @ z)) for R1, R2 in zip(Rs[:-1], Rs[1:])]
    assert max(steps) < 1e-2, max(steps)


@pytest.mark.parametrize('white', ['fixed', 'sampled'])
def test_sample_and_clogl_run_under_root(white):
    """PulsarLikelihood.sample and clogL return finite values under root, with the white noise fixed or sampled."""
    psr = _pulsar(5.94, 14.0, 145)
    m = ds.PulsarLikelihood([psr.residuals, ds.makenoise_measurement(psr, _noisedict(psr) if white == 'fixed' else {}),
                             ds.makegp_timing(psr, svd=True), signals.makegp_fftcov_dm(psr, signals.powerlaw, 31)])
    p = {**_params('dm145', *ROWS['posterior']), **({} if white == 'fixed' else _noisedict(psr))}
    _, draw = m.sample(jax.random.PRNGKey(0), p)
    assert np.asarray(draw).shape == psr.toas.shape and np.all(np.isfinite(np.asarray(draw)))
    cp = {**p, **{k: np.zeros(sl.stop - sl.start) for k, sl in m.N.index.items()}}
    assert np.isfinite(float(m.clogL(cp)))
