import re
import warnings

import numpy as np
import pytest
import scipy.integrate
import scipy.stats

import jax
import jax.numpy as jnp

import discovery as ds
from discovery import prior


def makefunc(params, value=0.0):
    return type('Func', (), {'params': list(params),
                             '__call__': staticmethod(lambda p: value)})()


SCALAR_SPECS = {
    'Uniform': [-18.0, -11.0],
    'Normal': [-14.5, 0.6, 'Normal'],
    'TruncatedNormal': [-14.5, 0.6, -16.0, -13.0, 'TruncatedNormal'],
    'UniformWithOutliers': [-16.0, -13.0, 0.15, -20.0, -11.0, 'UniformWithOutliers'],
    'NormalWithOutliers': [-14.5, 0.6, 0.15, -20.0, -11.0, 'NormalWithOutliers'],
    'TruncatedNormalWithOutliers': [-14.5, 0.6, -16.0, -13.0, 0.15, -20.0, -11.0,
                                    'TruncatedNormalWithOutliers'],
    'NormalWithNormalOutliers': [-14.5, 0.6, 0.15, 7.0, 'NormalWithNormalOutliers'],
    'TruncatedNormalWithNormalOutliers': [-14.5, 0.6, -18.0, -11.0, 0.15, 7.0,
                                          'TruncatedNormalWithNormalOutliers'],
}

RED_GROUP = {
    'family': 'TruncatedMultivariateNormal',
    'mu': [-13.8754, 0.7716], 'sigma': [0.7114, 0.7988],
    'rho': [[1.0, 0.7779], [0.7779, 1.0]],
    'bounds': [(-18.0, -11.0), (0.0, 7.0)],
}
RED_KEY = ('(?P<inst>.*_)?red_noise_log10_A', '(?P<inst>.*_)?red_noise_gamma')


@pytest.fixture(autouse=True)
def restore_jointpriors():
    saved = dict(prior.jointpriors_standard)
    yield
    prior.jointpriors_standard.clear()
    prior.jointpriors_standard.update(saved)


# --- backward compatibility -------------------------------------------------

def test_uniform_only_models_reproduce_the_legacy_tanh_arithmetic():
    params = ['crn_gamma', 'crn_log10_A', 'crn_log10_rho(5)', 'gw_gamma', 'cw_psi']
    t = prior.makelogtransform_uniform(makefunc(params))

    a, b = [], []
    for par in params:
        lo, hi = prior.getprior_uniform(par)
        n = int(par[par.index('(') + 1:par.index(')')]) if '(' in par else 1
        a.extend([lo] * n)
        b.extend([hi] * n)
    a, b = jnp.array(a), jnp.array(b)

    rng = np.random.default_rng(0)
    for _ in range(50):
        ys = jnp.array(rng.normal(size=9) * 3.0)

        expected_x = 0.5 * (b + a + (b - a) * jnp.tanh(ys))
        got = t.to_dict(ys)
        flat = jnp.concatenate([jnp.atleast_1d(got[par]) for par in params])
        assert np.array_equal(np.asarray(flat).view(np.int64),
                              np.asarray(expected_x).view(np.int64))

        expected_lp = jnp.sum(jnp.log(2.0) - 2.0 * jnp.logaddexp(ys, -ys))
        assert (np.float64(t.logprior(ys)).view(np.int64)
                == np.float64(expected_lp).view(np.int64))


def test_makelogtransform_uniform_dispatches_to_makelogtransform(monkeypatch):
    """It is a wrapper, not an import-time alias.

    Bound as `makelogtransform_uniform = makelogtransform` the two were identical, which
    is what this test used to assert -- but it also meant replacing
    prior.makelogtransform left every caller of the alias on the original, including the
    sampler entry points that take it as a default argument value.
    """
    import inspect

    assert (inspect.signature(prior.makelogtransform_uniform)
            == inspect.signature(prior.makelogtransform))

    sentinel = object()
    monkeypatch.setattr(prior, 'makelogtransform', lambda *a, **k: sentinel)
    assert prior.makelogtransform_uniform(None) is sentinel


def test_vector_parameters_stay_arrays_when_the_flat_length_equals_the_parameter_count():
    # one length-1 vector plus scalars: the flat length matches len(params), which
    # previously selected the scalar branch and dropped the array shape
    t = prior.makelogtransform_uniform(makefunc(['crn_log10_rho(1)', 'crn_gamma']))
    got = t.to_dict(jnp.zeros(2))

    assert np.shape(got['crn_log10_rho(1)']) == (1,)
    assert np.shape(got['crn_gamma']) == ()


def test_transform_exposes_the_documented_attributes():
    t = prior.makelogtransform_uniform(makefunc(['crn_gamma']))

    for name in ('params', 'logprior', 'logL', 'to_dict', 'to_vec', 'to_df', 'base_scale'):
        assert hasattr(t, name)


# --- specification parsing --------------------------------------------------

def test_parse_spec_accepts_a_bare_range_and_a_tagged_list():
    assert prior.parse_spec([-18.0, -11.0]) == ('Uniform', (-18.0, -11.0))
    assert prior.parse_spec([-18.0, -11.0, 'Uniform']) == ('Uniform', (-18.0, -11.0))
    assert prior.parse_spec([-14.5, 0.6, 'Normal']) == ('Normal', (-14.5, 0.6))


def test_parse_spec_rejects_an_unknown_family():
    with pytest.raises(KeyError, match='Unknown prior family'):
        prior.parse_spec([1.0, 2.0, 'Cauchy'])


def test_parse_spec_rejects_the_wrong_number_of_arguments():
    with pytest.raises(ValueError, match='takes 2 numbers'):
        prior.parse_spec([-14.5, 0.6, 1.0, 'Normal'])


def test_parse_spec_rejects_an_untagged_list_that_is_not_a_range():
    with pytest.raises(ValueError, match='neither'):
        prior.parse_spec([-14.5, 0.6, 1.0])


def test_a_tagged_spec_is_never_read_as_a_uniform_range():
    # [mean, std, 'Normal'] read as a box would give a > b and silently invert the map
    spec = {'mypar': [-14.5, 0.6, 'Normal']}

    with pytest.raises(ValueError, match='no uniform range'):
        prior.getprior_uniform('mypar', spec)
    with pytest.raises(ValueError, match='no uniform range'):
        prior.sample_uniform(['mypar'], spec)


def test_getsupport_reports_the_link_support_of_every_family():
    expected = {'Uniform': (-18.0, -11.0), 'Normal': (-np.inf, np.inf),
                'TruncatedNormal': (-16.0, -13.0), 'UniformWithOutliers': (-20.0, -11.0),
                'NormalWithOutliers': (-np.inf, np.inf),
                'TruncatedNormalWithOutliers': (-20.0, -11.0),
                'NormalWithNormalOutliers': (-np.inf, np.inf),
                'TruncatedNormalWithNormalOutliers': (-18.0, -11.0)}

    for name, spec in SCALAR_SPECS.items():
        assert prior.getsupport('mypar', {'mypar': spec}) == expected[name]


# --- scalar families --------------------------------------------------------

@pytest.mark.parametrize('name', sorted(SCALAR_SPECS))
def test_every_scalar_family_is_a_normalised_density(name):
    t = prior.makelogtransform(makefunc(['mypar']), priordict={'mypar': SCALAR_SPECS[name]})
    logprior = jax.jit(t.logprior)

    integral, _ = scipy.integrate.quad(
        lambda y: float(np.exp(logprior(jnp.array([y])))), -60, 60, limit=800)

    assert integral == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize('name', sorted(SCALAR_SPECS))
def test_every_scalar_family_round_trips_and_has_a_finite_gradient(name):
    t = prior.makelogtransform(makefunc(['mypar']), priordict={'mypar': SCALAR_SPECS[name]})

    for y in np.linspace(-6.0, 6.0, 25):
        x = float(t.to_dict(jnp.array([y]))['mypar'])
        assert float(t.to_vec({'mypar': x})[0]) == pytest.approx(y, abs=1e-8)

        g = jax.grad(lambda v: t.logprior(v))(jnp.array([y]))
        assert np.isfinite(g).all()


def test_the_normal_family_is_exactly_whitened():
    t = prior.makelogtransform(makefunc(['mypar']), priordict={'mypar': [-14.5, 0.6, 'Normal']})

    for y in (-2.0, 0.0, 0.7, 3.0):
        # log p(y) = -y^2/2 - log(2 pi)/2 for any mean and width
        assert float(t.logprior(jnp.array([y]))) == pytest.approx(
            -0.5 * y * y - 0.5 * np.log(2 * np.pi), abs=1e-12)
        assert float(jax.grad(lambda v: t.logprior(v))(jnp.array([y]))[0]) == pytest.approx(-y)


def test_the_truncated_normal_matches_scipy():
    mu, sd, lo, hi = -14.5, 0.6, -16.0, -13.0
    t = prior.makelogtransform(makefunc(['mypar']),
                               priordict={'mypar': [mu, sd, lo, hi, 'TruncatedNormal']})
    ref = scipy.stats.truncnorm((lo - mu) / sd, (hi - mu) / sd, loc=mu, scale=sd)

    # differencing removes the link Jacobian only if it is evaluated; use the
    # analytic Jacobian of the tanh map instead
    fwd = lambda y: t.to_dict(jnp.array([y]))['mypar']
    for y in (-1.5, -0.3, 0.0, 0.9, 2.2):
        x = float(fwd(y))
        logjac = float(jnp.log(jnp.abs(jax.grad(fwd)(y))))
        assert float(t.logprior(jnp.array([y]))) - logjac == pytest.approx(
            float(ref.logpdf(x)), abs=1e-10)


def test_the_truncated_normal_normalisation_survives_a_far_tail_box():
    # log(Phi(be) - Phi(al)) cancels catastrophically if formed as a plain difference
    got = prior._log_ndtr_diff_np(np.array([10.0, 38.0, 50.0]), np.array([11.0, 39.0, 60.0]))

    assert np.isfinite(got).all()
    assert got[0] == pytest.approx(-53.2313102, abs=1e-6)


def test_the_jax_and_numpy_log_ndtr_differences_agree():
    al = np.array([-3.0, -0.5, 2.0, 10.0, 30.0])
    be = np.array([-1.0, 0.5, 4.0, 11.0, 31.0])

    assert np.allclose(np.asarray(prior._log_ndtr_diff(jnp.array(al), jnp.array(be))),
                       prior._log_ndtr_diff_np(al, be), rtol=1e-12)


def test_a_mixture_reaches_the_outlier_region_outside_its_core_box():
    spec = SCALAR_SPECS['TruncatedNormalWithOutliers']
    t = prior.makelogtransform(makefunc(['mypar']), priordict={'mypar': spec})

    # the core is [-16, -13]; the link must cover the outlier box [-20, -11]
    reached = [float(t.to_dict(jnp.array([y]))['mypar']) for y in (-8.0, 8.0)]

    assert reached[0] < -16.0 and reached[1] > -13.0
    assert np.isfinite(t.logprior(jnp.array([7.0])))


def test_chi_is_the_outlier_weight():
    core, out = (-16.0, -13.0), (-20.0, -11.0)
    for chi in (0.05, 0.5):
        spec = [core[0], core[1], chi, out[0], out[1], 'UniformWithOutliers']
        t = prior.makelogtransform(makefunc(['mypar']), priordict={'mypar': spec})
        fwd = lambda y: t.to_dict(jnp.array([y]))['mypar']

        # a point inside the outlier box but outside the core carries weight chi alone
        y = 8.0
        x = float(fwd(y))
        assert x > core[1]
        logjac = float(jnp.log(jnp.abs(jax.grad(fwd)(y))))
        density = float(t.logprior(jnp.array([y]))) - logjac
        assert density == pytest.approx(np.log(chi) - np.log(out[1] - out[0]), abs=1e-10)


def test_a_uniform_core_inside_a_wider_outlier_box_warns_about_its_step():
    # core [-16, -13] at weight 0.9 against outlier [-20, -11] at weight 0.1
    # steps by log(1 + 9 * 9/3) = 3.33 nats
    spec = [-16.0, -13.0, 0.1, -20.0, -11.0, 'UniformWithOutliers']

    with pytest.warns(UserWarning, match='steps by 3.33 nats'):
        prior.makelogtransform(makefunc(['mypar']), priordict={'mypar': spec})


def test_a_matching_outlier_box_does_not_warn():
    spec = [-16.0, -13.0, 0.1, -16.0, -13.0, 'UniformWithOutliers']

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        prior.makelogtransform(makefunc(['mypar']), priordict={'mypar': spec})


def test_a_joint_prior_with_a_wider_outlier_box_warns_about_its_step():
    spec = dict(RED_GROUP, outlier={'chi': 1e-6, 'bounds': [(-40.0, 10.0), (-40.0, 40.0)]})

    with pytest.warns(UserWarning, match='steps by'):
        prior.makelogtransform(makefunc(['J0030+0451_red_noise_gamma',
                                         'J0030+0451_red_noise_log10_A']),
                               jointpriors={RED_KEY: spec})


# --- to_vec at and beyond the prior edge ------------------------------------

def test_to_vec_is_finite_at_a_prior_edge_and_nan_outside_it():
    t = prior.makelogtransform_uniform(makefunc(['crn_gamma', 'crn_log10_A']))
    lo, hi = prior.getprior_uniform('crn_gamma')

    at_edge = t.to_vec({'crn_gamma': hi, 'crn_log10_A': -14.0})
    assert np.isfinite(at_edge).all()
    assert float(at_edge[0]) == pytest.approx(18.714973875118524, abs=1e-9)

    outside = t.to_vec({'crn_gamma': hi + 1.0, 'crn_log10_A': -14.0})
    assert np.isnan(np.asarray(outside)[0])


def test_to_vec_at_an_edge_maps_back_to_that_edge():
    t = prior.makelogtransform_uniform(makefunc(['crn_gamma', 'crn_log10_A']))
    lo, hi = prior.getprior_uniform('crn_gamma')

    ys = t.to_vec({'crn_gamma': hi, 'crn_log10_A': -14.0})
    assert float(t.to_dict(ys)['crn_gamma']) == hi


def test_to_vec_leaves_interior_points_bit_identical_to_the_unclamped_map():
    t = prior.makelogtransform_uniform(makefunc(['crn_gamma', 'crn_log10_A']))
    a = jnp.array([prior.getprior_uniform(p)[0] for p in ('crn_gamma', 'crn_log10_A')])
    b = jnp.array([prior.getprior_uniform(p)[1] for p in ('crn_gamma', 'crn_log10_A')])

    rng = np.random.default_rng(3)
    for _ in range(200):
        xs = jnp.array(rng.uniform(np.asarray(a), np.asarray(b)))
        got = t.to_vec(dict(zip(('crn_gamma', 'crn_log10_A'), xs)))
        expected = jnp.arctanh((a + b - 2 * xs) / (a - b))
        assert np.array_equal(np.asarray(got).view(np.int64),
                              np.asarray(expected).view(np.int64))


def test_makelogtransform_classic_clamps_the_same_way():
    t = prior.makelogtransform_classic(makefunc(['crn_gamma', 'crn_log10_A']))
    hi = prior.getprior_uniform('crn_gamma')[1]

    assert np.isfinite(t.to_vec({'crn_gamma': hi, 'crn_log10_A': -14.0})).all()


# --- joint priors -----------------------------------------------------------

def group_logdensity(t, params, ys):
    """logprior with the link Jacobian removed, for comparison against a density."""
    fwd = lambda v: jnp.array([t.to_dict(v)[p] for p in params])
    return float(t.logprior(ys)) - float(jnp.linalg.slogdet(jax.jacfwd(fwd)(ys))[1])


def test_a_joint_prior_reproduces_the_truncated_multivariate_normal():
    params = ['J0030+0451_red_noise_gamma', 'J0030+0451_red_noise_log10_A']
    t = prior.makelogtransform(makefunc(params), jointpriors={RED_KEY: RED_GROUP})

    mu, sigma = np.array(RED_GROUP['mu']), np.array(RED_GROUP['sigma'])
    cov = np.outer(sigma, sigma) * np.array(RED_GROUP['rho'])
    mvn = scipy.stats.multivariate_normal(mean=mu, cov=cov)
    bounds = np.array(RED_GROUP['bounds'])
    mass = (mvn.cdf(bounds[:, 1]) - mvn.cdf([bounds[0, 0], bounds[1, 1]])
            - mvn.cdf([bounds[0, 1], bounds[1, 0]]) + mvn.cdf(bounds[:, 0]))

    rng = np.random.default_rng(7)
    for _ in range(40):
        ys = jnp.array(rng.normal(size=2) * 1.5)
        x = np.array([t.to_dict(ys)[p] for p in params])
        # params sort gamma before log10_A; the group order is (log10_A, gamma)
        ref = float(mvn.logpdf([x[1], x[0]]) - np.log(mass))
        assert group_logdensity(t, params, ys) == pytest.approx(ref, abs=1e-9)


def test_a_joint_prior_is_a_normalised_density():
    params = ['J0030+0451_red_noise_gamma', 'J0030+0451_red_noise_log10_A']
    t = prior.makelogtransform(makefunc(params), jointpriors={RED_KEY: RED_GROUP})
    logprior = jax.jit(t.logprior)

    integral, _ = scipy.integrate.dblquad(
        lambda a, b: float(np.exp(logprior(jnp.array([b, a])))), -30, 30, -30, 30, epsabs=1e-10)

    assert integral == pytest.approx(1.0, abs=1e-6)


def test_a_joint_prior_gathers_members_that_are_not_adjacent():
    # func.params is sorted, so chrom_gp_gamma sits between the group's members
    key = ('(?P<inst>.*_)?chrom_gp_log10_A', '(?P<inst>.*_)?chrom_gp_alpha')
    spec = {'family': 'TruncatedMultivariateNormal',
            'mu': [-14.3, 5.5], 'sigma': [0.5, 1.2], 'rho': [[1.0, -0.2], [-0.2, 1.0]],
            'bounds': [(-18.0, -11.0), (3.0, 14.0)]}
    params = sorted(['J0030+0451_chrom_gp_alpha', 'J0030+0451_chrom_gp_gamma',
                     'J0030+0451_chrom_gp_log10_A'])
    assert params[1] == 'J0030+0451_chrom_gp_gamma'

    t = prior.makelogtransform(makefunc(params), jointpriors={key: spec})
    got = t.to_dict(jnp.zeros(3))

    assert float(got['J0030+0451_chrom_gp_log10_A']) == pytest.approx(-14.3, abs=1e-9)
    assert float(got['J0030+0451_chrom_gp_alpha']) == pytest.approx(5.5, abs=1e-9)
    # the ungrouped member keeps its own uniform prior, so y = 0 is its box midpoint
    lo, hi = prior.getprior_uniform('J0030+0451_chrom_gp_gamma')
    assert float(got['J0030+0451_chrom_gp_gamma']) == pytest.approx(0.5 * (lo + hi))


def test_a_joint_prior_instantiates_once_per_pulsar():
    params = sorted(['J0030+0451_red_noise_gamma', 'J0030+0451_red_noise_log10_A',
                     'J1909-3744_red_noise_gamma', 'J1909-3744_red_noise_log10_A'])
    t = prior.makelogtransform(makefunc(params), jointpriors={RED_KEY: RED_GROUP})

    single = prior.makelogtransform(
        makefunc(['J0030+0451_red_noise_gamma', 'J0030+0451_red_noise_log10_A']),
        jointpriors={RED_KEY: RED_GROUP})

    ys = jnp.array([0.4, -0.7, 0.4, -0.7])
    assert float(t.logprior(ys)) == pytest.approx(2.0 * float(single.logprior(ys[:2])), abs=1e-9)


def test_a_joint_prior_rejects_an_instance_missing_a_member():
    params = ['J0030+0451_red_noise_gamma', 'J0030+0451_red_noise_log10_A',
              'J1909-3744_red_noise_gamma']

    with pytest.raises(ValueError, match='no parameter matching'):
        prior.makelogtransform(makefunc(params), jointpriors={RED_KEY: RED_GROUP})


def test_a_joint_prior_rejects_a_non_positive_definite_correlation():
    spec = dict(RED_GROUP, rho=[[1.0, 1.5], [1.5, 1.0]])

    with pytest.raises(ValueError, match='not positive definite'):
        prior.makelogtransform(makefunc(['J0030+0451_red_noise_gamma',
                                         'J0030+0451_red_noise_log10_A']),
                               jointpriors={RED_KEY: spec})


def test_a_joint_prior_rejects_a_box_holding_no_gaussian_mass():
    spec = dict(RED_GROUP, mu=[-13.8754, 60.0])

    with pytest.raises(ValueError, match='no Gaussian mass'):
        prior.makelogtransform(makefunc(['J0030+0451_red_noise_gamma',
                                         'J0030+0451_red_noise_log10_A']),
                               jointpriors={RED_KEY: spec})


def test_a_gaussian_outlier_never_steps_at_the_core_boundary():
    # a uniform outlier wider than the core steps; a gaussian one sharing the box
    # and the mean does not, whatever the mixing fraction
    for chi in (0.01, 0.5, 0.99):
        spec = [-14.5, 0.6, -18.0, -11.0, chi, 7.0, 'TruncatedNormalWithNormalOutliers']
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            prior.makelogtransform(makefunc(['mypar']), priordict={'mypar': spec})


def test_a_joint_prior_may_take_a_gaussian_outlier_component():
    spec = dict(RED_GROUP, outlier={'chi': 0.2261, 'sigma': [7.0, 6.0]})
    params = ['J0030+0451_red_noise_gamma', 'J0030+0451_red_noise_log10_A']
    t = prior.makelogtransform(makefunc(params), jointpriors={RED_KEY: spec},
                               prior_whiten=False)
    logprior = jax.jit(t.logprior)

    integral, _ = scipy.integrate.dblquad(
        lambda a, b: float(np.exp(logprior(jnp.array([b, a])))), -30, 30, -30, 30, epsabs=1e-10)

    assert integral == pytest.approx(1.0, abs=1e-6)

    # the outlier shares the core's box, so the support is unchanged; bounds are
    # in group order (log10_A, gamma) while params sort gamma first
    (a_lo, a_hi), (g_lo, g_hi) = RED_GROUP['bounds']
    for ys in ([6.0, -6.0], [-6.0, 6.0], [0.0, 0.0]):
        x = t.to_dict(jnp.array(ys))
        assert a_lo <= float(x['J0030+0451_red_noise_log10_A']) <= a_hi
        assert g_lo <= float(x['J0030+0451_red_noise_gamma']) <= g_hi


def test_a_joint_gaussian_outlier_rejects_a_bad_width():
    for bad in ([0.0, 6.0], [7.0]):
        spec = dict(RED_GROUP, outlier={'chi': 0.2, 'sigma': bad})
        with pytest.raises(ValueError):
            prior.makelogtransform(makefunc(['J0030+0451_red_noise_gamma',
                                             'J0030+0451_red_noise_log10_A']),
                                   jointpriors={RED_KEY: spec})


def test_a_joint_prior_with_a_matching_outlier_box_stays_normalised():
    # a uniform outlier component holds density right up to the box edge, so the
    # quadrature needs the whole link range; unwhitened coordinates saturate by |y| ~ 19
    spec = dict(RED_GROUP, outlier={'chi': 0.2261, 'bounds': RED_GROUP['bounds']})
    params = ['J0030+0451_red_noise_gamma', 'J0030+0451_red_noise_log10_A']
    t = prior.makelogtransform(makefunc(params), jointpriors={RED_KEY: spec},
                               prior_whiten=False)
    logprior = jax.jit(t.logprior)

    integral, _ = scipy.integrate.dblquad(
        lambda a, b: float(np.exp(logprior(jnp.array([b, a])))), -30, 30, -30, 30, epsabs=1e-10)

    assert integral == pytest.approx(1.0, abs=1e-6)


def test_whitening_is_a_reparametrisation_that_leaves_the_physical_density_alone():
    params = ['J0030+0451_red_noise_gamma', 'J0030+0451_red_noise_log10_A']
    spec = dict(RED_GROUP, outlier={'chi': 0.2261, 'bounds': RED_GROUP['bounds']})
    on = prior.makelogtransform(makefunc(params), jointpriors={RED_KEY: spec}, prior_whiten=True)
    off = prior.makelogtransform(makefunc(params), jointpriors={RED_KEY: spec}, prior_whiten=False)

    rng = np.random.default_rng(11)
    for _ in range(20):
        ys = jnp.array(rng.normal(size=2) * 1.5)
        x = {p: on.to_dict(ys)[p] for p in params}
        # compare at the same physical point, reached through each parametrisation
        assert (group_logdensity(on, params, ys)
                == pytest.approx(group_logdensity(off, params, off.to_vec(x)), abs=1e-9))


def test_jointpriors_standard_is_merged_like_priordict_standard():
    params = ['J0030+0451_red_noise_gamma', 'J0030+0451_red_noise_log10_A']
    prior.jointpriors_standard[RED_KEY] = RED_GROUP

    t = prior.makelogtransform(makefunc(params))
    got = t.to_dict(jnp.zeros(2))

    assert float(got['J0030+0451_red_noise_log10_A']) == pytest.approx(-13.8754, abs=1e-9)


def test_a_joint_prior_member_needs_no_scalar_prior_of_its_own():
    key = ('(?P<inst>.*_)?sw_gp_log10_ell', '(?P<inst>.*_)?sw_gp_log10_sigma')
    spec = {'family': 'TruncatedMultivariateNormal',
            'mu': [1.92, 0.64], 'sigma': [1.028, 0.171],
            'rho': [[1.0, 0.997], [0.997, 1.0]], 'bounds': [(1.0, 4.0), (-2.0, 1.3)]}
    params = ['J1909-3744_sw_gp_log10_ell', 'J1909-3744_sw_gp_log10_sigma']

    for par in params:
        with pytest.raises(KeyError):
            prior.getprior_uniform(par)

    t = prior.makelogtransform(makefunc(params), jointpriors={key: spec})

    assert float(t.to_dict(jnp.zeros(2))['J1909-3744_sw_gp_log10_ell']) == pytest.approx(1.92,
                                                                                         abs=1e-9)


def test_a_joint_prior_matching_nothing_is_ignored():
    t = prior.makelogtransform(makefunc(['crn_gamma']), jointpriors={RED_KEY: RED_GROUP})

    assert float(t.to_dict(jnp.zeros(1))['crn_gamma']) == pytest.approx(3.5)


# --- hierarchical joint priors ----------------------------------------------

HIER_PSRS = ['J0030+0451', 'J1713+0747', 'J1909-3744']
HIER_PARAMS = sorted(f'{p}_red_noise_{s}' for p in HIER_PSRS for s in ('log10_A', 'gamma'))

HIER_SPEC = {
    'family': 'MultivariateNormal',
    'link': ['identity', ('logistic', 0.0, 7.0)],
    'mu': ['spin_log10_A_mu', 'spin_gamma_mu'],
    'chol': [['spin_L_amp', 0.0], ['spin_L_12', 'spin_L_gamma']],
}
HIER_PRIORS = {'spin_log10_A_mu': [-17.0, -12.0], 'spin_gamma_mu': [-4.0, 4.0],
               'spin_L_amp': [0.3, 3.0], 'spin_L_gamma': [0.3, 3.5],
               'spin_L_12': [-1.5, 1.5], 'spin_Q': [0.0, 1.0]}


def hier_transform(spec=None, priordict=None, params=None):
    return prior.makelogtransform(makefunc(params or HIER_PARAMS),
                                  priordict={**HIER_PRIORS, **(priordict or {})},
                                  jointpriors={RED_KEY: spec or HIER_SPEC})


def test_hierarchical_hyperparameters_join_the_sampled_vector():
    t = hier_transform()

    assert set(t.params) == set(HIER_PARAMS) | {'spin_log10_A_mu', 'spin_gamma_mu',
                                                'spin_L_amp', 'spin_L_12', 'spin_L_gamma'}
    assert list(t.params)[:len(HIER_PARAMS)] == HIER_PARAMS


def test_hierarchical_hyperparameters_take_their_own_scalar_priors():
    t = hier_transform()
    got = t.to_dict(jnp.zeros(len(t.params)))

    # y = 0 is the midpoint of each hyperparameter's uniform box
    assert float(got['spin_L_amp']) == pytest.approx(1.65)
    assert float(got['spin_gamma_mu']) == pytest.approx(0.0)


def test_the_noncentered_foreground_is_exactly_a_unit_normal():
    t = hier_transform()
    members = [t.params.index(p) for p in HIER_PARAMS]
    hyper = [i for i in range(len(t.params)) if i not in members]

    rng = np.random.default_rng(5)
    for _ in range(5):
        ys = jnp.array(rng.normal(size=len(t.params)))
        h = np.asarray(ys)[hyper]
        # the hyperparameters keep their own tanh log-prior
        hyper_lp = float(np.sum(np.log(2.0) - 2.0 * np.logaddexp(h, -h)))
        zt = np.asarray(ys)[members]

        assert float(t.logprior(ys)) - hyper_lp == pytest.approx(
            float(np.sum(-0.5 * zt ** 2 - 0.5 * np.log(2 * np.pi))), abs=1e-9)


def test_the_logistic_link_keeps_gamma_inside_its_interval():
    t = hier_transform()

    rng = np.random.default_rng(6)
    for _ in range(50):
        got = t.to_dict(jnp.array(rng.normal(size=len(t.params)) * 4.0))
        for psr in HIER_PSRS:
            assert 0.0 < float(got[f'{psr}_red_noise_gamma']) < 7.0


def test_a_hierarchical_prior_round_trips_through_its_sampled_hyperparameters():
    t = hier_transform()

    rng = np.random.default_rng(7)
    for _ in range(20):
        ys = jnp.array(rng.normal(size=len(t.params)))
        assert np.allclose(np.asarray(t.to_vec(t.to_dict(ys))), np.asarray(ys), atol=1e-9)


def test_centered_and_noncentered_differ_by_the_cholesky_determinant():
    off = hier_transform()
    on = hier_transform(dict(HIER_SPEC, parametrisation='centered'))

    rng = np.random.default_rng(8)
    for _ in range(5):
        ys = jnp.array(rng.normal(size=len(off.params)))
        x = off.to_dict(ys)
        yc = on.to_vec(x)

        # the two reach the same physical point
        for p in HIER_PARAMS:
            assert float(on.to_dict(yc)[p]) == pytest.approx(float(x[p]), abs=1e-9)

        # z = mu + L z~ carries |L| per instance
        logdet = np.log(abs(float(x['spin_L_amp']))) + np.log(abs(float(x['spin_L_gamma'])))
        assert float(off.logprior(ys)) - float(on.logprior(yc)) == pytest.approx(
            len(HIER_PSRS) * logdet, abs=1e-8)


def test_a_hierarchical_prior_has_finite_gradients_everywhere_it_is_sampled():
    t = hier_transform(dict(HIER_SPEC, outlier={'chi': 'spin_Q',
                                                'chol': [[7.0, 0.0], [0.0, 6.0]]}))
    grad = jax.jit(jax.grad(t.logprior))

    rng = np.random.default_rng(9)
    for _ in range(100):
        ys = jnp.array(rng.normal(size=len(t.params)) * 2.0)
        assert np.isfinite(float(t.logprior(ys)))
        assert np.isfinite(np.asarray(grad(ys))).all()


def test_a_covariate_shifts_the_population_mean_per_instance():
    values = dict(zip((f'{p}_' for p in HIER_PSRS), (1.0, -2.0, 0.5)))
    spec = dict(HIER_SPEC,
                covariates=[{'coefficient': 'red_alpha', 'coord': 0, 'values': values}])
    t = hier_transform(spec, priordict={'red_alpha': [0.5, 0.5]})

    got = t.to_dict(jnp.zeros(len(t.params)))
    mu = float(got['spin_log10_A_mu'])

    assert float(got['red_alpha']) == pytest.approx(0.5)
    for psr, cov in zip(HIER_PSRS, (1.0, -2.0, 0.5)):
        assert float(got[f'{psr}_red_noise_log10_A']) == pytest.approx(mu + 0.5 * cov, abs=1e-9)


def test_a_zero_width_range_fixes_a_parameter_in_the_general_path():
    # the idiom used to pin a parameter, e.g. priordict={'gw_gamma': [13/3, 13/3]}
    t = prior.makelogtransform(makefunc(['gw_gamma', 'mypar']),
                               priordict={'gw_(.*_)?gamma': [13 / 3, 13 / 3],
                                          'mypar': [-14.5, 0.6, 'Normal']})

    for y in (-3.0, 0.0, 4.0):
        got = t.to_dict(jnp.array([y, 0.3]))
        assert float(got['gw_gamma']) == pytest.approx(13 / 3)

    # the fixed coordinate keeps the log-prior the legacy transform gave it
    only = prior.makelogtransform(makefunc(['mypar']), priordict={'mypar': [-14.5, 0.6, 'Normal']})
    legacy = float(jnp.log(2.0) - 2.0 * jnp.logaddexp(1.7, -1.7))
    assert float(t.logprior(jnp.array([1.7, 0.3]))) == pytest.approx(
        float(only.logprior(jnp.array([0.3]))) + legacy, abs=1e-12)


@pytest.mark.parametrize('variant', ['plain', 'outlier', 'centered', 'covariate'])
def test_a_hierarchical_group_maps_a_whole_chain_at_once(variant):
    values = dict(zip((f'{p}_' for p in HIER_PSRS), (1.0, -2.0, 0.5)))
    spec = {'plain': HIER_SPEC,
            'outlier': dict(HIER_SPEC, outlier={'chi': 'spin_Q',
                                                'chol': [[7.0, 0.0], [0.0, 6.0]]}),
            'centered': dict(HIER_SPEC, parametrisation='centered'),
            'covariate': dict(HIER_SPEC, covariates=[{'coefficient': 'red_alpha',
                                                      'coord': 0, 'values': values}]),
            }[variant]
    t = hier_transform(spec, priordict={'red_alpha': [-2.0, 2.0]})

    rng = np.random.default_rng(12)
    chain = jnp.array(rng.normal(size=(8, len(t.params))))
    df = t.to_df(chain)

    assert len(df) == 8
    assert set(HIER_PARAMS) <= set(df.columns)
    assert np.isfinite(df.to_numpy()).all()

    # a batched map must agree row by row with mapping each sample on its own
    single = t.to_dict(chain[3])
    for par in HIER_PARAMS + ['spin_L_amp']:
        assert float(df.iloc[3][par]) == pytest.approx(float(single[par]), abs=1e-12)


def test_a_truncated_multivariate_normal_refuses_sampled_hyperparameters():
    spec = dict(RED_GROUP, mu=['spin_log10_A_mu', 0.7716])

    with pytest.raises(ValueError, match='cannot sample its hyperparameters'):
        prior.makelogtransform(makefunc(HIER_PARAMS), priordict=HIER_PRIORS,
                               jointpriors={RED_KEY: spec})


def test_a_hyperparameter_may_not_shadow_a_model_parameter():
    spec = dict(HIER_SPEC, mu=['J0030+0451_red_noise_gamma', 'spin_gamma_mu'])

    with pytest.raises(ValueError, match='already model parameters'):
        hier_transform(spec)


def test_an_upper_triangular_cholesky_entry_is_rejected():
    spec = dict(HIER_SPEC, chol=[['spin_L_amp', 'spin_L_12'], [0.0, 'spin_L_gamma']])

    with pytest.raises(ValueError, match='lower-triangular'):
        hier_transform(spec)


# --- physical-space covariates -----------------------------------------------

COV_VALUES = dict(zip((f'{p}_' for p in HIER_PSRS), (1.0, -2.0, 0.5)))


def bounded_covariate_transform(space, coeff=0.5, mu=None):
    spec = dict(HIER_SPEC, link=[('logistic', -18.0, -11.0), ('logistic', 0.0, 7.0)],
                covariates=[{'coefficient': 'red_alpha', 'coord': 0,
                             'values': COV_VALUES, 'space': space}])
    priors = {'red_alpha': [coeff, coeff]}
    if mu is not None:
        priors['spin_log10_A_mu'] = [mu, mu]
    return hier_transform(spec, priordict=priors)


@pytest.mark.parametrize('mu', [-2.5, 0.0, 2.5])
def test_a_physical_covariate_is_the_slope_wherever_the_mean_sits(mu):
    # the defining property: d(parameter)/d(covariate) is the coefficient itself, the
    # same number at every population mean, which a latent-space shift is not
    coeff = 0.5
    t = bounded_covariate_transform('physical', coeff=coeff, mu=mu)
    ys = jnp.zeros(len(t.params))
    got = t.to_dict(ys)

    base = None
    for psr, cov in zip(HIER_PSRS, (1.0, -2.0, 0.5)):
        x = float(got[f'{psr}_red_noise_log10_A'])
        if base is None:
            base, base_cov = x, cov
        assert x - base == pytest.approx(coeff * (cov - base_cov), abs=1e-9)


def test_a_latent_covariate_slope_depends_on_the_mean_but_a_physical_one_does_not():
    slopes = {}
    for space in ('latent', 'physical'):
        for mu in (0.0, 2.5):
            got = bounded_covariate_transform(space, mu=mu).to_dict(
                jnp.zeros(len(HIER_PARAMS) + 5))
            xs = [float(got[f'{p}_red_noise_log10_A']) for p in HIER_PSRS]
            # covariates 1.0 and -2.0 are three apart
            slopes[space, mu] = (xs[0] - xs[1]) / 3.0

    assert slopes['physical', 0.0] == pytest.approx(0.5, abs=1e-9)
    assert slopes['physical', 2.5] == pytest.approx(0.5, abs=1e-9)
    assert abs(slopes['latent', 0.0] - slopes['latent', 2.5]) > 0.1


def test_a_physical_covariate_bounds_the_de_scaled_value_not_the_parameter():
    coeff = 2.0
    t = bounded_covariate_transform('physical', coeff=coeff)

    rng = np.random.default_rng(31)
    outside = 0
    for _ in range(50):
        got = t.to_dict(jnp.array(rng.normal(size=len(t.params)) * 4.0))
        for psr, cov in zip(HIER_PSRS, (1.0, -2.0, 0.5)):
            x = float(got[f'{psr}_red_noise_log10_A'])
            assert -18.0 <= x - coeff * cov <= -11.0
            outside += not (-18.0 <= x <= -11.0)

    # a shift of +-2 dex must actually take the parameter out of the box sometimes,
    # or this test would pass on a build that ignored the covariate
    assert outside > 0


def test_a_physical_covariate_round_trips_through_to_vec():
    # the coefficient is sampled, not pinned: a pinned hyperparameter maps back to
    # y = 0 whatever its value, which would hide a broken inverse
    spec = dict(HIER_SPEC, link=[('logistic', -18.0, -11.0), ('logistic', 0.0, 7.0)],
                covariates=[{'coefficient': 'red_alpha', 'coord': 0,
                             'values': COV_VALUES, 'space': 'physical'}])
    t = hier_transform(spec, priordict={'red_alpha': [-2.0, 2.0]})

    rng = np.random.default_rng(32)
    for _ in range(20):
        ys = jnp.array(rng.normal(size=len(t.params)))
        assert np.allclose(np.asarray(t.to_vec(t.to_dict(ys))), np.asarray(ys), atol=1e-9)


def test_a_physical_covariate_leaves_the_latent_density_alone():
    # the shift is constant given the hyperparameters, so it moves the parameter
    # without touching the population's log-density or its link Jacobian
    off = bounded_covariate_transform('latent', coeff=0.0)
    on = bounded_covariate_transform('physical', coeff=0.75)

    rng = np.random.default_rng(33)
    for _ in range(10):
        ys = jnp.array(rng.normal(size=len(off.params)))
        assert float(on.logprior(ys)) == pytest.approx(float(off.logprior(ys)), abs=1e-9)


def test_a_covariate_on_a_bounded_coordinate_demands_a_space():
    spec = dict(HIER_SPEC, link=[('logistic', -18.0, -11.0), ('logistic', 0.0, 7.0)],
                covariates=[{'coefficient': 'red_alpha', 'coord': 0,
                             'values': COV_VALUES}])

    with pytest.raises(ValueError, match="needs 'space'"):
        hier_transform(spec, priordict={'red_alpha': [-2.0, 2.0]})


def test_an_unknown_covariate_space_is_named_in_the_error():
    spec = dict(HIER_SPEC, covariates=[{'coefficient': 'red_alpha', 'coord': 0,
                                        'values': COV_VALUES, 'space': 'sky'}])

    with pytest.raises(ValueError, match='covariate space'):
        hier_transform(spec, priordict={'red_alpha': [-2.0, 2.0]})


def test_an_identity_linked_covariate_needs_no_space_and_the_two_agree():
    # HIER_SPEC's coordinate 0 is identity-linked, where a latent shift and a physical
    # shift are the same operation, so the key stays optional there
    bare = hier_transform(dict(HIER_SPEC, covariates=[
        {'coefficient': 'red_alpha', 'coord': 0, 'values': COV_VALUES}]),
        priordict={'red_alpha': [0.5, 0.5]})
    named = hier_transform(dict(HIER_SPEC, covariates=[
        {'coefficient': 'red_alpha', 'coord': 0, 'values': COV_VALUES,
         'space': 'physical'}]), priordict={'red_alpha': [0.5, 0.5]})

    rng = np.random.default_rng(34)
    for _ in range(10):
        ys = jnp.array(rng.normal(size=len(bare.params)))
        for p in HIER_PARAMS:
            assert float(named.to_dict(ys)[p]) == pytest.approx(
                float(bare.to_dict(ys)[p]), abs=1e-9)


# --- uniform (sampling-prior) outliers on a hierarchical group ---------------

# both coordinates logistic-linked, so the box a uniform outlier is flat over exists
BOXED_SPEC = dict(HIER_SPEC, link=[('logistic', -18.0, -11.0), ('logistic', 0.0, 7.0)])


def uniform_outlier_transform(chi):
    spec = dict(BOXED_SPEC, outlier={'chi': 'spin_Q', 'kind': 'uniform'})
    return hier_transform(spec, priordict={'spin_Q': [chi, chi]})


def test_a_uniform_outlier_is_exactly_flat_in_the_physical_parameter():
    # chi = 1 is the pure outlier: the density of each member must be 1/(hi - lo) in
    # every coordinate, whatever the population mean, Cholesky factor or latent point
    t = uniform_outlier_transform(1.0)
    members = [t.params.index(p) for p in HIER_PARAMS]
    hyper = [i for i in range(len(t.params)) if i not in members]
    widths = np.array([7.0 if p.endswith('log10_A') else 7.0 for p in HIER_PARAMS])

    # the Jacobian of the members' physical values, taken from the transform itself
    # rather than rebuilt, and at fixed hyperparameters: the claim is about the
    # conditional density of the population given its hyperparameters
    def members_of(ys):
        got = t.to_dict(ys)
        return jnp.stack([got[p] for p in HIER_PARAMS])

    jac = jax.jit(jax.jacobian(members_of))

    rng = np.random.default_rng(21)
    for _ in range(20):
        ys = jnp.array(rng.normal(size=len(t.params)) * 2.0)
        h = np.asarray(ys)[hyper]
        hyper_lp = float(np.sum(np.log(2.0) - 2.0 * np.logaddexp(h, -h)))
        _, logjac = np.linalg.slogdet(np.asarray(jac(ys))[:, members])

        # the group's log-density in y, minus the Jacobian, is the density in x
        logpx = float(t.logprior(ys)) - hyper_lp - float(logjac)
        assert logpx == pytest.approx(float(-np.sum(np.log(widths))), abs=1e-9)


def test_a_uniform_outlier_at_zero_weight_leaves_the_population_untouched():
    t, off = uniform_outlier_transform(1e-300), hier_transform(BOXED_SPEC)

    rng = np.random.default_rng(22)
    for _ in range(10):
        ys = jnp.array(rng.normal(size=len(off.params)))
        # spin_Q sits last, and is pinned, so the population members line up; a
        # pinned hyperparameter still carries its own scalar log-prior at y = 0
        full = jnp.concatenate([ys, jnp.zeros(1)])
        assert float(t.logprior(full)) + np.log(2.0) == pytest.approx(
            float(off.logprior(ys)), abs=1e-9)


def test_a_uniform_outlier_needs_no_width_and_rejects_one():
    spec = dict(BOXED_SPEC, outlier={'chi': 'spin_Q', 'kind': 'uniform',
                                     'chol': [[7.0, 0.0], [0.0, 6.0]]})

    with pytest.raises(ValueError, match="takes no 'chol'"):
        hier_transform(spec)


def test_a_uniform_outlier_refuses_an_unbounded_coordinate():
    # HIER_SPEC leaves log10_A identity-linked, so it has no box to be uniform over
    spec = dict(HIER_SPEC, outlier={'chi': 'spin_Q', 'kind': 'uniform'})

    with pytest.raises(ValueError, match='bounded link on every coordinate'):
        hier_transform(spec)


def test_an_unknown_outlier_kind_is_named_in_the_error():
    spec = dict(BOXED_SPEC, outlier={'chi': 'spin_Q', 'kind': 'cauchy'})

    with pytest.raises(ValueError, match='outlier kind'):
        hier_transform(spec)


def test_a_covariate_space_is_not_read_as_a_hyperparameter_name():
    t = bounded_covariate_transform('physical')

    assert 'physical' not in t.params
    assert 'red_alpha' in t.params


def test_the_outlier_kind_is_not_read_as_a_hyperparameter_name():
    t = uniform_outlier_transform(0.1)

    assert 'uniform' not in t.params
    assert set(t.params) == set(HIER_PARAMS) | {'spin_log10_A_mu', 'spin_gamma_mu',
                                               'spin_L_amp', 'spin_L_12', 'spin_L_gamma',
                                               'spin_Q'}


def test_a_uniform_outlier_has_finite_gradients_where_it_is_sampled():
    spec = dict(BOXED_SPEC, outlier={'chi': 'spin_Q', 'kind': 'uniform'})
    t = hier_transform(spec)
    grad = jax.jit(jax.grad(t.logprior))

    rng = np.random.default_rng(23)
    for _ in range(100):
        ys = jnp.array(rng.normal(size=len(t.params)) * 3.0)
        assert np.isfinite(float(t.logprior(ys)))
        assert np.isfinite(np.asarray(grad(ys))).all()


def test_a_uniform_outlier_maps_a_whole_chain_at_once():
    spec = dict(BOXED_SPEC, outlier={'chi': 'spin_Q', 'kind': 'uniform'})
    t = hier_transform(spec)

    rng = np.random.default_rng(24)
    chain = jnp.array(rng.normal(size=(8, len(t.params))))
    df = t.to_df(chain)

    assert len(df) == 8
    assert np.isfinite(df.to_numpy()).all()
    single = t.to_dict(chain[3])
    for par in HIER_PARAMS:
        assert float(df.iloc[3][par]) == pytest.approx(float(single[par]), abs=1e-12)


def test_an_unknown_joint_family_is_named_in_the_error():
    with pytest.raises(KeyError, match='Unknown joint prior family'):
        prior.makelogtransform(makefunc(HIER_PARAMS),
                               jointpriors={RED_KEY: dict(HIER_SPEC, family='Wishart')})


@pytest.mark.integration
def test_a_hierarchical_prior_drives_a_real_likelihood():
    from pathlib import Path

    data_dir = Path(__file__).resolve().parent.parent / 'data'
    psrs = [ds.Pulsar.read_feather(data_dir / f'v1p1_de440_pint_bipm2019-{name}.feather')
            for name in ('B1855+09', 'J0023+0923')]

    model = ds.GlobalLikelihood(
        [ds.PulsarLikelihood([psr.residuals,
                              ds.makenoise_measurement(psr),
                              ds.makegp_fourier(psr, ds.powerlaw, 10, name='red_noise')])
         for psr in psrs])

    key = ('(?P<inst>.*_)?red_noise_log10_A', '(?P<inst>.*_)?red_noise_gamma')
    t = prior.makelogtransform(model.logL, priordict=HIER_PRIORS, jointpriors={key: HIER_SPEC})

    assert 'spin_L_amp' in t.params
    ys = jnp.zeros(len(t.params))

    # the extra hyperparameter keys in to_dict must not disturb the likelihood
    assert np.isfinite(float(t.logL(ys)))
    assert np.isfinite(float(t(ys)))
    assert float(t(ys)) == pytest.approx(float(t.logL(ys)) + float(t.logprior(ys)), abs=1e-9)
    assert np.isfinite(float(jax.jit(t)(ys)))


# --- drawing from the prior -------------------------------------------------

@pytest.mark.parametrize('name', sorted(SCALAR_SPECS))
def test_sample_prior_draws_inside_the_support(name):
    spec = {'mypar': SCALAR_SPECS[name]}
    lo, hi = prior.getsupport('mypar', spec)

    draws = prior.sample_prior(['mypar'], spec, n=500)['mypar']

    assert np.all(draws >= lo) and np.all(draws <= hi)
    assert np.shape(draws) == (500,)
    assert np.isscalar(prior.sample_prior(['mypar'], spec)['mypar'])


def test_sample_prior_handles_vector_parameters():
    draws = prior.sample_prior(['crn_log10_rho(30)'])['crn_log10_rho(30)']

    assert np.shape(draws) == (30,)


def test_sample_prior_reproduces_the_normal_family_moments():
    spec = {'mypar': [-14.5, 0.6, 'Normal']}

    np.random.seed(42)
    draws = prior.sample_prior(['mypar'], spec, n=200000)['mypar']

    assert np.mean(draws) == pytest.approx(-14.5, abs=0.01)
    assert np.std(draws) == pytest.approx(0.6, abs=0.01)


def test_sample_uniform_is_unchanged_for_uniform_priors():
    np.random.seed(1)
    got = prior.sample_uniform(['crn_gamma', 'crn_log10_rho(3)'])

    assert 0.0 <= got['crn_gamma'] <= 7.0
    assert np.shape(got['crn_log10_rho(3)']) == (3,)


# --- sampler integration ----------------------------------------------------

def test_base_scale_is_ten_for_bounded_coordinates_and_wide_for_unbounded_ones():
    t = prior.makelogtransform(makefunc(['mypar', 'crn_gamma']),
                               priordict={'mypar': [-14.5, 0.6, 'Normal']})
    order = {par: i for i, par in enumerate(t.params)}

    assert float(t.base_scale[order['mypar']]) > 100.0
    assert float(t.base_scale[order['crn_gamma']]) == 10.0


def test_the_uniform_fast_path_keeps_the_legacy_base_scale():
    t = prior.makelogtransform_uniform(makefunc(['crn_gamma', 'crn_log10_A']))

    assert np.all(np.asarray(t.base_scale) == 10.0)


def test_nested_sampling_refuses_a_non_uniform_prior():
    jaxns = pytest.importorskip('discovery.samplers.jaxns')

    with pytest.raises(NotImplementedError, match='uniform priors only'):
        jaxns.makemodel(makefunc(['mypar']), priordict={'mypar': [-14.5, 0.6, 'Normal']})


def test_to_df_columns_are_unchanged_for_a_mixed_family_model():
    params = ['crn_log10_rho(3)', 'mypar']
    t = prior.makelogtransform(makefunc(params), priordict={'mypar': [-14.5, 0.6, 'Normal']})

    df = t.to_df(jnp.zeros((2, 4)))

    assert list(df.columns) == ['crn_log10_rho[0]', 'crn_log10_rho[1]',
                                'crn_log10_rho[2]', 'mypar']
