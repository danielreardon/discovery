#!/usr/bin/env python3
"""Tests for discovery.params.Params -- the single-leaf parameter container.

A small set of unit tests confirming the container runs: dict round-trips,
the Mapping protocol, functional updates, and -- the point of the exercise --
that it is a single JAX pytree leaf and survives jit/grad. No pulsar data
needed; the tests operate on hand-built parameter-name lists.
"""

import numpy as np
import pytest

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

from discovery.params import Params, make_layout


VEC_NAME = 'B1855+09_red_noise_log10_rho(30)'          # array-valued: 30 columns
MIXED = [VEC_NAME, 'crn_log10_A', 'crn_gamma']         # flat size 30 + 1 + 1 = 32


def _mixed_dict():
    return {VEC_NAME: np.arange(30, dtype=float),
            'crn_log10_A': -14.5,
            'crn_gamma': 3.5}


@pytest.mark.unit
class TestParams:

    def test_make_layout(self):
        layout, size = make_layout(MIXED)
        assert size == 32
        assert layout[0] == (VEC_NAME, 0, 30, (30,))
        assert layout[1] == ('crn_log10_A', 30, 31, ())
        # a 2-D suffix expands too
        layout2, size2 = make_layout(['fourierGP_var(6,4)'])
        assert size2 == 24 and layout2[0][3] == (6, 4)

    def test_from_dict_and_getitem(self):
        d = _mixed_dict()
        p = Params.from_dict(d, MIXED)
        assert p.size == 32 and len(p) == 3
        assert np.shape(p[VEC_NAME]) == (30,)
        assert np.ndim(p['crn_log10_A']) == 0
        assert np.allclose(p[VEC_NAME], d[VEC_NAME])
        assert float(p['crn_log10_A']) == -14.5

    def test_round_trip(self):
        d = _mixed_dict()
        rt = Params.from_dict(d, MIXED).to_dict()
        assert set(rt) == set(d)
        for k in d:
            assert np.allclose(np.asarray(rt[k]), np.asarray(d[k]))

    def test_mapping_protocol(self):
        p = Params.from_dict(_mixed_dict(), MIXED)
        assert list(p) == MIXED                       # iteration in layout order
        assert set(p.keys()) == set(MIXED)
        assert 'crn_gamma' in p and 'nope' not in p
        assert p.get('nope', 'default') == 'default'

    def test_dict_unpacking(self):
        # logL does `{**params}` in one path -- it must materialise a plain dict
        p = Params.from_dict(_mixed_dict(), MIXED)
        d = {**p}
        assert isinstance(d, dict) and set(d) == set(MIXED)

    def test_update_is_functional(self):
        p = Params.from_dict(_mixed_dict(), MIXED)
        p2 = p.update('crn_gamma', 5.0)
        assert float(p2['crn_gamma']) == 5.0
        assert float(p['crn_gamma']) == 3.5           # original untouched
        assert p2 is not p

    def test_updates_many_including_array(self):
        p = Params.from_dict(_mixed_dict(), MIXED)
        p2 = p.updates({'crn_log10_A': -13.0, VEC_NAME: np.full(30, -7.0)})
        assert float(p2['crn_log10_A']) == -13.0
        assert np.allclose(p2[VEC_NAME], -7.0)
        assert np.allclose(p[VEC_NAME], np.arange(30))  # original untouched

    def test_zeros(self):
        p = Params.zeros(MIXED)
        assert p.size == 32
        assert np.all(np.asarray(p.raw) == 0.0)

    def test_single_pytree_leaf(self):
        # the whole point: one leaf, not one-per-parameter
        p = Params.from_dict(_mixed_dict(), MIXED)
        leaves = jax.tree_util.tree_leaves(p)
        assert len(leaves) == 1
        assert leaves[0].shape == (32,)

    def test_jit_round_trip(self):
        p = Params.from_dict(_mixed_dict(), MIXED)

        @jax.jit
        def f(params):
            return params['crn_log10_A'] * 2.0 + jnp.sum(params[VEC_NAME])

        assert np.allclose(f(p), -14.5 * 2.0 + np.arange(30).sum())

    def test_grad_cotangent_is_params(self):
        p = Params.from_dict(_mixed_dict(), MIXED)

        def f(params):
            return params['crn_gamma'] ** 2 + jnp.sum(params[VEC_NAME] ** 2)

        g = jax.grad(f)(p)
        assert isinstance(g, Params)                   # cotangent keeps Params structure
        assert float(g['crn_gamma']) == pytest.approx(2 * 3.5)

    def test_scatter_under_jit(self):
        # zeros() builds a NumPy raw; updating it with tracers inside jit must
        # still work -- exercises the use_jax detection on the *values*
        @jax.jit
        def build(a, g):
            p = Params.zeros(MIXED).updates({'crn_log10_A': a, 'crn_gamma': g})
            return p['crn_log10_A'] + p['crn_gamma']

        assert float(build(-13.0, 2.5)) == pytest.approx(-10.5)
