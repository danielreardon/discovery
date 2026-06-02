# Daniel J. Reardon -- danieljohnreardon@gmail.com #
# BlackJAX Nested Slice Sampling (NSS) backend for discovery.
# Mirrors discovery.samplers.jaxns in structure/usage.

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp

from blackjax.ns import nss, utils

from .. import prior


# ----------------------------------------------------------------------
# Model construction
# ----------------------------------------------------------------------
class NSModel:
    """Lightweight container holding the prior/likelihood callables and the
    metadata discovery needs to build live points and to reconstruct a chain.

    Particle positions are represented as a PyTree (dict) keyed by the *base*
    parameter name. Scalar parameters map to a scalar; vector parameters of the
    form ``name(size)`` map to an array of shape ``(size,)``.
    """

    def __init__(self, mylogl, priordict={}):
        self.params = list(mylogl.params)

        # Build per-parameter uniform bounds and (optional) vector sizes.
        self.bounds = {}   # base name -> (low, high)
        self.sizes = {}    # base name -> None (scalar) or int (vector length)
        for par in self.params:
            low, high = prior.getprior_uniform(par, priordict)
            if '(' in par:
                base = par[:par.index('(')]
                size = int(par[par.index('(') + 1: par.index(')')])
            else:
                base = par
                size = None
            self.bounds[base] = (float(low), float(high))
            self.sizes[base] = size

        # Total dimensionality (vectors expanded). num_inner_steps should be a
        # multiple of this for good slice-sampling mixing.
        self.ndim = sum(1 if s is None else s for s in self.sizes.values())

        # Map full parameter name -> base name for likelihood calls.
        self._par_to_base = {
            par: (par[:par.index('(')] if '(' in par else par) for par in self.params
        }

        def logprior_fn(position):
            # Uniform prior; jax.scipy.stats.uniform.logpdf returns -inf outside
            # [a, b], which is exactly what we want for the box constraint.
            lp = 0.0
            for base, (a, b) in self.bounds.items():
                x = position[base]
                lp = lp + jax.scipy.stats.uniform.logpdf(x, a, b - a).sum()
            return lp

        def loglikelihood_fn(position):
            params_dict = {par: position[self._par_to_base[par]] for par in self.params}
            ll = mylogl(params_dict)
            # The NSS inner kernel only checks the likelihood contour, not the
            # prior, so fold the box constraint into the likelihood to keep
            # particles inside the uniform prior support.
            lp = logprior_fn(position)
            return jnp.where(jnp.isfinite(lp), ll, -jnp.inf)

        self.logprior_fn = logprior_fn
        self.loglikelihood_fn = loglikelihood_fn

    def init_particles(self, rng_key, num_live):
        """Draw ``num_live`` initial live points uniformly from the prior box."""
        position = {}
        for i, (base, (a, b)) in enumerate(self.bounds.items()):
            k = jax.random.fold_in(rng_key, i)
            size = self.sizes[base]
            shape = (num_live,) if size is None else (num_live, size)
            position[base] = jax.random.uniform(k, shape, minval=a, maxval=b)
        return position

    def to_df(self, position):
        """Flatten a resampled position PyTree into a discovery-style DataFrame,
        using ``name`` for scalars and ``name[i]`` for vector components."""
        data = {}
        for base, size in self.sizes.items():
            arr = np.asarray(position[base])
            if arr.ndim == 1:
                data[base] = arr
            else:
                for j in range(arr.shape[1]):
                    data[f"{base}[{j}]"] = arr[:, j]
        return pd.DataFrame(data)


def makemodel(mylogl, priordict={}):
    return NSModel(mylogl, priordict=priordict)


# Provided for API parity with jaxns/numpyro. BlackJAX NSS samples directly in
# the (bounded) parameter space, so no unconstrained transform is needed.
def makemodel_transformed(mylogl, transform=prior.makelogtransform_uniform, priordict={}):
    return makemodel(mylogl, priordict=priordict)


# ----------------------------------------------------------------------
# Sampler construction
# ----------------------------------------------------------------------
def makesampler_nested(model, num_live=1000, num_delete=None, num_inner_steps=None,
                       termination_frac=1e-3, max_iterations=100000, **kwargs):
    """Build a BlackJAX Nested Slice Sampler around a discovery NSModel.

    Parameters
    ----------
    model : NSModel
        Output of ``makemodel``.
    num_live : int
        Number of live points.
    num_delete : int
        Number of live points deleted/replaced per NS step (compression rate).
        Defaults to ``num_live // 2`` for fast, vectorised steps.
    num_inner_steps : int
        Number of slice-sampling steps per new particle. Defaults to
        ``5 * model.ndim`` (should be a multiple of the dimensionality).
    termination_frac : float
        Stop once the fraction of remaining evidence held by the live points
        drops below this value.
    max_iterations : int
        Hard safety cap on the number of NS steps.
    """
    if num_delete is None:
        num_delete = max(1, num_live // 2)
    if num_inner_steps is None:
        num_inner_steps = 5 * model.ndim

    algo = nss.as_top_level_api(
        model.logprior_fn,
        model.loglikelihood_fn,
        num_inner_steps=num_inner_steps,
        num_delete=num_delete,
        **kwargs,
    )

    class Sampler:
        def __init__(self, model, algo, num_live, termination_frac, max_iterations):
            self.model = model
            self.algo = algo
            self.num_live = num_live
            self.termination_frac = termination_frac
            self.max_iterations = max_iterations
            self.state = None
            self.dead = None

        def run(self, key):
            key, init_key = jax.random.split(key)
            particles = self.model.init_particles(init_key, self.num_live)
            self.state = self.algo.init(particles, rng_key=init_key)

            step = jax.jit(self.algo.step)
            log_frac_threshold = jnp.log(self.termination_frac)

            dead = []
            for i in range(self.max_iterations):
                key, subkey = jax.random.split(key)
                self.state, info = step(subkey, self.state)
                dead.append(info)

                logZ = self.state.integrator.logZ
                logZ_live = self.state.integrator.logZ_live
                # Remaining evidence fraction held by live points.
                log_frac = logZ_live - jnp.logaddexp(logZ, logZ_live)
                if bool(log_frac < log_frac_threshold):
                    break
            else:
                print(f"Warning: hit max_iterations={self.max_iterations} before termination.")

            self._key = key
            self.dead = dead
            self.results = utils.finalise(self.state, dead, update_info=False)

            logZ = float(self.state.integrator.logZ)
            ess = float(utils.ess(key, self.results))
            print(f"Nested sampling finished in {len(dead)} steps.")
            print(f"  log Z = {logZ:.3f}")
            print(f"  ESS   = {ess:.1f}")
            return self.state

        def to_df(self, S=None):
            if self.results is None:
                raise RuntimeError("Run the sampler before accessing results.")

            n_dead = int(self.results.particles.loglikelihood.shape[0])
            S = int(S or n_dead)

            key, subkey = jax.random.split(self._key)
            resampled = utils.sample(subkey, self.results, shape=S)

            df = self.model.to_df(resampled.position)
            df['logl'] = np.asarray(resampled.loglikelihood)
            df['logposterior'] = np.asarray(resampled.logdensity + resampled.loglikelihood)

            df['logZ'] = float(self.state.integrator.logZ)
            df['ESS'] = float(utils.ess(key, self.results))
            return df

        def make_plots(self, save_name=None):
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import corner

            if self.results is None:
                raise RuntimeError("Run the sampler before making plots.")

            reserved = {'logl', 'logposterior', 'logZ', 'ESS'}
            df = self.to_df()
            labels = [c for c in df.columns if c not in reserved]
            data = df[labels].values

            fig = corner.corner(
                data, labels=labels,
                show_titles=True, title_fmt=".2f",
                title_kwargs={"fontsize": 10}, label_kwargs={"fontsize": 9},
                plot_datapoints=True, hist_kwargs={"color": "C0"},
                contour_kwargs={"colors": ["C0"]},
            )
            plt.tight_layout()
            if save_name:
                plt.savefig(save_name + "_corner.png")
            plt.close()

    return Sampler(model, algo, num_live, termination_frac, max_iterations)