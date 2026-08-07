import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
tfpd = tfp.distributions

from jaxns import Model, Prior, NestedSampler, resample
from jaxns.nested_samplers.common.types import TerminationCondition
from .. import prior

def makemodel_transformed(mylogl, transform=prior.makelogtransform_uniform, priordict={}):
    return makemodel(mylogl, priordict)


def makemodel(mylogl, priordict={}):
    params = list(mylogl.params)

    def prior_model():
        values = []
        for par in params:
            try:
                low, high = prior.getprior_uniform(par, priordict)
            except ValueError as exc:
                raise NotImplementedError(
                    f"Nested sampling builds its prior as an explicit box, so it supports "
                    f"uniform priors only: {exc}") from exc
            if '(' in par:
                base = par.split('(')[0]
                size = int(par[par.index('(') + 1: par.index(')')])
                low_arr = jnp.full((size,), low)
                high_arr = jnp.full((size,), high)
                val = yield Prior(tfpd.Uniform(low=low_arr, high=high_arr), name=base)
            else:
                val = yield Prior(tfpd.Uniform(low=low, high=high), name=par)
            values.append(val)
        if len(values) == 1:
            return values[0]

        return tuple(values)

    def log_likelihood(*args):
        params_dict = {}
        i = 0
        for par in params:
            val = args[i]
            params_dict[par] = val
            i += 1
        return mylogl(params_dict)

    return Model(prior_model=prior_model, log_likelihood=log_likelihood)


def makesampler_nested(model, max_samples=1e6, num_live_points=None, init_efficiency_threshold=0.1, difficult_model=True, gradient_guided=False, parameter_estimation=True, **kwargs):
    ns = NestedSampler(
        model=model,
        init_efficiency_threshold=init_efficiency_threshold,
        parameter_estimation=parameter_estimation,
        difficult_model=difficult_model,
        gradient_guided=gradient_guided,
        max_samples=max_samples,
        num_live_points=num_live_points,
        verbose=True,
        **kwargs
    )

    class Sampler:
        def __init__(self, nested_sampler):
            self.ns = nested_sampler
            self.termination = TerminationCondition(dlogZ=1e-3, max_samples=int(1e6))
            self.state = None
            self.results = None

        def run(self, key):
            self.reason, self.state = self.ns(key, term_cond=self.termination)
            self.results = self.ns.to_results(termination_reason=self.reason, state=self.state)
            self.ns.summary(self.results)

        def _flatten_samples(self, samples_dict):
            data = {}
            for name, arr in samples_dict.items():
                arr_np = np.asarray(arr)
                if arr_np.ndim == 1:
                    data[name] = arr_np
                else:
                    arr2 = arr_np.reshape((arr_np.shape[0], -1))
                    for j in range(arr2.shape[1]):
                        data[f"{name}[{j}]"] = arr2[:, j]
            return data

        def to_df(self, equal_weight=True, S=None, seed=0, include_weights=False):
            if self.results is None:
                raise RuntimeError("Run the sampler before accessing results.")

            r = self.results
            logw = np.asarray(r.log_dp_mean)
            samples = {k: np.asarray(v) for k, v in r.samples.items()}
            logl_all = np.asarray(r.log_L_samples)
            logpost_all = np.asarray(r.log_posterior_density)

            if equal_weight:
                S = int(S or logw.shape[0])
                rs = resample(jax.random.PRNGKey(seed),
                              samples=samples,
                              log_weights=logw,
                              S=S,
                              replace=True)
                rs_all = resample(jax.random.PRNGKey(seed),
                                  samples=dict(samples, logl=logl_all, logposterior=logpost_all),
                                  log_weights=logw,
                                  S=S,
                                  replace=True)
                data = self._flatten_samples(rs)
                data['logl'] = np.asarray(rs_all['logl'])
                data['logposterior'] = np.asarray(rs_all['logposterior'])
                df = pd.DataFrame(data)
            else:
                w = np.exp(logw - logw.max())
                w /= w.sum()
                data = self._flatten_samples(samples)
                data['logl'] = logl_all
                data['logposterior'] = logpost_all
                if include_weights:
                    data['weight'] = w
                df = pd.DataFrame(data)

            df['logZ'] = float(np.asarray(r.log_Z_mean))
            df['logZ_uncert'] = float(np.asarray(r.log_Z_uncert))
            df['ESS'] = float(np.asarray(r.ESS))
            return df

        def make_plots(self, save_name=None, diagnostics=False,
                       include_aux=False, equal_weight=True):
            import matplotlib.pyplot as plt
            import corner

            if self.results is None:
                raise RuntimeError("Run the sampler before making plots.")

            reserved = {'logZ', 'logZ_uncert', 'ESS'}
            if include_aux:
                reserved = reserved
            else:
                reserved = reserved | {'logl', 'logposterior', 'weight'}

            if not equal_weight:
                df = self.to_df(equal_weight=False, include_weights=True)
                labels = [c for c in df.columns if c not in reserved]
                data = df[labels].values
                weights = df['weight'].values if 'weight' in df else None
                fig = corner.corner(
                    data, labels=labels, weights=weights,
                    show_titles=True, title_fmt=".2f",
                    title_kwargs={"fontsize": 10}, label_kwargs={"fontsize": 9},
                    plot_datapoints=False, hist_kwargs={"color": "C0"},
                    contour_kwargs={"colors": ["C0"]}
                )
            else:
                df = self.to_df(equal_weight=True)
                labels = [c for c in df.columns if c not in reserved]
                data = df[labels].values
                fig = corner.corner(
                    data, labels=labels,
                    show_titles=True, title_fmt=".2f",
                    title_kwargs={"fontsize": 10}, label_kwargs={"fontsize": 9},
                    plot_datapoints=True, hist_kwargs={"color": "C0"},
                    contour_kwargs={"colors": ["C0"]}
                )

            plt.tight_layout()
            if save_name:
                plt.savefig(save_name + "_corner.png")
            plt.close()

            if diagnostics:
                self.ns.plot_diagnostics(self.results,
                                         save_name=None if save_name is None else save_name + "_diagnostics.png")

    return Sampler(ns)
