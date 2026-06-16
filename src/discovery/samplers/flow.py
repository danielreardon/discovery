import numpy as np

import jax
import jax.numpy as jnp

from .. import prior


def makesampler_flow(mylogl, priordict={}, flow_layers=16, knots=9, tanh_max_val=3.0,
                     num_samples=1024, learning_rate=1e-2, steps=1001, anneal_steps=500,
                     num_posterior=8192, logl_batch=1024, show_progress=True):
    """Variational normalizing-flow sampler with the makesampler_nuts/_nested interface.

    Fits a triangular-spline normalizing flow to the (uniform-prior transformed) posterior by
    ELBO maximisation, then draws posterior samples from the trained flow. Returns an object with
    ``run(key)``, ``to_df()`` and ``make_plots(save_name)`` so it drops into the same pipeline as
    the NUTS (:mod:`discovery.samplers.numpyro`) and nested (:mod:`discovery.samplers.jaxns`)
    samplers. Unlike a single NUTS chain, a sufficiently flexible flow can represent multiple
    modes, so this is an alternative for multimodal posteriors.

    Parameters
    ----------
    mylogl : callable
        A discovery log-likelihood (e.g. ``model.logL``) with a ``.params`` attribute.
    priordict : dict
        Prior overrides merged over ``prior.priordict_standard`` (as for the other samplers).
    flow_layers, knots, tanh_max_val :
        Triangular-spline flow architecture (see flowjax.flows.triangular_spline_flow).
    num_samples : int
        Number of ELBO Monte-Carlo draws per optimisation step.
    learning_rate, steps, anneal_steps :
        Adam learning rate, number of optimisation steps, and the linear beta-annealing horizon
        (``beta = min(1, 0.5 + 0.5*i/anneal_steps)``).
    num_posterior : int
        Number of samples drawn from the trained flow for the returned chain.
    logl_batch : int
        Batch size used when evaluating the model log-likelihood on the drawn samples.
    show_progress : bool
        Show the training progress bar.
    """
    from flowjax.flows import triangular_spline_flow
    from flowjax.distributions import StandardNormal
    from ..flow import VariationalFit, value_and_grad_ElboLoss

    logx = prior.makelogtransform_uniform(mylogl, priordict=priordict)
    loss = value_and_grad_ElboLoss(logx, num_samples=num_samples)

    class FlowSampler:
        def __init__(self):
            self.logx = logx
            self.trained_flow = None
            self.losses = None

        def run(self, key):
            # VariationalFit AOT-compiles with a typed key (jax.random.key); accept either a
            # typed or a legacy PRNGKey by reseeding into a typed key deterministically.
            key = jax.random.key(int(jax.random.bits(key, shape=(), dtype=jnp.uint32)))
            flow_key, train_key = jax.random.split(key)
            flow = triangular_spline_flow(flow_key, base_dist=StandardNormal((len(logx.params),)),
                                          cond_dim=None, flow_layers=flow_layers, knots=knots,
                                          tanh_max_val=tanh_max_val, invert=False, init=None)
            trainer = VariationalFit(dist=flow, loss_fn=loss, multibatch=1,
                                     learning_rate=learning_rate,
                                     annealing_schedule=lambda i: min(1.0, 0.5 + 0.5 * i / anneal_steps),
                                     show_progress=show_progress)
            self.train_key, self.trained_flow = trainer.run(train_key, steps=steps)
            self.losses = trainer.losses

        def to_df(self):
            if self.trained_flow is None:
                raise RuntimeError("Run the flow sampler before accessing samples.")
            samples = self.trained_flow.sample(self.train_key, sample_shape=(num_posterior,))
            df = self.logx.to_df(samples).drop(columns=['logl'], errors='ignore')
            # model log-likelihood per sample (batched to bound memory), for write_ml_json/plots
            logl_fn = jax.jit(jax.vmap(self.logx.logL))
            ll = np.concatenate([np.asarray(logl_fn(samples[i:i + logl_batch]))
                                 for i in range(0, samples.shape[0], logl_batch)])
            df['logl'] = ll
            return df

        def make_plots(self, save_name=None, diagnostics=False):
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import corner
            import re

            df = self.to_df()
            reserved = [r'^logl$', r'^(.*_)?alpha_scaling\[\d+\]$']
            labels = [c for c in df.columns if not any(re.match(r, c) for r in reserved)]
            corner.corner(df[labels].values, labels=labels, show_titles=True, title_fmt=".2f",
                          title_kwargs={"fontsize": 10}, label_kwargs={"fontsize": 9},
                          plot_datapoints=True, hist_kwargs={"color": "C0"},
                          contour_kwargs={"colors": ["C0"]})
            plt.tight_layout()
            if save_name:
                plt.savefig(f"{save_name}_corner.png")
            plt.close()

            if diagnostics and save_name and self.losses is not None:
                plt.figure()
                plt.plot(self.losses)
                plt.xlabel("iteration")
                plt.ylabel("ELBO loss")
                plt.savefig(f"{save_name}_flowloss.png")
                plt.close()

    return FlowSampler()
