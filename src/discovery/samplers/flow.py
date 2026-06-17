import numpy as np
from scipy.special import logsumexp

import jax
import jax.numpy as jnp

from .. import prior


def makesampler_flow(mylogl, priordict={}, flow_layers=16, knots=9, tanh_max_val=3.0,
                     num_samples=1024, multibatch=1, learning_rate=1e-2, steps=1001, anneal_steps=500,
                     beta0=0.5, annealing_schedule=None,
                     num_posterior=1024, logl_batch=None, show_progress=True):
    """Variational normalizing-flow sampler with the makesampler_nuts/_nested interface.

    Fits a triangular-spline normalizing flow to the (uniform-prior transformed) posterior by
    ELBO maximisation, then draws posterior samples from the trained flow. Returns an object with
    ``run(key)``, ``to_df()``, ``estimate_evidence()`` and ``make_plots(save_name)`` so it drops
    into the same pipeline as the NUTS (:mod:`discovery.samplers.numpyro`) and nested
    (:mod:`discovery.samplers.jaxns`) samplers. A sufficiently flexible flow can represent
    multiple modes, but note the ELBO is reverse-KL (mode-seeking), so it can still miss/under-
    weight modes -- cross-check multimodal results against nested sampling.

    Memory note
    -----------
    Peak memory scales with ``num_samples`` (the ELBO Monte-Carlo batch is vmapped *and*
    differentiated). ``multibatch`` instead accumulates gradients over sequential minibatches, so
    it raises the *effective* number of ELBO samples (``num_samples * multibatch``) **without**
    raising peak memory. For large models (many components, band/legendre variable bases) that hit
    out-of-memory, lower ``num_samples`` and raise ``multibatch`` (e.g. num_samples=64,
    multibatch=4 ~ 256 effective samples at ~1/4 the peak memory of num_samples=256).

    Parameters
    ----------
    mylogl : callable
        A discovery log-likelihood (e.g. ``model.logL``) with a ``.params`` attribute.
    priordict : dict
        Prior overrides merged over ``prior.priordict_standard`` (as for the other samplers).
    flow_layers, knots, tanh_max_val :
        Triangular-spline flow architecture (see flowjax.flows.triangular_spline_flow).
    num_samples : int
        ELBO Monte-Carlo draws per minibatch (the vmapped+differentiated batch; sets peak memory).
    multibatch : int
        Number of sequential minibatches accumulated per optimisation step (raises effective
        sample count without raising peak memory).
    learning_rate, steps, anneal_steps :
        Adam learning rate, number of optimisation steps, and the beta-annealing horizon (the
        iteration at which the tempered target reaches the full posterior, beta = 1).
    beta0 : float
        Initial inverse temperature for the default linear schedule
        ``beta = min(1, beta0 + (1 - beta0)*i/anneal_steps)``. Training starts on the flattened
        target ``p**beta0`` -- a low ``beta0`` shrinks the valleys between separated modes so the
        flow can spread mass across multiple basins before the target sharpens, which helps it
        capture multiple modes. Lower it further (e.g. 0.02-0.05) for deeper inter-mode barriers,
        at the cost of more exploration time. Ignored if ``annealing_schedule`` is given.
    annealing_schedule : callable, optional
        Custom ``beta(iteration)`` schedule; overrides ``beta0``/``anneal_steps``.
    num_posterior : int
        Number of samples drawn from the trained flow for the returned chain / evidence estimate.
    logl_batch : int or None
        Batch size when evaluating the model log-likelihood / target on drawn samples in
        ``to_df``/``estimate_evidence``. Defaults to ``num_samples`` (a forward-only eval at that
        size is guaranteed to fit, since training did forward+backward at it); lower it further if
        the post-processing still runs tight on memory.
    show_progress : bool
        Show the training progress bar.
    """
    from flowjax.flows import triangular_spline_flow
    from flowjax.distributions import StandardNormal
    from ..flow import VariationalFit, value_and_grad_ElboLoss

    logx = prior.makelogtransform_uniform(mylogl, priordict=priordict)
    loss = value_and_grad_ElboLoss(logx, num_samples=num_samples)

    # Evaluate post-hoc log-likelihoods in batches no larger than the training batch: a
    # forward-only eval at num_samples is guaranteed to fit since training (forward+backward at
    # the same size) did. Larger batches can OOM at the very end (in to_df / estimate_evidence).
    if logl_batch is None:
        logl_batch = num_samples

    def _batched(fn, samples):
        f = jax.jit(jax.vmap(fn))
        return np.concatenate([np.asarray(f(samples[i:i + logl_batch]))
                               for i in range(0, samples.shape[0], logl_batch)])

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
            schedule = annealing_schedule
            if schedule is None:
                schedule = lambda i: min(1.0, beta0 + (1.0 - beta0) * i / anneal_steps)
            trainer = VariationalFit(dist=flow, loss_fn=loss, multibatch=multibatch,
                                     learning_rate=learning_rate,
                                     annealing_schedule=schedule,
                                     show_progress=show_progress)
            self.train_key, self.trained_flow = trainer.run(train_key, steps=steps)
            self.losses = trainer.losses

        def to_df(self, with_logl=True):
            if self.trained_flow is None:
                raise RuntimeError("Run the flow sampler before accessing samples.")
            samples = self.trained_flow.sample(self.train_key, sample_shape=(num_posterior,))
            df = self.logx.to_df(samples).drop(columns=['logl'], errors='ignore')
            if with_logl:
                # model log-likelihood per sample (batched; the memory-heavy step). Use
                # with_logl=False to get the cheap parameter samples without this evaluation.
                df['logl'] = _batched(self.logx.logL, samples)
            return df

        def estimate_evidence(self, n=None, key=None):
            """Importance-sampling evidence estimate using the trained flow as the proposal.

            ``logZ = logmeanexp(logtarget - logq)`` where ``logtarget`` is the trained target
            ``logx(ys) = logL + log[pi(theta)|dtheta/dys|]`` (the uniform-prior normalization is
            already included in the transform, so this integrates directly to the normalized
            evidence) and ``logq`` is the flow density. Returns ``logZ``, an approximate standard
            error, and the importance-sampling effective sample size -- a flow-quality diagnostic:
            ``ess`` much less than ``n`` means the flow matches the posterior poorly.

            WARNING: if the flow has missed a posterior mode (the ELBO is mode-seeking), this
            UNDERESTIMATES logZ and the ESS can still look fine. For multimodal posteriors,
            cross-check the evidence against nested sampling (jaxns).
            """
            if self.trained_flow is None:
                raise RuntimeError("Run the flow sampler before estimating the evidence.")
            n = int(n or num_posterior)
            if key is None:
                key = jax.random.split(self.train_key)[0]
            samples, logq = self.trained_flow.sample_and_log_prob(key, (n,))
            logq = np.asarray(logq)
            logtarget = _batched(self.logx, samples)          # logL + log[pi(theta)|dtheta/dys|]
            logw = logtarget - logq                            # log importance weights
            logmeanw = logsumexp(logw) - np.log(n)
            relvar = max(float(np.exp(logsumexp(2 * logw) - np.log(n) - 2 * logmeanw) - 1.0), 0.0)
            return {'logZ': float(logmeanw),
                    'logZ_err': float(np.sqrt(relvar / n)),
                    'ess': float(n / (1.0 + relvar)),
                    'ess_frac': float(1.0 / (1.0 + relvar)),
                    'n': n}

        def make_plots(self, save_name=None, diagnostics=False):
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import corner
            import re

            df = self.to_df()
            reserved = [r'^logl$', r'^logZ.*$', r'^ess$', r'^(.*_)?alpha_scaling\[\d+\]$']
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
