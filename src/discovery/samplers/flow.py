import numpy as np
from scipy.special import logsumexp

import jax
import jax.numpy as jnp

from .. import prior


def makesampler_flow(mylogl, priordict={}, flow_layers=16, knots=9, tanh_max_val=3.0,
                     num_samples=1024, multibatch=1, learning_rate=1e-2, steps=1001, anneal_steps=500,
                     beta0=0.1, annealing_schedule=None, base_loc=None,
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
    base_loc : array, optional
        Mean of the flow's (Gaussian) base distribution in the unconstrained ``ys`` space, used to
        seed training toward a region. Defaults to a zero-mean StandardNormal base. Set by
        :func:`run_flow_multistart`; see there.
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
    from flowjax.distributions import StandardNormal, Normal
    from ..flow import VariationalFit, value_and_grad_ElboLoss

    # The ELBO is a Monte-Carlo mean of log-posterior differences of order 1e4-1e5, and the
    # evidence is an importance-sampling estimate over the same quantity. Single precision
    # leaves ~1e-2 nats of rounding noise per term, which both slows ELBO convergence and
    # biases logZ. discovery enables x64 on import; fail loudly if that has been undone.
    if not jax.config.jax_enable_x64:
        raise RuntimeError(
            "makesampler_flow: jax_enable_x64 is False. The ELBO and the importance-sampling "
            "evidence are built from log-posterior differences of order 1e4-1e5, which lose "
            "~1e-2 nats to rounding in float32. Call "
            "jax.config.update('jax_enable_x64', True) before building the sampler."
        )

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
            if base_loc is None:
                parlen = sum(int(p[p.index('(') + 1:p.index(')')]) if '(' in p else 1
                             for p in logx.params)
                base = StandardNormal((parlen,))
            else:
                bl = jnp.asarray(base_loc, dtype=float)
                base = Normal(loc=bl, scale=jnp.ones_like(bl))   # seed the start toward a region
            flow = triangular_spline_flow(flow_key, base_dist=base,
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


def run_flow_multistart(mylogl, init_grid, rng_key, priordict={}, evidence_n=None, **flow_kwargs):
    """Train several normalizing flows from dispersed starts and keep the highest-evidence one.

    The ELBO is reverse-KL (mode-seeking), so a single flow can settle in a sub-dominant mode and
    miss the dominant one (which then shows up as a much lower max log-likelihood and evidence).
    This is the flow analogue of :func:`discovery.samplers.numpyro.run_nuts_multistart`: it trains
    one flow per entry in ``init_grid`` -- each seeded toward a different region by shifting the
    flow's base mean (entries may be ``None`` for a plain prior-centred restart) -- estimates each
    flow's evidence by importance sampling, and returns the flow with the highest ``logZ``. Because
    a dominant mode's evidence is far larger, ``logZ`` selects it unambiguously.

    Note that base-seeding only *diversifies* the starting region (the untrained spline does not
    preserve the base mean exactly), so it improves coverage rather than guaranteeing a flow lands
    exactly on a given mode; the ``logZ`` ranking is what makes the selection robust.

    Parameters
    ----------
    mylogl : callable
        Discovery log-likelihood (e.g. ``model.logL``) with a ``.params`` attribute.
    init_grid : list of (dict or None)
        One entry per restart. A dict maps physical parameter names to starting values (others
        default to the prior midpoint); ``None`` starts from the standard prior-centred base.
    rng_key : jax.random.PRNGKey
        Base key; restart ``i`` uses ``jax.random.fold_in(rng_key, i)``.
    priordict : dict
        Prior overrides merged onto ``prior.priordict_standard``.
    evidence_n : int or None
        Samples for each restart's evidence estimate (defaults to the flow's ``num_posterior``).
    **flow_kwargs :
        Forwarded to :func:`makesampler_flow` (flow_layers, knots, num_samples, multibatch, steps,
        beta0, ...).

    Returns
    -------
    best_sampler : FlowSampler
        The trained flow sampler with the highest ``logZ``.
    best_chain : pandas.DataFrame
        Its posterior samples (with ``logl`` and ``logZ``/``logZ_err``/``ess`` columns).
    summary : list of dict
        Per-restart records ``{'start', 'init', 'logZ', 'logZ_err', 'ess'}`` ranked best-first.
    """
    logx = prior.makelogtransform_uniform(mylogl, priordict=priordict)
    valid = set(logx.params)
    parlen = sum(int(p[p.index('(') + 1:p.index(')')]) if '(' in p else 1 for p in logx.params)
    base_phys = logx.to_dict(jnp.zeros(parlen))   # prior midpoints

    summary = []
    for i, overrides in enumerate(init_grid):
        if overrides is None:
            base_loc, label = None, 'prior-centred'
        else:
            phys = dict(base_phys)
            for name, val in overrides.items():
                if name not in valid:
                    raise KeyError(f"run_flow_multistart: '{name}' is not a model parameter.")
                phys[name] = val
            ys = logx.to_vec(phys)
            if not bool(jnp.all(jnp.isfinite(ys))):
                raise ValueError(f"run_flow_multistart: start {i} maps to non-finite init values; "
                                 f"check that overrides lie strictly within their priors: {overrides}")
            base_loc, label = np.asarray(ys), overrides

        print(f"[flow-multistart] restart {i}/{len(init_grid) - 1}  seed = {label}")
        s = makesampler_flow(mylogl, priordict=priordict, base_loc=base_loc, **flow_kwargs)
        s.run(jax.random.fold_in(rng_key, i))
        ev = s.estimate_evidence(n=evidence_n)
        print(f"[flow-multistart] restart {i}: logZ = {ev['logZ']:.2f} +/- {ev['logZ_err']:.2f}  "
              f"ess = {ev['ess']:.0f}/{ev['n']}")
        summary.append({'start': i, 'init': overrides, 'logZ': ev['logZ'],
                        'logZ_err': ev['logZ_err'], 'ess': ev['ess'], '_sampler': s})

    summary.sort(key=lambda r: r['logZ'], reverse=True)
    best = summary[0]
    print(f"[flow-multistart] best restart = {best['start']} (logZ = {best['logZ']:.2f}); "
          f"seed = {best['init']}")

    best_sampler = best['_sampler']
    best_chain = best_sampler.to_df()
    ev = best_sampler.estimate_evidence(n=evidence_n)
    best_chain['logZ'], best_chain['logZ_err'], best_chain['ess'] = ev['logZ'], ev['logZ_err'], ev['ess']
    for r in summary:
        r.pop('_sampler', None)
    return best_sampler, best_chain, summary
