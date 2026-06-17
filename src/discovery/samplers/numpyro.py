import inspect
import pickle
import warnings
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import pandas as pd

import numpyro
from numpyro import infer
from numpyro import distributions as dist

from .. import prior
from ..pulsar import save_chain


def makemodel_transformed(mylogl, transform=prior.makelogtransform_uniform, priordict={}):
    logx = transform(mylogl, priordict=priordict)

    parlen = sum(int(par[par.index('(')+1:par.index(')')]) if '(' in par else 1 for par in logx.params)

    def numpyro_model():
        pars = numpyro.sample('pars', dist.Normal(0, 10).expand([parlen]))
        logl = logx.logL(pars)
        numpyro.deterministic('log_likelihood', logl)
        numpyro.factor('logl', logx(pars))
    numpyro_model.to_df = lambda chain: logx.to_df(chain['pars'])
    # expose the transform and dimensionality so samplers can map physical starting values
    # into the unconstrained 'pars' vector (used by run_nuts_multistart)
    numpyro_model.transform = logx
    numpyro_model.parlen = parlen

    return numpyro_model


def makemodel(mylogl, priordict={}):
    def numpyro_model():
        logl = mylogl({par: numpyro.sample(par, dist.Uniform(*prior.getprior_uniform(par, priordict)))
                       for par in mylogl.params})
        numpyro.deterministic('log_likelihood', logl)
        numpyro.factor('logl', logl)
    numpyro_model.to_df = lambda chain: pd.DataFrame(chain)

    return numpyro_model


def makesampler_nuts(numpyro_model, num_warmup=512, num_samples=1024, num_chains=1, **kwargs):
    nutsargs = dict(max_tree_depth=8, dense_mass=False,
                    forward_mode_differentiation=False, target_accept_prob=0.8,
                    **{arg: val for arg in kwargs.items() if arg in inspect.getfullargspec(infer.NUTS).args})

    mcmcargs = dict(num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains,
                    chain_method='vectorized', progress_bar=True,
                    **{arg: val for arg in kwargs.items() if arg in inspect.getfullargspec(infer.MCMC).kwonlyargs})

    sampler = infer.MCMC(infer.NUTS(numpyro_model, **nutsargs), **mcmcargs)

    def _to_df():
        samples = sampler.get_samples()

        df = numpyro_model.to_df(samples)

        if 'log_likelihood' in samples:
            df = df.drop(columns=['log_likelihood'], errors='ignore')
            df['logl'] = np.asarray(samples['log_likelihood'])

        return df

    def _make_plots(save_name=None, diagnostics=False):
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
        import corner
        import re

        df = sampler.to_df()
        reserved = [r'^logl$', r'^(.*_)?alpha_scaling\[\d+\]$'] # don't plot likelihood or outlier parameters
        labels = [c for c in df.columns if not any(re.match(r, c) for r in reserved)]
        data = df[labels].values

        fig = corner.corner(
            data,
            labels=labels,
            show_titles=True,
            title_fmt=".2f",
            title_kwargs={"fontsize": 10},
            label_kwargs={"fontsize": 9},
            plot_datapoints=True,
            hist_kwargs={"color": "C0"},
            contour_kwargs={"colors": ["C0"]},
        )
        plt.tight_layout()
        if save_name:
            plt.savefig(f"{save_name}_corner.png")
        plt.close()

    sampler.to_df = _to_df
    sampler.make_plots = _make_plots

    return sampler


def print_summary(chain, prob=0.9, exclude=('logl', 'logZ', 'logZ_err', 'ess', 'weight', 'logposterior')):
    """Print per-parameter split-R-hat and effective sample size for a posterior chain.

    Runs numpyro's diagnostics on the full ``chain`` DataFrame (named physical parameters). This is
    used instead of ``MCMC.print_summary`` because after checkpointed or multistart runs the MCMC
    object only retains the final chunk (and reports the raw ``pars`` vector, not the physical
    names). A single chain is split into two halves to give the split-R-hat convergence diagnostic;
    ``n_eff`` is the effective sample size accounting for autocorrelation.
    """
    import numpyro.diagnostics as diag
    cols = [c for c in chain.columns if c not in exclude]
    n = len(chain)
    h = n // 2
    if h < 2 or not cols:
        print("print_summary: chain too short (or no parameters) for diagnostics.")
        return
    samples = {c: np.asarray(chain[c].to_numpy()[:2 * h], dtype=float).reshape(2, h) for c in cols}
    diag.print_summary(samples, prob=prob, group_by_chain=True)


def run_nuts_with_checkpoints(
    sampler,
    num_samples_per_checkpoint,
    rng_key,
    outdir="chains",
    resume=False,
    init_params=None,
):
    """Run NumPyro MCMC and save checkpoints.

    This function performs multiple iterations of MCMC sampling, saving checkpoints
    after each iteration. It saves samples to feather files and the NumPyro MCMC
    state to a pickle file.

    Parameters
    ----------
    sampler : numpyro.infer.MCMC
        A NumPyro MCMC sampler object.
    num_samples_per_checkpoint : int
        The number of samples to save in each checkpoint.
    rng_key : jax.random.PRNGKey
        The random number generator key for JAX.
    outdir : str | Path
        The directory for output files.
    resume : bool
        Whether to look for a state to resume from.
    init_params : dict, optional
        Initial parameter values for the chain, passed to ``sampler.warmup``. Used by
        :func:`run_nuts_multistart` to start chains in different basins. Ignored on resume.

    Returns
    -------
    None
        This function doesn't return any value but saves the results to disk.

    Side Effects
    ------------
    - Runs the warmup/adaptation phase once and checkpoints the post-warmup state before the
      first sampling chunk, so an interruption during sampling never has to repeat warmup.
    - Runs the MCMC sampler for the number of iterations required to reach the total sample number.
    - Saves samples data to feather files after each iteration.
    - Writes the NumPyro sampler state to a pickle file after each iteration.

    Example
    -------
    >>> import discovery.samplers.numpyro as ds_numpyro
    >>> # Assume `model` is configured
    >>> npsampler = ds_numpyro.makesampler_nuts(model, num_samples=100, num_warmup=50)
    >>> ds_numpyro.run_nuts_with_checkpoints(npsampler, 10, jax.random.key(42))

    """
    # convert to pathlib object and make the output directory if needed
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True, parents=True)

    samples_file = outdir / "numpyro-samples.feather"
    checkpoint_file = outdir / "numpyro-checkpoint.pickle"

    if resume and checkpoint_file.is_file():
        # A checkpoint exists, so warmup has already completed (its state is saved before the
        # first sampling chunk). Restore that state and skip warmup. The samples file may not
        # exist yet if the run was interrupted between warmup and the first chunk.
        with checkpoint_file.open("rb") as f:
            sampler.post_warmup_state = pickle.load(f)

        if samples_file.is_file():
            df = pd.read_feather(samples_file)
            num_samples_saved = df.shape[0]
        else:
            df = None
            num_samples_saved = 0

        total_sample_num = sampler.num_samples - num_samples_saved

    else:
        df = None
        num_samples_saved = 0
        total_sample_num = sampler.num_samples

        # Run the warmup/adaptation phase once and checkpoint the post-warmup state
        # immediately, before any sampling. Warmup itself cannot be resumed mid-way (its
        # step-size and mass-matrix adaptation schedule must run contiguously), but saving
        # here means an interruption during the first sampling chunk never repeats warmup.
        sampler.warmup(rng_key, init_params=init_params)
        with checkpoint_file.open("wb") as f:
            pickle.dump(sampler.post_warmup_state, f)
        rng_key, _ = jax.random.split(rng_key)

    num_checkpoints = int(jnp.ceil(total_sample_num / num_samples_per_checkpoint))
    remainder_samples = int(total_sample_num % num_samples_per_checkpoint)

    for checkpoint in range(num_checkpoints):
        if checkpoint == 0:
            sampler.num_samples = num_samples_per_checkpoint
            sampler._set_collection_params()  # Need this to update num_samples
        elif checkpoint == num_checkpoints - 1:
            # We won't need to update the collection params because we've set the post warmup state,
            # and that accomplishes the same goal.
            sampler.num_samples = remainder_samples if remainder_samples != 0 else num_samples_per_checkpoint

        sampler.run(rng_key)

        df_new = sampler.to_df()

        df = pd.concat([df, df_new]) if df is not None else df_new

        save_chain(df, samples_file)

        with checkpoint_file.open("wb") as f:
            pickle.dump(sampler.last_state, f)

        sampler.post_warmup_state = sampler.last_state

        rng_key, _ = jax.random.split(rng_key)


def _laplace_logz(df):
    """Approximate Laplace log-evidence of a single-mode chain (uniform prior).

    ``logZ ~ max(logL) + 0.5 * ( d*log(2*pi) + log|Cov| )``, omitting the common uniform-prior
    log-density so values are comparable across modes of the same model up to a shared constant.
    This is a Gaussian approximation and is biased for strongly curved/degenerate modes -- it is
    only used to flag when two modes have comparable posterior mass, not to reweight.
    """
    if 'logl' not in df.columns:
        return float('nan')
    if len(df) < 2:
        return float(df['logl'].max())
    param_cols = [c for c in df.columns if c != 'logl']
    X = np.asarray(df[param_cols].to_numpy(), dtype=float)
    X = X[:, X.std(axis=0) > 0]  # drop fixed columns
    if X.shape[1] == 0:
        return float(df['logl'].max())
    cov = np.atleast_2d(np.cov(X, rowvar=False))
    d = cov.shape[0]
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0 or not np.isfinite(logdet):
        # singular full covariance (e.g. d > n or perfect correlations): fall back to diagonal
        logdet = float(np.sum(np.log(np.clip(np.diag(cov), 1e-300, None))))
    return float(df['logl'].max() + 0.5 * (d * np.log(2.0 * np.pi) + logdet))


def run_nuts_multistart(
    sampler,
    model,
    init_grid,
    num_samples_per_checkpoint,
    rng_key,
    outdir="chains",
    resume=False,
    comparable_logz_threshold=5.0,
    mode_atol_sigma=4.0,
    scout_samples=100,
):
    """Run NUTS from several initialisations to escape a multimodal posterior.

    A single NUTS chain is a local sampler: it stays in whichever basin it starts in and
    cannot cross the low-probability valleys between well-separated modes (e.g. the band-noise
    alpha/gamma/log10_A modes). This starts one checkpointed NUTS chain per entry in
    ``init_grid`` -- each overriding chosen physical parameters at its starting point.

    By default it runs in two stages to avoid paying for the full sampling phase in every basin:
    a cheap **scout** stage runs warmup plus ``scout_samples`` samples for every start and scores
    each basin by its maximum log-likelihood; then only the **best** start is promoted to a full
    chain, *reusing its already-completed warmup* (it resumes from the scout checkpoint). Set
    ``scout_samples=None`` to disable scouting and run a full chain for every start instead.

    Scoring by the end-of-warmup/scout log-likelihood ranks basins by depth (peak), which agrees
    with ranking by posterior mass only when the modes are well separated -- exactly the regime
    where keeping the single best mode is valid. The per-mode Laplace ``logZ`` check below warns
    when that assumption breaks.

    Chains are run **sequentially**, each checkpointed in ``outdir/start{i}``, so peak memory
    stays that of a single chain (important for large PTA models). If you have the memory and
    prefer parallel chains, run a vectorised ``num_chains>1`` sampler instead.

    Parameters
    ----------
    sampler : numpyro.infer.MCMC
        Sampler from :func:`makesampler_nuts` (num_chains=1).
    model : callable
        The model from :func:`makemodel_transformed`; carries ``.transform`` and ``.parlen``,
        used to map physical starting values into the sampler's unconstrained ``pars`` vector.
    init_grid : list of dict
        One dict per chain, mapping a physical parameter name to its starting value. Names must
        be in ``model.transform.params`` and values must lie within their prior ranges.
        Parameters not listed start at their prior midpoint.
    num_samples_per_checkpoint : int
        Samples per checkpoint within the full chain (see :func:`run_nuts_with_checkpoints`).
    rng_key : jax.random.PRNGKey
        Base RNG key; each start uses ``jax.random.fold_in(rng_key, i)``.
    outdir : str | Path
        Parent directory; start ``i`` is checkpointed in ``outdir/start{i}``.
    resume : bool
        Passed through to each start, so interrupted starts resume from their checkpoints.
    comparable_logz_threshold : float
        If the two highest-mass distinct modes have Laplace ``logZ`` within this many nats, a
        warning is emitted: the single returned mode is then NOT representative of the posterior
        and a multimodal sampler should be used instead.
    mode_atol_sigma : float
        Starts whose MAP points agree to within this many within-chain standard deviations on
        every parameter are treated as the same mode (greedy clustering).
    scout_samples : int or None
        Samples drawn (after warmup) when scouting each basin before promoting the best start to
        a full chain. ``None`` disables scouting and runs a full chain for every start.

    Returns
    -------
    best_df : pandas.DataFrame
        Full posterior chain from the start with the highest maximum log-likelihood.
    summary : list of dict
        Per-start records ``{'start', 'init', 'max_logl', 'laplace_logz', 'mode', 'samples_file'}``
        ranked best-first. With scouting, all but the best are scout-length chains.

    Notes
    -----
    A per-mode Laplace ``logZ`` (Gaussian approximation, uniform prior) is estimated only to
    detect when modes have comparable posterior mass. When they do, returning the single best
    mode is statistically wrong (the modes should appear in proportion to their mass); reach for
    nested sampling (jaxns, the ``-ns`` option) or a normalizing-flow sampler (``discovery.flow``)
    instead. No reweighting across modes is performed here.
    """
    transform = model.transform
    valid = set(transform.params)
    # ys = 0 maps to the prior midpoint for every parameter (with correct shapes for vector params)
    base_phys = transform.to_dict(jnp.zeros(model.parlen))
    original_num_samples = sampler.num_samples

    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True, parents=True)

    # map each start's physical overrides into the unconstrained 'pars' init vector
    inits = []
    for i, overrides in enumerate(init_grid):
        phys = dict(base_phys)
        for name, val in overrides.items():
            if name not in valid:
                hint = [p for p in transform.params if 'band' in p] or list(transform.params)[:6]
                raise KeyError(f"run_nuts_multistart: '{name}' is not a model parameter. "
                               f"Examples of valid names: {hint}")
            phys[name] = val
        ys = transform.to_vec(phys)
        if not bool(jnp.all(jnp.isfinite(ys))):
            raise ValueError(f"run_nuts_multistart: start {i} maps to non-finite init values; "
                             f"check that overrides lie strictly within their priors: {overrides}")
        inits.append({'pars': ys})

    def _run_start(i, target, per_checkpoint, do_resume):
        sampler.num_samples = target          # checkpointing mutates this; set per call
        sampler.post_warmup_state = None       # fresh warmup (overwritten on resume)
        start_dir = outdir / f"start{i}"
        run_nuts_with_checkpoints(sampler, per_checkpoint, jax.random.fold_in(rng_key, i),
                                  outdir=start_dir, resume=do_resume, init_params=inits[i])
        samples_file = start_dir / "numpyro-samples.feather"
        return pd.read_feather(samples_file).reset_index(drop=True), samples_file

    scouting = scout_samples is not None
    stage = "scout" if scouting else "full"
    target = scout_samples if scouting else original_num_samples
    chunk = scout_samples if scouting else num_samples_per_checkpoint

    summary = []
    for i, overrides in enumerate(init_grid):
        print(f"[multistart] {stage} start {i+1}/{len(init_grid)}  init = {overrides}")
        df, samples_file = _run_start(i, target, chunk, resume)
        max_logl = float(df['logl'].max()) if 'logl' in df.columns else float('nan')
        print(f"[multistart] {stage} start {i+1}/{len(init_grid)} max logl = {max_logl:.2f}")
        summary.append({'start': i, 'init': overrides, 'max_logl': max_logl,
                        'samples_file': str(samples_file), '_df': df})

    # --- per-start Laplace log-evidence (relative across modes; Gaussian approximation) ---
    for r in summary:
        r['laplace_logz'] = _laplace_logz(r['_df'])

    # --- cluster starts into distinct modes by MAP proximity (in within-chain-sigma units) ---
    param_cols = [c for c in summary[0]['_df'].columns if c != 'logl']
    scales = np.array([max(float(np.median([r['_df'][c].std() for r in summary])), 1e-12)
                       for c in param_cols])
    maps = [r['_df'][param_cols].to_numpy(dtype=float)[r['_df']['logl'].to_numpy().argmax()]
            for r in summary]
    mode_reps = []  # representative MAP per distinct mode
    for i, m in enumerate(maps):
        mode = next((k for k, rep in enumerate(mode_reps)
                     if np.max(np.abs(m - rep) / scales) < mode_atol_sigma), None)
        if mode is None:
            mode = len(mode_reps)
            mode_reps.append(m)
        summary[i]['mode'] = mode

    # --- per-mode evidence = Laplace logZ of that mode's highest-likelihood member ---
    modes = {}
    for r in summary:
        k = r['mode']
        if k not in modes or r['max_logl'] > modes[k]['max_logl']:
            modes[k] = {'mode': k, 'max_logl': r['max_logl'], 'laplace_logz': r['laplace_logz']}
    ranked = sorted(modes.values(), key=lambda d: d['laplace_logz'], reverse=True)

    print(f"[multistart] found {len(modes)} distinct mode(s):")
    for d in ranked:
        print(f"    mode {d['mode']}: laplace_logZ = {d['laplace_logz']:.2f}, "
              f"max logl = {d['max_logl']:.2f}")

    if len(ranked) >= 2:
        gap = ranked[0]['laplace_logz'] - ranked[1]['laplace_logz']
        if gap < comparable_logz_threshold:
            msg = (f"run_nuts_multistart: the top two modes have COMPARABLE posterior mass "
                   f"(Laplace dlogZ = {gap:.2f} < {comparable_logz_threshold}). The single best "
                   f"mode returned here is NOT representative of the full posterior; the modes "
                   f"should appear in proportion to their mass. Use a sampler that handles "
                   f"multimodality correctly -- nested sampling (jaxns, the -ns option) or a "
                   f"normalizing-flow sampler (discovery.flow) -- for this model.")
            print("\n*** WARNING: " + msg + "\n")
            warnings.warn(msg, RuntimeWarning, stacklevel=2)
        else:
            print(f"[multistart] best mode is clearly separated (Laplace dlogZ = {gap:.2f} >= "
                  f"{comparable_logz_threshold}); returning it is appropriate.")

    # best start = highest maximum log-likelihood (basin depth)
    summary.sort(key=lambda r: (r['max_logl'] if np.isfinite(r['max_logl']) else -np.inf), reverse=True)
    best = summary[0]

    # when scouting, promote only the winner to a full chain, reusing its warmup (resume from
    # the scout checkpoint). The losing starts keep their scout-length chains.
    if scouting:
        print(f"[multistart] promoting best scout start {best['start']+1} (mode {best['mode']}) "
              f"to a full chain of {original_num_samples} samples...")
        df, samples_file = _run_start(best['start'], original_num_samples,
                                      num_samples_per_checkpoint, do_resume=True)
        best['_df'] = df
        best['samples_file'] = str(samples_file)
        best['max_logl'] = float(df['logl'].max()) if 'logl' in df.columns else best['max_logl']

    print(f"[multistart] best start = {best['start']} (mode {best['mode']}, "
          f"max logl = {best['max_logl']:.2f}); init = {best['init']}")

    best_df = best['_df']
    for r in summary:
        r.pop('_df', None)  # don't return heavy frames in the summary
    return best_df, summary
