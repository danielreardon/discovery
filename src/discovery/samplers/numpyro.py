import inspect
import numpy as np
import pandas as pd

import numpyro
from numpyro import infer
from numpyro import distributions as dist

from .. import prior


def makemodel_transformed(mylogl, transform=prior.makelogtransform_uniform, priordict={}):
    logx = transform(mylogl, priordict=priordict)

    parlen = sum(int(par[par.index('(')+1:par.index(')')]) if '(' in par else 1 for par in logx.params)

    def numpyro_model():
        pars = numpyro.sample('pars', dist.Normal(0, 10).expand([parlen]))
        logl = logx.logL(pars)
        numpyro.deterministic('log_likelihood', logl)
        numpyro.factor('logl', logx(pars))
    numpyro_model.to_df = lambda chain: logx.to_df(chain['pars'])

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
