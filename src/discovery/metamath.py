"""metamath: graph-based kernel/GP classes built on the metamatrix DSL.

Kernel-math methods in this file return metamatrix **graphs**, not closures.
Sub-objects (`N.make_solve`, `P.make_inv`, reparams, ExtSignal coeff maps,
priors, ...) compose into those graphs as leaves so `fold_constants` can
decide what runs at trace time vs runtime. `mm.func` belongs at the outer
boundary in `likelihood.py`, not inside methods here. `make_sample` is the
one documented exception.

See `dev_architecture/metamatrix/metamatrix_architecture.md` for the full
conventions and porting guidance before adding new methods.
"""

import functools

import jax
import jax.numpy as jnp
import jax.scipy as jsp

from . import signals
from . import metamatrix as mm
from .utils import (
    Kernel,
    make_uind,
    smup_ind_correct,
    vsmup_ind_correct,
    vsmdp_ind,
)



@mm.graph
def noisesolve(graph, y, N):
    result = N.solve(y)


@mm.graph
def noiseinvfunc(graph, P):
    result = P


@mm.graph
def noiseinv(graph, P):
    result = P.inv()


@mm.graph
def normal(g, y, Nsolve):
    Nmy, lN = Nsolve(y).split()
    logp = -0.5 * (y.T @ Nmy) - 0.5 * lN


# this is actually pretty inefficient on CPU
stacksolve = False

@mm.graph
def woodbury(g, y, Nsolve, F, Pinv):
    if stacksolve:
        (Nmy, NmF), lN = g.stacksolve(Nsolve, y, F)
    else:
        Nmy, lN = Nsolve(y)
        NmF, _ = Nsolve(F)

    FtNmy = g.dot(NmF, y) # FtNmy = g.dot(F, Nmy)
    FtNmF = g.dot(F, NmF)

    Pm, lP = Pinv # should be a call even without parameters
    cf, lS = g.cho_factor(Pm + FtNmF)
    mu = g.cho_solve(cf, FtNmy)

    # Single-precision (stage 2): pin the white-noise logdet lN and the prior
    # logdet lP to float64. lS (the small Phi^-1 + FtNmF Cholesky logdet) is left
    # in the working dtype on purpose -- pinning it would force that Cholesky into
    # float64, which is the cost we are avoiding. ld stays working dtype: it is the
    # final sum, and protecting that sum is Piece 2's job, not the pins'.
    g.pin_f64(lN)
    g.pin_f64(lP)
    ld = g.node(lambda lN, lP, lS: lN + lP + lS, [lN, lP, lS], description=f'{lN.name} + {lP.name} + {lS.name}')

    cond = g.pair(mu, cf, name='cond')
    solve = g.pair(Nmy - NmF @ mu, ld, name='solve')

    # ytNmy = y^T N^-1 y is the catastrophic-cancellation term: pin it to float64
    # so its ntoa-long accumulation is built accurately even under a float32
    # working dtype.
    ytNmy = g.pin_f64(g.dot(y, Nmy))
    # Half A: combine the final logL in float64 -- ytNmy/lN/lP are read in as f64
    # (no loss), FtNmy.mu and lS are cast up -- so the O(1) result is not rounded
    # to the float32 ulp of a ~1e6 term. The float32 Cholesky / mu upstream are
    # untouched (that accuracy is Half B / reference+delta).
    logp = g.combine_logp_f64(ytNmy, g.dot(FtNmy, mu), [lN, lP, lS])

    # more readable, but doing this keeps y.T @ Nmy from being cached
    # logp = g.node(lambda y, Nmy, FtNmy, mu, ld: -0.5 * (y.T @ Nmy - FtNmy.T @ mu) - 0.5 * ld,
    #               [y, Nmy, FtNmy, mu, ld], description=f'-0.5 * (y.T @ Nmy - FtNmy.T @ mu) - 0.5 * ld')


@mm.graph
def woodburykernelsolve(g, y, T, Nsolve, F, Pinv):
    """Return (T^T Σ^-1 y, T^T Σ^-1 T) for Σ = N + F P F^T.

    Used by GlobalLikelihood.conditional to project per-pulsar Woodburys
    onto the global GP basis T.
    """
    Nmy, _ = Nsolve(y)
    NmF, _ = Nsolve(F)
    NmT, _ = Nsolve(T)

    FtNmy = g.dot(NmF, y)   # F.T @ N^-1 y
    FtNmF = g.dot(F, NmF)   # F.T @ N^-1 F
    FtNmT = g.dot(NmF, T)   # F.T @ N^-1 T  (N symmetric, so == (N^-1 F)^T T)

    TtNmy = g.dot(NmT, y)
    TtNmF = g.dot(NmT, F)
    TtNmT = g.dot(NmT, T)

    Pm, lP = Pinv
    cf, lS = g.cho_factor(Pm + FtNmF)

    TtSy = TtNmy - TtNmF @ g.cho_solve(cf, FtNmy)
    TtST = TtNmT - TtNmF @ g.cho_solve(cf, FtNmT)

    result = g.pair(TtSy, TtST)


@mm.graph
def woodburylatent(g, y, Nsolve, F, Psolve, getc):
    c = getc

    yp = y - F @ c
    Nmyp, lN = Nsolve(yp)

    Pmc, lP = Psolve(c)

    logp = -0.5 * (yp.T @ Nmyp + c.T @ Pmc + lP + lN)


@mm.graph
def globalwoodbury(g, ys, Nsolves, Fs, Pinv):
    ytNmys, FtNmys, FtNmFs = [], [], []

    if isinstance(Nsolves, (tuple, list)):
        for y, F, Nsolve in zip(ys, Fs, Nsolves):
            Nmy, lN = Nsolve(y)
            NmF, _ = Nsolve(F)

            ytNmys.append(g.dot(y, Nmy) + lN)
            FtNmys.append(g.dot(NmF, y))
            # FtNmys.append(g.dot(F, Nmy))
            FtNmFs.append(g.dot(F, NmF))
    else:
        # ingest output from vectorwoodburysolve
        Nmy_lNs = Nsolves(*ys)
        NmF_lNs = Nsolves(*Fs)

        # we can't include Nmy_lNs/NmF_lNs in the zip
        # since it's a tuple Sym that doesn't know its length
        for i, (y, F) in enumerate(zip(ys, Fs)):
            Nmy, lN = Nmy_lNs[i].split()
            NmF, _  = NmF_lNs[i].split()

            ytNmys.append(g.dot(y, Nmy) + lN)
            FtNmys.append(g.dot(F, Nmy))
            FtNmFs.append(g.dot(F, NmF))

    # Single-precision (stage 2): pin the array-wide data term and prior logdet to
    # float64, mirroring woodbury(). Each ytNmys element is (y^T N^-1 y + lN), so
    # pinning the sum_all pulls every per-pulsar quadratic + white-noise logdet --
    # and the y-path N -- into float64, where the cross-pulsar ytNmy cancellation
    # actually bites. The separate Nsolve(F) path stays working dtype, so the big
    # FtNmF Cholesky remains float32. lS is left working dtype on purpose (pinning
    # it would force that Cholesky into float64).
    ytNmy = g.pin_f64(g.sum_all(ytNmys))
    FtNmy = g.hstack(FtNmys)
    FtNmF = g.block_diag(FtNmFs)

    Pm, lP = Pinv
    cf, lS = g.cho_factor(Pm + FtNmF)

    lP = g.pin_f64(lP.sum()) # g.node(lambda x: jnp.sum(x), inputs=[lP]) # should be an op

    logp = g.combine_logp_f64(ytNmy, g.dot(FtNmy, g.cho_solve(cf, FtNmy)), [lP, lS])  # Half A: f64 final combine

@mm.graph
def globalwoodbury_fused(g, projected, Pinv):
    ytNmy_proj, ld, FtNmy_proj, FtNmF_proj = (projected[0], projected[1],
                                                projected[2], projected[3])
    ytNmy = g.sum(ytNmy_proj)
    total_ld = g.sum(ld)
    FtNmy = g.node(lambda x: x.reshape(-1), [FtNmy_proj])         # (67*28,)
    FtNmF = g.node(lambda x: jsp.linalg.block_diag(*x), [FtNmF_proj])  # (1876, 1876)
    Pm, lP = Pinv
    cf, lS = g.cho_factor(Pm + FtNmF)
    # Outer (global GP) prior logdet -> float64. The outer Cholesky logdet lS and
    # total_ld stay working dtype (pinning lS would force the global f64 Cholesky;
    # ytNmy is not pinned here because its cone reaches the inner Cholesky via
    # ytNmy_proj -- the raw accumulation is already pinned at source in jointsolve).
    logp = g.combine_logp_f64(ytNmy, g.dot(FtNmy, g.cho_solve(cf, FtNmy)),
                              [g.pin_f64(lP.sum()), total_ld, lS])  # Half A: f64 final combine


@mm.graph
def vectorwoodburyjointsolve(g, ys, Fs_outer, Nsolves, Fs_inner, Pinv):
    """Jointly solve — provides both TOA-space and projected outputs."""
    Nmys, NmFs_out, NmFs_in = [], [], []
    FtNmys_in, FtNmFs_in = [], []
    lNs = []

    for y, F_out, F_in, Nsolve in zip(ys, Fs_outer, Fs_inner, Nsolves):
        Nmy, lN = Nsolve(y)
        NmF_out, _ = Nsolve(F_out)
        NmF_in, _ = Nsolve(F_in)
        # Single-precision (stage 2): pin the per-pulsar white-noise logdet to
        # float64 at source (mirrors woodbury). The downstream combine into `ld`
        # stays working dtype -- that is Piece 2's job, not the pins'.
        lNs.append(g.pin_f64(lN))
        Nmys.append(Nmy)
        NmFs_out.append(NmF_out)
        NmFs_in.append(NmF_in)
        FtNmys_in.append(g.dot(F_in, Nmy))
        FtNmFs_in.append(g.dot(F_in, NmF_in))

    FtNmy_in = g.array(FtNmys_in)
    FtNmF_in = g.array(FtNmFs_in)

    Pm, lP = Pinv
    lP = g.pin_f64(lP)   # inner GP prior logdet -> float64 (pinned at source)
    cf, lS = g.cho_factor(Pm + FtNmF_in)
    mu_y = g.cho_solve(cf, FtNmy_in)

    FtNmFs_cross = [g.dot(F_in, NmF_out) for F_in, NmF_out in zip(Fs_inner, NmFs_out)]
    FtNmF_cross = g.array(FtNmFs_cross)
    mu_F = g.cho_solve(cf, FtNmF_cross)

    # --- TOA-space output (for make_solve etc.) ---
    solves = []
    for i in range(len(ys)):
        ld = lNs[i] + lP[i] + lS[i]
        y_corr = Nmys[i] - NmFs_in[i] @ mu_y[i, :]
        F_corr = NmFs_out[i] - NmFs_in[i] @ mu_F[i, :, :]
        solves.append(g.pair(g.pair(y_corr, ld), g.pair(F_corr, ld)))
    # if you just want the results normally you can prune to here
    g.named(g.ntuple(solves), 'result')

    # --- Projected output (for globalwoodbury_fused) ---
    # Pin each per-pulsar raw y^T N^-1 y to float64 at source (mirrors woodbury's
    # ytNmy pin). Cone = Nsolve(y) -> N only; the inner Cholesky cf is the *other*
    # operand of the ytNmy_proj subtraction, not an ancestor, so it stays float32.
    ytNmy_consts = g.array([g.pin_f64(g.dot(ys[i], Nmys[i])) for i in range(len(ys))])  # (67,)
    FtNmy_out = g.array([g.dot(Fs_outer[i], Nmys[i]) for i in range(len(ys))])       # (67, 28)
    FtNmF_cross_out = g.array([g.dot(Fs_outer[i], NmFs_in[i]) for i in range(len(ys))])  # (67, 28, 60)
    FtNmF_out = g.array([g.dot(Fs_outer[i], NmFs_out[i]) for i in range(len(ys))])   # (67, 28, 28)
    ld = g.array(lNs) + lP + lS                                                       # (67,)

    # Batched runtime: ~5 nodes instead of ~2000
    ytNmy_proj = ytNmy_consts - g.node(lambda a, b: jnp.einsum('ij,ij->i', a, b),
                                        [FtNmy_in, mu_y])                              # (67,)
    FtNmy_proj = FtNmy_out - g.node(lambda A, x: jnp.einsum('ijk,ik->ij', A, x),
                                     [FtNmF_cross_out, mu_y])                          # (67, 28)
    FtNmF_proj = FtNmF_out - FtNmF_cross_out @ mu_F                                   # (67, 28, 60) @ (67, 60, 28) = (67, 28, 28)

    g.named(g.ntuple([ytNmy_proj, ld, FtNmy_proj, FtNmF_proj]), 'projected')

@mm.graph
def vectorwoodbury(g, ys, Nsolves, Fs, Pinv):
    ytNmys, FtNmys, FtNmFs = [], [], []

    for y, F, Nsolve in zip(ys, Fs, Nsolves):
        Nmy, lN = Nsolve(y)
        NmF, _ = Nsolve(F)

        ytNmys.append(y.T @ Nmy + lN)
        FtNmys.append(F.T @ Nmy)
        FtNmFs.append(F.T @ NmF)

    # Single-precision (stage 2): same pins as globalwoodbury(). Each ytNmys
    # element is (y^T N^-1 y + lN); pin the array sum_all to float64 (the
    # cross-pulsar cancellation term) and the prior logdet sum. lS stays working
    # dtype so the batched FtNmF Cholesky remains float32.
    ytNmy = g.pin_f64(g.sum_all(ytNmys))
    FtNmy = g.array(FtNmys)
    FtNmF = g.array(FtNmFs)

    Pm, lP = Pinv
    cf, lS = g.cho_factor(Pm + FtNmF)
    mu = g.cho_solve(cf, FtNmy)

    cond = g.pair(mu, cf, name='cond')
    logp = g.combine_logp_f64(ytNmy, g.sum(FtNmy * mu), [g.pin_f64(lP.sum()), lS.sum()])  # Half A: f64 final combine


@mm.graph
def gaussian_coefficient_logprior(g, c_for_prior, Pinv):
    """Gaussian log-prior on per-pulsar GP coefficients: `-0.5 c.T Pm c - 0.5 sum(ldP)`.

    General-purpose; not tied to any particular compound topology. Used as the
    `prior` leaf of `vectorgpcomponent` whenever the prior precision collapses
    into a single `Pinv` leaf.

    Inputs:
      c_for_prior : (npsr, k)        — per-pulsar coefficient matrix.
      Pinv        : GraphLeaf returning (Pm, ldP) with
                      Pm  shape (npsr, k, k)  — batched precision per pulsar
                                                (batched-diagonal under
                                                 metamatrix.matrix_inv);
                      ldP shape (npsr,)       — per-pulsar log|Phi|.

    Output:
      scalar `logpr = -0.5 * einsum('ij,ijk,ik->', c, Pm, c) - 0.5 * sum(ldP)`.
    """
    Pm, ldP = Pinv
    quad = g.node(
        lambda c_, P_: jnp.einsum('ij,ijk,ik->', c_, P_, c_),
        [c_for_prior, Pm], description='c_for_prior.T @ Pm @ c_for_prior')
    logpr = -0.5 * quad - 0.5 * g.sum(ldP)


@mm.graph
def vectorgpcomponent(g, ys, Nsolves, Fs, prior, coeffs, means, ext_coeffs, ext_Fs):
    """ArrayLikelihood.clogL: log p(y, c) at fixed GP coefficients c.

    Mirrors `vectorwoodbury`'s per-pulsar precompute, but the GP coefficients
    come from outside (a FuncLeaf folding params -> (c, ldL)) instead of being
    solved for as the MAP. The prior on c is centered on `means(params)` when
    provided. ExtSignals contribute their own basis cross-terms.

    Leaves:
      ys         : per-pulsar residual ArgLeafs.
      Nsolves    : per-pulsar `N.make_solve` GraphLeafs (each (y) -> (Nmy, lN)).
      Fs         : per-pulsar GP basis (Const/Func/GraphLeafs).
      prior      : GraphLeaf `(c_for_prior) -> scalar logpr`. The single-Pinv
                   case wraps `P.make_inv` via `gaussian_coefficient_logprior`;
                   the mixed-Phi case (commongp + globalgp(HD)) supplies a
                   per-GP-sum graph built by `CompoundGP._build_mixed_logprior`.
                   Same interface.
      coeffs     : FuncLeaf params -> (c, ldL); c shape (npsr, k); ldL is the
                   summed log-|J| from reparams.
      means      : FuncLeaf params -> a0 of shape (npsr, k), or ConstLeaf(0.0)
                   if no means are configured.
      ext_coeffs : list of FuncLeafs (one per ExtSignal): params -> ccw.
      ext_Fs     : list of lists of per-pulsar Fcw leaves (one list per ExtSignal).

    Named outputs:
      'logp'   — scalar log-likelihood.
      'staged' — pair (logp, c); used when reparams were applied so the caller
                 can see the post-reparam coefficients.
    """
    NmFs, ldNs, ytNmys, NmFtys, FtNmFs = [], [], [], [], []
    for y, F, Nsolve in zip(ys, Fs, Nsolves):
        Nmy, lN = Nsolve(y)
        NmF, _ = Nsolve(F)
        NmFs.append(NmF)
        ldNs.append(lN)
        ytNmys.append(g.dot(y, Nmy))
        NmFtys.append(g.dot(NmF, y))   # F.T @ N^-1 y
        FtNmFs.append(g.dot(F, NmF))   # F.T @ N^-1 F

    ytNmy = g.sum_all(ytNmys)
    ldN   = g.sum_all(ldNs)
    NmFty = g.array(NmFtys)            # (npsr, k)
    FtNmF = g.array(FtNmFs)            # (npsr, k, k)

    c, ldL = coeffs.split()
    c_for_prior = c - means            # means is ConstLeaf(0.0) when absent

    logpr = prior(c_for_prior)         # Apply on GraphLeaf -> scalar

    quad_c = g.node(
        lambda c_, F_: jnp.einsum('ij,ijk,ik->', c_, F_, c_),
        [c, FtNmF], description='c.T @ FtNmF @ c')
    cNmFty = g.node(lambda c_, n_: jnp.sum(c_ * n_), [c, NmFty],
                    description='sum(c * NmFty)')
    data_term = -0.5 * ytNmy + cNmFty - 0.5 * quad_c - 0.5 * ldN

    logp = data_term + logpr + ldL
    for ccw, Fcw_list in zip(ext_coeffs, ext_Fs):
        NmFcws = [Nsolve(Fcw_i) for Nsolve, Fcw_i in zip(Nsolves, Fcw_list)]
        FcwNmy    = g.array([g.dot(NmFcw[0], y) for NmFcw, y in zip(NmFcws, ys)])
        FtNmFcw   = g.array([g.dot(F, NmFcw[0]) for F, NmFcw in zip(Fs, NmFcws)])
        FcwtNmFcw = g.array([g.dot(Fcw_i, NmFcw[0])
                             for Fcw_i, NmFcw in zip(Fcw_list, NmFcws)])
        cwdata = g.node(lambda x_, B_: jnp.sum(x_ * B_), [ccw, FcwNmy],
                        description='sum(ccw * FcwNmy)')
        cross = g.node(lambda c_, A_, x_: jnp.einsum('ij,ijk,ik->', c_, A_, x_),
                       [c, FtNmFcw, ccw], description='c.T @ FtNmFcw @ ccw')
        cwself = g.node(lambda x_, A_: jnp.einsum('ij,ijk,ik->', x_, A_, x_),
                        [ccw, FcwtNmFcw], description='ccw.T @ FcwtNmFcw @ ccw')
        logp = logp + cwdata - cross - 0.5 * cwself

    g.named(logp, 'logp')
    g.named(g.pair(logp, c), 'staged')


@mm.graph
def vectorwoodburysolve(g, ys, Nsolves, Fs, Pinv):
    Nmys, NmFs, FtNmys, FtNmFs, lNs = [], [], [], [], []

    for y, F, Nsolve in zip(ys, Fs, Nsolves):
        Nmy, lN = Nsolve(y) # Nmy: (n_i,), lN: scalar
        NmF, _ = Nsolve(F)  # NmF: (n_i, k)

        lNs.append(lN)
        Nmys.append(Nmy)
        NmFs.append(NmF)
        FtNmys.append(g.dot(F, Nmy))
        FtNmFs.append(g.dot(F, NmF)) # F.T @ Nmy: (k,); F.T @ NmF: (k, k)

    FtNmy = g.array(FtNmys)            # FtNmy: (np, k)
    FtNmF = g.array(FtNmFs)            # FtNmF: (np, k, k)

    Pm, lP = Pinv                      # Pm: (np, k, k), lP: (np,)
    cf, lS = g.cho_factor(Pm + FtNmF)  # cf: (np, k, k), lS: (np,)
    mu = g.cho_solve(cf, FtNmy)        # cfFtNmy: (np, k)

    solves = []
    for i, (Nmy, NmF) in enumerate(zip(Nmys, NmFs)):
       solves.append(g.pair(Nmy - NmF @ mu[i, :], lNs[i] + lP[i] + lS[i]))

    solve = g.ntuple(solves)



@mm.graph
def concat(g, a, b):
    if isinstance(a, (list, tuple)):
        result = [g.node(lambda x, y: jnp.hstack([x, y]), [ai, bi]) for ai, bi in zip(a, b)]
    else:
        result = g.node(lambda x, y: jnp.hstack([x, y]), [a, b])

@mm.graph
def delay(g, y, d):
    result = y - d



def _materialize(x):
    """Resolve x — array, callable, or metamath graph — to (const_array, fn).

    Exactly one of the returned values is non-None. `fn` (when set) has a
    `.params` attribute and is called as `fn(params=params)`.
    """
    if isinstance(x, dict):           # metamath graph
        fn = mm.func(x)
        if not fn.args and not fn.params:
            return jnp.asarray(fn(params={})), None
        return None, fn
    if callable(x):
        return None, x
    return jnp.asarray(x), None


def _sm_apply(y, Np, Uind, P):
    """Apply K^-1 to y for K = diag(N) + F P F^T (exposure-indexed Sherman-Morrison).

    Dispatches on y.ndim so the same node serves both `Nsolve(y)` (1D) and
    `Nsolve(F)` (2D) call-sites in the Woodbury graph. Takes the already-
    padded Np (so `pad(N)` can fold when N is constant).
    """
    if y.ndim == 1:
        yp = jnp.pad(y, ((1, 0),), constant_values=0.0)
        return smup_ind_correct(yp, Np, Uind, P)
    else:
        # y has shape (d, k); vsmup_ind_correct expects (k, d+1).
        Tp = jnp.pad(y.T, ((0, 0), (1, 0)), constant_values=0.0)
        return vsmup_ind_correct(Tp, Np, Uind, P).T


@mm.graph
def smsolve(g, y, N, Uind, P):
    """Sherman-Morrison-Woodbury solve `K^-1 y` for `K = diag(N) + F P F^T`.

    F is the ecorr exposure matrix (0/1 epoch indicators), encoded compactly
    as ``Uind`` (per-epoch list of TOA indices, padded with 0 to a common
    length; index 0 of Np is a sentinel +inf so padded entries contribute 0).

    The math is decomposed across graph nodes so the constant pieces fold:

      ``logN``   = ``sum(log N)``           — folds when N is const
      ``Np``     = ``pad(N)`` with +inf     — folds when N is const
      ``log1pt`` = ``sum log1p(P · F^T diag(1/N) F)`` (the F^T·F is
                   diagonal by epoch, gathered via ``Uind``)
                                            — folds when both N and P are const
      ``Kmy``    = ``K^-1 y``               — runtime (depends on the
                                              residual / basis matrix)

    Returns (Kmy, logdet). Equivalent to ``matrix.SM_1d_indexed`` (1D y) /
    ``matrix.SM_2d_indexed`` (2D y) but exposed as a graph so folding can
    bake the N-only and (N,P)-only precompute when noisedict / ecorr-prior
    are pinned.
    """
    logN = g.node(lambda N_: jnp.sum(jnp.log(N_)),
                  [N], description='sum(log N)')
    Np = g.node(lambda N_: jnp.pad(N_, ((1, 0),), constant_values=jnp.inf),
                [N], description='pad(N) with +inf at idx 0')
    log1pt = g.node(lambda Np_, P_, U_: jnp.sum(vsmdp_ind(Np_, P_, U_)),
                    [Np, P, Uind],
                    description='sum log1p(P · F^T diag(1/N) F)')
    logdet = logN + log1pt

    Kmy = g.node(_sm_apply, [y, Np, Uind, P], description='K^-1 y')
    result = g.pair(Kmy, logdet)


class NoiseMatrixSM(Kernel):
    """metamath analog of matrix.NoiseMatrixSM_var.

    K = diag(N) + F P F^T where F is the ecorr exposure matrix (0/1 indicators
    per epoch). Uses the indexed Sherman-Morrison solve via a graph node.
    """
    def __init__(self, N, F, P):
        # signature matches matrix.NoiseMatrixSM_var(getnoise, F, getP)
        self.N = N            # callable: params -> diag noise (n_toa,)
        self.F = F            # exposure matrix (n_toa, n_epoch), constant
        self.P = P            # callable: params -> per-epoch variance (n_epoch,)
        self.Uind = jnp.array(make_uind(F))

    @property
    def make_solve(self):
        return smsolve(None, self.N, self.Uind, self.P)

    @property
    def make_inv(self):
        # not needed for Woodbury, but provide for completeness via dense fallback
        raise NotImplementedError("NoiseMatrixSM.make_inv not implemented; "
                                  "use make_solve inside a Woodbury.")

    def make_kernelproduct(self, y):
        return normal(y, self.make_solve)

    def make_sample(self):
        """Sample noise ~ N(0, K) where K = diag(N) + F P F^T.

        Equivalent to drawing N(0, diag(N)) + F @ N(0, diag(P)).
        """
        N_const, N_fn = _materialize(self.N)
        P_const, P_fn = _materialize(self.P)
        F = jnp.asarray(self.F)

        def sample(key, params={}):
            N_arr = N_const if N_fn is None else N_fn(params=params)
            P_arr = P_const if P_fn is None else P_fn(params=params)

            key, k1, k2 = jax.random.split(key, 3)
            noise = jax.random.normal(k1, N_arr.shape) * jnp.sqrt(N_arr)
            ecorr = jax.random.normal(k2, P_arr.shape) * jnp.sqrt(P_arr)
            return key, noise + F @ ecorr

        sample.params = sorted(set(
            (list(N_fn.params) if N_fn is not None else []) +
            (list(P_fn.params) if P_fn is not None else [])
        ))
        return sample


class NoiseMatrix(Kernel):
    def __init__(self, N):
        self.N = N
        # matrix.NoiseMatrix*_var classes expose the callable as `getN`; alias
        # for compat with signals.py construction (e.g. NoiseMatrixSM_var(..., egp.Phi.getN))
        self.getN = N

    def make_sample(self):
        """Sample x ~ N(0, N). Dispatches on N.ndim at runtime: 1D → diag, 2D → cholesky."""
        N_const, N_fn = _materialize(self.N)

        def sample(key, params={}):
            N_arr = N_const if N_fn is None else N_fn(params=params)
            key, subkey = jax.random.split(key)
            if N_arr.ndim == 1:
                return key, jax.random.normal(subkey, N_arr.shape) * jnp.sqrt(N_arr)
            else:
                L = jsp.linalg.cholesky(N_arr, lower=True)
                return key, L @ jax.random.normal(subkey, (N_arr.shape[0],))

        sample.params = list(N_fn.params) if N_fn is not None else []
        return sample

    @property
    def make_solve(self):
        return noisesolve(None, self.N)

    @property
    def make_inv(self):
        if getattr(self, 'inv', None):
            # shortcut for GPs with custom make_inv
            return noiseinvfunc(self.inv)
        else:
            return noiseinv(self.N)

    def make_kernelproduct(self, y):
        return normal(y, self.make_solve)


class NoiseMatrix1D(NoiseMatrix):
    """Marker subclass for 1D (diagonal) noise. Same impl as NoiseMatrix.

    Exists so that `isinstance(x, matrix.NoiseMatrix1D_var)` dimension dispatch
    in likelihood.py still discriminates 1D vs 2D when the monkeypatch is active.
    """


class NoiseMatrix2D(NoiseMatrix):
    """Marker subclass for 2D (full) noise."""


def NoiseMatrix12D(getN):
    """Dispatch on getN.type, matching matrix.NoiseMatrix12D_var."""
    is_2d = getattr(getN, "type", None) is jax.Array
    return (NoiseMatrix2D if is_2d else NoiseMatrix1D)(getN)


class WoodburyKernel(Kernel):
    def __init__(self, N, F, P):
        self.N, self.F, self.P = N, F, P

    def make_sample(self):
        """Sample y ~ N(0, N + F P F^T) as noise + F @ coefficients."""
        N_sample = self.N.make_sample()
        P_sample = self.P.make_sample()
        F_const, F_fn = _materialize(self.F)

        F_params = list(F_fn.params) if F_fn is not None else []

        def sample(key, params={}):
            key, noise = N_sample(key, params)
            key, c = P_sample(key, params)
            F_arr = F_const if F_fn is None else F_fn(params=params)
            return key, noise + F_arr @ c

        sample.params = sorted(set(N_sample.params + P_sample.params + F_params))
        return sample

    @property
    def make_solve(self):
        return mm.prune_graph(woodbury(None, self.N.make_solve, self.F, self.P.make_inv), output='solve')

    def make_kernelproduct(self, y):
        return woodbury(y, self.N.make_solve, self.F, self.P.make_inv)

    def make_kernelsolve(self, y, T):
        """Callable returning (T^T Σ^-1 y, T^T Σ^-1 T) for Σ = N + F P F^T.

        Matches the contract of matrix.WoodburyKernel_var*.make_kernelsolve
        so likelihood.py's GlobalLikelihood.conditional path works unchanged.
        """
        graph = woodburykernelsolve(y, T, self.N.make_solve, self.F, self.P.make_inv)
        f = mm.func(graph)
        def call(params={}):
            return f(params=params)
        call.params = f.params
        return call

    def make_conditional(self, y):
        return mm.prune_graph(woodbury(y, self.N.make_solve, self.F, self.P.make_inv), output='cond')

    def make_coefficientproduct(self, y):
        cvars = list(self.index.keys())
        def getc(params):
            return jnp.concatenate([params[cvar] for cvar in cvars])
        getc.params = cvars

        return woodburylatent(y, self.N.make_solve, self.F, self.P.make_solve, getc)


class GlobalWoodburyKernel(Kernel):
    def __init__(self, Ns, Fs, P):
        self.Ns, self.Fs, self.P = Ns, Fs, P

    def make_kernelproduct(self, ys):
        if isinstance(self.Ns, (tuple, list)):
            # old path: list of per-pulsar noise kernels
            return globalwoodbury(ys, [N.make_solve for N in self.Ns], self.Fs, self.P.make_inv)
        elif hasattr(self.Ns, 'Ns'):
            # compound kernel: vectorized path
            joint_graph = vectorwoodburyjointsolve(
                ys, self.Fs, [N.make_solve for N in self.Ns.Ns],
                self.Ns.Fs, self.Ns.P.make_inv)
            proj_graph = mm.prune_graph(joint_graph, output='projected')
            return globalwoodbury_fused(proj_graph, self.P.make_inv)
        else:
            # single noise matrix
            return globalwoodbury(ys, self.Ns.make_solve, self.Fs, self.P.make_inv)


class VectorWoodburyKernel(Kernel):
    def __init__(self, Ns, Fs, P):
        self.Ns, self.Fs, self.P = Ns, Fs, P

    @property
    def make_solve(self):
        return vectorwoodburysolve([None] * len(self.Ns),
                                   [N.make_solve for N in self.Ns],
                                   self.Fs, self.P.make_inv)

    def make_kernelproduct(self, ys):
        return vectorwoodbury(ys, [N.make_solve for N in self.Ns], self.Fs, self.P.make_inv)

    def make_conditional(self, ys):
        return mm.prune_graph(vectorwoodbury(ys, [N.make_solve for N in self.Ns], self.Fs, self.P.make_inv), output='cond')

    def make_joint_solve(self, Fs_outer):
        return vectorwoodburyjointsolve(
            [None] * len(self.Ns),   # ys are args
            Fs_outer,                 # global GP bases, constants
            [N.make_solve for N in self.Ns],
            self.Fs,
            self.P.make_inv
        )

    def make_kernelproduct_gpcomponent(self, ys, transform=None, extsignals=None):
        """ArrayLikelihood.clogL path. Returns a metamatrix graph.

        Composes the GP-coefficient log-likelihood from `vectorgpcomponent`:

            xi --[reparams]--> c  --(prior on c - means)--  data sees c

        Leaves go in as graphs / FuncLeafs / arrays; folding decides what
        runs at trace time vs runtime. The output is a graph with named
        subgraphs 'logp' and 'staged'; the method prunes to whichever
        matches the reparam state.

        - ``transform``: callable or list of ``rp(params, c) -> (c, ldL)``.
        - ``self.means``: callable ``params -> a0``; centers the GP prior.
        - ``extsignals``: list of ``ExtSignal`` (each with .coeffs, .Fs).
        """
        if transform is None:
            reparams = []
        elif isinstance(transform, (list, tuple)):
            reparams = list(transform)
        else:
            reparams = [transform]

        # Prior on c_for_prior: either the uniform-Phi wrap of `P.make_inv` or,
        # for a mixed-Phi compound (commongp + HD globalgp), the per-GP-sum
        # graph supplied by `CompoundGP._build_mixed_logprior`. Both look the
        # same to `vectorgpcomponent` — a GraphLeaf taking c_for_prior.
        if hasattr(self, 'prior') and self.prior is not None:
            prior_graph = self.prior
        else:
            prior_graph = gaussian_coefficient_logprior(None, self.P.make_inv)

        # Build the params -> (c, ldL) FuncLeaf that bundles fold + reparams.
        # cvarsall: list-per-pulsar of {coeff-name: slice}.
        cvarsall = (self.index if isinstance(self.index, list)
                    else [{par: sl} for par, sl in self.index.items()])
        cvar_params = sum([list(cvars) for cvars in cvarsall], [])

        def _fold(params):
            return jnp.array([jnp.concatenate([params[cvar] for cvar in cvars])
                              for cvars in cvarsall])

        def _coeffs(params):
            c = _fold(params)
            ldL = 0.0
            for rp in reparams:
                c, dldL = rp(params, c)
                ldL = ldL + dldL
            return (c, ldL)

        _coeffs.params = sorted(set(
            cvar_params + sum([list(rp.params) for rp in reparams], [])))

        means_leaf = self.means if getattr(self, 'means', None) is not None else 0.0

        ext_coeffs = [es.coeffs for es in (extsignals or [])]
        ext_Fs     = [list(es.Fs) for es in (extsignals or [])]

        graph = vectorgpcomponent(
            ys,
            [N.make_solve for N in self.Ns],
            list(self.Fs),
            prior_graph,
            _coeffs,
            means_leaf,
            ext_coeffs,
            ext_Fs,
        )
        return mm.prune_graph(graph, output=('staged' if reparams else 'logp'))


class CompoundGP:
    def __new__(cls, x):
        if not isinstance(x, (list, tuple)):
            # if this is a single GP, just return it
            return x
        else:
            return super().__new__(cls)
    @staticmethod
    def _is_vector(gp):
        # per-pulsar F: commongp exposes F as a tuple; globalgp carries Fs as a list.
        if hasattr(gp, 'F') and isinstance(gp.F, tuple):
            return True
        if hasattr(gp, 'Fs') and isinstance(gp.Fs, (list, tuple)):
            return True
        return False

    def __init__(self, gplist):
        self.gplist = gplist
        # Mixed-Phi compound: at least one gp's Phi is dense (NoiseMatrix2D,
        # e.g. HD globalgp) while another is per-pulsar diagonal
        # (NoiseMatrix1D, e.g. commongp). The Phi.N arrays can't be concat'd
        # into a single batched Phi.
        phi_kinds = {('2D' if isinstance(gp.Phi, NoiseMatrix2D) else '1D')
                     for gp in gplist}
        self._mixed_phi = len(phi_kinds) > 1

        # The per-GP coefficient log-prior (with Phi=None) is only for the
        # vector / decentered path — commongp/globalgp whose coefficients are
        # sampled (clogL), where the joint prior decomposes as a sum of per-GP
        # contributions. A mixed compound of *marginalized* GPs (e.g. a constant
        # timing GP + a constant fourier-variance GP in a single PulsarLikelihood)
        # has no `.index`; it instead needs a real combined Phi (see `Phi`).
        self._mixed_vector = (self._mixed_phi
                              and all(hasattr(gp, 'index') for gp in gplist)
                              and all(self._is_vector(gp) for gp in gplist))
        if self._mixed_vector:
            self.prior = self._build_mixed_logprior(gplist)
        if all(hasattr(gp, 'index') for gp in gplist):
            # vector commongp (per-pulsar F tuples) or commongp + globalgp
            # (the globalgp carries Fs as a list). Either way → list-of-dicts,
            # one per pulsar, matching matrix.VectorCompoundGP.
            if all(self._is_vector(gp) for gp in gplist):
                self.index = [dict(g) for g in
                              zip(*[gp.index.items() for gp in gplist])]
            else:
                # non-vector (single PulsarLikelihood): cumulative-offset flat dict
                index, cnt = {}, 0
                for gp in gplist:
                    for var, sli in gp.index.items():
                        width = sli.stop - sli.start
                        index[var] = slice(cnt, cnt + width)
                        cnt += width
                self.index = index
    def _concat(self, vecmats):
        return functools.reduce(lambda x, y: concat(x, y), vecmats)
    @property
    def F(self):
        # for VectorWoodburyKernel: each gp exposes either F (tuple of per-pulsar
        # arrays, commongp-style) or Fs (list, globalgp-style). Normalize both.
        def _per_psr(gp):
            return gp.F if hasattr(gp, 'F') else gp.Fs
        if all(isinstance(_per_psr(gp), (tuple, list)) for gp in self.gplist):
            return tuple(self._concat(Fs) for Fs in zip(*(_per_psr(gp) for gp in self.gplist)))
        else:
            return self._concat([_per_psr(gp) for gp in self.gplist])

    @staticmethod
    def _build_mixed_logprior(gplist):
        """Per-GP-sum log-prior graph for a mixed-Phi compound.

        Input  arg : c_for_prior of shape (npsr, k_total).
        Output     : scalar `Σ_g log p_g(c[:, g_slice])`.

        Per-GP contribution (matches matrix.VectorCompoundGP.priorfunc):
          - diagonal Phi (NoiseMatrix1D, per-pulsar):
              `-0.5 sum(c_g^2 / Phi_g) - 0.5 sum(log|Phi_g|)`
          - dense Phi (NoiseMatrix2D, e.g. HD globalgp):
              `-0.5 c_flat^T Phi^-1 c_flat - 0.5 logdet(Phi)`,
            where c_flat is c_g.reshape(-1) (row-major: per-pulsar order
            matches gp.index slice ordering).
        """
        b = mm.GraphBuilder()
        c_for_prior = b.leaf(None, name='c_for_prior')

        widths = [next(iter(gp.index.values())).stop -
                  next(iter(gp.index.values())).start
                  for gp in gplist]
        offsets = [0]
        for w in widths:
            offsets.append(offsets[-1] + w)

        def _slice_op(s, e):
            return lambda c: c[:, s:e]

        def _dense_contrib(c_, Phi_):
            cf = c_.reshape(-1)
            return (-0.5 * cf @ jnp.linalg.solve(Phi_, cf)
                    - 0.5 * jnp.linalg.slogdet(Phi_)[1])

        def _diag_contrib(c_, Phi_):
            return (-0.5 * jnp.sum(c_ * c_ / Phi_)
                    - 0.5 * jnp.sum(jnp.log(jnp.abs(Phi_))))

        contribs = []
        for i, gp in enumerate(gplist):
            s, e = offsets[i], offsets[i + 1]
            phi_n_leaf = b.leaf(gp.Phi.N, name=f'gp{i}_phiN')
            c_slice = b.node(_slice_op(s, e), [c_for_prior],
                             description=f'c_for_prior[:,{s}:{e}]')
            if isinstance(gp.Phi, NoiseMatrix2D):
                contrib = b.node(_dense_contrib, [c_slice, phi_n_leaf],
                                 description=f'gp{i} dense logprior')
            else:
                contrib = b.node(_diag_contrib, [c_slice, phi_n_leaf],
                                 description=f'gp{i} diag logprior')
            contribs.append(contrib)

        logpr = contribs[0]
        for c in contribs[1:]:
            logpr = logpr + c
        b.named(logpr, 'logpr')
        return b.graph

    def _mixed_dense_Phi(self):
        """Combined dense Phi for a mixed 1D/2D *marginalized* compound.

        Promotes each per-pulsar-diagonal (1D) block to a dense diagonal and
        block-diagonals everything into one dense covariance, mirroring
        matrix.CompoundGP's mixed-const / mixed-var handling. Handles constant
        (static arrays) and variable (callables) Phi blocks.
        """
        parts = [(_materialize(gp.Phi.N), isinstance(gp.Phi, NoiseMatrix2D))
                 for gp in self.gplist]

        if all(fn is None for (const, fn), _ in parts):
            blocks = [const if is2d else jnp.diag(const)
                      for (const, fn), is2d in parts]
            return NoiseMatrix2D(jsp.linalg.block_diag(*blocks))

        def getN(params):
            blocks = []
            for (const, fn), is2d in parts:
                v = const if fn is None else fn(params=params)
                blocks.append(v if is2d else jnp.diag(v))
            return jsp.linalg.block_diag(*blocks)
        getN.params = sorted(set().union(
            *[set(fn.params) for (const, fn), _ in parts if fn is not None]))
        getN.type = jax.Array
        return NoiseMatrix2D(getN)

    @property
    def Phi(self):
        if self._mixed_vector:
            # Vector / decentered path: no single combined Phi; the prior lives
            # on `self.prior` and is consumed via VectorWoodburyKernel's `prior`
            # branch. likelihood.py threads this None through as P and never
            # calls P.make_inv when self.prior is set.
            return None
        if self._mixed_phi:
            # mixed 1D/2D but marginalized (no sampled coefficients): build a
            # real combined dense Phi for the Woodbury.
            return self._mixed_dense_Phi()
        N = self._concat([gp.Phi.N for gp in self.gplist])
        nm = NoiseMatrix(N)

        # combined_inv only available when every gp supplies a Phi_inv;
        # otherwise fall through to the default noiseinv graph.
        if all(getattr(gp, 'Phi_inv', None) is not None for gp in self.gplist):
            phi_invs = [gp.Phi_inv for gp in self.gplist]

            def combined_inv(params):
                results = [f(params) for f in phi_invs]
                precisions = [r[0] for r in results]
                logdets = [r[1] for r in results]
                return (jax.scipy.linalg.block_diag(*precisions), sum(logdets))

            combined_inv.args = list(dict.fromkeys(
                arg for f in phi_invs for arg in getattr(f, 'args', [])
            ))
            combined_inv.params = list(dict.fromkeys(
                p for f in phi_invs for p in getattr(f, 'params', [])
            ))
            nm.inv = combined_inv

        return nm

def CompoundDelay(residuals, delays):
    return functools.reduce(lambda x, y: mm.func(delay(x, y)), [residuals, *delays])


### experimental

@mm.graph
def woodburyfast(g, y, allsolve, F, Pinv):
    ytNmy, FtNmy, FtNmF, lN = allsolve(y, F)

    Pm, lP = Pinv
    cf, lS = g.cho_factor(Pm + FtNmF)
    mu = g.cho_solve(cf, FtNmy)
    ld = lP + lS + lN

    # cond = g.pair(mu, cf, name='cond')
    # solve = g.pair(Nmy - NmF @ mu, ld, name='solve')

    logp = g.combine_logp_f64(ytNmy, g.dot(FtNmy, mu), [lP, lS, lN])  # Half A: f64 final combine

# yt Km y = yt Nm y - yt Nm F (Pinv + FtNmF)^-1 Ft Nm y
# Tt Km y = Tt Nm y - Tt Nm F (Pinv + FtNmF)^-1 Ft Nm y
# Tt Km T = Tt Nm T - Tt Nm F (Pinv + FtNmF)^-1 Ft Nm T
# quindi mi mancano TtNmy, TtNmF, TtNmT;
# il primo e l'ultimo si possono ottenere da allsolve(y, T), ma TtNmF?

@mm.graph
def noiseallsolve(graph, y, F, N):
    result = N.allsolve(y, F)
