# Non-uniform prior support in discovery — implementation plan

Branch `priors`, off `consolidated-merge` (`86fbf32`).
Worktree `/fred/oz002/dreardon/discovery_dev/discovery/.claude/worktrees/priors`.

Scoped by six independent sub-agents over `src/discovery/prior.py`, the sampler
back-ends, `platypus/infer_hyperpriors.py`, `platypus/hyper_utils.py`, the
`hierarchical_models*.txt` manifests, and `/fred/oz002/dreardon/hbm_outliers_example.py`.

---

## Status

| phase | state |
|---|---|
| 0 foundations | done |
| 1 scalar families | done |
| 2 outlier mixtures | done, plus `NormalWithNormalOutliers` |
| 3 joint N-D, fixed hyperparameters | done, 2-D and 3-D |
| 4 nested sampling | done: raises, naming the family |
| 5 platypus plumbing | done in `hyper_utils.py`; **uncommitted** |
| 6 simultaneous hierarchical | done except the solar-wind profile |
| 7 validation | unit and integration done; the array-level science runs are not |

339 tests pass (262 pre-existing, 77 new in `tests/test_prior.py`).

**Manifest v2 turned out to be unnecessary.** The plan assumed a new file format
was needed to name a distribution family per entry. It is not: the family is
already in the result-JSON filename, decoded by `_parse_hyper_label`, so the
existing bare-path manifests carry gauss fits unchanged. Only the hard rejection
at `hyper_utils.py:1156` had to go.

**Validated against the fitting code.** The `TruncatedMultivariateNormal` density
reproduces `hyper_utils.truncated_normal_2d` to `5.3e-14` over 200 random points
at the `red_gauss_A4.0e-15` ML row. Every family integrates to 1 by quadrature.
The uniform-only path is bit-identical to the previous implementation for
`to_dict`, `logprior` and `to_vec` over 400 random points including a vector
parameter.

**Solar wind carries no physical scaling.** `sigma_SW` is not significantly
predicted by ecliptic latitude and the production chains do not use it, so the
asymmetric-Gaussian profile was removed from `infer_hyperpriors.py` and
`hyper_utils.py` rather than ported. `-physical sw` is now silently dropped at
the fitting end, and loading a `sw_*_physical_*` result raises. Every remaining
scaling — red, dm, chrom, ecorr — is linear in one coefficient times one
per-pulsar covariate, which the `covariates` field expresses exactly.

**Sampler coverage.** NUTS and the normalising-flow sampler both work with every
family: uniform, TruncatedNormal, a fixed joint TruncatedMultivariateNormal, and
a hierarchical MultivariateNormal with sampled hyperparameters. Nested sampling
raises, since it builds its prior as an explicit box and never sees the
transform; `-ns` is only used for the broad uniform evidence runs that precede
the hierarchical fit, so this costs nothing.

**Not done:** generalising the hard-coded `KAT_MKBF` white-noise keys in
`infer_hyperpriors.py`; the matched-ECORR control run of §6; and the array-level
demonstration that a Gaussian population prior no longer truncates
`red_noise_gamma`.

---

## 0. What the scoping changed about the brief

Five findings alter the shape of the work. They are recorded here because each of
them invalidates part of the original request.

### 0.1 `infer_hyperpriors.py` already fits correlated 2-D and 3-D models

The request treats the correlated case as a future ambition. It is not — it is the
default, and it is already fitted:

| component | dimension | free hyperparameters |
|---|---|---|
| `red`, `dm`, `sw` | 2 | `mu_A, mu_G, sigma_A, sigma_G, rho` |
| `chrom` | 3 | `mu_A, mu_G, mu_alpha, sigma_A, sigma_G, sigma_alpha, rho_AG, rho_Aalpha, rho_Galpha` |
| `efac`, `equad`, `ecorr` | 1 | `mu, sigma` |

`hyper_utils.py:8-51` (`truncated_normal_2d`) and `:162-219` (`truncated_normal_3d`).
The fitted correlations are strong and are not a nuisance term to be dropped:
`red_rho = 0.778`, `sw_rho = 0.997` (ML rows on disk).

So the design target is **N-dimensional, N ∈ {1,2,3}**, not "2-D support".

### 0.2 The fitted family is *not* the family in the colleague's example

These are two different distributions and they will not agree even at matched moments.

| | `infer_hyperpriors.py` | `hbm_outliers_example.py` |
|---|---|---|
| covariance | `(sigma, rho)`, cov rebuilt in the density, PSD by eigenvalue rejection | **Cholesky `scale_tril`**, PSD by construction |
| bounded support | **rectangular truncation** + renormalisation by the bivariate-normal CDF box mass `Z` | **logistic squash**, `gamma = 7/(1+exp(-z))`; Gaussian lives in unconstrained space |
| marginal on gamma | truncated normal — closed support `[0,7]`, non-zero density at the edges | logit-normal — open support `(0,7)`, no normalising constant |
| outlier component | **uniform** over the sampling box | broad zero-mean MVN, `scale_tril = diag(7,6)` |
| outlier weight | `Chi` = **outlier** fraction | `spin_Q` = **foreground** fraction, so `Chi ≡ 1 - Q` |

This fork is the answer to the 2-D question and is resolved in §3.

### 0.3 `Chi` is the outlier weight — confirmed, not assumed

`hyper_utils.py:301-342`:
```python
def mixture_fn(data, Chi, **kwargs):
    return (1.0 - Chi) * model_fn(data, **kwargs) + Chi * sampling_prior_fn(data)
```
`Chi = 0` is a pure hypermodel. The outlier component is the **product of uniform
sampling-prior densities over the `SAMPLING_PRIORS` box** (`infer_hyperpriors.py:124-139`),
not a broad Gaussian.

One structural detail that a consumer must reproduce: the mixture wraps **each
component block separately** with a single shared `Chi`, so the joint model is a
*product of mixtures*, not a mixture of products —

```
p(theta) = prod_c [ (1-Chi)*p_c_hyper(theta_c) + Chi*p_c_unif(theta_c) ]
```

A pulsar can be an outlier in `red` but not in `dm` within the same draw.

### 0.4 The manifest has no per-pulsar rows at all

`hierarchical_models.txt` is a bare list of result-JSON paths, one per line. Every
semantic lives in the *filename*, decoded by `_parse_hyper_label`
(`hyper_utils.py:958-1001`). There is no field layout to extend — "a per-pulsar entry
naming a distribution family" is a **new file format**, not a new column.

The per-pulsar *keying* infrastructure does already exist and is correct, including
`re.escape` on the `+` in J-names (`hyper_utils.py:1196-1204`, verified: pattern
`J0030\+0451_red_noise_gamma\Z` matches the real name and not the unescaped one).
That part is reusable as-is.

The single line that blocks everything today is `hyper_utils.py:1156-1160`:
```python
if not is_uniform:
    raise ValueError(f"{fname}: prior overrides are built from the uniform-hypermodel "
                     f"[min, max] box, but label '{label}' is not a uniform fit. ...")
```
Every Gaussian fit, every `rho`, and every `Chi` is rejected there.

### 0.5 The `to_vec` bug is real but its diagnosis in the brief is wrong

Measured in float64 (`jax 0.5.3`, x64 confirmed on):

- `to_dict` saturates to a bound at **|y| ≈ 18.26–18.51**, a,b-dependent — not a
  constant ~19.
- `to_vec` at `x == b` returns `+inf`. **It does not produce a NaN gradient.**
  `to_dict(inf)` is fine (`tanh(inf)=1`), `logL` is finite there, and
  `grad logL` is exactly `-0.0` because `sech^2(inf) = 0`. NaN arises only for `x`
  strictly *outside* the box.
- The scipy NaN comes from scipy's own line search doing `inf - inf`
  (`_linesearch.py:161`), not from JAX.
- L-BFGS-B does return `nit=0` — but with `success=False`, `status=2`,
  message `ABNORMAL:`, and `‖jac‖ = 2.22`, **not a small gradient**. It only looks
  silent if the caller never inspects `.success`. BFGS and SLSQP *are* silent
  (`success=True`) but with a permanently frozen coordinate, not a NaN.

On "77% of 344 fits had a parameter within 0.1% of a bound": in this
parametrisation, 0.1% of range corresponds to **|y| > 3.45**, an utterly
unremarkable coordinate. Unpenalised ML *should* run to the rail whenever the
likelihood is monotone in a parameter — the norm for red-noise `log10_A` in a quiet
pulsar. And the bug needs |y| ≳ 18.3 to bite, which is unreachable from a sane start
because the `logL` gradient has underflowed to ~1e-14 long before. That statistic is
evidence about the *likelihood*, not about `arctanh`.

**Verdict:** a real, contained bug worth fixing (a noisedict value sitting exactly on
a prior edge maps to an infinite coordinate silently); *not* a symptom of the
box-only design — it would appear under logit, probit, or any bounded transform whose
inverse has a pole at the boundary; and almost certainly *not* the cause of the 77%
statistic. Fix it on its own merits, decoupled from this work.

---

## 1. Representation of a prior spec

### 1.1 The tagged-list syntax, and the one thing that makes it dangerous

The proposed trailing-string tag is adopted as the **user-facing** syntax. It is
compact, JSON-round-trippable (which matters for the manifest, §5), and backward
compatible by construction.

It has exactly one sharp edge, and it is a silent-wrong-answer edge.
`prior.py:138-143`:

```python
a.append(therange[0])
b.append(therange[1])
```

A 3-element `['mean', 'std', 'Normal']` is **not rejected here** — the extra element
is ignored and the entry is read as the box `[mean, std]`. Whenever `std < mean`
(always, for `log10_A`) this gives `a > b`, and both `to_dict`'s
`0.5*(b+a+(b-a)*tanh(y))` and `to_vec`'s `(a+b-2x)/(a-b)` silently invert their
sense. Same at `prior.py:204-205` in `makelogtransform_classic`.

Every *other* site fails loudly (`TypeError` from `uniform(par, *range)` and
`np.random.uniform(*range)`, `ValueError` from `low, high = ...`). This one does not.

**Mitigation, and it is Phase 0 work:** normalise every entry through a single
`parse_spec()` at the boundary, and add an explicit shape guard to all five
first-match-wins loops (`prior.py:86, 97, 126, 203, 257`) so that a spec which is
not an untagged 2-list can never reach `therange[0]`/`therange[1]`.

### 1.2 Canonical form

```python
Family = namedtuple('Family', 'nargs support logpdf')   # support(args)->(lo,hi)
FAMILIES = {}                                            # name -> Family

def parse_spec(spec):
    """[a,b] -> ('Uniform', (a,b));  [..., 'Name'] -> ('Name', tuple(rest))"""
```

Dispatch is a dict lookup on the trailing string, **resolved once at build time in
Python**. Nothing is dispatched inside a traced function.

### 1.3 Families

| spec | link | notes |
|---|---|---|
| `[a, b]` or `[a, b, 'Uniform']` | tanh onto `[a,b]` | fast path, byte-identical to today |
| `[mu, sd, 'Normal']` | **whitened affine** `x = mu + sd*y` | see §2.1 |
| `[mu, sd, a, b, 'TruncatedNormal']` | tanh onto `[a,b]` | stable `logZ`, §2.2 |
| `[a, b, chi, A, B, 'UniformWithOutliers']` | tanh onto `[min(a,A), max(b,B)]` | §2.3 |
| `[mu, sd, chi, A, B, 'NormalWithOutliers']` | whitened affine (support is **R**) | §2.3 |
| `[mu, sd, a, b, chi, A, B, 'TruncatedNormalWithOutliers']` | tanh onto the union | §2.3 |
| `[mu, sd, chi, sd_out, 'NormalWithNormalOutliers']` | whitened affine | **added**, see §2.4 |

---

## 2. Scalar families — mathematics

Every family contributes `log p_X(x) + log|dx/dy|`. The existing `logprior`
(`prior.py:177`) is already an *exactly normalised* log-density, not a bare
Jacobian —

```
log p_X(x) + log|dx/dy| = -log(b-a) + log h + log sech^2(y) = log2 - 2*logaddexp(y,-y)
```

with `h=(b-a)/2`. This must be preserved: `flow.py:130-159` `estimate_evidence` and
`numpyro.py:302-306` `_laplace_logz` both depend on it.

### 2.1 Normal — whiten, do not use tanh

Support is all of R, so tanh cannot reach the tails. With `x = mu + sd*y`:

```
logprior_i(y) = -0.5*((x-mu)/sd)^2 - log(sd) - 0.5*log(2pi) + log(sd)
              = -0.5*y^2 - 0.5*log(2pi)
```

**`sd` cancels exactly** — the transformed prior is N(0,1) for every Normal
parameter regardless of `mu` and `sd`.

Whitening matters even though NUTS's diagonal mass matrix adapts, for three
non-asymptotic reasons: (i) during the first adaptation window there is one global
step size, so a stiffer coordinate starves the loose ones of movement and corrupts
their variance estimate — a feedback loop that can survive the whole warmup;
(ii) `y=0` is the initialisation used everywhere in this repo
(`numpyro.py:399`, `flow.py:235` both call `to_dict(zeros(parlen))` and call it
"prior midpoints"), and under an identity map `y=0` may be tens of sigma from `mu`;
(iii) the base-distribution factor of §4.1 is only harmless in whitened units.

### 2.2 TruncatedNormal — and the `logZ` that must not cancel

With `z=(x-mu)/sd`, `al=(a-mu)/sd`, `be=(b-mu)/sd`:

```
logprior_i(y) = -0.5*z^2 - log(sd) - 0.5*log(2pi) - logZ + log(h) + log sech^2(y+y0)
logZ = log(Phi(be) - Phi(al))
```

`y0 = arctanh((clip(mu,a,b) - m)/h)` puts `y=0` at the prior mode.

The naive `log(ndtr(be) - ndtr(al))` returns `-inf` at `(al,be)=(10,11)` where the
true value is `-53.23`. The obvious repair still fails because
**`jax.scipy.special.log_ndtr(x)` underflows to exactly `-0.0` for x ≳ 37.5**
(measured: `log_ndtr(37.0) = -5.73e-300`, `log_ndtr(38.0) = -0.0`).

Recommended form — mirror the **inputs**, not the branches, so there is one
expression and no NaN gradient through an unselected `-inf` branch:

```python
def _log_ndtr_diff(al, be):
    """log(Phi(be) - Phi(al)) for be > al."""
    flip = (al + be) > 0.0
    u = jnp.where(flip, -be, al)         # far-tail endpoint, always
    v = jnp.where(flip, -al, be)
    lu, lv = log_ndtr(u), log_ndtr(v)
    return lv + jnp.log(-jnp.expm1(lu - lv))
```

Verified: `(10,11) -> -53.2313`, `(38,39) -> -726.557`, `(50,60) -> -1254.83`,
`(-1e30,1e30) -> 0.0`; `d logZ/dmu` finite at `mu = -14.5, -30, +5, -60`.
Relative error vs mpmath stays `<= 1.5e-14` down to interval widths of `1e-4` sigma
and only degrades past `1e-13` sigma — a degenerate spec, not a physical one.

**Critical shortcut:** when `mu, sd, a, b` are fixed numbers — the whole of Phases
1–4 — `logZ` is a **build-time constant**. Compute it once in scipy/mpmath and bake
it in as a Python float. The JAX path above is needed only in Phase 6.

### 2.3 Mixtures — the support rule

```
log p_X(x) = logaddexp( log(1-chi) + log f_core(x),
                        log(chi)   + log f_out(x) )
```
with `chi` the **outlier** weight (§0.3) and each `log f` returning `-inf` outside
its own support.

**The link must map onto the *union* of the component supports.** If it maps only
onto the core box, the outlier component is not merely under-sampled — it is
unreachable, and what gets sampled is the mixture *renormalised to the core box*,
i.e. a different distribution with effective outlier weight

```
chi_eff = chi*(b-a)/(B-A) / [ (1-chi) + chi*(b-a)/(B-A) ]
```

Worked example: core `[-16,-13]`, outlier `[-20,-11]`, `chi = 0.15` → the mixture
integrates to `0.85 + 0.15*(3/9) = 0.90` over the core box, so you would sample an
effective outlier fraction of 5.6% having specified 15%, with no error raised.

Normalisation of the recommended construction was checked by quadrature:
`integral p_X(x) dx = 1.0000000000000002`, and after the tanh map
`integral exp(logprior(y)) dy = 0.99999999`.

### 2.4 The discontinuity hazard — why `NormalWithNormalOutliers` is added

A uniform core inside a wider uniform outlier has a **step discontinuity in
`log p`** at the core boundaries, of height `log(1 + (1-chi)(B-A)/(chi(b-a)))`.
Measured: **3.33 nats** for `chi=0.1`, core `[-16,-13]`, outlier `[-20,-11]`.

Hamiltonian dynamics does not conserve energy across a discontinuity. Every
leapfrog trajectory crossing one incurs an energy error equal to the jump, so it is
rejected with probability `1 - exp(-dH)` — at 3.3 nats, a 96% rejection rate for
boundary crossings. The outlier region becomes practically unreachable *even though
the support is correct*, and the global step size is driven down to compensate.

This is precisely why `hbm_outliers_example.py:24-26` abandons the uniform
background:
```python
#bg_spin_dist = dist.Uniform(low=jnp.array([-20, 0]), high=jnp.array([-13, 7]))
bg_spin_dist = dist.MultivariateNormal(scale_tril=jnp.array([[7, 0],[0, 6]]))
```

Actions:
1. Add `NormalWithNormalOutliers` (and its MVN counterpart) — the family that
   actually behaves.
2. Keep the uniform-outlier families, because they are what `infer_hyperpriors.py`
   fits and we must be able to reproduce existing results.
3. Emit a build-time warning quoting the jump height when it exceeds ~1 nat.
4. Optional logistic taper of build-time width `delta` on the core edges,
   renormalised by 1-D quadrature at build time, for users who want the uniform core
   without the rejection penalty.

### 2.5 Discrete latents

NUTS cannot sample a categorical assignment — there is no gradient with respect to
it. Confirmed that numpyro never samples the `Categorical` in a `MixtureGeneral`
either: `_MixtureBase.log_prob` logsumexps over `component_log_probs`
(numpyro 0.18.0). **The mixture must be marginalised analytically**, and `logaddexp`
is exactly the two-component case of what numpyro does internally. No design choice
here.

Worth exposing as a deterministic: the per-instance responsibility
`r = softmax(component log-densities)`. One `softmax` of quantities already
computed, and it is the diagnostic a user actually wants
(`hbm_outliers_example.py:43` does this).

---

## 3. The N-dimensional case — the answer to the stuck question

### 3.1 The fork, and why it resolves cleanly

There are two incompatible correlated families in play (§0.2). The resolution is
that **they serve different phases, and each is easy in its own phase**:

| | consuming a *fitted* population prior | sampling the population *simultaneously* |
|---|---|---|
| phase | 3–5 | 6 |
| family | truncated MVN, `(mu, sigma, rho)` | logit-normal, Cholesky `scale_tril` |
| bounded support | rectangular truncation | logistic squash |
| `Z` | **build-time constant** — scipy `multivariate_normal.cdf`, computed once, baked in as a float | **does not exist** — no normalising constant needed |
| prior lives on | physical `x` | latent `z` |
| link Jacobian | **required** | **must NOT be applied** |
| matches | `infer_hyperpriors.py` exactly | `hbm_outliers_example.py` exactly |

The blocker everyone expects — "there is no closed form for the bivariate normal
CDF and it must be differentiable" — **evaporates in Phases 3–5**, because with
fixed hyperparameters `Z` is a constant. And in Phase 6, where the hyperparameters
are sampled, the logit-normal family has no `Z` at all. At no point is a
differentiable bivariate normal CDF required.

The one thing that must be reproduced bit-for-bit from `hyper_utils.py:118-127` is
the guard: a box whose Gaussian mass `Z <= 1e-12` is treated as **zero density**
(`Zsafe = inf`). Its justification depends on the `Chi` mixture absorbing those
pulsars, so with `-outliers` off a deep-tail pulsar contributes exactly zero and
drives `logL -> -inf`. That is a semantic decision a port must make explicitly
rather than inherit silently.

### 3.2 The Jacobian trap

This is the single most important line to get right, and it is a silent,
plausible-looking bug if got wrong.

- **Phase 3–5** — the prior is specified on physical `x`. Link Jacobians are part
  of the density. Omitting them gives a wrong but finite posterior.
- **Phase 6** — the prior is specified on the latent `z`; `x` is only a
  deterministic re-expression fed to the likelihood. The contribution is
  `logL(x(z)) + log p_Z(z)` and **nothing else**. Adding a spurious `log|dx/dz|`
  here is wrong.

Both conventions will coexist in `prior.py`. Each family declares which it uses,
and the docstring says so in one line.

Two consequences to document for Phase 6: the prior is Gaussian in
`(log10_A, logit(gamma/7))`, not in `(log10_A, gamma)`; and the reported population
mean must be pushed through the link — `E[gamma] != 7*sigmoid(mu_1)`.

### 3.3 How to express a group in a per-parameter regex dict

Three options were developed.

**(a) A separate `jointpriors` structure keyed by a tuple of regexes.** Backward
compatibility is perfect (`priordict` untouched). Per-pulsar expansion is natural.
Readability is best — mean, covariance, links and outlier spec live in one literal,
in the order the tuple declares. Cost is moderate: one parser, one grouping pass,
one gather/scatter, one density. Con: two sources of truth, needing an explicit
precedence rule and a warning when a grouped parameter also matches a `priordict`
entry.

**(b) A group tag inside per-parameter entries, e.g. `['...', 'MVN:spin:0']`.**
Rejected. The per-parameter entry cannot carry the group's mean/covariance — those
are joint — so a second block is needed *anyway*, giving two sources of truth *plus*
the tags. The component index is encoded in a string (`:0`, `:1`), unvalidatable
until build time and silently wrong if reordered. It also breaks the shape of a
spec: every other entry is numbers-then-family-name; this one is a name with no
numbers.

**(c) A separate `makelogtransform(func, priordict=, jointpriors=)` function.**
Not an alternative data structure — it is (a) plus a signature decision. As a
*parallel* function it fails on plumbing: `samplers/numpyro.py:19`,
`samplers/flow.py:76,232` and `samplers/jaxns.py:12` all default to
`prior.makelogtransform_uniform`, and `flow.py:76`/`232` do not even accept a
`transform=` kwarg. Joint priors would be unreachable from the flow sampler.

**Recommendation: (a) as the data structure, on the existing function via a new
keyword, with the new name aliased to the old object.**

```python
def makelogtransform(func, priordict={}, jointpriors={},
                     parametrisation='noncentered', prior_whiten=True): ...

makelogtransform_uniform = makelogtransform     # same object: all sampler defaults keep working
```

Aliasing to the *same object* is what makes joint priors work through `flow.py` and
`samplers/numpyro.py` without touching either.

Spec shape is a **dict**, not a list — a group has structure that positional lists
render unreadable:

```python
jointpriors = {
    ('(?P<inst>.*_)?red_noise_log10_A', '(?P<inst>.*_)?red_noise_gamma'): {
        'family':  'TruncatedMultivariateNormal',
        'mu':      [-13.8754, 0.7716],
        'sigma':   [0.7114, 0.7988],
        'rho':     [[1.0, 0.7779], [0.7779, 1.0]],
        'bounds':  [(-18.0, -11.0), (0.0, 7.0)],
        'outlier': {'chi': 0.2261, 'bounds': [(-18.0, -11.0), (0.0, 7.0)]},
    }
}
```

These are the real ML values from `red_gauss_A4.0e-15_result.json` and
`red_gauss_physical_outliers_A4.0e-15_result.json`.

### 3.4 Instance discovery and the mandatory gather

**`func.params` is `sorted(set(...))` at every level of `likelihood.py`** (lines
316, 378, 590, 621, 685, 812, 1194). Group members are therefore **not adjacent**.
Live counterexample from `priordict_standard` itself: a `(alpha, log10_A)` group on
the chromatic GP sorts as `chrom_gp_alpha`, `chrom_gp_gamma`, `chrom_gp_log10_A` —
`gamma` sits between them. `(red_noise_gamma, red_noise_log10_A)` happens to be
adjacent, but only by accident of spelling. **The index gather is required for
correctness, not as an optimisation.**

Build-time procedure:

1. `re.fullmatch` each regex in the tuple key against every name in `func.params`.
2. Instance key = text captured by the named group `(?P<inst>...)`; fall back to
   `match.group(1)` if exactly one group, else `''` (single global instance).
   `J0030+0451_red_noise_log10_A` → instance key `J0030+0451_`.
3. Bucket by instance key. Every instance must supply **exactly one** parameter per
   tuple slot — raise otherwise. This catches typos and, importantly, pulsars
   missing one of the two parameters, which would otherwise join a malformed group.
4. Build `idx`, an `(n_inst, d)` int array of **flat** `ys` offsets, reusing the
   `slices` machinery at `prior.py:111-119` so vector parameters are handled.
   Validate that group members are scalar.
5. Record the complement mask so the scalar path skips group members.

At trace time: one gather `zs = ys[idx]` → `(n_inst, d)`, one vectorised density,
one scatter. Negligible against a PTA likelihood, fully `jit`/`vmap`-safe.

### 3.5 Group density, Phase 3 form (fixed hyperparameters, prior on `x`)

```python
def _logpdf_tmvn(x, mu, sigma, rho, bounds, logZ):
    """x: (n_inst, d) physical. logZ: build-time float, the rectangle mass."""
    z    = (x - mu) / sigma                          # (n_inst, d)
    Linv = _chol_inv(rho)                            # build-time constant (d,d)
    w    = z @ Linv.T
    core = (-0.5 * jnp.sum(w**2, -1) - jnp.sum(jnp.log(sigma))
            - jnp.sum(jnp.log(jnp.diag(_chol(rho)))) - 0.5*d*LOG_2PI - logZ)
    inside = jnp.all((x >= lo) & (x <= hi), axis=-1)
    return jnp.where(inside, core, -jnp.inf)
```

plus the per-coordinate tanh link Jacobians (§3.2), and `logaddexp` against the
uniform outlier term when `chi` is present.

`logZ` comes from `scipy.stats.multivariate_normal.cdf` inclusion–exclusion at
build time: 4 terms in 2-D, 8 in 3-D (`hyper_utils.py:195-204`).

---

## 4. Two pre-existing defects that must be fixed as part of this

### 4.1 The base-distribution factor in `makemodel_transformed`

`samplers/numpyro.py:25-28`:
```python
pars = numpyro.sample('pars', dist.Normal(0, 10).expand([parlen]))   # contributes log N(pars; 0, 10)
logl = logx.logL(pars)
numpyro.deterministic('log_likelihood', logl)
numpyro.factor('logl', logx(pars))                                   # contributes logL + logprior
```

The sampled target is `logL + logprior + logN(y; 0, 10)` — an **extra N(0,10)
factor on the unconstrained vector**. For tanh coordinates this is harmless (it
excludes only `|y| ≳ 40`, i.e. `x` within ~1e-35 of the box edge). For the new
**unbounded** families it is not: a whitened Normal prior gets precision `1 + 0.01`,
shrinking the posterior sd by 0.5%; an unwhitened identity map would be
catastrophic. It also hits the non-centered `z~` coordinates in Phase 6, where it
shrinks the inferred population scatter — the quantity being measured.

Fix, one line, zero effect on existing runs:
```python
scale = getattr(logx, 'base_scale', 10.0)
pars  = numpyro.sample('pars', dist.Normal(0.0, scale).expand([parlen]))
```
with `base_scale = 10.0` for every tanh coordinate (bit-identical to today) and
something large — 1e3, or `dist.ImproperUniform` — for unbounded and non-centered
coordinates.

### 4.2 The `to_vec` clamp

Recommended fix (§0.5), applied identically to `prior.py:163` and `prior.py:218`:

```python
u = (a + b - 2*xs)/(a - b)

# arctanh(+/-1) is infinite: a parameter sitting exactly on a prior edge maps to
# an infinite coordinate. Pull |u| == 1 back by half an ulp, giving the largest
# finite coordinate the transform can represent. |u| > 1 is left alone so that
# out-of-prior values still give NaN.
eps = jnp.finfo(u.dtype).eps
u = jnp.where(jnp.abs(u) == 1.0, jnp.sign(u) * (1.0 - 0.5*eps), u)

return jnp.arctanh(u)
```

`0.5*eps = 1.11e-16` clamps to `nextafter(1, 0)`, whose `arctanh` is
`18.714973875118524` — already the largest finite value the unmodified expression
can return. **The clamp therefore cannot alter any previously-finite value**; it
only replaces `±inf` with `±18.715`.

Verified: `to_dict` output changes by exactly `0.0` bitwise for every input;
`to_vec` is bitwise identical on 200,000 random interior points; over a dense grid
of `y ∈ [-25, 25]`, `max|new - old| = 0.0` wherever the old code was finite.
Sampling, likelihoods, evidences and stored chains are unaffected by construction.

Note in the commit message: the `isfinite` guard at `numpyro.py:416` will stop
firing for overrides sitting *exactly* on a prior edge (it still fires for overrides
outside the prior). Its message already says "strictly within their priors".

`samplers/flow.py:247` has **no** such guard and should gain one.

---

## 5. Phasing

Each phase is independently testable and independently revertable. Nothing is
committed without review, and **the installed clone at `/fred/oz002/dreardon/discovery`
is not touched until a phase has landed and the queue is drained** — live jobs
import that tree at startup.

### Phase 0 — foundations, no behaviour change

- `parse_spec()` + `FAMILIES` registry + `getsupport()`.
- Shape guards in all five first-match-wins loops (`prior.py:86, 97, 126, 203, 257`)
  so a tagged spec can never be read as a box (§1.1).
- `to_vec` clamp, both transforms (§4.2). `flow.py:247` finiteness guard.
- `base_scale` attribute plumbed into `makemodel_transformed` (§4.1).
- **Test: bit-comparability.** `makelogtransform(f)` vs the pre-change
  `makelogtransform_uniform(f)`, compared on raw int64 bit patterns over a few
  hundred random `ys`. Segmenting the `jnp.sum` by family would reorder float
  additions and change the last ulp, so an all-uniform **fast path taking today's
  code verbatim** is the only way to guarantee bit-identity rather than mere
  equality.

### Phase 1 — scalar families

Uniform (fast path), Normal (whitened), TruncatedNormal (tanh + baked `logZ`).
`sample_prior()` added; `sample_uniform` keeps working for uniform entries and
raises a clear error for others. `getprior_uniform` keeps its 2-tuple contract for
Uniform specs and raises `"family X has no uniform box; use getsupport()"` for the
rest — it is called from `samplers/numpyro.py:40` and `samplers/jaxns.py:22`, both
of which genuinely need a box.

### Phase 2 — outlier mixtures

The three requested families plus `NormalWithNormalOutliers` (§2.4). Union-support
rule (§2.3). Build-time discontinuity warning. Responsibility exposed as a
deterministic.

### Phase 3 — joint N-D priors, fixed hyperparameters

`jointpriors` (§3.3), instance discovery and gather (§3.4),
`TruncatedMultivariateNormal` in 2-D and 3-D with build-time `Z` (§3.5), optional
uniform-outlier mixture. Reproduces `infer_hyperpriors.py` exactly.

### Phase 4 — nested sampling

`samplers/jaxns.py:16-46` builds the prior explicitly as
`tfpd.Uniform(low=..., high=...)` and never sees `transformed` at all. Non-uniform
families cannot work there without real work. Scope: raise a clear
`NotImplementedError` naming the family, and document that `-ns` runs are
uniform-only. Revisit separately.

### Phase 5 — platypus plumbing

- Manifest v2: an optional per-parameter family line alongside the existing bare
  path list, which keeps working (§0.4).
- `load_hyperprior_overrides` gains a gauss path — delete the hard raise at
  `hyper_utils.py:1156-1160`, emit the `(mu, sigma, rho, chi)` spec instead of a
  box.
- Preserve the in-place mutation of `prior.priordict_standard` and the discarded
  return value at `common_noise.py:176`. **This is not incidental — it is the only
  wiring.** Nothing passes a `priordict` to the sampler
  (`common_noise.py:273`, `single_pulsar_noise.py:168` both call
  `ds_numpyro.makemodel_transformed(model.logL)` with no `priordict` argument), so a
  non-mutating version would silently sample the *default* `[0,7]` box with no error.
- Note the stale docstring at `hyper_utils.py:1129-1131`: the returned dict maps
  `key -> ([lo,hi], is_regex)`, a 2-tuple, so it is **not** usable as a `priordict=`
  argument as it claims.
- Fix the dead `ecorr_beta` path (`hyper_utils.py:1042, 1117`) — no fit model
  defines it, so `beta` is always `0.0`.
- Fix the `(physical)` log tag at `hyper_utils.py:1178`, printed from
  `component in physical_set` rather than from whether a shift was applied. Every
  red/dm/chrom **gamma** line in `job_outputs/` is labelled `(physical)` while
  carrying an unshifted, array-wide-identical box.
- Generalise the hard-coded `KAT_MKBF` white-noise keys
  (`infer_hyperpriors.py:151-188`) before this is used on IPTA data.

### Phase 6 — simultaneous hierarchical sampling

The logit-normal family (§3.1), non-centered by default.

**Hyperparameters ride in `to_dict`.** They are not in `func.params`, so the
transform owns them: `transform.params = list(func.params) + hypernames`.
`logL(ys) = func(to_dict(ys))` is safe because discovery's likelihood closures index
the dict by name and never enumerate it — verified by grep over `likelihood.py`,
`signals.py`, `matrix.py`: only build-time name-list checks, no runtime
`params.items()`. `parlen` in `numpyro.py:22` and `flow.py:234` is computed by
parsing `(n)` off `logx.params`, so appended scalar names just work, and `to_df`
gains hyperparameter columns — which is what you want in the chain.

Sharp edge: `to_vec(params)` now needs values for the hyperparameters. Default
missing keys to the prior midpoint and document it, or the override machinery at
`numpyro.py:415` / `flow.py:247` breaks.

**`logprior` needs no signature change** — it already takes the whole `ys`; only its
internals change from closed-over constants to gathering hyperparameter slots out of
`ys`. This is the key API observation: **the existing design already supports
hierarchical priors.** The only new surface is a way to declare a hyperparameter and
reference it.

Declaration rule — **any string where a number is expected names a
hyperparameter**, whose own scalar prior is looked up in `priordict` as usual. No
new syntax:

```python
('(?P<inst>.*_)?red_noise_log10_A', '(?P<inst>.*_)?red_noise_gamma'): {
    'family': 'MultivariateNormal',
    'link':   ('identity', ('logistic', 0.0, 7.0)),
    'mu':     ['spin_log10_A_mu', 'spin_gamma_mu'],
    'chol':   [['spin_L_amp', 0.0],
               ['spin_L_12',  'spin_L_gamma']],
    'outlier': {'chi': 'spin_Q', 'chol': [[7.0, 0.0], [0.0, 6.0]]},
}
priordict = {..., 'spin_log10_A_mu': [-17, -12], 'spin_gamma_mu': [-4, 4],
                  'spin_L_amp': [0.3, 3.0], 'spin_L_gamma': [0.3, 3.5],
                  'spin_L_12': [-1.5, 1.5], 'spin_Q': [0.0, 1.0]}
```

Those boxes are exactly `hbm_outliers_example.py:5-9, 32`. Positivity of `L_jj` is
handled by the ordinary tanh box.

**Centered vs non-centered.** Centered is Neal's funnel: as `L_jj -> 0` the prior
tightens as `L^-n_inst` while all `z_k` collapse to within `O(L)` of `mu`. Symptoms
are divergences and `L` **biased low** — the sampler cannot reach the tip, so it
under-reports the population scatter, which is exactly the quantity being measured.
In PTA red noise a large fraction of pulsars have essentially unconstrained spin
noise, which is precisely the weak-data regime where centered fails.
**Recommendation: non-centered by default**, `'centered'` available.

The example has non-centered commented out (lines 46-47) because a mixture of two
MVNs with different covariances cannot be non-centered by a single linear map. It
**can** be handled exactly, and those lines should be revived:

```
z = mu + L z~,   |dz/dz~| = |L| = prod L_jj
chi_fg * N(z; mu, LL^T) * |L| == chi_fg * N(z~; 0, I)          exactly

log p~(z~) = logaddexp( log(chi_fg) + logN(z~; 0, I),
                        log(chi_out) + logN(mu + L z~; mu, Sigma_out) + sum_j log L_jj )
```

The funnel is **fully removed for the foreground**. Residual `L`-dependence sits
only in the outlier term, and `Sigma_out` is fixed in the example
(`scale_tril=[[7,0],[0,6]]`), so that term is smooth and mild. Partial rather than
complete mitigation — say so in the docstring — but a large improvement over
centered.

**Label switching:** none while `Sigma_out` is fixed and much wider than `LL^T`. If
`Sigma_out` ever becomes sampled, the components can swap and `chi` goes bimodal at
`chi` and `1-chi`. Guard with an ordering constraint (`det Sigma_out > det LL^T`) or
keep it fixed; check this in the spec parser.

**Physical scaling** enters as a deterministic per-instance shift of `mu`, exactly
as `hyper_utils.py:388-405, 487-545` does it:

| component | shifted coord | shift | covariate | coefficient |
|---|---|---|---|---|
| red | `log10_A` | `alpha * log10(Edot/1e33)` | P0, P1 (Shklovskii- and Galactic-corrected) | `red_alpha` |
| dm | `log10_A` | `alpha * log10(DM/30)` | DM | `dm_alpha` |
| chrom | `log10_A` | `alpha * log10(DM/30)` | DM | `chrom_pop_alpha` |
| sw | `log10_sigma` | asymmetric Gaussian in `sin|ELAT|` | ELAT | `sw_elat_mu`, `sw_elat_sigma_lo/hi` |
| ecorr | `log10_ecorr` | `-0.5*log10(tobs/3600) + alpha*log10(P0/1e-3)` | tobs, P0 | `ecorr_alpha` |

`tobs` is the **median of the `-tobs` flags in the .tim file** — per-observation
integration time, not the data span. A jitter law, so it must read the same flag.

Note `make_sw_gauss_physical` (`hyper_utils.py:541-545`) **swaps the coordinate
order** when calling `truncated_normal_2d_perpsr`, because the shift acts on `G` and
the per-pulsar-broadcast slot is always the first coordinate. Easy to get wrong.

White noise stays fixed in this phase, per the brief.

### Phase 7 — validation

1. **Exact regression.** All-uniform specs reproduce current behaviour bitwise
   (Phase 0 test, extended to a real MPTA model build).
2. **Population reproduction.** With a fitted Gaussian population prior loaded, the
   array's `red_noise_gamma` distribution is no longer truncated: check the
   posterior has support above 2.42 and that the induced population matches the
   `red_gauss_*` ML row.
3. **The J0437 test.** See §6.
4. **Round-trip.** `to_vec(to_dict(y))` finite and monotone for all `y` including
   `inf`, every family.
5. **Normalisation.** Quadrature of `exp(logprior(y))` = 1 for every family.
6. **Not blocking, but the cheapest empirical check on the conditioning claims in
   §2.1:** a synthetic 20-parameter target mixing a `sd=0.05` Normal with wide
   Uniforms, comparing divergences and `n_eff` for whitened vs identity.

---

## 6. The motivating case — corrected

Independent recomputation from the MPTA 6-yr chains found the stated numbers
oversold. Recording the corrections so the case is stated on ground that holds.

| claim | verdict |
|---|---|
| box truncates `[0,7]` → `[0.601, 2.416]` | **partly** — the real box is `[0.5943, 2.4207]`; no result file on disk yields `[0.601, 2.416]` |
| 80.7–85.5% of free-fit medians above the ceiling | **not confirmed** — actual range across 14 model variants is **68.7–81.9%** (57–68 of 83); 85.5% is unreachable |
| only 9–11% of samples near an edge | **confirmed** at eps≈0.05 two-sided, but wildly definition-dependent (1.7%–38% over eps ∈ [0.01, 0.20]) |
| pivot 0.048/yr, `dlgP(f1)` = +0.71…+1.06 | **confirmed** — pivot 0.035–0.051/yr, `dlgP(f1)` = +0.83…+0.99 median |
| "+3 to +4 dex by 12/yr" | **confirmed** (+3.91…+4.06), though 12/yr is mid-band, not the edge: with `fftint=True, max_cadence_days=14`, `psd2cov` gives a band of `[1/T, 26/yr]` |
| "adds power to **every** pulsar" | **false** — `dlgP(f1) > 0` for 69–80% of pulsars; a fifth to a third *lose* low-frequency power |

The gamma-ceiling statistic is close to vacuous on its own: free-fit gamma is
prior-dominated (median KS distance to `Uniform(0,7)` is 0.08–0.14; the 68% credible
width is 0.63× the prior width against 0.68 for a pure uniform), so "most medians
exceed 2.42" largely restates "2.42 < 3.5". Per-pulsar `P(gamma > 2.4207)` has
median 0.58–0.61 against the prior value 0.654 — the data mildly prefer *lower*
gamma. **Only 1 of 83 pulsars excludes the ceiling at 95% credibility.**

Likewise, railing fails to fire not because the truncation is subtle but because the
constrained posterior *is* the prior: median KS distance to `Uniform(box)` is 0.068,
pooled median gamma 1.516 against a box midpoint of 1.508. No diagnostic built on
posterior shape can ever fire against that.

**What does survive, and is the real case:**

1. **`dlgA` = +1.10 dex median.** Forcing gamma from ~3.1 to ~1.5 while holding
   low-frequency power roughly fixed drives the amplitude up by an order of
   magnitude, and because the spectrum is now flat that amplitude propagates to
   every frequency. The mechanism reproduces.
2. **The amplitude floor is doing at least as much damage as the gamma ceiling, and
   the brief ignores it.** The same box truncates `log10_A` from `[-18,-11]` to
   `[-15.78, -12.38]`, and **33 of 83 pulsars have free-fit `log10_A` medians below
   -15.78** — for 40% of the array the prior imposes a red-noise amplitude floor the
   data do not support.
3. **Negligible for the median pulsar, severe for a minority.** Integrating `P(f)`
   over `[5/yr, 26/yr]`, added rms is 9 ns median against a 516 ns median
   epoch-averaged TOA uncertainty — 2.4%, harmless. But 23/83 pulsars exceed 10% of
   their epoch sigma and **6 exceed 100%**: J1017-7156 (3.2x), J0900-3144 (2.2x),
   J1643-1224 (1.9x), **J0437-4715 (1.5x)**. J0437 free-fits to gamma 3.18 /
   lgA -15.85 (below the box floor) and is constrained to gamma 1.33 / lgA -14.50,
   turning a 0.07 ps high-band red rms into 11.3 ps against a 7.4 ps epoch sigma.

So the defensible motivating sentence is not "the box truncates 80% of pulsars and
adds power to every one", but:

> The box is a hard rectangle taken from a single ML sample of the hierarchical
> fit; it is inconsistent with the free-fit amplitude posterior for 40% of the
> array; and for ~7 pulsars including J0437-4715 it injects red power at or above
> the white-noise level in band.

Caveat on the comparison: the `_hyperprior` runs use 2/3-mode correlated Legendre
ECORR while the free-fit runs use 1-mode or plain/quadratic. No matched
non-hyperprior 2/3-mode run exists in the tree, so free-vs-constrained conflates the
prior box with the ECORR model. Bounded: gamma medians across the four free ECORR
variants agree to 0.03–0.20 and lgA to <0.15 dex, against +1.10 dex
free→constrained, so the box dominates — but it is not a clean control. **Running
that control is worth doing before the science claim is made.**

---

## 7. Open questions

1. **The `Z <= 1e-12 -> zero density` convention** (`hyper_utils.py:118-127`) is
   only well-justified when `-outliers` is on. With outliers off, a deep-tail pulsar
   contributes exactly zero and drives `logL -> -inf` (masked by bilby's
   `np.nan_to_num`). A port must decide this explicitly.
2. **Should `to_df` report `x` or `z` for group members?** Assumed `x`, consistent
   with the rest of `to_df` — but the population diagnostics and any comparison
   against `mu` live in `z`. Suggest `x` plus optional `z` columns under `raw=True`.
3. **Prior on `L`.** Flat boxes on Cholesky entries (as in the example) imply a
   non-obvious prior on the correlation. LKJ or a half-Normal on `L_jj` would be
   better. Not blocking; note as future work.
4. **`UniformWithOutliers` is the weakest family in the set** (§2.4). Needed to
   reproduce `infer_hyperpriors.py`, but if it is wanted only for symmetry, prefer
   the tapered variant or drop it.
5. **`prior_whiten` for TruncatedNormal needs a build-time 1-D quadrature** to pick
   the scale. Cheap and exact, but it adds a scipy call at trace-construction time.
   `kappa ~ h/sigma` clipped to `[1,100]` captures most of the benefit analytically
   if that is unwelcome.

## 8. Incidental defects found while scoping

Not part of this work; recorded so they are not lost.

- `infer_hyperpriors.py:194-212` pops pulsars missing par/tim metadata from
  `chains`, but `psr_names` was captured **before** the pop, so
  `p0 = [params[p]["P0"] for p in psr_names]` raises `KeyError` for exactly those
  pulsars. The drop is ineffective, and `posteriors` was already built, so it would
  desynchronise anyway.
- `-physical efac` / `-physical equad` pass validation and reach the label
  (`_phys-efac`) but have no physical variant in the univariate branch — a
  non-physical fit with a physical label.
- `hyper_utils.py` defines `normal_1d` and `normal_2d` (untruncated) which are never
  called.
- `flow.py:102` uses `StandardNormal((len(logx.params),))` where `flow.py:234`
  correctly computes `parlen` — wrong dimension whenever any vector parameter
  exists.
- `flow.py:227` does `p1, p2 = self.logx.params[-2:]` then indexes the DataFrame by
  bare name — already raises `KeyError` if the last parameter is a vector.
- `prior.py:150`'s `if len(a) != len(func.params)` is a proxy for "are there vector
  parameters" and is wrong for a single-element vector `x(1)` alongside scalars.
- `scripts/test_buildguard.py:94-95` declares `--hyperpriors` and branches on it but
  calls `mpta.update_priordict_standard_mpta()` instead of
  `load_hyperprior_overrides` — it does not exercise the path its docstring
  advertises.
- `data/*/results/hyperpriors.txt` does not exist for any dataset despite
  `make_manifests.py:13,63` generating it and every `launch_common.sh` line
  referencing it (all commented out).
