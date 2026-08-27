"""priordict_standard is one shared dict: the priors must follow the model, not the imports.

The model modules are imported inside a fixture rather than at module scope. Importing
either runs its updater, and at module scope that happens during COLLECTION, mutating the
shared dict before any test in any other file runs -- which breaks the files asserting
discovery's own defaults.
"""

import inspect
from pathlib import Path

import pytest

import discovery as ds
from discovery import prior


DATA = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture
def models():
    from discovery.models import mpta, ppta
    return mpta, ppta


def box(par):
    return [float(v) for v in prior._matchprior(par, prior.priordict_standard)]


def production_state(models):
    """Both models imported, then the built model's updater applied, as a run reaches it."""
    mpta, ppta = models
    ppta.update_priordict_standard_ppta()
    mpta.update_priordict_standard_mpta()


# --- the collision, and that building a model settles it -------------------------

def test_the_two_models_really_do_collide(models):
    """Both declare the same keys with different values, each right for its own array."""
    mpta, ppta = models

    mpta.update_priordict_standard_mpta()
    assert (box('J0437-4715_KAT_efac'),
            box('J0437-4715_red_noise_log10_fc')) == ([0.5, 2.0], [-10.5, -7.5])

    ppta.update_priordict_standard_ppta()
    assert (box('J0437-4715_KAT_efac'),
            box('J0437-4715_red_noise_log10_fc')) == ([0.1, 5.0], [-11.54, -7.5])


def test_building_an_mpta_model_installs_mptas_boxes(models):
    """models.ppta imported last used to leave every mpta run on ppta's efac box."""
    mpta, ppta = models

    f = DATA / "v1p1_de440_pint_bipm2019-J0030+0451.feather"
    if not f.exists():
        pytest.skip("pulsar data fixture missing")
    psr = ds.Pulsar.read_feather(f)

    ppta.update_priordict_standard_ppta()
    assert box(f'{psr.name}_430_ASP_efac') == [0.1, 5.0]

    mpta.single_pulsar_noise(psr, turnover=('red',))
    assert box(f'{psr.name}_430_ASP_efac') == [0.5, 2.0]
    assert box(f'{psr.name}_red_noise_log10_fc') == [-10.5, -7.5]


def test_building_a_ppta_model_installs_pptas_boxes(models):
    """The other half of the regression: mpta imported last must not win either."""
    mpta, ppta = models

    f = DATA / "v1p1_de440_pint_bipm2019-J0030+0451.feather"
    if not f.exists():
        pytest.skip("pulsar data fixture missing")
    psr = ds.Pulsar.read_feather(f)

    mpta.update_priordict_standard_mpta()
    assert box(f'{psr.name}_430_ASP_efac') == [0.5, 2.0]

    ppta.single_pulsar_noise(psr, red=False, dm=False, chrom=False, chrom_poly=False)
    assert box(f'{psr.name}_430_ASP_efac') == [0.1, 5.0]
    assert box(f'{psr.name}_red_noise_log10_fc') == [-11.54, -7.5]


def test_every_model_entry_point_installs_its_own_priors(models):
    """Structural guard against a NEW entry point, which no built model can express."""
    mpta, ppta = models
    assert 'update_priordict_standard_mpta()' in inspect.getsource(mpta.single_pulsar_noise)
    assert 'update_priordict_standard_mpta()' in inspect.getsource(mpta.common_noise)
    assert 'update_priordict_standard_ppta(' in inspect.getsource(ppta.single_pulsar_noise)
    assert 'update_priordict_standard_ppta(' in inspect.getsource(ppta.common_noise)


def test_a_global_ecorr_is_detected_from_the_chain(models):
    """{psr}_log10_ecorr, not {psr}_ecorr, which matched no parameter mpta produces."""
    mpta, _ = models

    f = DATA / "v1p1_de440_pint_bipm2019-J0030+0451.feather"
    if not f.exists():
        pytest.skip("pulsar data fixture missing")
    psr = ds.Pulsar.read_feather(f)

    kw = dict(red=False, dm=False, chrom=False, sw=False, chrom_poly=False)
    with_global = mpta.single_pulsar_noise(psr, global_ecorr=True, **kw).logL.params
    without = mpta.single_pulsar_noise(psr, global_ecorr=False, **kw).logL.params

    probe = f'{psr.name}_log10_ecorr'
    assert any(probe in c for c in with_global)
    assert not any(probe in c for c in without)


def test_ppta_common_noise_delegation_keeps_its_own_boxes(models):
    """ppta.common_noise wraps mpta.common_noise, which must not reinstall MPTA's.

    Resolves the box after the call rather than checking that install_priors=False
    appears in the source. The source-text version passed while the flag was defeated one
    level down, by mpta.common_noise's own calls to single_pulsar_noise.
    """
    mpta, ppta = models

    f = DATA / "v1p1_de440_pint_bipm2019-J0030+0451.feather"
    if not f.exists():
        pytest.skip("pulsar data fixture missing")
    psr = ds.Pulsar.read_feather(f)

    ppta.update_priordict_standard_ppta()
    mpta.single_pulsar_noise(psr, turnover=('red',), install_priors=False)

    assert box(f'{psr.name}_430_ASP_efac') == [0.1, 5.0]
    assert box(f'{psr.name}_red_noise_log10_fc') == [-11.54, -7.5]


def test_install_priors_reaches_the_per_pulsar_rebuilds(models):
    """common_noise builds each pulsar through single_pulsar_noise, which installs too."""
    mpta, _ = models
    assert 'install_priors' in inspect.signature(mpta.single_pulsar_noise).parameters
    assert inspect.getsource(mpta.common_noise).count(
        'install_priors=install_priors') == 2


def test_a_front_inserted_per_pulsar_override_survives_the_reinstall(models):
    """Hyperprior overrides are inserted ahead of the generic keys and must stay there.

    dict.update leaves the position of an existing key alone and appends only new ones,
    so the front-inserted override still wins the first-match scan. Rebuilding the dict
    with clear() would drop it silently, which is why this is pinned.
    """
    mpta, _ = models

    key = r'J0437\-4715_red_noise_log10_A'
    rest = dict(prior.priordict_standard)
    prior.priordict_standard.clear()
    prior.priordict_standard.update({key: [-14.5, -13.5], **rest})
    assert box('J0437-4715_red_noise_log10_A') == [-14.5, -13.5]

    mpta.update_priordict_standard_mpta()
    assert box('J0437-4715_red_noise_log10_A') == [-14.5, -13.5]


def test_mpta_tnequad_declaration_actually_takes_effect(models):
    """'(.*_)?log10_tnequad' was dead behind the stock '(.*_)?tnequad'."""
    mpta, _ = models
    mpta.update_priordict_standard_mpta()
    assert box('J0437-4715_KAT_log10_tnequad') == [-10.0, -5.0]


def _stage1_chain(psr, split=None):
    """A minimal stage-1 chain, optionally written by a run that split the white noise.

    split=None names the efacs per backend, as the default does; split='chan' names them
    {psr}_chan<N>_efac, as white_selection='chan' does.
    """
    import numpy as np
    import pandas as pd

    if split is None:
        labels = sorted(set(np.asarray(psr.backend_flags).tolist()))
    else:
        labels = sorted({f'{split}{v}' for v in set(np.asarray(psr.flags[split]).tolist())})

    cols = [f'{psr.name}_red_noise_log10_A', f'{psr.name}_red_noise_gamma']
    for lab in labels:
        cols += [f'{psr.name}_{lab}_efac', f'{psr.name}_{lab}_log10_tnequad',
                 f'{psr.name}_{lab}_log10_ecorr']

    df = pd.DataFrame({c: np.linspace(-8.0, -7.0, 8) if 'log10' in c
                       else np.linspace(0.9, 1.1, 8) for c in cols})
    df.attrs['noisedict'] = {}
    return df


def test_a_white_selection_mismatch_is_reported(models, capsys):
    """A stage-1 run split per channel names its efacs {psr}_chan0_efac.

    A rebuild at the default per-backend split never matches those names, so the values
    are not applied and the parameters fall back on their priors -- silently, because
    from the model's point of view nothing is missing.
    """
    mpta, _ = models

    f = DATA / "v1p1_de440_pint_bipm2019-J0030+0451.feather"
    if not f.exists():
        pytest.skip("pulsar data fixture missing")
    psr = ds.Pulsar.read_feather(f)
    if 'chan' not in psr.flags:
        pytest.skip("fixture has no chan flag")

    mismatched = _stage1_chain(psr, split='chan')

    mpta.common_noise([psr], [mismatched], fd=False, noise_point='median')
    assert 'efac parameter(s) this rebuild cannot produce' in capsys.readouterr().out

    # naming the split the stage-1 run used silences it
    mpta.common_noise([psr], [mismatched], fd=False, noise_point='median',
                      white_selection='chan')
    assert 'cannot produce' not in capsys.readouterr().out


def test_a_matching_white_split_is_not_reported(models, capsys):
    """Nothing to warn about when the names line up, whatever is left free."""
    mpta, _ = models

    f = DATA / "v1p1_de440_pint_bipm2019-J0030+0451.feather"
    if not f.exists():
        pytest.skip("pulsar data fixture missing")
    psr = ds.Pulsar.read_feather(f)

    matching = _stage1_chain(psr)
    mpta.common_noise([psr], [matching], fd=False, noise_point='median')
    assert 'cannot produce' not in capsys.readouterr().out

    # a whole family deliberately absent leaves the NAMES intact, so still nothing
    dropped = matching.drop(columns=[c for c in matching.columns if 'ecorr' in c])
    mpta.common_noise([psr], [dropped], fd=False, noise_point='median')
    assert 'cannot produce' not in capsys.readouterr().out


def test_white_selection_is_threaded_to_the_per_pulsar_rebuilds(models):
    """Structural guard: the argument must reach both call sites, not just exist."""
    mpta, _ = models
    assert 'white_selection' in inspect.signature(mpta.common_noise).parameters
    assert inspect.getsource(mpta.common_noise).count(
        'white_selection=white_selection') == 2


# --- the guard against the next one -----------------------------------------------

def test_shadowed_priors_catches_the_log10_absorption():
    d = {'(.*_)?tnequad': [-8.5, -5], '(.*_)?log10_tnequad': [-10, -5]}
    shadowed, unchecked = prior.shadowed_priors(d)
    assert shadowed == [('(.*_)?log10_tnequad', '(.*_)?tnequad', False)]
    assert not unchecked


def test_shadowed_priors_does_not_fire_on_a_merely_overlapping_key():
    """An anchored key cannot shadow a prefixed one: it never matches J0000+0000_x."""
    d = {'curn_log10_A': [-18, -11], '(.*_)?curn_log10_A': [-18, -11]}
    assert prior.shadowed_priors(d)[0] == []


def test_shadowed_priors_reports_a_trailing_star_as_redundant():
    """re.match is unanchored at the end, so a trailing .* adds nothing."""
    d = {'(.*_)?red_noise_gamma': [0, 7], '(.*_)?red_noise_gamma.*': [0, 7]}
    assert prior.shadowed_priors(d)[0] == [
        ('(.*_)?red_noise_gamma.*', '(.*_)?red_noise_gamma', True)]


# The dead keys in the state a production run reaches. Every one is harmless -- the key
# shadowing it carries the same box -- and they are recorded so a NEW dead declaration
# fails here rather than silently. A key whose box DISAGREES with the one in force is a
# live bug, covered separately below.
KNOWN_SHADOWED = {
    ('(.*_)?log10_ecorr_q.*', '(.*_)?log10_ecorr'),
    ('(.*_)?red_noise2_gamma.*', '(.*_)?red_noise2_gamma'),
    ('(.*_)?red_noise2_log10_A.*', '(.*_)?red_noise2_log10_A'),
    ('(.*_)?red_noise_gamma.*', '(.*_)?red_noise_gamma'),
    ('(.*_)?red_noise_log10_A.*', '(.*_)?red_noise_log10_A'),
    ('gw_gamma', 'gw_(.*_)?gamma'),
    ('gw_log10_A', 'gw_(.*_)?log10_A'),
}


def test_no_new_prior_key_is_shadowed(models):
    production_state(models)
    shadowed, _ = prior.shadowed_priors()
    assert {(d, by) for d, by, _ in shadowed} == KNOWN_SHADOWED


def test_no_shadowed_key_disagrees_with_the_key_that_shadows_it(models):
    """A dead declaration whose box differs from the one in force is a live bug."""
    production_state(models)
    shadowed, _ = prior.shadowed_priors()
    assert [(d, by) for d, by, agree in shadowed if not agree] == []


def test_the_orbital_dm_gaussian_has_a_prior_for_every_parameter_it_samples(models):
    """Its amplitude is named dm_orb_amp, which '(.*_)?orbital_dm_amp' never matched."""
    production_state(models)
    for par in ('J0437-4715_orbital_dm_dm_orb_amp', 'J0437-4715_orbital_dm_phi0',
                'J0437-4715_orbital_dm_sigma_phi'):
        prior._matchprior(par, prior.priordict_standard)  # KeyError if absent


def test_the_orbital_dm_fourier_component_is_gone(models):
    mpta, _ = models
    assert 'orbital_dm_fourier' not in inspect.signature(mpta.single_pulsar_noise).parameters
    assert 'orbital_dm_fourier' not in inspect.getsource(mpta.single_pulsar_noise)
    production_state(models)
    assert not [k for k in prior.priordict_standard if 'orbital_dm_fourier' in k]
    assert not [k for k in prior.priordict_standard if 'orbital_dm_gp' in k]


# --- the alias -------------------------------------------------------------------

def test_the_uniform_transform_alias_keeps_the_same_signature():
    assert (inspect.signature(prior.makelogtransform_uniform)
            == inspect.signature(prior.makelogtransform))


def test_the_uniform_transform_alias_resolves_at_call_time(monkeypatch):
    """Bound as an import-time alias, patching makelogtransform never reached it."""
    sentinel = object()
    monkeypatch.setattr(prior, 'makelogtransform', lambda *a, **k: sentinel)
    assert prior.makelogtransform_uniform(None) is sentinel
