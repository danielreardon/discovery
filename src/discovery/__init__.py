"""Discovery"""
from __future__ import annotations

import jax
jax.config.update("jax_enable_x64", True)

from .const import *
from .matrix import *
from .prior import *
from .signals import *
from .likelihood import *
from .params import *
from .optimal import *
from .solar import *
from .pulsar import *
from .deterministic import *


_KERNELS = "matrix"
_GP_ORIG = None
_LIKELIHOOD_CLASSES = ("PulsarLikelihood", "GlobalLikelihood", "ArrayLikelihood")


def config(kernels=None):
    """Select the kernel-implementation subsystem the top-level likelihoods use.

    Parameters
    ----------
    kernels : {'matrix', 'metamath'}, optional
        - 'matrix'   : the legacy closure-based path (`matrix.py` + `likelihood.py`).
        - 'metamath' : the graph-based path (`metamath.py` + `likelihood_metamath.py`).

    Returns the current kernels setting if called with no arguments.

    Notes
    -----
    This branch's ``signals.py`` is not migrated to the ``_kernels`` factory; it
    constructs kernels via ``matrix.*``. So for the metamath path we (a) set the
    ``_kernels`` factory mode (for any code that *is* migrated, e.g. the new
    ``measurement_noise``/``recipes`` modules) and (b) install the
    ``matrix.* -> metamath`` monkeypatch via ``_kernel_switch`` so the existing
    ``matrix.*`` call sites resolve to metamath classes. We also rebind the
    top-level likelihood classes. Call ``config()`` *before* constructing models;
    class references already imported into user code are not updated.
    """
    global _KERNELS

    if kernels is None:
        return _KERNELS

    if kernels not in ("matrix", "metamath"):
        raise ValueError(
            f"unknown kernels {kernels!r}; expected 'matrix' or 'metamath'"
        )

    from . import _kernels
    from . import _kernel_switch
    from . import matrix as _matrix
    from . import utils as _utils

    _kernels.set_mode(kernels)

    # This branch's signals.py builds GP containers via ``matrix.ConstantGP`` /
    # ``matrix.VariableGP``; ``likelihood_metamath`` classifies GPs with
    # ``utils.ConstantGP`` / ``utils.VariableGP`` (isinstance). Without swapping
    # these too, the metamath likelihood finds zero GPs and silently drops them.
    global _GP_ORIG
    if kernels == "metamath":
        _kernel_switch.apply_patches()
        if _GP_ORIG is None:
            _GP_ORIG = (_matrix.ConstantGP, _matrix.VariableGP)
            _matrix.ConstantGP, _matrix.VariableGP = _utils.ConstantGP, _utils.VariableGP
        from . import likelihood_metamath as _src
    else:
        _kernel_switch.restore_patches()
        if _GP_ORIG is not None:
            _matrix.ConstantGP, _matrix.VariableGP = _GP_ORIG
            _GP_ORIG = None
        from . import likelihood as _src

    import sys
    pkg = sys.modules[__name__]
    for name in _LIKELIHOOD_CLASSES:
        setattr(pkg, name, getattr(_src, name))

    _KERNELS = kernels


__version__ = "0.5"
