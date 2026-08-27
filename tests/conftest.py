#   ---------------------------------------------------------------------------------
#   Copyright (c) Microsoft Corporation. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root for information.
#   ---------------------------------------------------------------------------------
"""
This is a configuration file for pytest containing customizations and fixtures.

In VSCode, Code Coverage is recorded in config.xml. Delete this file to reset reporting.
"""

from __future__ import annotations

from typing import List

import pytest
from _pytest.nodes import Item


def pytest_collection_modifyitems(items: list[Item]):
    for item in items:
        if "_int_" in item.nodeid:
            item.add_marker(pytest.mark.integration)


@pytest.fixture
def unit_test_mocks(monkeypatch: None):
    """Include Mocks here to execute all commands offline and fast."""
    pass


@pytest.fixture(scope="session")
def _priordict_baseline():
    """The shared prior dict as collection left it, before any test has run."""
    from discovery import prior

    return dict(prior.priordict_standard)


@pytest.fixture(autouse=True)
def restore_priordict(_priordict_baseline):
    """Give every test the same priordict_standard, whatever the previous one did to it.

    prior.priordict_standard is a module-level dict that every model mutates: importing
    discovery.models.mpta or building a model through it installs that model's boxes for
    the rest of the process. Tests that assert the stock defaults -- or that a parameter
    has NO prior, which several do -- therefore depend on which other tests ran first,
    and on nothing else. Restoring the baseline after each test removes the coupling.
    """
    from discovery import prior

    yield
    if prior.priordict_standard != _priordict_baseline:
        prior.priordict_standard.clear()
        prior.priordict_standard.update(_priordict_baseline)
