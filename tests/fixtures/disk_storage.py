"""Helpers for tests of disk-backed holder storage."""

import contextlib
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import List

from policyengine_core.country_template import CountryTaxBenefitSystem
from policyengine_core.experimental import MemoryConfig
from policyengine_core.simulations import Simulation, SimulationBuilder

REPO_ROOT = Path(__file__).resolve().parents[2]


def build_simulation(on_disk: bool = True, count: int = 1) -> Simulation:
    # A tax-benefit system keeps the last simulation built on it, so give
    # each simulation its own for the tests that release one.
    simulation = SimulationBuilder().build_default_simulation(
        CountryTaxBenefitSystem(), count=count
    )
    if on_disk:
        simulation.memory_config = MemoryConfig(max_memory_occupation=0)
    # Holders made before ``memory_config`` is set have no disk storage, so
    # start without any: each holder is created where it is first used.
    make_holders_again(simulation)
    return simulation


def make_holders_again(simulation: Simulation) -> None:
    """Drop every holder of ``simulation``, so each is made again where it is
    next used, as rebuilding the populations does."""
    for population in simulation.populations.values():
        population._holders = {}


def file_count(folder: str) -> int:
    """How many files are in ``folder`` and the folders in it."""
    return sum(len(files) for _, _, files in os.walk(folder))


def run_in_new_interpreter(script: str, tmp_path, **environment) -> str:
    """Run ``script`` in a new interpreter that makes temporary folders in
    ``tmp_path``, and return what it printed. Tests that fork do it there:
    forking pytest's process, which may run threads, can deadlock."""
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        cwd=REPO_ROOT,
        env={**os.environ, "TMPDIR": str(tmp_path), **environment},
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


@contextlib.contextmanager
def unraisable_exceptions():
    """Collect exceptions Python can only print, such as those in finalizers."""
    errors: List[str] = []
    previous_hook = sys.unraisablehook
    sys.unraisablehook = lambda unraisable: errors.append(repr(unraisable.exc_value))
    try:
        yield errors
    finally:
        sys.unraisablehook = previous_hook
