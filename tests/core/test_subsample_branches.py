"""``subsample`` drops the branches made before it, which hold the whole population.

A branch is a copy of the simulation as it was when the branch was made.
``get_branch`` returns the branch already made under a name, so a formula
that branches (as policyengine-us's itemizing branches do) found, after
subsampling, a branch of the full population, and failed to set its input
on it.
"""

import numpy as np

from policyengine_core.country_template import Microsimulation
from tests.fixtures.subsample_inputs import (
    BRANCH_BASE,
    DATASET_YEAR,
    build,
    via_branch,
)

SEED = "branches"


def subsampled(prior=()):
    """A simulation with ``via_branch``, subsampled after calculating it for
    each period in ``prior``."""
    simulation = build(variables=[via_branch])
    for period in prior:
        simulation.calculate("via_branch", period)
    simulation.subsample(n=4, seed=SEED, time_period=DATASET_YEAR)
    return simulation


def test_formula_that_branches_runs_on_the_subsample():
    fresh = subsampled()
    used = subsampled(prior=[DATASET_YEAR, "2023"])
    for period in (DATASET_YEAR, "2023"):
        expected = [2 * BRANCH_BASE] * used.persons.count
        assert used.calculate("via_branch", period).tolist() == expected
        assert fresh.calculate("via_branch", period).tolist() == expected


def test_branches_made_before_subsample_are_dropped():
    simulation = build()
    branch = simulation.get_branch("scratch")
    branch.set_input("base", DATASET_YEAR, np.full(simulation.persons.count, 1000.0))
    branch.calculate("doubled", DATASET_YEAR)

    simulation.subsample(n=4, seed=SEED, time_period=DATASET_YEAR)

    assert simulation.branches == {}
    again = simulation.get_branch("scratch")
    assert again is not branch
    assert again.persons.count == simulation.persons.count
    # Made from the subsample: its inputs, not the dropped branch's.
    assert (
        again.calculate("doubled", DATASET_YEAR).tolist()
        == (2 * simulation.calculate("base", DATASET_YEAR)).tolist()
    )


def test_only_the_baseline_branch_is_made_again():
    simulation = Microsimulation(reform={"taxes.income_tax_rate": {"2022-01-01": 0.42}})
    simulation.get_branch("scratch")
    simulation.baseline.get_branch("nested")

    simulation.subsample(n=3, seed=SEED, time_period=DATASET_YEAR)

    assert set(simulation.branches) == {"baseline"}
    assert simulation.baseline is simulation.branches["baseline"]
    assert simulation.baseline.branches == {}
    assert simulation.baseline.persons.count == simulation.persons.count
