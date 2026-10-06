"""``subsample`` samples households by the weights the simulation was given.

It rebuilds the simulation from its inputs, and quantizes the input weights.
A weight the simulation calculated (by a formula, carried over or uprated)
is calculated again after subsampling, from the quantized input weights, so
households are sampled by the input weights too: sampling by a calculated
weight would apply its formula twice. Which weights the simulation happened
to calculate before subsampling, or while exporting its inputs, no longer
changes the sample.
"""

import numpy as np
import pytest

from policyengine_core.country_template import Microsimulation
from policyengine_core.country_template.entities import Household
from policyengine_core.periods import YEAR
from policyengine_core.variables import Variable
from tests.fixtures.subsample_inputs import (
    DATASET_YEAR,
    HOUSEHOLD_OF_PERSON,
    build,
    dataframe,
    growing_household_weight,
    person_ids,
    stored,
)

SEED = "weights"
LATER_YEAR = "2024"


class flat_household_weight(Variable):
    value_type = float
    entity = Household
    definition_period = YEAR
    label = "Household weight: 100 for every household after the dataset year"

    def formula(household, period):
        if period.start.year <= int(DATASET_YEAR):
            return household.filled_array(0.0)
        return household.filled_array(100.0)


flat_household_weight.__name__ = "household_weight"


def household_ids(simulation) -> set:
    return set(simulation.calculate("household_id", DATASET_YEAR).tolist())


def test_weights_calculated_for_the_sampling_year_leave_the_sample_unchanged():
    def subsampled(prior, time_period=LATER_YEAR):
        simulation = build(variables=[growing_household_weight])
        for period in prior:
            simulation.calculate("household_weight", period)
        simulation.subsample(n=6, seed=SEED, time_period=time_period)
        return simulation

    fresh = subsampled(prior=[])
    used = subsampled(prior=[LATER_YEAR])
    assert stored(used) == stored(fresh)
    # No weights were given for the later year, so the given ones are used.
    assert stored(fresh) == stored(subsampled(prior=[], time_period=DATASET_YEAR))


def test_microsimulation_samples_by_given_weights_not_ones_its_export_calculates():
    # Exporting ``base`` for 2023 makes a Microsimulation calculate 2023
    # weights; ``household_weight`` comes after ``base``, so its export used
    # to include them.
    data = dataframe()
    data["base__2023"] = data[f"base__{DATASET_YEAR}"] + 1

    def subsampled(time_period):
        simulation = build(
            variables=[growing_household_weight],
            data=data,
            simulation_class=Microsimulation,
        )
        simulation.subsample(n=6, seed=SEED, time_period=time_period)
        return simulation

    assert stored(subsampled("2023")) == stored(subsampled(DATASET_YEAR))


def test_household_given_no_weight_is_never_sampled():
    data = dataframe()
    household_six = data[f"household_id__{DATASET_YEAR}"] == 6
    data.loc[household_six, f"household_weight__{DATASET_YEAR}"] = 0.0
    simulation = build(variables=[flat_household_weight], data=data)
    calculated = simulation.calculate("household_weight", LATER_YEAR)
    assert calculated.tolist() == [100.0] * 6

    simulation.subsample(n=60, seed=SEED, time_period=LATER_YEAR)

    assert 6 not in household_ids(simulation)


def given_weights(weights_2023=None, data_weights=None):
    """A simulation given household weights for 2022 (``data_weights``, by
    household) and, with ``weights_2023``, for 2023 with ``set_input``."""
    data = dataframe()
    if data_weights is not None:
        data[f"household_weight__{DATASET_YEAR}"] = [
            data_weights[household] for household in HOUSEHOLD_OF_PERSON
        ]
    simulation = build(data=data)
    if weights_2023 is not None:
        simulation.set_input("household_weight", "2023", np.array(weights_2023))
    return simulation


ONLY_FIRST = {1: 100.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0}
ONLY_LAST = [0.0, 0.0, 0.0, 0.0, 0.0, 600.0]


@pytest.mark.parametrize(
    "time_period, sampled",
    [
        (DATASET_YEAR, {1}),
        ("2023", {6}),
        # Given for no later year: the latest given before it, which is what
        # the simulation carries over to it.
        (LATER_YEAR, {6}),
        # Given for no earlier year: the earliest given.
        ("2021", {1}),
    ],
)
def test_time_period_picks_the_household_weights_given_for_it(time_period, sampled):
    def subsampled(prior):
        simulation = given_weights(weights_2023=ONLY_LAST, data_weights=ONLY_FIRST)
        for variable, period in prior:
            simulation.calculate(variable, period)
        simulation.subsample(n=3, seed=SEED, time_period=time_period)
        return simulation

    fresh = subsampled(prior=[])
    used = subsampled(
        prior=[("household_weight", year) for year in ("2021", LATER_YEAR, "2025")]
    )
    assert household_ids(fresh) == sampled
    assert stored(used) == stored(fresh)


def test_subsample_without_household_ids_in_the_dataset():
    data = dataframe().drop(columns=[f"household_id__{DATASET_YEAR}"])

    def subsampled(prior):
        simulation = build(data=data)
        for variable in prior:
            simulation.calculate(variable, DATASET_YEAR)
        simulation.subsample(n=4, seed=SEED, time_period=DATASET_YEAR)
        return simulation

    fresh = subsampled(prior=[])
    used = subsampled(prior=["household_id"])
    assert stored(used) == stored(fresh)
    # Whole households are kept, each person with the others of theirs.
    kept = person_ids(fresh)
    households = {HOUSEHOLD_OF_PERSON[person - 1] for person in kept}
    assert sorted(kept) == [
        index + 1
        for index, household in enumerate(HOUSEHOLD_OF_PERSON)
        if household in households
    ]
    assert fresh.populations["household"].count == len(households)
    assert 1 < len(households) < 6


def test_subsample_needs_household_weights_the_simulation_was_given():
    data = dataframe().drop(columns=[f"household_weight__{DATASET_YEAR}"])
    simulation = build(data=data)
    simulation.calculate("household_weight", DATASET_YEAR)

    with pytest.raises(ValueError, match="household_weight"):
        simulation.subsample(n=2, seed=SEED, time_period=DATASET_YEAR)
