"""Properties of ``subsample``, for any history before it.

1. What was calculated before subsampling leaves no trace. Take any
   sequence of calculations (formula variables, weights and inputs, at the
   dataset year and other years, including formulas that make branches) and
   of changes to the inputs (``set_input``, ``delete_arrays``), with
   auto-carry-over on or off, on a ``Simulation`` or ``Microsimulation``.
   The subsampled simulation stores exactly what one does that made only the
   changes to the inputs, holds only inputs, has no branch of the whole
   population, and every later result agrees.
2. Subsampling is a rebuild from the sampled households' inputs: the result
   stores what a simulation built straight from those households' rows of
   the dataset (with the subsample's weights) stores, and calculates the
   same. It keeps whole households, and the person total of each weight.
"""

import numpy as np
import pytest

from policyengine_core.country_template import Microsimulation, Simulation
from tests.fixtures.subsample_inputs import (
    DATASET_YEAR,
    FORMULA_VARIABLES,
    HOUSEHOLD_OF_PERSON,
    build,
    dataframe,
    growing_household_weight,
    person_ids,
    stored,
    via_branch,
)

hypothesis = pytest.importorskip("hypothesis")
st = hypothesis.strategies

VARIABLES = FORMULA_VARIABLES + [
    "base",
    "overridden",
    "household_weight",
    "via_branch",
]
YEARS = ["2021", DATASET_YEAR, "2023", "2024"]
CHANGED = ["base", "overridden", "doubled", "household_weight"]

CALCULATE = st.tuples(
    st.just("calculate"), st.sampled_from(VARIABLES), st.sampled_from(YEARS)
)
SCRATCH_BRANCH = st.tuples(st.just("scratch_branch"), st.sampled_from(YEARS))
DELETE = st.tuples(st.just("delete"), st.sampled_from(CHANGED), st.sampled_from(YEARS))
SET_INPUT = st.tuples(
    st.just("set_input"),
    st.sampled_from(CHANGED),
    st.sampled_from(YEARS),
    st.integers(0, 3),
)
ACTIONS = st.one_of(CALCULATE, SCRATCH_BRANCH, DELETE, SET_INPUT)
CHANGES_INPUTS = {"delete", "set_input"}


def apply(simulation, action):
    kind, *arguments = action
    if kind == "calculate":
        variable, period = arguments
        simulation.calculate(variable, period)
    elif kind == "scratch_branch":
        (period,) = arguments
        branch = simulation.get_branch("scratch")
        branch.set_input("base", period, np.full(simulation.persons.count, 5.0))
        branch.calculate("household_total", period)
    elif kind == "delete":
        variable, period = arguments
        simulation.delete_arrays(variable, period)
    else:
        variable, period, seed = arguments
        population = simulation.get_variable_population(variable)
        # Positive, so given weights can be sampled by.
        values = 1.0 + (np.arange(population.count) * (seed + 1)) % 7
        simulation.set_input(variable, period, values)


def subsampled(options, actions, n, seed, time_period):
    """``(simulation, error)``: a simulation built with ``options`` after
    ``actions``, subsampled, or the error ``subsample`` raised."""
    simulation = build(**options)
    for action in actions:
        apply(simulation, action)
    try:
        simulation.subsample(n=n, seed=seed, time_period=time_period)
    except ValueError as error:
        return simulation, str(error)
    return simulation, None


@hypothesis.settings(max_examples=150, deadline=None)
@hypothesis.given(
    carry_over=st.booleans(),
    simulation_class=st.sampled_from([Simulation, Microsimulation]),
    growing_weights=st.booleans(),
    actions=st.lists(ACTIONS, min_size=1, max_size=8),
    later=st.lists(
        st.tuples(st.sampled_from(VARIABLES), st.sampled_from(YEARS)),
        min_size=1,
        max_size=4,
    ),
    n=st.integers(1, 8),
    seed=st.sampled_from(["a", "b", "c"]),
    time_period=st.sampled_from(YEARS),
)
def test_calculations_before_subsample_leave_no_trace(
    carry_over, simulation_class, growing_weights, actions, later, n, seed, time_period
):
    variables = [via_branch] + ([growing_household_weight] if growing_weights else [])
    options = dict(
        carry_over=carry_over, variables=variables, simulation_class=simulation_class
    )
    input_changes = [action for action in actions if action[0] in CHANGES_INPUTS]
    fresh, fresh_error = subsampled(options, input_changes, n, seed, time_period)
    used, used_error = subsampled(options, actions, n, seed, time_period)
    assert used_error == fresh_error
    hypothesis.assume(fresh_error is None)

    assert stored(used) == stored(fresh)
    for simulation in (fresh, used):
        for population in simulation.populations.values():
            for holder in population._holders.values():
                for branch, period in holder.get_known_branch_periods():
                    assert not holder.is_derived(period, branch)
        assert set(simulation.branches) <= {"baseline"}
    for variable, period in later:
        assert (
            np.asarray(used.calculate(variable, period)).tolist()
            == np.asarray(fresh.calculate(variable, period)).tolist()
        )
    for name in ("probe", "scratch"):
        assert used.get_branch(name).persons.count == used.persons.count
    assert stored(used) == stored(fresh)


@hypothesis.settings(max_examples=60, deadline=None)
@hypothesis.given(
    n=st.integers(1, 10),
    seed=st.sampled_from(["a", "b", "c", "d"]),
    quantize_weights=st.booleans(),
    prior=st.lists(
        st.tuples(st.sampled_from(VARIABLES), st.sampled_from(YEARS)), max_size=4
    ),
)
def test_subsample_is_a_rebuild_from_the_sampled_rows(n, seed, quantize_weights, prior):
    weight = f"household_weight__{DATASET_YEAR}"
    simulation = build(variables=[via_branch])
    for variable, period in prior:
        simulation.calculate(variable, period)
    simulation.subsample(
        n=n, seed=seed, time_period=DATASET_YEAR, quantize_weights=quantize_weights
    )

    # Whole households: each kept person's household keeps every person.
    kept = person_ids(simulation)
    households = {HOUSEHOLD_OF_PERSON[person - 1] for person in kept}
    assert kept == [
        index + 1
        for index, household in enumerate(HOUSEHOLD_OF_PERSON)
        if household in households
    ]
    assert simulation.populations["household"].count == len(households)
    # The person total of the weight is kept.
    assert float(
        np.sum(simulation.calculate("household_weight", DATASET_YEAR, map_to="person"))
    ) == pytest.approx(float(dataframe()[weight].sum()), rel=1e-5)

    # The same as building from the sampled households' rows of the dataset,
    # with the weights the subsample gave them.
    rows = dataframe()
    rows = rows[rows[f"person_id__{DATASET_YEAR}"].isin(kept)].reset_index(drop=True)
    rows[weight] = np.asarray(
        simulation.calculate("household_weight", DATASET_YEAR, map_to="person")
    )
    reference = build(variables=[via_branch], data=rows)
    assert stored(simulation) == stored(reference)
    for variable in VARIABLES:
        for period in YEARS:
            assert (
                np.asarray(simulation.calculate(variable, period)).tolist()
                == np.asarray(reference.calculate(variable, period)).tolist()
            )
