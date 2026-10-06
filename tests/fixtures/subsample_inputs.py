"""A small dataset-backed simulation for testing what ``subsample`` keeps.

Shared by ``tests/core/test_subsample_inputs_only.py`` and its Hypothesis
property module, which pytest must be able to skip on its own when
Hypothesis is not installed.
"""

import numpy as np
import pandas as pd

from policyengine_core.country_template import CountryTaxBenefitSystem, Simulation
from policyengine_core.country_template.entities import Household, Person
from policyengine_core.data import Dataset
from policyengine_core.periods import YEAR
from policyengine_core.variables import Variable

DATASET_YEAR = "2022"
HOUSEHOLD_OF_PERSON = [1, 1, 2, 2, 2, 3, 4, 4, 5, 6]
BASE = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
WEIGHTS = {1: 100.0, 2: 200.0, 3: 300.0, 4: 400.0, 5: 500.0, 6: 600.0}
BRANCH_BASE = 1.0


class base(Variable):
    value_type = float
    entity = Person
    definition_period = YEAR
    label = "Input amount"


class doubled(Variable):
    value_type = float
    entity = Person
    definition_period = YEAR
    label = "Twice the input amount"

    def formula(person, period):
        return 2 * person("base", period)


class ended(Variable):
    value_type = float
    entity = Person
    definition_period = YEAR
    label = "Seven, until its formula ends with the dataset year"
    end = f"{DATASET_YEAR}-12-31"

    def formula(person, period):
        return person.filled_array(7.0)


class household_total(Variable):
    value_type = float
    entity = Household
    definition_period = YEAR
    label = "Household sum of the doubled amount"

    def formula(household, period):
        return household.sum(household.members("doubled", period))


class overridden(Variable):
    value_type = float
    entity = Person
    definition_period = YEAR
    label = "A formula variable the dataset gives values for"

    def formula(person, period):
        return person.filled_array(-1.0)


class via_branch(Variable):
    value_type = float
    entity = Person
    definition_period = YEAR
    label = "Doubled on a branch with its own input, as policyengine-us's itemizing branches are"

    def formula(person, period):
        branch = person.simulation.get_branch("probe")
        branch.set_input("base", period, np.full(person.count, BRANCH_BASE))
        return branch.calculate("doubled", period)


class growing_household_weight(Variable):
    value_type = float
    entity = Household
    definition_period = YEAR
    label = "Household weight, growing unevenly after the dataset year"

    def formula(household, period):
        if period.start.year <= int(DATASET_YEAR):
            return household.filled_array(0.0)
        odd = household("household_id", period) % 2 == 1
        growth = np.where(odd, 10.0, 0.1)
        return household("household_weight", period.last_year) * growth + 50


growing_household_weight.__name__ = "household_weight"


FORMULA_VARIABLES = ["doubled", "ended", "household_total", "person_weight"]
OVERRIDDEN = [float(index) for index in range(len(BASE))]


def dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            f"person_id__{DATASET_YEAR}": list(range(1, len(BASE) + 1)),
            f"household_id__{DATASET_YEAR}": HOUSEHOLD_OF_PERSON,
            f"person_household_id__{DATASET_YEAR}": HOUSEHOLD_OF_PERSON,
            f"household_weight__{DATASET_YEAR}": [
                WEIGHTS[household] for household in HOUSEHOLD_OF_PERSON
            ],
            f"base__{DATASET_YEAR}": BASE,
            f"overridden__{DATASET_YEAR}": OVERRIDDEN,
        }
    )


def system(carry_over: bool = True, variables=()) -> CountryTaxBenefitSystem:
    """The template system with this module's variables, and ``variables``
    added (one the system has is replaced, and moves last in its order)."""
    tax_benefit_system = CountryTaxBenefitSystem()
    tax_benefit_system.auto_carry_over_input_variables = carry_over
    tax_benefit_system.add_variables(base, doubled, ended, household_total, overridden)
    for variable in variables:
        tax_benefit_system.replace_variable(variable)
    return tax_benefit_system


def build(
    carry_over: bool = True,
    variables=(),
    data: pd.DataFrame = None,
    simulation_class=Simulation,
):
    """A ten-person, six-household simulation loaded from ``dataframe()``
    (or ``data``)."""
    return simulation_class(
        tax_benefit_system=system(carry_over, variables),
        dataset=Dataset.from_dataframe(
            dataframe() if data is None else data, DATASET_YEAR
        ),
    )


def stored(simulation) -> dict:
    """Every stored value: ``{(variable, branch, period): values}``."""
    values = {}
    for population in simulation.populations.values():
        for name, holder in population._holders.items():
            for branch, period in holder.get_known_branch_periods():
                array = holder.get_array(period, branch)
                values[(name, branch, str(period))] = np.asarray(array).tolist()
    return values


def person_ids(simulation) -> list:
    """The IDs of the persons in ``simulation``, in order."""
    return np.asarray(simulation.calculate("person_id", DATASET_YEAR)).tolist()
