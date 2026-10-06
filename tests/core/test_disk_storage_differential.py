"""Storing holder values on disk gives the results storing them in memory gives.

A Hypothesis property over random sequences of branching, cloning, inputs,
calculations, derivatives, reforms, holders made again and drops, run on a
disk-backed and a memory-backed simulation family side by side.
"""

from __future__ import annotations

import gc
import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pytest

# The country-package smoke job installs no dev dependencies.
hypothesis = pytest.importorskip("hypothesis")
st = hypothesis.strategies

from policyengine_core.data_storage.on_disk_storage import REPLACEMENTS_DIR
from policyengine_core.simulations import Simulation
from tests.fixtures.disk_storage import (
    build_simulation,
    make_holders_again,
    unraisable_exceptions,
)

MONTHS = ["2017-01", "2017-02"]
NAMES = ["a", "b", "c"]
INPUTS = ["salary", "rent", "housing_occupancy_status", "birth"]
CALCULATED = [
    "salary",
    "income_tax",
    "disposable_income",
    "basic_income",
    "housing_allowance",
    "housing_occupancy_status",
    "total_taxes",
    "household_income",
]
HOUSING_STATUSES = ["owner", "tenant", "free_lodger", "homeless"]
COUNT = 2


def _input_value(variable: str, seed: int) -> np.ndarray:
    values = [seed, seed // 7 + 1]
    if variable == "housing_occupancy_status":
        return np.array([HOUSING_STATUSES[value % 4] for value in values])
    if variable == "birth":
        return np.array(
            [f"{1950 + value % 60}-01-01" for value in values], dtype="datetime64[D]"
        )
    return np.array(values, dtype=float)


@dataclass
class _Member:
    simulation: Simulation
    parent: Optional["_Member"]
    branch_name: Optional[str]


class _Family:
    """A simulation and every branch or clone made from it, still in use."""

    def __init__(self, on_disk: bool):
        self.members = [_Member(build_simulation(on_disk, count=COUNT), None, None)]

    def pick(self, index: int) -> _Member:
        return self.members[index % len(self.members)]

    def apply(self, operation: tuple):
        kind, index, *arguments = operation
        if kind == "drop":
            # Before taking any reference this frame would hold through the
            # collection.
            self.drop(index)
            return None
        member = self.pick(index)
        simulation = member.simulation
        if kind == "branch":
            (name,) = arguments
            branch = simulation.get_branch(name)
            if not any(other.simulation is branch for other in self.members):
                self.members.append(_Member(branch, member, name))
        elif kind == "clone":
            self.members.append(_Member(simulation.clone(), member, None))
        elif kind == "set":
            variable, period, seed = arguments
            simulation.set_input(variable, period, _input_value(variable, seed))
        elif kind == "calculate":
            variable, period = arguments
            return _outcome(lambda: simulation.calculate(variable, period))
        elif kind == "derivative":
            (period,) = arguments
            return _outcome(
                lambda: simulation.derivative("income_tax", "salary", period, delta=100)
            )
        elif kind == "reform":
            (rate,) = arguments
            simulation.apply_reform({"taxes.income_tax_rate": rate})
        elif kind == "delete_arrays":
            (variable,) = arguments
            simulation.delete_arrays(variable)
        elif kind == "new_holders":
            make_holders_again(simulation)

    def drop(self, index: int) -> None:
        """Drop a branch or clone, and every branch or clone made from it,
        and collect them."""
        member = self.pick(index)
        if member.parent is None:
            return
        dropped = [member]
        for other in self.members:
            ancestor = other.parent
            while ancestor is not None and ancestor not in dropped:
                ancestor = ancestor.parent
            if ancestor is not None:
                dropped.append(other)
        self.members = [other for other in self.members if other not in dropped]
        if member.branch_name is not None:
            branches = member.parent.simulation.branches
            if branches.get(member.branch_name) is member.simulation:
                del branches[member.branch_name]
        # Nothing in this frame may keep a dropped member alive.
        del dropped, member, other, ancestor
        gc.collect()

    def disk_storages(self) -> list:
        return [
            holder._disk_storage
            for member in self.members
            for population in member.simulation.populations.values()
            for holder in population._holders.values()
            if holder._disk_storage is not None
        ]

    def missing_files(self) -> List[str]:
        return [
            path
            for storage in self.disk_storages()
            for path in storage._files.values()
            if not os.path.isfile(path)
        ]

    def unread_replacements(self) -> List[str]:
        """The files storages wrote instead of writing over shared ones that
        no storage alive reads or may write again, once each family of
        storages has removed those it may (as it does before writing a new
        one)."""
        # A storage whose holder is gone may still read files, kept alive by
        # a clone of it: count every storage of each family.
        storages = {
            id(member): member
            for storage in self.disk_storages()
            for member in storage._family_storages
        }.values()
        for storage in storages:
            storage._remove_unread_replacements()
        read = {
            path
            for storage in storages
            for paths in (storage._files, storage._own_paths)
            for path in paths.values()
        }
        folders = {
            os.path.join(storage.storage_dir, REPLACEMENTS_DIR) for storage in storages
        }
        return sorted(
            path
            for folder in folders
            if os.path.isdir(folder)
            for path in (os.path.join(folder, name) for name in os.listdir(folder))
            if path not in read
        )


def _outcome(calculate):
    try:
        return np.asarray(calculate()).tolist()
    except Exception as error:  # Both storages must fail the same way.
        return type(error).__name__


_index = st.integers(0, 15)
_operations = st.lists(
    st.one_of(
        st.tuples(st.just("branch"), _index, st.sampled_from(NAMES)),
        st.tuples(st.just("clone"), _index),
        st.tuples(
            st.just("set"),
            _index,
            st.sampled_from(INPUTS),
            st.sampled_from(MONTHS),
            st.integers(0, 5_000),
        ),
        st.tuples(
            st.just("calculate"),
            _index,
            st.sampled_from(CALCULATED),
            st.sampled_from(MONTHS),
        ),
        st.tuples(st.just("derivative"), _index, st.sampled_from(MONTHS)),
        st.tuples(st.just("reform"), _index, st.sampled_from([0.1, 0.2, 0.3])),
        st.tuples(
            st.just("delete_arrays"), _index, st.sampled_from(INPUTS + CALCULATED)
        ),
        st.tuples(st.just("new_holders"), _index),
        st.tuples(st.just("drop"), _index),
    ),
    max_size=30,
)


@hypothesis.settings(
    max_examples=75,
    deadline=None,
    suppress_health_check=[hypothesis.HealthCheck.too_slow],
)
@hypothesis.given(operations=_operations)
# Counterexamples on master 2e22c8f8. A holder made again for a variable made
# its disk storage in the folder its earlier one used: it wrote the default it
# calculated over the file a clone read, and the earlier storage, collected
# with the clone, removed the folder and the new holder's files with it.
@hypothesis.example(
    operations=[("set", 0, "salary", "2017-01", 0), ("clone", 0), ("new_holders", 0)]
)
@hypothesis.example(
    operations=[
        ("set", 0, "salary", "2017-01", 0),
        ("clone", 0),
        ("new_holders", 0),
        ("set", 0, "salary", "2017-01", 5),
        ("drop", 1),
    ]
)
# The files ``derivative``'s clone wrote instead of writing over shared ones
# stayed on disk until the simulation was collected.
@hypothesis.example(
    operations=[
        ("set", 0, "salary", "2017-01", 1),
        ("calculate", 0, "income_tax", "2017-01"),
        ("derivative", 0, "2017-01"),
        ("reform", 0, 0.1),
        ("calculate", 0, "income_tax", "2017-01"),
    ]
)
def test_disk_storage_gives_the_results_memory_storage_gives(operations):
    """For any sequence of branching, cloning, inputs, calculations,
    derivatives, reforms, holders made again and drops:

    - every calculation returns what it returns with values held in memory;
    - every file a live storage maps is on disk;
    - once the dropped simulations are collected, no file is left that a
      storage wrote instead of writing over a shared one and that no live
      storage reads, once each family has removed those it may;
    - once the simulations are gone, their directory is removed, and no
      finalizer raised.
    """
    with unraisable_exceptions() as errors:
        in_memory = _Family(on_disk=False)
        on_disk = _Family(on_disk=True)
        for operation in operations:
            expected = in_memory.apply(operation)
            assert on_disk.apply(operation) == expected, operation
            assert on_disk.missing_files() == [], operation

        for member_in_memory, member_on_disk in zip(in_memory.members, on_disk.members):
            for variable in CALCULATED:
                for period in MONTHS:
                    assert _outcome(
                        lambda: member_on_disk.simulation.calculate(variable, period)
                    ) == _outcome(
                        lambda: member_in_memory.simulation.calculate(variable, period)
                    ), (variable, period)

        gc.collect()
        assert on_disk.unread_replacements() == []

        directory = on_disk.members[0].simulation._data_storage_dir
        del in_memory, on_disk, member_in_memory, member_on_disk
        gc.collect()

    assert errors == []
    assert directory is None or not os.path.exists(directory)
