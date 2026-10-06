"""Disk-backed holder storage keeps the values each simulation reads.

``test_data_storage_dir.py`` and ``test_on_disk_storage_clones.py`` cover the
folder a simulation stores values in, and the files clones, copies and forked
processes share. These tests cover what those leave open:

* a disk storage a holder makes takes a folder no other disk storage uses,
  so a holder made again for a variable, or another simulation given the
  same folder, neither writes over nor removes files an earlier storage
  reads;
* a folder the caller chose (``_data_storage_dir``) is made if it does not
  exist yet;
* a file a storage wrote instead of writing over a shared one is removed once
  no storage reads it, so making and dropping branches, or taking
  derivatives, again and again leaves a steady number of files;
* a plain value stored over an enum reads back plain;

and a few ways simulations share values that pass through those.
``test_disk_storage_differential.py`` checks the first and third as
properties over random sequences of operations.
"""

from __future__ import annotations

import copy
import gc
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pytest

from policyengine_core import periods
from policyengine_core.country_template import CountryTaxBenefitSystem, entities
from policyengine_core.data_storage import OnDiskStorage
from policyengine_core.data_storage.on_disk_storage import REPLACEMENTS_DIR
from policyengine_core.enums import EnumArray
from policyengine_core.experimental import MemoryConfig
from policyengine_core.simulations import Simulation, SimulationBuilder
from policyengine_core.variables import Variable
from tests.fixtures.disk_storage import (
    build_simulation,
    file_count,
    make_holders_again,
    run_in_new_interpreter,
)

PERIOD = "2017-01"


def _drop_branch(simulation: Simulation, name: str) -> None:
    del simulation.branches[name]
    gc.collect()


def _replacements(storage: OnDiskStorage) -> list:
    folder = os.path.join(storage.storage_dir, REPLACEMENTS_DIR)
    return sorted(os.listdir(folder)) if os.path.isdir(folder) else []


# ----- A folder of its own for each disk storage a holder makes ----- #


def test_a_clone_keeps_its_values_when_its_source_makes_its_holders_again():
    """From the differential property's shrunk counterexample: the source's
    new holder wrote the default it calculated over the file the clone
    reads."""
    simulation = build_simulation(count=2)
    simulation.set_input("salary", PERIOD, np.array([0.0, 1.0]))
    clone = simulation.clone()

    make_holders_again(simulation)
    gc.collect()
    # An input with no formula: calculating it stores its default.
    np.testing.assert_array_equal(simulation.calculate("salary", PERIOD), [0, 0])

    np.testing.assert_array_equal(clone.calculate("salary", PERIOD), [0, 1])


def test_a_holder_made_again_keeps_its_values_when_the_earlier_one_goes():
    """The earlier holder's disk storage removed its folder when collected,
    with the files the new one had written there."""
    simulation = build_simulation()
    simulation.set_input("salary", PERIOD, np.array([1.0]))
    earlier = simulation.get_holder("salary")
    earlier_dir = earlier._disk_storage.storage_dir

    make_holders_again(simulation)
    simulation.set_input("salary", PERIOD, np.array([7.0]))
    new_dir = simulation.get_holder("salary")._disk_storage.storage_dir
    del earlier
    gc.collect()

    # The first disk storage of a variable keeps the folder named for it;
    # the next takes a new one named after it.
    assert earlier_dir == os.path.join(simulation.data_storage_dir, "salary")
    assert os.path.dirname(new_dir) == simulation.data_storage_dir
    assert os.path.basename(new_dir).startswith("salary_")
    assert not os.path.exists(earlier_dir)
    np.testing.assert_array_equal(
        simulation.get_holder("salary").get_array(PERIOD), [7.0]
    )


def test_simulations_given_one_folder_keep_their_own_values(tmp_path):
    first, second = build_simulation(), build_simulation()
    first._data_storage_dir = second._data_storage_dir = str(tmp_path)

    first.set_input("salary", PERIOD, np.array([1.0]))
    second.set_input("salary", PERIOD, np.array([2.0]))

    np.testing.assert_array_equal(first.get_holder("salary").get_array(PERIOD), [1])
    np.testing.assert_array_equal(second.get_holder("salary").get_array(PERIOD), [2])
    del second
    gc.collect()
    np.testing.assert_array_equal(first.get_holder("salary").get_array(PERIOD), [1])
    del first
    gc.collect()
    # Each removed the folder its storage made; the folder itself stays.
    assert list(tmp_path.iterdir()) == []


# ----- A folder the caller chose that does not exist yet ----- #


def test_a_folder_the_caller_chose_is_made_if_it_does_not_exist(tmp_path):
    folder = tmp_path / "simulation"
    simulation = build_simulation()
    simulation._data_storage_dir = str(folder)

    simulation.set_input("salary", PERIOD, np.array([1.0]))

    storage_dir = simulation.get_holder("salary")._disk_storage.storage_dir
    assert os.path.dirname(storage_dir) == str(folder)
    np.testing.assert_array_equal(simulation.calculate("salary", PERIOD), [1.0])
    del simulation
    gc.collect()
    assert folder.is_dir()
    assert list(folder.iterdir()) == []


@pytest.mark.parametrize("in_a_simulations_folder", [False, True])
def test_a_clone_stores_in_a_folder_the_caller_chose_that_does_not_exist(
    tmp_path, in_a_simulations_folder
):
    """The clone stores first, so it makes the folder its own goes in, also
    when that is in a temporary folder another simulation made."""
    if in_a_simulations_folder:
        owner = build_simulation()
        folder = Path(owner.data_storage_dir) / "simulation"
    else:
        folder = tmp_path / "simulation"
    simulation = build_simulation()
    simulation._data_storage_dir = str(folder)
    clone = simulation.clone()

    clone.set_input("salary", PERIOD, np.array([2.0]))

    assert Path(clone.data_storage_dir).parent == folder
    np.testing.assert_array_equal(clone.calculate("salary", PERIOD), [2.0])


def test_the_folders_leading_to_a_folder_the_caller_chose_are_not_made(tmp_path):
    """A folder on a disk that is not mounted, say, is not made elsewhere."""
    simulation = build_simulation()
    simulation._data_storage_dir = str(tmp_path / "missing" / "simulation")

    with pytest.raises(FileNotFoundError):
        simulation.set_input("salary", PERIOD, np.array([1.0]))

    assert not (tmp_path / "missing").exists()


_FORK_WITH_A_CHOSEN_FOLDER_NOT_MADE_YET = """
    import gc, json, os, tempfile, traceback
    import numpy as np
    from tests.fixtures.disk_storage import build_simulation

    chosen = os.path.join(tempfile.mkdtemp(), "chosen")
    simulation = build_simulation()
    simulation._data_storage_dir = chosen
    report_r, report_w = os.pipe()
    pid = os.fork()
    if pid == 0:
        status = 1
        try:
            simulation.set_input("salary", "2017-01", np.array([1.0]))
            storage_dir = simulation.get_holder("salary")._disk_storage.storage_dir
            report = {
                "in the chosen folder": os.path.dirname(os.path.dirname(storage_dir))
                == chosen,
                "value": simulation.calculate("salary", "2017-01").tolist(),
            }
            del simulation
            gc.collect()
            os.write(report_w, json.dumps(report).encode())
            status = 0
        except BaseException:
            traceback.print_exc()
        finally:
            os._exit(status)
    os.close(report_w)
    child = json.loads(os.read(report_r, 256).decode() or "null")
    _, status = os.waitpid(pid, 0)
    print(json.dumps({"status": os.waitstatus_to_exitcode(status), "child": child}))
"""


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
def test_a_forked_process_makes_a_folder_the_caller_chose_that_does_not_exist(
    tmp_path,
):
    """It stores in a folder of its own inside it, as when the folder exists,
    rather than in the default temporary folder."""
    printed = run_in_new_interpreter(
        _FORK_WITH_A_CHOSEN_FOLDER_NOT_MADE_YET, tmp_path, NUMEXPR_MAX_THREADS="1"
    )

    assert json.loads(printed.splitlines()[-1]) == {
        "status": 0,
        "child": {"in the chosen folder": True, "value": [1.0]},
    }


_FORK_THEN_THE_PARENT_REMOVES_ITS_FOLDER = """
    import gc, json, os, traceback
    import numpy as np
    from tests.fixtures.disk_storage import build_simulation

    simulation = build_simulation()
    folder = simulation.data_storage_dir
    ready_r, ready_w = os.pipe()
    go_r, go_w = os.pipe()
    report_r, report_w = os.pipe()
    pid = os.fork()
    if pid == 0:
        status = 1
        try:
            # The child's own folder goes in the one it inherited; a clone's
            # would too.
            simulation.data_storage_dir
            gc.collect()
            clone = simulation.clone()
            os.write(ready_w, b"x")
            os.read(go_r, 1)
            clone.set_input("salary", "2017-01", np.array([1.0]))
            report = {
                "value": clone.calculate("salary", "2017-01").tolist(),
                "folder made again": os.path.exists(folder),
            }
            del clone
            gc.collect()
            os.write(report_w, json.dumps(report).encode())
            status = 0
        except BaseException:
            traceback.print_exc()
        finally:
            os._exit(status)
    os.close(ready_w)
    os.close(report_w)
    os.read(ready_r, 1)
    del simulation
    gc.collect()
    removed = not os.path.exists(folder)
    os.write(go_w, b"x")
    child = json.loads(os.read(report_r, 256).decode() or "null")
    _, status = os.waitpid(pid, 0)
    print(json.dumps(
        {"status": os.waitstatus_to_exitcode(status), "removed": removed, "child": child}
    ))
"""


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
def test_a_clone_in_a_forked_process_makes_no_folder_the_parent_removed(tmp_path):
    """The parent's folder, made by a simulation and since removed, is not
    made again, where nothing would remove it: the clone stores in the
    default temporary folder instead."""
    printed = run_in_new_interpreter(
        _FORK_THEN_THE_PARENT_REMOVES_ITS_FOLDER, tmp_path, NUMEXPR_MAX_THREADS="1"
    )

    assert json.loads(printed.splitlines()[-1]) == {
        "status": 0,
        "removed": True,
        "child": {"value": [1.0], "folder made again": False},
    }


# ----- Files nothing reads any more ----- #


def test_a_file_a_dropped_clone_wrote_is_removed_at_the_next_new_file(tmp_path):
    storage = OnDiskStorage(str(tmp_path / "salary"))
    os.mkdir(storage.storage_dir)
    storage.put(np.array([1.0]), PERIOD)
    clone = storage.clone()
    clone.put(np.array([2.0]), PERIOD)
    assert len(_replacements(storage)) == 1

    del clone
    gc.collect()
    # Shared with the clone, so a new file: the clone's goes first.
    storage.put(np.array([3.0]), PERIOD)

    (written,) = _replacements(storage)
    assert os.path.join(storage.storage_dir, REPLACEMENTS_DIR, written) in (
        storage._files.values()
    )
    np.testing.assert_array_equal(storage.get(PERIOD), [3.0])


def test_a_key_only_a_dropped_clone_stored_is_removed_at_the_next_new_file(
    tmp_path,
):
    """A clone writes a key no storage of its family stored to a new file
    too, so the family can remove it once the clone is collected."""
    storage = OnDiskStorage(str(tmp_path / "salary"))
    os.mkdir(storage.storage_dir)
    storage.put(np.array([1.0]), PERIOD)
    clone = storage.clone()
    clone.put(np.array([2.0]), "2017-02")
    assert len(_replacements(storage)) == 1

    del clone
    gc.collect()
    storage.put(np.array([3.0]), PERIOD)

    (written,) = _replacements(storage)
    assert written.startswith(f"default_{PERIOD}.")
    assert sorted(os.listdir(storage.storage_dir)) == [
        f"default_{PERIOD}.npy",
        REPLACEMENTS_DIR,
    ]


def test_a_file_a_clone_reads_stays_after_the_storage_that_wrote_it_drops_it(
    tmp_path,
):
    storage = OnDiskStorage(str(tmp_path / "salary"))
    os.mkdir(storage.storage_dir)
    storage.put(np.array([1.0]), PERIOD)
    storage.clone()  # Collected at once: the named file is shared all the same.
    storage.put(np.array([2.0]), PERIOD)  # A new file of its own...
    reader = storage.clone()  # ...which this clone reads.

    storage.delete(PERIOD)
    storage.put(np.array([4.0]), PERIOD)  # Another new file.

    assert len(_replacements(storage)) == 2
    np.testing.assert_array_equal(reader.get(PERIOD), [2.0])
    np.testing.assert_array_equal(storage.get(PERIOD), [4.0])


@pytest.mark.parametrize(
    "copier",
    [copy.copy, copy.deepcopy, lambda storage: pickle.loads(pickle.dumps(storage))],
    ids=["copy", "deepcopy", "pickle"],
)
def test_a_file_a_copy_reads_is_never_removed(tmp_path, copier):
    """A copy is no clone: the storages it was copied from cannot tell when
    it stops reading a file, so they never remove one it was copied with."""
    storage = OnDiskStorage(str(tmp_path / "salary"))
    os.mkdir(storage.storage_dir)
    storage.put(np.array([1.0]), PERIOD)
    clone = storage.clone()
    clone.put(np.array([2.0]), PERIOD)
    copied = copier(clone)

    del clone
    gc.collect()
    storage.put(np.array([3.0]), PERIOD)

    np.testing.assert_array_equal(copied.get(PERIOD), [2.0])
    np.testing.assert_array_equal(storage.get(PERIOD), [3.0])


_FORK_THEN_DROP_A_CLONE = """
    import gc, json, os, tempfile, traceback
    import numpy as np
    from policyengine_core.data_storage import OnDiskStorage

    storage = OnDiskStorage(os.path.join(tempfile.mkdtemp(), "salary"))
    os.mkdir(storage.storage_dir)
    storage.put(np.array([1.0]), "2017-01")
    clone = storage.clone()
    clone.put(np.array([2.0]), "2017-01")
    go_r, go_w = os.pipe()
    report_r, report_w = os.pipe()
    pid = os.fork()
    if pid == 0:
        status = 1
        try:
            os.read(go_r, 1)
            os.write(report_w, json.dumps(clone.get("2017-01").tolist()).encode())
            status = 0
        except BaseException:
            traceback.print_exc()
        finally:
            os._exit(status)
    os.close(report_w)
    # The parent drops the clone; the child still reads its copy's file.
    del clone
    gc.collect()
    storage.put(np.array([3.0]), "2017-01")
    os.write(go_w, b"x")
    child = os.read(report_r, 256).decode() or "nothing"
    _, status = os.waitpid(pid, 0)
    print(json.dumps({"status": os.waitstatus_to_exitcode(status), "child": child,
                      "parent": storage.get("2017-01").tolist()}))
"""


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
def test_a_file_a_forked_process_may_read_is_never_removed(tmp_path):
    printed = run_in_new_interpreter(
        _FORK_THEN_DROP_A_CLONE, tmp_path, NUMEXPR_MAX_THREADS="1"
    )
    result = json.loads(printed.splitlines()[-1])

    assert result == {"status": 0, "child": "[2.0]", "parent": [3.0]}


def test_repeated_derivatives_and_reforms_leave_a_steady_number_of_files():
    simulation = build_simulation()
    simulation.set_input("salary", PERIOD, np.array([3_000.0]))
    simulation.calculate("income_tax", PERIOD)
    counts = []

    for cycle in range(6):
        rate = (0.1, 0.2)[cycle % 2]
        simulation.derivative("income_tax", "salary", PERIOD, delta=100)
        simulation.apply_reform({"taxes.income_tax_rate": rate})
        np.testing.assert_allclose(
            simulation.calculate("income_tax", PERIOD), [3_000.0 * rate]
        )
        gc.collect()
        counts.append(file_count(simulation.data_storage_dir))

    # Each cycle's files replace the last's.
    assert counts[1:] == [counts[1]] * 5, counts


@pytest.mark.parametrize("new_names", [False, True], ids=["same_name", "new_names"])
def test_branches_made_and_dropped_again_and_again_leave_a_steady_number_of_files(
    new_names,
):
    simulation = build_simulation()
    simulation.set_input("salary", PERIOD, np.array([3_000.0]))
    simulation.calculate("disposable_income", PERIOD)
    counts = []

    for cycle in range(6):
        # A branch's name is part of the keys it stores under.
        name = f"branch_{cycle}" if new_names else "branch"
        branch = simulation.get_branch(name)
        branch.set_input("salary", PERIOD, np.array([3_000.0 + cycle]))
        branch.calculate("disposable_income", PERIOD)
        del branch
        _drop_branch(simulation, name)
        simulation.apply_reform({"taxes.income_tax_rate": (0.1, 0.2)[cycle % 2]})
        simulation.calculate("disposable_income", PERIOD)
        gc.collect()
        counts.append(file_count(simulation.data_storage_dir))

    assert counts[1:] == [counts[1]] * 5, counts


# ----- Enums ----- #


def test_a_plain_array_stored_over_an_enum_reads_back_plain(tmp_path):
    storage = OnDiskStorage(str(tmp_path))
    status = CountryTaxBenefitSystem().variables["housing_occupancy_status"]
    storage.put(status.possible_values.encode(np.array(["owner"])), PERIOD)
    assert isinstance(storage.get(PERIOD), EnumArray)

    storage.put(np.array([3]), PERIOD)

    assert not isinstance(storage.get(PERIOD), EnumArray)
    np.testing.assert_array_equal(storage.get(PERIOD), [3])


# ----- Simulations sharing values ----- #


@pytest.mark.parametrize("on_disk", [False, True])
def test_derivative_leaves_the_simulation_inputs(on_disk):
    """``derivative`` perturbs a clone's input, not the simulation's own."""
    simulation = build_simulation(on_disk)
    simulation.set_input("salary", PERIOD, np.array([3_000.0]))

    derivative = simulation.derivative("income_tax", "salary", PERIOD, delta=100)

    np.testing.assert_allclose(derivative, [0.15])
    np.testing.assert_array_equal(simulation.calculate("salary", PERIOD), [3_000.0])


def test_branches_with_the_same_name_keep_their_own_values():
    simulation = build_simulation()
    simulation.set_input("salary", PERIOD, np.array([1_000.0]))
    first = simulation.get_branch("first").get_branch("nested")
    second = simulation.get_branch("second").get_branch("nested")

    first.set_input("salary", PERIOD, np.array([1.0]))
    second.set_input("salary", PERIOD, np.array([2.0]))

    np.testing.assert_array_equal(first.calculate("salary", PERIOD), [1.0])
    np.testing.assert_array_equal(second.calculate("salary", PERIOD), [2.0])
    np.testing.assert_array_equal(simulation.calculate("salary", PERIOD), [1_000.0])


@pytest.mark.parametrize("on_disk", [False, True])
def test_branch_keeps_the_values_it_started_with(on_disk):
    """A parent replacing a value after branching leaves the branch's value."""
    simulation = build_simulation(on_disk)
    simulation.set_input("salary", PERIOD, np.array([1_000.0]))
    branch = simulation.get_branch("branch")

    simulation.delete_arrays("salary", PERIOD)
    simulation.set_input("salary", PERIOD, np.array([7.0]))

    np.testing.assert_array_equal(branch.calculate("salary", PERIOD), [1_000.0])
    np.testing.assert_array_equal(simulation.calculate("salary", PERIOD), [7.0])


def test_simulation_stores_on_disk_after_its_branch_storage_is_removed():
    simulation = build_simulation()
    simulation.data_storage_dir
    branch = simulation.get_branch("branch")
    branch.set_input("salary", PERIOD, np.array([1.0]))
    del branch
    _drop_branch(simulation, "branch")

    simulation.set_input("salary", PERIOD, np.array([2.0]))

    np.testing.assert_array_equal(simulation.calculate("salary", PERIOD), [2.0])


def test_derivative_of_a_carried_over_input_leaves_it():
    """A year input carried over to later years is not perturbed by
    ``derivative`` (from the carry-over-order review's repro)."""

    class carried(Variable):
        value_type = float
        entity = entities.Person
        definition_period = periods.YEAR
        label = "Carried-over input"

    tax_benefit_system = CountryTaxBenefitSystem()
    tax_benefit_system.auto_carry_over_input_variables = True
    simulation = SimulationBuilder().build_default_simulation(
        tax_benefit_system, count=1
    )
    simulation.memory_config = MemoryConfig(max_memory_occupation=0)
    # Its holder is created after memory_config, so it stores on disk.
    tax_benefit_system.add_variables(carried)
    simulation.set_input("carried", "2012", [100])

    derivative = simulation.derivative("carried", "carried", "2012", delta=1)

    np.testing.assert_array_equal(derivative, [1.0])
    np.testing.assert_array_equal(simulation.get_array("carried", "2012"), [100.0])
    np.testing.assert_array_equal(simulation.calculate("carried", "2013"), [100.0])
