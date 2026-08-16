# Tests

Please note, these tests are written by Claude.
The test the project at a point in time that it was known to be running smoothly.
However, they test it without real oversight, so if you feel it's appropriate to break one,
the scripts in the scripts folder are the ultimate test for consequences.

```bash
pip install pytest pytest-mock showerdata
pytest                      # from the repo root
pytest tests/test_read_write.py -k orientation
pytest tests/test_dataset.py -k ShowerData
```

`pytest.ini` belongs at the repo root, not in `tests/`.

## Scope

All three data formats: `padded`, `padded_unordered` and `showerdata`.
`test_dataset.py` exercises `ShowerDataDataset` against real ShowerData files
written with the `showerdata` package (it is on PyPI); when the package is not
installed it is stubbed in `conftest.py` so the rest of the suite still runs,
and the ShowerData tests skip themselves. `test_diffusion.py` skips itself if
`k_diffusion` is missing, because `src/diffusion.py` subclasses out of it and
cannot be stubbed.

`conftest.py` writes real HDF5 files with the `config/default.yaml` key names
(`events`, `energy`, `n_points`, `p_norm_local`, plus a `pdg` column) and
covers both the `(n_events, n_points, n_features)` and `roll_axis`
`(n_events, n_features, n_points)` layouts, plus `(n_events,)` and
`(n_events, 1)` per-event columns. `make_showerdata_file` writes real
ShowerData files and returns a truth dict under the on-disk attribute names
(`points`, `energies`, `directions`, `pdg`).

`base_config` mirrors `config/default.yaml`, including `simulate_pdgs: [22]`
and `incident_pdg_key: null`, so by default no pdg filtering happens and the
older single-pdg tests read exactly as before; tests that want several pdgs
set `simulate_pdgs` and the key themselves. With `fmt="showerdata"` it mirrors
`config/showerdata.yaml` instead: the showerdata key names, `incident_pdg` as
a condition, and `simulate_pdgs: [-11, 11, 22]`. Its `detector` section
carries the ecal/hcal pair rather than a single `cell_thickness`, with
`hcal_start` of 2 over three layers, so layers 0 and 1 are ecal and layer 2 is
hcal — a stack small enough to read but which still exercises the split. The
`data` and `detector` sections deliberately disagree on every number, so a
test that passes has read from the section it meant to.

An autouse fixture clears the `lru_cache` on `get_possible_files`, `get_files`
and `get_n_events` around every test. Without it results leak between tests.

## Behaviour these tests pin down

Tests carrying an explanatory docstring or comment record current behaviour
that looks unintended. Worth a look before changing the code:

| Where | What |
| --- | --- |
| `choose_idxs` | The last event is unreachable once the dataset is longer than `2 * bs`, so one event never appears in an epoch. `PointCloudDatasetUnordered` does not have this problem. |
| `choose_idxs` | An odd `batch_size` gives `bs - 1` events in every middle batch and `bs` at the ends, so the batch size is not constant across an epoch. |
| `_make_index_list` | `n_points` shaped `(n_events, 1)` breaks the ragged `np.array(...)` call. Marked `xfail`. |
| `pdgs_to_onehot` | A pdg missing from `simulate_pdgs` encodes as an all-zero row rather than raising; the index-list filter normally keeps such events out of batches. Handed an `(n, 1)` column it returns a 3-D array, which `ShowerDataDataset.__getitem__` relies on its own `squeeze()` to flatten. |
| `ShowerDataDataset._make_index_list` | The pdg filter always reads the `pdg` attribute; `incident_pdg_key: null` does not switch it off, unlike the padded formats. |
| `ShowerDataDataset.__getitem__` | `ShowerDataFile[i]` keeps a leading length-1 axis, so the batch handed to `_event_processing` is 4-D and the trim cuts that axis instead of the point axis. Batches keep the full on-disk width (values are still right, the padding `PointCloudDataset` would remove stays), and the computed "trim length" is read off the fourth point slot — a batch whose chosen events are all zero there comes back with zero-width points. |
| `_event_processing` | `config["data"]["padding"]` is trusted, not verified. Naming the wrong end returns batches of pure padding, with no warning. `ShowerDataDataset` never consults it at all — even a bogus value builds. |
| `_get_prior_event_axes` | A `cond_features` entry whose `*_key` is `None` stays in `keys_to_include` but is dropped here, so `__getitem__` raises `KeyError`. |
| `fuzz_parallel`, `fuzz_perpendicular` | Neither consults the energy column when moving points, so zero-energy padding is displaced too. `fuzz_perpendicular` builds a `done` mask and never reads it. |
| `get_n_events` | Returns a scalar for one file and a list for several. If some files are empty, several files can still yield a scalar, which `read_raw_regaxes` then indexes with `n_events[i]`. |
| `get_possible_files` | `sorted()` is lexicographic, so with 10+ files `data_10.h5` lands between `data_1.h5` and `data_2.h5` and the train/val/test ranges no longer match the file numbers. |
| `_read_padded` | Files with no selected events are still sliced with an empty index array (`test_selection_confined_to_one_of_several_files`). |
| `floors_ceilings` | Raises `TypeError` on a plain list, so callers must wrap `layer_bottom_pos` in `np.array` first. Every caller in `detector_map` now does. |
| `confine_to_box` | The Y ceiling is the top of the last layer, which is half a cell below that layer's own band ceiling. Only the buffer above the topmost layer is affected, where there is no detector anyway. |
| `confine_to_box` | Two-dimensional input is silently wrong: `np.where(...)[0]` indexes by row, so rows are returned whole and repeated, and the output is longer than the input. |
| `create_map` | The `confine` argument is never read; confinement always happens, in detector coordinates. |
| `create_map` | A detector layer the muon map does not cover raises `IndexError` from `unique_X[0]` rather than giving an empty grid. |
| `_cell_centers_in_row` | The gap test compares each unique position with its predecessor, not with the last accepted centre, so a chain of near-duplicates can walk past a genuinely new cell. |
| `get_layer_centers`, `find_layers` | An unknown `coordinates` value gives `UnboundLocalError` rather than a clear error. |
| `_validate_orientations` | Only the prefix is checked; a nonsense suffix such as `local:abc` fails later inside `events_to_local`. |
| `_resolve_value` | An empty key path returns the entire config; a path running past a leaf raises `TypeError`, which the `except KeyError` does not catch. |
| `leaf_like` | A list of key paths counts as a leaf and is then handed to `_resolve_value`, which cannot use lists as dict keys. |
