# Tests

```bash
pip install pytest pytest-mock
pytest                      # from the repo root
pytest tests/test_read_write.py -k orientation
```

`pytest.ini` belongs at the repo root, not in `tests/`.

## Scope

Only the `padded` and `padded_unordered` formats. `showerdata` is stubbed in
`conftest.py` so the package need not be installed; `test_diffusion.py` skips
itself if `k_diffusion` is missing, because `src/diffusion.py` subclasses out of
it and cannot be stubbed.

`conftest.py` writes real HDF5 files with the `config/default.yaml` key names
(`events`, `energy`, `n_points`, `p_norm_local`) and covers both the
`(n_events, n_points, n_features)` and `roll_axis`
`(n_events, n_features, n_points)` layouts, plus `(n_events,)` and
`(n_events, 1)` per-event columns.

An autouse fixture clears the `lru_cache` on `get_possible_files`, `get_files`
and `get_n_events` around every test. Without it results leak between tests.

## Behaviour these tests pin down

Tests carrying an explanatory docstring or comment record current behaviour
that looks unintended. Worth a look before changing the code:

| Where | What |
| --- | --- |
| `floors_ceilings` | Raises `TypeError` on a plain list. `dataset.fuzz_perpendicular` passes `config["data"]["layer_bottom_pos"]` straight from YAML, i.e. a list — so unquantized training would break. |
| `_make_index_list` | `n_points` shaped `(n_events, 1)` breaks the ragged `np.array(...)` call. Marked `xfail`. |
| `get_n_events` | Returns a scalar for one file and a list for several. If some files are empty, several files can still yield a scalar, which `read_raw_regaxes` then indexes with `n_events[i]`. |
| `get_possible_files` | `sorted()` is lexicographic, so with 10+ files `data_10.h5` lands between `data_1.h5` and `data_2.h5` and the train/val/test ranges no longer match the file numbers. |
| `confine_to_box` | In the `detector_coords` branch the Y ceiling uses `data.cell_thickness`, not `detector.cell_thickness`. |
| `create_map` | The `confine` argument is never read. |
| `get_layer_centers`, `find_layers` | An unknown `coordinates` value gives `UnboundLocalError` rather than a clear error. |
| `_validate_orientations` | Only the prefix is checked; a nonsense suffix such as `local:abc` fails later inside `events_to_local`. |
| `_resolve_value` | An empty key path returns the entire config; a path running past a leaf raises `TypeError`, which the `except KeyError` does not catch. |
| `leaf_like` | A list of key paths counts as a leaf and is then handed to `_resolve_value`, which cannot use lists as dict keys. |
| `_read_padded` | Files with no selected events are still sliced with an empty index array (`test_selection_confined_to_one_of_several_files`). |
