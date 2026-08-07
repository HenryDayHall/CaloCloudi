"""
Read and write dat aon disk, creates a common interface.
"""

import glob
from functools import lru_cache

import h5py
import numpy as np
import showerdata


@lru_cache(maxsize=1)
def get_possible_files(dataset_path):
    """
    Given a dataset path, pottentially with unfilled format entries,
    find all the files that are accessible and match the pattern.

    Parameters
    ----------
    dataset_path: str
        Path to the dataset
        The path can contain format placeholders, e.g. "data_{}.h5"

    Returns
    -------
    matching_files: list of str
        List of all the files that match the pattern
        Sorted in deterministic order.

    Raises
    ------
    FileNotFoundError
        If no files are found with the pattern
    """
    replacements = ["*" for _ in range(dataset_path.count("{"))]
    subbed_name = dataset_path.format(*replacements)
    matching_files = sorted(glob.glob(subbed_name))
    if not matching_files:
        raise FileNotFoundError(f"No files found with pattern {subbed_name}")
    return matching_files


@lru_cache(maxsize=1)
def get_files(dataset_path, file_range_start, file_range_end):
    """
    Deterministically get the requested number of files from the dataset.

    Parameters
    ----------
    dataset_path: str
        Path to the dataset
        The path can contain format placeholders, e.g. "data_{}.h5"
    file_range_start: int
        First file to read
    file_range_end: int
        Last file to read

    Returns
    -------
    files: list of str
        List of the files to be read
        If n_files is 0, all files are returned
        Sorted into deterministic order.
    """
    files = get_possible_files(dataset_path)
    assert (
        len(files) >= file_range_end
    ), f"Requested files up to {file_range_end}, but only {len(files)} available"
    return files[file_range_start:file_range_end]


@lru_cache(maxsize=1)
def get_n_events(dataset_path, file_range_start, file_range_end, dataset_format):
    """
    Get the number of events in the dataset

    Parameters
    ----------
    dataset_path: str
        Path to the dataset
    file_range_start: int
        First file to read
    file_range_end: int
        Last file to read

    Returns
    -------
    n_events: int or list of ints
        Number of events in the dataset, or in each file if n_files > 0

    """
    n_events = []
    for file_name in get_files(dataset_path, file_range_start, file_range_end):
        if dataset_format == "showerdata":
            loaded = showerdata.ShowerDataFile(file_name)
            n_events.append(len(loaded))
        else:
            with h5py.File(file_name, "r") as on_disk:
                events_array_shape = on_disk["events"].shape
                if np.sum(events_array_shape):
                    n_events.append(on_disk["events"].shape[-3])
    if len(n_events) < 2:
        n_events = np.sum(n_events)
    return n_events


def n_events_in_part(config, part):
    file_range_start = config["data"][f"{part}_range_start"]
    file_range_end = config["data"][f"{part}_range_end"]
    n_events = get_n_events(
        config["data"]["dataset_path"],
        file_range_start,
        file_range_end,
        config["data"]["format"],
    )
    return n_events


def events_to_local(events, orientation):
    """
    Rotate axes order so that the shower progresses along the z-axis.
    Modifes in place.
    To get local coordinates from global coordinates,
    the axis along which the shower is developing is swapped with the z-axis.
    To revert it, the swap is performed again.

    Parameters
    ----------
    events : array
        The input showers tensor.
    orientation : str
        The relationship between orientation on disk and local coordinates.

    """
    if 0 in events.shape:
        return
    local_orientation = _validate_orientations(orientation)
    column_target = ["xyz".index(c) for c in local_orientation]
    events[..., column_target] = events[..., [0, 1, 2]]


def _validate_orientations(orientation, orientation_global=None):
    """
    Check the format of orientation strings, return the local and global orientations
    relative to the disk.

    Parameters
    ----------
    orientation : str
        The relationship between orientation on disk and local coordinates.
    orientation_global : str, optional
        The relationship between orientation on disk and global coordinates.
        If None, doesn't need checking

    Returns
    -------
    short_orientation : str
        The relationship between orientation on disk and local coordinates.
    short_orientation_global : str (Optional)
        The relationship between orientation on disk and global coordinates.

    """
    expected_start = "hdf5:xyz==local:"
    if not orientation.startswith(expected_start):
        raise NotImplementedError(
            f"Do not know how to interpret orientation {orientation}"
        )
    short_orientation = orientation[len(expected_start) :]
    if orientation_global is not None:
        expected_start = "hdf5:xyz==global:"
        if not orientation_global.startswith(expected_start):
            raise NotImplementedError(
                f"Do not know how to interpret orientation {orientation_global}"
            )
        short_orientation_global = orientation_global[len(expected_start) :]
        return short_orientation, short_orientation_global
    return short_orientation


def local_to_global(events, orientation, orientation_global):
    """
    Rotate events given in a local coordinate system to the global coordinate system.
    Two orientations strings are supplied becuase the metadata saves the relationship
    bettween the disk coordinates and the local coordinates, and the disk
    coordinates and the global coordinates.

    Does not act in place.

    Parameters
    ----------
    events : array [..., 4]
        The input points, in local coordinates, with last axis in xyze format.
    orientation : str
        The relationship between orientation on disk and local coordinates.
    orientation_global : str
        The relationship between orientation on disk and global coordinates.

    Returns
    -------
    events : array [..., 4]
        The input points, in global coordinates, with last axis in xyze format.

    """
    if not len(events):
        return events
    local_to_disk, disk_to_global = _validate_orientations(
        orientation, orientation_global
    )
    to_global = [local_to_disk.index(c) for c in disk_to_global]
    new_events = events.copy()
    new_events[..., to_global] = events[..., [0, 1, 2]]
    return new_events


def global_to_local(events, orientation, orientation_global):
    """
    Rotate events given in a global coordinate system to the local coordinate system.
    Two orientations strings are supplied becuase the metadata saves the relationship
    bettween the disk coordinates and the local coordinates, and the disk
    coordinates and the global coordinates.

    Does not act in place.

    Parameters
    ----------
    events : array [..., 4]
        The input points, in global coordinates, with last axis in xyze format.
    orientation : str
        The relationship between orientation on disk and local coordinates.
    orientation_global : str
        The relationship between orientation on disk and global coordinates.

    Returns
    -------
    events : array [..., 4]
        The input points, in local coordinates, with last axis in xyze format.

    """
    if not len(events):
        return events
    local_to_disk, disk_to_global = _validate_orientations(
        orientation, orientation_global
    )
    to_local = [disk_to_global.index(c) for c in local_to_disk]
    new_events = events.copy()
    new_events[..., to_local] = events[..., [0, 1, 2]]
    return new_events


def read_raw_regaxes(
    config, part="test", pick_events=None, total_size=None, per_event_cols=None
):
    """
    Draw raw events form the dataset, reshape, and move to local axis orientation,
    but don't alter values.

    Parameters
    ----------
    config: config.Configs
        The configuration object
    pick_events: list of ints or slice
        Indices of the events to pick
        Optional, default is None, then a linearly spaced selection is made
    total_size: int
        Total number of events to pick
        Ignored if pick_events is not None
        If set to -1, read all events.
        Optional, default is 100
    per_event_cols: list of str
        The event level metadata columns to read
        Optional, default is None, then only ["energy"] is read
        if it is present.

    Returns
    -------
    per_event: array
        Event level variables, such as
        Incident energies of the events
        The first index is the event number
    events: array
        The events themselves
        The First index is the event number,
        the second is the point number,
        the third is the coordinate number in xyze format

    """
    if per_event_cols is None:
        per_event_cols = ["energy"]
    file_range_start = config["data"][f"{part}_range_start"]
    file_range_end = config["data"][f"{part}_range_end"]
    n_events = get_n_events(
        config["data"]["dataset_path"],
        file_range_start,
        file_range_end,
        config["data"]["format"],
    )
    n_total_events = np.sum(n_events)
    total_size = min(100 if total_size is None else total_size, n_total_events)
    if pick_events is None:
        if total_size == 0:
            return np.zeros(0), np.zeros(0)
        if total_size == -1:
            pick_events = np.arange(n_total_events)
        else:
            print("Selecting evenly spaced events")
            pick_events = np.linspace(0, n_total_events - 1, max(total_size, 1)).astype(
                int
            )
    elif isinstance(pick_events, slice):
        pick_events = np.arange(n_total_events)[pick_events]
    else:
        pick_events = np.array(pick_events, dtype=int)
        assert (
            np.max(pick_events, initial=0) < n_total_events
        ), "Event index out of range in pick_events"

    file_names = get_files(
        config["data"]["dataset_path"], file_range_start, file_range_end
    )
    file_indices = []
    if len(file_names) == 1:
        file_indices.append(pick_events)
    else:
        file_start = 0
        for i, file_name in enumerate(file_names):
            file_end = file_start + n_events[i]
            file_indices.append(
                pick_events[(pick_events >= file_start) & (pick_events < file_end)]
                - file_start
            )
            file_start = file_end

    if config["data"]["format"] == "showerdata":
        per_event, events = _read_showerdata(
            config, file_names, file_indices, per_event_cols
        )
    else:
        per_event, events = _read_padded(
            config, file_names, file_indices, per_event_cols
        )

    # postprocess
    per_event = np.vstack(per_event)
    if per_event.shape[1] == 1:
        per_event = per_event[:, 0]

    if len(events) == 1 or len(set(e.shape[1] for e in events)) == 1:
        events = np.vstack(events)
    else:  # pad to max len
        # print("Padding to max len in case it is not done beforehand.")
        to_pad = [e for e in events if e.shape[0] > 0]
        max_len = max(e.shape[1] for e in events)
        to_pad = np.array(
            [
                np.concatenate(
                    [e, np.zeros((e.shape[0], max_len - e.shape[1], e.shape[-1]))],
                    axis=1,
                )
                for e in to_pad
            ]
        )
        events = np.vstack(to_pad)

    events_to_local(events, config["data"]["orientation"])

    return per_event, events


def _read_showerdata(config, file_names, file_indices, per_event_cols):
    per_event = []
    events = []
    for name, indices in zip(file_names, file_indices):
        dataset = showerdata.ShowerDataFile(name)[indices]
        events_here = dataset.points
        # treat empty files
        if len(events_here) == 0:
            continue
        events.append(events_here)

        per_event_here = []
        for col, name in enumerate(per_event_cols):
            values = getattr(dataset, name)
            if len(values.shape) == 1:
                values = values[:, None]
            per_event_here.append(values)
        per_event.append(np.concatenate(per_event_here, axis=1))

    return per_event, events


def _read_padded(config, file_names, file_indices, per_event_cols):
    per_event = []
    events = []
    for name, indices in zip(file_names, file_indices):
        with h5py.File(name, "r") as dataset:
            events_here = dataset["events"]
            # treat empty files
            if len(events_here) == 0:
                continue
            # account for the variations in shower axes layout
            n_events_here = events_here.shape[-3]
            events_here = events_here[..., indices, :, :]
            # and make the axes order regular
            if config["data"]["roll_axis"]:
                events_here = events_here.transpose(0, 2, 1)

            events.append(events_here)

            # event level variables have more randomness again
            # can be (n_events,) or (1, n_events) or even (1, n_events, 1)
            # or even have other additional axes
            per_event_here = []
            for col, name in enumerate(per_event_cols):
                shape_here = dataset[name].shape
                found_slice = False
                found_event_axis = False
                filter_idxs = []
                event_axis = 0
                for s in shape_here:
                    if not found_event_axis and s == n_events_here:
                        filter_idxs.append(indices.tolist())
                        found_event_axis = True
                    elif s == 1:
                        filter_idxs.append(0)
                    else:
                        if not found_event_axis:
                            event_axis += 1
                        assert not found_slice, "Only one slice allowed"
                        filter_idxs.append(slice(None))
                        found_slice = True
                here = dataset[name][tuple(filter_idxs)]
                if not found_slice:
                    # give it the dimensionalty needed to concatenate
                    here = here[..., None]
                if event_axis != 0:
                    here = here.swapaxes(0, event_axis)
                per_event_here.append(here)
            per_event.append(np.concatenate(per_event_here, axis=1))

    return per_event, events


def check_regaxes(incedent_energy, events):
    """
    Check if the input data is in the regular shape.

    Parameters
    ----------
    incedent_energy: array (n_events,)
        Incident energies of the events
        The first index is the event number
    events: array (n_events, n_points, 4)
        The events themselves
        The First index is the event number,
        the second is the point number,
        the third is the coordinate number in xyze format

    Raises
    ------
    ValueError
        If the input data is not in the regular shape
    """
    if len(incedent_energy.shape) != 1:
        raise ValueError("Incident energy must be 1D")
    if len(events.shape) != 3:
        raise ValueError("Events must be 3D")
    n_events = incedent_energy.shape[0]
    if n_events != events.shape[0]:
        raise ValueError("Number of events in energy and events do not match")
    if events.shape[2] != 4:
        raise ValueError("Events must have 4 coordinates")


def write_raw_regaxes(destination, incedent_energy, events):
    """
    Write data to the disk, require the data be in a regular shape.
    Assumes the axis order is the correct one for the local_xyz_orientation
    as specified in the metadata.

    Parameters
    ----------
    destination: str
        Path to the destination file
    incedent_energy: array (n_events,)
        Incident energies of the events
        The first index is the event number
    events: array (n_events, n_points, 4)
        The events themselves
        The First index is the event number,
        the second is the point number,
        the third is the coordinate number in xyze format

    Raises
    ------
    ValueError
        If the input data is not in the regular shape
    """
    check_regaxes(incedent_energy, events)
    with h5py.File(destination, "w") as dataset:
        dataset.create_dataset("energy", data=incedent_energy)
        dataset.create_dataset("events", data=events)
