from torch.utils.data import Dataset
import numpy as np
import h5py
import showerdata

from ..detector_map import floors_ceilings
from .read_write import get_files, events_to_local


class AbstractBase(Dataset):
    @classmethod
    def get_n_points(cls, data, axis=-1):
        """
        Can operate on an event, or a batch of events.
        """
        n_points_arr = (data[..., axis] > 0.0).sum(-1)
        return n_points_arr

    def choose_idxs(self, idx):
        if idx > self.bs and idx < self.__len__() - self.bs:
            idxs = slice(idx - int(self.bs / 2), idx + int(self.bs / 2))
        elif idx < self.bs:
            idxs = slice(idx, idx + self.bs)
        else:
            idxs = slice(idx - self.bs, idx)
        return idxs

    def fuzz_parallel(self, event):
        shape = event.shape[:2]
        pos_offset_x = np.random.uniform(-self.offset * 0.5, self.offset * 0.5, shape)
        pos_offset_y = np.random.uniform(-self.offset * 0.5, self.offset * 0.5, shape)
        event[:, :, 0] = event[:, :, 0] + pos_offset_x
        event[:, :, 1] = event[:, :, 1] + pos_offset_y

    def fuzz_perpendicular(self, event):
        layer_bottom_pos = np.array(self.config["data"]["layer_bottom_pos"])
        cell_thickness = self.config["data"]["cell_thickness"]
        layer_floors, layer_ceilings = floors_ceilings(
            layer_bottom_pos,
            cell_thickness,
            percent_buffer=0,
        )
        select_from, select_to = floors_ceilings(
            layer_bottom_pos,
            cell_thickness,
            percent_buffer=0.5,
        )

        done = np.zeros(event.shape[:-1], dtype=bool)
        # not real points don't need moving
        done[event[..., 3] <= 0] = True
        perpendicular_axis = 2

        for i, (floor, ceiling) in enumerate(zip(layer_floors, layer_ceilings)):
            mask = (event[..., perpendicular_axis] >= select_from[i]) & (
                event[..., perpendicular_axis] < select_to[i]
            )
            event[..., perpendicular_axis][mask] = np.random.uniform(
                floor, ceiling, mask.sum()
            )
            done[mask] = True

    def __len__(self):
        return self._len


class PointCloudDataset(AbstractBase):
    # these can be accessed without instantiating the class
    energy_scale = 1000  # MeV to GeV

    def __init__(
        self,
        config,
        dataset_part="train",
    ):
        """
        Base class for point cloud open_files.
        Iterable, torch.utils.data.Dataset subclass.

        Parameters
        ----------
        config: dict
            config...
        """
        self.config = config
        file_path = self.config["data"]["dataset_path"]
        self.keys_to_include = {
            name: self.config["data"].get(f"{name}_key", name)
            for name in self.config["model"]["cond_features"] + ["points"]
        }
        self.open_files = self._open_data_files(file_path, dataset_part)

        self._prior_event_axes = self._get_prior_event_axes()

        self.max_ds_seq_len = config["training"]["max_points_per_event"]
        self.index_list = self._make_index_list()
        self.front_padded = self._is_front_padded()
        self.bs = config["training"]["batch_size"]

        self.retain_quantized = self.config["training"]["retain_quantized"]
        self.offset = (
            self.config["data"]["cell_size"] / self.config["data"]["divisions_per_cell"]
        )

        # avoid repeat calculation
        self._len = len(self.index_list)

    def _open_data_files(self, file_path, dataset_part):
        """
        Open all the data files, and return them.
        We don't bother closing them, because they are
        read-only and will be closed when the program
        exits.
        N.B. if we end up with memory issues,
        we might need to rethink this.

        Parameters
        ----------
        file_path : str
            Path to the HDF5 file containing the dataset.
            If n_files is > 0, then the file_path should
            contain one or more "{}" to be formatted with
            the file number.
        dataset_part : str
            Either "train", "val", or "test"

        Returns
        -------
        list
            List of h5py.File objects.
        """
        file_range_start = self.config["data"][f"{dataset_part}_range_start"]
        file_range_end = self.config["data"][f"{dataset_part}_range_end"]
        all_files = [
            h5py.File(path, "r")
            for path in get_files(file_path, file_range_start, file_range_end)
        ]
        if not all_files:
            raise FileNotFoundError(f"No files found at {file_path}")
        return all_files

    def _make_index_list(self):
        """
        To allow the data to be iterated in order of
        number of points if desired, make a list of
        indices that can be used to access the data
        in that order.

        Returns
        -------
        index_list : numpy.ndarray (n_points, 3)
            Array with cols (n_points, file_idx, event_idx)
            that can be used to access the data in order
            of number of points.
        """
        index_list = []
        event_key = self.config["data"]["points_key"]
        for file_idx, dataset in enumerate(self.open_files):
            if "n_points" in dataset:
                n_points = dataset["n_points"][:]
            else:
                if self.config["data"]["roll_axis"]:
                    events = np.moveaxis(dataset[event_key], -1, -2)
                    n_points = self.get_n_points(events)
                else:
                    n_points = self.get_n_points(dataset[event_key])
            n_points[n_points > self.max_ds_seq_len] = self.max_ds_seq_len
            index_list += [(n, file_idx, i) for i, n in enumerate(n_points)]
        # sort the index list by 'n_points'
        index_list.sort(key=lambda x: x[0])
        index_list = np.array(index_list, dtype=int)
        return index_list

    def _get_prior_event_axes(self):
        """
        For each key in the cond_features plus the points_key
        find out which axis is the length of th number of events
        in that file.
        Create a number of empty slices to pad out the indexing to that point.
        We make the assumption that this is the same
        for all files in the dataset.
        """
        file_0 = self.open_files[0]
        n_events_in_file0 = file_0[self.config["data"]["points_key"]].shape[0]
        axes = {
            name: [slice(None)] * file_0[key].shape.index(n_events_in_file0)
            for name, key in self.keys_to_include.items()
            if key is not None
        }
        return axes

    def _is_front_padded(self, check_file=0):
        padding = self.config["data"]["padding"]
        if padding == "front":
            is_front_padded = True
        elif padding == "back":
            is_front_padded = False
        else:
            raise ValueError(
                f"padding must be 'front' or 'back', not {padding}"
            )
        return is_front_padded

    def _event_processing(self, event):
        if self.config["data"]["roll_axis"]:
            event = np.moveaxis(event, -1, -2)

        # Ensure the shower runs along the z axis
        events_to_local(event, self.config["data"]["orientation"])

        # Trim padding
        max_len = (event[:, :, 3] > 0).sum(axis=1).max()
        trim_len = min(max_len, self.max_ds_seq_len)
        if self.front_padded:
            event = event[:, -trim_len:]
        else:
            event = event[:, :trim_len]

        if not self.retain_quantized:
            self.fuzz_parallel(event)
            self.fuzz_perpendicular(event)

        return event

    def __getitem__(self, idx):
        idxs = self.choose_idxs(idx)
        batch = {}
        for name_in_batch, name_on_disk in self.keys_to_include.items():
            padding = self._prior_event_axes[name_in_batch]
            data = np.array(
                [
                    self.open_files[file_n][name_on_disk][(*padding, event_n)]
                    for n_pts, file_n, event_n in self.index_list[idxs]
                ]
            )

            if name_in_batch == "points":
                data = self._event_processing(data)

            if len(data.shape) == 1:
                data = data[..., np.newaxis]
            batch[name_in_batch] = data

        # expose the number of points per event
        batch["n_points"] = self.index_list[idxs, 0, np.newaxis]

        return batch


class PointCloudDatasetUnordered(PointCloudDataset):
    def choose_idxs(self, idx):
        rng = np.random.default_rng(seed=idx)
        bs = min(self._len, self.bs)
        idxs = rng.choice(self._len, bs, replace=False)
        idxs.sort()
        return idxs


class ShowerDataDataset(AbstractBase):
    def __init__(
        self,
        config,
        dataset_part="train",
    ):
        """
        Base class for loading ShowerData files.
        Iterable, torch.utils.data.Dataset subclass.

        Parameters
        ----------
        config: dict
            config...
        """
        self.config = config
        file_path = self.config["data"]["dataset_path"]
        self.keys_to_include = {
            name: self.config["data"].get(f"{name}_key", name)
            for name in self.config["model"]["cond_features"] + ["points"]
        }
        self.open_files = self._open_data_files(file_path, dataset_part)

        self.max_ds_seq_len = config["training"]["max_points_per_event"]
        self.index_list = self._make_index_list()
        self.front_padded = False
        self.bs = config["training"]["batch_size"]

        self.retain_quantized = self.config["training"]["retain_quantized"]
        self.offset = (
            self.config["data"]["cell_size"] / self.config["data"]["divisions_per_cell"]
        )

        # avoid repeat calculation
        self._len = len(self.index_list)

    def _open_data_files(self, file_path, dataset_part):
        """
        Open all the data files, and return them.
        We don't bother closing them, because they are
        read-only and will be closed when the program
        exits.
        N.B. if we end up with memory issues,
        we might need to rethink this.

        Parameters
        ----------
        file_path : str
            Path to the HDF5 file containing the dataset.
            If n_files is > 0, then the file_path should
            contain one or more "{}" to be formatted with
            the file number.
        dataset_part : str
            Either "train", "val", or "test"

        Returns
        -------
        list
            List of h5py.File objects.
        """
        file_range_start = self.config["data"][f"{dataset_part}_range_start"]
        file_range_end = self.config["data"][f"{dataset_part}_range_end"]
        all_files = [
            showerdata.ShowerDataFile(path)
            for path in get_files(file_path, file_range_start, file_range_end)
        ]
        if not all_files:
            raise FileNotFoundError(f"No files found at {file_path}")
        return all_files

    def _make_index_list(self):
        """
        To allow the data to be iterated in order of
        number of points if desired, make a list of
        indices that can be used to access the data
        in that order.

        Returns
        -------
        index_list : numpy.ndarray (n_points, 3)
            Array with cols (n_points, file_idx, event_idx)
            that can be used to access the data in order
            of number of points.
        """
        index_list = []
        for file_idx, dataset in enumerate(self.open_files):
            n_points = self.get_n_points(dataset[:].points)
            n_points[n_points > self.max_ds_seq_len] = self.max_ds_seq_len
            index_list += [(n, file_idx, i) for i, n in enumerate(n_points)]
        # sort the index list by 'n_points'
        index_list.sort(key=lambda x: x[0])
        index_list = np.array(index_list, dtype=int)
        return index_list

    def _event_processing(self, event):
        # Trim padding
        max_len = (event[:, :, 3] > 0).sum(axis=1).max()
        trim_len = min(max_len, self.max_ds_seq_len)
        # always back padded
        event = event[:, :trim_len]

        if not self.retain_quantized:
            self.fuzz_parallel(event)
            self.fuzz_perpendicular(event)

        return event

    def __getitem__(self, idx):
        idxs = self.choose_idxs(idx)
        batch = {}
        for name_in_batch, name_on_disk in self.keys_to_include.items():
            data = np.array(
                [
                    getattr(self.open_files[file_n][event_n], name_on_disk)
                    for n_pts, file_n, event_n in self.index_list[idxs]
                ]
            )

            if name_in_batch == "points":
                data = self._event_processing(data)

            if len(data.shape) == 1:
                data = data[..., np.newaxis]
            batch[name_in_batch] = data

        # expose the number of points per event
        batch["n_points"] = self.index_list[idxs, 0, np.newaxis]

        return batch


def from_config(config, dataset_part="train"):
    if config["data"]["format"] == "padded":
        return PointCloudDataset(config, dataset_part=dataset_part)
    elif config["data"]["format"] == "padded_unordered":
        return PointCloudDatasetUnordered(config, dataset_part=dataset_part)
    elif config["data"]["format"] == "showerdata":
        return ShowerDataDataset(config, dataset_part=dataset_part)
    else:
        raise NotImplementedError
