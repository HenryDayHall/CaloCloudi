from torch.utils.data import Dataset
import warnings
import numpy as np
import h5py

from ..utils.detector_map import floors_ceilings
from .read_write import get_files, events_to_local


class PointCloudDataset(Dataset):
    # these can be accessed without instantiating the class
    energy_scale = 1000  # MeV to GeV
    def __init__(
        self,
        file_path,
        configs,
        bs=32,
        max_ds_seq_len=6000,
        n_files=0,
    ):
        """
        Base class for point cloud open_files.
        Iterable, torch.utils.data.Dataset subclass.

        Parameters
        ----------
        file_path : str
            Path to the HDF5 file containing the dataset.
            If n_files is > 0, then the file_path should
            contain one or more "{}" to be formatted with
            the file number.
        configs: dict
            config...
        bs : int, optional
            Batch size, number of events returned in each
            iteration.
            Default is 32.
        max_ds_seq_len : int, optional
            Maximium number of points/hits in each event.
            Data will be trimmed to this length if longer on
            the disk.
            Default is 6000.
        n_files : int, optional
            If this is > 0, then the file_path should
            contain one or more "{}" to be formatted with
            the file number.
            If it's 0, then the file_path should be a
            single file.
        """
        self.configs = configs
        self.open_files = self._open_data_files(file_path, n_files)

        self._prior_event_axes = self._get_prior_event_axes()

        self.max_ds_seq_len = max_ds_seq_len
        self.index_list = self._make_index_list()
        self.front_padded = self._is_front_padded()
        self.bs = bs

        self.retain_quantized = self.config["training"]["retain_quantized"]
        self.offset = (
            self.config["data"]["cell_size"]
            / self.config["data"]["divisions_per_cell"]
        )

        self.keys_to_include = {
                name: self.configs["data"][f"{name}_key"]
                for name in 
                self.configs["model"]["cond_features"] + "points"]
        }

        # avoid repeat calculation
        self._len = len(self.index_list)

    def _open_data_files(self, file_path, n_files):
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
        n_files : int
            If this is > 0, then the file_path should
            contain one or more "{}" to be formatted with
            the file number.
            If it's 0, then the file_path should be a
            single file.

        Returns
        -------
        list
            List of h5py.File objects.
        """
        all_files = [h5py.File(path, "r") for path in get_files(file_path, n_files)]
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
        event_key = self.configs["data"]["points_key"]
        for file_idx, dataset in enumerate(self.open_files):
            if "n_points" in dataset:
                n_points = dataset["n_points"][:]
            else:
                if self.configs["data"]["roll_axis"]:
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
        n_events_in_file0 = file_0[self.configs["data"]["points_key"]].shape[0]
        axes = {
            name: [slice(None)] * file_0[key].shape.index(n_events_in_file0)
            for name, key in self.keys_to_include.items()
            if key is not None
        }
        return axes

    def _is_front_padded(self, check_file=0):
        padding = self.configs["data"]["padding"]
        if padding == "front":
            is_front_padded = True
        elif padding == "back":
            is_front_padded = False
        elif check_file >= len(self.open_files) - 1:
            # stop before we run out of files to check
            is_front_padded = False
        else:
            is_front_padded = self._is_front_padded(check_file + 1)
        return is_front_padded

    @classmethod
    def get_n_points(cls, data, axis=-1):
        """
        Can operate on an event, or a batch of events.
        """
        n_points_arr = (data[..., axis] != 0.0).sum(1)
        return n_points_arr

    def _choose_idxs(self, idx):
        if idx > self.bs and idx < self.__len__() - self.bs:
            idxs = slice(idx - int(self.bs / 2), idx + int(self.bs / 2))
        elif idx < self.bs:
            idxs = slice(idx, idx + self.bs)
        else:
            idxs = slice(idx - self.bs, idx)
        return idxs

    def _fuzz_parallel(self, event):
        pos_offset_x = np.random.uniform(0, self.offset, 1)
        pos_offset_y = np.random.uniform(0, self.offset, 1)
        event[:, :, 0] = event[:, :, 0] + pos_offset_x
        event[:, :, 1] = event[:, :, 1] + pos_offset_y

    def _fuzz_perpendicular(self, event):
        layer_bottom_pos = self.configs["data"]["layer_bottom_pos"]
        cell_thickness = self.configs["data"]["cell_thickness"]
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
        assert done.all()

    def normalize_xyze(self, event):
        # correct for the sim-E... datasets
        Xmean, Ymean, Zmean = (
            self.configs["data"]["Xmean"],
            self.configs["data"]["Ymean"],
            self.configs["data"]["Zmean"],
        )
        Xstd, Ystd, Zstd = (
            self.configs["data"]["Xstd"],
            self.configs["data"]["Ystd"],
            self.configs["data"]["Zstd"],
        )

        if self.configs["model"]["logarithmic_point_energy"]:
            Emean, Estd = (
                self.configs["data"]["log_Emean"],
                self.configs["data"]["log_Estd"],
            )
            # TODO consider setting padding to nan, so that it gets caught in the mask
            event[..., 3] = (
                (np.log(event[..., 3] + 1e-12) - Emean) / Estd / 2
            )  # energy transformation

        event[..., 0] = (event[..., 0] - Xmean) / Xstd / 2  # x coordinate normalization
        event[..., 1] = (event[..., 1] - Ymean) / Ystd / 2  # y coordinate normalization
        event[..., 2] = (event[..., 2] - Zmean) / Zstd / 2  # z coordinate normalization

    def _event_processing(self, event):
        if self._roll_axis:
            event = np.moveaxis(event, -1, -2)

        # Ensure the shower runs along the z axis
        events_to_local(event, self.configs["data"]["orientation"])

        # Trim padding
        max_len = (event[:, :, 3] > 0).sum(axis=1).max()
        trim_len = min(max_len, self.max_ds_seq_len)
        if self.front_padded:
            event = event[:, -trim_len:]
        else:
            event = event[:, :trim_len]

        if not self.retain_quantized:
            self._fuzz_parallel(event)
            self._fuzz_perpendicular(event)

        self.normalize_xyze(event)

        return event

    def __getitem__(self, idx):
        idxs = self._choose_idxs(idx)
        batch = {}
        # TODO do I need a way to apply rescales to the incident energy here?
        for name_in_batch, name_on_disk in self.keys_to_include.items():
            if name_in_batch == "points":
                continue

            padding = self._prior_event_axes[name_in_batch]
            data = np.array(
                [
                    self.open_files[file_n][name_on_disk][(*padding, event_n)]
                    for n_pts, file_n, event_n in self.index_list[idxs]
                ]
            )

            if name_in_batch == "event":
                data = self._event_processing(data)

            if len(data.shape) == 1:
                data = data[..., np.newaxis]
            batch[name_in_batch] = data

        # special case for points as it's already been processed
        if "points" in self.keys_to_include:
            batch["points"] = self.index_list[idxs, 0, np.newaxis]

        return batch

    def __len__(self):
        return self._len
