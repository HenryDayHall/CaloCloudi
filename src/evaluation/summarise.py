import os
import yaml
import ot

from scipy.stats import wasserstein_distance_nd
import numpy as np

from . import inference
from ..data import read_write
from ..detector_map import find_layers, get_layer_centers


def emd(reference, predicted):
    """
    Earth movers distance between reference and predicted datasets.
    """
    n_events = reference.shape[0]
    distances = np.empty(n_events)
    for event_n in range(n_events):
        u_values = reference[event_n][:, :3]
        v_values = predicted[event_n][:, :3]
        u_weights = reference[event_n][:, 3]
        v_weights = predicted[event_n][:, 3]
        distances[event_n] = wasserstein_distance_nd(
            u_values, v_values, u_weights, v_weights
        )
    return distances


class ReferenceBase:
    # subclasses should change this
    save_prefix = None
    # subclasses should implement calculate_reference

    def __init__(
        self,
        config,
        data_part="test",
        pick_events=None,
        total_size=1_000,
        printer=print,
    ):
        self.config = config
        self.data_part = data_part
        self.pick_events = pick_events
        self.total_size = total_size
        self.printer = printer
        reference_path = self.get_output_path()
        if os.path.exists(reference_path):
            self.printer(f"Loading precalculated reference from {reference_path}")
            self.reference = np.load(reference_path)
        else:
            self.printer(f"Calculating reference and saving to {reference_path}")
            self.reference = self.calculate_reference()
            np.savez(reference_path, **self.reference)
        self.cond = self.reference["cond"]
        self.printer(f"Have {len(self.cond)} reference events")

    def get_output_path(self):
        dataset_name = os.path.basename(self.config["data"]["dataset_path"])
        dataset_name = dataset_name.split(".")[0].split("{")[0]
        out_dir = self.config["output_path"]
        precalc_dir = os.path.join(out_dir, "precalculated_reference", dataset_name)
        os.makedirs(precalc_dir, exist_ok=True)
        file_name = f"{self.save_prefix}{self.data_part}_Total{int(self.total_size)}"
        if self.pick_events is not None:
            picky = f"_Pick{self.pick_events}"
            file_name += "".join(p for p in picky if p.isalnum())
        else:
            file_name += "_NoPick"
        file_name += ".npz"
        path = os.path.join(precalc_dir, file_name)
        return path


class EMDCalculator(ReferenceBase):
    save_prefix = "emd"

    def calculate_reference(self):
        sampler = inference.Sampler(self.config)
        self.printer("Getting condition from reference")
        cond, points, target = sampler.get_cond(
            self.data_part, self.pick_events, self.total_size, return_target=True
        )
        self.printer("Getting points per layer from reference")
        points_per_layer = inference.points_per_layer_from_target(target, self.config)
        self.printer(f"Max points per layer: {np.max(points_per_layer)}")
        self.printer("Target to physical")
        physical_points, point_layer_ids = target_to_physical(target, self.config)
        del target
        self.printer("Unshifting physical target")
        physical_points = inference.unshift_points(
            physical_points, point_layer_ids, cond, self.config
        )
        self.printer("Physical target to cells")
        cells = inference.physical_to_cells(
            physical_points, point_layer_ids, self.config
        )
        return {
            "cond": cond,
            "points": points,
            "points_per_layer": points_per_layer,
            "cells": cells,
        }

    @property
    def points(self):
        return self.reference["points"]

    @property
    def points_per_layer(self):
        return self.reference["points_per_layer"]

    @property
    def reference_cells(self):
        return self.reference["cells"]

    def run_model(self, model, output_path=None):
        sampler = inference.Sampler(self.config, model)
        total_points_to_sample = len(self.points)
        batch_length = 1
        batches = int(np.ceil(total_points_to_sample / batch_length))
        self.printer("Calculating EDM from model to reference")
        emds = np.empty(total_points_to_sample)
        for i in range(batches):
            if i % 10 == 0:
                self.printer(f"Calculating batch {i}/{batches}")
            start = i * batch_length
            end = min((i + 1) * batch_length, total_points_to_sample)
            sample = sampler.sample(self.cond[start:end], self.points[start:end])
            physical_points, point_layer_ids = inference.sample_to_physical(
                sample, self.points_per_layer[start:end], self.config
            )
            del sample
            physical_points = inference.unshift_points(
                physical_points, point_layer_ids, self.cond[start:end], self.config
            )
            cells = inference.physical_to_cells(
                physical_points, point_layer_ids, self.config
            )
            del physical_points, point_layer_ids
            new_emds = emd(self.reference_cells[start:end], cells)
            emds[start:end] = new_emds
        if output_path is not None:
            np.save(output_path, emds)
        return emds

    @staticmethod
    def get_output_path_from_model_path(model_path):
        output_path = ".".join(model_path.split(".")[:-1]) + "_emd.npz"
        return output_path

    @classmethod
    def from_model_path(
        cls,
        model_path,
        data_part="test",
        pick_events=None,
        total_size=1_000,
        printer=print,
        save_summary=True,
    ):
        printer(f"Loading model from {model_path}")
        sampler = inference.Sampler.from_model_path(model_path)
        config = sampler.config
        this = cls(
            config,
            data_part=data_part,
            pick_events=pick_events,
            total_size=total_size,
            printer=printer,
        )
        output_path = None
        if save_summary:
            output_path = cls.get_output_path_from_model_path(model_path)
            printer(f"Saving summary to {output_path}")
        this.run_model(sampler.model, output_path=output_path)
        return this


class SlicedWassersteinHL:
    def __init__(
        self,
        model_summary,
        reference_summary,
        n_projections=1000,
        force_redo=False,
        printer=print,
    ):
        self.model_summary = model_summary
        self.printer = printer
        self.n_projections = n_projections
        self.reference_summary = reference_summary
        ref_cond = self.reference_summary.singulars["cond"]
        mod_cond = self.model_summary.singulars["cond"]
        if not np.allclose(ref_cond, mod_cond):
            raise ValueError("Reference and model conditions do not match")
        if os.path.exists(self.output_path) and not force_redo:
            self.results = self.read_output(self.output_path)
        else:
            self.num_projections = n_projections
            distance, error = self.calculate()
            self.results = {"distance": distance, "error": error}
            have_pdgs = self.model_summary.config["simulate_pdgs"]
            if len(have_pdgs) > 1:
                for pdg in have_pdgs:
                    distance, error = self.calculate(pdg)
                    self.results[f"pdg_{pdg}_distance"] = distance
                    self.results[f"pdg_{pdg}_error"] = error
            self.save()

    def calculate(self, pdg=None):
        clip_to = 300
        self.printer("Creating arrays")
        ref_array = self.construct_array(self.reference_summary, pdg)
        model_array = self.construct_array(self.model_summary, pdg)
        self.printer("Running sliced wasserstein")
        distance, log = ot.sliced_wasserstein_distance(
            model_array[:clip_to],
            ref_array[:clip_to],
            n_projections=self.n_projections,
            log=True,
        )
        error = (np.std(log["projected_emds"]) / self.n_projections) ** 0.5
        self.printer(f"found distance {distance} +- {error}")
        return distance, error

    @staticmethod
    def read_output(output_path):
        with open(output_path, "r") as f:
            results = yaml.safe_load(f)
        return results

    def save(self):
        results = {key: float(value) for key, value in self.results.items()}
        with open(self.output_path, "w") as f:
            yaml.dump(results, f)

    @staticmethod
    def construct_array(summary, pdg=None):
        if pdg is None:
            n_events = summary.singulars["cond"].shape[0]
        else:
            n_events = summary.singulars[f"pdg_{pdg}_cond"].shape[0]
        inputs = []
        for key in summary.singulars.keys():
            if pdg is None and key.startswith("pdg_"):
                continue
            if pdg is not None and not key.startswith(f"pdg_{pdg}"):
                continue
            if key == "cond":
                continue
            if summary.singulars[key].shape[0] != n_events:
                continue
            values = summary.singulars[key]
            if len(values.shape) == 1:
                values = values.reshape(-1, 1)
            inputs.append(values)
        return np.concatenate(inputs, axis=1)

    @property
    def output_path(self):
        summary_path = self.model_summary.output_path
        summary_tag = "_summary"
        summary_pos = summary_path.rfind(summary_tag)
        start_part = summary_path[:summary_pos]
        end_part = summary_path[summary_pos + len(summary_tag) :]
        output_path = start_part + f"_slicedHLwas{self.n_projections}" + end_part
        return output_path

    @classmethod
    def from_model_path(cls, model_path, **model_summary_kwargs):
        model_summary = ModelSummary.from_model_path(model_path, **model_summary_kwargs)
        ref_kwargs = ["data_part", "pick_events", "total_size", "printer"]
        ref_kwargs = {
            key: model_summary_kwargs[key]
            for key in ref_kwargs
            if key in model_summary_kwargs
        }
        reference_summary = ReferenceSummary.from_model_path(model_path, **ref_kwargs)
        return cls(model_summary, reference_summary)


def _cell_mask(cells):
    """Boolean mask selecting real (non-padding) cells based on positive energy."""
    return cells[:, :, 3] > 0


def _radial_distances(cells, directions):
    """
    Perpendicular distance of each cell from the line through the origin
    along ``directions``.

    Parameters
    ----------
    cells : np.ndarray
        Array of shape ``[n_events, n_cells, 4]`` with features x, y, z, e.
    directions : np.ndarray
        Array of shape ``[n_events, 3]`` with features x, y, z.

    Returns
    -------
    np.ndarray
        Array of shape ``[n_events, n_cells]`` with perpendicular distances.
    """
    positions = cells[:, :, :3]
    unit_dirs = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    # projection of each position onto the direction
    projection = np.einsum("ecd,ed->ec", positions, unit_dirs)
    parallel = projection[:, :, None] * unit_dirs[:, None, :]
    perpendicular = positions - parallel
    return np.linalg.norm(perpendicular, axis=2)


def pca(cells, energy_fraction=1.0):
    """
    Energy-weighted principal component analysis of the spatial cell
    distribution for each event.

    Parameters
    ----------
    cells : np.ndarray
        Array of shape ``[n_events, n_cells, 4]`` with features x, y, z, e.
    energy_fraction : float, optional
        Fraction of cells to include in the analysis. Default is
        ``1.0``.

    Returns
    -------
    np.ndarray
        Array of shape ``[n_events, 3]`` with the eigenvalues of the
        energy-weighted covariance matrix, sorted in descending order.
    """
    n_events = cells.shape[0]
    eigenvalues = np.zeros((n_events, 3))
    mask = _cell_mask(cells)
    for event_n in range(n_events):
        event_mask = mask[event_n]
        # get the top energy_fraction of the cells
        energies = cells[event_n, event_mask, 3]
        energy_order = np.argsort(np.argsort(energies))
        order_cut = int(len(energies) * (1 - energy_fraction))
        energy_fraction_mask = energy_order > order_cut
        weights = energies[energy_fraction_mask]
        positions = cells[event_n, event_mask, :3][energy_fraction_mask]

        total_weight = weights.sum()
        if total_weight <= 0 or positions.shape[0] < 2:
            continue
        mean = np.average(positions, axis=0, weights=weights)
        centered = positions - mean
        cov = (weights[:, None] * centered).T @ centered / total_weight
        vals = np.linalg.eigvalsh(cov)
        eigenvalues[event_n] = np.sort(vals)[::-1]
    return eigenvalues


def event_energy(cells):
    """
    Total energy per event.

    Parameters
    ----------
    cells : np.ndarray
        Array of shape ``[n_events, n_cells, 4]`` with features x, y, z, e.

    Returns
    -------
    np.ndarray
        Array of shape ``[n_events]`` with the summed cell energy.
    """
    return cells[:, :, 3].sum(axis=1)


def cell_energies(cells, bins=None):
    """
    Histogram of cell energies for each event.

    Parameters
    ----------
    cells : np.ndarray
        Array of shape ``[n_events, n_cells, 4]`` with features x, y, z, e.
    bins : int, or list of bins, optional
        Energy bins, by default 100 logarithmically spaced bins between
        ``0.000001`` and ``100``.

    Returns
    -------
    counts : np.ndarray
        Array of shape ``[n_events, n_bins]`` with the cell count per energy
        bin for each event.
    bin_edges : np.ndarray
        Array of shape ``[n_bins + 1]`` with the energy bin edges.
    """
    mask = _cell_mask(cells)
    energies = cells[:, :, 3]
    n_events = cells.shape[0]
    if bins is None:
        bin_edges = np.logspace(np.log10(10 ** (-6)), np.log10(100), 100)
    else:
        bin_edges = np.histogram_bin_edges(energies[mask], bins=bins)
    counts = np.empty((n_events, len(bin_edges) - 1))
    for event_n in range(n_events):
        counts[event_n], _ = np.histogram(
            energies[event_n, mask[event_n]], bins=bin_edges
        )
    return counts, bin_edges


def radial_energy(cells, directions, bins=None):
    """
    Energy binned by perpendicular distance from the incident direction.

    Parameters
    ----------
    cells : np.ndarray
        Array of shape ``[n_events, n_cells, 4]`` with features x, y, z, e.
    directions : np.ndarray
        Array of shape ``[n_events, 3]`` with features x, y, z.
    bins : int, or list of bins, optional
        Number of radial bins, by default 50 bins of width 5 mm.

    Returns
    -------
    counts : np.ndarray
        Array of shape ``[n_events, n_bins]`` with the summed energy per radial bin.
    bin_edges : np.ndarray
        Array of shape ``[n_events, n_bins + 1]`` with the radial bin edges.
    """
    mask = _cell_mask(cells)
    distances = _radial_distances(cells, directions)
    energies = cells[:, :, 3]
    n_events = cells.shape[0]
    if bins is None:
        bin_edges = np.arange(0, 250, 5)
    else:
        bin_edges = np.histogram_bin_edges(distances[mask], bins=bins)
    counts = np.empty((n_events, len(bin_edges) - 1))
    for event_n in range(n_events):
        event_mask = mask[event_n]
        counts[event_n], _ = np.histogram(
            distances[event_n, event_mask],
            bins=bin_edges,
            weights=energies[event_n, event_mask],
        )
    return counts, bin_edges


def layer_energies(cells, config):
    """
    Total energy deposited in each detector layer.

    Parameters
    ----------
    cells : np.ndarray
        Array of shape ``[n_events, n_cells, 4]`` with features x, y, z, e.
    config : dict
        Configuration used to obtain the detector layer centers.

    Returns
    -------
    np.ndarray
        Array of shape ``[n_events, n_layers]`` with the summed energy per layer.
    """
    layer_ids = find_layers(config, cells, coordinates="detector")
    n_layers = len(config["data"]["layer_bottom_pos"])
    energies = cells[:, :, 3]
    mask = _cell_mask(cells) & (layer_ids >= 0)
    n_events = cells.shape[0]
    counts = np.zeros((n_events, n_layers))
    for event_n in range(n_events):
        event_mask = mask[event_n]
        counts[event_n] = np.bincount(
            layer_ids[event_n, event_mask],
            weights=energies[event_n, event_mask],
            minlength=n_layers,
        )
    return counts


def event_occupancies(cells):
    """
    Number of active cells per event.

    Parameters
    ----------
    cells : np.ndarray
        Array of shape ``[n_events, n_cells, 4]`` with features x, y, z, e.

    Returns
    -------
    np.ndarray
        Array of shape ``[n_events]`` with the count of non-padding cells.
    """
    return _cell_mask(cells).sum(axis=1)


def radial_occupancies(cells, directions, bins=None):
    """
    Number of active cells binned by perpendicular distance from the
    incident direction.

    Parameters
    ----------
    cells : np.ndarray
        Array of shape ``[n_events, n_cells, 4]`` with features x, y, z, e.
    directions : np.ndarray
        Array of shape ``[n_events, 3]`` with features x, y, z.
    bins : int, or list of bins, optional
        Number of radial bins, by default 50 bins of width 5 mm.

    Returns
    -------
    counts : np.ndarray
        Array of shape ``[n_events, n_bins]`` with the cell count per radial bin.
    bin_edges : np.ndarray
        Array of shape ``[n_bins + 1]`` with the radial bin edges.
    """
    mask = _cell_mask(cells)
    distances = _radial_distances(cells, directions)
    n_events = cells.shape[0]
    if bins is None:
        bin_edges = np.arange(0, 250, 5)
    else:
        bin_edges = np.histogram_bin_edges(distances[mask], bins=bins)
    counts = np.empty((n_events, len(bin_edges) - 1))
    for event_n in range(n_events):
        counts[event_n], _ = np.histogram(
            distances[event_n, mask[event_n]], bins=bin_edges
        )
    return counts, bin_edges


def layer_occupancies(cells, config):
    """
    Number of active cells in each detector layer, summed over all events.

    Parameters
    ----------
    cells : np.ndarray
        Array of shape ``[n_events, n_cells, 4]`` with features x, y, z, e.
    config : dict
        Configuration used to obtain the detector layer centers.

    Returns
    -------
    np.ndarray
        Array of shape ``[n_events, n_layers]``
        with the count of active cells per layer.
    """
    n_layers = len(config["data"]["layer_bottom_pos"])
    layer_ids = find_layers(config, cells, coordinates="detector")
    mask = _cell_mask(cells) & (layer_ids >= 0)
    n_events = cells.shape[0]
    counts = np.zeros((n_events, n_layers), dtype=int)
    for event_n in range(n_events):
        counts[event_n] = np.bincount(
            layer_ids[event_n, mask[event_n]], minlength=n_layers
        )
    return counts


def target_to_physical(points, config):
    physical_points = np.zeros_like(points)
    point_layer_ids = find_layers(config, points, coordinates="data")
    layer_centers = get_layer_centers(config, coordinates="detector")
    mask = (points[:, :, 3] > 0) & (point_layer_ids >= 0)
    physical_points[mask, 3] = points[mask, 3]

    physical_points[mask, 1] = layer_centers[point_layer_ids[mask]]

    data_low_x = config["data"]["Xmin"]
    detector_low_z = config["data"]["Zmin_in_detector"]
    data_x_range = config["data"]["Xmax"] - data_low_x
    detector_z_range = config["data"]["Zmax_in_detector"] - detector_low_z
    shift_0 = detector_low_z - data_low_x
    scale_0 = detector_z_range / data_x_range
    physical_points[mask, 2] = (points[mask, 0] + shift_0) * scale_0

    data_low_y = config["data"]["Ymin"]
    detector_low_x = config["data"]["Xmin_in_detector"]
    data_y_range = config["data"]["Ymax"] - data_low_y
    detector_x_range = config["data"]["Xmax_in_detector"] - detector_low_x
    shift_1 = detector_low_x - data_low_y
    scale_1 = detector_x_range / data_y_range
    physical_points[mask, 0] = (points[mask, 1] + shift_1) * scale_1

    return physical_points, point_layer_ids


class SingularsMixin:
    def calculate_singulars(self, cond, cells, bins=None):
        if bins is None:
            bins = {}
        singulars = self._calculate_singulars(cond, cells, bins=bins)
        pdg_list = self.config["simulate_pdgs"]
        n_pdgs_simulated = len(pdg_list)
        if n_pdgs_simulated > 1:
            pdg_col_num = inference.get_pdg_col_number(self.config)
            pdg_onehot = (
                cond[:, pdg_col_num : pdg_col_num + n_pdgs_simulated]
                .astype(int)
                .astype(bool)
            )
            for pdg_n, pdg in enumerate(pdg_list):
                tag = f"pdg_{pdg}"
                my_bins = {
                    k[len(tag) :]: bins[k] for k in bins.keys() if k.startswith(tag)
                }
                pdg_mask = pdg_onehot[:, pdg_n]
                pdg_singulars = self._calculate_singulars(
                    cond[pdg_mask], cells[pdg_mask], my_bins
                )
                for k, v in pdg_singulars.items():
                    key = f"{tag}_{k}"
                    singulars[key] = v
        return singulars

    def _calculate_singulars(self, cond, cells, bins):
        singulars = {}
        direction_start, direction_end = inference.get_col_range_in_cond(
            self.config, "incident_direction"
        )
        directions = cond[:, direction_start:direction_end]
        directions = directions[:, [1, 2, 0]]

        singulars["cond"] = cond
        # not binned
        singulars["pca"] = pca(cells)
        # not binned
        singulars["pca_top4"] = pca(cells, energy_fraction=0.04)
        # not binned
        singulars["event_energy"] = event_energy(cells)
        cell_energy_edges = bins.get("cell_energy_edges", None)
        cell_energy_counts, cell_energy_edges = cell_energies(cells, cell_energy_edges)
        singulars["cell_energies"] = cell_energy_counts
        singulars["cell_energies_edges"] = cell_energy_edges
        # not binned
        singulars["layer_energies"] = layer_energies(cells, self.config)
        # not binned
        singulars["layer_occupancies"] = layer_occupancies(cells, self.config)
        # not binned
        singulars["event_occupancies"] = event_occupancies(cells)

        # need to move the cells to 0 to get the the radials
        floored_cells = np.copy(cells)
        floored_cells[:, :, 1] -= self.config["detector"]["layer_bottom_pos"][0]
        del cells
        radial_energy_edges = bins.get("radial_energy_edges", None)
        radial_energy_counts, radial_energy_edges = radial_energy(
            floored_cells, directions, radial_energy_edges
        )
        assert radial_energy_counts.shape[1] == radial_energy_edges.shape[0] - 1
        singulars["radial_energy"] = radial_energy_counts
        singulars["radial_energy_edges"] = radial_energy_edges
        radial_occ_edges = bins.get("radial_occ_edges", None)
        radial_occ_counts, radial_occ_edges = radial_occupancies(
            floored_cells, directions, radial_occ_edges
        )
        singulars["radial_occupancies"] = radial_occ_counts
        singulars["radial_occupancies_edges"] = radial_occ_edges
        for key in singulars.keys():
            if key.endswith("edges"):
                bins[key] = singulars[key]
        return singulars


class ModelSummary(SingularsMixin):
    def __init__(
        self,
        config,
        model=None,
        sample_cells=None,
        cond=None,
        points_per_layer=None,
        energy_per_layer=None,
        data_part="test",
        pick_events=None,
        total_size=1_000,
        printer=print,
        output_path=None,
        rescale_energy=False,
        force_redo=False,
    ):
        self.config = config
        self.data_part = data_part
        self.pick_events = pick_events
        self.total_size = total_size
        self.printer = printer
        self.rescale_energy = rescale_energy
        # generate these later as needed, or they can be given as arguments to use pcfm
        self._cond = cond
        if self._cond is not None:
            self.external_cond = True
        self._points_per_layer = points_per_layer
        if self._points_per_layer is None:
            self._points = None
        else:
            self._points = np.sum(self._points_per_layer, axis=1)
        self._energy_per_layer = energy_per_layer
        self.output_path = output_path
        if output_path is not None and os.path.exists(output_path) and not force_redo:
            self.printer(f"Loading singulars from {output_path}")
            self.singulars = np.load(output_path)
        elif model is not None:
            self.printer(f"Have model, and output_path is {output_path}")
            assert sample_cells is None
            self.singulars = self.add_model(model, output_path=output_path)
        elif sample_cells is not None:
            self.printer(f"Have sample cells, and output_path is {output_path}")
            self.singulars = self.add_sample_cells(
                sample_cells, output_path=output_path
            )

    def _make_sampling_kit(self):
        self.printer("Making sampling kit")
        sampler = inference.Sampler(self.config)
        self.printer("Getting condition from reference")
        self._cond, self._points, target = sampler.get_cond(
            self.data_part, self.pick_events, self.total_size, return_target=True
        )
        self.printer("Getting points per layer from reference")
        if self.rescale_energy:
            (
                self._points_per_layer,
                self._energy_per_layer,
            ) = inference.points_per_layer_from_target(
                target, self.config, return_energy=True
            )
        else:
            self._points_per_layer = inference.points_per_layer_from_target(
                target, self.config
            )
        del target
        self.printer(f"Max points per layer: {np.max(self._points_per_layer)}")

    @property
    def cond(self):
        if self._cond is None:
            self._make_sampling_kit()
        return self._cond

    @property
    def points(self):
        if self._points is None:
            self._make_sampling_kit()
        return self._points

    @property
    def points_per_layer(self):
        if self._points_per_layer is None:
            self._make_sampling_kit()
        return self._points_per_layer

    @property
    def energy_per_layer(self):
        assert self.rescale_energy, "Energy not rescaled"
        if self._energy_per_layer is None:
            self._make_sampling_kit()
        return self._energy_per_layer

    @classmethod
    def from_model_path(
        cls,
        model_path,
        cond=None,
        points_per_layer=None,
        energy_per_layer=None,
        data_part="test",
        pick_events=None,
        total_size=1_000,
        printer=print,
        save_summary=True,
        rescale_energy=False,
        force_redo=False,
    ):
        printer(f"Loading model from {model_path}")
        output_path = None
        if save_summary:
            output_path = cls.get_output_path_from_model_path(
                model_path,
                external_cond=(cond is not None),
                rescale_energy=rescale_energy,
            )
            printer(f"Saving summary to {output_path}")
        config = inference.Sampler.get_config_from_model_path(model_path)
        this = cls(
            config,
            model_path,
            cond=cond,
            points_per_layer=points_per_layer,
            energy_per_layer=energy_per_layer,
            data_part=data_part,
            pick_events=pick_events,
            total_size=total_size,
            printer=printer,
            output_path=output_path,
            rescale_energy=rescale_energy,
            force_redo=force_redo,
        )
        return this

    def add_model(self, model, output_path=None):
        sampler = inference.Sampler(self.config, model)
        total_points_to_sample = len(self.points)
        batch_length = 32
        batches = int(np.ceil(total_points_to_sample / batch_length))
        self.printer("Sampling the model using the reference")
        singulars = {}
        bins = {}
        energy_units_correction = 1
        for i in range(batches):
            if i % 10 == 0:
                self.printer(f"Sampling batch {i}/{batches}")
            start = i * batch_length
            end = min((i + 1) * batch_length, total_points_to_sample)
            sample = sampler.sample(self.cond[start:end], self.points[start:end])
            physical_points, point_layer_ids = inference.sample_to_physical(
                sample, self.points_per_layer[start:end], self.config
            )
            del sample
            if self.rescale_energy:
                physical_points = inference.energy_corrections(
                    physical_points, point_layer_ids, self.energy_per_layer[start:end]
                )
            if True:  # "Padded_photon_full_" in self.config["data"]["dataset_path"]:
                physical_points *= energy_units_correction
                energy_mask = physical_points[..., 3] > 0
                mean_energy = np.mean(physical_points[energy_mask][..., 3])
                self.printer(f"Mean point energy: {mean_energy}")
                if mean_energy > 0.1:
                    self.printer("Warning, might be having an issue with energy units")
                    if energy_units_correction != 1:
                        raise RuntimeError(
                            "Energy units appear to need more than one correction"
                            f" previous energy correction: {energy_units_correction}"
                            " but mean energy ({mean_energy}) > 0.1"
                        )
                    energy_units_correction *= 10 ** (-3)
                    physical_points[..., 3] *= energy_units_correction
                elif mean_energy < 0.000001:
                    self.printer("Warning, might be having an issue with energy units")
                    if energy_units_correction != 1:
                        raise RuntimeError(
                            "Energy units appear to need more than one correction"
                            f" previous energy correction: {energy_units_correction}"
                            " but mean energy ({mean_energy}) < 0.000001"
                        )
                    energy_units_correction *= 10 ** (3)
                    physical_points[..., 3] *= energy_units_correction
            physical_points = inference.unshift_points(
                physical_points, point_layer_ids, self.cond[start:end], self.config
            )
            cells = inference.physical_to_cells(
                physical_points, point_layer_ids, self.config
            )
            del physical_points, point_layer_ids
            new_singulars = self.calculate_singulars(
                self.cond[start:end], cells, bins=bins
            )
            for key in new_singulars:
                if key not in singulars:
                    singulars[key] = []
                singulars[key].append(new_singulars[key])
        singulars = {
            k: (v[0] if k.endswith("_edges") else np.concatenate(v, axis=0))
            for k, v in singulars.items()
        }
        if output_path is not None:
            np.savez(output_path, **singulars)
        return singulars

    def add_sample_cells(self, sample_cells, output_path=None):
        """
        Assume that the user has already established this isn't too memory intensive
        """
        mean_energy = np.mean(sample_cells[..., 3])
        print(f"Mean cell energy: {mean_energy}")
        if mean_energy > 0.001:
            self.printer(
                "Warning, might be having a cell level issue with energy units"
            )
            sample_cells[..., 3] *= 10 ** (-3)
        if mean_energy < 0.0000001:
            self.printer(
                "Warning, might be having a a cell level issue with energy units"
            )
            sample_cells[..., 3] *= 10 ** (3)
        sample_singulars = self.calculate_singulars(self.cond, sample_cells)
        if output_path is not None:
            np.savez(output_path, **sample_singulars)
        return sample_singulars

    @staticmethod
    def get_output_path_from_model_path(model_path, external_cond, rescale_energy):
        output_path = ".".join(model_path.split(".")[:-1])
        if external_cond:
            output_path += "_external_cond"
        if rescale_energy:
            output_path += "_rescaled_energy"
        output_path += "_summary.npz"
        return output_path


class ReferenceSummary(ReferenceBase, SingularsMixin):
    """
    Much less memory intensive than the model summary.
    """

    # need this for the ReferenceBase
    save_prefix = "pre"

    # also need to implement calculate_reference
    def calculate_reference(self):
        cond, cells = self.fetch_reference()
        return self.calculate_singulars(cond, cells)

    def fetch_reference(self):
        cond_columns = [
            self.config["data"][f"{name}_key"]
            for name in self.config["model"]["cond_features"]
        ]
        cond, target = read_write.read_raw_regaxes(
            self.config,
            part=self.data_part,
            pick_events=self.pick_events,
            total_size=self.total_size,
            per_event_cols=cond_columns,
        )
        cond = inference.pdg_to_onehot_in_full_cond(self.config, cond)
        physical_points, point_layer_ids = target_to_physical(target, self.config)
        del target
        physical_points = inference.unshift_points(
            physical_points, point_layer_ids, cond, self.config
        )
        cells = inference.physical_to_cells(
            physical_points, point_layer_ids, self.config
        )
        return cond, cells

    @property
    def singulars(self):
        return self.reference

    @classmethod
    def from_model_path(
        cls,
        model_path,
        data_part="test",
        pick_events=None,
        total_size=1_000,
        printer=print,
    ):
        config = inference.Sampler.get_config_from_model_path(model_path)
        this = cls(
            config,
            data_part=data_part,
            pick_events=pick_events,
            total_size=total_size,
            printer=printer,
        )
        return this


def get_reference(model_path):
    output_path = ReferenceSummary.get_output_path_from_model_path(model_path)
    singulars = np.load(output_path)
    return singulars


def complete_model(model_path, n_events, **model_summary_kwargs):
    force = model_summary_kwargs.pop("force", False)
    external_cond = "cond" in model_summary_kwargs
    rescale_energy = model_summary_kwargs.get("rescale_energy", False)
    output_path = ModelSummary.get_output_path_from_model_path(
        model_path, external_cond, rescale_energy
    )
    model_summary_kwargs["output_path"] = output_path
    model_summary_kwargs["force_redo"] = force
    if not os.path.exists(output_path) or force:
        print(f"Summarising to {output_path}")
        ModelSummary.from_model_path(
            model_path, data_part="test", total_size=n_events, **model_summary_kwargs
        )
    else:
        print(f"Already summarised to {output_path}")
    ema_model_path = model_path.replace("_model.pt", "_ema_model.pt")
    if os.path.exists(ema_model_path):
        print("EMA model exists")
        output_path = ModelSummary.get_output_path_from_model_path(
            ema_model_path, external_cond, rescale_energy
        )
        model_summary_kwargs["output_path"] = output_path
        if not os.path.exists(output_path) or force:
            print(f"Summarising to {output_path}")
            ModelSummary.from_model_path(
                ema_model_path,
                data_part="test",
                total_size=n_events,
                **model_summary_kwargs,
            )
        else:
            print(f"Already summarised to {output_path}")
    else:
        print("EMA model does not exist")
    ReferenceSummary.from_model_path(model_path, data_part="test", total_size=n_events)
    print("Done summaries")
