"""
An inference only implementation of the network with minimal imports,
and the tools to compile it for optimal inference speed.

For now this stops at the raw sampled points (after low energy culling).
The physical post processing (``sample_to_physical`` / ``energy_corrections``
in :mod:`src.evaluation.inference`) is deliberately left out and can be added
later.

Only distilled (consistency) models are supported: sampling is a single
network evaluation.  ``max_points`` is fixed when the model is compiled so the
graph shape stays static.
"""

import torch

from ..diffusion import Diffusion
from ..data import transforms


class CompiledDiffusion(torch.nn.Module):
    """A compiled, inference only, distilled diffusion sampler.

    Parameters
    ----------
    config : dict
        Full configuration dictionary.
    model_path : str
        Checkpoint holding the distilled network weights.  Loaded when the
        model is compiled.
    max_points : int
        Number of points generated per event.  Fixed at compile time to keep
        the graph shape static; events asking for fewer points have their
        lowest energy points culled to zero.

    Notes
    -----
    ``infer`` accepts and returns ``torch.Tensor`` objects only; no numpy is
    used at inference time.  The physical mapping to detector coordinates is
    intentionally omitted for now.
    """

    def __init__(self, config: dict, model_path: str, max_points: int) -> None:
        super().__init__()
        self.config = config
        self.max_points = int(max_points)
        self.device = config["device"]
        self.dtype = getattr(torch, config["training"]["dtype"])

        model = Diffusion(config, distillation=True)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device, dtype=self.dtype)
        model.eval()
        self.model = model

        self.preprocess_conditioning = transforms.preprocessing(
            config, "conditioning"
        ).to(self.device, dtype=self.dtype)
        self.preprocess_features = transforms.preprocessing(
            config, "features"
        ).to(self.device, dtype=self.dtype)

        self._sample = torch.compile(self._sample_impl)

    def __getstate__(self):
        # the compiled wrapper captures _sample_impl by reference and is not
        # picklable; drop it and rebuild it in __setstate__
        state = self.__dict__.copy()
        state.pop("_sample", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._sample = torch.compile(self._sample_impl)

    def _sample_impl(self, cond: torch.Tensor) -> torch.Tensor:
        """Preprocess, sample ``max_points`` points, inverse transform.

        Parameters
        ----------
        cond : torch.Tensor (batch_size, cond_features)
            Raw conditioning.

        Returns
        -------
        torch.Tensor (batch_size, max_points, feature_dim)
            Sampled points in feature space.
        """
        preprocessed_cond = self.preprocess_conditioning.forward(cond)
        output = self.model.sample(preprocessed_cond, self.max_points)
        return self.preprocess_features.inverse(output)

    @staticmethod
    def _cull_low_energy(
        points: torch.Tensor, num_points: torch.Tensor
    ) -> torch.Tensor:
        """Zero the lowest energy points so each event keeps ``num_points``.

        Parameters
        ----------
        points : torch.Tensor (batch_size, max_points, feature_dim)
            Sampled points; column ``3`` is energy.
        num_points : torch.Tensor (batch_size,)
            Number of points to keep per event.

        Returns
        -------
        torch.Tensor
            ``points`` with culled rows set to zero.
        """
        max_points = points.shape[1]
        energies = points[:, :, 3]
        energy_order = energies.argsort(dim=1).argsort(dim=1)
        remove_from_event = max_points - num_points
        remove = energy_order < remove_from_event[:, None]
        points = points.clone()
        points[remove] = 0
        return points

    @torch.no_grad()
    def infer(
        self,
        conditioning: torch.Tensor,
        points_per_layer: torch.Tensor,
        energy_per_layer: torch.Tensor,
    ) -> torch.Tensor:
        """Sample points for a batch of events.

        Parameters
        ----------
        conditioning : torch.Tensor (batch_size, cond_features)
            Raw conditioning, one hot encoded pdgs already expanded.
        points_per_layer : torch.Tensor (batch_size, n_layers)
            Number of points requested in each layer; the total per event sets
            how many points survive the low energy cull.
        energy_per_layer : torch.Tensor (batch_size, n_layers)
            Reserved for the (not yet implemented) energy correction step.

        Returns
        -------
        torch.Tensor (batch_size, max_points, feature_dim)
            Sampled points in feature space, low energy points culled to zero.
        """
        conditioning = conditioning.to(self.device, dtype=self.dtype)
        num_points = points_per_layer.to(self.device).sum(dim=1)
        points = self._sample(conditioning)
        return self._cull_low_energy(points, num_points)

    def forward(
        self,
        conditioning: torch.Tensor,
        points_per_layer: torch.Tensor,
        energy_per_layer: torch.Tensor,
    ) -> torch.Tensor:
        return self.infer(conditioning, points_per_layer, energy_per_layer)
