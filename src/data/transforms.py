"""
Borrowed from https://github.com/FLC-QU-hep/PointCountFM
"""

import math

import torch
from torch import nn

__all__ = [
    "Transformation",
    "Sequence",
    "Partial",
    "Identity",
    "Log",
    "LogIt",
    "Affine",
    "Clamp",
    "StandardScaler",
    "Dequantize",
    "compose",
]


class Transformation(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.forward(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


class Sequence(Transformation):
    def __init__(self, modules: list[Transformation]) -> None:
        super().__init__()
        self.sub_modules = nn.ModuleList(modules)

    def fit(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        for module in self.sub_modules:
            if isinstance(module, Transformation):
                x = module.fit(x, mask)
            else:
                raise ValueError("All sub-modules must be of type Transformation")
        return x

    def forward(self, x: torch.Tensor):
        for module in self.sub_modules:
            x = module.forward(x)
        return x

    def inverse(self, x: torch.Tensor):
        for module in self.sub_modules[::-1]:
            if isinstance(module, Transformation):
                x = module.inverse(x)
            else:
                raise ValueError("All sub-modules must be of type Transformation")
        return x


class Partial(Transformation):
    """Apply a separate transformation to each slice along one axis.

    ``split_indices`` are the boundaries between slices, as for
    :func:`torch.tensor_split`: ``n`` indices describe ``n + 1`` slices, and
    exactly that many ``components`` must be given.  Each component is either a
    :class:`Transformation`, a specification for :func:`compose`, or ``None``
    to leave that slice unchanged.

    The transformed slices are concatenated into a new tensor, so the input is
    not modified and dtypes are promoted rather than truncated.  Components are
    held in a :class:`~torch.nn.ModuleList`, so nested buffers take part in
    ``state_dict``, ``.to()`` and ``.parameters()``, and ``fit`` is passed down
    to them along with the matching slice of the mask.

    Parameters
    ----------
    split_indices : list of int
        Strictly increasing, positive slice boundaries along ``axis``.
    components : list
        One entry per slice.
    axis : int, optional
        The axis to split, default the last one.

    Raises
    ------
    ValueError
        If ``split_indices`` are not strictly increasing positive integers, or
        if the number of ``components`` does not match the number of slices.
    TypeError
        If a component is a module that is not a :class:`Transformation`.
    """

    def __init__(
        self,
        split_indices: list[int],
        components: list[Transformation | list | dict | None],
        axis: int = -1,
    ) -> None:
        super().__init__()
        self.split_indices = self._check_split_indices(split_indices)
        self.axis = axis
        components = list(components)
        expected = len(self.split_indices) + 1
        if len(components) != expected:
            raise ValueError(
                f"split_indices={self.split_indices} describes {expected} slices "
                f"along axis {axis}, but {len(components)} components were given"
            )
        self.components = nn.ModuleList(
            self._build_component(component) for component in components
        )

    @staticmethod
    def _check_split_indices(split_indices: list[int]) -> list[int]:
        indices = list(split_indices)
        for index in indices:
            if isinstance(index, bool) or not isinstance(index, int):
                raise TypeError(f"split_indices must be integers, got {index!r}")
        if any(index < 1 for index in indices):
            raise ValueError(
                f"split_indices must be positive, got {indices}; an index of 0 "
                "would make an empty first slice"
            )
        if any(later <= earlier for earlier, later in zip(indices, indices[1:])):
            raise ValueError(
                f"split_indices must be strictly increasing, got {indices}; "
                "repeated indices would make empty slices"
            )
        return indices

    @staticmethod
    def _build_component(
        component: Transformation | list | dict | None,
    ) -> Transformation:
        if isinstance(component, Transformation):
            return component
        if isinstance(component, nn.Module):
            raise TypeError(
                "components must be Transformations, got "
                f"{type(component).__name__}"
            )
        return compose(component)

    def _chunk(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Split ``x`` into one view per component, checking it is big enough."""
        if not -x.ndim <= self.axis < x.ndim:
            raise ValueError(
                f"axis {self.axis} is out of range for a {x.ndim}d tensor"
            )
        length = x.shape[self.axis]
        if self.split_indices and self.split_indices[-1] >= length:
            raise ValueError(
                f"split_indices={self.split_indices} do not fit along axis "
                f"{self.axis}, which has length {length}"
            )
        return torch.tensor_split(x, self.split_indices, dim=self.axis)

    def fit(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        chunks = self._chunk(x)
        masks = self._chunk(mask) if mask is not None else [None] * len(chunks)
        return torch.cat(
            [
                component.fit(chunk, chunk_mask)
                for component, chunk, chunk_mask in zip(self.components, chunks, masks)
            ],
            dim=self.axis,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                component.forward(chunk)
                for component, chunk in zip(self.components, self._chunk(x))
            ],
            dim=self.axis,
        )

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                component.inverse(chunk)
                for component, chunk in zip(self.components, self._chunk(x))
            ],
            dim=self.axis,
        )


class Identity(Transformation):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return x


class Log(Transformation):
    def __init__(self, alpha: float = 1e-6, base: float = math.e) -> None:
        super().__init__()
        self.alpha = alpha
        self.log_base = math.log(base)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(x + self.alpha) / self.log_base

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.log_base * x) - self.alpha


class LogIt(Transformation):
    def __init__(self, alpha: float = 1e-6) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (1 - 2 * self.alpha) * x + self.alpha
        return torch.log(x / (1 - x))

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.sigmoid(x)
        return (x - self.alpha) / (1 - 2 * self.alpha)


class Affine(Transformation):
    def __init__(
        self,
        scale: float = 1.0,
        shift: float = 0.0,
        inverse_scale: float = None,
        neg_shift: float = None,
    ) -> None:
        super().__init__()
        if inverse_scale is not None:
            scale = 1 / inverse_scale
        if neg_shift is not None:
            shift = -neg_shift
        self.a = scale
        self.b = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.a * (x + self.b)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return x / self.a - self.b


class Clamp(Transformation):
    def __init__(self, min: float = 0.0, max: float = 1.0) -> None:
        super().__init__()
        self.min = min
        self.max = max

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, self.min, self.max)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return x


class StandardScaler(Transformation):
    def __init__(self, shape: tuple[int]) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(shape))
        self.register_buffer("std", torch.ones(shape))
        self.shape = shape

    def fit(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            mask = torch.ones_like(x, dtype=torch.bool)
        dims = tuple(torch.where(torch.tensor(self.shape) == 1)[0].tolist())
        mean = torch.sum(x * mask, dim=dims, keepdim=True)
        mean /= torch.sum(mask, dim=dims, keepdim=True)
        self.mean = mean
        x = x - mean
        std = torch.sqrt(torch.sum(x**2 * mask, dim=dims, keepdim=True))
        std /= torch.sqrt(torch.sum(mask, dim=dims, keepdim=True) - 1)
        std[std == 0] = 1
        self.std = std
        x = x / std
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean


class Dequantize(Transformation):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.rand_like(x)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return torch.floor(x)


def compose(transformation: list[list[str | dict | list | None]] | None) -> Sequence:
    """Build a :class:`Sequence` of transformations from a specification."""
    if transformation is None:
        return Sequence([Identity()])
    trafo_list = []
    attrs = globals()
    for element in transformation:
        if (element[0] not in __all__) or (
            element[0] in ["Transformation", "Sequence", "compose"]
        ):
            raise ValueError(f"Invalid transformation: {element[0]}")
        Trafo = attrs[element[0]]
        assert issubclass(Trafo, Transformation)
        if len(element) == 1 or element[1] is None:
            trafo_list.append(Trafo())
        elif isinstance(element[1], list):
            trafo_list.append(Trafo(*element[1]))
        elif isinstance(element[1], dict):
            trafo_list.append(Trafo(**element[1]))
        else:
            raise ValueError(
                f"argument for {element[0]} must be a list or a dict not {type(element[1])}"
            )

    return Sequence(trafo_list)


def _resolve_value(configs: dict, value):
    """Recursively resolve config keys within ``value``."""
    if (not hasattr(value, "__iter__")) or isinstance(value, str):
        return value
    try:
        part = configs
        for key in value:
            if not isinstance(key, str):
                return value
            part = part[key]
        return part
    except KeyError:
        return value


def leaf_like(value):
    """Return ``True`` if ``value`` is a leaf of the transformation structure."""
    if not hasattr(value, "__iter__"):
        return True
    if isinstance(value, list):
        return all(leaf_like(v) for v in value)
    return isinstance(value, str)


def fetch_values(configs: dict, nested_iterable):
    """Recursively resolve config keys within a nested structure."""
    if leaf_like(nested_iterable):
        return _resolve_value(configs, nested_iterable)
    if isinstance(nested_iterable, dict):
        resolved = {}
        for key, value in nested_iterable.items():
            resolved[key] = fetch_values(configs, value)
        return resolved
    if isinstance(nested_iterable, list):
        return [fetch_values(configs, element) for element in nested_iterable]
    return nested_iterable


def preprocessing(configs, part):
    """Build the preprocessing transformation for a given part."""
    transformations = configs["preprocessing"][part]
    transformations = fetch_values(configs, transformations)
    return compose(transformations)
