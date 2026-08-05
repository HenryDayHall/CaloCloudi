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
    def __init__(
        self,
        split_indices: list[int],
        components: list[Transformation | list | dict | None],
        axis: int = -1,
    ):
        super().__init__()
        self.split_indices = split_indices
        self.components = [self._build_component(c) for c in components]
        self.axis = axis
        self._setup_splits()

    @staticmethod
    def _build_component(
        component: Transformation | list | dict | None,
    ) -> Transformation:
        if isinstance(component, Transformation):
            return component
        return compose(component)

    def _setup_splits(self):
        self._splits = []
        starts = [0] + self.split_indices
        ends = self.split_indices + [None]
        for start, end in zip(starts, ends):
            here = slice(start, end)
            if self.axis < 0:
                split = [Ellipsis, here] + [slice(None)] * (-self.axis - 1)
            else:
                split = [slice(None)] * self.axis + [here]
            self._splits.append(split)

    def forward(self, x: torch.Tensor):
        for split, component in zip(self._splits, self.components):
            x[*split] = component.forward(x[*split])
        return x

    def inverse(self, x: torch.Tensor):
        for split, component in zip(self._splits, self.components):
            x[*split] = component.inverse(x[*split])
        return x


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
    """Build a :class:`Sequence` of transformations from a specification.

    Parameters
    ----------
    transformation : list of list or None
        A list of transformation specifications. Each element is a list where
        the first item is the transformation name and the optional second item
        is either a list of positional arguments or a dict of keyword
        arguments. If ``None``, an identity transformation is returned.

    Returns
    -------
    Sequence
        The composed sequence of transformations.

    Raises
    ------
    ValueError
        If a transformation name is invalid or its arguments are not a list or
        a dict.
    """
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
    """Recursively resolve config keys within ``value``.

    A ``value`` that is a list of strings forming a valid path into ``configs``
    is replaced by the value found there. Otherwise ``value`` is treated as a
    raw value or a container to recurse into.

    Parameters
    ----------
    configs : dict
        Nested configuration dictionary.
    value : object
        The value to resolve.

    Returns
    -------
    object
        The resolved value.
    """
    if not hasattr(value, "__iter__"):
        return value
    try:
        part = configs
        for key in value:
            part = part[key]
        return part
    except KeyError:
        return value


def leaf_like(value):
    """Return ``True`` if ``value`` is a leaf of the transformation structure.

    A leaf is either a non-iterable value or a list containing only leaves
    (i.e. a list of strings representing a config key path).

    Parameters
    ----------
    value : object
        The value to check.

    Returns
    -------
    bool
        Whether ``value`` should be treated as a leaf.
    """
    if not hasattr(value, "__iter__"):
        return True
    if isinstance(value, list):
        return all(leaf_like(v) for v in value)
    return isinstance(value, str)


def fetch_values(
    configs: dict,
    nested_iterable,
):
    """Recursively resolve config keys within a nested structure.

    The ``nested_iterable`` describes a transformation (or part of one) and may
    contain nested lists and dicts. Any leaf that is a list of strings forming a
    valid path into ``configs`` is replaced by the value found there. Raw values
    are left untouched.

    Parameters
    ----------
    configs : dict
        Nested configuration dictionary.
    nested_iterable : object
        The structure to resolve. May contain lists, dicts, and leaf values.

    Returns
    -------
    object
        The structure with config keys replaced by their values.
    """
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
    """Build the preprocessing transformation for a given part.

    Parameters
    ----------
    configs : dict
        Nested configuration dictionary containing a ``"preprocessing"`` entry.
    part : str
        The part of the preprocessing to build (e.g. ``"features"`` or
        ``"conditioning"``).

    Returns
    -------
    Sequence
        The composed preprocessing transformation with config keys resolved.
    """
    transformations = configs["preprocessing"][part]
    transformations = fetch_values(configs, transformations)
    return compose(transformations)
