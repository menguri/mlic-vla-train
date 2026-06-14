#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import collections
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torchvision.transforms import v2
from torchvision.transforms.v2 import (
    Transform,
    functional as F,  # noqa: N812
)


class RandomSubsetApply(Transform):
    """Apply a random subset of N transformations from a list of transformations.

    Args:
        transforms: list of transformations.
        p: represents the multinomial probabilities (with no replacement) used for sampling the transform.
            If the sum of the weights is not 1, they will be normalized. If ``None`` (default), all transforms
            have the same probability.
        n_subset: number of transformations to apply. If ``None``, all transforms are applied.
            Must be in [1, len(transforms)].
        random_order: apply transformations in a random order.
    """

    def __init__(
        self,
        transforms: Sequence[Callable],
        p: list[float] | None = None,
        n_subset: int | None = None,
        random_order: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(transforms, Sequence):
            raise TypeError("Argument transforms should be a sequence of callables")
        if p is None:
            p = [1] * len(transforms)
        elif len(p) != len(transforms):
            raise ValueError(
                f"Length of p doesn't match the number of transforms: {len(p)} != {len(transforms)}"
            )

        if n_subset is None:
            n_subset = len(transforms)
        elif not isinstance(n_subset, int):
            raise TypeError("n_subset should be an int or None")
        elif not (1 <= n_subset <= len(transforms)):
            raise ValueError(f"n_subset should be in the interval [1, {len(transforms)}]")

        self.transforms = transforms
        total = sum(p)
        self.p = [prob / total for prob in p]
        self.n_subset = n_subset
        self.random_order = random_order

        self.selected_transforms = None

    def forward(self, *inputs: Any) -> Any:
        needs_unpacking = len(inputs) > 1

        selected_indices = torch.multinomial(torch.tensor(self.p), self.n_subset)
        if not self.random_order:
            selected_indices = selected_indices.sort().values

        self.selected_transforms = [self.transforms[i] for i in selected_indices]

        for transform in self.selected_transforms:
            outputs = transform(*inputs)
            inputs = outputs if needs_unpacking else (outputs,)

        return outputs

    def extra_repr(self) -> str:
        return (
            f"transforms={self.transforms}, "
            f"p={self.p}, "
            f"n_subset={self.n_subset}, "
            f"random_order={self.random_order}"
        )


class SharpnessJitter(Transform):
    """Randomly change the sharpness of an image or video.

    Similar to a v2.RandomAdjustSharpness with p=1 and a sharpness_factor sampled randomly.
    While v2.RandomAdjustSharpness applies — with a given probability — a fixed sharpness_factor to an image,
    SharpnessJitter applies a random sharpness_factor each time. This is to have a more diverse set of
    augmentations as a result.

    A sharpness_factor of 0 gives a blurred image, 1 gives the original image while 2 increases the sharpness
    by a factor of 2.

    If the input is a :class:`torch.Tensor`,
    it is expected to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        sharpness: How much to jitter sharpness. sharpness_factor is chosen uniformly from
            [max(0, 1 - sharpness), 1 + sharpness] or the given
            [min, max]. Should be non negative numbers.
    """

    def __init__(self, sharpness: float | Sequence[float]) -> None:
        super().__init__()
        self.sharpness = self._check_input(sharpness)

    def _check_input(self, sharpness):
        if isinstance(sharpness, (int | float)):
            if sharpness < 0:
                raise ValueError("If sharpness is a single number, it must be non negative.")
            sharpness = [1.0 - sharpness, 1.0 + sharpness]
            sharpness[0] = max(sharpness[0], 0.0)
        elif isinstance(sharpness, collections.abc.Sequence) and len(sharpness) == 2:
            sharpness = [float(v) for v in sharpness]
        else:
            raise TypeError(f"{sharpness=} should be a single number or a sequence with length 2.")

        if not 0.0 <= sharpness[0] <= sharpness[1]:
            raise ValueError(f"sharpness values should be between (0., inf), but got {sharpness}.")

        return float(sharpness[0]), float(sharpness[1])

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        sharpness_factor = torch.empty(1).uniform_(self.sharpness[0], self.sharpness[1]).item()
        return {"sharpness_factor": sharpness_factor}

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        sharpness_factor = params["sharpness_factor"]
        return self._call_kernel(F.adjust_sharpness, inpt, sharpness_factor=sharpness_factor)


class GammaJitter(Transform):
    """Randomly change image gamma without shifting hue."""

    def __init__(self, gamma: float | Sequence[float], gain: float = 1.0) -> None:
        super().__init__()
        self.gamma = self._check_input(gamma)
        self.gain = gain

    def _check_input(self, gamma):
        if isinstance(gamma, (int | float)):
            if gamma <= 0:
                raise ValueError("If gamma is a single number, it must be positive.")
            gamma = [max(1e-6, 1.0 - gamma), 1.0 + gamma]
        elif isinstance(gamma, collections.abc.Sequence) and len(gamma) == 2:
            gamma = [float(v) for v in gamma]
        else:
            raise TypeError(f"{gamma=} should be a single number or a sequence with length 2.")

        if not 0.0 < gamma[0] <= gamma[1]:
            raise ValueError(f"gamma values should be in (0., inf), but got {gamma}.")

        return float(gamma[0]), float(gamma[1])

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        gamma = torch.empty(1).uniform_(self.gamma[0], self.gamma[1]).item()
        return {"gamma": gamma}

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        return self._call_kernel(F.adjust_gamma, inpt, gamma=params["gamma"], gain=self.gain)


class GaussianNoise(Transform):
    """Add weak zero-mean image noise to tensor images."""

    def __init__(self, std: float | Sequence[float]) -> None:
        super().__init__()
        self.std = self._check_input(std)

    def _check_input(self, std):
        if isinstance(std, (int | float)):
            std = [0.0, float(std)]
        elif isinstance(std, collections.abc.Sequence) and len(std) == 2:
            std = [float(v) for v in std]
        else:
            raise TypeError(f"{std=} should be a single number or a sequence with length 2.")

        if not 0.0 <= std[0] <= std[1]:
            raise ValueError(f"std values should be between 0 and inf, but got {std}.")

        return float(std[0]), float(std[1])

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        std = torch.empty(1).uniform_(self.std[0], self.std[1]).item()
        return {"std": std}

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        if not isinstance(inpt, torch.Tensor):
            return inpt
        if params["std"] == 0:
            return inpt
        if inpt.is_floating_point():
            return (inpt + torch.randn_like(inpt) * params["std"]).clamp(0.0, 1.0)
        noise = torch.randn_like(inpt.float()) * params["std"] * 255.0
        return (inpt.float() + noise).clamp(0, 255).to(inpt.dtype)


class CompressionJitter(Transform):
    """Approximate weak compression artifacts with per-image value quantization."""

    def __init__(self, levels: int | Sequence[int]) -> None:
        super().__init__()
        self.levels = self._check_input(levels)

    def _check_input(self, levels):
        if isinstance(levels, int):
            levels = [levels, levels]
        elif isinstance(levels, collections.abc.Sequence) and len(levels) == 2:
            levels = [int(v) for v in levels]
        else:
            raise TypeError(f"{levels=} should be an int or a sequence with length 2.")

        if not 2 <= levels[0] <= levels[1]:
            raise ValueError(f"levels should be between 2 and inf, but got {levels}.")

        return int(levels[0]), int(levels[1])

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        levels = torch.randint(self.levels[0], self.levels[1] + 1, (1,)).item()
        return {"levels": levels}

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        if not isinstance(inpt, torch.Tensor):
            return inpt
        levels = params["levels"]
        if inpt.is_floating_point():
            return (torch.round(inpt.clamp(0.0, 1.0) * levels) / levels).to(inpt.dtype)
        return (torch.round(inpt.float() / 255.0 * levels) / levels * 255.0).clamp(0, 255).to(inpt.dtype)


@dataclass
class ImageTransformConfig:
    """
    For each transform, the following parameters are available:
      weight: This represents the multinomial probability (with no replacement)
            used for sampling the transform. If the sum of the weights is not 1,
            they will be normalized.
      type: The name of the class used. This is either a class available under torchvision.transforms.v2 or a
            custom transform defined here.
      kwargs: Lower & upper bound respectively used for sampling the transform's parameter
            (following uniform distribution) when it's applied.
    """

    weight: float = 1.0
    type: str = "Identity"
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class ImageTransformsConfig:
    """
    These transforms are all using standard torchvision.transforms.v2
    You can find out how these transformations affect images here:
    https://pytorch.org/vision/0.18/auto_examples/transforms/plot_transforms_illustrations.html
    We use a custom RandomSubsetApply container to sample them.
    """

    # Set this flag to `true` to enable transforms during training
    enable: bool = False
    # This is the maximum number of transforms (sampled from these below) that will be applied to each frame.
    # It's an integer in the interval [1, number_of_available_transforms].
    max_num_transforms: int = 3
    # By default, transforms are applied in Torchvision's suggested order (shown below).
    # Set this to True to apply them in a random order.
    random_order: bool = False
    tfs: dict[str, ImageTransformConfig] = field(
        default_factory=lambda: {
            "brightness": ImageTransformConfig(
                weight=1.0,
                type="ColorJitter",
                kwargs={"brightness": (0.8, 1.2)},
            ),
            "contrast": ImageTransformConfig(
                weight=1.0,
                type="ColorJitter",
                kwargs={"contrast": (0.8, 1.2)},
            ),
            "saturation": ImageTransformConfig(
                weight=1.0,
                type="ColorJitter",
                kwargs={"saturation": (0.5, 1.5)},
            ),
            "hue": ImageTransformConfig(
                weight=1.0,
                type="ColorJitter",
                kwargs={"hue": (-0.05, 0.05)},
            ),
            "sharpness": ImageTransformConfig(
                weight=1.0,
                type="SharpnessJitter",
                kwargs={"sharpness": (0.5, 1.5)},
            ),
            "affine": ImageTransformConfig(
                weight=1.0,
                type="RandomAffine",
                kwargs={"degrees": (-5.0, 5.0), "translate": (0.05, 0.05)},
            ),
        }
    )


def make_transform_from_config(cfg: ImageTransformConfig):
    custom_transforms = {
        "SharpnessJitter": SharpnessJitter,
        "GammaJitter": GammaJitter,
        "GaussianNoise": GaussianNoise,
        "CompressionJitter": CompressionJitter,
    }
    if cfg.type in custom_transforms:
        return custom_transforms[cfg.type](**cfg.kwargs)

    transform_cls = getattr(v2, cfg.type, None)
    if isinstance(transform_cls, type) and issubclass(transform_cls, Transform):
        return transform_cls(**cfg.kwargs)

    raise ValueError(
        f"Transform '{cfg.type}' is not valid. It must be a class in "
        f"torchvision.transforms.v2 or one of {sorted(custom_transforms)}."
    )


class ImageTransforms(Transform):
    """A class to compose image transforms based on configuration."""

    def __init__(self, cfg: ImageTransformsConfig) -> None:
        super().__init__()
        self._cfg = cfg

        self.weights = []
        self.transforms = {}
        for tf_name, tf_cfg in cfg.tfs.items():
            if tf_cfg.weight <= 0.0:
                continue

            self.transforms[tf_name] = make_transform_from_config(tf_cfg)
            self.weights.append(tf_cfg.weight)

        n_subset = min(len(self.transforms), cfg.max_num_transforms)
        if n_subset == 0 or not cfg.enable:
            self.tf = v2.Identity()
        else:
            self.tf = RandomSubsetApply(
                transforms=list(self.transforms.values()),
                p=self.weights,
                n_subset=n_subset,
                random_order=cfg.random_order,
            )

    def forward(self, *inputs: Any) -> Any:
        return self.tf(*inputs)
