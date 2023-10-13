from typing import Dict, Iterable, List

import numpy as np
import torch
import tqdm

from .utils import _desc

__all__ = [
    "LinearityTest",
]


class LinearityTest:
    def __init__(
            self,
            epsilons: Iterable[float],
            delta: float = 1e-4,
            bound: bool = False,
            use_cuda: bool = False,
            verbose: bool = False,
    ) -> None:
        if not isinstance(epsilons, list):
            epsilons = list(epsilons)

        self.epsilons = epsilons
        self.delta = delta
        self.bound = bound
        self.use_cuda = use_cuda
        self.machine = "cuda" if use_cuda else "cpu"
        self.verbose = verbose

        self.angles = dict(epsilon=epsilons)

    def __call__(
            self,
            model,
            x: torch.Tensor,
            d: torch.Tensor,
    ) -> Dict[str, List[float]]:
        self.init_dict(model, x)

        for epsilon in tqdm.tqdm(
                self.epsilons,
                desc=_desc("Linearity test", model),
                disable=not self.verbose,
        ):
            self.compute_angle(model, x, d, epsilon)

        return self.angles

    def feature_extractor(
            self,
            model,
    ):
        from ..xray import FeatureExtractor

        return FeatureExtractor(
            model=model,
            penultimate_only=False,
            use_cuda=self.use_cuda,
        )

    def init_dict(
            self,
            model,
            x: torch.Tensor,
    ) -> None:
        assert len(self.angles) == 1

        layers = list(self.feature_extractor(model)(x).keys())

        for layer in layers:
            self.angles[layer] = list()

    def move(
            self,
            x: torch.Tensor,
            d: torch.Tensor,
            eps: float,
    ) -> torch.Tensor:
        _x = x + eps * d

        if self.bound:
            _x = _x.clamp(0, 1)

        return _x

    def compute_angle(
            self,
            model,
            x: torch.Tensor,
            d: torch.Tensor,
            epsilon: float,
    ) -> None:
        def angle_of_three_points(
                i: torch.Tensor,
                f1: torch.Tensor,
                f2: torch.Tensor,
                eps: float = 1e-7,
        ) -> float:
            assert len(i.shape) * len(f1.shape) * len(f2.shape) == 1

            v1 = f1 - i
            v2 = f2 - i

            v1 = v1 / (torch.norm(v1, p="fro") + eps)
            v2 = v2 / (torch.norm(v2, p="fro") + eps)

            angle = torch.acos(torch.dot(v1, v2).clamp(-1, 1))

            return float(angle)

        x_eps = self.move(x, d, epsilon)
        x_eps_l = self.move(x, d, epsilon - self.delta)
        x_eps_r = self.move(x, d, epsilon + self.delta)

        y_eps = self.feature_extractor(model)(x_eps.to(self.machine))
        y_eps_l = self.feature_extractor(model)(x_eps_l.to(self.machine))
        y_eps_r = self.feature_extractor(model)(x_eps_r.to(self.machine))

        for layer in y_eps.keys():
            angle = angle_of_three_points(
                i=y_eps[layer].reshape(-1),
                f1=y_eps_l[layer].reshape(-1),
                f2=y_eps_r[layer].reshape(-1),
            )

            self.angles[layer].append(np.pi - angle)
