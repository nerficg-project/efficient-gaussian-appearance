"""FasterGSVDA/Appearance/SV.py"""

from typing import Callable

import torch

import Framework
from Methods.FasterGSVDA.Appearance.Base import AppearanceModel
from Methods.FasterGSVDA.Appearance.utils import fibonacci_sphere


class SV(AppearanceModel):
    """View-dependent appearance via a soft spherical Voronoi partition of the direction sphere."""

    def __init__(self, appearance_config: Framework.ConfigParameterList) -> None:
        super().__init__(appearance_config)
        self.max_degree = self.appearance_config.SV.N_SITES
        # the reference trains all sites from the first iteration, so the degree schedule starts saturated
        self.active_degree = self.max_degree

    def _initialize_residual_parameters(self, n_primitives: int) -> None:
        """Creates and registers the initial spherical Voronoi site parameters."""
        # raw temperatures start at 0 (exp -> tau = 1); raw site colors start at 0
        residual_params = torch.zeros((n_primitives, self.appearance_config.SV.N_SITES, 7), dtype=torch.float32, device='cuda')
        initial_sites = fibonacci_sphere(self.appearance_config.SV.N_SITES)  # raw sites: the shared unit Fibonacci lattice, like the reference
        residual_params[..., 0:3] = initial_sites
        self._residual_params = torch.nn.Parameter(residual_params)

    def param_groups(self, optimizer_config: Framework.ConfigParameterList) -> list[dict]:
        """Returns the optimizer parameter groups; sites, temperatures, and colors have separate learning rates."""
        # the fused optimizer cycles a learning rate list over the packed [site xyz | tau | color rgb] columns
        residual_lrs = [optimizer_config.LEARNING_RATE_SV_SITES] * 3 + [optimizer_config.LEARNING_RATE_SV_TAUS] + [optimizer_config.LEARNING_RATE_SV_COLORS] * 3
        return [
            {'params': [self._base_colors], 'lr': optimizer_config.LEARNING_RATE_BASE_COLORS, 'name': 'base_colors'},
            {'params': [self._residual_params], 'lr': residual_lrs, 'name': 'residual_params'},
        ]

    def lr_schedulers(self, optimizer_config: Framework.ConfigParameterList) -> dict[str, Callable]:
        """Returns the cosine schedule for the sites and temperatures (site and base colors stay constant, like the reference)."""
        if not optimizer_config.APPEARANCE_COSINE_DECAY.USE:
            return {}
        sites_policy = self._cosine_decay_policy(optimizer_config, optimizer_config.LEARNING_RATE_SV_SITES)
        taus_policy = self._cosine_decay_policy(optimizer_config, optimizer_config.LEARNING_RATE_SV_TAUS)
        colors_lr = optimizer_config.LEARNING_RATE_SV_COLORS

        def residual_schedule(iteration: int) -> list[float]:
            return [sites_policy(iteration)] * 3 + [taus_policy(iteration)] + [colors_lr] * 3

        return {'residual_params': residual_schedule}

    def increase_degree(self) -> None:
        """For Spherical Voronoi, all sites are activated at once."""
        self.active_degree = self.max_degree

    def residual_colors(self, directions: torch.Tensor) -> torch.Tensor:
        """Evaluates the soft spherical Voronoi mixture for all Gaussians (N, 3)."""
        sites = torch.nn.functional.normalize(self._residual_params[:, :self.active_degree, 0:3], dim=-1)
        taus = self._residual_params[:, :self.active_degree, 3].exp()
        distances = torch.linalg.norm(sites - directions[:, None], dim=-1)
        weights = torch.softmax(-taus * distances, dim=-1)
        site_colors = self.residual_activation(self._residual_params[:, :self.active_degree, 4:7])
        return (weights[..., None] * site_colors).sum(dim=1)
