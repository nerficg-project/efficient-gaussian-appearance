"""FasterGSVDA/Appearance/NASGabor.py"""

import math
from typing import Callable

import torch

import Framework
from Methods.FasterGSVDA.Appearance.Base import AppearanceModel
from Optim.knn_utils import compute_knn_distances


class NASGabor(AppearanceModel):
    """View-dependent appearance via normalized anisotropic spherical Gabor lobes."""

    def __init__(self, appearance_config: Framework.ConfigParameterList) -> None:
        super().__init__(appearance_config)
        self.max_degree = self.appearance_config.NASGABOR.N_LOBES

    def _initialize_residual_parameters(self, n_primitives: int) -> None:
        """Creates and registers the initial NASGabor lobe parameters."""
        residual_params = torch.empty((n_primitives, self.appearance_config.NASGABOR.N_LOBES, 9), dtype=torch.float32, device='cuda')
        residual_params[..., 0:3] = 0.0            # lobe frame angles: tanh -> cos(theta) = cos(phi) = cos(tau) = 0
        residual_params[..., 3:5] = math.log(0.5)  # sharpness lambda and anisotropy a: exp -> 0.5
        residual_params[..., 5] = -1.6             # Gabor frequency: (tanh(-1.6) + 1) * 20 -> k = 1.57
        residual_params[..., 6:9] = 0.5            # rgb lobe weights: amplitude tanh(0.5) = 0.46 with the reference tanh residual activation
        self._residual_params = torch.nn.Parameter(residual_params)

    def auto_scale_learning_rates(self, optimizer_config: Framework.ConfigParameterList, camera_centers: torch.Tensor) -> None:
        """Scales the lobe parameter learning rate by the training view spacing (part of the reference recipe, always active)."""
        n_neighbors = max(1, min(3, camera_centers.shape[0] - 1))
        mean_knn_distance = compute_knn_distances(camera_centers, n_neighbors=n_neighbors).mean().item()  # FIXME: should use simple_knn when n_neighbors=3
        optimizer_config.LEARNING_RATE_RESIDUAL_PARAMS *= (0.3536 / mean_knn_distance) ** 2

    def lr_schedulers(self, optimizer_config: Framework.ConfigParameterList) -> dict[str, Callable[[int], float]]:
        """Returns cosine-decay schedulers for the base color and lobe parameter learning rates."""
        if not optimizer_config.APPEARANCE_COSINE_DECAY.USE:
            return {}
        return {
            'base_colors': self._cosine_decay_policy(optimizer_config, optimizer_config.LEARNING_RATE_BASE_COLORS),
            'residual_params': self._cosine_decay_policy(optimizer_config, optimizer_config.LEARNING_RATE_RESIDUAL_PARAMS),
        }

    def residual_colors(self, directions: torch.Tensor) -> torch.Tensor:
        """Evaluates the active NASGabor lobes for all Gaussians (N, 3)."""
        # right-handed lobe frame; only x and z are needed to evaluate a lobe
        cos_theta, cos_phi, cos_tau = self._residual_params[:, :self.active_degree, 0:3].tanh().clamp(-0.999999, 0.999999).unbind(dim=-1)
        sin_theta = torch.sqrt(1.0 - cos_theta * cos_theta)
        sin_phi = torch.sqrt(1.0 - cos_phi * cos_phi)
        sin_tau = torch.sqrt(1.0 - cos_tau * cos_tau)
        frame_x = torch.stack([
            cos_theta * cos_phi * cos_tau - sin_theta * sin_tau,
            sin_theta * cos_phi * cos_tau + cos_theta * sin_tau,
            -sin_phi * cos_tau,
        ], dim=-1)
        frame_z = torch.stack([cos_theta * sin_phi, sin_theta * sin_phi, cos_phi], dim=-1)
        sharpness = self._residual_params[:, :self.active_degree, 3].exp().clamp_max(1e4)
        anisotropy = self._residual_params[:, :self.active_degree, 4].exp().clamp_max(1e4)
        frequency = (self._residual_params[:, :self.active_degree, 5].tanh() + 1.0) * 20.0
        v_z = (directions[:, None] * frame_z).sum(dim=-1)
        v_x = (directions[:, None] * frame_x).sum(dim=-1)
        v_z_clamped = v_z.clamp(-0.99999988, 0.99999988)
        K = 0.5 * (v_z_clamped + 1.0)
        K_e = 5e-6 + anisotropy * v_x * v_x / (1.0 - v_z_clamped * v_z_clamped)
        E = K ** K_e
        N = sharpness * torch.sqrt(1.0 + anisotropy) / (2.0 * math.pi * (1.0 + 1e-8 - torch.exp(-2.0 * sharpness)))
        G = 0.5 * (1.0 + torch.cos(frequency * v_x))
        p = torch.exp(2.0 * sharpness * (E * K - 1.0))
        pdf = torch.where(v_z >= 0.99999988, 1.0, torch.where(v_z > -0.99999988, p * E * G * N, 0.0))
        lobe_colors = self.residual_activation(self._residual_params[:, :self.active_degree, 6:9])
        return (pdf[..., None] * lobe_colors).sum(dim=1)
