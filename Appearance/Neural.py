"""FasterGSVDA/Appearance/Neural.py"""

from typing import Callable

import torch

import Framework
from Logging import Logger
from Methods.FasterGSVDA.Appearance.Base import AppearanceModel
from Methods.FasterGSVDA.Appearance.utils import frequency_encode, compute_sh_degrees
from Methods.FasterGSVDA.FasterGSVDACudaBackend import initialize_residual_mlp
from Optim.lr_utils import LRDecayPolicy
import Thirdparty.TinyCudaNN as tcnn


class Neural(AppearanceModel):
    """View-dependent appearance via a tiny, direction-conditioned MLP."""

    def __init__(self, appearance_config: Framework.ConfigParameterList) -> None:
        super().__init__(appearance_config)
        sh_degrees = list(self.appearance_config.NEURAL.SH_DEGREES)
        if sh_degrees != sorted(set(sh_degrees)) or any(degree < 0 or degree > 6 for degree in sh_degrees):
            raise Framework.ModelError(f'APPEARANCE.NEURAL.SH_DEGREES must be a strictly ascending list of SH degrees in [0, 6], got {sh_degrees}')
        if self.appearance_config.NEURAL.BASE_INPUT and self.appearance_config.NEURAL.FEATURE_DIM <= 3:
            raise Framework.ModelError(f'APPEARANCE.NEURAL.BASE_INPUT requires FEATURE_DIM > 3 (3 of the {self.appearance_config.NEURAL.FEATURE_DIM} inputs are the base values)')
        self.n_feature_dims = self.appearance_config.NEURAL.FEATURE_DIM
        if self.appearance_config.NEURAL.BASE_INPUT:
            self.n_feature_dims -= 3
        n_encoded_feature_dims = self.appearance_config.NEURAL.FEATURE_DIM
        if self.appearance_config.NEURAL.FEATURE_N_FREQUENCIES > 0:
            n_encoded_feature_dims *= 2 * self.appearance_config.NEURAL.FEATURE_N_FREQUENCIES
        self.max_degree = sum(1 for degree in sh_degrees if degree > 0)
        n_sh_degree_dims = sum(2 * degree + 1 for degree in sh_degrees)
        n_input_dims = n_encoded_feature_dims + n_sh_degree_dims
        if n_input_dims != 32:
            Logger.log_warning(
                f'mlp input width is {n_input_dims} ({n_encoded_feature_dims} encoded features + {n_sh_degree_dims} SH degree values): '
                f'all preset variants are designed for exactly 32 (tcnn pads the input to a multiple of 16)'
            )
        self.residual_mlp = tcnn.NetworkWithInputEncoding(
            n_input_dims=n_input_dims,
            n_output_dims=3,
            encoding_config={'otype': 'Identity'},  # all inputs are assembled explicitly, matching the fused kernels
            network_config={
                'otype': 'FullyFusedMLP',
                'activation': self.appearance_config.NEURAL.HIDDEN_ACTIVATION,
                'output_activation': 'None',
                'n_neurons': self.appearance_config.NEURAL.N_NEURONS,
                'n_hidden_layers': self.appearance_config.NEURAL.N_HIDDEN_LAYERS,
            },
        )
        # zero the final weight matrix (packed last by tcnn) so the residual is exactly 0 when its degree activates (softplus residual activation maps 0 to ln(2)/beta instead)
        self.residual_mlp.params.data[-self.appearance_config.NEURAL.N_NEURONS * ((self.residual_mlp.n_output_dims + 15) // 16 * 16):].zero_()

    def initialize_backend(self) -> None:
        """Configures the rasterizer's jit-compiled kernels and creates the fused-side residual mlp."""
        super().initialize_backend()
        initialize_residual_mlp(
            self.residual_mlp.n_input_dims,
            self.residual_mlp.n_output_dims,
            self.residual_mlp.encoding_config,
            self.residual_mlp.network_config,
            self.appearance_config.NEURAL.BASE_INPUT,
            self.appearance_config.NEURAL.FEATURE_N_FREQUENCIES,
            self.appearance_config.NEURAL.SH_DEGREES,
        )

    def _initialize_residual_parameters(self, n_primitives: int) -> None:
        """Creates and registers zero-initialized per-Gaussian mlp input features."""
        residual_params = torch.zeros((n_primitives, self.n_feature_dims), dtype=torch.float32, device='cuda')
        self._residual_params = torch.nn.Parameter(residual_params)

    def param_groups(self, optimizer_config: Framework.ConfigParameterList) -> list[dict]:
        """Adds the optimizer parameter group of the shared mlp weights (excluded from densification)."""
        # the mlp receives dense gradients from every visible Gaussian each step, so its second moment
        # can track a much shorter horizon than the sparsely updated per-Gaussian groups (beta2 0.99)
        return super().param_groups(optimizer_config) + [
            {'params': [self.residual_mlp.params], 'lr': optimizer_config.LEARNING_RATE_MLP_WEIGHTS, 'betas': (0.9, 0.99), 'name': 'mlp_weights'},
        ]

    def lr_schedulers(self, optimizer_config: Framework.ConfigParameterList) -> dict[str, Callable[[int], float]]:
        """Returns the always-on mlp weight schedule and cosine-decay schedulers for the base colors and features."""
        lr_schedulers = {
            'mlp_weights': LRDecayPolicy(
                lr_init=optimizer_config.LEARNING_RATE_MLP_WEIGHTS,
                lr_final=optimizer_config.LEARNING_RATE_MLP_WEIGHTS * 0.1,
                lr_delay_steps=3_000,
                lr_delay_mult=1e-8,
                max_steps=30_000,
            ),
        }
        if optimizer_config.APPEARANCE_COSINE_DECAY.USE:
            lr_schedulers['base_colors'] = self._cosine_decay_policy(optimizer_config, optimizer_config.LEARNING_RATE_BASE_COLORS)
            lr_schedulers['residual_params'] = self._cosine_decay_policy(optimizer_config, optimizer_config.LEARNING_RATE_RESIDUAL_PARAMS)
        return lr_schedulers

    @property
    def raw_parameters(self) -> dict[str, torch.Tensor]:
        """Returns the raw appearance parameters consumed by the fused rasterizer kernels."""
        return {'base_colors': self._base_colors, 'residual_params': self._residual_params, 'mlp_weights': self.residual_mlp.params}

    def residual_colors(self, directions: torch.Tensor) -> torch.Tensor:
        """Evaluates the residual mlp via tcnn's python bindings for all Gaussians (N, 3)."""
        sh_values = compute_sh_degrees(directions, self.active_degree, self.appearance_config.NEURAL.SH_DEGREES)
        features = self._residual_params
        if self.appearance_config.NEURAL.BASE_INPUT:
            features = torch.cat([features, self._base_colors], dim=-1)
        if self.appearance_config.NEURAL.FEATURE_N_FREQUENCIES > 0:
            features = frequency_encode(features, self.appearance_config.NEURAL.FEATURE_N_FREQUENCIES)
        mlp_outputs = self.residual_mlp(torch.cat([features, sh_values], dim=-1)).float()
        return self.residual_activation(mlp_outputs)
