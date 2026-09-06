"""FasterGSVDA/Appearance/Base.py

Base class for the appearance models. Each one owns the appearance-specific per-Gaussian parameters,
their initialization, optimizer setup, degree schedule, the rasterizer input mapping, and the ply export.
During densification and pruning the Gaussians module hands masks/indices to the model's
select/copy/prune/sort/extend parameter methods, which manage the appearance optimizer groups internally.
"""

from typing import Callable
from functools import partial

import torch
import numpy as np

import Framework
from Logging import Logger
from Datasets.utils import View
from Methods.FasterGSVDA.Appearance.utils import SH_C0, base_activation, inverse_base_activation, \
    residual_activation, color_activation, inverse_color_activation
from Methods.FasterGSVDA.FasterGSVDACudaBackend import FusedAdam, initialize_appearance
from Optim.adam_utils import extend_param_groups, prune_param_groups, reset_state, sort_param_groups
from Optim.lr_utils import CosineDecayPolicy
import Thirdparty.TinyCudaNN as tcnn


class AppearanceModel(torch.nn.Module):
    """Base class for appearance models."""

    # names of the per-Gaussian optimizer groups, used to exclude shared groups (e.g., mlp_weights) from densification
    _per_gaussian_params = ['base_colors', 'residual_params']

    def __init__(self, appearance_config: Framework.ConfigParameterList) -> None:
        super().__init__()
        self.name = type(self).__name__.lower()
        self.appearance_config = appearance_config
        if self.appearance_config.BASE_ACTIVATION not in ('none', 'exp'):
            raise Framework.ModelError(f'MODEL.APPEARANCE.BASE_ACTIVATION must be one of none/exp, got {self.appearance_config.BASE_ACTIVATION}')
        if self.appearance_config.RESIDUAL_ACTIVATION not in ('none', 'tanh', 'softplus'):
            raise Framework.ModelError(f'MODEL.APPEARANCE.RESIDUAL_ACTIVATION must be one of none/tanh/softplus, got {self.appearance_config.RESIDUAL_ACTIVATION}')
        if self.appearance_config.COLOR_ACTIVATION not in ('none', 'relu', 'softplus', 'sigmoid', 'hardsigmoid', 'satexp'):
            raise Framework.ModelError(f'MODEL.APPEARANCE.COLOR_ACTIVATION must be one of none/relu/softplus/sigmoid/hardsigmoid/satexp, got {self.appearance_config.COLOR_ACTIVATION}')
        if self.appearance_config.COLOR_ACTIVATION == 'satexp' and (self.appearance_config.BASE_ACTIVATION != 'exp' or self.appearance_config.RESIDUAL_ACTIVATION != 'softplus'):
            Logger.log_warning('COLOR_ACTIVATION satexp expects BASE_ACTIVATION exp and RESIDUAL_ACTIVATION softplus: signed base/residual terms can produce negative colors')
        if self.appearance_config.BASE_ACTIVATION == 'exp' and self.appearance_config.COLOR_ACTIVATION != 'satexp':
            Logger.log_warning(f'BASE_ACTIVATION exp (non-negative intensity) cannot darken below the 0.5 gray value of the 0-centered {self.appearance_config.COLOR_ACTIVATION} color activation')
        # the activation attributes are the configured callables; their names remain accessible via appearance_config
        self.base_activation = partial(base_activation, activation=self.appearance_config.BASE_ACTIVATION)
        self.inverse_base_activation = partial(inverse_base_activation, activation=self.appearance_config.BASE_ACTIVATION)
        self.residual_activation = partial(residual_activation, activation=self.appearance_config.RESIDUAL_ACTIVATION)
        self.color_activation = partial(color_activation, activation=self.appearance_config.COLOR_ACTIVATION)
        self.inverse_color_activation = partial(inverse_color_activation, activation=self.appearance_config.COLOR_ACTIVATION)
        self.register_parameter('_base_colors', None)
        self.register_parameter('_residual_params', None)
        self.max_degree = 0
        self.active_degree = 0
        self.optimizer = None
        self._lr_schedulers = {}

    @property
    def raw_parameters(self) -> dict[str, torch.Tensor]:
        """Returns the raw appearance parameters consumed by the fused rasterizer kernels."""
        return {'base_colors': self._base_colors, 'residual_params': self._residual_params}

    @property
    def base_colors(self) -> torch.Tensor:
        """Returns the base colors after the configured base activation (N, 3)."""
        return self.base_activation(self._base_colors)

    def residual_colors(self, directions: torch.Tensor) -> torch.Tensor:
        """Evaluates the view-dependent color residual for the given unit view directions (N, 3)."""
        raise NotImplementedError

    def precomputed_colors(
        self,
        means: torch.Tensor,
        view: View,
        render_base: bool = True,
        render_residual: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Returns the primitive colors evaluated outside the rasterizer (reference/debug path)."""
        return {'precomputed_colors': self(means, view, render_base, render_residual).contiguous()}

    def forward(
        self,
        means: torch.Tensor,
        view: View,
        render_base: bool = True,
        render_residual: bool = True,
    ) -> torch.Tensor:
        """Computes the primitive colors as activation(base + residual) for all Gaussians (N, 3)."""
        # early exit if view-dependent residual is disabled
        if not render_residual or self.active_degree == 0:
            return self.color_activation(self.base_colors) if render_base else torch.zeros_like(self._base_colors)

        # compute view directions
        if not self.appearance_config.DIRECTION_GRADIENT:
            means = means.detach()
        view_directions = torch.nn.functional.normalize(means - view.position)

        # compute the full colors as activation(base + residual)
        base_colors = self.base_colors
        full_colors = self.color_activation(base_colors + self.residual_colors(view_directions))

        # compute residual-only visualization as abs(activation(base + residual) - activation(base))
        if not render_base:
            return (full_colors - self.color_activation(base_colors)).abs()

        return full_colors

    def increase_degree(self) -> None:
        """Activates the next appearance degree."""
        if self.active_degree < self.max_degree:
            self.active_degree += 1

    def get_extra_state(self) -> dict:
        """Stores the appearance schedule position in the checkpoint's state dict."""
        return {'active_degree': self.active_degree}

    def set_extra_state(self, state: dict) -> None:
        """Restores the appearance schedule position from a checkpoint."""
        self.active_degree = state['active_degree']

    def initialize_backend(self) -> None:
        """Configures the rasterizer's jit-compiled kernels for this appearance model."""
        initialize_appearance(
            self.name,
            self.max_degree,
            self.appearance_config.BASE_ACTIVATION,
            self.appearance_config.RESIDUAL_ACTIVATION,
            self.appearance_config.COLOR_ACTIVATION,
            self.appearance_config.DIRECTION_GRADIENT,
        )

    def initialize_parameters(self, n_primitives: int, rgbs: torch.Tensor | None) -> None:
        """Creates and registers the initial appearance parameters for the given point cloud colors."""
        rgbs = torch.full((n_primitives, 3), fill_value=0.5, dtype=torch.float32, device='cuda') if rgbs is None else rgbs.cuda()
        base_colors = self.inverse_base_activation(self.inverse_color_activation(rgbs))
        self._base_colors = torch.nn.Parameter(base_colors.contiguous())
        self._initialize_residual_parameters(n_primitives)

    def _initialize_residual_parameters(self, n_primitives: int) -> None:
        """Creates and registers the initial residual features."""
        raise NotImplementedError

    def auto_scale_learning_rates(self, optimizer_config: Framework.ConfigParameterList, camera_centers: torch.Tensor) -> None:
        """Adjusts appearance learning rates based on the training view distribution, before the optimizer is created."""
        return

    def param_groups(self, optimizer_config: Framework.ConfigParameterList) -> list[dict]:
        """Returns the optimizer parameter groups for the appearance parameters."""
        return [
            {'params': [self._base_colors], 'lr': optimizer_config.LEARNING_RATE_BASE_COLORS, 'name': 'base_colors'},
            {'params': [self._residual_params], 'lr': optimizer_config.LEARNING_RATE_RESIDUAL_PARAMS, 'name': 'residual_params'},
        ]

    def lr_schedulers(self, optimizer_config: Framework.ConfigParameterList) -> dict[str, Callable[[int], float]]:
        """Returns the lr schedulers by optimizer group name; groups without a scheduler keep their constant lr."""
        raise NotImplementedError

    @staticmethod
    def _cosine_decay_policy(optimizer_config: Framework.ConfigParameterList, lr_init: float) -> Callable[[int], float]:
        """Builds a cosine-decay policy for the given initial learning rate from the shared APPEARANCE_COSINE_DECAY schedule."""
        return CosineDecayPolicy(
            lr_init=lr_init,
            lr_final=optimizer_config.APPEARANCE_COSINE_DECAY.FINAL_FACTOR * lr_init,
            start_iteration=optimizer_config.APPEARANCE_COSINE_DECAY.START_ITERATION,
            end_iteration=optimizer_config.APPEARANCE_COSINE_DECAY.MAX_STEPS,
            lr_warmup=optimizer_config.APPEARANCE_COSINE_DECAY.WARMUP_FACTOR * lr_init,
        )

    def setup_optimizer(self, optimizer_config: Framework.ConfigParameterList) -> None:
        """Creates the optimizer and learning rate schedulers for the appearance parameters."""
        self.optimizer = FusedAdam(self.param_groups(optimizer_config), lr=0.0, eps=1e-15)
        self._lr_schedulers = self.lr_schedulers(optimizer_config)

    def update_learning_rate(self, iteration: int) -> None:
        """Computes the current appearance learning rates for the given iteration."""
        for param_group in self.optimizer.param_groups:
            if (scheduler := self._lr_schedulers.get(param_group['name'])) is not None:
                param_group['lr'] = scheduler(iteration)

    def optimizer_step(self) -> None:
        """Performs an optimizer step on the appearance parameters."""
        self.optimizer.step()
        self.optimizer.zero_grad()

    def _assign_parameters(self, param_groups: dict[str, torch.nn.Parameter]) -> None:
        """Assign the given Gaussian appearance parameters."""
        self._base_colors = param_groups['base_colors']
        self._residual_params = param_groups['residual_params']

    def prune_parameters(self, valid_mask: torch.Tensor) -> None:
        """Keeps only the appearance parameters of the given Gaussians, rebuilding the optimizer groups."""
        self._assign_parameters(prune_param_groups(self.optimizer, valid_mask, group_names=self._per_gaussian_params))

    def sort_parameters(self, ordering: torch.Tensor) -> None:
        """Applies the given ordering to the appearance parameters, rebuilding the optimizer groups."""
        self._assign_parameters(sort_param_groups(self.optimizer, ordering, group_names=self._per_gaussian_params))

    def select_parameters(self, selector: torch.Tensor, n_copies: int = 1) -> dict[str, torch.Tensor]:
        """Returns appearance parameters of the selected Gaussians (boolean mask or index tensor), repeated n_copies times."""
        base_colors = self._base_colors[selector]
        residual_params = self._residual_params[selector]
        if n_copies > 1:
            base_colors = base_colors.expand(n_copies, -1, -1).flatten(end_dim=1)
            residual_params = residual_params.expand(n_copies, *([-1] * residual_params.dim())).flatten(end_dim=1)
        return {'base_colors': base_colors, 'residual_params': residual_params}

    def copy_parameters(self, target_indices: torch.Tensor, source_indices: torch.Tensor) -> None:
        """Copies the appearance parameters between the given Gaussians in place."""
        self._base_colors[target_indices] = self._base_colors[source_indices]
        self._residual_params[target_indices] = self._residual_params[source_indices]

    def extend_parameters(self, *new_parameters: dict[str, torch.Tensor]) -> None:
        """Appends appearance parameters for new Gaussians (concatenating multiple selections), rebuilding the optimizer groups."""
        merged = new_parameters[0] if len(new_parameters) == 1 else \
            {name: torch.cat([parameters[name] for parameters in new_parameters]) for name in new_parameters[0]}
        self._assign_parameters(extend_param_groups(self.optimizer, merged))

    def reset_optimizer_state(self, indices: torch.Tensor) -> None:
        """Resets the optimizer state of the appearance parameters for the given Gaussians."""
        reset_state(self.optimizer, group_names=self._per_gaussian_params, indices=indices)

    def teardown_optimizer(self) -> None:
        """Clears any leftover gradients and deletes the optimizer."""
        self.optimizer.zero_grad()
        self.optimizer = None

    @torch.no_grad()
    def ply_attributes(self) -> tuple[np.ndarray, list[str]]:
        """Returns the appearance model's ply attribute columns (N, C) and their names."""
        Logger.log_warning('ply export contains only the base color, the view-dependent residual is not included')
        # most viewers reconstruct the diffuse color as max(0.5 + SH_C0 * f_dc, 0)
        base_color = self.color_activation(self.base_colors)
        sh_0 = ((base_color - 0.5) / SH_C0).contiguous().cpu().numpy()
        return sh_0, ['f_dc_0', 'f_dc_1', 'f_dc_2']  # 0-th SH degree coefficients
