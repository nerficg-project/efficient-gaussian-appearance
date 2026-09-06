"""FasterGSVDA/Appearance/SH.py"""

from typing import Callable

import torch
import numpy as np

import Framework
from Methods.FasterGSVDA.Appearance.Base import AppearanceModel
from Methods.FasterGSVDA.Appearance.utils import SH_C0


class SH(AppearanceModel):
    """Explicit view-dependent appearance via spherical harmonics."""

    def __init__(self, appearance_config: Framework.ConfigParameterList) -> None:
        super().__init__(appearance_config)
        self.max_degree = self.appearance_config.SH.SH_DEGREE

    def _initialize_residual_parameters(self, n_primitives: int) -> None:
        """Creates and registers zero-initialized coefficients for the view-dependent SH degrees."""
        residual_params = torch.zeros((n_primitives, (self.appearance_config.SH.SH_DEGREE + 1) ** 2 - 1, 3), dtype=torch.float32, device='cuda')
        self._residual_params = torch.nn.Parameter(residual_params)

    def lr_schedulers(self, optimizer_config: Framework.ConfigParameterList) -> dict[str, Callable[[int], float]]:
        """Returns cosine-decay schedulers for the base color and SH coefficient learning rates."""
        if not optimizer_config.APPEARANCE_COSINE_DECAY.USE:
            return {}
        return {
            'base_colors': self._cosine_decay_policy(optimizer_config, optimizer_config.LEARNING_RATE_BASE_COLORS),
            'residual_params': self._cosine_decay_policy(optimizer_config, optimizer_config.LEARNING_RATE_RESIDUAL_PARAMS),
        }

    def residual_colors(self, directions: torch.Tensor) -> torch.Tensor:
        """Evaluates the band sum beyond the constant band for all Gaussians (N, 3)."""
        x, y, z = directions[:, :, None].unbind(dim=1)
        result = -0.48860251190291987 * y * self._residual_params[:, 0] \
               + 0.48860251190291987 * z * self._residual_params[:, 1] \
               - 0.48860251190291987 * x * self._residual_params[:, 2]
        if self.active_degree > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, xz, yz = x * y, x * z, y * z
            result = result + 1.0925484305920792 * xy * self._residual_params[:, 3] \
                            - 1.0925484305920792 * yz * self._residual_params[:, 4] \
                            + (0.94617469575755997 * zz - 0.31539156525251999) * self._residual_params[:, 5] \
                            - 1.0925484305920792 * xz * self._residual_params[:, 6] \
                            + 0.54627421529603959 * (xx - yy) * self._residual_params[:, 7]
            if self.active_degree > 2:
                result = result + y * (0.59004358992664352 * yy - 1.7701307697799304 * xx) * self._residual_params[:, 8] \
                                + 2.8906114426405538 * xy * z * self._residual_params[:, 9] \
                                + y * (0.45704579946446572 - 2.2852289973223288 * zz) * self._residual_params[:, 10] \
                                + z * (1.865881662950577 * zz - 1.1195289977703462) * self._residual_params[:, 11] \
                                + x * (0.45704579946446572 - 2.2852289973223288 * zz) * self._residual_params[:, 12] \
                                + 1.4453057213202769 * z * (xx - yy) * self._residual_params[:, 13] \
                                + x * (1.7701307697799304 * yy - 0.59004358992664352 * xx) * self._residual_params[:, 14]
        return self.residual_activation(result)

    @torch.no_grad()
    def ply_attributes(self) -> tuple[np.ndarray, list[str]]:
        """Returns the SH coefficients in the standard 3DGS layout."""
        # non-standard activation setups use the baked base-only export
        if self.appearance_config.BASE_ACTIVATION != 'none' or self.appearance_config.RESIDUAL_ACTIVATION != 'none' or self.appearance_config.COLOR_ACTIVATION != 'relu':
            return super().ply_attributes()
        sh_0 = self._base_colors.detach().div(SH_C0).contiguous().cpu().numpy()
        sh_rest = self._residual_params.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        attribute_names = (
              ['f_dc_0', 'f_dc_1', 'f_dc_2']                     # 0-th SH degree coefficients
            + [f'f_rest_{i}' for i in range(sh_rest.shape[-1])]  # remaining SH degree coefficients
        )
        return np.concatenate((sh_0, sh_rest), axis=1), attribute_names
