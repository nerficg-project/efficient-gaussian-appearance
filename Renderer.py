"""FasterGSVDA/Renderer.py"""

import math

import torch

import Framework
from Cameras.Perspective import PerspectiveCamera
from Datasets.Base import BaseDataset
from Datasets.utils import View
from Logging import Logger
from Methods.Base.Renderer import BaseModel
from Methods.Base.Renderer import BaseRenderer
from Methods.FasterGSVDA.Model import FasterGSVDAModel
from Methods.FasterGSVDA.FasterGSVDACudaBackend import diff_rasterize, rasterize, update_pruning_scores, RasterizerSettings


def extract_settings(
    view: View,
    bg_color: torch.Tensor,
    appearance_degree: int,
    proper_antialiasing: bool,
    render_base: bool,
    render_residual: bool,
) -> RasterizerSettings:
    if not isinstance(view.camera, PerspectiveCamera):
        raise Framework.RendererError('FasterGSVDA renderer only supports perspective cameras')
    if view.camera.distortion is not None:
        Logger.log_warning('found distortion parameters that will be ignored by the rasterizer')
    return RasterizerSettings(
        view.w2c,
        view.position,
        bg_color,
        appearance_degree,
        view.camera.width,
        view.camera.height,
        view.camera.focal_x,
        view.camera.focal_y,
        view.camera.center_x,
        view.camera.center_y,
        view.camera.near_plane,
        view.camera.far_plane,
        proper_antialiasing,
        render_base,
        render_residual,
    )


@Framework.Configurable.configure(
    SCALE_MODIFIER=1.0,
    PROPER_ANTIALIASING=False,
    RENDER_BASE_COLOR=True,  # if False, the view-independent base color is omitted from the rendered color
    RENDER_RESIDUAL_COLOR=True,  # if False, the view-dependent residual is omitted from the rendered color
    USE_FUSED_APPEARANCE=True,  # if False, colors are precomputed outside the rasterizer
    FORCE_OPTIMIZED_INFERENCE=False,
)
class FasterGSVDARenderer(BaseRenderer):
    """Wrapper around the rasterization module from 3DGS."""

    def __init__(self, model: 'BaseModel') -> None:
        super().__init__(model, [FasterGSVDAModel])
        if not Framework.config.GLOBAL.GPU_INDICES:
            raise Framework.RendererError('FasterGSVDA renderer not implemented in CPU mode')
        if len(Framework.config.GLOBAL.GPU_INDICES) > 1:
            Logger.log_warning(f'FasterGSVDA renderer not implemented in multi-GPU mode: using GPU {Framework.config.GLOBAL.GPU_INDICES[0]}')

    def render_image(self, view: View, to_chw: bool = False, benchmark: bool = False) -> dict[str, torch.Tensor]:
        """Renders an image for a given view."""
        if benchmark or self.FORCE_OPTIMIZED_INFERENCE:
            return self.render_image_benchmark(view, to_chw=to_chw or benchmark)
        elif self.model.training:
            raise Framework.RendererError('please directly call render_image_training() instead of render_image() during training')
        else:
            return self.render_image_inference(view, to_chw)

    def render_image_training(self, view: View, update_densification_info: bool, bg_color: torch.Tensor) -> torch.Tensor:
        """Renders an image for a given view."""
        image = diff_rasterize(
            means=self.model.gaussians.means,
            scales=self.model.gaussians.raw_scales,
            rotations=self.model.gaussians.raw_rotations,
            opacities=self.model.gaussians.raw_opacities,
            **(self.model.gaussians.raw_appearance_parameters if self.USE_FUSED_APPEARANCE else self.model.gaussians.precomputed_colors(view)),
            densification_info=self.model.gaussians.densification_info if update_densification_info else torch.empty(0),
            rasterizer_settings=extract_settings(
                view, bg_color, self.model.gaussians.appearance.active_degree,
                self.PROPER_ANTIALIASING, True, True,
            ),
        )
        if self.model.ppisp is not None:
            image = self.model.ppisp(image, view)
        return image

    @torch.no_grad()
    def render_image_inference(self, view: View, to_chw: bool = False) -> dict[str, torch.Tensor]:
        """Renders an image for a given view."""
        apply_ppisp = self.model.ppisp is not None and self.RENDER_BASE_COLOR  # PPISP does not make sense for the residual-only visualization
        image = rasterize(
            means=self.model.gaussians.means,
            scales=self.model.gaussians.raw_scales + math.log(max(self.SCALE_MODIFIER, 1e-6)),
            rotations=self.model.gaussians.raw_rotations,
            opacities=self.model.gaussians.raw_opacities,
            **(self.model.gaussians.raw_appearance_parameters if self.USE_FUSED_APPEARANCE else
               self.model.gaussians.precomputed_colors(view, render_base=self.RENDER_BASE_COLOR, render_residual=self.RENDER_RESIDUAL_COLOR)),
            rasterizer_settings=extract_settings(
                view, view.camera.background_color, self.model.gaussians.appearance.active_degree,
                self.PROPER_ANTIALIASING, self.RENDER_BASE_COLOR, self.RENDER_RESIDUAL_COLOR,
            ),
            to_chw=to_chw,
            clamp_output=not apply_ppisp,
        )
        if apply_ppisp:
            image = self.model.ppisp(image, view)
        return {'rgb': image}

    @torch.inference_mode()
    def render_image_benchmark(self, view: View, to_chw: bool = False) -> dict[str, torch.Tensor]:
        """Renders an image for a given view."""
        image = rasterize(
            means=self.model.gaussians.means,
            scales=self.model.gaussians.raw_scales,
            rotations=self.model.gaussians.raw_rotations,
            opacities=self.model.gaussians.raw_opacities,
            **(self.model.gaussians.raw_appearance_parameters if self.USE_FUSED_APPEARANCE else self.model.gaussians.precomputed_colors(view)),
            rasterizer_settings=extract_settings(
                view, view.camera.background_color, self.model.gaussians.appearance.active_degree,
                self.PROPER_ANTIALIASING, True, True,
            ),
            to_chw=to_chw,
            clamp_output=self.model.ppisp is None,
        )
        if self.model.ppisp is not None:
            image = self.model.ppisp(image, view)
        return {'rgb': image}

    def ppisp_controller_distillation(self, view: View) -> torch.Tensor:
        """Renders an image for a given view where only the PPISP module will receive gradients."""
        image = rasterize(
            means=self.model.gaussians.means,
            scales=self.model.gaussians.raw_scales,
            rotations=self.model.gaussians.raw_rotations,
            opacities=self.model.gaussians.raw_opacities,
            **(self.model.gaussians.raw_appearance_parameters if self.USE_FUSED_APPEARANCE else self.model.gaussians.precomputed_colors(view)),
            rasterizer_settings=extract_settings(
                view, view.camera.background_color, self.model.gaussians.appearance.active_degree,
                self.PROPER_ANTIALIASING, True, True,
            ),
            to_chw=True,
            clamp_output=False,
        )
        image = self.model.ppisp(image, view)
        return image

    @torch.inference_mode()
    def compute_pruning_scores(self, dataset: BaseDataset) -> torch.Tensor:
        """Computes the pruning scores for the current dataset."""
        scores = torch.zeros(self.model.gaussians.means.shape[0], device=self.model.gaussians.means.device, dtype=torch.float32)
        for view in dataset:
            update_pruning_scores(
                scores=scores,
                means=self.model.gaussians.means,
                scales=self.model.gaussians.raw_scales,
                rotations=self.model.gaussians.raw_rotations,
                opacities=self.model.gaussians.raw_opacities,
                **(self.model.gaussians.raw_appearance_parameters if self.USE_FUSED_APPEARANCE else self.model.gaussians.precomputed_colors(view)),
                rasterizer_settings=extract_settings(
                    view, view.camera.background_color, self.model.gaussians.appearance.active_degree,
                    self.PROPER_ANTIALIASING, True, True,
                ),
            )
        return scores

    def postprocess_outputs(self, outputs: dict[str, torch.Tensor], *_) -> dict[str, torch.Tensor]:
        """Postprocesses the model outputs, returning tensors of shape 3xHxW."""
        return {'rgb': outputs['rgb']}
