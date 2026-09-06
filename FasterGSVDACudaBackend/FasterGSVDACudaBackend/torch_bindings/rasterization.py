from typing import NamedTuple, Any
import torch
from torch.autograd.function import once_differentiable

from FasterGSVDACudaBackend import _C

# module-level rasterizer instance owning the jit-compiled kernels and tcnn modules
_rasterizer = _C.Rasterizer()


def initialize_appearance(
    appearance_model: str,
    max_degree: int,
    base_activation: str,
    residual_activation: str,
    color_activation: str,
    direction_gradient: bool,
) -> None:
    """Configures the rasterizer's jit-compiled kernels for the given appearance model."""
    # ids match the backend enums (see appearance.h)
    appearance_model_ids = {'sh': 1, 'sv': 2, 'nasg': 3, 'nasgabor': 4, 'neural': 5}  # 0 is the internal precomputed-colors variant
    base_activation_ids = {'none': 0, 'exp': 1}
    residual_activation_ids = {'none': 0, 'tanh': 1, 'softplus': 2}
    color_activation_ids = {'none': 0, 'relu': 1, 'softplus': 2, 'sigmoid': 3, 'hardsigmoid': 4, 'satexp': 5}
    _rasterizer.set_appearance(
        appearance_model_ids[appearance_model],
        max_degree,
        base_activation_ids[base_activation],
        residual_activation_ids[residual_activation],
        color_activation_ids[color_activation],
        direction_gradient
    )


def initialize_residual_mlp(
    n_input_dims: int,
    n_output_dims: int,
    encoding_config: dict[str, Any],
    network_config: dict[str, Any],
    base_input: bool,
    n_frequencies: int,
    sh_degrees: list[int],
) -> None:
    """Creates the tcnn residual mlp from a tinycudann-style configuration (requires initialize_appearance('neural', ...) first)."""
    sh_degree_mask = sum(1 << degree for degree in sh_degrees)
    _rasterizer.set_residual_mlp(n_input_dims, n_output_dims, encoding_config, network_config, base_input, n_frequencies, sh_degree_mask)


class RasterizerSettings(NamedTuple):
    w2c: torch.Tensor  # affine transformation from model/world space to view space
    cam_position: torch.Tensor  # camera position in world space
    bg_color: torch.Tensor  # background color in RGB format
    appearance_degree: int  # appearance degree used for rendering
    width: int  # width of the image plane in pixels
    height: int  # height of the image plane in pixels
    focal_x: float  # focal length in x direction in pixels
    focal_y: float  # focal length in y direction in pixels
    center_x: float  # x coordinate of the image center in pixels (positive -> right)
    center_y: float  # y coordinate of the image center in pixels (positive -> down)
    near_plane: float  # near clipping plane distance
    far_plane: float  # far clipping plane distance
    proper_antialiasing: bool  # whether to use proper antialiasing
    render_base: bool  # whether the view-independent base color contributes to the rendered color
    render_residual: bool  # whether the view-dependent appearance term contributes (if False, the appearance degree is treated as 0)

    def as_tuple(self) -> tuple:
        return (
            self.w2c,
            self.cam_position,
            self.bg_color,
            self.appearance_degree,
            self.width,
            self.height,
            self.focal_x,
            self.focal_y,
            self.center_x,
            self.center_y,
            self.near_plane,
            self.far_plane,
            self.proper_antialiasing,
            self.render_base,
            self.render_residual,
        )


class _Rasterize(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        means: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        opacities: torch.Tensor,
        precomputed_colors: torch.Tensor,
        base_colors: torch.Tensor,
        residual_params: torch.Tensor,
        mlp_weights: torch.Tensor,
        densification_info: torch.Tensor,
        rasterizer_settings: RasterizerSettings,
    ) -> torch.Tensor:
        mlp_weights_half = mlp_weights.detach().half().contiguous() if mlp_weights.numel() > 0 else torch.empty(0, dtype=torch.float16, device='cuda')
        (
            image,
            primitive_buffers, tile_buffers, instance_buffers, bucket_buffers, mlp_outputs, fwd_ctx,
            n_instances, n_buckets, instance_primitive_indices_selector
        ) = _C.forward(
            _rasterizer,
            means,
            scales,
            rotations,
            opacities,
            precomputed_colors,
            base_colors,
            residual_params,
            mlp_weights_half,
            *rasterizer_settings.as_tuple(),
        )
        ctx.rasterizer_settings = rasterizer_settings
        ctx.buffer_state = (n_instances, n_buckets, instance_primitive_indices_selector)
        ctx.save_for_backward(
            image,
            means,
            scales,
            rotations,
            opacities,
            precomputed_colors,
            base_colors,
            residual_params,
            mlp_weights_half,
            mlp_outputs,
            fwd_ctx,
            primitive_buffers,
            tile_buffers,
            instance_buffers,
            bucket_buffers,
        )
        ctx.densification_info = densification_info
        ctx.mark_non_differentiable(densification_info)
        return image

    @staticmethod
    @once_differentiable
    def backward(
        ctx: Any,
        grad_image: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, None, None]:
        settings = ctx.rasterizer_settings
        (
            grad_means, grad_scales, grad_rotations, grad_opacities,
            grad_precomputed_colors, grad_base_colors, grad_residual_params, grad_mlp_weights
        ) = _C.backward(
            _rasterizer,
            ctx.densification_info,
            grad_image,
            *ctx.saved_tensors,
            *settings.as_tuple(),
            *ctx.buffer_state,
        )
        return (
            grad_means,
            grad_scales,
            grad_rotations,
            grad_opacities,
            grad_precomputed_colors if grad_precomputed_colors.numel() > 0 else None,
            grad_base_colors if grad_base_colors.numel() > 0 else None,
            grad_residual_params if grad_residual_params.numel() > 0 else None,
            grad_mlp_weights if grad_mlp_weights.numel() > 0 else None,
            None,  # densification_info
            None,  # rasterizer_settings
        )


def diff_rasterize(
    means: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    opacities: torch.Tensor,
    densification_info: torch.Tensor,
    rasterizer_settings: RasterizerSettings,
    precomputed_colors: torch.Tensor | None = None,
    base_colors: torch.Tensor | None = None,
    residual_params: torch.Tensor | None = None,
    mlp_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    return _Rasterize.apply(
        means,
        scales,
        rotations,
        opacities,
        precomputed_colors if precomputed_colors is not None else torch.empty(0, dtype=torch.float32, device='cuda'),
        base_colors if base_colors is not None else torch.empty(0, dtype=torch.float32, device='cuda'),
        residual_params if residual_params is not None else torch.empty(0, dtype=torch.float32, device='cuda'),
        mlp_weights if mlp_weights is not None else torch.empty(0, dtype=torch.float32, device='cuda'),
        densification_info,
        rasterizer_settings,
    )


def rasterize(
    means: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    opacities: torch.Tensor,
    rasterizer_settings: RasterizerSettings,
    to_chw: bool,
    clamp_output: bool,
    precomputed_colors: torch.Tensor | None = None,
    base_colors: torch.Tensor | None = None,
    residual_params: torch.Tensor | None = None,
    mlp_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    return _C.inference(
        _rasterizer,
        means,
        scales,
        rotations,
        opacities,
        precomputed_colors if precomputed_colors is not None else torch.empty(0, dtype=torch.float32, device='cuda'),
        base_colors if base_colors is not None else torch.empty(0, dtype=torch.float32, device='cuda'),
        residual_params if residual_params is not None else torch.empty(0, dtype=torch.float32, device='cuda'),
        mlp_weights.detach().half().contiguous() if mlp_weights is not None else torch.empty(0, dtype=torch.float16, device='cuda'),
        *rasterizer_settings.as_tuple(),
        to_chw,
        clamp_output,
    )


def update_pruning_scores(
    scores: torch.Tensor,
    means: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    opacities: torch.Tensor,
    rasterizer_settings: RasterizerSettings,
    precomputed_colors: torch.Tensor | None = None,
    base_colors: torch.Tensor | None = None,
    residual_params: torch.Tensor | None = None,
    mlp_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    return _C.pruning_scores(
        _rasterizer,
        scores,
        means,
        scales,
        rotations,
        opacities,
        precomputed_colors if precomputed_colors is not None else torch.empty(0, dtype=torch.float32, device='cuda'),
        base_colors if base_colors is not None else torch.empty(0, dtype=torch.float32, device='cuda'),
        residual_params if residual_params is not None else torch.empty(0, dtype=torch.float32, device='cuda'),
        mlp_weights.detach().half().contiguous() if mlp_weights is not None else torch.empty(0, dtype=torch.float16, device='cuda'),
        *rasterizer_settings.as_tuple(),
    )
