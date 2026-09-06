#include "rasterization_api.h"
#include "appearance_params.h"
#include "forward.h"
#include "backward.h"
#include "inference.h"
#include "pruning_scores.h"
#include "torch_utils.h"
#include "rasterization_config.h"
#include "utils.h"
#include <stdexcept>
#include <functional>
#include <tuple>

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int, int, int>
faster_gs::rasterization::forward_wrapper(
    Rasterizer& rasterizer,
    const torch::Tensor& means,
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    const torch::Tensor& opacities,
    const torch::Tensor& precomputed_colors,
    const torch::Tensor& base_colors,
    const torch::Tensor& residual_params,
    const torch::Tensor& mlp_weights,
    const torch::Tensor& w2c,
    const torch::Tensor& cam_position,
    const torch::Tensor& bg_color,
    const int appearance_degree,
    const int width,
    const int height,
    const float focal_x,
    const float focal_y,
    const float center_x,
    const float center_y,
    const float near_plane,
    const float far_plane,
    const bool proper_antialiasing,
    const bool render_base,
    const bool render_residual)
{
    // all optimizable tensors must be passed as contiguous CUDA floating point tensors
    CHECK_INPUT(config::debug, means, "means");
    CHECK_INPUT(config::debug, scales, "scales");
    CHECK_INPUT(config::debug, rotations, "rotations");
    CHECK_INPUT(config::debug, opacities, "opacities");
    if (precomputed_colors.numel() > 0) CHECK_INPUT(config::debug, precomputed_colors, "precomputed_colors");
    if (base_colors.numel() > 0) CHECK_INPUT(config::debug, base_colors, "base_colors");
    if (residual_params.numel() > 0) CHECK_INPUT(config::debug, residual_params, "residual_params");
    if (mlp_weights.numel() > 0) CHECK_INPUT(config::debug, mlp_weights, "mlp_weights");

    const int n_primitives = means.size(0);
    const torch::TensorOptions float_options = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
    const torch::TensorOptions half_options = torch::TensorOptions().dtype(torch::kHalf).device(torch::kCUDA);
    const torch::TensorOptions byte_options = torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA);
    torch::Tensor image = torch::empty({3, height, width}, float_options);
    torch::Tensor mlp_outputs = torch::empty({0}, half_options);
    torch::Tensor fwd_ctx = torch::empty({0}, byte_options);
    torch::Tensor primitive_buffers = torch::empty({0}, byte_options);
    torch::Tensor tile_buffers = torch::empty({0}, byte_options);
    torch::Tensor instance_buffers = torch::empty({0}, byte_options);
    torch::Tensor bucket_buffers = torch::empty({0}, byte_options);
    const std::function<char*(size_t)> resize_primitive_buffers = resize_function_wrapper(primitive_buffers);
    const std::function<char*(size_t)> resize_tile_buffers = resize_function_wrapper(tile_buffers);
    const std::function<char*(size_t)> resize_instance_buffers = resize_function_wrapper(instance_buffers);
    const std::function<char*(size_t)> resize_bucket_buffers = resize_function_wrapper(bucket_buffers);

    const bool precomputed = precomputed_colors.numel() > 0;
    const bool fused_mlp = !precomputed && rasterizer.is_neural();
    if (fused_mlp) {
        rasterizer.upload_params(mlp_weights);
        if (rasterizer.needs_mlp_outputs()) mlp_outputs = torch::empty({3, n_primitives}, half_options);
        const int64_t n_ctx_threads = div_round_up(n_primitives, config::block_size_preprocess) * config::block_size_preprocess;
        fwd_ctx = torch::empty({n_ctx_threads * rasterizer.fwd_ctx_bytes()}, byte_options);
    }
    const AppearanceInputs appearance {
        precomputed ? reinterpret_cast<const float3*>(precomputed_colors.data_ptr<float>()) : nullptr,
        base_colors.numel() > 0 ? reinterpret_cast<const float3*>(base_colors.data_ptr<float>()) : nullptr,
        residual_params.numel() > 0 ? residual_params.data_ptr<float>() : nullptr,
        fused_mlp ? rasterizer.params_jit() : nullptr,
        mlp_outputs.numel() > 0 ? reinterpret_cast<__half*>(mlp_outputs.data_ptr<torch::Half>()) : nullptr,
        fused_mlp ? fwd_ctx.data_ptr<uint8_t>() : nullptr,
    };

    auto [n_instances, n_buckets, instance_primitive_indices_selector] = forward(
        resize_primitive_buffers,
        resize_tile_buffers,
        resize_instance_buffers,
        resize_bucket_buffers,
        precomputed ? rasterizer.precomputed_forward_kernel() : rasterizer.preprocess_forward_kernel(),
        reinterpret_cast<float3*>(means.data_ptr<float>()),
        reinterpret_cast<float3*>(scales.data_ptr<float>()),
        reinterpret_cast<float4*>(rotations.data_ptr<float>()),
        opacities.data_ptr<float>(),
        appearance,
        reinterpret_cast<float4*>(w2c.contiguous().data_ptr<float>()),
        reinterpret_cast<float3*>(cam_position.contiguous().data_ptr<float>()),
        reinterpret_cast<float3*>(bg_color.contiguous().data_ptr<float>()),
        image.data_ptr<float>(),
        n_primitives,
        render_residual ? appearance_degree : 0,  // residual-off renders at degree 0 (base only)
        width,
        height,
        focal_x,
        focal_y,
        center_x,
        center_y,
        near_plane,
        far_plane,
        proper_antialiasing,
        render_base
    );

    return {
        image,
        primitive_buffers, tile_buffers, instance_buffers, bucket_buffers, mlp_outputs, fwd_ctx,
        n_instances, n_buckets, instance_primitive_indices_selector
    };
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
faster_gs::rasterization::backward_wrapper(
    Rasterizer& rasterizer,
    torch::Tensor& densification_info,
    const torch::Tensor& grad_image,
    const torch::Tensor& image,
    const torch::Tensor& means,
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    const torch::Tensor& opacities,
    const torch::Tensor& precomputed_colors,
    const torch::Tensor& base_colors,
    const torch::Tensor& residual_params,
    const torch::Tensor& mlp_weights,
    const torch::Tensor& mlp_outputs,
    const torch::Tensor& fwd_ctx,
    const torch::Tensor& primitive_buffers,
    const torch::Tensor& tile_buffers,
    const torch::Tensor& instance_buffers,
    const torch::Tensor& bucket_buffers,
    const torch::Tensor& w2c,
    const torch::Tensor& cam_position,
    const torch::Tensor& bg_color,
    const int appearance_degree,
    const int width,
    const int height,
    const float focal_x,
    const float focal_y,
    const float center_x,
    const float center_y,
    const float near_plane,
    const float far_plane,
    const bool proper_antialiasing,
    const bool render_base,
    const bool render_residual,
    const int n_instances,
    const int n_buckets,
    const int instance_primitive_indices_selector)
{
    const int n_primitives = means.size(0);
    const torch::TensorOptions float_options = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
    torch::Tensor grad_means = torch::zeros_like(means);
    torch::Tensor grad_scales = torch::zeros_like(scales);
    torch::Tensor grad_rotations = torch::zeros_like(rotations);
    torch::Tensor grad_opacities = torch::zeros_like(opacities);
    torch::Tensor grad_precomputed_colors = torch::zeros_like(precomputed_colors);
    torch::Tensor grad_base_colors = torch::zeros_like(base_colors);
    torch::Tensor grad_residual_params = torch::zeros_like(residual_params);
    torch::Tensor grad_mlp_weights = torch::zeros_like(mlp_weights);
    torch::Tensor grad_mean2d_helper = torch::zeros({n_primitives, 2}, float_options);
    torch::Tensor grad_conic_helper = torch::zeros({3, n_primitives}, float_options);
    torch::Tensor grad_color_helper = torch::zeros({3, n_primitives}, float_options);

    const bool precomputed = precomputed_colors.numel() > 0;
    const bool fused_mlp = !precomputed && rasterizer.is_neural();
    if (fused_mlp) rasterizer.upload_params(mlp_weights);
    const AppearanceInputs appearance {
        precomputed ? reinterpret_cast<const float3*>(precomputed_colors.data_ptr<float>()) : nullptr,
        base_colors.numel() > 0 ? reinterpret_cast<const float3*>(base_colors.data_ptr<float>()) : nullptr,
        residual_params.numel() > 0 ? residual_params.data_ptr<float>() : nullptr,
        fused_mlp ? rasterizer.params_jit() : nullptr,
        mlp_outputs.numel() > 0 ? reinterpret_cast<__half*>(mlp_outputs.data_ptr<torch::Half>()) : nullptr,
        fused_mlp ? fwd_ctx.data_ptr<uint8_t>() : nullptr,
    };
    const AppearanceGradients grad_appearance {
        precomputed ? reinterpret_cast<float3*>(grad_precomputed_colors.data_ptr<float>()) : nullptr,
        precomputed ? nullptr : reinterpret_cast<float3*>(grad_base_colors.data_ptr<float>()),
        precomputed ? nullptr : grad_residual_params.data_ptr<float>(),
        fused_mlp ? reinterpret_cast<__half*>(grad_mlp_weights.data_ptr<torch::Half>()) : nullptr,
    };

    backward(
        precomputed ? rasterizer.precomputed_backward_kernel() : rasterizer.preprocess_backward_kernel(),
        grad_image.contiguous().data_ptr<float>(),
        reinterpret_cast<float3*>(means.data_ptr<float>()),
        reinterpret_cast<float3*>(scales.data_ptr<float>()),
        reinterpret_cast<float4*>(rotations.data_ptr<float>()),
        opacities.data_ptr<float>(),
        appearance,
        reinterpret_cast<float4*>(w2c.contiguous().data_ptr<float>()),
        reinterpret_cast<float3*>(cam_position.contiguous().data_ptr<float>()),
        reinterpret_cast<float3*>(bg_color.contiguous().data_ptr<float>()),
        image.data_ptr<float>(),
        reinterpret_cast<char*>(primitive_buffers.data_ptr()),
        reinterpret_cast<char*>(tile_buffers.data_ptr()),
        reinterpret_cast<char*>(instance_buffers.data_ptr()),
        reinterpret_cast<char*>(bucket_buffers.data_ptr()),
        reinterpret_cast<float2*>(grad_mean2d_helper.data_ptr<float>()),
        grad_conic_helper.data_ptr<float>(),
        grad_color_helper.data_ptr<float>(),
        reinterpret_cast<float3*>(grad_means.data_ptr<float>()),
        reinterpret_cast<float3*>(grad_scales.data_ptr<float>()),
        reinterpret_cast<float4*>(grad_rotations.data_ptr<float>()),
        grad_opacities.data_ptr<float>(),
        grad_appearance,
        densification_info.size(0) > 0 ? densification_info.data_ptr<float>() : nullptr,
        n_primitives,
        n_instances,
        n_buckets,
        instance_primitive_indices_selector,
        fused_mlp ? rasterizer.backward_shmem_bytes() : 0,
        render_residual ? appearance_degree : 0,  // residual-off renders at degree 0 (base only)
        width,
        height,
        focal_x,
        focal_y,
        center_x,
        center_y,
        proper_antialiasing,
        render_base
    );

    if (fused_mlp) grad_mlp_weights = grad_mlp_weights.to(torch::kFloat).mul_(1.0f / config::mlp_grad_loss_scale);

    return {grad_means, grad_scales, grad_rotations, grad_opacities, grad_precomputed_colors, grad_base_colors, grad_residual_params, grad_mlp_weights};
}

torch::Tensor
faster_gs::rasterization::inference_wrapper(
    Rasterizer& rasterizer,
    const torch::Tensor& means,
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    const torch::Tensor& opacities,
    const torch::Tensor& precomputed_colors,
    const torch::Tensor& base_colors,
    const torch::Tensor& residual_params,
    const torch::Tensor& mlp_weights,
    const torch::Tensor& w2c,
    const torch::Tensor& cam_position,
    const torch::Tensor& bg_color,
    const int appearance_degree,
    const int width,
    const int height,
    const float focal_x,
    const float focal_y,
    const float center_x,
    const float center_y,
    const float near_plane,
    const float far_plane,
    const bool proper_antialiasing,
    const bool render_base,
    const bool render_residual,
    const bool to_chw,
    const bool clamp_output)
{
    const int n_primitives = means.size(0);
    const torch::TensorOptions float_options = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
    const torch::TensorOptions byte_options = torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA);
    torch::Tensor image = to_chw ? torch::empty({3, height, width}, float_options) : torch::empty({height, width, 3}, float_options);
    torch::Tensor primitive_buffers = torch::empty({0}, byte_options);
    torch::Tensor tile_buffers = torch::empty({0}, byte_options);
    torch::Tensor instance_buffers = torch::empty({0}, byte_options);
    const std::function<char*(size_t)> resize_primitive_buffers = resize_function_wrapper(primitive_buffers);
    const std::function<char*(size_t)> resize_tile_buffers = resize_function_wrapper(tile_buffers);
    const std::function<char*(size_t)> resize_instance_buffers = resize_function_wrapper(instance_buffers);

    const bool precomputed = precomputed_colors.numel() > 0;
    const bool fused_mlp = !precomputed && rasterizer.is_neural();
    if (fused_mlp) rasterizer.upload_params(mlp_weights);
    const AppearanceInputs appearance {
        precomputed ? reinterpret_cast<const float3*>(precomputed_colors.data_ptr<float>()) : nullptr,
        base_colors.numel() > 0 ? reinterpret_cast<const float3*>(base_colors.data_ptr<float>()) : nullptr,
        residual_params.numel() > 0 ? residual_params.data_ptr<float>() : nullptr,
        fused_mlp ? rasterizer.params_jit() : nullptr,
        nullptr,
        nullptr,
    };

    inference(
        resize_primitive_buffers,
        resize_tile_buffers,
        resize_instance_buffers,
        precomputed ? rasterizer.precomputed_forward_kernel() : rasterizer.preprocess_forward_kernel(),
        reinterpret_cast<float3*>(means.data_ptr<float>()),
        reinterpret_cast<float3*>(scales.data_ptr<float>()),
        reinterpret_cast<float4*>(rotations.data_ptr<float>()),
        opacities.data_ptr<float>(),
        appearance,
        reinterpret_cast<float4*>(w2c.contiguous().data_ptr<float>()),
        reinterpret_cast<float3*>(cam_position.contiguous().data_ptr<float>()),
        reinterpret_cast<float3*>(bg_color.contiguous().data_ptr<float>()),
        image.data_ptr<float>(),
        n_primitives,
        render_residual ? appearance_degree : 0,  // residual-off renders at degree 0 (base only)
        width,
        height,
        focal_x,
        focal_y,
        center_x,
        center_y,
        near_plane,
        far_plane,
        proper_antialiasing,
        render_base,
        to_chw,
        clamp_output
    );

    return image;
}

void
faster_gs::rasterization::pruning_scores_wrapper(
    Rasterizer& rasterizer,
    torch::Tensor& scores,
    const torch::Tensor& means,
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    const torch::Tensor& opacities,
    const torch::Tensor& precomputed_colors,
    const torch::Tensor& base_colors,
    const torch::Tensor& residual_params,
    const torch::Tensor& mlp_weights,
    const torch::Tensor& w2c,
    const torch::Tensor& cam_position,
    const torch::Tensor& bg_color,
    const int appearance_degree,
    const int width,
    const int height,
    const float focal_x,
    const float focal_y,
    const float center_x,
    const float center_y,
    const float near_plane,
    const float far_plane,
    const bool proper_antialiasing,
    const bool render_base,
    const bool render_residual)
{
    const int n_primitives = means.size(0);
    const torch::TensorOptions byte_options = torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA);
    torch::Tensor primitive_buffers = torch::empty({0}, byte_options);
    torch::Tensor tile_buffers = torch::empty({0}, byte_options);
    torch::Tensor instance_buffers = torch::empty({0}, byte_options);
    const std::function<char*(size_t)> resize_primitive_buffers = resize_function_wrapper(primitive_buffers);
    const std::function<char*(size_t)> resize_tile_buffers = resize_function_wrapper(tile_buffers);
    const std::function<char*(size_t)> resize_instance_buffers = resize_function_wrapper(instance_buffers);

    const bool precomputed = precomputed_colors.numel() > 0;
    const bool fused_mlp = !precomputed && rasterizer.is_neural();
    if (fused_mlp) rasterizer.upload_params(mlp_weights);
    const AppearanceInputs appearance {
        precomputed ? reinterpret_cast<const float3*>(precomputed_colors.data_ptr<float>()) : nullptr,
        base_colors.numel() > 0 ? reinterpret_cast<const float3*>(base_colors.data_ptr<float>()) : nullptr,
        residual_params.numel() > 0 ? residual_params.data_ptr<float>() : nullptr,
        fused_mlp ? rasterizer.params_jit() : nullptr,
        nullptr,
        nullptr,
    };

    pruning_scores(
        resize_primitive_buffers,
        resize_tile_buffers,
        resize_instance_buffers,
        precomputed ? rasterizer.precomputed_forward_kernel() : rasterizer.preprocess_forward_kernel(),
        reinterpret_cast<float3*>(means.data_ptr<float>()),
        reinterpret_cast<float3*>(scales.data_ptr<float>()),
        reinterpret_cast<float4*>(rotations.data_ptr<float>()),
        opacities.data_ptr<float>(),
        appearance,
        reinterpret_cast<float4*>(w2c.contiguous().data_ptr<float>()),
        reinterpret_cast<float3*>(cam_position.contiguous().data_ptr<float>()),
        reinterpret_cast<float3*>(bg_color.contiguous().data_ptr<float>()),
        scores.data_ptr<float>(),
        n_primitives,
        render_residual ? appearance_degree : 0,  // residual-off renders at degree 0 (base only)
        width,
        height,
        focal_x,
        focal_y,
        center_x,
        center_y,
        near_plane,
        far_plane,
        proper_antialiasing,
        render_base
    );
}
