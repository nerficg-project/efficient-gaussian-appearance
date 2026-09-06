#include "densification_api.h"
#include "mcmc.h"
#include "densification_config.h"
#include "helper_math.h"
#include "torch_utils.h"
#include <tuple>
#include <cstdint>

std::tuple<torch::Tensor, torch::Tensor>
faster_gs::densification::relocation_wrapper(
    const torch::Tensor& old_opacities,
    const torch::Tensor& old_scales,
    const torch::Tensor& n_samples_per_primitive)
{
    const int n_primitives = old_opacities.size(0);
    const torch::TensorOptions float_options = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
    torch::Tensor new_opacities = torch::empty({n_primitives, 1}, float_options);
    torch::Tensor new_scales = torch::empty({n_primitives, 3}, float_options);

    relocation_adjustment(
        old_opacities.contiguous().data_ptr<float>(),
        reinterpret_cast<float3*>(old_scales.contiguous().data_ptr<float>()),
        n_samples_per_primitive.contiguous().data_ptr<int64_t>(),
        new_opacities.data_ptr<float>(),
        reinterpret_cast<float3*>(new_scales.data_ptr<float>()),
        n_primitives
    );

    return std::make_tuple(new_opacities, new_scales);
}

void
faster_gs::densification::add_noise_wrapper(
    const torch::Tensor& raw_scales,
    const torch::Tensor& raw_rotations,
    const torch::Tensor& raw_opacities,
    const torch::Tensor& random_samples,
    torch::Tensor& means,
    const float current_lr)
{
    // tensors must be contiguous CUDA float tensors
    CHECK_INPUT(config::debug, raw_scales, "raw_scales");
    CHECK_INPUT(config::debug, raw_rotations, "raw_rotations");
    CHECK_INPUT(config::debug, raw_opacities, "raw_opacities");
    CHECK_INPUT(config::debug, random_samples, "random_samples");
    CHECK_INPUT(config::debug, means, "means");

    add_noise(
        reinterpret_cast<float3*>(raw_scales.data_ptr<float>()),
        reinterpret_cast<float4*>(raw_rotations.data_ptr<float>()),
        raw_opacities.data_ptr<float>(),
        reinterpret_cast<float3*>(random_samples.data_ptr<float>()),
        reinterpret_cast<float3*>(means.data_ptr<float>()),
        raw_scales.size(0),
        current_lr
    );

}
