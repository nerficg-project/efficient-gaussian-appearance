#include "adam.h"
#include "adam_config.h"
#include "torch_utils.h"
#include "utils.h"
#include <cmath>

namespace faster_gs::adam {

    // per-column step sizes, cycled via idx % count; passed by value to live in the kernel parameter space
    struct StepSizes {
        float values[config::max_step_sizes];
        int count;
    };

    // based on https://github.com/pytorch/pytorch/blob/9d32aa9789fc0ef0cad01a788157ecc2121db810/torch/csrc/api/src/optim/adam.cpp#L72-L142
    __global__ void adam_step_cu(
        const float* param_grad,
        float* param,
        float* exp_avg,
        float* exp_avg_sq,
        const int n_elements,
        const StepSizes step_sizes,
        const float beta1,
        const float beta2,
        const float epsilon,
        const float bias_correction2_sqrt_rcp)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n_elements) return;
        const float grad = param_grad[idx];
        const float exp = exp_avg[idx];
        const float exp_sq = exp_avg_sq[idx];
        const float grad_sq = grad * grad;
        const float moment1 = fmaf(beta1, exp - grad, grad);
        const float moment2 = fmaf(beta2, exp_sq - grad_sq, grad_sq);
        const float denom = sqrtf(moment2) * bias_correction2_sqrt_rcp + epsilon;
        param[idx] -= step_sizes.values[idx % step_sizes.count] * moment1 / denom;
        exp_avg[idx] = moment1;
        exp_avg_sq[idx] = moment2;
    }

    void adam_step_wrapper(
        const torch::Tensor& param_grad,
        torch::Tensor& param,
        torch::Tensor& exp_avg,
        torch::Tensor& exp_avg_sq,
        const int step_count,
        const std::variant<double, std::vector<double>>& learning_rate,
        const double beta1,
        const double beta2,
        const double epsilon)
    {
        CHECK_INPUT(config::debug, param, "param");
        CHECK_INPUT(config::debug, exp_avg, "exp_avg");
        CHECK_INPUT(config::debug, exp_avg_sq, "exp_avg_sq");
        CHECK_INPUT(config::debug, param_grad, "param_grad");

        const auto learning_rates = std::visit([](const auto& value) { return std::vector<double>{value}; }, learning_rate);
        TORCH_CHECK(!learning_rates.empty() && learning_rates.size() <= config::max_step_sizes, "between 1 and ", config::max_step_sizes, " learning rates are supported");

        const int n_elements = param.numel();

        const double bias_correction1_rcp = 1.0 / (1.0 - std::pow(beta1, step_count));
        const double bias_correction2_sqrt_rcp = 1.0 / std::sqrt(1.0 - std::pow(beta2, step_count));

        StepSizes step_sizes;
        step_sizes.count = static_cast<int>(learning_rates.size());
        for (int i = 0; i < step_sizes.count; i++) step_sizes.values[i] = static_cast<float>(learning_rates[i] * bias_correction1_rcp);

        adam_step_cu<<<div_round_up(n_elements, config::block_size_adam_step), config::block_size_adam_step>>>(
            param_grad.data_ptr<float>(),
            param.data_ptr<float>(),
            exp_avg.data_ptr<float>(),
            exp_avg_sq.data_ptr<float>(),
            n_elements,
            step_sizes,
            static_cast<float>(beta1),
            static_cast<float>(beta2),
            static_cast<float>(epsilon),
            static_cast<float>(bias_correction2_sqrt_rcp)
        );
        CHECK_CUDA(config::debug, "adam_step")
    }

}
