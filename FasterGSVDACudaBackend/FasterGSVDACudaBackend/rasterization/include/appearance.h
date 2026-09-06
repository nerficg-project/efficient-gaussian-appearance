#pragma once

#include <torch/extension.h>
#include <tiny-cuda-nn/network_with_input_encoding.h>
#include <cuda_fp16.h>
#include <memory>
#include <string>

namespace faster_gs::rasterization {

    // registers the directories used for jit runtime compilation (called once at import time)
    void set_rtc_dirs(const std::string& cache_dir, const std::string& include_dir, const std::string& kernel_dir);

    enum class AppearanceModel : int { Precomputed = 0, SH = 1, SV = 2, NASG = 3, NASGabor = 4, Neural = 5 };

    // owns the jit-compiled preprocess kernels of the configured appearance model, an appearance-independent
    // kernel pair for the precomputed-colors path, and (neural only) the jit-fused tcnn residual mlp
    class Rasterizer {
    public:
        Rasterizer() = default;
        ~Rasterizer();

        // selects the appearance model and activation triple (ids match the python-side initialize_appearance);
        // max_degree is unused for neural (see set_residual_mlp); direction_gradient is baked in as the
        // JIT_DIRECTION_GRADIENT token (color backward through the view direction)
        void set_appearance(
            const int appearance_model,
            const int max_degree,
            const int base_activation,
            const int residual_activation,
            const int color_activation,
            const bool direction_gradient);

        // creates the tcnn residual mlp for the jit-fused kernels (requires the neural appearance model);
        // sh_degree_mask is a bitmask of the view-direction SH degrees fed to the mlp (bit l set = degree l)
        void set_residual_mlp(
            const int n_input_dims,
            const int n_output_dims,
            const nlohmann::json& encoding_config,
            const nlohmann::json& network_config,
            const bool base_input,
            const int n_frequencies,
            const int sh_degree_mask);

        bool is_neural() const { return m_appearance == AppearanceModel::Neural; }

        // lazily built jit-compiled preprocess kernels (configured appearance / precomputed-colors path)
        tcnn::CudaRtcKernel* preprocess_forward_kernel();
        tcnn::CudaRtcKernel* preprocess_backward_kernel();
        tcnn::CudaRtcKernel* precomputed_forward_kernel();
        tcnn::CudaRtcKernel* precomputed_backward_kernel();

        // fused mlp only
        bool needs_mlp_outputs() const;
        void upload_params(const torch::Tensor& mlp_weights);
        const __half* params_jit() const;
        int fwd_ctx_bytes() const;
        int backward_shmem_bytes() const;

    private:
        bool m_appearance_set = false;
        AppearanceModel m_appearance = AppearanceModel::SH;
        int m_base_activation = 0;
        int m_residual_activation = 0;
        int m_color_activation = 0;
        int m_sh_rest_bases = 0;
        int m_nasg_n_lobes = 0;
        int m_nasgabor_n_lobes = 0;
        int m_sv_n_sites = 0;
        bool m_direction_gradient = false;
        std::unique_ptr<tcnn::NetworkWithInputEncoding<__half>> m_model;
        torch::Tensor m_params_jit;
        int m_feature_dim = 0;
        bool m_base_input = false;
        int m_n_frequencies = 0;
        int m_sh_degree_mask = 0;
        std::unique_ptr<tcnn::CudaRtcKernel> m_preprocess_forward_kernel;
        std::unique_ptr<tcnn::CudaRtcKernel> m_preprocess_backward_kernel;
        std::unique_ptr<tcnn::CudaRtcKernel> m_precomputed_forward_kernel;
        std::unique_ptr<tcnn::CudaRtcKernel> m_precomputed_backward_kernel;

        void check_appearance_set() const;
        void check_residual_mlp() const;
        void invalidate_appearance_kernels();
        std::string kernel_name(const std::string& kernel_base, const bool precomputed) const;
        std::string assemble_rtc_source(const std::string& kernel_filename, const bool with_backward, const bool precomputed) const;
        void reset_appearance(const AppearanceModel appearance, const int base_activation, const int residual_activation, const int color_activation);
    };

}
