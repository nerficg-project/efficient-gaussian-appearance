#include "appearance.h"
#include "rasterization_config.h"

#include <tiny-cuda-nn/rtc_kernel.h>

#include <c10/cuda/CUDAStream.h>

#include <fstream>
#include <sstream>

namespace faster_gs::rasterization {

    static std::string rtc_include_dir;
    static std::string rtc_kernel_dir;

    void set_rtc_dirs(const std::string& cache_dir, const std::string& include_dir, const std::string& kernel_dir) {
        tcnn::rtc_set_cache_dir(cache_dir);
        tcnn::rtc_set_include_dir(include_dir);
        rtc_include_dir = include_dir;
        rtc_kernel_dir = kernel_dir;
    }

    Rasterizer::~Rasterizer() = default;

    void Rasterizer::check_appearance_set() const {
        if (!m_appearance_set) throw std::runtime_error("FasterGSVDA: appearance model is not initialized");
    }

    void Rasterizer::check_residual_mlp() const {
        if (m_model == nullptr) throw std::runtime_error("FasterGSVDA: residual mlp is not initialized");
    }

    void Rasterizer::invalidate_appearance_kernels() {
        m_preprocess_forward_kernel = nullptr;
        m_preprocess_backward_kernel = nullptr;
        // the precomputed-colors kernels are appearance independent and stay cached
        m_params_jit = torch::Tensor();
    }

    static std::string read_rtc_file(const std::string& path) {
        std::ifstream file(path);
        if (!file) throw std::runtime_error("FasterGSVDA: could not read rtc source: " + path);
        std::stringstream stream;
        stream << file.rdbuf();
        return stream.str();
    }

    static std::string read_rtc_kernel_file(const std::string& filename) {
        return read_rtc_file(rtc_kernel_dir + "/" + filename);
    }

    // round-trip exact float literal for the generated rtc source
    static std::string float_literal(const float value) {
        std::stringstream stream;
        stream.precision(9);
        stream << std::showpoint << value << "f";
        return stream.str();
    }

    static const char* appearance_suffix(const AppearanceModel appearance) {
        switch (appearance) {
            case AppearanceModel::Precomputed: return "precomputed";
            case AppearanceModel::SH: return "sh";
            case AppearanceModel::SV: return "sv";
            case AppearanceModel::NASG: return "nasg";
            case AppearanceModel::NASGabor: return "nasgabor";
            case AppearanceModel::Neural: return "neural";
        }
        return "";
    }

    std::string Rasterizer::kernel_name(const std::string& kernel_base, const bool precomputed) const {
        // per-variant kernel names keep the compiled variants distinguishable in profiler traces and the rtc ptx cache
        const AppearanceModel appearance = precomputed ? AppearanceModel::Precomputed : m_appearance;
        return kernel_base + "_" + appearance_suffix(appearance) + "_cu";
    }

    std::string Rasterizer::assemble_rtc_source(const std::string& kernel_filename, const bool with_backward, const bool precomputed) const {
        const AppearanceModel appearance = precomputed ? AppearanceModel::Precomputed : m_appearance;
        const std::string kernel_base = kernel_filename.substr(0, kernel_filename.size() - 3);  // strip ".cu"
        // define the jit tokens; unused tokens are defined regardless so shared code always compiles,
        // and the precomputed-colors variant zeroes every appearance-specific token so its source
        // (and therefore tcnn's ptx cache entry) is independent of the configured appearance model
        std::stringstream preamble;
        preamble << "#define JIT_KERNEL_NAME " << kernel_name(kernel_base, precomputed) << "\n";
        preamble << "#define JIT_APPEARANCE_MODEL " << static_cast<int>(appearance) << "\n";
        preamble << "#define JIT_BASE_ACTIVATION " << (precomputed ? 0 : m_base_activation) << "\n";
        preamble << "#define JIT_RESIDUAL_ACTIVATION " << (precomputed ? 0 : m_residual_activation) << "\n";
        preamble << "#define JIT_COLOR_ACTIVATION " << (precomputed ? 0 : m_color_activation) << "\n";
        preamble << "#define JIT_SH_REST_BASES " << (precomputed ? 0 : m_sh_rest_bases) << "u\n";
        preamble << "#define JIT_N_INPUT_DIMS " << (!precomputed && m_model != nullptr ? static_cast<int>(m_model->input_width()) : 0) << "\n";
        preamble << "#define JIT_N_FEATURE_DIMS " << (precomputed ? 0 : m_feature_dim) << "u\n";
        preamble << "#define JIT_LOSS_SCALE " << float_literal(config::mlp_grad_loss_scale) << "\n";
        preamble << "#define JIT_BASE_INPUT " << (!precomputed && m_base_input ? "true" : "false") << "\n";
        preamble << "#define JIT_N_FREQUENCIES " << (precomputed ? 0 : m_n_frequencies) << "u\n";
        preamble << "#define JIT_CTX_BYTES " << (!precomputed && m_model != nullptr ? static_cast<int>(m_model->device_function_fwd_ctx_bytes()) : 0) << "\n";
        preamble << "#define JIT_WITH_BACKWARD " << (with_backward ? 1 : 0) << "\n";
        preamble << "#define JIT_SH_DEGREE_MASK " << (precomputed ? 0 : m_sh_degree_mask) << "u\n";
        preamble << "#define JIT_NASG_N_LOBES " << (precomputed ? 0 : m_nasg_n_lobes) << "u\n";
        preamble << "#define JIT_NASGABOR_N_LOBES " << (precomputed ? 0 : m_nasgabor_n_lobes) << "u\n";
        preamble << "#define JIT_SV_N_SITES " << (precomputed ? 0 : m_sv_n_sites) << "u\n";
        preamble << "#define JIT_DIRECTION_GRADIENT " << (!precomputed && m_direction_gradient ? "true" : "false") << "\n";

        // the rtc source receives host-side code through two mechanisms: definitions that must match
        // exactly are spliced verbatim (appearance_params.h and tile_culling.cuh below), while values that
        // need selection or type adaptation are emitted as generated text (this config subset from rasterization_config.h)
        preamble << "namespace faster_gs { namespace config {\n";
        preamble << "constexpr float dilation = " << float_literal(config::dilation) << ";\n";
        preamble << "constexpr float dilation_proper_antialiasing = " << float_literal(config::dilation_proper_antialiasing) << ";\n";
        preamble << "constexpr bool detach_dilation_proper_antialiasing_from_cov2d = " << (config::detach_dilation_proper_antialiasing_from_cov2d ? "true" : "false") << ";\n";
        preamble << "constexpr float min_cov2d_determinant = " << float_literal(config::min_cov2d_determinant) << ";\n";
        preamble << "constexpr bool original_opacity_interpretation = " << (config::original_opacity_interpretation ? "true" : "false") << ";\n";
        preamble << "constexpr float min_alpha_threshold_rcp = " << float_literal(config::min_alpha_threshold_rcp) << ";\n";
        preamble << "constexpr float min_alpha_threshold = " << float_literal(config::min_alpha_threshold) << ";\n";
        preamble << "constexpr float max_power_threshold = " << float_literal(config::max_power_threshold) << ";\n";
        preamble << "constexpr uint32_t tile_width = " << config::tile_width << "u;\n";
        preamble << "constexpr uint32_t tile_height = " << config::tile_height << "u;\n";
        preamble << "constexpr uint32_t n_sequential_threshold = " << config::n_sequential_threshold << "u;\n";
        preamble << "} }\n";

        // the kernel sources (and the appearance struct header) are inlined into a single string (instead of using
        // includes) so that tcnn's source-hash based ptx cache is invalidated whenever any of the files change
        std::string source = preamble.str() + "\n" + read_rtc_file(rtc_include_dir + "/appearance_params.h")
            + "\n" + read_rtc_file(rtc_include_dir + "/tile_culling.cuh")
            + "\n" + read_rtc_kernel_file("preprocess_common.cuh");
        switch (appearance) {
            case AppearanceModel::SH: source += "\n" + read_rtc_kernel_file("appearance_sh.cuh"); break;
            case AppearanceModel::SV: source += "\n" + read_rtc_kernel_file("appearance_sv.cuh"); break;
            case AppearanceModel::NASG: source += "\n" + read_rtc_kernel_file("appearance_nasg.cuh"); break;
            case AppearanceModel::NASGabor: source += "\n" + read_rtc_kernel_file("appearance_nasgabor.cuh"); break;
            case AppearanceModel::Neural: source += "\n" + read_rtc_kernel_file("appearance_neural.cuh"); break;
            case AppearanceModel::Precomputed: break;
        }
        source += "\n" + read_rtc_kernel_file(kernel_filename);
        if (appearance == AppearanceModel::Neural) {
            std::string device_functions = m_model->generate_device_function("eval_model");
            if (with_backward) device_functions += "\n" + m_model->generate_backward_device_function("backward_eval_model", config::block_size_preprocess);
            source = device_functions + "\n" + source;
        }
        return source;
    }

    void Rasterizer::reset_appearance(const AppearanceModel appearance, const int base_activation, const int residual_activation, const int color_activation) {
        if (base_activation < 0 || base_activation > 1) throw std::runtime_error("FasterGSVDA: invalid base activation id");
        if (residual_activation < 0 || residual_activation > 2) throw std::runtime_error("FasterGSVDA: invalid residual activation id");
        if (color_activation < 0 || color_activation > 5) throw std::runtime_error("FasterGSVDA: invalid color activation id");
        m_appearance_set = true;
        m_appearance = appearance;
        m_base_activation = base_activation;
        m_residual_activation = residual_activation;
        m_color_activation = color_activation;
        m_sh_rest_bases = 0;
        m_model = nullptr;
        m_feature_dim = 0;
        m_base_input = false;
        m_n_frequencies = 0;
        m_sh_degree_mask = 0;
        m_nasg_n_lobes = 0;
        m_nasgabor_n_lobes = 0;
        m_sv_n_sites = 0;
        m_direction_gradient = false;
        invalidate_appearance_kernels();
    }

    void Rasterizer::set_appearance(const int appearance_model, const int max_degree, const int base_activation, const int residual_activation, const int color_activation, const bool direction_gradient) {
        if (appearance_model < 1 || appearance_model > 5) throw std::runtime_error("FasterGSVDA: invalid appearance model id");
        const AppearanceModel appearance = static_cast<AppearanceModel>(appearance_model);
        reset_appearance(appearance, base_activation, residual_activation, color_activation);
        m_direction_gradient = direction_gradient;
        switch (appearance) {
            case AppearanceModel::SH: m_sh_rest_bases = (max_degree + 1) * (max_degree + 1) - 1; break;
            case AppearanceModel::SV: m_sv_n_sites = max_degree; break;
            case AppearanceModel::NASG: m_nasg_n_lobes = max_degree; break;
            case AppearanceModel::NASGabor: m_nasgabor_n_lobes = max_degree; break;
            case AppearanceModel::Neural: break;  // the residual mlp is configured via set_residual_mlp
            case AppearanceModel::Precomputed: break;  // unreachable, the id is validated above
        }
    }

    void Rasterizer::set_residual_mlp(
        const int n_input_dims,
        const int n_output_dims,
        const nlohmann::json& encoding_config,
        const nlohmann::json& network_config,
        const bool base_input,
        const int n_frequencies,
        const int sh_degree_mask)
    {
        if (!m_appearance_set || m_appearance != AppearanceModel::Neural) throw std::runtime_error("FasterGSVDA: set_appearance must select the neural appearance model before set_residual_mlp");
        invalidate_appearance_kernels();
        m_model = std::make_unique<tcnn::NetworkWithInputEncoding<__half>>(n_input_dims, n_output_dims, encoding_config, network_config);
        int n_degree_dims = 0;
        for (int degree = 0; degree <= 6; degree++) if (sh_degree_mask & (1 << degree)) n_degree_dims += 2 * degree + 1;
        m_base_input = base_input;
        m_n_frequencies = n_frequencies;
        m_sh_degree_mask = sh_degree_mask;
        // the inputs besides the (optionally frequency-encoded) feature block hold the sh degree values;
        // with base_input, 3 of the encoder's raw dims are the base values, so the learned feature dim is the total minus 3
        m_feature_dim = (n_input_dims - n_degree_dims) / (n_frequencies > 0 ? 2 * n_frequencies : 1) - (base_input ? 3 : 0);
    }

    static void check_jit_support() {
        if (!tcnn::supports_jit_fusion()) throw std::runtime_error("FasterGSVDA: the jit-compiled preprocess kernels require jit fusion support (compute capability 7.5+ and CUDA 11.8+)");
    }

    tcnn::CudaRtcKernel* Rasterizer::preprocess_forward_kernel() {
        check_appearance_set();
        if (m_appearance == AppearanceModel::Neural) check_residual_mlp();
        if (m_preprocess_forward_kernel == nullptr) {
            check_jit_support();
            m_preprocess_forward_kernel = std::make_unique<tcnn::CudaRtcKernel>(kernel_name("preprocess_forward", false), assemble_rtc_source("preprocess_forward.cu", false, false));
        }
        return m_preprocess_forward_kernel.get();
    }

    tcnn::CudaRtcKernel* Rasterizer::preprocess_backward_kernel() {
        check_appearance_set();
        if (m_appearance == AppearanceModel::Neural) check_residual_mlp();
        if (m_preprocess_backward_kernel == nullptr) {
            check_jit_support();
            m_preprocess_backward_kernel = std::make_unique<tcnn::CudaRtcKernel>(kernel_name("preprocess_backward", false), assemble_rtc_source("preprocess_backward.cu", true, false));
        }
        return m_preprocess_backward_kernel.get();
    }

    tcnn::CudaRtcKernel* Rasterizer::precomputed_forward_kernel() {
        if (m_precomputed_forward_kernel == nullptr) {
            check_jit_support();
            m_precomputed_forward_kernel = std::make_unique<tcnn::CudaRtcKernel>(kernel_name("preprocess_forward", true), assemble_rtc_source("preprocess_forward.cu", false, true));
        }
        return m_precomputed_forward_kernel.get();
    }

    tcnn::CudaRtcKernel* Rasterizer::precomputed_backward_kernel() {
        if (m_precomputed_backward_kernel == nullptr) {
            check_jit_support();
            m_precomputed_backward_kernel = std::make_unique<tcnn::CudaRtcKernel>(kernel_name("preprocess_backward", true), assemble_rtc_source("preprocess_backward.cu", true, true));
        }
        return m_precomputed_backward_kernel.get();
    }

    bool Rasterizer::needs_mlp_outputs() const {
        if (m_model == nullptr) return false;
        return m_residual_activation != 0;
    }

    void Rasterizer::upload_params(const torch::Tensor& mlp_weights) {
        check_residual_mlp();
        if (!m_params_jit.defined()) {
            const torch::TensorOptions half_options = torch::TensorOptions().dtype(torch::kHalf).device(torch::kCUDA);
            m_params_jit = torch::empty({static_cast<int>(m_model->n_params())}, half_options);
        }
        m_params_jit.copy_(mlp_weights);
        __half* params = reinterpret_cast<__half*>(m_params_jit.data_ptr<torch::Half>());
        m_model->set_params(params, params, nullptr);
        m_model->convert_params_to_jit_layout(c10::cuda::getCurrentCUDAStream(), false);
    }

    const __half* Rasterizer::params_jit() const {
        return reinterpret_cast<const __half*>(m_params_jit.data_ptr<torch::Half>());
    }

    int Rasterizer::fwd_ctx_bytes() const {
        if (m_model == nullptr) return 0;
        return static_cast<int>(m_model->device_function_fwd_ctx_bytes());
    }

    int Rasterizer::backward_shmem_bytes() const {
        if (m_model == nullptr) return 0;
        return static_cast<int>(m_model->backward_device_function_shmem_bytes(config::block_size_preprocess, tcnn::GradientMode::Accumulate));
    }

}
