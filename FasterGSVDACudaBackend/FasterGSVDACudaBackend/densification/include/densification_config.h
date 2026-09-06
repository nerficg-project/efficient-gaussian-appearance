#pragma once

#include "helper_math.h"

#define DEF inline constexpr

namespace faster_gs::densification::config {
    DEF bool debug = false;
    // mcmc constants
    DEF int mcmc_max_n_samples = 50;
    // block size constants
    DEF int block_size_relocation = 256;
    DEF int block_size_add_noise = 256;
}

namespace config = faster_gs::densification::config;

#undef DEF
