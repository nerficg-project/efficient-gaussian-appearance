#pragma once

#define DEF inline constexpr

namespace faster_gs::adam::config {
    DEF bool debug = false;
    // block size constants
    DEF int block_size_adam_step = 256;
    DEF int max_step_sizes = 7;  // maximum number of per-column step sizes
}

namespace config = faster_gs::adam::config;

#undef DEF