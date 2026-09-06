#pragma once

#define DEF inline constexpr

namespace faster_gs::filter3d::config {
    DEF bool debug = false;
    DEF int block_size_update_3d_filter = 256;
}

namespace config = faster_gs::filter3d::config;

#undef DEF
