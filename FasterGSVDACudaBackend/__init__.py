from pathlib import Path

import Framework

extension_dir = Path(__file__).parent
__extension_name__ = extension_dir.name
__install_command__ = [
    'pip', 'install',
    str(extension_dir),
    '--no-build-isolation',  # to build the extension using the current environment instead of creating a new one
]

try:
    from .FasterGSVDACudaBackend.torch_bindings.rasterization import diff_rasterize, rasterize, update_pruning_scores, RasterizerSettings, initialize_appearance, initialize_residual_mlp
    from .FasterGSVDACudaBackend.torch_bindings.adam import FusedAdam
    from .FasterGSVDACudaBackend.torch_bindings.filter3d import update_3d_filter
    from .FasterGSVDACudaBackend.torch_bindings.densification import relocation_adjustment, add_noise
    __all__ = [
        'diff_rasterize', 'rasterize', 'update_pruning_scores', 'RasterizerSettings', 'initialize_appearance', 'initialize_residual_mlp',
        'FusedAdam',
        'update_3d_filter',
        'relocation_adjustment', 'add_noise'
    ]
    # set up jit runtime compilation
    import os
    from FasterGSVDACudaBackend import _C
    _rtc_dir = os.path.join(os.path.dirname(__file__), 'rtc')
    _rtc_cache_dir = os.path.join(_rtc_dir, 'cache')
    os.makedirs(_rtc_cache_dir, exist_ok=True)
    _rtc_include_dir = os.path.join(_rtc_dir, 'include')
    _rtc_kernel_dir = os.path.join(os.path.dirname(__file__), 'FasterGSVDACudaBackend', 'rasterization', 'rtc')
    _C.set_rtc_dirs(_rtc_cache_dir, _rtc_include_dir, _rtc_kernel_dir)
except ImportError:
    raise Framework.ExtensionError(name=__extension_name__, install_command=__install_command__)
