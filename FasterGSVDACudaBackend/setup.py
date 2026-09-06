import os
import shutil
from glob import glob
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

__author__ = 'Florian Hahlbohm'
__description__ = 'A refactored CUDA implementation of the 3DGS rasterizer with runtime-compiled appearance models.'

ENABLE_FASTMATH = True  # set to False to disable fast math optimizations (e.g., for debugging)
ENABLE_NVCC_LINEINFO = False  # set to True for profiling kernels with Nsight Compute (overhead is minimal)

module_root = Path(__file__).parent.absolute()
extension_name = module_root.name
extension_root = module_root / extension_name
cuda_modules = [d.name for d in Path(extension_root).iterdir() if d.is_dir() and d.name not in ['utils', 'torch_bindings']]

# gather source files
sources = [str(extension_root / 'torch_bindings' / 'bindings.cu')]
for module in cuda_modules:
    sources += glob(str(extension_root / module / 'src' / '**'/ '*.cpp'), recursive=True)
    sources += glob(str(extension_root / module / 'src' / '**' / '*.cu'), recursive=True)

# gather include directories
include_dirs = [str(extension_root / 'utils')]
for module in cuda_modules:
    include_dirs.append(str(extension_root / module / 'include'))

# add tcnn files
tcnn_root_dir = module_root / 'tiny-cuda-nn'
sources += [
    f'{tcnn_root_dir}/dependencies/fmt/src/format.cc',
    f'{tcnn_root_dir}/dependencies/fmt/src/os.cc',
    f'{tcnn_root_dir}/src/cpp_api.cu',
    f'{tcnn_root_dir}/src/common_host.cu',
    f'{tcnn_root_dir}/src/encoding.cu',
    f'{tcnn_root_dir}/src/object.cu',
    f'{tcnn_root_dir}/src/rtc_kernel.cu',
    f'{tcnn_root_dir}/src/network.cu',
    f'{tcnn_root_dir}/src/cutlass_mlp.cu',
    f'{tcnn_root_dir}/src/fully_fused_mlp.cu',
]
include_dirs += [
    f'{tcnn_root_dir}',
    f'{tcnn_root_dir}/include',
    f'{tcnn_root_dir}/dependencies',
    f'{tcnn_root_dir}/dependencies/cutlass/include',
    f'{tcnn_root_dir}/dependencies/cutlass/tools/util/include',
    f'{tcnn_root_dir}/dependencies/fmt/include',
]

# copy headers required by RTC at runtime
rtc_dir = module_root / 'rtc'
rtc_include_dir = rtc_dir / 'include'
rtc_cache_dir = rtc_dir / 'cache'
shutil.rmtree(rtc_dir, ignore_errors=True)
os.makedirs(rtc_include_dir, exist_ok=True)
os.makedirs(rtc_cache_dir, exist_ok=True)

nvcc_path = shutil.which('nvcc')
if nvcc_path is None:
    print(f'WARNING: could not find CUDA include directory. JIT compilation will not be supported.')
else:
    cuda_include_dir = os.path.join(os.path.dirname(os.path.dirname(nvcc_path)), 'include')
    cuda_headers = glob(f'{cuda_include_dir}/cuda_fp16*') + glob(f'{cuda_include_dir}/vector*')
    tcnn_headers = glob(f'{tcnn_root_dir}/include/tiny-cuda-nn/*', recursive=True)
    pcg32_headers = glob(f'{tcnn_root_dir}/dependencies/pcg32/*')
    rasterization_headers = [
        f'{extension_root}/rasterization/include/appearance_params.h',
        f'{extension_root}/rasterization/include/tile_culling.cuh',
    ]
    def copy_files(whence, files):
        for h in files:
            if not os.path.isfile(h):
                continue
            tgt = os.path.join(rtc_include_dir, os.path.relpath(h, whence))
            os.makedirs(os.path.dirname(tgt), exist_ok=True)
            shutil.copyfile(h, tgt)
    copy_files(cuda_include_dir, cuda_headers)
    copy_files(tcnn_root_dir / 'include', tcnn_headers)
    copy_files(tcnn_root_dir / 'dependencies', pcg32_headers)
    copy_files(extension_root / 'rasterization' / 'include', rasterization_headers)

    # patch an upstream tiny-cuda-nn bug in the JIT backward: the generated hidden layers store
    # POST-activation values in the forward context (network.cu: activate() before the ctx store),
    # but mma.h's activate_bwd computed the activation derivative via vec_activation_backward_in,
    # which expects PRE-activation inputs -- i.e. it evaluates f'(f(x)) instead of f'(x). This is
    # coincidentally correct for sign-preserving activations (ReLU/LeakyReLU/None, derivative
    # depends only on the argument's sign) but wrong for all smooth hidden activations
    # (Softplus/Squareplus/Sigmoid/Tanh). Fixed by recovering the derivative from the stored
    # output instead. Note: the output-layer ctx stores pre-activation values, so this patch
    # assumes output_activation None (which activate_bwd no-ops), as used by this method.
    mma_path = os.path.join(rtc_include_dir, 'tiny-cuda-nn', 'mma.h')
    with open(mma_path) as f:
        mma_source = f.read()
    mma_fixes = [
        ('*(vec_t*)this = vec_activation_backward_in(act, *(vec_t*)this, *(vec_t*)&fwd_in);',
         '*(vec_t*)this = vec_activation_backward(act, *(vec_t*)this, *(const vec_t*)&fwd_in);'),
        ('*(vec_t*)this = vec_activation_backward_in<act, __half, N_ELEMS, 16>(*(vec_t*)this, *(const vec_t*)&fwd_in);',
         '*(vec_t*)this = vec_activation_backward(act, *(vec_t*)this, *(const vec_t*)&fwd_in);'),
    ]
    for old, new in mma_fixes:
        if old not in mma_source:
            raise RuntimeError(f'mma.h activation-backward patch target not found (tcnn updated?): {old}')
        mma_source = mma_source.replace(old, new)
    with open(mma_path, 'w') as f:
        f.write(mma_source)
    print('patched rtc mma.h: activation backward now recovers derivatives from stored outputs')

# set up compiler flags
cxx_flags = [
    '/std:c++17' if os.name == 'nt' else '-std=c++17',
    '-DTCNN_HALF_PRECISION=1',
]
nvcc_flags = [
    '-std=c++17',
    '--extended-lambda',
    '--expt-relaxed-constexpr',
    '-U__CUDA_NO_HALF_OPERATORS__',
    '-U__CUDA_NO_HALF_CONVERSIONS__',
    '-U__CUDA_NO_HALF2_OPERATORS__',
    '-Xcompiler=-Wno-float-conversion',
    '-Xcompiler=-fno-strict-aliasing',
    '-DTCNN_PARAMS_UNALIGNED',
    '-DTCNN_RTC',
    '-DTCNN_HALF_PRECISION=1',
]
if ENABLE_FASTMATH:
    cxx_flags.append('-O3')
    nvcc_flags.append('-O3')
    nvcc_flags.append('-use_fast_math')
    nvcc_flags.append('-DTCNN_RTC_USE_FAST_MATH')
if ENABLE_NVCC_LINEINFO:
    nvcc_flags.append('-lineinfo')

# define the CUDA extension
extension = CUDAExtension(
    name=f'{extension_name}._C',
    sources=sources,
    include_dirs=include_dirs,
    extra_compile_args={
        'cxx': cxx_flags,
        'nvcc': nvcc_flags
    },
    libraries=['cuda', 'nvrtc'],
)

# set up the package
setup(
    name=extension_name,
    author=__author__,
    packages=[f'{extension_name}.torch_bindings'],
    ext_modules=[extension],
    description=__description__,
    cmdclass={'build_ext': BuildExtension}
)
