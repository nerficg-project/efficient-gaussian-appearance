"""FasterGSVDA/Appearance/utils.py"""

import math

import torch


def fibonacci_sphere(n: int) -> torch.Tensor:
    """Distributes n points near-uniformly on the unit sphere via the Fibonacci lattice (n, 3)."""
    indices = torch.arange(n, dtype=torch.float32) + 0.5
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    y = 1.0 - 2.0 * indices / n
    radius = torch.sqrt(1.0 - y * y)
    theta = golden_angle * indices
    return torch.stack([torch.cos(theta) * radius, y, torch.sin(theta) * radius], dim=1)


def frequency_encode(features: torch.Tensor, n_frequencies: int) -> torch.Tensor:
    """Frequency-encodes per-Gaussian features following tcnn's ordering: per feature sin(2^f*x), cos(2^f*x) for f in [0, n_frequencies)."""
    scaled = features[..., None] * torch.exp2(torch.arange(n_frequencies, dtype=torch.float32, device=features.device))
    return torch.stack([scaled.sin(), scaled.cos()], dim=-1).flatten(start_dim=-3)


SH_C0 = 0.28209479177387814  # 1/(2*sqrt(pi))


def compute_sh_degrees(directions: torch.Tensor, active_degree: int, sh_degrees: list[int]) -> torch.Tensor:
    """Evaluates the SH basis of the configured degrees for unit directions (N, sum(2l+1)).

    sh_degrees must be strictly ascending; degree 0 is the direction-independent input and is never
    gated, while the i-th configured degree >= 1 is zeroed unless i < active_degree.
    Polynomials taken from tiny-cuda-nn's sh_enc.
    """
    x, y, z = directions.unbind(dim=-1)
    xy, xz, yz, x2, y2, z2 = x * y, x * z, y * z, x * x, y * y, z * z
    x4, y4, z4 = x2 * x2, y2 * y2, z2 * z2
    x6, y6, z6 = x4 * x2, y4 * y2, z4 * z2
    degree_values = {
        0: lambda: [
            SH_C0,
        ],
        1: lambda: [
            -0.48860251190291987 * y,
            0.48860251190291987 * z,
            -0.48860251190291987 * x,
        ],
        2: lambda: [
            1.0925484305920792 * xy,
            -1.0925484305920792 * yz,
            0.94617469575755997 * z2 - 0.31539156525251999,
            -1.0925484305920792 * xz,
            0.54627421529603959 * x2 - 0.54627421529603959 * y2,
        ],
        3: lambda: [
            0.59004358992664352 * y * (-3.0 * x2 + y2),
            2.8906114426405538 * xy * z,
            0.45704579946446572 * y * (1.0 - 5.0 * z2),
            0.3731763325901154 * z * (5.0 * z2 - 3.0),
            0.45704579946446572 * x * (1.0 - 5.0 * z2),
            1.4453057213202769 * z * (x2 - y2),
            0.59004358992664352 * x * (-x2 + 3.0 * y2),
        ],
        4: lambda: [
            2.5033429417967046 * xy * (x2 - y2),
            1.7701307697799304 * yz * (-3.0 * x2 + y2),
            0.94617469575756008 * xy * (7.0 * z2 - 1.0),
            0.66904654355728921 * yz * (3.0 - 7.0 * z2),
            -3.1735664074561294 * z2 + 3.7024941420321507 * z4 + 0.31735664074561293,
            0.66904654355728921 * xz * (3.0 - 7.0 * z2),
            0.47308734787878004 * (x2 - y2) * (7.0 * z2 - 1.0),
            1.7701307697799304 * xz * (-x2 + 3.0 * y2),
            -3.7550144126950569 * x2 * y2 + 0.62583573544917614 * x4 + 0.62583573544917614 * y4,
        ],
        5: lambda: [
            0.65638205684017015 * y * (10.0 * x2 * y2 - 5.0 * x4 - y4),
            8.3026492595241645 * xy * z * (x2 - y2),
            -0.48923829943525038 * y * (3.0 * x2 - y2) * (9.0 * z2 - 1.0),
            4.7935367849733241 * xy * z * (3.0 * z2 - 1.0),
            0.45294665119569694 * y * (14.0 * z2 - 21.0 * z4 - 1.0),
            0.1169503224534236 * z * (-70.0 * z2 + 63.0 * z4 + 15.0),
            0.45294665119569694 * x * (14.0 * z2 - 21.0 * z4 - 1.0),
            2.3967683924866621 * z * (x2 - y2) * (3.0 * z2 - 1.0),
            -0.48923829943525038 * x * (x2 - 3.0 * y2) * (9.0 * z2 - 1.0),
            2.0756623148810411 * z * (-6.0 * x2 * y2 + x4 + y4),
            0.65638205684017015 * x * (10.0 * x2 * y2 - x4 - 5.0 * y4),
        ],
        6: lambda: [
            1.3663682103838286 * xy * (-10.0 * x2 * y2 + 3.0 * x4 + 3.0 * y4),
            2.3666191622317521 * yz * (10.0 * x2 * y2 - 5.0 * x4 - y4),
            2.0182596029148963 * xy * (x2 - y2) * (11.0 * z2 - 1.0),
            -0.92120525951492349 * yz * (3.0 * x2 - y2) * (11.0 * z2 - 3.0),
            0.92120525951492349 * xy * (-18.0 * z2 + 33.0 * z4 + 1.0),
            0.58262136251873131 * yz * (30.0 * z2 - 33.0 * z4 - 5.0),
            6.6747662381009842 * z2 - 20.024298714302954 * z4 + 14.684485723822165 * z6 - 0.31784601133814211,
            0.58262136251873131 * xz * (30.0 * z2 - 33.0 * z4 - 5.0),
            0.46060262975746175 * (x2 - y2) * (11.0 * z2 * (3.0 * z2 - 1.0) - 7.0 * z2 + 1.0),
            -0.92120525951492349 * xz * (x2 - 3.0 * y2) * (11.0 * z2 - 3.0),
            0.50456490072872406 * (11.0 * z2 - 1.0) * (-6.0 * x2 * y2 + x4 + y4),
            2.3666191622317521 * xz * (10.0 * x2 * y2 - x4 - 5.0 * y4),
            10.247761577878714 * x2 * y4 - 10.247761577878714 * x4 * y2 + 0.6831841051919143 * x6 - 0.6831841051919143 * y6,
        ],
    }
    n_dims = sum(2 * degree + 1 for degree in sh_degrees)
    sh_values = torch.empty((directions.shape[0], n_dims), dtype=torch.float32, device=directions.device)
    n_active = active_degree + (sh_degrees[0] == 0)  # always include degree 0 if present
    offset = 0
    for degree in sh_degrees[:n_active]:
        for value in degree_values[degree]():
            sh_values[:, offset] = value
            offset += 1
    sh_values[:, offset:] = 0.0
    return sh_values


def inverse_torch_softplus(y: torch.Tensor, beta: float = 1.0, threshold: float = 20.0) -> torch.Tensor:
    """Inverse of PyTorch's softplus function."""
    return torch.where(y > threshold / beta, y, (beta * y).expm1().clamp_min(torch.finfo(y.dtype).tiny).log() / beta)


def inverse_torch_sigmoid(y: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Inverse of PyTorch's sigmoid function."""
    return y.logit(eps=eps)


def inverse_torch_hardsigmoid(y: torch.Tensor) -> torch.Tensor:
    """Inverse of PyTorch's hardsigmoid function."""
    return 6.0 * (y - 0.5)


def base_activation(raw_base: torch.Tensor, activation: str) -> torch.Tensor:
    """Reference implementation of the rasterizer's base activation; scales must match rtc/preprocess_common.cuh."""
    match activation:
        case 'none':
            return raw_base
        case 'exp':
            return torch.exp(3.0 * raw_base)
        case _:
            raise ValueError(f'unsupported base activation: {activation}')


def inverse_base_activation(pre_activation_base: torch.Tensor, activation: str) -> torch.Tensor:
    """Maps target pre-activation base values to the raw base reproducing them under the given base activation."""
    match activation:
        case 'none':
            return pre_activation_base
        case 'exp':
            return torch.log(pre_activation_base.clamp_min(1e-4)) / 3.0
        case _:
            raise ValueError(f'unsupported base activation: {activation}')


def residual_activation(raw_residual: torch.Tensor, activation: str) -> torch.Tensor:
    """Reference implementation of the rasterizer's residual activation; scales must match rtc/preprocess_common.cuh."""
    match activation:
        case 'none':
            return raw_residual
        case 'tanh':
            return torch.tanh(raw_residual)
        case 'softplus':
            return torch.nn.functional.softplus(raw_residual, beta=10.0)
        case _:
            raise ValueError(f'unsupported residual activation: {activation}')


def color_activation(raw_color: torch.Tensor, activation: str) -> torch.Tensor:
    """Reference implementation of the rasterizer's color activation; scales must match rtc/preprocess_common.cuh."""
    match activation:
        case 'none':
            return raw_color + 0.5
        case 'relu':
            return torch.relu(raw_color + 0.5)
        case 'softplus':
            return torch.nn.functional.softplus(raw_color + 0.5, beta=10.0)
        case 'sigmoid':
            return torch.sigmoid(4.0 * raw_color)
        case 'hardsigmoid':
            return torch.nn.functional.hardsigmoid(6.0 * raw_color)
        case 'satexp':
            return -torch.expm1(-raw_color)
        case _:
            raise ValueError(f'unsupported color activation: {activation}')


def inverse_color_activation(rgb: torch.Tensor, activation: str) -> torch.Tensor:
    """Maps target rgb values to the raw color reproducing them under the given color activation."""
    match activation:
        case 'none':
            return rgb - 0.5
        case 'relu':
            return rgb - 0.5
        case 'softplus':
            return inverse_torch_softplus(rgb, beta=10.0) - 0.5
        case 'sigmoid':
            return inverse_torch_sigmoid(rgb) / 4.0
        case 'hardsigmoid':
            return inverse_torch_hardsigmoid(rgb) / 6.0
        case 'satexp':
            return -torch.log1p(-rgb.clamp(1e-4, 1.0 - 1e-4))
        case _:
            raise ValueError(f'unsupported color activation: {activation}')
