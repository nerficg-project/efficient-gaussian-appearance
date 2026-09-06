"""FasterGSVDA/Appearance: Dynamically provides access to the implemented appearance models.

Each module in this package implements one appearance model as a class named like its file
(SH.py -> SH); MODEL.APPEARANCE.TYPE selects one by class name. The rasterizer's preprocess
kernels are jit-compiled for the configured appearance model and activations
(see rasterization/rtc/ in the CUDA backend).
"""

import importlib
from pathlib import Path

import Framework

from Methods.FasterGSVDA.Appearance.Base import AppearanceModel

options = tuple(sorted(f.stem for f in Path(__file__).parent.iterdir() if f.suffix == '.py' and f.stem not in ('__init__', 'Base', 'utils')))


def create_appearance_model(appearance_config: Framework.ConfigParameterList) -> AppearanceModel:
    """Creates the appearance model strategy selected by MODEL.APPEARANCE.TYPE and configures the rasterizer for it."""
    if appearance_config.TYPE not in options:
        raise Framework.ModelError(f'MODEL.APPEARANCE.TYPE must be one of {list(options)}, got {appearance_config.TYPE}')
    module = importlib.import_module(f'Methods.FasterGSVDA.Appearance.{appearance_config.TYPE}')
    appearance_class = getattr(module, appearance_config.TYPE)
    appearance = appearance_class(appearance_config)
    appearance.initialize_backend()
    return appearance
