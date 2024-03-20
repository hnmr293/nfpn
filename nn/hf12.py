import torch
from .base import to_hf, HfLinear, HfConv2d


_FMT_NAME = 'HF12'

def to_hf12(module: torch.nn.Module, convert_linear: bool = True, convert_conv2d: bool = False):
    target_modules = []
    
    if convert_linear:
        target_modules.append((torch.nn.Linear, Linear))
    
    if convert_conv2d:
        target_modules.append((torch.nn.Conv2d, Conv2d))
    
    return to_hf(module, target_modules, _FMT_NAME)


class Linear(HfLinear):
    def __init__(self, base: torch.nn.Linear) -> None:
        super().__init__(_FMT_NAME, base)


class Conv2d(HfConv2d):
    def __init__(self, base: torch.nn.Conv2d):
        super().__init__(_FMT_NAME, base)
