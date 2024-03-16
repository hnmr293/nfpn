from typing import Optional
import torch
import torch.nn.functional as F

from .. import convert


def to_hf8(module: torch.nn.Module, convert_linear: bool = True, convert_conv2d: bool = False):
    target_modules = []
    
    if convert_linear:
        target_modules.append((torch.nn.Linear, Linear))
    
    if convert_conv2d:
        target_modules.append((torch.nn.Conv2d, Conv2d))
    
    for name, mod in list(module.named_children()):
        for orig_class, hf_class in target_modules:
            if isinstance(mod, orig_class):
                try:
                    new_mod = hf_class(mod)
                except Exception as e:
                    print(f'[nfpn] *** WARN *** failed to convert module to HF8: {name} {str(e)}')
                    break
                
                delattr(module, name)
                del mod
                
                setattr(module, name, new_mod)
                break
    
    return module


def _get_param(data: torch.Tensor):
    if convert.HF8_MAX <= data.abs().max():
        print('[nfpn] *** WARN *** max(abs(data)) >= HF8_MAX')
    
    hf8 = convert.to_hf8(data)
    
    hf8.requires_grad_(False)
    
    hf8 = torch.nn.Parameter(hf8, requires_grad=False)
    
    return (hf8,)


class Linear(torch.nn.Module):
    def __init__(self, base: torch.nn.Linear) -> None:
        super().__init__()
        self.weight = _get_param(base.weight)
        self.weight_shape = base.weight.shape
        if base.bias is not None:
            self.bias = _get_param(base.bias)
            self.bias_shape = base.bias.shape
        else:
            self.bias = None
            self.bias_shape = None
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.to(base.weight.device)
    
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        weight = convert.hf8_to_fp16(*self.weight).reshape(self.weight_shape)
        bias = convert.hf8_to_fp16(*self.bias).reshape(self.bias_shape) if self.bias else None
        return F.linear(x, weight, bias)
    
    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse)
        self.weight = [fn(p) for p in self.weight]
        if self.bias:
            self.bias = [fn(p) for p in self.bias]
        return self


class Conv2d(torch.nn.Module):
    def __init__(self, base: torch.nn.Conv2d):
        super().__init__()
        self.weight = _get_param(base.weight)
        self.weight_shape = base.weight.shape
        if base.bias is not None:
            self.bias = _get_param(base.bias)
            self.bias_shape = base.bias.shape
        else:
            self.bias = None
            self.bias_shape = None
        
        self.padding_mode = base.padding_mode
        self._reversed_padding_repeated_twice = base._reversed_padding_repeated_twice
        self.stride = base.stride
        self.dilation = base.dilation
        self.groups = base.groups
        self.padding = base.padding
        
        self.to(base.weight.device)
    
    def _conv_forward(self, input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, bias, self.stride,
                            (0, 0), self.dilation, self.groups)
        return F.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        weight = convert.hf8_to_fp16(*self.weight).reshape(self.weight_shape)
        bias = convert.hf8_to_fp16(*self.bias).reshape(self.bias_shape) if self.bias else None
        return self._conv_forward(x, weight, bias)
    
    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse)
        self.weight = [fn(p) for p in self.weight]
        if self.bias:
            self.bias = [fn(p) for p in self.bias]
        return self
