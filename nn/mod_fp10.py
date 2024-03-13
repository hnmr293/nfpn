from typing import Optional
import torch
import torch.nn.functional as F

from nfpn import to_fp10, fp10_to_fp16, FP10_MAX


def get_param(data: torch.Tensor):
    if FP10_MAX <= data.abs().max():
        print('[WARN] max(abs(data)) >= FP10_MAX')
    
    exp, frac = to_fp10(data)
    
    exp.requires_grad_(False)
    frac.requires_grad_(False)
    
    exp = torch.nn.Parameter(exp, requires_grad=False)
    frac = torch.nn.Parameter(frac, requires_grad=False)
    
    return exp, frac


class Linear(torch.nn.Module):
    def __init__(self, base: torch.nn.Linear) -> None:
        super().__init__()
        self.weight = get_param(base.weight)
        self.weight_shape = base.weight.shape
        if base.bias is not None:
            self.bias = get_param(base.bias)
            self.bias_shape = base.bias.shape
        else:
            self.bias = None
            self.bias_shape = None
        self.to(base.weight.device)
    
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        weight = fp10_to_fp16(*self.weight).reshape(self.weight_shape)
        bias = fp10_to_fp16(*self.bias).reshape(self.bias_shape) if self.bias else None
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
        self.weight = get_param(base.weight)
        self.weight_shape = base.weight.shape
        if base.bias is not None:
            self.bias = get_param(base.bias)
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
        weight = fp10_to_fp16(*self.weight).reshape(self.weight_shape)
        bias = fp10_to_fp16(*self.bias).reshape(self.bias_shape) if self.bias else None
        return self._conv_forward(x, weight, bias)
    
    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse)
        self.weight = [fn(p) for p in self.weight]
        if self.bias:
            self.bias = [fn(p) for p in self.bias]
        return self
