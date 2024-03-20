from typing import List, Tuple, Optional
import torch
import torch.nn.functional as F

from .. import convert


class HfModule(torch.nn.Module):
    
    def __init__(self, fmt_name: str):
        super().__init__()
        self.__fmt_name = fmt_name
        self.__param_names = []
    
    def init_hf(self, *parameters: Tuple[str, torch.Tensor]):
        device = None
        
        for name, data in parameters:
            if name is not None and data is not None:
                setattr(self, name, self._get_param(data))
                setattr(self, f'{name}_shape', data.shape)
                self.__param_names.append(name)
                if device is None:
                    device = data.device
            else:
                setattr(self, name, None)
                setattr(self, f'{name}_shape', None)
        
        if device is not None:
            self.to(device)
    
    def _apply(self, fn, *args, **kwargs):
        super()._apply(fn, *args, **kwargs)
        for name in self.__param_names:
            applied = tuple(fn(p) for p in getattr(self, name))
            setattr(self, name, applied)
        return self

    def _get_param(self, data: torch.Tensor):
        fmt = self.__fmt_name
        
        if getattr(convert, f'{fmt.upper()}_MAX') <= data.abs().max():
            print(f'[nfpn] *** WARN *** max(abs(data)) >= {fmt.upper()}_MAX')
        
        fn = getattr(convert, f'to_{fmt.lower()}')
        hf = fn(data)
        
        hf.requires_grad_(False)
        
        hf = torch.nn.Parameter(hf, requires_grad=False)
        
        if not isinstance(hf, tuple):
            hf = (hf,)
        
        return hf
    
    def _get_fp16(self, name: str):
        if not hasattr(self, name):
            return None
        
        data = getattr(self, name)
        
        if data is None:
            return None
        
        fn = getattr(convert, f'{self.__fmt_name.lower()}_to_fp16')
        shape = getattr(self, f'{name}_shape')
        
        return fn(*data).reshape(shape)
    

class HfLinear(HfModule):
    
    def __init__(self, fmt_name: str, base: torch.nn.Linear):
        super().__init__(fmt_name)
        
        self.in_features = base.in_features
        self.out_features = base.out_features
        
        self.init_hf(
            ('weight', base.weight),
            ('bias', base.bias),
        )
    
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        weight = self._get_fp16('weight')
        bias = self._get_fp16('bias')
        return F.linear(x, weight, bias)


class HfConv2d(HfModule):
    def __init__(self, fmt_name: str, base: torch.nn.Conv2d):
        super().__init__(fmt_name)
        
        self.padding_mode = base.padding_mode
        self._reversed_padding_repeated_twice = base._reversed_padding_repeated_twice
        self.stride = base.stride
        self.dilation = base.dilation
        self.groups = base.groups
        self.padding = base.padding
        
        self.init_hf(
            ('weight', base.weight),
            ('bias', base.bias),
        )
    
    def _conv_forward(self, input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, bias, self.stride,
                            (0, 0), self.dilation, self.groups)
        return F.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        weight = self._get_fp16('weight')
        bias = self._get_fp16('bias')
        return self._conv_forward(x, weight, bias)


def to_hf(module: torch.nn.Module, target_modules: List[Tuple[torch.nn.Module, HfModule]], fmt_name: str):
    for name, mod in list(module.named_modules()):
        for orig_class, hf_class in target_modules:
            if isinstance(mod, orig_class):
                try:
                    new_mod = hf_class(mod)
                except Exception as e:
                    print(f'[nfpn] *** WARN *** failed to convert module to {fmt_name}: {name} {str(e)}')
                    break
                
                current = module
                names = name.split('.')
                
                while len(names) != 1:
                    current = getattr(current, names.pop(0))
                
                setattr(current, names[0], new_mod)
                break
    
    return module
