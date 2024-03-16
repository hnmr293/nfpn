import torch

# 0111_1111
HF8X_MAX = 1 * (1+0.5+0.25+0.125)

# 0000_0000
HF8X_MIN = 2 ** (-15)


def to_hf8x(data: torch.Tensor):
    data = data.to(dtype=torch.float16)
    
    assert (data.abs() <= HF8X_MAX).all().item(), f'max value = {data.abs().max().item()}'
    
    # fp16: sEEE_EEff_ffff_ffff
    #
    # hf8x: sEEE_Efff
    #       exponential = E-15 = -15..0
    #
    # * significant bits
    #       000 = 1 + 0
    #       100 = 1 + 0.5
    #       010 = 1 + 0.25
    #       001 = 1 + 0.125
    #
    
    idata = data.view(dtype=torch.int16).view(size=(-1,))
    
    mask_s = 0b1000_0000_0000_0000
    mask_e = 0b0111_1100_0000_0000
    mask_f = 0b0000_0011_1111_1111
    
    # sign
    s = (idata >> 8).to(dtype=torch.uint8) & 0b1000_0000
    
    # exponential
    e = ((idata & mask_e) >> 10).to(dtype=torch.uint8)
    e = e.clamp(0, 15) # 4 bits
    e.bitwise_left_shift_(3)
    
    # significants
    f = ((idata & mask_f) >> 7).to(dtype=torch.uint8)
    
    return s + e + f

def hf8x_to_fp16(hf8x: torch.Tensor):
    assert hf8x.dtype == torch.uint8
    assert hf8x.ndim == 1
    
    mask_s  = 0b1000_0000
    mask_ef = 0b0111_1111
    
    s = (hf8x & mask_s).to(dtype=torch.int16)
    s.bitwise_left_shift_(8)
    ef = (hf8x & mask_ef).to(dtype=torch.int16)
    ef.bitwise_left_shift_(7)
    
    FP16 = s.add_(ef)
    
    return FP16.view(dtype=torch.float16)
