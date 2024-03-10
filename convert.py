import torch

# 0000_1111_1111
FP12_MAX = 0.5 * (1+0.5+0.25+0.125+0.0625+0.03125)

# 0000_0000_0000
FP12_MIN = 2 ** (-15)


def to_fp12(data: torch.Tensor):
    data = data.to(dtype=torch.float16)
    
    assert (data.abs() <= FP12_MAX).all().item(), f'max value = {data.abs().max().item()}'
    
    # fp16: sEEE_EEff_ffff_ffff
    #
    # fp12 type-a: sEEE_ffff_ffff
    #                where EEE != 000
    #       mantissa = E-12 = -5..-11
    # fp12 type-b: s000_0GGf_ffff
    #       mantissa = G-15 = -12..-15
    # fp12 type-c: s000_1HHf_ffff
    #       mantissa = H-4 = -1..-4
    #
    # * significant bits
    #       0000_0000 = 1 + 0
    #       1000_0000 = 1 + 0.5
    #       0100_0000 = 1 + 0.25
    #       0010_0000 = 1 + 0.125
    #       0001_0000 = 1 + 0.0625
    #       0000_1000 = 1 + 0.03125
    #       0000_0100 = 1 + 0.015625
    #       0000_0010 = 1 + 0.0078125
    #       0000_0001 = 1 + 0.00390625
    #
    
    idata = data.view(dtype=torch.int16).view(size=(-1,))
    #assert idata.numel() % 2 == 0
    
    mask_s = 0b1000_0000_0000_0000
    mask_e = 0b0111_1100_0000_0000
    mask_f = 0b0000_0011_1111_1111
    
    # sign
    s = ((idata & mask_s) >> 12).to(dtype=torch.uint8) & 0b1000
    
    # exponential
    e = ((idata & mask_e) >> 10) - 15
    e = e.clamp(-15, -1)
    
    E_mask = torch.logical_and(-11 <= e, e <= -5)
    
    E = (e + 12).to(dtype=torch.int8).view(dtype=torch.uint8)
    E[~E_mask] = 0
    E += s
    
    E[::2] <<= 4
    E[::2] += E[1::2]
    E = E[::2]
    
    # significants
    f = idata & mask_f
    F = (f >> 2).to(dtype=torch.uint8)
    
    ## significants of type-b
    G_mask = e < -11
    Ge = (e + 15).to(dtype=torch.int8).view(dtype=torch.uint8)
    Ge = Ge.clamp(0, 3)
    F[G_mask] = (F[G_mask] >> 3) + (Ge[G_mask] << 5)
    
    ## significants of type-c
    H_mask = -5 < e
    He = (e + 4).to(dtype=torch.int8).view(dtype=torch.uint8)
    He = He.clamp(0, 3) + 0b100
    F[H_mask] = (F[H_mask] >> 3) + (He[H_mask] << 5)
    
    return E, F
    

def fp12_to_fp16(exp: torch.Tensor, frac: torch.Tensor):
    assert exp.dtype == torch.uint8
    assert frac.dtype == torch.uint8
    assert exp.ndim == 1
    assert frac.ndim == 1
    assert exp.size(0) * 2 == frac.size(0)
    
    FP16 = torch.zeros(frac.shape, dtype=torch.int16, device=frac.device)
    
    exp = exp[..., None].expand(size=(-1,2)).contiguous().view((-1,))
    exp[::2] >>= 4
    exp[1::2] &= 0b0000_1111
    
    # sign
    FP16[:] = exp & 0b0000_1000
    FP16 <<= 12
    
    # exponential
    exp_e = (exp & 0b0111).to(dtype=torch.int16)
    exp_gh = ((frac >> 5) & 0b11).to(dtype=torch.int16)
    
    E_mask = exp_e != 0
    G_mask = torch.logical_and(exp_e == 0, (frac >> 7) == 0)
    H_mask = torch.logical_and(exp_e == 0, (frac >> 7) == 1)
    
    # EEE | EEE+3 | << 10               | e      |
    # --- | ----- | ------------------- | ------ |
    # 001 | 0100  | s001_0000_0000_0000 | 00100 = 4 -> -11
    
    # GG | << 10               | e      |
    # ---| ------------------- | ------ |
    # 00 | s000_0000_0000_0000 | 00000 = 0 -> -15
    # 01 | s000_0100_0000_0000 | 00001 = 1 -> -14
    # 10 | s000_1000_0000_0000 | 00010 = 2 -> -13
    # 11 | s000_1100_0000_0000 | 00011 = 3 -> -12
    
    # HH | HH+11 | << 10               | e      |
    # ---| ----  | ------------------- | ------ |
    # 00 | 1011  | s010_1100_0000_0000 | 01011 = 11 -> -4
    # 01 | 1100  | s011_0000_0000_0000 | 01100 = 12 -> -3
    # 10 | 1101  | s011_0100_0000_0000 | 01101 = 13 -> -2
    # 11 | 1110  | s011_1000_0000_0000 | 01110 = 14 -> -1
    
    FP16 += (
        ## type-a
        E_mask * (exp_e + 3) +
        #                 ^ EEE - 12 = -15 + a <=> a = EEE + 3
        ## type-b
        G_mask * exp_gh +
        #              ^ GG - 15 = -15 + a <=> a = GG
        ## type-c
        H_mask * (exp_gh + 11)
        #                  ^ HH - 4 = -15 + a <=> a = HH + 11
    ) << 10
    
    # significants
    FP16 += (
        ## type-a
        E_mask * frac +
        ## type-b and type-c
        (G_mask + H_mask) * (frac << 3)
    ).to(dtype=torch.int16) << 2  # TODO pad random value?
    
    return FP16.view(dtype=torch.float16)
