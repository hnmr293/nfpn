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
    

def fp12_to_fp16_2(exp: torch.Tensor, frac: torch.Tensor):
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


# s_eee_fff (fp12) -> S_EEEEE (fp16)
EXP_MAP = torch.tensor([
    # -15        -14        -13        -12        -4         -3         -2         -1
    # 0_000_000  0_000_001  0_000_010  0_000_011  0_000_100  0_000_101  0_000_110  0_000_111
    0b0_00000, 0b0_00001, 0b0_00010, 0b0_00011, 0b0_01011, 0b0_01100, 0b0_01101, 0b0_01110, 
    
    # -11
    # 0_001_000  0_001_001  0_001_010  0_001_011  0_001_100  0_001_101  0_001_110  0_001_111
    0b0_00100, 0b0_00100, 0b0_00100, 0b0_00100, 0b0_00100, 0b0_00100, 0b0_00100, 0b0_00100, 
    
    # -10
    # 0_010_000  0_010_001  0_010_010  0_010_011  0_010_100  0_010_101  0_010_110  0_010_111
    0b0_00101, 0b0_00101, 0b0_00101, 0b0_00101, 0b0_00101, 0b0_00101, 0b0_00101, 0b0_00101, 
    
    # -9
    # 0_011_000  0_011_001  0_011_010  0_011_011  0_011_100  0_011_101  0_011_110  0_011_111
    0b0_00110, 0b0_00110, 0b0_00110, 0b0_00110, 0b0_00110, 0b0_00110, 0b0_00110, 0b0_00110, 
    
    # -8
    # 0_100_000  0_100_001  0_100_010  0_100_011  0_100_100  0_100_101  0_100_110  0_100_111
    0b0_00111, 0b0_00111, 0b0_00111, 0b0_00111, 0b0_00111, 0b0_00111, 0b0_00111, 0b0_00111, 
    
    # -7
    # 0_101_000  0_101_001  0_101_010  0_101_011  0_101_100  0_101_101  0_101_110  0_101_111
    0b0_01000, 0b0_01000, 0b0_01000, 0b0_01000, 0b0_01000, 0b0_01000, 0b0_01000, 0b0_01000, 
    
    # -6
    # 0_110_000  0_110_001  0_110_010  0_110_011  0_110_100  0_110_101  0_110_110  0_110_111
    0b0_01001, 0b0_01001, 0b0_01001, 0b0_01001, 0b0_01001, 0b0_01001, 0b0_01001, 0b0_01001, 
    
    # -5
    # 0_111_000  0_111_001  0_111_010  0_111_011  0_111_100  0_111_101  0_111_110  0_111_111
    0b0_01010, 0b0_01010, 0b0_01010, 0b0_01010, 0b0_01010, 0b0_01010, 0b0_01010, 0b0_01010, 
    
    # -15        -14        -13        -12        -4         -3         -2         -1
    # 1_000_000  1_000_001  1_000_010  1_000_011  1_000_100  1_000_101  1_000_110  1_000_111
    0b1_00000, 0b1_00001, 0b1_00010, 0b1_00011, 0b1_01011, 0b1_01100, 0b1_01101, 0b1_01110, 
    
    # -11
    # 1_001_000  1_001_001  1_001_010  1_001_011  1_001_100  1_001_101  1_001_110  1_001_111
    0b1_00100, 0b1_00100, 0b1_00100, 0b1_00100, 0b1_00100, 0b1_00100, 0b1_00100, 0b1_00100, 
    
    # -10
    # 1_010_000  1_010_001  1_010_010  1_010_011  1_010_100  1_010_101  1_010_110  1_010_111
    0b1_00101, 0b1_00101, 0b1_00101, 0b1_00101, 0b1_00101, 0b1_00101, 0b1_00101, 0b1_00101, 
    
    # -9
    # 1_011_000  1_011_001  1_011_010  1_011_011  1_011_100  1_011_101  1_011_110  1_011_111
    0b1_00110, 0b1_00110, 0b1_00110, 0b1_00110, 0b1_00110, 0b1_00110, 0b1_00110, 0b1_00110, 
    
    # -8
    # 1_100_000  1_100_001  1_100_010  1_100_011  1_100_100  1_100_101  1_100_110  1_100_111
    0b1_00111, 0b1_00111, 0b1_00111, 0b1_00111, 0b1_00111, 0b1_00111, 0b1_00111, 0b1_00111, 
    
    # -7
    # 1_101_000  1_101_001  1_101_010  1_101_011  1_101_100  1_101_101  1_101_110  1_101_111
    0b1_01000, 0b1_01000, 0b1_01000, 0b1_01000, 0b1_01000, 0b1_01000, 0b1_01000, 0b1_01000, 
    
    # -6
    # 1_110_000  1_110_001  1_110_010  1_110_011  1_110_100  1_110_101  1_110_110  1_110_111
    0b1_01001, 0b1_01001, 0b1_01001, 0b1_01001, 0b1_01001, 0b1_01001, 0b1_01001, 0b1_01001, 
    
    # -5
    # 1_111_000  1_111_001  1_111_010  1_111_011  1_111_100  1_111_101  1_111_110  1_111_111
    0b1_01010, 0b1_01010, 0b1_01010, 0b1_01010, 0b1_01010, 0b1_01010, 0b1_01010, 0b1_01010, 
], dtype=torch.int16) << 10

EXP_MASK = torch.tensor([
    0b1111_0000,
    0b0000_1111
], dtype=torch.uint8)

def fp12_to_fp16(exp: torch.Tensor, frac: torch.Tensor):
    global EXP_MAP, EXP_MASK
    
    assert exp.dtype == torch.uint8
    assert frac.dtype == torch.uint8
    assert exp.ndim == 1
    assert frac.ndim == 1
    assert exp.size(0) * 2 == frac.size(0)
    
    if EXP_MAP.device != exp.device:
        EXP_MAP = EXP_MAP.to(exp.device)
    if EXP_MASK.device != exp.device:
        EXP_MASK = EXP_MASK.to(exp.device)
    
    exp = exp.repeat_interleave(2).view((-1,2))
    exp.bitwise_and_(EXP_MASK)
    exp[..., 0] >>= 1
    exp[..., 1] <<= 3
    exp = exp.view((-1,))
    # 0111_1000 0111_1000
    
    E_mask = torch.logical_and(exp != 0, exp != 0b0100_0000).to(dtype=torch.int16)
    
    # take upper 6-bits from bitmap
    exp = torch.take(EXP_MAP, (exp + (frac >> 5)).long())
    
    f_mask = (0b1110_0000 * E_mask) + 0b0001_1111
    f_shift = 5 - 3 * E_mask
    frac = frac.to(dtype=torch.int16)
    frac.bitwise_and_(f_mask)
    frac.bitwise_left_shift_(f_shift)
    
    FP16 = exp.add_(frac)
    
    return FP16.view(dtype=torch.float16)
