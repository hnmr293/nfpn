import torch

# 00_0011_1111
FP10_MAX = 0.5 * (1+0.5+0.25+0.125)

# 00_0000_0000
FP10_MIN = 2 ** (-15)


def to_fp10(data: torch.Tensor):
    data = data.to(dtype=torch.float16)
    
    assert (data.abs() <= FP10_MAX).all().item(), f'max value = {data.abs().max().item()}'
    
    # fp16: sEEE_EEff_ffff_ffff
    #
    # fp10 type-a: sE_EEff_ffff  where EEE != 000
    #       mantissa = E-12 = -5..-11
    # fp10 type-b: s0_00ff_f0GG
    #       mantissa = G-15 = -12..-15
    # fp10 type-c: s0_00ff_f1HH
    #       mantissa = H-4 = -1..-4
    #
    # * significant bits
    #       00_0000 = 1 + 0
    #       10_0000 = 1 + 0.5
    #       01_0000 = 1 + 0.25
    #       00_1000 = 1 + 0.125
    #       00_0100 = 1 + 0.0625
    #       00_0010 = 1 + 0.03125
    #       00_0001 = 1 + 0.015625
    #
    
    idata = data.view(dtype=torch.int16).view(size=(-1,))
    assert idata.numel() % 4 == 0
    
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
    
    E1 = (E & 0b1100) >> 2
    E2 = E & 0b0011
    
    E1 = E1.view((-1,4))
    E1.bitwise_left_shift_(torch.tensor([6, 4, 2, 0]))
    E1 = E1.sum(dim=-1, dtype=torch.uint8)
    
    E2 <<= 6
    
    # significants
    f = idata & mask_f
    F = (f >> 4).to(dtype=torch.uint8)
    
    ## significants of type-b
    G_mask = e < -11
    Ge = (e + 15).to(dtype=torch.int8).view(dtype=torch.uint8)
    Ge = Ge.clamp(0, 3)
    F[G_mask] = (F[G_mask] & 0b11_1000) + Ge[G_mask]
    
    ## significants of type-c
    H_mask = -5 < e
    He = (e + 4).to(dtype=torch.int8).view(dtype=torch.uint8)
    He = He.clamp(0, 3)
    F[H_mask] = (F[H_mask] & 0b11_1000) + He[H_mask] + 0b100
    
    F.add_(E2)
    
    return E1, F

def fp10_to_fp16(exp: torch.Tensor, frac: torch.Tensor):
    assert exp.dtype == torch.uint8
    assert frac.dtype == torch.uint8
    assert exp.ndim == 1
    assert frac.ndim == 1
    assert exp.size(0) * 4 == frac.size(0)
    
    FP16 = torch.zeros(frac.shape, dtype=torch.int16, device=frac.device)
    
    exp = exp[..., None].repeat_interleave(4)
    exp.view((-1,4)).bitwise_right_shift_(torch.tensor([6, 4, 2, 0], device=exp.device))
    exp.bitwise_and_(0b11)
    
    # sign
    FP16[:] = exp & 0b10
    FP16 <<= 14
    
    # exponential
    exp_e = (((exp & 1) << 2) + (frac >> 6)).to(dtype=torch.int16)
    exp_gh = (frac & 0b11).to(dtype=torch.int16)
    
    E_mask = exp_e != 0
    G_mask = torch.logical_and(exp_e == 0, (frac & 0b100) == 0)
    H_mask = torch.logical_and(exp_e == 0, (frac & 0b100) == 4)
    
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
        E_mask * (frac & 0b111_111) +
        ## type-b and type-c
        (G_mask + H_mask) * (frac & 0b111_000)
    ).to(dtype=torch.int16) << 4  # TODO pad random value?
    
    return FP16.view(dtype=torch.float16)
