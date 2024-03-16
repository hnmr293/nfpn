import torch
from nfpn.convert import hf8_to_fp16

ns = torch.arange(0, 1 << 8, dtype=torch.uint8, device='cuda:0')

vs = hf8_to_fp16(ns).view(dtype=torch.int16)
vs = vs.view((-1, 16))

for j, row in enumerate(vs):
    row = ', '.join([f'0b{x.item()&0xffff:016b}' for x in row])
    print(row)
