import torch
from nfpn import hf12_to_fp16

ns = torch.arange(0, 0b1111_1111_1111 + 1, dtype=torch.int16, device='cuda:0')

es = ((ns >> 8) & 0b1111).to(dtype=torch.uint8)
es[::2] <<= 4
es[::2] += es[1::2]
es = es[::2]

fs = (ns & 0b1111_1111).to(dtype=torch.uint8)

vs = hf12_to_fp16(es, fs).view(dtype=torch.int16)
vs = vs.view((-1, 16))

for j, row in enumerate(vs):
    row = ', '.join([f'0b{x.item()&0xffff:016b}' for x in row])
    print(row)
