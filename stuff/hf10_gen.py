import torch
from nfpn import hf10_to_fp16

ns = torch.arange(0, 0b11_1111_1111 + 1, dtype=torch.int16, device='cuda:0')

es = ((ns >> 8) & 0b11).to(dtype=torch.uint8).view((-1, 4))
es.bitwise_left_shift_(torch.tensor([6, 4, 2, 0], device='cuda:0'))
es = es.sum(dim=-1, dtype=torch.uint8)

fs = (ns & 0b1111_1111).to(dtype=torch.uint8)

vs = hf10_to_fp16(es, fs).view(dtype=torch.int16)
vs = vs.view((-1, 16))

for j, row in enumerate(vs):
    row = ', '.join([f'0b{x.item()&0xffff:016b}' for x in row])
    print(row)
