from collections import defaultdict
import torch
from diffusers import StableDiffusionXLPipeline

path = 'D:/sd/models/SDXL/animagineXLV3_v30.safetensors'

unet = StableDiffusionXLPipeline.from_single_file(path, torch_dtype=torch.float16).unet.cuda()

result = defaultdict(lambda: 0)
result2 = defaultdict(lambda: 0)

for name, mod in unet.named_modules():
    if isinstance(mod, torch.nn.Linear):
        shape = mod.weight.shape
        w = mod.weight.clone().view((-1,)).view(dtype=torch.int16)
        w.bitwise_right_shift_(10)
        w.bitwise_and_(0b011111)
        vs, cs = w.unique(return_counts=True)
        for v, c in zip(vs, cs):
            result[v.item() - 15] += c.item()
        
        w = w.view(shape)
        for row in w:
            min = row.min()
            row[1::].sub_(min)
        vs, cs = w.unique(return_counts=True)
        for v, c in zip(vs, cs):
            result2[v.item()] += c.item()
        
        

for v, c in result.items():
    print(f'{v}: {c}')

print()

for v, c in result2.items():
    print(f'{v}: {c}')

# -15:   5160891
# -14:   5171614
# -13:  10327845
# -12:  20646282
# -11:  41211986
# -10:  81813038
# -9:  159720956
# -8:  301214975
# -7:  508863860
# -6:  643685247
# -5:  411904731
# -4:   42789799
# -3:     132446
# -2:       3805
# -1:        197
# -0:          8


# 0:    5510995
# 1:    5520604
# 2:   11026937
# 3:   22040284
# 4:   43969430
# 5:   87233838
# 6:  170138594
# 7:  318533261
# 8:  523975358
# 9:  627887717
# 10: 378977929
# 11:  37733441
# 12:     96232
# 13:      2881
# 14:       171
# 15:         8
