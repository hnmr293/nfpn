# SD-fp12

12-bit floating point weight for Stable diffusion.

## Result

### VRAM usage and performance

```
[Generation Info]
Model: Animagine XL 3.0
Prompt: close up of a cute girl sitting in flower garden, insanely frilled white dress, absurdly long brown hair, small silver tiara, long sleeves highneck dress
Negative Prompt: (low quality, worst quality:1.4)
```

```
# measurement

@contextlib.contextmanager
def cuda_profiler(device: str):
    cuda_start = torch.cuda.Event(enable_timing=True)
    cuda_end = torch.cuda.Event(enable_timing=True)

    obj = {}
    
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    cuda_start.record()
    
    try:
        yield obj
    finally:
        pass

    cuda_end.record()
    torch.cuda.synchronize()
    obj['time'] = cuda_start.elapsed_time(cuda_end)
    obj['memory'] = torch.cuda.max_memory_allocated(device)

pipe = StableDiffusionXLPipeline.from_single_file(...)
with cuda_profiler(DEVICE) as prof:
    pipe.unet = pipe.unet.to('cuda:0')
print(prof['memory'], prof['time'])
```

|     | VRAM usage of U-Net (MiB) | Peak VRAM usage on sampling (MiB) | Sampling time (ms/step) |
| --- | --- | --- | --- |
| Original | 4998.2 | 7869.4 | 1245.2 |
| Replace all attn's Linear layers | 4541.5 | 7411.5 | 1446.0 |
| Replace all `Linear` layers | 3890.4 | 6760.1 | 1744.0 |
| Replace all `Linear` and `Conv2d` layers | 3737.8 | 6613.6 | 1829.5 |

![image](./images/illust.png)

## Usage

See [example.py](./examples/example.py) `to_fp12` function.

