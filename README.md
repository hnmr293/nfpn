# nfpn

Narrowed floating point number format for Pytorch. Specifically intended for application to the Stable diffusion.

## VRAM usage and performance with FP12

```
[Generation Info]
Base Model: Animagine XL 3.0 (fp16)
Prompt: close up of a cute girl sitting in flower garden, insanely frilled white dress, absurdly long brown hair, small silver tiara, long sleeves highneck dress
Negative Prompt: (low quality, worst quality:1.4)
```

```python
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
with cuda_profiler('cuda:0') as prof:
    pipe.unet = pipe.unet.to('cuda:0')
print(prof['memory'], prof['time'])
```

|     | VRAM usage of U-Net (MiB) | Peak VRAM usage on sampling (MiB) | Sampling time (ms/step) |
| --- | --- | --- | --- |
| Original | 4998.2 (1.00) | 7869.4 (1.00) | **1245.2 (1.00)** |
| Replace all attn's Linear layers | 4541.5 (0.91) | 7411.5 (0.94) | 1261.7 (1.01) |
| Replace all `Linear` layers | 3890.4 (0.78) | 6760.1 (0.86) | 1344.9 (1.08) |
| Replace all `Linear` and `Conv2d` layers | **3737.8 (0.75)** | **6613.6 (0.84)** | 1369.4 (1.10) |

![image](./images/illust.png)

## Usage

See [example.py](./examples/example.py) `to_fp12` function.

