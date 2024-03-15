import os
import contextlib
import torch
from diffusers import (
    DiffusionPipeline,
    StableDiffusionXLPipeline,
    EulerAncestralDiscreteScheduler,
)

import nfpn


# Model setting
PATH_TO_MODEL = 'D:/sd/models/SDXL/animagineXLV3_v30.safetensors'

# Prompts
PROMPT = '1girl, cute princess, frilled white dress, long sleeves highneck dress, elaborated lace, brown hair, sitting, tiara, red eyes, eyelashes, twinkle eyes, smile, flower garden, cinematic lighting, necklace, masterpiece, best quality'
NEGATIVE_PROMPT = 'nsfw, collarbone, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name, bad eyes'

# Generation settings
NUM_IMAGES = 4
WIDTH = 1024
HEIGHT = 1024
STEPS = 30
CFG = 6.0
SEED = 1

# 
# HF settings
# 
USE_HF = False
HF_BITS = 8
HF_ONLY_ATTN = False
HF_APPLY_LINEAR = True
HF_APPLY_CONV = False

# Other settings
IMAGE_DIR = './'
DEVICE = 'cuda:0'
USE_AMP = False


# ==============================================================================
# Model loading
# ==============================================================================

def free_memory():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def replace_hf(pipe: DiffusionPipeline):
    for name, mod in pipe.unet.named_modules():
        if HF_ONLY_ATTN and 'attn' not in name:
            continue
        to_hf(mod)
    return pipe


def to_hf(module: torch.nn.Module):
    fn = None
    
    if HF_BITS == 8:
        fn = nfpn.nn.to_hf8
    elif HF_BITS == 10:
        fn = nfpn.nn.to_hf10
    elif HF_BITS == 12:
        fn = nfpn.nn.to_hf12
    else:
        raise ValueError(f'unknown HF_BITS value: {HF_BITS}')
    
    return fn(module)


def load_model(path: str):
    pipe = StableDiffusionXLPipeline.from_single_file(
        path,
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    
    return pipe


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


# ==============================================================================
# Generation
# ==============================================================================

def generate(pipe: DiffusionPipeline, prompt: str, negative_prompt: str, seed: int, device: str, use_amp: bool = False):
    import contextlib
    import torch.amp
    
    context = (
        torch.amp.autocast_mode.autocast if use_amp
        else contextlib.nullcontext
    )

    with torch.no_grad(), context(device):
        rng = torch.Generator(device=device)
        if 0 <= seed:
            rng = rng.manual_seed(seed)
        
        latents, *_ = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=WIDTH,
            height=HEIGHT,
            num_inference_steps=STEPS,
            guidance_scale=CFG,
            num_images_per_prompt=NUM_IMAGES,
            generator=rng,
            device=device,
            return_dict=False,
            output_type='latent',
        )
        
        return latents
        
def save_image(pipe, latents):
    with torch.no_grad():
        needs_upcasting = pipe.vae.dtype == torch.float16 and pipe.vae.config.force_upcast
        if needs_upcasting:
            pipe.upcast_vae()
            latents = latents.to(next(iter(pipe.vae.post_quant_conv.parameters())).dtype)
        images = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
        images = pipe.image_processor.postprocess(images, output_type='pil')
    
    for i, image in enumerate(images):
        image.save(os.path.join(IMAGE_DIR, f'{i:02d}.png'))


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    
    def _str_to_bool(v):
        return str(v) == 'True'
    
    p.add_argument('--model', type=str, required=True)
    p.add_argument('--prompt', type=str, default='')
    p.add_argument('--negative_prompt', type=str, default='')
    p.add_argument('--num_images', type=int, default=4)
    p.add_argument('--width', type=int, default=1024)
    p.add_argument('--height', type=int, default=1024)
    p.add_argument('--steps', type=int, default=30)
    p.add_argument('--cfg', type=float, default=6.0)
    p.add_argument('--seed', type=int, default=-1)
    p.add_argument('--image_dir', type=str, default='./')
    
    p.add_argument('--hf_bits', type=int, default=0)
    p.add_argument('--hf_only_attn', type=_str_to_bool, default=False)
    p.add_argument('--hf_linear', type=_str_to_bool, default=True)
    p.add_argument('--hf_conv', type=_str_to_bool, default=False)
    
    args = p.parse_args()
    
    PATH_TO_MODEL = args.model
    PROMPT = args.prompt
    NEGATIVE_PROMPT = args.negative_prompt
    NUM_IMAGES = args.num_images
    WIDTH = args.width
    HEIGHT = args.height
    STEPS = args.steps
    CFG = args.cfg
    SEED = args.seed
    IMAGE_DIR = args.image_dir
    USE_HF = args.hf_bits != 0
    HF_BITS = args.hf_bits
    HF_ONLY_ATTN = args.hf_only_attn
    HF_APPLY_LINEAR = args.hf_linear
    HF_APPLY_CONV = args.hf_conv
    
    os.makedirs(IMAGE_DIR, exist_ok=True)
    
    
    pipe = load_model(PATH_TO_MODEL)
    
    if USE_HF:
        pipe = replace_hf(pipe)
    
    free_memory()
    with cuda_profiler(DEVICE) as prof:
        pipe.unet = pipe.unet.to(DEVICE)
    print('LOAD VRAM', prof['memory'])
    print('LOAD TIME', prof['time'])
    
    pipe = pipe.to(DEVICE)
    pipe.vae = pipe.vae.to('cpu')
    
    free_memory()
    with cuda_profiler(DEVICE) as prof:
        latents = generate(pipe, PROMPT, NEGATIVE_PROMPT, SEED, DEVICE, USE_AMP)
    print('UNET VRAM', prof['memory'])
    print('UNET TIME', prof['time'])
    
    pipe.vae = pipe.vae.to(DEVICE)
    pipe.vae.enable_slicing()
    
    free_memory()
    save_image(pipe, latents)
