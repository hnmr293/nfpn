import torch
from diffusers import DiffusionPipeline, StableDiffusionXLPipeline

from fp12 import Linear, Conv2d


PATH_TO_MODEL = "D:/sd/models/SDXL/animagineXLV3_v30.safetensors"
PROMPT = "close up of a cute girl sitting in flower garden, insanely frilled white dress, absurdly long brown hair, small silver tiara, long sleeves highneck dress"
NEGATIVE_PROMPT = "(low quality, worst quality:1.4)"
SEED = 1

DEVICE = 'cuda:0'
USE_AMP = False

FP12_ONLY_ATTN = True
FP12_APPLY_LINEAR = True
FP12_APPLY_CONV = False


# ==============================================================================
# Model loading
# ==============================================================================

def free_memory():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def to_fp12(module: torch.nn.Module):
    target_modules = []
    
    if FP12_APPLY_LINEAR:
        target_modules.append((torch.nn.Linear, Linear))
    
    if FP12_APPLY_CONV:
        target_modules.append((torch.nn.Conv2d, Conv2d))
    
    for name, mod in list(module.named_children()):
        for orig_class, fp12_class in target_modules:
            if isinstance(mod, orig_class):
                try:
                    new_mod = fp12_class(mod)
                except Exception as e:
                    print(f'  -> failed: {name} {str(e)}')
                    continue
                
                delattr(module, name)
                del mod
                
                setattr(module, name, new_mod)


def load_model(path: str, device: str):
    pipe = StableDiffusionXLPipeline.from_single_file(
        path,
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to(device)
    pipe.enable_vae_slicing()
    return pipe

def replace_fp12(pipe: DiffusionPipeline):
    for name, mod in pipe.unet.named_modules():
        if FP12_ONLY_ATTN and 'attn' not in name:
            continue
        print('[fp12] REPLACE', name)
        to_fp12(mod)
    return pipe


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
        
        images, *_ = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=1024,
            height=1024,
            num_inference_steps=20,
            guidance_scale=3.0,
            num_images_per_prompt=4,
            generator=rng,
            return_dict=False,
        )
        
        for i, image in enumerate(images):
            image.save(f'{i:02d}.png')


if __name__ == '__main__':
    free_memory()
    pipe = load_model(PATH_TO_MODEL, DEVICE)
    pipe = replace_fp12(pipe)
    generate(pipe, PROMPT, NEGATIVE_PROMPT, SEED, DEVICE, USE_AMP)
