## glide_util.py
# Utilities for tokenizing, padding, and batching data and sampling from GLIDE.

import os
from typing import Tuple

import PIL
import numpy as np
import torch as th
from glide_finetune.train_util import pred_to_pil
from glide_text2im.download import load_checkpoint
from glide_text2im.model_creation import (
    create_gaussian_diffusion,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_upsampler,
)
from glide_text2im.tokenizer.bpe import Encoder
from glide_finetune.set2im_models import (
    SuperResInpaintSet2ImUnet,
    InpaintSet2ImUNet,
    SuperResSet2ImUNet,
    Set2ImUNet,
)
from vit import sVIT
import numpy as np
MODEL_TYPES = ["base", "upsample", "base-inpaint", "upsample-inpaint"]

def create_set_model(
    vit_options,
    image_size,
    num_channels,
    num_res_blocks,
    channel_mult,
    attention_resolutions,
    num_heads,
    num_head_channels,
    num_heads_upsample,
    use_scale_shift_norm,
    dropout,
    text_ctx,
    xf_width,
    xf_layers,
    xf_heads,
    xf_final_ln,
    xf_padding,
    resblock_updown,
    use_fp16,
    cache_text_emb,
    inpaint,
    super_res,
):
    if channel_mult == "":
        if image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
    else:
        channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))
        assert 2 ** (len(channel_mult) + 2) == image_size

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    if inpaint and super_res:
        model_cls = SuperResInpaintSet2ImUnet
    elif inpaint:
        model_cls = InpaintSet2ImUNet
    elif super_res:
        model_cls = SuperResSet2ImUNet
    else:
        model_cls = Set2ImUNet
    return model_cls(
        vit_options=vit_options,
        xf_width=xf_width,
        image_size=image_size,
        num_channels=num_channels,
        xf_final_ln=xf_final_ln,
        in_channels=3,
        model_channels=num_channels,
        out_channels=6,
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        use_fp16=use_fp16,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        cache_text_emb=cache_text_emb,
    )

def create_model_and_diffusion(
    vit_options,
    image_size,
    num_channels,
    num_res_blocks,
    channel_mult,
    attention_resolutions,
    num_heads,
    num_head_channels,
    num_heads_upsample,
    use_scale_shift_norm,
    dropout,
    text_ctx,
    xf_width,
    xf_layers,
    xf_heads,
    xf_final_ln,
    xf_padding,
    resblock_updown,
    use_fp16,
    cache_text_emb,
    inpaint,
    super_res,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    **kwargs,
):
    model = create_set_model(
        vit_options=vit_options,
        image_size=image_size,
        num_channels=num_channels,
        num_res_blocks=num_res_blocks,
        channel_mult=channel_mult,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
        text_ctx=text_ctx,
        xf_width=xf_width,
        xf_layers=xf_layers,
        xf_heads=xf_heads,
        xf_final_ln=xf_final_ln,
        xf_padding=xf_padding,
        resblock_updown=resblock_updown,
        use_fp16=use_fp16,
        cache_text_emb=cache_text_emb,
        inpaint=inpaint,
        super_res=super_res,
    )
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        noise_schedule=noise_schedule,
        timestep_respacing=timestep_respacing,
    )
    return model, diffusion

def get_uncond_tokens_mask(tokenizer: Encoder):
    uncond_tokens, uncond_mask = tokenizer.padded_tokens_and_mask([], 128)
    return th.tensor(uncond_tokens), th.tensor(uncond_mask, dtype=th.bool)


def get_tokens_and_mask(
    tokenizer: Encoder, prompt: str = "", context_len: int = 128
) -> Tuple[th.tensor, th.tensor]:
    if len(prompt) == 0:
        return get_uncond_tokens_mask(tokenizer)
    else:
        tokens = tokenizer.encode(prompt)
        tokens, mask = tokenizer.padded_tokens_and_mask(tokens, context_len)
        tokens = th.tensor(tokens)  # + uncond_tokens)
        mask = th.tensor(mask, dtype=th.bool)  # + uncond_mask, dtype=th.bool)
        return tokens, mask

def set_encoder_options(dim=1024):
    return {
        "image_size": 256,
        "patch_size": 16,
        "num_classes": dim,
        "channels": 3,
        "dim": dim,
        "depth": 6,
        "heads": 16,
        "mlp_dim": 2048,
        "dropout": 0.1,
        "emb_dropout": 0.1,
    }

def load_model(
    glide_path: str = "",
    vit_options: dict = {},
    use_fp16: bool = False,
    freeze_transformer: bool = False,
    freeze_diffusion: bool = False,
    activation_checkpointing: bool = False,
    model_type: str = "base"
):
    assert model_type in MODEL_TYPES, f"Model must be one of {MODEL_TYPES}. Exiting."
    if model_type in ["base", "base-inpaint"]:
        options = model_and_diffusion_defaults()
    elif model_type in ["upsample", "upsample-inpaint"]:
        options = model_and_diffusion_defaults_upsampler()
    if "inpaint" in model_type:
        options["inpaint"] = True
    options["use_fp16"] = use_fp16
    glide_model, glide_diffusion = create_model_and_diffusion(vit_options=vit_options, **options)
    if activation_checkpointing:
        glide_model.use_checkpoint = True

    glide_model.requires_grad_(True)
    # if freeze_transformer:
    #     glide_model.transformer.requires_grad_(False)
    #     glide_model.transformer_proj.requires_grad_(False)
    #     glide_model.token_embedding.requires_grad_(False)
    #     glide_model.padding_embedding.requires_grad_(False)
    #     glide_model.positional_embedding.requires_grad_(False)
    if freeze_diffusion:
        glide_model.out.requires_grad_(False)
        glide_model.input_blocks.requires_grad_(False)
        glide_model.middle_block.requires_grad_(False)
        glide_model.output_blocks.requires_grad_(False)
    if len(glide_path) > 0:  # user provided checkpoint
        assert os.path.exists(glide_path), "glide path does not exist"
        model_dict = glide_model.state_dict()
        weights = th.load(glide_path, map_location="cpu")
        weights = {k: v for k, v in weights.items() if
                       (k in model_dict) and (model_dict[k].shape == weights[k].shape)}
        glide_model.load_state_dict(weights, strict=False)
    else:  # use default checkpoint from openai
        print('Using openai checkpoint')
        model_dict = glide_model.state_dict()
        open_ai_dict = load_checkpoint(model_type, "cpu")
        open_ai_dict = {k: v for k, v in open_ai_dict.items() if
                       (k in model_dict) and (model_dict[k].shape == open_ai_dict[k].shape)}
        glide_model.load_state_dict(
            open_ai_dict, strict=False
        )  # always load to cpu, saves memory
    if use_fp16:
        glide_model.convert_to_fp16()
        print("Converted to fp16, likely gradients will explode")
    return glide_model, glide_diffusion, options

def load_set_encoder(model_path, context_dim=1280):
    options = set_encoder_options(dim=context_dim)
    encoder = sVIT(**options)
    if model_path != "":
        model_state_dict = load_state_dict(model_path, map_location="cpu")
        encoder.load_state_dict(model_state_dict, strict=False)
    return encoder

def read_image(path: str, shape: Tuple[int, int]):
    pil_img = PIL.Image.open(path).convert('RGB')
    pil_img = pil_img.resize(shape, resample=PIL.Image.BICUBIC)
    img = np.array(pil_img)
    return th.from_numpy(img)[None].permute(0, 3, 1, 2).float() / 127.5 - 1

def read_image_with_mask(path: str, size: int = 256) -> Tuple[th.Tensor, th.Tensor]:
    pil_img = PIL.Image.open(path).convert('RGB')
    pil_img = np.array(pil_img)
    w = pil_img.shape[1] // 3
    mask, pil_img = pil_img[:, :w], pil_img[:, w:2*w]
    pil_img = PIL.Image.fromarray(pil_img).resize((size, size), resample=Image.BICUBIC)
    img = np.array(pil_img)
    return th.from_numpy(img)[None].permute(0, 3, 1, 2).float() / 127.5 - 1

# Sample from the base model.

@th.inference_mode()
def sample(
    glide_model,
    glide_options,
    side_x,
    side_y,
    prompt="",
    batch_size=1,
    guidance_scale=4,
    device="cpu",
    prediction_respacing="100",
    upsample_enabled=False,
    image_to_upsample='',
    upsample_temp=0.997,
    **kwargs
):
    glide_model.del_cache()
    eval_diffusion = create_gaussian_diffusion(
        steps=glide_options["diffusion_steps"],
        noise_schedule=glide_options["noise_schedule"],
        timestep_respacing=prediction_respacing,
    )
    # Create the text tokens to feed to the model.
    # Create the classifier-free guidance tokens (empty)
    full_batch_size = batch_size * 2

    # Pack the tokens together into model kwargs.
    model_kwargs = dict(
        **kwargs
    )

    def cfg_model_fn(x_t, ts, **kwargs):
        half = x_t[: len(x_t) // 2]
        combined = th.cat([half, half], dim=0)
        model_out = glide_model(combined, ts, **kwargs)
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = th.split(eps, len(eps) // 2, dim=0)
        beta = eval_diffusion.betas[
            int(
                ts.flatten()[0].item()
                / glide_options["diffusion_steps"]
                * len(eval_diffusion.betas)
            )
        ]
        half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
        eps = th.cat([half_eps, half_eps], dim=0)
        current_prediction_pil = pred_to_pil(
            (x_t - eps * (beta**0.5))[:batch_size]
        )
        current_prediction_pil.save("current_prediction.png")
        return th.cat([eps, rest], dim=1)

    model_fn = cfg_model_fn # so we use CFG for the base model.
    if upsample_enabled:
        # assert image_to_upsample != '', "You must specify a path to an image to upsample."
        # low_res_samples = read_image_with_mask(image_to_upsample, size=(side_x, side_y))
        # model_kwargs['low_res'] = low_res_samples
        noise = th.randn((batch_size, 3, side_y, side_x), device=device) * upsample_temp
        model_kwargs['noise'] = noise
        model_fn = glide_model # just use the base model, no need for CFG.

    samples = eval_diffusion.p_sample_loop(
        model_fn,
        (full_batch_size, 3, side_y, side_x),  # only thing that's changed
        device=device,
        clip_denoised=True,
        progress=True,
        model_kwargs=model_kwargs,
        cond_fn=None,
    )[:batch_size]
    glide_model.del_cache()
    return samples