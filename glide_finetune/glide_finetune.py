"""
TODO: Make inference chi's painting vs others
"""
import os
from typing import Tuple

import torch as th
from glide_text2im.respace import SpacedDiffusion
from glide_text2im.text2im_model import Text2ImUNet
from wandb import wandb
import gc
from glide_finetune import glide_util, train_util
import numpy as np
import random

def preprocess_set_imgs(set_imgs):
    for i in range(set_imgs.shape[0]):
        for j in range(set_imgs.shape[1]):
            if random.random() < .2:
                set_imgs[i, j, ...] = 0
        # if random.random() < .05:
        #     set_imgs[i, ...] = 0
    return set_imgs
def base_train_step(
    glide_model: Text2ImUNet,
    glide_diffusion: SpacedDiffusion,
    batch: Tuple[th.Tensor, th.Tensor, th.Tensor],
    device: str,
    inpainting=False
):
    """
    Perform a single training step.

        Args:
            glide_model: The model to train.
            glide_diffusion: The diffusion to use.
            batch: A tuple of (tokens, masks, reals) where tokens is a tensor of shape (batch_size, seq_len), masks is a tensor of shape (batch_size, seq_len) and reals is a tensor of shape (batch_size, 3, side_x, side_y) normalized to [-1, 1].
            device: The device to use for getting model outputs and computing loss.
        Returns:
            The loss.
    """
    set_imgs, reals = [x.to(device) for x in batch]
    set_imgs = preprocess_set_imgs(set_imgs)
    img_masks = None
    if not(img_masks is None):
        inpainting = True
    timesteps = th.randint(
        0, len(glide_diffusion.betas) - 1, (reals.shape[0],), device=device
    )
    noise = th.randn_like(reals, device=device)
    # print('noise max min')
    # print(th.max(noise), th.min(noise))
    x_t = glide_diffusion.q_sample(reals, timesteps, noise=noise).to(device)
    # print('x_t max min')
    # print(th.max(x_t), th.min(x_t))
    _, C = x_t.shape[:2]
    if inpainting:
        model_output = glide_model(
            x_t.to(device),
            timesteps.to(device),
            inpaint_image=reals,
            inpaint_mask=img_masks,
            set_imgs=set_imgs
        )
    else:
        model_output = glide_model(
            x_t.to(device),
            timesteps.to(device),
            set_imgs=set_imgs
        )
    epsilon, _ = th.split(model_output, C, dim=1)
    # print('epsilon max min')
    # print(th.max(epsilon), th.min(epsilon))
    # print('loss')
    # print(th.nn.functional.mse_loss(epsilon, noise.to(device).detach()))
    return th.nn.functional.mse_loss(epsilon, noise.to(device).detach())

def upsample_train_step(
    glide_model: Text2ImUNet,
    glide_diffusion: SpacedDiffusion,
    batch: Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor],
    device: str,
):
    """
    Perform a single training step.

        Args:
            glide_model: The model to train.
            glide_diffusion: The diffusion to use.
            batch: A tuple of (tokens, masks, low_res, high_res) where 
                - tokens is a tensor of shape (batch_size, seq_len), 
                - masks is a tensor of shape (batch_size, seq_len) with dtype torch.bool
                - low_res is a tensor of shape (batch_size, 3, base_x, base_y), normalized to [-1, 1]
                - high_res is a tensor of shape (batch_size, 3, base_x*4, base_y*4), normalized to [-1, 1]
            device: The device to use for getting model outputs and computing loss.
        Returns:
            The loss.
    """
    tokens, masks, low_res_image, high_res_image, high_res_mask = [ x.to(device) for x in batch ]
    if not(high_res_mask is None):
        inpainting = True
    timesteps = th.randint(0, len(glide_diffusion.betas) - 1, (low_res_image.shape[0],), device=device)
    noise = th.randn_like(high_res_image, device=device) # Noise should be shape of output i think
    noised_high_res_image = glide_diffusion.q_sample(high_res_image, timesteps, noise=noise).to(device)
    _, C = noised_high_res_image.shape[:2]
    if inpainting:
        model_output = glide_model(
            noised_high_res_image.to(device),
            timesteps.to(device),
            low_res=low_res_image.to(device),
            tokens=tokens.to(device),
            mask=masks.to(device))
    else:
        model_output = glide_model(
            noised_high_res_image.to(device),
            timesteps.to(device),
            inpaint_image=high_res_image,
            inpaint_mask=high_res_mask,
            low_res=low_res_image.to(device),
            tokens=tokens.to(device),
            mask=masks.to(device))
    epsilon, _ = th.split(model_output, C, dim=1)
    return th.nn.functional.mse_loss(epsilon, noise.to(device).detach())


def run_glide_finetune_epoch(
    glide_model: Text2ImUNet,
    glide_diffusion: SpacedDiffusion,
    glide_options: dict,
    dataloader: th.utils.data.DataLoader,
    optimizer: th.optim.Optimizer,
    sample_bs: int,  # batch size for inference
    sample_gs: float = 4.0,  # guidance scale for inference
    sample_respacing: str = '100', # respacing for inference
    prompt: str = "",  # prompt for inference, not training
    side_x: int = 64,
    side_y: int = 64,
    outputs_dir: str = "./outputs",
    checkpoints_dir: str = "./finetune_checkpoints",
    device: str = "cpu",
    log_frequency: int = 100,
    wandb_run=None,
    gradient_accumualation_steps=1,
    epoch: int = 0,
    train_upsample: bool = False,
    upsample_factor=4,
    image_to_upsample='low_res_face.png',
    inpainting=False,
    dataset=None
):
    print(f"checkpoints dir is {checkpoints_dir}")
    if train_upsample: train_step = upsample_train_step
    else: train_step = base_train_step

    glide_model.to(device)
    glide_model.train()
    log = {}
    gc.collect()

    for train_idx, batch in enumerate(dataloader):
        gc.collect()
        # print('------------------------------')
        accumulated_loss = train_step(
            glide_model=glide_model,
            glide_diffusion=glide_diffusion,
            batch=batch,
            device=device,
        )
        gc.collect()
        accumulated_loss.backward()
        # th.nn.utils.clip_grad_norm_(glide_model.parameters(), 1)
        # print('number of invalid gradients')
        # print(len([th.any(th.isnan(x.grad)) for x in glide_model.parameters() if (x.requires_grad and th.any(th.isnan(x.grad)))]))
        # gc.collect()
        # for name, param in glide_model.named_parameters():
        #     if param.requires_grad:
        #         print(name)
        #         print(th.max(abs(param.grad)))

        # optimizer.step(accumulated_loss)
        optimizer.step()
        gc.collect()
        glide_model.zero_grad()
        log = {**log, "iter": train_idx, "loss": accumulated_loss.item() / gradient_accumualation_steps}
        # Sample from the model
        if train_idx > 0 and train_idx % log_frequency == 0:
            print(f"loss: {accumulated_loss.item():.4f}")
            print(f"Sampling from model at iteration {train_idx}")
            gc.collect()
            th.cuda.empty_cache()
            target_sets = "Chi,Pablo_Picasso,William_Turner"
            # target_sets = 'Chi,Pablo_Picasso'
            log_dict = {}
            for target_set in target_sets.split(','):
                target_set_data = th.Tensor.repeat(th.Tensor(dataset.get_set_imgs(target_set, '')[None, ...]), [sample_bs, 1, 1, 1, 1]).half().to(device)
                kwargs = {}
                real_imgs, mask_imgs = None, None
                if inpainting:
                    if train_upsample:
                        _, _, low_res_image, high_res_image, high_res_mask = [ x.to(device) for x in batch ]
                        real_imgs = high_res_image
                        mask_imgs = high_res_mask
                        kwargs = dict(
                            low_res=low_res_image[:sample_bs].repeat(2, 1, 1, 1),
                            inpaint_image=high_res_image[:sample_bs].repeat(2, 1, 1, 1),
                            inpaint_mask=high_res_mask[:sample_bs].repeat(2, 1, 1, 1),
                        )
                    else:
                        _, _, reals, img_masks = [x.to(device) for x in batch]
                        real_imgs = reals
                        mask_imgs = img_masks
                        kwargs = dict(
                            inpaint_image=reals[:sample_bs].repeat(2, 1, 1, 1),
                            inpaint_mask=img_masks[:sample_bs].repeat(2, 1, 1, 1),
                        )
                else:
                    if not train_upsample:
                        uncond_set_imgs = th.zeros_like(target_set_data[:sample_bs]).type(target_set_data.dtype).to(device)
                        kwargs = dict(
                            set_imgs=th.cat([target_set_data[:sample_bs], uncond_set_imgs], dim=0)
                        )
                if train_upsample:
                    output_x, output_y = side_x*upsample_factor, side_y*upsample_factor
                else:
                    output_x, output_y = side_x, side_y
                samples = glide_util.sample(
                    glide_model=glide_model,
                    glide_options=glide_options,
                    side_x=output_x,
                    side_y=output_y,
                    prompt=prompt,
                    batch_size=sample_bs,
                    guidance_scale=sample_gs,
                    device=device,
                    prediction_respacing=sample_respacing,
                    image_to_upsample=image_to_upsample,
                    **kwargs
                )
                log_dict[target_set] = np.transpose(samples.cpu().detach().numpy()[0], [1, 2, 0])
                print("got output for")
                print(target_set)
                del target_set_data
                gc.collect()
                th.cuda.empty_cache() 
            # combining imgs
            output_img = []
            for key in log_dict:
                output_img.append(log_dict[key])
            output_img =np.concatenate(output_img, axis=1)
            if wandb_run is not None:
                log_dict["iter"] = train_idx
                wandb_run.log({
                    "iter": train_idx,
                    "output_img": wandb.Image(output_img, caption=target_sets)
                })
           
            sample_save_path = os.path.join(outputs_dir, f"{epoch}_{train_idx}.png")
            img_outputs = samples
            if inpainting:
                img_outputs = th.concat([samples, real_imgs[:sample_bs]*mask_imgs[:sample_bs], real_imgs[:sample_bs]], axis = -1)
            train_util.pred_to_pil(img_outputs).save(sample_save_path)
            print(f"Saved sample {sample_save_path}")
        if train_idx % 5000 == 0 and train_idx > 0:
            train_util.save_model(glide_model, checkpoints_dir, train_idx, epoch)
            print(
                f"Saved checkpoint {train_idx} to {checkpoints_dir}/glide-ft-{train_idx}.pt"
            )
        wandb_run.log(log)
    print(f"Finished training, saving final checkpoint")
    train_util.save_model(glide_model, checkpoints_dir, train_idx, epoch)
