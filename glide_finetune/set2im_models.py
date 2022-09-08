from vit import sVIT
from glide_text2im.unet import UNetModel
from glide_text2im.xf import LayerNorm
from glide_text2im.nn import timestep_embedding
import torch as th
import torch.nn as nn
import torch.nn.functional as F

class Set2ImUNet(UNetModel):
    """
    A UNetModel that conditions on text with an encoding transformer.

    Expects an extra kwarg `tokens` of text.

    :param text_ctx: number of text tokens to expect.
    :param xf_width: width of the transformer.
    :param xf_layers: depth of the transformer.
    :param xf_heads: heads in the transformer.
    :param xf_final_ln: use a LayerNorm after the output layer.
    :param tokenizer: the text tokenizer for sampling/vocab size.
    """

    def __init__(
        self,
        vit_options,
        xf_width,
        xf_final_ln,
        *args,
        cache_text_emb=False,
        xf_padding=False,
        share_unemb=False,
        **kwargs,
    ):
        self.xf_width = xf_width
        if not self.xf_width:
            super().__init__(*args, **kwargs, encoder_channels=None)
        else:
            super().__init__(*args, **kwargs, encoder_channels=self.xf_width)
        self.vit = sVIT(**vit_options)
        
        if self.xf_width:
            if xf_final_ln:
                self.final_ln = LayerNorm(self.xf_width)
            else:
                self.final_ln = None

            self.transformer_proj = nn.Linear(self.xf_width, self.model_channels * 4)

        self.cache = None

    def convert_to_fp16(self):
        super().convert_to_fp16()
        if self.xf_width:
            self.vit.convert_module_to_f16()
            self.transformer_proj.to(th.float16)

    def get_set_emb(self, set_imgs, timesteps):
        # print(self.dtype)
        xf_out = self.vit(set_imgs.to(self.dtype), timesteps)
        # print('xf_out max min')
        # print(th.max(xf_out), th.min(xf_out))
        if self.final_ln is not None:
            xf_out = self.final_ln(th.permute(xf_out, [0, 2, 1]))
            xf_out = th.permute(xf_out, [0, 2, 1])
        xf_proj = self.transformer_proj(xf_out[:, :, -1])
        # print('xf_proj max min')
        # print(th.max(xf_proj), th.min(xf_proj))
        # xf_out = xf_out.permute(0, 2, 1)  # NLC -> NCL

        outputs = dict(xf_proj=xf_proj, xf_out=xf_out)

        # if self.cache_text_emb:
        #     self.cache = dict(
        #         tokens=tokens,
        #         xf_proj=xf_proj.detach(),
        #         xf_out=xf_out.detach() if xf_out is not None else None,
        #     )

        return outputs

    def del_cache(self):
        self.cache = None

    def forward(self, x, timesteps, set_imgs=None):
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        # print('emb max min')
        # print(th.max(emb), th.min(emb))
        
        if self.xf_width:
            set_outputs = self.get_set_emb(set_imgs, timesteps)
            xf_proj, xf_out = set_outputs["xf_proj"], set_outputs["xf_out"]
            # print('xf_proj max min')
            # print(th.max(xf_proj), th.min(xf_proj))
            # print('xf_out max min')
            # print(th.max(xf_out), th.min(xf_out))
            emb = emb + xf_proj.to(emb)
        else:
            xf_out = None
        # print('emb max min')
        # print(th.max(emb), th.min(emb))
        h = x.type(self.dtype)
        # print('h max min')
        # print(th.max(h), th.min(h))
        # print('input block')
        for module in self.input_blocks:
            # print(type(module).__name__)
            h = module(h, emb, xf_out)
            # print(h.shape)
            # print('h max min')
            # print(th.max(h), th.min(h))
            hs.append(h)
        # print('h max min')
        # print(th.max(h), th.min(h))
        # print('middle block')
        h = self.middle_block(h, emb, xf_out)
        # print(h.shape)
        # print('h max min')
        # print(th.max(h), th.min(h))
        # print('output block')
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb, xf_out)
            # print(h.shape)
        h = h.type(x.dtype)
        # print('h max min')
        # print(th.max(h), th.min(h))
        h = self.out(h)
        # print('output')
        # print(th.max(h), th.min(h))
        return h

class SuperResSet2ImUNet(Set2ImUNet):
    """
    A text2im model that performs super-resolution.
    Expects an extra kwarg `low_res` to condition on a low-resolution image.
    """

    def __init__(self, *args, **kwargs):
        if "in_channels" in kwargs:
            kwargs = dict(kwargs)
            kwargs["in_channels"] = kwargs["in_channels"] * 2
        else:
            # Curse you, Python. Or really, just curse positional arguments :|.
            args = list(args)
            args[1] = args[1] * 2
        super().__init__(*args, **kwargs)

    def forward(self, x, timesteps, low_res=None, **kwargs):
        _, _, new_height, new_width = x.shape
        upsampled = F.interpolate(
            low_res, (new_height, new_width), mode="bilinear", align_corners=False
        )
        x = th.cat([x, upsampled], dim=1)
        return super().forward(x, timesteps, **kwargs)


class InpaintSet2ImUNet(Set2ImUNet):
    """
    A text2im model which can perform inpainting.
    """

    def __init__(self, *args, **kwargs):
        if "in_channels" in kwargs:
            kwargs = dict(kwargs)
            kwargs["in_channels"] = kwargs["in_channels"] * 2 + 1
        else:
            # Curse you, Python. Or really, just curse positional arguments :|.
            args = list(args)
            args[1] = args[1] * 2 + 1
        super().__init__(*args, **kwargs)

    def forward(self, x, timesteps, inpaint_image=None, inpaint_mask=None, **kwargs):
        if inpaint_image is None:
            inpaint_image = th.zeros_like(x)
        if inpaint_mask is None:
            inpaint_mask = th.zeros_like(x[:, :1])
        # print(x.shape, inpaint_image.shape, inpaint_mask.shape)
        try:
            return super().forward(
                th.cat([x, inpaint_image * inpaint_mask, inpaint_mask], dim=1),
                timesteps,
                **kwargs,
            )
        except Exception as e:
            print("Error with")
            print(x.shape, inpaint_image.shape, inpaint_mask.shape)
            raise Exception()


class SuperResInpaintSet2ImUnet(Set2ImUNet):
    """
    A text2im model which can perform both upsampling and inpainting.
    """

    def __init__(self, *args, **kwargs):
        if "in_channels" in kwargs:
            kwargs = dict(kwargs)
            kwargs["in_channels"] = kwargs["in_channels"] * 3 + 1
        else:
            # Curse you, Python. Or really, just curse positional arguments :|.
            args = list(args)
            args[1] = args[1] * 3 + 1
        super().__init__(*args, **kwargs)

    def forward(
        self,
        x,
        timesteps,
        inpaint_image=None,
        inpaint_mask=None,
        low_res=None,
        **kwargs,
    ):
        if inpaint_image is None:
            inpaint_image = th.zeros_like(x)
        if inpaint_mask is None:
            inpaint_mask = th.zeros_like(x[:, :1])
        _, _, new_height, new_width = x.shape
        upsampled = F.interpolate(
            low_res, (new_height, new_width), mode="bilinear", align_corners=False
        )
        return super().forward(
            th.cat([x, inpaint_image * inpaint_mask, inpaint_mask, upsampled], dim=1),
            timesteps,
            **kwargs,
        )