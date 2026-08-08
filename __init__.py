r"""ComfyUI-H3-ContactSheet — five coordinated views from one reference image.

Contact-Sheet diffusion for MiniMax-H3: five standalone image latents packed
on the model's time axis, jointly denoised, independently decoded. Pair with
a five-view LoRA from https://huggingface.co/matlod/minimax-h3-turnaround
loaded through the STOCK LoraLoaderModelOnly (keys verified compatible).

Graph shape (all other nodes stock):

  UNETLoader -> LoraLoaderModelOnly -> BasicGuider ----------------\
  CLIPLoader ----------\                                            v
  VAELoader (video) --> H3ContactSheet --(cond, latent)--> SamplerCustomAdvanced
                                                                    |
  VAELoader (video) --> H3ContactSheetDecode <---- (denoised) ------/
                          -> IMAGE batch of 5 + one strip

The denoise itself is any stock sampler; recommended res_multistep/simple,
28 steps, denoise 1.0. Works at 512/1024/2048 per view.

Tested against ComfyUI master a464ac33. The layout/payload interface this
rides (minimax_refs conditioning + latent-shape-driven packing) is internal
to comfy_extras/nodes_minimax_h3.py and may drift — pin your ComfyUI if the
sheet quality changes after an update.
"""

import math

import torch

import comfy.model_management
import comfy.nested_tensor
import comfy.utils
import node_helpers
from comfy_api.latest import ComfyExtension, io

CANVAS_MULTIPLE = 32
SHEET_SLOTS = 5
# 17-frame bookkeeping at 40 Hz audio latents: silent audio rides along
SHEET_AUDIO_LATENTS = 28


def _resize(image, width, height):
    samples = image[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, width, height, "lanczos", "disabled")
    return samples.movedim(1, -1)


class H3ContactSheet(io.ComfyNode):
    """Prompt + one reference image -> conditioning + five-slot AV latent."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3ContactSheet",
            display_name="H3 Contact Sheet (five views)",
            category="model/conditioning/minimax",
            description="Five coordinated views of one subject from one reference "
                        "image. Reference rides as <Picture 1> — keep that tag in "
                        "the prompt. Needs a contact-sheet LoRA on the model.",
            inputs=[
                io.Clip.Input("clip"),
                io.Vae.Input("vae"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True,
                                default="the camera orbits the subject of <Picture 1> "
                                        "ninety degrees clockwise"),
                io.Image.Input("ref_image"),
                io.Int.Input("size", default=1024, min=512, max=2048, step=CANVAS_MULTIPLE,
                             tooltip="Square size of each generated view. 512 = LoRA "
                                     "training size (fastest), 1024 = sweet spot, 2048 "
                                     "works (rotation arrives in the later views)."),
            ],
            outputs=[io.Conditioning.Output(display_name="positive"), io.Latent.Output()],
        )

    @classmethod
    def execute(cls, clip, vae, prompt, ref_image, size) -> io.NodeOutput:
        # aspect-preserving down-only scale of the ref to the generation's
        # pixel area, /32 snap — identical to the ref2va "match" policy and
        # the training-time match-resize
        h, w = ref_image.shape[1], ref_image.shape[2]
        scale = min(1.0, math.sqrt((size * size) / (w * h)))
        tw = max(CANVAS_MULTIPLE, round(w * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        th = max(CANVAS_MULTIPLE, round(h * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        resized = _resize(ref_image[:1], tw, th)
        z = vae.encode(resized)

        tokens = clip.tokenize(prompt, minimax_ref_items=[{"type": "image", "data": resized}])
        cond = clip.encode_from_tokens_scheduled(tokens)
        cond = node_helpers.conditioning_set_values(cond, {
            "minimax_refs": [{"kind": "image", "latent_h": th // 16,
                              "latent_w": tw // 16, "latent": z}],
        })

        device = comfy.model_management.intermediate_device()
        video = torch.zeros([1, 24, SHEET_SLOTS, size // 16, size // 16], device=device)
        audio = torch.zeros([1, 32, 2, SHEET_AUDIO_LATENTS], device=device)
        latent = {"samples": comfy.nested_tensor.NestedTensor((video, audio))}
        return io.NodeOutput(cond, latent)


class H3ContactSheetDecode(io.ComfyNode):
    """Denoised five-slot latent -> five independently-decoded views + strip.

    Slots MUST be decoded one at a time (each is a standalone one-frame
    image); a whole-latent video decode smears them together.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3ContactSheetDecode",
            display_name="H3 Contact Sheet Decode",
            category="latent/minimax",
            inputs=[
                io.Vae.Input("vae"),
                io.Latent.Input("samples"),
            ],
            outputs=[
                io.Image.Output(display_name="views"),
                io.Image.Output(display_name="sheet"),
            ],
        )

    @classmethod
    def execute(cls, vae, samples) -> io.NodeOutput:
        latent = samples["samples"]
        video = latent.unbind()[0] if hasattr(latent, "unbind") else latent
        if video.ndim == 4:  # plain image latent (e.g. from stock VAEEncode)
            video = video.unsqueeze(2)
        views = []
        for s in range(video.shape[2]):
            # a single latent frame is OFF the VAE's 17k+5 grid (minimum legal
            # clip = 2 latent tokens = 5 px frames) and decodes with heavy
            # artifacts. Duplicate the slot token to a legal 2-token clip and
            # keep pixel frame 0 — measured cleaner than any T=1 path.
            slot = video[:, :, s:s + 1]
            frames = vae.decode(torch.cat([slot, slot], dim=2))  # [1, T, H, W, C]
            if frames.ndim == 5:
                frames = frames[:, 0]
            elif frames.shape[0] > 1:  # some paths return frames on batch dim
                frames = frames[:1]
            views.append(frames[0])
        batch = torch.stack(views)                     # [5, H, W, C]
        sheet = torch.cat(list(batch), dim=1)[None]    # [1, H, 5W, C]
        return io.NodeOutput(batch, sheet)


class H3ContactSheetExtension(ComfyExtension):
    async def get_node_list(self):
        return [H3ContactSheet, H3ContactSheetDecode]


def comfy_entrypoint():
    return H3ContactSheetExtension()
