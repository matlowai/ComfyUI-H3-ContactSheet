# ComfyUI-H3-ContactSheet

Five coordinated views of one subject from **one reference image**, inside
ComfyUI — Contact-Sheet diffusion for MiniMax-H3. Pair with a Turnaround
LoRA from **[matlod/minimax-h3-turnaround](https://huggingface.co/matlod/minimax-h3-turnaround)**.

Two nodes; everything else in the graph is stock:

```
UNETLoader -> LoraLoaderModelOnly (stock!) -> BasicGuider ---------\
CLIPLoader (type: minimax) ---\                                     v
VAELoader (video) ------------+--> H3ContactSheet ---> SamplerCustomAdvanced
                                     (cond + latent)                |
VAELoader (video) --> H3ContactSheetDecode <--- denoised latent ----/
                        -> 5 views (IMAGE batch) + 1 strip (IMAGE)
```

- **H3 Contact Sheet (five views)** — prompt + reference image + per-view
  size → conditioning (+ the five-slot AV latent). The reference rides as
  `<Picture 1>`; keep that tag in your prompt.
- **H3 Contact Sheet Decode** — decodes the five slots **independently**
  (a whole-video decode would smear them) and also emits a `[v0..v4]`
  strip.

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/matlowai/ComfyUI-H3-ContactSheet
# restart ComfyUI
```

Models: the MiniMax-H3 **ref2va** DiT (pruned int8 works), the Qwen3-VL
minimax text encoder, the H3 video VAE, and a Turnaround LoRA — the LoRA
loads through the **stock** `LoraLoaderModelOnly` (key format verified).
Sampler settings that match the LoRA's training: `res_multistep` /
`simple`, 28 steps, denoise 1.0, LoRA strength 1.0 (0.7 favors scene
fidelity over rotation).

Sizes 512 / 1024 / 2048 per view all work (~10 s / ~57 s / ~227 s on an
RTX PRO 6000 at 450 W). VRAM floor is the base model, ~41 GiB.

`example_api_workflow.py` submits a complete graph over the ComfyUI API if
you'd rather script it.

## Known issue: upstream VAE decode quality (2026-08-07)

Sheet fidelity in ComfyUI is currently limited by an upstream bug in the
H3 video VAE **decode** (not this node, not the LoRA). Measured on one
image + the same fp16 VAE file, encode->decode roundtrip mean abs error:

| path | error |
|---|---|
| reference implementation (ai-toolkit extension) | **4.6** |
| ComfyUI decode, tiled (its default) | 31.4 — visible 256px tile seams / banding |
| ComfyUI decode, untiled | 93.7 — strong ViT patch-grid artifacts |

ComfyUI's **encode** is fine (latent cosine 0.9997 vs reference). Both
implementations use the same 256px/64px-overlap windowing design; comfy's
version of it mis-blends. Upstream issue filed. Until it's fixed, use the
[CLI](https://github.com/matlowai/h3-contact-sheet) for final-quality
sheets; node output is fine for composition/iteration.

## Honest notes

- Verified equivalent to the research sampler (matched checkpoint / ref /
  seed side-by-side).
- This rides ComfyUI's internal `minimax_refs` conditioning interface
  (`comfy_extras/nodes_minimax_h3.py`). **Tested against master
  `a464ac33`** — if sheets degrade after a ComfyUI update, pin near that
  commit and file an issue here.
- Model behavior (rotation axis follows subject pose, 2048 rotation
  arrives late, don't stack the LoRA on normal video) is documented on
  the [model card](https://huggingface.co/matlod/minimax-h3-turnaround).
