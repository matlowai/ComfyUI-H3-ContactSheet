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

Sizes 512 / 1024 / 2048 per view all work. Measured through this node on
an RTX PRO 6000 (450 W cap, --gpu-only so everything stays in VRAM):
2048 sheet in **175 s** (~20% faster than the reference sampler), peak
~54 GiB VRAM; 512 sheets in ~25 s warm. Output verified artifact-free at
2048 (flat-region crops).

**Smaller GPUs:** those numbers are NOT a requirement. Stock ComfyUI
offloads/streams weights to system RAM on smaller cards, exactly as with
any big model — if your setup runs MiniMax-H3 video generation at all,
it runs this node (a sheet is computationally a ~17-frame clip). Expect
slower steps and a hefty system-RAM footprint (~50 GB weights mirror),
as usual for H3 on consumer cards.

`example_api_workflow.py` submits a complete graph over the ComfyUI API if
you'd rather script it.

## Decode fidelity note (resolved in-node, 2026-08-07)

Early versions decoded each slot as a raw single-latent-frame — a shape
that is **off the VAE's 17k+5 frame grid** and decodes with heavy
artifacts in ComfyUI (roundtrip error 31.4 tiled / 93.7 untiled, vs 3.9
on-grid; [Comfy-Org/ComfyUI#15416](https://github.com/Comfy-Org/ComfyUI/issues/15416)).
The node now duplicates each slot token to a legal 2-token clip and keeps
pixel frame 0 — output is visually indistinguishable from the reference
sampler. Nothing to configure; noted here for anyone building their own
per-slot decode.

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
