# MiniMax H3 latent upscaler, vendored

A learned 3D upscaler for MiniMax H3 video latents, from **ComfyUI Minimax H3 Latent Upscaler**
by LBH-123-AI.

- Upstream: <https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler>
- Taken at commit `40316cf008b2fd8663263270669eb4da23f89d2c` (2026-09-17)
- Licence: MIT, retained verbatim in `LICENSE` beside this file
- Weights: <https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler>, Apache-2.0

**No weights are bundled.** They are fetched from Hugging Face or placed by hand in
`ComfyUI/models/latent_upscale_models`.

## Files taken from upstream

| File | Upstream path | Changed |
|---|---|---|
| `network.py` | `nodes/minimax_h3_latent_upscaler_3d.py` | network classes and latent statistics only, inference only |

## Every change made

1. **Network and statistics only.** The device helpers, checkpoint scanning and loading, the
   resize-mode arithmetic and the ComfyUI node are not taken; the pack loads checkpoints and
   registers the network with ComfyUI's model management itself.
2. **No attention blocks.** `AttnBlock3D` and the `attn` argument are gone, along with the
   `einops` import it needed. Upstream builds the network with attention off at inference.
3. **One pass over the whole clip.** Upstream's `forward` splits a long clip into 32-frame
   segments blended over the temporal kernel's width; here `forward` is upstream's
   `_forward_seg`, run once over every frame it is given.
4. **`forward(x, scale, size)`** takes the scale and the target `(time, height, width)` together,
   and returns the resized latent even when the size is unchanged.
5. **Dropout** in `ResBlockEmb3D` is an `nn.Identity` at the same position, so checkpoint keys
   still line up, and the zero initialisation of the last convolutions is gone, since every
   parameter is replaced by the checkpoint.
6. **`LATENTS_MEAN` and `LATENTS_STD`** are tuples rather than lists.
