# FlowSeek, vendored

Optical flow network from **FlowSeek: Optical Flow Made Easier with Depth Foundation Models and
Motion Bases** (ICCV 2025), Poggi, Tosi.

- Upstream: <https://github.com/mattpoggi/flowseek>
- Taken at commit `ba6407370bceeb96244cad4d60d37489f6eb471e` (2026-01-13)
- Licence: Apache-2.0, retained verbatim in `LICENSE` beside this file

FlowSeek is built on SEA-RAFT, whose parts it imports from `../sea_raft` (BSD-3-Clause, see
`../sea_raft/LICENSE`). It reads each frame with Depth Anything V2
(<https://github.com/DepthAnything/Depth-Anything-V2>, Apache-2.0), whose backbone is DINOv2
(<https://github.com/facebookresearch/dinov2>, Apache-2.0, Copyright (c) Meta Platforms, Inc.
and affiliates).

**No weights are bundled.** They are fetched from Hugging Face or placed by hand in
`ComfyUI/models/optical_flow`. A FlowSeek checkpoint carries its Depth Anything V2 weights.
See `docs/MODELS.md`.

## Files taken from upstream

| File | Upstream path | Changed |
|---|---|---|
| `flowseek.py` | `core/flowseek.py` | forward split into `encode` and `estimate`, inference only |
| `depth_anything_v2/dpt.py` | `core/depth_anything_v2/dpt.py`, `util/blocks.py` | image preprocessing removed, blocks merged in |
| `depth_anything_v2/dinov2.py` | `core/depth_anything_v2/dinov2.py`, `dinov2_layers/` | inference path only, layers merged in |

## Every change made

1. **No third-party imports.** `cv2`, `torchvision`, `numpy`, `huggingface_hub` and `xformers`
   are gone, along with the image preprocessing in `DepthAnythingV2.infer_image` that needed
   them. Attention is `torch.nn.functional.scaled_dot_product_attention`.
2. **No fixed device.** Every `.cuda()` call is gone; tensors are made on the device of the
   frames they belong to. The constructor no longer reads `weights/depth_anything_v2_*.pth`
   from the working directory; those weights arrive with the checkpoint.
3. **`forward` split in two**, as in `../sea_raft`: `encode(images)` runs the depth network,
   the motion bases, their network and the feature network once per frame, and
   `estimate(first, second, iters)` runs the rest once per ordered pair. The normalisation the
   released weights were trained with, `image / mean - std`, is kept.
4. **DINOv2 reduced to inference.** Stochastic depth, nested tensors, masking, register tokens,
   block chunking, SwiGLU and weight initialisation are gone, and blocks past the last layer
   the depth head reads are not run. Parameter names are unchanged.
5. **`FloatFunctional` adds** in the fusion blocks are plain additions.

No parameter or buffer name differs from upstream, so the released checkpoints load with
`strict=True`.

`modules/vendor/` holds code this repository did not write. It is kept as close to upstream as
the changes above allow, and is not restyled.
