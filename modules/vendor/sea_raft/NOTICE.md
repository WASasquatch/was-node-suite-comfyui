# SEA-RAFT, vendored

Optical flow network from **SEA-RAFT: Simple, Efficient, Accurate RAFT for Optical Flow**
(ECCV 2024), Wang, Lipson, Deng.

- Upstream: <https://github.com/princeton-vl/SEA-RAFT>
- Taken at commit `9137517ba24e628442aec097d3afe71d03503b75` (2026-03-26)
- Licence: BSD-3-Clause, retained verbatim in `LICENSE` beside this file

**No weights are bundled.** They are fetched from Hugging Face or placed by hand in
`ComfyUI/models/optical_flow`. See `docs/MODELS.md`.

## Files taken from upstream

| File | Upstream path | Changed |
|---|---|---|
| `layer.py` | `core/layer.py` | training-only gradient clipping removed, `LayerNorm` kept to its channels-last form |
| `extractor.py` | `core/extractor.py` | constructor takes keywords, ImageNet initialisation removed |
| `update.py` | `core/update.py` | constructors take keywords, unused `FlowHead` removed |
| `corr.py` | `core/corr.py`, `core/utils/utils.py` | lookup offsets built once, on-demand lookup added, unused samplers removed |
| `raft.py` | `core/raft.py`, `core/utils/utils.py` | forward split into `encode` and `estimate`, inference only |

## Every change made

1. **No third-party imports.** `huggingface_hub`, `torchvision`, `scipy` and `numpy` are gone.
   The model class is a plain `nn.Module`; `ResNetFPN` no longer downloads ImageNet weights
   while it is built, since every parameter is replaced by the checkpoint.
2. **`forward` split in two.** `encode(images)` runs the feature network once per frame and
   `estimate(first, second, iters)` runs the context network and the refinement once per
   ordered pair of encoded frames. `forward(image1, image2, iters)` is the two composed and
   returns the final flow, which is upstream's `forward(..., test_mode=True)["final"]`.
3. **Inference only.** The per-iteration flow list, the uncertainty output, the mixture of
   Laplace loss and `flow_gt` are gone. The flow is upsampled once, after the last pass,
   which is the one upstream returns as `final`.
4. **Lookup offsets** in `CorrBlock` are built once per pair rather than once per level per
   pass, and the dilation of ones they were multiplied by is gone. The pyramid stops at its
   last level instead of downsampling once more.
5. **On-demand correlation.** `CorrBlock(..., local=k)` builds the all-pairs volume for all but
   its `k` finest levels; those it correlates at each lookup by sampling the second feature
   pyramid. The values are the same; memory for those levels grows with the frame's area
   rather than its square. The volume is divided by the square root of the channel count in
   place.
6. **Padding** is a pair of functions, `padding` and `unpad`, in place of `InputPadder`, with
   the same arithmetic.

No parameter or buffer name differs from upstream, so the released checkpoints load with
`strict=True`. A checkpoint saved with only one of a shared BatchNorm's two names, `bn3` and
`downsample.1`, has the other filled from it before loading.

`modules/vendor/` holds code this repository did not write. It is kept as close to upstream as
the changes above allow, and is not restyled.
