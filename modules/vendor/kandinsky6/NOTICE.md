# Kandinsky 6, vendored

The joint video and audio transformer, the PiFlow policy and the audio autoencoder of
**Kandinsky 6**, Kandinsky Lab.

- Upstream: <https://github.com/kandinskylab/kandinsky-6>
- Taken at commit `bf676db9b1335d22ebca85089355732687de3d0a` (2026-10-07)
- Licence: MIT, Kandinsky Lab, retained verbatim in `LICENSE` beside this file

The audio autoencoder upstream carries from two further projects, whose licences are in
`licenses/`:

| Project | Licence | Text |
|---|---|---|
| [MMAudio](https://github.com/hkchengrex/MMAudio), the mel autoencoder and mel converter | MIT, Sony Research Inc. | `licenses/LICENSE.mmaudio` |
| [BigVGAN](https://github.com/NVIDIA/BigVGAN), the vocoder | MIT, NVIDIA Corporation | `licenses/LICENSE.bigvgan` |
| The projects BigVGAN itself carries from | MIT, Apache-2.0 and BSD-3-Clause | `licenses/LICENSE.bigvgan_1` to `_8` |

**No weights are bundled.** The checkpoints are downloaded by the user. See `docs/MODELS.md`.

## Files taken from upstream

| File | Upstream path | Changed |
|---|---|---|
| `dit.py` | `kandinsky/core/components/dit.py`, `rope.py`, `kandinsky/core/tensors.py` | layers built from ComfyUI's operations, joint video and audio path only |
| `piflow.py` | `kandinsky/core/algo/piflow_math.py` | none |
| `audio.py` | `comfyui/kandinsky6/mmaudio/ext/autoencoder/`, `ext/mel_converter.py`, `ext/bigvgan_v2/` | the three folded into one file, inference only |

## Every change made

1. **Layers from ComfyUI.** Every `nn.Linear`, `nn.LayerNorm`, `nn.RMSNorm` and `nn.Embedding`
   in `dit.py` is built from the `operations` ComfyUI passes in, with its `device` and `dtype`.
   The layers upstream runs in float32, the timestep MLP and every modulation projection, read
   their weights through `comfy.ops.CastBiasWeightContext` at float32.
2. **Attention** goes through `comfy.ldm.modules.attention.optimized_attention` in place of
   upstream's engine dispatch, so the backend ComfyUI was started with is the one used.
3. **Rotary tables are computed per call** by `rope_1d` and `rope_3d`, with the same positions,
   frequencies and scale factors as upstream's `RoPE1D` and `RoPE3D` tables. `apply_rotary`
   makes the same two products upstream sums, accumulated in place.
4. **One forward.** `DiffusionTransformer3D.forward` is upstream's fused video and audio path. The
   video-only and audio-only paths, the text projection cache, attention masks for padded text
   and the MagCache stage helpers are gone. A reference frame at the end of the clip reuses the
   rotary position of the first, which is what upstream's pipeline builds for image-to-video.
5. **`TimeEmbeddings` takes the output dtype** as an argument in place of reading the weight's.
6. **The audio autoencoder is the inference half.** `MPConv1D` reads a weight that already
   carries its normalisation, which is how the released file stores it, so `remove_weight_norm`
   and the weight-norm parametrisations in BigVGAN are gone. The posterior returns its mean. The
   mel filter bank is read from the checkpoint rather than built with librosa.
7. **BigVGAN's configuration** is the released 44.1 kHz, 128 band, 512x one, as constructor
   defaults. `Snake` and `AMPBlock2`, which that configuration never builds, are gone.

No parameter or buffer name differs from the released Diffusers checkpoints, so the transformer
and the audio file load with `strict=True`.

`modules/vendor/` holds code this repository did not write. It is kept as close to upstream as
the changes above allow, and is not restyled.
