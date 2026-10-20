# Model weights

Nothing is downloaded at startup. A checkpoint is fetched by the first run that needs it, and
only with `features.network: true`. With that off, the run stops and names the file and every
folder it searched.

## EMA-VFI frame interpolation

Published at <https://huggingface.co/WAS/EMA-VFI>. Goes in `ComfyUI/models/EMA-VFI`.

| File | Size | Quality | Multiplier above 2 |
|---|---|---|---|
| [`ours_t.safetensors`](https://huggingface.co/WAS/EMA-VFI/blob/main/ours_t.safetensors) | 263 MB | best | **yes** |
| [`ours.safetensors`](https://huggingface.co/WAS/EMA-VFI/blob/main/ours.safetensors) | 263 MB | best | no |
| [`ours_small_t.safetensors`](https://huggingface.co/WAS/EMA-VFI/blob/main/ours_small_t.safetensors) | 58 MB | lower | **yes** |
| [`ours_small.safetensors`](https://huggingface.co/WAS/EMA-VFI/blob/main/ours_small.safetensors) | 58 MB | lower | no |

A `_t` file is the one to take: it lands a frame anywhere between two, so a multiplier above 2
works.

## Optical flow

**Video Motion Model Loader** builds the network that **Video Motion** and **Video Temporal
Upscale** read. Files go in `ComfyUI/models/optical_flow`, or a folder an `extra_model_paths.yaml`
entry names under the key `optical_flow`. With `features.network: true`, a file in this table is
fetched by the first run that picks it.

| Save as | Network | Size | Download |
|---|---|--:|---|
| `sea_raft_m_spring.safetensors` | SEA-RAFT M, the default | 79 MB | `model.safetensors` from [MemorySlices/Tartan-C-T-TSKH-spring540x960-M](https://huggingface.co/MemorySlices/Tartan-C-T-TSKH-spring540x960-M) |
| `sea_raft_s_spring.safetensors` | SEA-RAFT S, about 1.3x faster | 36 MB | `model.safetensors` from [MemorySlices/Tartan-C-T-TSKH-spring540x960-S](https://huggingface.co/MemorySlices/Tartan-C-T-TSKH-spring540x960-S) |
| `sea_raft_m_ct.safetensors` | SEA-RAFT M, general purpose | 79 MB | `model.safetensors` from [MemorySlices/Tartan-C-T432x960-M](https://huggingface.co/MemorySlices/Tartan-C-T432x960-M) |
| `flowseek_t_ct.safetensors` | FlowSeek T, reads depth too, about 1.5x slower | 171 MB | `optical_flow/flowseek_t_ct.safetensors` from [WAS/was-node-suite-weights](https://huggingface.co/WAS/was-node-suite-weights/tree/main/optical_flow) |
| `flowseek_t_tskh.safetensors` | FlowSeek T, the TSKH training | 171 MB | `optical_flow/flowseek_t_tskh.safetensors` from [WAS/was-node-suite-weights](https://huggingface.co/WAS/was-node-suite-weights/tree/main/optical_flow) |

Every SEA-RAFT repository names its file `model.safetensors`. A copy downloaded by hand is saved
under the name in the first column. Upstream's own `.pth` releases load under the names they are
published with: SEA-RAFT's `Tartan-*.pth` from [princeton-vl/SEA-RAFT](https://github.com/princeton-vl/SEA-RAFT)
and FlowSeek's `flowseek_*.pth` from [mattpoggi/flowseek](https://github.com/mattpoggi/flowseek).

| Weights | Licence |
|---|---|
| SEA-RAFT | BSD-3-Clause, Princeton Vision & Learning Lab |
| FlowSeek, with the Depth Anything V2 weights inside it | Apache-2.0 |

## MiniMax H3

| File | Goes in | Size | Read by | Download |
|---|---|--:|---|---|
| `minimax_h3_latent_upscaler_3d_conv_v1_fp16.safetensors` | `ComfyUI/models/latent_upscale_models` | 691 MB | **H3 Upscale Video**, `upscale_model` | `minimax_h3_latent_upscaler_3d_conv_v1/` in [LBH-123-AI/Minimax_h3_latent_Upscaler](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler) |
| `taeh3.safetensors` | `ComfyUI/models/vae_approx` | 22,709,752 bytes | **H3 Extend Window**, `live_preview`, drawn in the Prompt Timeline | `safetensors/taeh3.safetensors` in [madebyollin/taehv](https://github.com/madebyollin/taehv) |

The upscaler is fetched by the first run that picks it with `features.network: true`, and the
`_bf16` file beside it loads too. `taeh3.safetensors` is never fetched. Another file named
`taeh3`, the older per-frame decoder, is a different size and does not load. Both folders are
read through `extra_model_paths.yaml` only where a root carries an entry under the key
`latent_upscale_models` or `vae_approx`.

| Weights | Licence |
|---|---|
| The H3 latent upscaler | Apache-2.0, LBH-123-AI |
| `taeh3.safetensors` | MIT, Ollin Boer Bohan |

## HDR reconstruction

[`hdrcnn.safetensors`](https://huggingface.co/WAS/was-node-suite-weights/blob/main/hdr/hdrcnn.safetensors),
118 MB, from `hdr/` in [WAS/was-node-suite-weights](https://huggingface.co/WAS/was-node-suite-weights),
into `ComfyUI/models/hdr`. `models/hdr/hdr/` is read too, and an `extra_model_paths.yaml` entry
under the key `hdr`.

## Power Preprocessor

Searched in order before anything is fetched: `ComfyUI/models/<family>`, Hugging Face's own
cache, then the `ckpts` directory of any pack under `custom_nodes`. Nothing is copied or moved.
A model is held for the life of the process, so one loader feeds as many nodes as wanted at the
cost of one set of weights.

| Family | Folder |
|---|---|
| Depth | `ComfyUI/models/depth_anything` |
| Pose | `ComfyUI/models/pose` |
| Segmentation | `ComfyUI/models/segmentation` |
| Edge | `ComfyUI/models/hed`, `pidi`, `teed` |
| Lineart | `ComfyUI/models/lineart`, `lineart_anime`, `manga_line` |
| Line segments | `ComfyUI/models/mlsd` |
| AnyLine | `ComfyUI/models/teed` |
| Intrinsics | `ComfyUI/models/intrinsics` |
| Denoising | `ComfyUI/models/denoise` |
| Low light | `ComfyUI/models/low_light` |

| Model | Answers | Size |
|---|---|--:|
| `Depth Anything V2 Small` | Depth | 99 MB |
| `Depth Anything V2 Base` | Depth | 390 MB |
| `Depth Anything V2 Large` | Depth | 1.3 GB |
| `DPT SwinV2 Tiny` | Depth, the quickest here | 164 MB |
| `DPT Large` | Depth, a heavier reading than Depth Anything | 1.4 GB |
| `ViTPose Small` | Body pose | 130 MB |
| `ViTPose Wholebody` | Body pose with feet, face and both hands, 133 points | 97 MB |
| `ViTPose Animal` | Animal pose, 17 points | 97 MB |
| `ViTPose Base` | Body pose | 380 MB |
| `SegFormer B0 ADE20K` | 150 classes per pixel | 15 MB |
| `SegFormer B2 ADE20K` | 150 classes per pixel | 110 MB |
| `SegFormer B4 ADE20K` | 150 classes per pixel | 260 MB |
| `HED Soft Edge` | Soft edges | 29 MB |
| `PiDiNet Soft Edge` | Soft edges, crisper than HED | 3 MB |
| `TEED Soft Edge` | Soft edges, the lightest of the three | 2 MB |
| `Lineart` | Drawn lines, fine | 62 MB |
| `Lineart Coarse` | Drawn lines, heavier | 62 MB |
| `Lineart Anime` | Drawn lines, trained on anime | 218 MB |
| `Manga Line` | Manga and comic linework | 173 MB |
| `MLSD Line Segments` | Straight runs, for architecture and interiors | 6 MB |
| `AnyLine` | A fine edge pass merged with a lineart pass | 0.2 MB |
| `Marigold IID Appearance` | Albedo, roughness, metallicity and material | 1.9 GB |
| `Marigold IID Lighting` | Albedo, shading and residual | 1.9 GB, or 1.7 GB once the other is on disk |
| `SCUNet` | Denoising, the widest range of noise | 72 MB |
| `NAFNet SIDD width32` | Denoising, light sensor grain | 117 MB |
| `NAFNet SIDD width64` | Denoising, the same at greater width | 464 MB |
| `DarkIR` | Low light, and takes out blur and noise with it | 13 MB |
| `Retinexformer NTIRE` | Low light, the general choice | 6.5 MB |
| `Retinexformer LOL v1` | Low light | 6.5 MB |
| `Retinexformer LOL v2 Real` | Low light | 6.5 MB |
| `Retinexformer LOL v2 Synthetic` | Low light | 6.5 MB |
| `Retinexformer FiveK` | Low light, the gentlest lift | 6.5 MB |
| `Retinexformer Extreme Dark` | Low light, the hardest lift | 6.5 MB |
| `Retinexformer Dark Motion` | Low light, moving subjects | 6.5 MB |
| `Retinexformer Indoor Night` | Low light, interiors | 6.5 MB |
| `Retinexformer Outdoor Night` | Low light, exteriors | 6.5 MB |
| `HVI-CIDNet Generalization` | Low light, a second opinion | 8 MB |
| `HVI-CIDNet FiveK` | Low light, photographic tone | 8 MB |
| `HVI-CIDNet SICE` | Low light, mixed exposure | 8 MB |
| `HVI-CIDNet Extreme Dark` | Low light, near black | 8 MB |

The six intrinsic maps, albedo, roughness, metallicity, material, shading and residual,
come from Marigold. They run in torch on the published safetensors and need nothing
installed. Both repositories publish the same autoencoder, and it is fetched once: the
second checkpoint reads the copy the first one left in `ComfyUI/models/intrinsics`.

### Marigold v2

`Marigold v2` is a model for `depth_map`, `normal_map` and `albedo`, and the one model the
node does not download. Picking it reads the `model`, `vae` and `conditioning_name`
inputs instead. Take the files from
[Comfy-Org/marigold-v2-0](https://huggingface.co/Comfy-Org/marigold-v2-0) and place them
yourself:

| File | Goes in | Size |
|---|---|--:|
| [`qwen_image_edit_2509_int8_convrot.safetensors`](https://huggingface.co/Comfy-Org/marigold-v2-0/blob/main/diffusion_models/qwen_image_edit_2509_int8_convrot.safetensors) | `ComfyUI/models/diffusion_models` | 20.5 GB |
| [`marigold_v2_depth_log_stage2.safetensors`](https://huggingface.co/Comfy-Org/marigold-v2-0/blob/main/loras/marigold_v2_depth_log_stage2.safetensors) | `ComfyUI/models/loras` | 1.7 GB |
| [`marigold_v2_normals.safetensors`](https://huggingface.co/Comfy-Org/marigold-v2-0/blob/main/loras/marigold_v2_normals.safetensors) | `ComfyUI/models/loras` | 1.7 GB |
| [`marigold_v2_albedo.safetensors`](https://huggingface.co/Comfy-Org/marigold-v2-0/blob/main/loras/marigold_v2_albedo.safetensors) | `ComfyUI/models/loras` | 1.7 GB |
| [`marigold_v2_depth_log_stage2_vae.safetensors`](https://huggingface.co/Comfy-Org/marigold-v2-0/blob/main/vae/marigold_v2_depth_log_stage2_vae.safetensors) | `ComfyUI/models/vae` | 254 MB |
| [`marigold_v2_normals_vae.safetensors`](https://huggingface.co/Comfy-Org/marigold-v2-0/blob/main/vae/marigold_v2_normals_vae.safetensors) | `ComfyUI/models/vae` | 254 MB |
| [`marigold_v2_albedo_vae.safetensors`](https://huggingface.co/Comfy-Org/marigold-v2-0/blob/main/vae/marigold_v2_albedo_vae.safetensors) | `ComfyUI/models/vae` | 254 MB |
| [`marigold_v2_depth_conditioning.safetensors`](https://huggingface.co/Comfy-Org/marigold-v2-0/blob/main/embeddings/marigold_v2_depth_conditioning.safetensors) | `ComfyUI/models/embeddings` | 2.4 MB |
| [`marigold_v2_normals_conditioning.safetensors`](https://huggingface.co/Comfy-Org/marigold-v2-0/blob/main/embeddings/marigold_v2_normals_conditioning.safetensors) | `ComfyUI/models/embeddings` | 2.5 MB |
| [`marigold_v2_albedo_conditioning.safetensors`](https://huggingface.co/Comfy-Org/marigold-v2-0/blob/main/embeddings/marigold_v2_albedo_conditioning.safetensors) | `ComfyUI/models/embeddings` | 2.6 MB |

Which map each answer takes:

| Answer | LoRA | VAE | Conditioning |
|---|---|---|---|
| `depth_map` | `..._depth_log_stage2` | `..._depth_log_stage2_vae` | `..._depth_conditioning` |
| `normal_map` | `..._normals` | `..._normals_vae` | `..._normals_conditioning` |
| `albedo` | `..._albedo` | `..._albedo_vae` | `..._albedo_conditioning` |

Choosing `Marigold v2` on the canvas wires **Load Diffusion Model** on the transformer into
`model`. A loader already in the graph on a `qwen_image_edit_2509` file is reused; otherwise
one is placed beside the node, on `qwen_image_edit_2509_int8_convrot.safetensors` or, where
the folder lacks it, another `qwen_image_edit_2509` build it holds. That is the only wire the answer needs: `adapter_name`,
`vae_name` and `conditioning_name` stay on `auto` and find that row's files by name, so one
transformer feeds every map and no LoRA or VAE loader is placed.

A file the folders lack is named on its widget instead of `auto`, and ComfyUI lists it under
missing models with its download. Choosing another model takes the loader's link off the
node and puts a named, missing file back to `auto`; changing the answer moves each widget to
that map's file.

The three overrides, where a file is named something else or a loader is wanted:

| Override | Use |
|---|---|
| `adapter_name` set to `already on the model` | A LoRA put on the transformer beforehand, at a strength of your own |
| The `vae` socket | A decoder from **Load VAE** or any node answering a VAE |
| The `conditioning` socket | A prompt embedding from a text encoder or a loaded conditioning |

`conditioning_name` stays on `auto`, which finds that map's embedding by name in
the embeddings folder. Naming a file overrides it. It is the text encoder's saved output, so
no text encoder is loaded.

The LoRA is what decides the map. Running the depth adapter against another map's embedding
moves the result by 0.005 on a 0 to 1 scale, and against a short one by 0.226, so an
override is worth setting only to a full length embedding.

One transformer serves all three answers, and only the LoRA, the VAE and the conditioning
change. It occupies about 19 GB of VRAM, so a 24 GB card runs it with little room for a
checkpoint beside it. The first answer costs about 10 seconds while the transformer loads,
and each one after that about 0.8 seconds a frame.

| Trouble | Cause |
|---|---|
| `model is not wired` | No transformer reaches `model`, as in a workflow saved before it was wired. Set `model_name` to another model and back to `Marigold v2`, or wire Load Diffusion Model by hand. |
| `Value not in list` on `unet_name`, `adapter_name`, `vae_name` or `conditioning_name` | The file named there is not downloaded yet. Download it from the missing models list, or from the table above, and reload the page. |
| `found no adapter` | That map's LoRA is not in the loras folder, or the model already carries one and `adapter_name` was not set to `already on the model`. |
| `found no decoder` | That map's VAE is not in the vae folder. |
| `found no prompt embedding` | That map's conditioning file is not in the embeddings folder. |
| A map that looks like noise | The LoRA and the VAE name different maps. |
| Depth flat or inverted | The answer already turns log depth over, so no invert node is wanted after it. |
| The `conditioning` menu is empty | The files are not in `ComfyUI/models/embeddings`. Put them there and reload the page. |

## Skin masks

**Image Skin Mask** reads a Sapiens2 body-part network from `ComfyUI/models/sapiens`.
**Image Simple Skin Mask** reads a clothing model from `ComfyUI/models/segmentation`. Both find
faces with `ViTPose Wholebody` from `ComfyUI/models/pose`, the file in the Power Preprocessor
table. With `features.network: true`, a file is fetched by the first run that picks it. Graph:
[`image-skin-mask.json`](workflows/image-skin-mask.json).

| File | Node | Size | Download | Licence |
|---|---|--:|---|---|
| `sapiens2_1b_seg.safetensors` | Image Skin Mask, the default | 5.9 GB | [facebook/sapiens2-seg-1b](https://huggingface.co/facebook/sapiens2-seg-1b/blob/main/sapiens2_1b_seg.safetensors) | Sapiens2 License |
| `sapiens2_0.4b_seg.safetensors` | Image Skin Mask, lighter, less sure of unusual bodies | 1.6 GB | [facebook/sapiens2-seg-0.4b](https://huggingface.co/facebook/sapiens2-seg-0.4b/blob/main/sapiens2_0.4b_seg.safetensors) | Sapiens2 License |
| `sapiens2_0.8b_seg.safetensors` | Image Skin Mask | 3.3 GB | [facebook/sapiens2-seg-0.8b](https://huggingface.co/facebook/sapiens2-seg-0.8b/blob/main/sapiens2_0.8b_seg.safetensors) | Sapiens2 License |
| `sapiens2_5b_seg.safetensors` | Image Skin Mask, about 10 GB of VRAM | 20.4 GB | [facebook/sapiens2-seg-5b](https://huggingface.co/facebook/sapiens2-seg-5b/blob/main/sapiens2_5b_seg.safetensors) | Sapiens2 License |
| `segformer_b3_clothes`, the whole repository | Image Simple Skin Mask | 190 MB | [sayeed99/segformer_b3_clothes](https://huggingface.co/sayeed99/segformer_b3_clothes) | MIT |

A Sapiens2 network runs in bfloat16 on a GPU that computes in it, so it holds about half its
file size in VRAM.

The [Sapiens2 License](https://github.com/facebookresearch/sapiens2/blob/main/LICENSE.md) binds
whoever runs the weights:

| | |
|---|---|
| Allows | Use, changes and redistribution, with a copy of the licence |
| Forbids | Surveillance, biometric processing, identifying people, deepfakes and impersonation, pornographic or defamatory use, and inferring health or other sensitive traits without consent |
| Also | Meta may audit a user and require the weights deleted, and may change the terms |

| Trouble | Cause |
|---|---|
| The model menu reads `put a sapiens2 seg checkpoint in models/sapiens` | Nothing on disk and `features.network` off. Turn it on, or place a file from the table and press **R** |
| `is not a Sapiens2 body-part checkpoint` | The file is another Sapiens2 task, such as pose or normals. Pick a `_seg` file |
| An arm or torso comes out half masked | `sapiens2_0.4b_seg` is unsure of that body. Pick `sapiens2_1b_seg` |
| Arms and legs of small, far-off figures are left out | Image Skin Mask reads them poorly in a crowd or a wide shot. Image Simple Skin Mask holds them |
| The top of a bald head is left out | Image Skin Mask reads a bald crown as hair. Image Simple Skin Mask keeps it |
| A bright highlight on skin is left out | Image Simple Skin Mask reads colour, and a highlight carries little. Image Skin Mask keeps it |

## PS-SR video super resolution

**Video Super Resolution (PS-SR)** reads a checkout placed by hand. Nothing here is fetched,
whatever `features.network` is set to, and the group is off until `pssr: true` is set under
`features:` in `config.yaml`.

The checkout goes in `ComfyUI/models/PS-SR`. `ComfyUI/models/ps-sr` and `ComfyUI/models/pssr`
are read too, an `extra_model_paths.yaml` entry under any of those three names keeps it on
another drive, and `PSSR_ROOT` names a directory ahead of all of them.

| File under the checkout | Holds |
|---|---|
| `checkpoints/pretrained_models/base.safetensors` | The steady restoration pass |
| `checkpoints/pretrained_models/draft.safetensors` | The sharp restoration pass |
| `dependent_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth` | The Wan VAE |
| `models/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl` | The umt5 tokenizer |
| `Wan_SR/pipelines/` | The two pipelines the node builds |

A checkout missing any of the first four is named in the error along with what is absent.

The transformer is not in that list. It comes in on the `model` input from **Load Diffusion
Model**, as Wan 2.1 T2V-1.3B or a finetune of it. The tagger, its LoRA and the 11 GB text
encoder are not read at all.

`diffsynth` is the one package the pack imports for this that ComfyUI does not already have.
The checkout's own requirements file names the rest.

## Kandinsky 6

ComfyUI's own loaders read every file, and nothing is fetched. The pack publishes them at
<https://huggingface.co/WAS/was-node-suite-weights> under `models/kandinsky6/`, laid out as
ComfyUI's `models` folder. The transformer files Kandinsky Lab publishes,
`transformer/diffusion_pytorch_model.safetensors` in each Kandinsky-6.0 repository, load as they
are.

| File | Goes in | Size | Loaded by |
|---|---|--:|---|
| [`kandinsky6_lite_distill_5s_int8_convrot.safetensors`](https://huggingface.co/WAS/was-node-suite-weights/blob/main/models/kandinsky6/diffusion_models/kandinsky6_lite_distill_5s_int8_convrot.safetensors) | `ComfyUI/models/diffusion_models` | 4.2 GB | Load Diffusion Model |
| [`kandinsky6_lite_distill_5s_w6a8.safetensors`](https://huggingface.co/WAS/was-node-suite-weights/blob/main/models/kandinsky6/diffusion_models/kandinsky6_lite_distill_5s_w6a8.safetensors) | `ComfyUI/models/diffusion_models` | 3.8 GB | Load Diffusion Model |
| [`kandinsky6_lite_5s_int8_convrot.safetensors`](https://huggingface.co/WAS/was-node-suite-weights/blob/main/models/kandinsky6/diffusion_models/kandinsky6_lite_5s_int8_convrot.safetensors) | `ComfyUI/models/diffusion_models` | 4.2 GB | Load Diffusion Model |
| [`kandinsky6_lite_5s_w6a8.safetensors`](https://huggingface.co/WAS/was-node-suite-weights/blob/main/models/kandinsky6/diffusion_models/kandinsky6_lite_5s_w6a8.safetensors) | `ComfyUI/models/diffusion_models` | 3.8 GB | Load Diffusion Model |
| [`kandinsky6_pro_distill_5s_int8_convrot.safetensors`](https://huggingface.co/WAS/was-node-suite-weights/blob/main/models/kandinsky6/diffusion_models/kandinsky6_pro_distill_5s_int8_convrot.safetensors) | `ComfyUI/models/diffusion_models` | 34.9 GB | Load Diffusion Model |
| [`kandinsky6_pro_distill_5s_w6a8.safetensors`](https://huggingface.co/WAS/was-node-suite-weights/blob/main/models/kandinsky6/diffusion_models/kandinsky6_pro_distill_5s_w6a8.safetensors) | `ComfyUI/models/diffusion_models` | 29.4 GB | Load Diffusion Model |
| [`kandinsky6_pro_5s_int8_convrot.safetensors`](https://huggingface.co/WAS/was-node-suite-weights/blob/main/models/kandinsky6/diffusion_models/kandinsky6_pro_5s_int8_convrot.safetensors) | `ComfyUI/models/diffusion_models` | 34.9 GB | Load Diffusion Model |
| [`kandinsky6_pro_5s_w6a8.safetensors`](https://huggingface.co/WAS/was-node-suite-weights/blob/main/models/kandinsky6/diffusion_models/kandinsky6_pro_5s_w6a8.safetensors) | `ComfyUI/models/diffusion_models` | 29.3 GB | Load Diffusion Model |
| [`qwen_2.5_vl_7b_fp8_scaled.safetensors`](https://huggingface.co/WAS/was-node-suite-weights/blob/main/models/kandinsky6/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors) | `ComfyUI/models/text_encoders` | 9.4 GB | DualCLIPLoader, type `kandinsky5` |
| [`clip_l.safetensors`](https://huggingface.co/WAS/was-node-suite-weights/blob/main/models/kandinsky6/text_encoders/clip_l.safetensors) | `ComfyUI/models/text_encoders` | 246 MB | DualCLIPLoader, type `kandinsky5` |
| [`kandinsky6_audio_vae.safetensors`](https://huggingface.co/WAS/was-node-suite-weights/blob/main/models/kandinsky6/vae/kandinsky6_audio_vae.safetensors) | `ComfyUI/models/vae` | 1.7 GB | Kandinsky 6 Audio VAE Loader |
| [`hunyuan_video_vae_bf16.safetensors`](https://huggingface.co/Comfy-Org/HunyuanVideo_repackaged/blob/main/split_files/vae/hunyuan_video_vae_bf16.safetensors) | `ComfyUI/models/vae` | 493 MB | Load VAE |

`hunyuan_video_vae_bf16.safetensors` is published by Comfy-Org at
<https://huggingface.co/Comfy-Org/HunyuanVideo_repackaged> under `split_files/vae/`.

| Transformer | Weights |
|---|---|
| `_int8_convrot` | 8-bit, run as 8-bit |
| `_w6a8` | 6-bit, run with 8-bit activations, the smallest |
| Kandinsky Lab's own file | bfloat16 |

| Transformer | KSampler |
|---|---|
| `_distill_` | 10 steps, CFG 1, `euler`, `simple` |
| the others | 50 steps, CFG 5, `euler`, `simple` |

| File | Licence |
|---|---|
| The Kandinsky 6 transformers | MIT, Kandinsky Lab |
| `qwen_2.5_vl_7b_fp8_scaled.safetensors` | Apache-2.0, Alibaba Cloud |
| `clip_l.safetensors` | MIT, OpenAI |
| `kandinsky6_audio_vae.safetensors` | **CC BY-NC 4.0, non-commercial use only**, the MMAudio autoencoder in it; MIT, the BigVGAN vocoder |
| `hunyuan_video_vae_bf16.safetensors` | Tencent Hunyuan Community License, which does not apply in the European Union, the United Kingdom or South Korea |

## Fibo 2.0

ComfyUI's own loaders read every file, and nothing is fetched. The transformers are Comfy-Org's
repackaging at <https://huggingface.co/Comfy-Org/Fibo-2.0>. Bria's own Diffusers transformer, the
`transformer/` folder of either Fibo-2.0 repository joined into one file, loads too.

| File | Goes in | Size | Loaded by |
|---|---|--:|---|
| [`fibo2_dev_bf16.safetensors`](https://huggingface.co/Comfy-Org/Fibo-2.0/blob/main/diffusion_models/fibo2_dev_bf16.safetensors) | `ComfyUI/models/diffusion_models` | 16.5 GB | Load Diffusion Model |
| [`fibo2_dev_int8_convrot.safetensors`](https://huggingface.co/Comfy-Org/Fibo-2.0/blob/main/diffusion_models/fibo2_dev_int8_convrot.safetensors) | `ComfyUI/models/diffusion_models` | 8.4 GB | Load Diffusion Model |
| [`fibo2_edit_turbo_bf16.safetensors`](https://huggingface.co/Comfy-Org/Fibo-2.0/blob/main/diffusion_models/fibo2_edit_turbo_bf16.safetensors) | `ComfyUI/models/diffusion_models` | 16.5 GB | Load Diffusion Model |
| [`fibo2_edit_turbo_int8_convrot.safetensors`](https://huggingface.co/Comfy-Org/Fibo-2.0/blob/main/diffusion_models/fibo2_edit_turbo_int8_convrot.safetensors) | `ComfyUI/models/diffusion_models` | 8.4 GB | Load Diffusion Model |
| [`qwen3vl_4b_bf16.safetensors`](https://huggingface.co/Comfy-Org/Fibo-2.0/blob/main/text_encoders/qwen3vl_4b_bf16.safetensors) | `ComfyUI/models/text_encoders` | 8.9 GB | Load CLIP, type `fibo2` where ComfyUI offers it, otherwise `krea2` |
| [`flux2-vae.safetensors`](https://huggingface.co/Comfy-Org/Fibo-2.0/blob/main/vae/flux2-vae.safetensors) | `ComfyUI/models/vae` | 336 MB | Load VAE |

| Transformer | Fibo 2 Scheduler | CFGGuider |
|---|---|---|
| Dev | 50 steps, 30 for a draft | CFG 5 |
| Edit Turbo | 4 steps | CFG 1 |

Sample with `euler` in SamplerCustomAdvanced. Dev generates and edits; Edit Turbo edits.

| File | Licence |
|---|---|
| The Fibo 2.0 transformers | **CC BY-NC 4.0, non-commercial use only**, Bria AI; commercial use needs an agreement with Bria at <https://platform.bria.ai/licensing/pricing?model=fibo-2> |
| `qwen3vl_4b_bf16.safetensors` | Apache-2.0, Alibaba Cloud |
| `flux2-vae.safetensors` | Apache-2.0, Black Forest Labs |

| What you see | What it means |
|---|---|
| `Fibo 2.0 reads its prompt with Qwen3-VL 4B` | Load CLIP holds another text encoder. Load `qwen3vl_4b_bf16.safetensors` |
| `An edit encodes its images with the FLUX.2 VAE` | An image is connected and `vae` is not. Connect Load VAE with `flux2-vae.safetensors` |
| `samples a FLUX.2 latent of 128 channels` | The latent came from another node. Sample the latent Text Encode Fibo 2 hands on |
| Lettering comes out garbled | The prompt is plain text. Write a JSON caption with a `text_render` entry, or let Generate Fibo 2 Prompt write one |

## Quantized models

**Quantize Model** writes to `ComfyUI/output/diffusion_models/` by default, numbered
`quantized_00001.safetensors` onward. Load Diffusion Model reads the file once it is in
`ComfyUI/models/diffusion_models`. Each quantized layer carries its own `comfy_quant` tag, so the
file loads with no loader setting.

| Format | Weights | Runs natively on |
|---|---|---|
| `int8 convrot` | 8-bit, each row rotated by a Hadamard block of 256, 64 or 16 | RTX 20 series and newer |
| `w6a8` | 6-bit in groups of 32, with 8-bit activations | RTX 30 series and newer |
| `w4a8` | 4-bit in groups of 16 with a codebook, with 8-bit activations | RTX 30 series and newer |
| `int8` | 8-bit, unrotated | RTX 20 series and newer |
| `fp8 e4m3fn` | 8-bit float, one scale per layer | RTX 40 series and newer |
| `mxfp8` | 8-bit float, a power-of-two scale per 32 weights | RTX 50 series, emulated on older cards |
| `nvfp4` | 4-bit float, a scale per 16 weights and one per layer | RTX 50 series, emulated on older cards |

With `layers`, `scales` and `attention` on `like Comfy-Org`, these releases come out tensor for
tensor from the bfloat16 file beside them in the same repository, apart from the hardware
differences below:

| Release | Layers quantized | Scales | Other tensors |
|---|---|---|---|
| [`flux3_image_klein_sd_int8_convrot.safetensors`](https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy) | The transformer blocks, with `qkv` on `fused` | Each row's maximum | As loaded |
| [`krea2_raw_int8_convrot.safetensors`](https://huggingface.co/Comfy-Org/Krea-2/tree/main/diffusion_models) | The 28 transformer blocks | Each row's lowest error | Float32 as bfloat16, norm scales kept |
| [`krea2_raw_fp8_scaled.safetensors`](https://huggingface.co/Comfy-Org/Krea-2/tree/main/diffusion_models) | The transformer and text-fusion blocks | One per layer | All bfloat16 |
| [`krea2_turbo_mxfp8.safetensors`](https://huggingface.co/Comfy-Org/Krea-2/tree/main/diffusion_models) | The transformer and text-fusion blocks | One per 32 weights | All bfloat16 |
| [`krea2_turbo_nvfp4.safetensors`](https://huggingface.co/Comfy-Org/Krea-2/tree/main/diffusion_models) | The transformer and text-fusion blocks | One per 16 weights and one per layer | All bfloat16 |

The Krea 2 fp8, mxfp8 and nvfp4 releases tag each block's attention gate, attention output and
feed-forward output to multiply at full precision, and so does Quantize Model. A model with no
release here takes the layers Comfy-Org's quantizer picks, scaled by row maximum.

Results depend on the hardware in two places:

| Where | What differs |
|---|---|
| `lowest error` row scales | Each row's error is a float32 sum. Where two clips tie to the last bit, the order the processor adds in picks between them, so a file made on other hardware can differ in those rows. Krea 2 Raw int8 convrot made on an RTX 4090 and a 24-thread desktop CPU differs from Comfy-Org's in 10 of its 1,691,648 rows. |
| `nvfp4` weights | A weight that lands exactly between two 4-bit levels rounds by the GPU's own division. Made without a GPU, those weights round to even and differ from Comfy-Org's. |

| Trouble | Cause |
|---|---|
| The file is not in Load Diffusion Model's menu | It is in the output folder. Move it into `ComfyUI/models/diffusion_models` and press **R** |
| Save Model fails with `Only Tensors of floating point and complex dtype can require gradients` | Core Save Model cannot write a quantized model. Give Quantize Model a `filename_prefix` instead |
| `is not available in this ComfyUI` | ComfyUI or comfy-kitchen is older than the format. Update both, or pick `int8 convrot` |

## The gated model nodes

| Group | Goes in | Widget | Repository |
|---|---|---|---|
| `sam` | `models/sams` | `ViT-H` | [facebook/sam-vit-huge](https://huggingface.co/facebook/sam-vit-huge) |
| | | `ViT-L` | [facebook/sam-vit-large](https://huggingface.co/facebook/sam-vit-large) |
| | | `ViT-B` | [facebook/sam-vit-base](https://huggingface.co/facebook/sam-vit-base) |
| `blip` | `models/blip` | caption | [Salesforce/blip-image-captioning-base](https://huggingface.co/Salesforce/blip-image-captioning-base) |
| | | question | [Salesforce/blip-vqa-base](https://huggingface.co/Salesforce/blip-vqa-base) |
| `clipseg` | `models/clipseg` | default | [CIDAS/clipseg-rd64-refined](https://huggingface.co/CIDAS/clipseg-rd64-refined) |
| `midas` | `models/midas` | `DPT_Large` | [Intel/dpt-large](https://huggingface.co/Intel/dpt-large) |
| | | `DPT_Hybrid` | [Intel/dpt-hybrid-midas](https://huggingface.co/Intel/dpt-hybrid-midas) |
| | | `DPT_Small` | [Intel/dpt-swinv2-tiny-256](https://huggingface.co/Intel/dpt-swinv2-tiny-256) |
| `diffusers` | `models/diffusers` | any | any diffusers-format repository, kept as `<owner>/<name>` |
| `birefnet` | `models/birefnet` | any | any full size BiRefNet `.safetensors`, published as [WAS/was-node-suite-weights](https://huggingface.co/WAS/was-node-suite-weights/tree/main/birefnet) |
| `ben2` | `models/ben2` | any | any BEN2 base `.safetensors`, published as `ben2-base.safetensors` |

Each is searched under its folder in these layouts, and Hugging Face's own cache after that.

| Layout | Example |
|---|---|
| The name alone | `models/blip/blip-image-captioning-base/` |
| Owner and name | `models/blip/Salesforce/blip-image-captioning-base/` |
| A Hugging Face cache tree | `models/blip/models--Salesforce--blip-image-captioning-base/snapshots/<rev>/` |

## Example workflows

The graphs in `docs/workflows` load these files through ComfyUI's own loaders. The Kandinsky 6,
Marigold v2, EMA-VFI and optical flow files they read are in those sections above. A graph names
some LoRAs inside a subfolder, as `H3/taomate_h3_3step_comfy.safetensors`: a file kept in that
subfolder loads as the graph names it, and one kept elsewhere is picked again from the loader.

### MiniMax H3

Graphs: every `minimax-h3-*.json` and `affine-minimax-h3-*.json`.

| File | Goes in | Size | Download | Licence |
|---|---|--:|---|---|
| `minimax_h3_fl2va_int8_convrot.safetensors` | `ComfyUI/models/diffusion_models` | 34.0 GB | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/diffusion_models/minimax_h3_fl2va_int8_convrot.safetensors) | MiniMax H3 Community License |
| `minimax_h3_fl2va_pruned_int8_convrot.safetensors` | `ComfyUI/models/diffusion_models` | 21.0 GB | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors) | MiniMax H3 Community License |
| `minimax_h3_fl2va_pruned_w6a8.safetensors` | `ComfyUI/models/diffusion_models` | 16.0 GB | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/diffusion_models/minimax_h3_fl2va_pruned_w6a8.safetensors) | MiniMax H3 Community License |
| `minimax_h3_ref2va_pruned_int8_convrot.safetensors` | `ComfyUI/models/diffusion_models` | 21.0 GB | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors) | MiniMax H3 Community License |
| `minimax_h3_ref2va_pruned_w6a8.safetensors` | `ComfyUI/models/diffusion_models` | 16.0 GB | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/diffusion_models/minimax_h3_ref2va_pruned_w6a8.safetensors) | MiniMax H3 Community License |
| `minimax_h3_ref2va_silverturbo_pruned_int8_convrot.safetensors` | `ComfyUI/models/diffusion_models` | 21.0 GB | [silveroxides/MiniMax-H3_tests](https://huggingface.co/silveroxides/MiniMax-H3_tests/blob/main/diffusion_models/minimax_h3_ref2va_silverturbo_pruned_int8_convrot.safetensors) | None stated |
| `singularity_minimax_h3_ref2va_pruned_w4a8_mixed.safetensors` | `ComfyUI/models/diffusion_models` | 11.8 GB | Withdrawn upstream. Its replacement is [`Minimax-h3_Singularity_ref2va_v1.3_Pruned_w4a8.safetensors`](https://huggingface.co/WarmBloodAban/Minimax-h3_Singularity/blob/main/Minimax-h3_Singularity_ref2va_v1.3_Pruned_w4a8.safetensors), picked in the loader in its place | Apache-2.0 |
| `minimax_h3_video_vae_fp16.safetensors` | `ComfyUI/models/vae` | 5.2 GB | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/vae/minimax_h3_video_vae_fp16.safetensors) | MiniMax H3 Community License |
| `minimax_h3_video_vae_int8_convrot.safetensors` | `ComfyUI/models/vae` | 2.8 GB | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/vae/minimax_h3_video_vae_int8_convrot.safetensors) | MiniMax H3 Community License |
| `minimax_h3_audio_vae_fp32.safetensors` | `ComfyUI/models/vae` | 605 MB | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/vae/minimax_h3_audio_vae_fp32.safetensors) | MiniMax H3 Community License |
| `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | `ComfyUI/models/text_encoders` | 15.7 GB | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors) | MiniMax H3 Community License |
| `qwen3vl_8b_fp8_scaled.safetensors` | `ComfyUI/models/text_encoders` | 10.6 GB | [Comfy-Org/Qwen3-VL](https://huggingface.co/Comfy-Org/Qwen3-VL/blob/main/text_encoders/qwen3vl_8b_fp8_scaled.safetensors) | Apache-2.0 |
| `minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors` | `ComfyUI/models/loras/H3` | 2.0 GB | [lightx2v/Minimax-h3-Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo/blob/main/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors) | Apache-2.0 |
| `minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors` | `ComfyUI/models/loras/H3` | 2.0 GB | [lightx2v/Minimax-h3-Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo/blob/main/minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors) | Apache-2.0 |
| `dmad_minimax_h3_4step_full_critic_comfyui.safetensors` | `ComfyUI/models/loras/H3` | 2.0 GB | [ZhengmingYu/DMAD](https://huggingface.co/ZhengmingYu/DMAD/blob/main/minimax_h3/dmad_minimax_h3_4step_full_critic_comfyui.safetensors) | MiniMax H3 Community License |
| `minimax_h3_DMAD_4step_full_lora_avg_rank_39_bf16.safetensors` | `ComfyUI/models/loras/H3` | 367 MB | [Minimax H3 Turbo LoRas on Civitai](https://civitai.com/models/2837571?modelVersionId=3382801), version `DMAD_4step_full` | On the Civitai page |
| `taomate_h3_3step_comfy.safetensors` | `ComfyUI/models/loras/H3` | 2.5 GB | [Robert1212star/TaoMate-H3-3Step-ComfyUI](https://huggingface.co/Robert1212star/TaoMate-H3-3Step-ComfyUI/blob/main/taomate_h3_3step_comfy.safetensors) | On the repository |

### Krea 2

Graphs: `affine-krea2.json`, `cns-krea2.json`.

| File | Goes in | Size | Download | Licence |
|---|---|--:|---|---|
| `krea2_raw_bf16.safetensors` | `ComfyUI/models/diffusion_models` | 26.3 GB | [Comfy-Org/Krea-2](https://huggingface.co/Comfy-Org/Krea-2/blob/main/diffusion_models/krea2_raw_bf16.safetensors) | Krea 2 Community License |
| `krea2_raw_fp8_scaled.safetensors` | `ComfyUI/models/diffusion_models` | 13.1 GB | [Comfy-Org/Krea-2](https://huggingface.co/Comfy-Org/Krea-2/blob/main/diffusion_models/krea2_raw_fp8_scaled.safetensors) | Krea 2 Community License |
| `krea2_turbo_fp8_scaled.safetensors` | `ComfyUI/models/diffusion_models` | 13.1 GB | [Comfy-Org/Krea-2](https://huggingface.co/Comfy-Org/Krea-2/blob/main/diffusion_models/krea2_turbo_fp8_scaled.safetensors) | Krea 2 Community License |
| `qwen3vl_4b_fp8_scaled.safetensors` | `ComfyUI/models/text_encoders` | 5.2 GB | [Comfy-Org/Krea-2](https://huggingface.co/Comfy-Org/Krea-2/blob/main/text_encoders/qwen3vl_4b_fp8_scaled.safetensors) | Krea 2 Community License |
| `qwen_image_vae.safetensors` | `ComfyUI/models/vae` | 254 MB | [Comfy-Org/Krea-2](https://huggingface.co/Comfy-Org/Krea-2/blob/main/vae/qwen_image_vae.safetensors) | Krea 2 Community License |
| `wan_2.1_vae.safetensors` | `ComfyUI/models/vae` | 254 MB | [Comfy-Org/Wan_2.1_ComfyUI_repackaged](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/blob/main/split_files/vae/wan_2.1_vae.safetensors) | Apache-2.0 |
| `krea2_darkbrush.safetensors` | `ComfyUI/models/loras` | 469 MB | [Comfy-Org/Krea-2](https://huggingface.co/Comfy-Org/Krea-2/blob/main/loras/krea2_darkbrush.safetensors) | Krea 2 Community License |
| `krea2_raw_to_turbo_r256_comfy.safetensors` | `ComfyUI/models/loras/Krea2` | 1.7 GB | [Krea 2 Turbo LoRA 256dim on Civitai](https://civitai.com/models/2727641?modelVersionId=3066079) | On the Civitai page |

### FLUX.2 klein 9B

Graph: `cns-klein9b.json`. Hugging Face asks for a signed-in account that has accepted the terms
before it serves `flux-2-klein-9b-fp8.safetensors`.

| File | Goes in | Size | Download | Licence |
|---|---|--:|---|---|
| `flux-2-klein-9b-fp8.safetensors` | `ComfyUI/models/diffusion_models` | 9.4 GB | [black-forest-labs/FLUX.2-klein-9b-fp8](https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-fp8) | **FLUX Non-Commercial License** |
| `qwen_3_8b_fp8mixed.safetensors` | `ComfyUI/models/text_encoders` | 8.7 GB | [Comfy-Org/vae-text-encorder-for-flux-klein-9b](https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-9b/blob/main/split_files/text_encoders/qwen_3_8b_fp8mixed.safetensors) | **FLUX Non-Commercial License**, as the repository is labelled |
| `flux2-vae.safetensors` | `ComfyUI/models/vae` | 336 MB | [Comfy-Org/flux2-dev](https://huggingface.co/Comfy-Org/flux2-dev/blob/main/split_files/vae/flux2-vae.safetensors) | **FLUX.1 [dev] Non-Commercial License**, as the repository is labelled |

### FLUX 3 Image Klein

Graphs: `flux3-klein-distill-lora.json`, `flux3-klein-tiled-diffusion.json`. Hugging Face asks
for a signed-in account that has accepted the terms before it serves the `black-forest-labs`
files.

| File | Goes in | Size | Download | Licence |
|---|---|--:|---|---|
| `flux3_image_klein_bf16.safetensors` | `ComfyUI/models/diffusion_models` | 11.1 GB | [black-forest-labs/flux-3-image-klein-comfy](https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy/blob/main/diffusion_models/flux3_image_klein_bf16.safetensors) | **FLUX Kommunity License** |
| `flux3_image_klein_sd_bf16.safetensors` | `ComfyUI/models/diffusion_models` | 11.1 GB | [black-forest-labs/flux-3-image-klein-comfy](https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy/blob/main/diffusion_models/flux3_image_klein_sd_bf16.safetensors) | **FLUX Kommunity License** |
| `flux3_image_klein_distill_r256.safetensors` | `ComfyUI/models/loras` | 1.3 GB | [WAS/flux3_klein](https://huggingface.co/WAS/flux3_klein/blob/main/flux3_image_klein_distill_r256.safetensors) | **FLUX Kommunity License** |
| `flux3_image_klein_distill_r64.safetensors` | `ComfyUI/models/loras` | 338 MB | [WAS/flux3_klein](https://huggingface.co/WAS/flux3_klein/blob/main/flux3_image_klein_distill_r64.safetensors) | **FLUX Kommunity License** |
| `flux3_image_klein_distill_r32.safetensors` | `ComfyUI/models/loras` | 169 MB | [WAS/flux3_klein](https://huggingface.co/WAS/flux3_klein/blob/main/flux3_image_klein_distill_r32.safetensors) | **FLUX Kommunity License** |
| `flux3_image_klein_distill_r16.safetensors` | `ComfyUI/models/loras` | 85 MB | [WAS/flux3_klein](https://huggingface.co/WAS/flux3_klein/blob/main/flux3_image_klein_distill_r16.safetensors) | **FLUX Kommunity License** |
| `qwen3vl_4b_int8_convrot.safetensors` | `ComfyUI/models/text_encoders` | 4.9 GB | [black-forest-labs/flux-3-image-klein-comfy](https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy/blob/main/text_encoders/qwen3vl_4b_int8_convrot.safetensors) | Apache-2.0, Qwen3-VL-4B-Instruct |
| `flux2-vae.safetensors` | `ComfyUI/models/vae` | 336 MB | [black-forest-labs/flux-3-image-klein-comfy](https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy/blob/main/vae/flux2-vae.safetensors) | **FLUX Kommunity License**, as the repository is labelled |

The four LoRAs are the distilled checkpoint minus the base, one per rank; the higher the rank,
the closer the base comes to the distilled picture at CFG 1. **Flux 3 Flow Map Patch** reads the
two `ref_time_in` weights from `flux3_image_klein_sd_bf16.safetensors` or
`flux3_image_klein_sd_int8_convrot.safetensors`, so one distilled file stays in
`ComfyUI/models/diffusion_models` beside the base.

| What you see | What it means |
|---|---|
| A fine stripe every 4 pixels down the rows, and a blue cast | The LoRA is on the base without **Flux 3 Flow Map Patch**. Put the patch between the LoRA and the sampler |
| `holds no ref_time_in weights` | The checkpoint menu names a base file. Pick a file with `_sd_` in its name |
| `takes a FLUX 3 Image model` | The model wired in is another architecture. Wire the Load Diffusion Model on `flux3_image_klein_bf16.safetensors` |

### MOSS-Image

Graph: `moss-image-high-res.json`.

| File | Goes in | Size | Download | Licence |
|---|---|--:|---|---|
| `moss_image_sft_int8_convrot.safetensors` | `ComfyUI/models/diffusion_models` | 5.4 GB | [Comfy-Org/MOSS-Image](https://huggingface.co/Comfy-Org/MOSS-Image/blob/main/diffusion_models/moss_image_sft_int8_convrot.safetensors) | Apache-2.0 |
| `moss_image_base_int8_convrot.safetensors` | `ComfyUI/models/diffusion_models` | 5.4 GB | [Comfy-Org/MOSS-Image](https://huggingface.co/Comfy-Org/MOSS-Image/blob/main/diffusion_models/moss_image_base_int8_convrot.safetensors) | Apache-2.0 |
| `qwen3.5_9b_int8_convrot.safetensors` | `ComfyUI/models/text_encoders` | 9.7 GB | [Comfy-Org/MOSS-Image](https://huggingface.co/Comfy-Org/MOSS-Image/blob/main/text_encoders/qwen3.5_9b_int8_convrot.safetensors) | Apache-2.0 |
| `flux2-vae.safetensors` | `ComfyUI/models/vae` | 336 MB | [Comfy-Org/MOSS-Image](https://huggingface.co/Comfy-Org/MOSS-Image/blob/main/vae/flux2-vae.safetensors) | Apache-2.0, as the repository is labelled |

**Model Block Restore** reads only the restored blocks from `moss_image_base_int8_convrot.safetensors`
or `moss_image_base_bf16.safetensors`, so the Base file stays in `ComfyUI/models/diffusion_models`
beside the SFT.

| What you see | What it means |
|---|---|
| `holds no layers.32 weights` | The checkpoint menu names another model. Pick `moss_image_base_int8_convrot.safetensors` |
| Faint vertical stripes across a smooth sky on MOSS-Image SFT | The model runs without **Model Block Restore**, or with `first_block` above 32. Restore blocks 32 to 41 from the Base file |
| `reads no RoPE scale` | **RoPE Extension Patch** is on a model other than MOSS-Image or FLUX 1 and 2. Leave the patch out |
| `has no shift to set` | **Flow Shift** is on a model that is not a flow model. Leave Flow Shift out for it |

### The other graphs

| File | Goes in | Size | Download | Licence | Graphs |
|---|---|--:|---|---|---|
| `autismmixSDXL_autismmixConfetti.safetensors` | `ComfyUI/models/checkpoints` | 6.9 GB | [AutismMix SDXL on Civitai](https://civitai.com/models/288584?modelVersionId=324524), version `AutismMix_confetti` | On the Civitai page | `fast-samplers-and-savers.json`, `hdr.json` |
| `qwen_3_4b.safetensors` | `ComfyUI/models/text_encoders` | 8.0 GB | [Comfy-Org/z_image_turbo](https://huggingface.co/Comfy-Org/z_image_turbo/blob/main/split_files/text_encoders/qwen_3_4b.safetensors) | Apache-2.0 | `fast-generate-text.json` |
| `4xNomosWebPhoto_RealPLKSR.safetensors` | `ComfyUI/models/upscale_models` | 30 MB | [Phips/4xNomosWebPhoto_RealPLKSR](https://huggingface.co/Phips/4xNomosWebPhoto_RealPLKSR/blob/main/4xNomosWebPhoto_RealPLKSR.safetensors) | CC-BY-4.0 | `vtc_temporal_upscale.json` |
| `skin_detail_slider_krea2_ZT-V1.safetensors` | `ComfyUI/models/loras/Krea2` | 1.2 MB | [Skin Detail Slider on Civitai](https://civitai.com/models/2936275?modelVersionId=3323867), version `V1` | On the Civitai page | `power-lora-stack.json` |
| `day_&_night_slider_0ts-krea2-v1.safetensors` | `ComfyUI/models/loras/Krea2` | 1.2 MB | [Day & Night Slider on Civitai](https://civitai.com/models/2957416?modelVersionId=3350109), version `V1` | On the Civitai page | `power-lora-stack.json` |

## Troubleshooting

| What you see | What it means |
|---|---|
| The menu reads `put a checkpoint in models/EMA-VFI` | Nothing on disk and `features.network` off. Turn it on, or place a file and restart |
| The menu reads `put a checkpoint in models/optical_flow` | Nothing on disk and `features.network` off. Turn it on, or place a file from the optical flow table and restart |
| The Prompt Timeline's Preview reads `No MiniMax H3 Preview VAE` | No `taeh3` file is in a `vae_approx` folder ComfyUI reads. Hover **Unavailable** for the folders searched |
| The Prompt Timeline's Preview reads `Invalid MiniMax H3 Preview VAE` | The file is not the `taeh3.safetensors` in the MiniMax H3 table. Replace it; the next run reads it |
| The Prompt Timeline's Preview reads `MiniMax H3 Preview Failed` | Decoding a step raised. Hover **Unavailable** for the error; sampling goes on without previews |
| The file is on disk but not in the menu | Press **R** on the canvas |
| A BiRefNet file is on disk but not in the cutout menu | It is a Lite file or another architecture. The menu lists full size BiRefNet and BEN2 base files only |
| `is a link to ..., which ComfyUI cannot open` | The model folder or file is a link that does not resolve from ComfyUI. Recreate it with `mklink` on Windows, or point `extra_model_paths.yaml` at the real folder |
| `was not found ... Setting features.network: true` | The run wanted a file you have not got, with fetching off |
| `could not be fetched` | Fetching is on and the download failed. The message carries the reason and the directory to place it in |
| `only trained to land halfway between two frames` | The multiplier is above 2 and this is not a `_t` file |
| `Could not detect model type` on a Kandinsky 6 file | The file is not a Kandinsky 6 transformer. Pick one from the Kandinsky 6 table, or `transformer/diffusion_pytorch_model.safetensors` from a Kandinsky-6.0 repository |
| `This file is not the Kandinsky 6 audio VAE` | Kandinsky 6 Audio VAE Loader is set to another file in `models/vae` |
| Out of memory on Kandinsky 6 Pro | The `_w6a8` file is the smallest. Pro runs on a 24 GB card in either quantised form, streamed through ComfyUI's VRAM management |
| ComfyUI quits with `Fatal Python error: Aborted` in VAE Decode after Kandinsky 6 Pro | Decode with VAE Decode (Tiled): `tile_size` 512, `overlap` 64, `temporal_size` 64, `temporal_overlap` 8 |
| Sound from a `pretrain` Kandinsky 6 transformer is off | The pack decodes every non-distilled transformer at audio scale 0.5302, the released checkpoints' value; the pretrained ones use 0.417 |
