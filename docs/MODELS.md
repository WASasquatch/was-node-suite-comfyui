# Model weights

Nothing is downloaded at startup. A checkpoint is fetched by the first run that needs it, and
only with `features.network: true`. With that off, the run stops and names the file and every
folder it searched.

## EMA-VFI frame interpolation

Published at <https://huggingface.co/WAS/was-node-suite-weights> under `EMA-VFI/`. Goes in
`ComfyUI/models/EMA-VFI`.

| File | Size | Quality | Multiplier above 2 |
|---|---|---|---|
| `ours_t.safetensors` | 250 MB | best | **yes** |
| `ours.safetensors` | 250 MB | best | no |
| `ours_small_t.safetensors` | 56 MB | lower | **yes** |
| `ours_small.safetensors` | 56 MB | lower | no |

A `_t` file is the one to take: it lands a frame anywhere between two, so a multiplier above 2
works.

## HDR reconstruction

`hdrcnn.safetensors`, 112 MB, from the same repository under `hdr/`, into `ComfyUI/models/hdr`.
`models/hdr/hdr/` is read too, and an `extra_model_paths.yaml` entry under the key `hdr`.

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
| `qwen_image_edit_2509_int8_convrot.safetensors` | `ComfyUI/models/diffusion_models` | 19.1 GB |
| `marigold_v2_depth_log_stage2.safetensors` | `ComfyUI/models/loras` | 1.6 GB |
| `marigold_v2_normals.safetensors` | `ComfyUI/models/loras` | 1.6 GB |
| `marigold_v2_albedo.safetensors` | `ComfyUI/models/loras` | 1.6 GB |
| `marigold_v2_depth_log_stage2_vae.safetensors` | `ComfyUI/models/vae` | 0.24 GB |
| `marigold_v2_normals_vae.safetensors` | `ComfyUI/models/vae` | 0.24 GB |
| `marigold_v2_albedo_vae.safetensors` | `ComfyUI/models/vae` | 0.24 GB |
| `marigold_v2_depth_conditioning.safetensors` | `ComfyUI/models/embeddings` | 2.4 MB |
| `marigold_v2_normals_conditioning.safetensors` | `ComfyUI/models/embeddings` | 2.5 MB |
| `marigold_v2_albedo_conditioning.safetensors` | `ComfyUI/models/embeddings` | 2.6 MB |

Which map each answer takes:

| Answer | LoRA | VAE | Conditioning |
|---|---|---|---|
| `depth_map` | `..._depth_log_stage2` | `..._depth_log_stage2_vae` | `..._depth_conditioning` |
| `normal_map` | `..._normals` | `..._normals_vae` | `..._normals_conditioning` |
| `albedo` | `..._albedo` | `..._albedo_vae` | `..._albedo_conditioning` |

Wire **Load Diffusion Model** on the transformer into `model`, and **Load VAE** on that
row's decoder into `vae`. The adapter is the node's own job: `adapter_name` stays on `auto` and it
finds that row's LoRA by name. No **LoraLoaderModelOnly** is needed, and
one transformer feeds every map.

Set `adapter_name` to `already on the model` where you want to apply the LoRA yourself, at a
strength of your own or stacked with others.

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
| `model and vae are not wired` | `Marigold v2` reads both. Wire them. |
| `found no adapter` | That map's LoRA is not in the loras folder, or the model already carries one and `adapter_name` was not set to `already on the model`. |
| `found no prompt embedding` | That map's conditioning file is not in the embeddings folder. |
| A map that looks like noise | The LoRA and the VAE name different maps. |
| Depth flat or inverted | The answer already turns log depth over, so no invert node is wanted after it. |
| The `conditioning` menu is empty | The files are not in `ComfyUI/models/embeddings`. Put them there and reload the page. |

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
| `birefnet` | `models/birefnet` | any | one `.safetensors` per variant, named as the widget lists it |
| `ben2` | `models/ben2` | 1 | `ben2-base.safetensors` |

Each is searched under its folder in these layouts, and Hugging Face's own cache after that.

| Layout | Example |
|---|---|
| The name alone | `models/blip/blip-image-captioning-base/` |
| Owner and name | `models/blip/Salesforce/blip-image-captioning-base/` |
| A Hugging Face cache tree | `models/blip/models--Salesforce--blip-image-captioning-base/snapshots/<rev>/` |

## Troubleshooting

| What you see | What it means |
|---|---|
| The menu reads `put a checkpoint in models/EMA-VFI` | Nothing on disk and `features.network` off. Turn it on, or place a file and restart |
| The file is on disk but not in the menu | Press **R** on the canvas |
| `was not found ... Setting features.network: true` | The run wanted a file you have not got, with fetching off |
| `could not be fetched` | Fetching is on and the download failed. The message carries the reason and the directory to place it in |
| `only trained to land halfway between two frames` | The multiplier is above 2 and this is not a `_t` file |
