# Built-in system prompts

The prompts listed under `Built-in/` in the system prompt library. A file whose front matter
carries `source` and `license` holds text from the project named below, under that project's
licence. Every other file is part of WAS Node Suite and under its licence.

| Licence | Text |
|---|---|
| Apache-2.0 | `LICENSE-Apache-2.0.txt` |
| MIT, Black Forest Labs Inc. | `LICENSE-MIT-Black-Forest-Labs.txt` |
| MIT, BRIA AI | `LICENSE-MIT-Bria.txt` |
| MIT, Kandinsky Lab | `LICENSE-MIT-Kandinsky-Lab.txt` |

## Sources

| Built-in files | Project | Taken at | Licence | Changed |
|---|---|---|---|---|
| `FLUX/FLUX.2 Prompt Upsampler.md`, `FLUX/FLUX.2 Edit Upsampler.md` | [black-forest-labs/flux2](https://github.com/black-forest-labs/flux2), `src/flux2/system_messages.py` | `50fe516` | Apache-2.0 | no |
| `FLUX/FLUX 3 Klein *.md`, eight files | [black-forest-labs/flux-3-image-klein](https://github.com/black-forest-labs/flux-3-image-klein), `src/flux3/pup_prompts.json` and `pup_prompts_external_vlm.json`. Copyright 2026 Black Forest Labs, Inc. | `bfc8889` | Apache-2.0 | no |
| `FLUX/flux-prompting/references/*.md`, ten files | [black-forest-labs/skills](https://github.com/black-forest-labs/skills), `skills/flux-image-best-practices/rules/` | `8907d51` | MIT | no |
| `Krea/Krea 2 Prompt Expansion.md` | [krea-ai/krea-2](https://github.com/krea-ai/krea-2), `docs/expansion.txt` | `db3984f` | Apache-2.0 | no |
| `MOSS-Image/moss-image-prompt-enhancer/SKILL.md` | [OpenMOSS/MOSS-Image](https://github.com/OpenMOSS/MOSS-Image), `skills/moss-image-prompt-enhancer/SKILL.md` | `2b46b1e` | Apache-2.0 | yes, below |
| `MOSS-Image/moss-image-prompt-enhancer/references/layout.md` | the same skill, `references/layout.md` | `2b46b1e` | Apache-2.0 | no |
| `MOSS-Image/MOSS-Image Recaption.md` | the same repository, `USERGUIDE.md`, the caption prompt | `2b46b1e` | Apache-2.0 | yes, below |
| `Fibo/Fibo 2 *.md`, seven files | [Bria-AI/vgl](https://github.com/Bria-AI/vgl), `src/bria_vgl/dialects/thales/prompts/` and `input_schema/` | `676b993` | MIT | yes, below |
| `LTX/LTX-Video *.md`, two files | [Lightricks/LTX-Video](https://github.com/Lightricks/LTX-Video), `ltx_video/utils/prompt_enhance_utils.py` | `4b2d053` | Apache-2.0 | no |
| `LTX/LTX-2 Text to Video Enhancer.md`, `LTX/LTX-2 Image to Video Enhancer.md`, `LTX/LTX-2.5 *.md` | [huggingface/diffusers](https://github.com/huggingface/diffusers), `src/diffusers/pipelines/ltx2/utils.py`. Copyright 2026 Lightricks and The HuggingFace Team | `40e551a` | Apache-2.0 | no |
| `Wan/Wan 2.2 *.md`, six files | [Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2), `wan/utils/system_prompt.py` | `1ea34ff` | Apache-2.0 | yes, below |
| `Wan/Wan First and Last Frame Extender*.md`, two files | [Wan-Video/Wan2.1](https://github.com/Wan-Video/Wan2.1), `wan/utils/prompt_extend.py` | `9737cba` | Apache-2.0 | yes, below |
| `Kandinsky/Kandinsky 6 *.md`, two files | [kandinskylab/kandinsky-6](https://github.com/kandinskylab/kandinsky-6), `comfyui/kandinsky6/beautifier_prompts/` | `7ceeefa` | MIT | no |
| `Qwen-Image/Qwen-Image Prompt Rewriter*.md`, `Qwen-Image/Qwen-Image 2512 *.md`, `Qwen-Image/Qwen-Image-Edit 2511 Instruction Rewriter.md` | [QwenLM/Qwen-Image](https://github.com/QwenLM/Qwen-Image), `src/examples/tools/prompt_utils.py` and `prompt_utils_2512.py` | `6b5e1f5` | Apache-2.0 | the edit rewriter, below |
| `Qwen-Image/Qwen-Image-Edit 2509 Instruction Rewriter.md` | [Qwen/Qwen-Image-Edit-2509](https://huggingface.co/spaces/Qwen/Qwen-Image-Edit-2509) Space, `app.py` | `757d88f` | Apache-2.0 | yes, below |
| `Z-Image/Z-Image Prompt Enhancer (Chinese).md` | [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/spaces/Tongyi-MAI/Z-Image-Turbo) Space, `pe.py` | `768cb50` | Apache-2.0 | yes, below |
| `ERNIE Image/ERNIE Image Prompt Enhancer.md` | [baidu/ERNIE-Image](https://huggingface.co/baidu/ERNIE-Image), `pe/chat_template.jinja` | `5346b31` | Apache-2.0 | yes, below |

## Changes

| File | Change |
|---|---|
| `MOSS-Image/moss-image-prompt-enhancer/SKILL.md` | The opening paragraph and the Compile and deliver section are replaced by an instruction to answer with the caption or layout JSON alone, and "Write a UTF-8 JSON file" reads "Answer with one JSON object". The sentences on asking the user, on outside research, on surfacing a size conflict and on the image-generation workflow are left out. The field list and composition checks of `references/layout.md` are folded into the Layout section, with a skeleton of the layout object. |
| `MOSS-Image/MOSS-Image Recaption.md` | The closing `Here is the original caption reference:` line and its `{{reference}}` slot are left out. |
| `Fibo/Fibo 2 *.md` | The field guide from `input_schema/` is inserted where the prompt reads `{llm_input_schema}`, and the `{style_preset_block}` slot is left empty, as the project's own prompt builder does. |
| `Wan/Wan 2.2 Text to Video Extender.md`, `Wan/Wan 2.2 Text to Video Extender (Chinese).md` | Rules 8 to 10, which replace the user's prompt with a different one for some kinds of content, are removed. |
| `Wan/Wan First and Last Frame Extender.md` | In rule 4, "The overall output should be in Chinese, retaining" reads "Retain". |
| `Qwen-Image/Qwen-Image-Edit 2511 Instruction Rewriter.md`, `Qwen-Image/Qwen-Image-Edit 2509 Instruction Rewriter.md` | The closing JSON output example is replaced by an instruction to answer in plain text. |
| `Z-Image/Z-Image Prompt Enhancer (Chinese).md` | The closing `用户输入 prompt: {prompt}` line is left out; the request is sent as the user message. |
| `ERNIE Image/ERNIE Image Prompt Enhancer.md` | The system prompt is taken out of the chat template that wraps it. |
