# EmbeddingGemma 2, ported

The text tower, the vision tower and the picture preprocessing of **EmbeddingGemma 2**, Google,
as implemented in Hugging Face **transformers**.

- Upstream: <https://github.com/huggingface/transformers>
- Taken at commit `6a9193deabe4ffcf495726897e3f2e651b8bbead` (2026-11-13)
- Licence: Apache-2.0, The Hugging Face team, retained verbatim in `LICENSE` beside this file
- Model weights: <https://huggingface.co/google/embeddinggemma-2>, Apache-2.0

**No weights are bundled.** The checkpoint is downloaded by the user into
`ComfyUI/models/LLM/google/embeddinggemma-2`.

## Files taken from upstream

| File | Upstream path | Changed |
|---|---|---|
| `model.py` | `src/transformers/models/embedding_gemma2/modeling_embedding_gemma2.py` | text tower, multimodal embedder and picture path, inference only |
| `model.py` | `src/transformers/models/gemma4/modeling_gemma4.py` | the vision tower, inference only |
| `model.py` | `src/transformers/models/embedding_gemma2/processing_embedding_gemma2.py` and `src/transformers/models/gemma4/image_processing_pil_gemma4.py` | the picture resize, patching and token layout |

## Every change made

1. **One file, plain torch.** The classes no longer derive from `PreTrainedModel`, read no
   `PreTrainedConfig` and use no transformers utilities. The configuration is read from the
   checkpoint's `config.json` as a dictionary.
2. **Attention** is `torch.nn.functional.scaled_dot_product_attention` with a boolean mask, scale
   1.0, and key and value heads repeated to the query heads, in place of upstream's attention
   interface dispatch. The bidirectional and sliding window masks are built in the forward with
   upstream's rule: every valid key, and within a sliding layer keys at most `sliding_window`
   positions away.
3. **Rotary tables** are computed per call from each layer type's head size and `rope_theta`,
   with upstream's default and axial layouts.
4. **The audio tower and its embedder are left out**, and their weights are not read.
5. **Pictures only.** The video and audio inputs, the processor class and the chat template are
   not ported. A picture is resized and patched by `picture_patches`, upstream's PIL path, and
   placed in the token sequence `<bos> <|image> <|image|>... <image|> <eos>` as the processor
   builds it.
6. **Mean pooling and normalisation**, which sentence-transformers applies upstream, are part of
   `EmbeddingGemma2.embed_ids` and `EmbeddingGemma2.embed_pictures`.
7. **Weights** are loaded from `model.safetensors` by upstream's tensor names into the model,
   which is then cast to the precision asked for.
