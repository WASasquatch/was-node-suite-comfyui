---
name: ComfyUI Node Authoring
description: How a ComfyUI custom node is built on the V3 API, for a standalone package that needs only ComfyUI. Schema, socket types, data formats, lists, lazy inputs, files, untrusted input, tooltips and a review checklist, to add to the Code Planner, Writer, Reviewer and Reviser prompts
model_family: [General]
tags: [code, comfyui, custom nodes, python, skill]
---

# ComfyUI node authoring

These rules apply whenever the code being planned, written, reviewed or revised is a ComfyUI custom node. The package stands alone: it imports ComfyUI, the Python standard library, torch, numpy and Pillow, its own files, and any other package it lists in its requirements. It never imports another custom node pack.

## The package

```
my_nodes/
  __init__.py        the entry point, below
  nodes.py           the node classes; a nodes/ folder for many
  requirements.txt   packages beyond torch, numpy and Pillow, one per line; omit when none
  web/               frontend scripts, only when a node draws its own interface
```

`__init__.py` registers the nodes through the V3 entry point and defines nothing else:

```python
from comfy_api.latest import ComfyExtension, io

from .nodes import ImageVignette


class MyNodesExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [ImageVignette]


async def comfy_entrypoint() -> MyNodesExtension:
    return MyNodesExtension()
```

Never also define `NODE_CLASS_MAPPINGS`: ComfyUI reads it first and then ignores `comfy_entrypoint`. Add `WEB_DIRECTORY = "./web"` beside the entry point only when a `web/` folder exists. Importing a module defines classes and nothing else: no downloads, no installs, no file writes, no `sys.path` changes.

## A node, whole

```python
"""Darken images toward their corners."""

from __future__ import annotations

import torch
from comfy_api.latest import io


class ImageVignette(io.ComfyNode):
    """Radial darkening toward the corners of every image in a batch."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MyNodesImageVignette",
            display_name="Image Vignette",
            category="My Nodes/Image",
            description=(
                "Darken images toward their corners, as a lens does, to draw the eye to the "
                "centre. Every image in a batch is darkened the same way."
            ),
            search_aliases=["vignette", "darken edges", "lens falloff"],
            inputs=[
                io.Image.Input("image", tooltip="The images to darken, a batch of any size."),
                io.Float.Input(
                    "strength", default=0.5, min=0.0, max=1.0, step=0.01,
                    tooltip="0 = no change; 0.5 = corners at half brightness; 1 = black corners.",
                ),
                io.Float.Input(
                    "radius", default=0.6, min=0.0, max=0.99, step=0.01,
                    tooltip="Where the darkening starts, as a share of the way to the corner: 0 = from the centre; 0.9 = only the corners.",
                ),
            ],
            outputs=[
                io.Image.Output(display_name="image", tooltip="The darkened images, same size and count."),
            ],
        )

    @classmethod
    def execute(cls, image, strength, radius) -> io.NodeOutput:
        """Scale each pixel by its distance from the centre.

        Args:
            image: Images shaped (batch, height, width, channels), values 0 to 1.
            strength: How dark the corners become.
            radius: Distance from the centre where darkening begins.

        Returns:
            The darkened images.
        """
        _, height, width, _ = image.shape
        ys = torch.linspace(-1.0, 1.0, height, device=image.device)
        xs = torch.linspace(-1.0, 1.0, width, device=image.device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        distance = torch.sqrt(grid_x ** 2 + grid_y ** 2) / (2 ** 0.5)
        falloff = ((distance - radius) / (1.0 - radius)).clamp(0.0, 1.0)
        gain = (1.0 - strength * falloff).to(image.dtype)
        return io.NodeOutput((image * gain[None, :, :, None]).clamp(0.0, 1.0))
```

What generalises:

- `define_schema` and `execute` are classmethods. `execute`'s parameter names are the input ids exactly; an optional input's parameter defaults to `None`. `execute` may be `async def`.
- `execute` returns `io.NodeOutput` with one positional value per output, in schema order.
- `define_schema` runs on every request for the node list, so it reads no files, scans no folders and calls no network. A menu built from a model folder comes from `folder_paths.get_filename_list(...)`, which caches.
- `cls` inside `execute` is a fresh copy for that run. Keep no per-run state on the class.

## The schema

`io.Schema` takes `node_id`, `display_name`, `category`, `description`, `inputs`, `outputs`, and where needed `search_aliases`, `hidden`, `is_output_node`, `is_input_list`, `not_idempotent`, `enable_expand`.

- `node_id` is written into every saved workflow. Give it a prefix of the package's own, so it never collides with another pack, and never rename it once published.
- `display_name` is the plain title; `category` is the Add Node menu path, the package's own name first.
- `description` is one to three sentences, at most 60 words, saying what the node is for and what it hands on.

Inputs:

| Socket | Widget keywords |
|---|---|
| `io.Int.Input(id, ...)` | `default`, `min`, `max`, `step`, `control_after_generate` |
| `io.Float.Input(id, ...)` | `default`, `min`, `max`, `step`, `round` |
| `io.String.Input(id, ...)` | `default`, `multiline`, `placeholder` |
| `io.Boolean.Input(id, ...)` | `default`, `label_on`, `label_off` |
| `io.Combo.Input(id, options=[...], ...)` | `default` |
| `io.Image`, `io.Mask`, `io.Latent`, `io.Model`, `io.Clip`, `io.Vae`, `io.Conditioning`, `io.Audio`, `io.Video` | none: sockets only |

Every input takes `tooltip`, `optional`, `display_name` and `lazy`. A widget input takes `force_input=True` to be a socket with no widget. `io.AnyType.Input(id)` takes any type. For an output that follows the type wired in, make one `t = io.MatchType.Template("value")` inside `define_schema`, then `io.MatchType.Input("value", template=t)` and `io.MatchType.Output(t, display_name="value")`. A type of the package's own is declared once at module scope, `MyData = io.Custom("MY_NODES_DATA")`, and used as `MyData.Input(...)` and `MyData.Output(...)`; it has no widget.

Outputs take `display_name`, `tooltip` and `is_output_list`. Name every output.

## Data on the wires

| Type | Value |
|---|---|
| IMAGE | float tensor (batch, height, width, channels), values 0 to 1, usually 3 channels |
| MASK | float tensor (batch, height, width), values 0 to 1 |
| LATENT | dict with `"samples"`: tensor (batch, channels, height, width), or (batch, channels, frames, height, width) for video; keep its other keys |
| CONDITIONING | list of `[tensor, dict]` pairs; copy before changing a dict |
| AUDIO | dict with `"waveform"` (batch, channels, samples) and `"sample_rate"` |
| MODEL, CLIP, VAE | ComfyUI's own objects; change a MODEL only on `model.clone()` |

Image and mask nodes take whole batches: loop over or broadcast across the first dimension and return every item. Indexing `[0]` drops the rest of a batch. Two inputs whose sizes or batch counts differ are matched explicitly, by resizing or repeating, with the rule stated in a tooltip. Keep the dtype and device the input arrived on, and move a second tensor onto the first before combining them: `mask.to(image)`. Get a device from `comfy.model_management.get_torch_device()` or `intermediate_device()`, never assume CUDA. Convert to PIL or numpy only where a library needs it, and back to a float tensor in 0 to 1.

## How a run treats a node

- Only nodes that lead to an output node run. A node that saves, shows or publishes something and feeds nothing sets `is_output_node=True`.
- `ui=` on `io.NodeOutput` sends something to the canvas: `ui.PreviewImage(images, cls=cls)`, `ui.PreviewMask(mask, cls=cls)`, `ui.PreviewText(text)`. Import `ui` from `comfy_api.latest`.
- A node reruns only when its inputs change. A node reading a file that can change on disk defines `fingerprint_inputs(cls, **inputs)` returning the file's modified time; one whose result differs on every call sets `not_idempotent=True`.
- `validate_inputs(cls, black_point, white_point)` returns `True`, or a string naming the fault and the fix. List only the inputs it checks: listing one turns off ComfyUI's own range and menu check for it, and `**inputs` turns it off for every input. It runs before anything executes, so a linked input arrives as `None` and an image, mask or other tensor is never there: skip a `None`, and check shapes and sizes in `execute`, raising a `ValueError` that names both sizes.
- `lazy=True` inputs are evaluated only when `check_lazy_status(cls, **inputs)` asks for them by name; an input not yet evaluated arrives as `None`. Use it to skip an unused branch.
- `is_input_list=True` hands every input over as a list, settings included, so read `settings[0]`. `is_output_list=True` on an output runs each downstream node once per item. A list output is never empty: raise a clear error instead.
- To stop the nodes after this one without failing the run, return `ExecutionBlocker(None)` from `comfy_execution.graph_utils` in that output's place.
- Hidden values come from `hidden=[io.Hidden.unique_id, io.Hidden.prompt, io.Hidden.extra_pnginfo]` and are read as `cls.hidden.unique_id`.
- Long work reports progress with `comfy.utils.ProgressBar(total)` and `.update(1)`, and calls `comfy.model_management.throw_exception_if_processing_interrupted()` between steps so Cancel works.
- Free large intermediate tensors before returning; load models through ComfyUI's model management so they unload when memory runs short.

## Files and folders

- Read and write only inside ComfyUI's folders: `folder_paths.get_input_directory()`, `get_output_directory()`, `get_temp_directory()`, and the model folders through `folder_paths.get_full_path_or_raise("loras", name)`.
- A file the user picks is a combo over a folder listing, never a typed absolute path. A typed name is joined to its folder and then checked: `os.path.realpath` of the result must lie inside `os.path.realpath` of the folder (`os.path.commonpath`), or the node refuses it. Refuse names holding `..`, a drive letter or a path separator where one name is meant.
- Save with `folder_paths.get_save_image_path(prefix, output_dir, width, height)` and its counter, so a save never replaces an existing file unless an overwrite input is on. Delete only files the node itself wrote, by name.
- A file written for people to share carries no absolute path of the machine that made it.

## Untrusted input

A workflow is a document somebody else wrote. Every value may arrive through a link or the API rather than the widget.

- Give every size, count, radius, factor and duration a real `min` and `max`; a pixel size or coordinate stops at `nodes.MAX_RESOLUTION`. Before allocating, work out the size of what the inputs ask for and refuse past a ceiling, naming the input to lower.
- Never `eval` or `exec` workflow text. Never call `str.format` on a template the workflow supplies; substitute named fields with a fixed pattern. A regular expression from a workflow can hang the server: prefer plain substring matching.
- Load weights with `safetensors`, or `torch.load(path, weights_only=True)`. Never unpickle a file a user can supply.
- Read sizes and counts in a file header against the file's real length before allocating from them.
- Reach the network only when the node exists to, from a fixed host, and never send tokens, paths or user names.

## Building on ComfyUI

- Call what ComfyUI already does: its samplers, loaders and model types. Copying core code forks it.
- Change a model only on `model.clone()`, through `set_model_*` and `model_options`. Never assign to an attribute of a ComfyUI module or class, never wrap a core function globally, and never edit a core registry or a file in the ComfyUI install.
- A package with heavy or optional dependencies imports them inside `execute`, and a missing one raises an error naming the package and the `pip install` command. Nothing is installed on the user's behalf.

## Text the node shows

- Every input and output has a tooltip of at most 45 words. Show the values, shortest first: `0 = off; 1 = full; 2 = doubled`. Give a short example of any text input's shape. Say what an input is for when the mechanism alone leaves the choice open.
- An error names the cause and the fix: `Image Vignette: radius 1.2 is past the corner. Use 0 to 0.99.` Prefix messages with the node's display name.
- Log names, counts and sizes, never prompt text, file contents or tokens.
- A docstring is a reference card: one line saying what the thing is or does, then `Args:`, `Returns:`, `Raises:`. An inline comment says what the next line does, in one line, where the code does not. No history, no reasons, no warnings in comments.

## Changing a published node

A saved workflow stores widget values by position and links by input id. Appending an input or an output, adding combo options, widening a range and rewording tooltips are free. Inserting, removing, reordering or retyping an input, renaming an input id, narrowing a range and changing a default break saved workflows silently. Append rather than insert, and keep every id.

## Review checklist

- The package registers through `comfy_entrypoint` and imports nothing from another pack.
- Every input id matches an `execute` parameter; optional ones default to `None`; outputs match `io.NodeOutput` in count and order.
- Batches are handled whole; shapes, dtypes, value ranges and devices match the table above.
- `validate_inputs` reads only widget values, survives `None` for a linked one, and never touches a tensor; a size mismatch between tensors raises a named error in `execute`.
- A node with no consumer that should run is an output node; a list output is never empty.
- Every number has a range, allocations are bounded, files stay in ComfyUI's folders, and nothing evaluates workflow text.
- `define_schema` touches no disk; module import has no side effects; heavy imports sit inside `execute`.
- Every input and output has a tooltip, the node has a description, errors name cause and fix.
