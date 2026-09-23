"""Custom socket type declarations.

Links are validated on the ``io_type`` string, not class identity. A custom type carries no
``default`` and renders no widget.
"""

from __future__ import annotations

from comfy_api.latest import io

__all__ = [
    "BLIP_MODEL",
    "BUS",
    "CLIPSEG_MODEL",
    "CONDITIONING_SEQ",
    "CROP_DATA",
    "DICT",
    "DOC",
    "EMA_VFI_MODEL",
    "H3_PROMPTS",
    "IMAGE_BOUNDS",
    "LIST",
    "LUT",
    "MIDAS_MODEL",
    "NUMBER",
    "REMBG_MODEL",
    "SAM_MODEL",
    "SAM_PARAMETERS",
    "SEED",
    "THREE_APP",
    "THREE_CAMERA",
    "THREE_EFFECT",
    "THREE_ENVIRONMENT",
    "THREE_GEOMETRY",
    "THREE_MATERIAL",
    "THREE_MODULE",
    "THREE_OBJECT",
    "THREE_SCENE",
    "THREE_TRACK",
    "THREE_TEXTURE",
    "WAS_COLOUR_PROFILE",
    "WAS_LOOP",
    "WAS_LORA_MERGE_OPTIONS",
    "WAS_VIDEO_METADATA",
    "YUNET_MODEL",
    "ZIP",
]

# ``io.Custom(name)`` returns a new class on each call, so two declarations of one name are
# distinct classes that still connect. One declaration per name keeps a single spelling per
# ``io_type`` and one module to resolve a socket type against.

#: An opened zip archive travelling from `Zip Open` to whatever reads out of it: the entries
#: it holds, their kinds and their sizes, and where the file is. Carries a
#: `modules.archive.container.Archive`, which holds the index rather than the bytes, so an
#: archive on a wire costs no memory and no open file handle, and a node handed something
#: that is not an archive can say so instead of failing inside a zip reader.
ZIP = io.Custom("ZIP")

#: `BLIP Model Loader` -> `BLIP Analyze Image`.
BLIP_MODEL = io.Custom("BLIP_MODEL")

#: One wire carrying `(model, clip, vae, positive, negative)` in and out of `Bus Node`.
BUS = io.Custom("BUS")

#: `CLIPSeg Model Loader` -> `CLIPSeg Masking`, `CLIPSEG2`.
CLIPSEG_MODEL = io.Custom("CLIPSEG_MODEL")

#: `(frame index, [tensor, dict])` pairs, from `WASCLIPTextEncodeList` to `WASKSamplerSeq`. A
#: schedule of conditionings rather than one conditioning, so it is kept off the
#: CONDITIONING wire that carries the single form.
CONDITIONING_SEQ = io.Custom("CONDITIONING_SEQ")

#: `(size, (left, top, right, bottom))`, from the four crop nodes to the three paste nodes.
CROP_DATA = io.Custom("CROP_DATA")

#: `MiniMax H3 Conditioning` -> `H3 Extend Window`. One encoded conditioning and one frame
#: count per extension pass, in pass order.
H3_PROMPTS = io.Custom("WAS_H3_PROMPTS")

#: `EMA-VFI Model Loader` -> `EMA-VFI Frame Interpolation`. Carries a
#: `modules.model.Backend` holding the interpolation network and the checkpoint it was
#: built from.
EMA_VFI_MODEL = io.Custom("EMA_VFI_MODEL")

#: The dictionary passed between the dictionary nodes. Wire-compatible with core's
#: `io.Dict`, which emits the same io_type; declared here so all custom sockets resolve
#: from one module.
DICT = io.Custom("DICT")

#: A whole document travelling between the document nodes: the markup, the metadata and any
#: file embedded in it. Carries a `modules.document.container.Document`, which holds the
#: container bytes alongside the parts read out of them, rather than the bytes on their own,
#: so a node handed something that is not a document can say so instead of failing inside a
#: zip reader.
DOC = io.Custom("DOC")

#: A list of `(rmin, rmax, cmin, cmax)` rows, one per image in the batch.
IMAGE_BOUNDS = io.Custom("IMAGE_BOUNDS")

#: A plain python list: palettes out of `Image Color Palette`, lines out of `Text List`.
#: Wire-compatible with core's `io.Array`, which emits the same io_type and carries the same
#: plain list; declared here so all custom sockets resolve from one module.
LIST = io.Custom("ARRAY")

#: A colour lookup table, from `WASNSLoadLUT` and `WASNSCombineLUT` to `WASNSApplyLUT` and
#: `WASNSSaveLUT`. Carries a `modules.image.lut.LUT`, not a bare array.
LUT = io.Custom("LUT")

#: `MiDaS Model Loader` -> `MiDaS Depth Approximation`.
MIDAS_MODEL = io.Custom("MIDAS_MODEL")

#: A numeric value of runtime-determined type, int or float. 39 sockets across 25 nodes.
#:
#: Intentionally not type-restrictive: `Number Operation` emits the same value on its
#: NUMBER, INT and FLOAT outputs, and every NUMBER input accepts both int-valued and
#: float-valued sources. Replacing it with `io.Int`/`io.Float` would disconnect existing
#: mixed-type graphs.
NUMBER = io.Custom("NUMBER")

#: `Image Remove Background Model Loader` -> `Image Remove Background`. Carries the built
#: cutout network rather than a model name.
REMBG_MODEL = io.Custom("REMBG_MODEL")

#: `SAM Model Loader` -> `SAM Image Mask`.
SAM_MODEL = io.Custom("SAM_MODEL")

#: The points/labels dict `SAM Parameters` builds and `SAM Parameters Combine` merges.
SAM_PARAMETERS = io.Custom("SAM_PARAMETERS")

#: `Seed` / `Number to Seed` -> `KSampler (WAS)`. `{"seed": n}`, not a bare int.
SEED = io.Custom("SEED")

#: A Three.js resource or scene-graph entry, as a descriptor `modules.threejs.spec` built.
#: Every one of these carries a plain dict, never a Three.js object: the browser resolves a
#: descriptor into the real thing and the server never holds one.

#: `Three Texture From Image` / `Three Texture URL` -> a material's map sockets.
THREE_TEXTURE = io.Custom("THREE_TEXTURE")

#: A geometry node -> `Three Mesh`. Shape only, carrying no material and no transform.
THREE_GEOMETRY = io.Custom("THREE_GEOMETRY")

#: A material node -> `Three Mesh`. Surface only, reusable across meshes.
THREE_MATERIAL = io.Custom("THREE_MATERIAL")

#: A mesh, light, group or helper -> a parent, or `Three Scene`. One occurrence per place it
#: is wired, since a Three.js object holds one parent at a time.
THREE_OBJECT = io.Custom("THREE_OBJECT")

#: A camera node -> `Three App`. The view the scene is rendered from.
THREE_CAMERA = io.Custom("THREE_CAMERA")

#: `Three Environment` -> `Three Scene`. The light every physical material reflects, and
#: optionally what is drawn behind the scene.
THREE_ENVIRONMENT = io.Custom("THREE_ENVIRONMENT")

#: An effect node -> the next effect, or `Three App`. A chain of passes the finished frame is
#: put through before it is shown.
THREE_EFFECT = io.Custom("THREE_EFFECT")

#: `Three Track` -> a camera. How one object follows or aims at another, resolved against the
#: object already in the scene rather than building a second copy of it.
THREE_TRACK = io.Custom("THREE_TRACK")

#: `Three Scene` -> `Three App`. The root of the scene graph, with its background and fog.
THREE_SCENE = io.Custom("THREE_SCENE")

#: `Three Script Module` -> the typed import nodes. Named exports from one body of
#: JavaScript, evaluated once per viewer load.
THREE_MODULE = io.Custom("THREE_MODULE")

#: `Three App` -> `Three Viewer`. A scene, a camera and the renderer settings together.
THREE_APP = io.Custom("THREE_APP")

#: `Image Load` -> `Image Save` / `Image Preview`. Carries a
#: `modules.image.colour_profile.Carried`, the profile a file was tagged with and the name it
#: gives itself. The pixels on the IMAGE wire beside it are always sRGB, so this says what the
#: file was, not what the numbers are.
WAS_COLOUR_PROFILE = io.Custom("WAS_COLOUR_PROFILE")

#: The `iterator` socket: `For Loop Open`/`While Loop Open` -> their matching `Close`, wired
#: straight across and never through the loop body. Holds a small dict identifying the Open node,
#: where the loop is up to, and what it has collected; nothing a body node ever needs to read.
WAS_LOOP = io.Custom("WAS_LOOP")

#: `WASNSPowerLoraMergerOptions` -> `WASNSPowerLoraMerger`. The advanced merge settings, as a
#: plain dictionary. Kept off the DICT wire so the merger's options socket accepts only a
#: set of merge settings and says so in the Add Node menu's link filter.
WAS_LORA_MERGE_OPTIONS = io.Custom("WAS_LORA_MERGE_OPTIONS")

#: `Load Video` / `Load Video (Upload)` -> `Video Metadata`. What one read measured, as a plain
#: dictionary: the rate, the frame count, the size and the duration of both the answer and the
#: file behind it. Kept off the DICT wire so the reader's socket accepts only a video's figures.
WAS_VIDEO_METADATA = io.Custom("WAS_VIDEO_METADATA")

#: `YuNet Model Loader` -> `Image Crop Face (YuNet)`. Carries a
#: `modules.model.yunet.Detector`, which holds the torch network and the input size
#: it was built for, rather than a bare session.
YUNET_MODEL = io.Custom("YUNET_MODEL")
