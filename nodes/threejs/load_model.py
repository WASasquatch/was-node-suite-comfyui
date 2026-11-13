"""A 3D model file placed in a Three.js scene."""

from __future__ import annotations

import os

import folder_paths
from comfy_api.latest import io

from ...modules.compat.types import (
    FILE_3D,
    FILE_3D_FBX,
    FILE_3D_GLB,
    FILE_3D_GLTF,
    FILE_3D_OBJ,
    FILE_3D_PLY,
    FILE_3D_STL,
    THREE_OBJECT,
)
from ...modules.interface import run_result
from ...modules.log import get_logger
from ...modules.threejs import model_report, models
from ...modules.threejs.spec import create_spec
from ...modules.util import file_listing, sandbox
from ...modules.io import picker

REQUIRES = "threejs"

logger = get_logger("nodes.threejs")

#: What the menu says when it found nothing.
NO_MODELS = "no model files found"

#: Where the model's origin is put, against the point of the model that lands there.
ORIGINS = ("centre", "base", "as saved")


def options() -> list[str]:
    """The menu's entries, or ``[NO_MODELS]`` when there are none."""
    return list(
        file_listing.labels(models.SUFFIXES, file_listing.ROOTS)
    ) or [NO_MODELS]


class ThreeLoadModel(io.ComfyNode):
    """Put a model file into the scene."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASThreeLoadModel",
            display_name="Three Load Model",
            search_aliases=[
                "WASThreeLoadModel",
                "Three Load Model",
                "gltf",
                "glb",
                "obj",
                "load 3d model",
                "tripo",
                "file 3d",
            ],
            category="WAS Suite/Three",
            description=(
                "Put a model into the scene as an object, so a mesh made elsewhere, such as a "
                "Tripo or Hunyuan3D generation, can be lit, animated and rendered here. Reads "
                ".glb, .gltf, .dae, .fbx, .obj, .3mf, .stl and .ply, compressed glTF included, "
                "and reports what the file holds."
            ),
            inputs=[
                picker.path_input(io.Combo.Input(
                    "file",
                    options=options(),
                    tooltip=(
                        "Which model to place. The menu lists the newest `.glb`, `.gltf`, `.dae`, "
                        "`.fbx`, `.obj`, `.3mf`, `.stl` and `.ply` files in ComfyUI's "
                        "input, output and temp folders and in any folder under "
                        "paths.allow_read."
                    ),
                )),
                io.String.Input(
                    "path",
                    default="",
                    multiline=False,
                    optional=True,
                    tooltip=(
                        "A model somewhere else, as a full path. Wire ComfyUI-Tripo's model_file "
                        "or Load 3D's mesh_path here. Filled in, it is used instead of the menu."
                    ),
                ),
                io.Float.Input(
                    "scale",
                    default=1.0,
                    min=0.0001,
                    max=10000.0,
                    step=0.01,
                    tooltip=(
                        "Multiplies the model's size, after fit_size where that is set. 1.0 "
                        "leaves it, 0.01 suits a model authored in centimetres."
                    ),
                ),
                io.Combo.Input(
                    "origin",
                    options=list(ORIGINS),
                    default="centre",
                    tooltip=(
                        "Which point of the model lands on the scene's origin. centre = its "
                        "middle; base = the middle of its underside, so it stands on a floor at "
                        "height 0; as saved = the coordinates in the file."
                    ),
                ),
                io.Boolean.Input(
                    "cast_shadow",
                    default=True,
                    tooltip="`true` lets every mesh in the model throw and receive shadows, `false` neither.",
                ),
                io.Float.Input(
                    "fit_size",
                    default=0.0,
                    min=0.0,
                    max=10000.0,
                    step=0.1,
                    optional=True,
                    tooltip=(
                        "0.0 = the model's own size; 2.0 = resized so its longest side is 2 units. "
                        "Generated models arrive at any size, so this frames each one the same."
                    ),
                ),
                io.MultiType.Input(
                    "model_3d",
                    types=[FILE_3D, FILE_3D_GLB, FILE_3D_GLTF, FILE_3D_FBX, FILE_3D_OBJ, FILE_3D_STL, FILE_3D_PLY],
                    optional=True,
                    tooltip=(
                        "A model from a 3D generator or Load 3D, such as the GLB from Tripo, "
                        "Hunyuan3D, Meshy or Rodin. Wired, it is used instead of path and the menu."
                    ),
                ),
                io.String.Input(
                    "keep_as",
                    default="",
                    multiline=False,
                    optional=True,
                    tooltip=(
                        "A name such as `tripo_assets/tent.glb` keeps a copy of the model that "
                        "came in on path or model_3d in ComfyUI's input folder, replacing an "
                        "older copy, so the menu still finds it with the generator bypassed. "
                        "Empty keeps nothing."
                    ),
                ),
                io.Float.Input(
                    "metalness",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    optional=True,
                    tooltip=(
                        "Multiplies how metallic every surface is. 1.0 keeps the file's metal; "
                        "0.3 turns hair or leather a generator marked as metal back into a "
                        "plain surface while armour keeps a sheen; 0.0 makes nothing metal."
                    ),
                ),
                io.Float.Input(
                    "min_roughness",
                    default=0.0,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    optional=True,
                    tooltip=(
                        "The least roughness any surface keeps. 0.0 keeps the file's; 0.45 "
                        "takes the mirror shine off surfaces a generator made glass smooth; "
                        "1.0 makes everything matt."
                    ),
                ),
            ],
            outputs=[
                THREE_OBJECT.Output(
                    display_name="object",
                    tooltip="The loaded model, for Three Group or the root socket on Three Scene.",
                ),
            ],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def execute(
        cls, file, scale, origin, cast_shadow, path="", fit_size=0.0, model_3d=None,
        keep_as="", metalness=1.0, min_roughness=0.0,
    ) -> io.NodeOutput:
        """Hold the model for the browser and describe where it goes.

        Raises:
            ValueError: No input names a readable model, its format has no loader, it needs
                an extension the browser cannot read, or keep_as cannot hold it.
            PathNotAllowed: The path resolved outside every permitted read root, or keep_as
                names somewhere outside ComfyUI's input folder.
        """
        carried, name = cls.model(file, path, model_3d)
        wired = model_3d is not None or bool(str(path or "").strip())
        if wired and str(keep_as or "").strip():
            cls.keep(carried, str(keep_as).strip())
        try:
            found = model_report.describe(carried.body, carried.format)
        except (ValueError, UnicodeDecodeError, OverflowError, TypeError, KeyError) as error:
            logger.debug("%s could not be described: %s", name, error)
            found = {"format": carried.format, "bytes": len(carried.body)}
        if found.get("unread"):
            held = ", ".join(model_report.UNREAD_EXTENSIONS[key] for key in found["unread"])
            raise ValueError(
                f"{name} needs {held} ({', '.join(found['unread'])}), which Three Load Model "
                f"does not read. Save it again with PNG, JPEG or WebP textures."
            )
        cls.report(name, found, str(cls.hidden.unique_id or "") or None)
        return io.NodeOutput(
            create_spec(
                "object",
                "ModelFile",
                params={
                    "url": carried.url,
                    "format": carried.format,
                    "scale": float(scale),
                    "fitSize": max(float(fit_size or 0.0), 0.0),
                    "origin": origin if origin in ORIGINS else "centre",
                    "castShadow": bool(cast_shadow),
                    "name": name,
                    "resources": carried.resources,
                    "decoders": list(found.get("decoders") or []),
                    "metalness": min(max(float(metalness), 0.0), 1.0),
                    "minRoughness": min(max(float(min_roughness), 0.0), 1.0),
                },
            )
        )

    @classmethod
    def model(cls, file, path, model_3d) -> tuple[models.Carried, str]:
        """The model the inputs name, held for the browser, and what to call it.

        Args:
            file: The menu's entry.
            path: A wired or typed path, used before the menu.
            model_3d: Core's 3D file, used before either.

        Returns:
            ``(carried, name)``, the name being the file's stem or ``model_3d``.

        Raises:
            ValueError: Nothing names a model, or its format has no loader.
            PathNotAllowed: A path resolved outside every permitted read root.
        """
        if model_3d is not None:
            fmt = str(getattr(model_3d, "format", "") or "").lower().lstrip(".")
            if getattr(model_3d, "is_disk_backed", False):
                resolved = sandbox.resolve_read(str(model_3d.get_source()))
                return models.carried(resolved), resolved.stem
            body = model_3d.get_bytes()
            return models.carried_bytes(body, fmt, f"the wired {fmt or 'model'}"), "model_3d"

        chosen = str(path or "").strip()
        if not chosen:
            if not file or file == NO_MODELS:
                raise ValueError(
                    "Three Load Model has no model to place. Put a .glb, .gltf, .dae, .fbx, "
                    ".obj, .3mf, .stl or .ply in ComfyUI's input folder and pick it, wire a "
                    "3D generator's model into model_3d, or wire a path into path."
                )
            chosen = file_listing.resolve(file, models.SUFFIXES, file_listing.ROOTS, paths=True) or file
        resolved = sandbox.resolve_read(chosen)
        return models.carried(resolved), resolved.stem

    @staticmethod
    def keep(carried: models.Carried, name: str) -> str:
        """Write the model into ComfyUI's input folder under ``name``.

        Args:
            carried: The model as held for the browser.
            name: The file name inside the input folder; sub-folders are allowed.

        Returns:
            The path written.

        Raises:
            ValueError: The name's suffix is not the model's format, or the model reads
                textures from files beside it.
            PathNotAllowed: The name leaves the input folder.
        """
        suffix = os.path.splitext(name)[1].lower().lstrip(".")
        if suffix != str(carried.format).lower().lstrip("."):
            raise ValueError(
                f"Three Load Model's keep_as `{name}` must end in .{carried.format}, the format "
                f"of the model it keeps."
            )
        if carried.resources:
            raise ValueError(
                "Three Load Model's keep_as keeps a model held in one file. This one reads "
                "textures from files beside it; save it as .glb to keep it."
            )
        target = sandbox.resolve_save_file(folder_paths.get_input_directory(), name)
        os.makedirs(target.parent, exist_ok=True)
        scratch = target.with_name(f".{target.name}.partial")
        try:
            with open(scratch, "wb") as handle:
                handle.write(carried.body)
            os.replace(scratch, target)
        except OSError:
            if scratch.exists():
                scratch.unlink()
            raise
        logger.info("Three Load Model kept the model as %s", target)
        return str(target)

    @staticmethod
    def report(name: str, found: dict, node_id: str | None) -> None:
        """Publish what the model holds for the node's own panel.

        Args:
            name: What the model is called.
            found: :func:`modules.threejs.model_report.describe`'s answer.
            node_id: The node's graph id.
        """
        try:
            megabytes = found.get("bytes", 0) / 1048576.0
            facts = {"file": f"{found.get('format', '?')}, {megabytes:.1f} MB"}
            counts = {}
            if "triangles" in found:
                counts = {
                    "triangles": found["triangles"],
                    "meshes": found["meshes"],
                    "materials": found["materials"],
                    "textures": found["textures"],
                    "clips": found["clip_count"],
                    "bones": found["joints"],
                }
                bounds = found.get("bounds")
                if bounds:
                    size = [high - low for low, high in zip(*bounds)]
                    facts["size"] = " x ".join(f"{value:.3g}" for value in size) + " units"
                sizes = found.get("texture_sizes") or []
                if sizes:
                    largest = max(sizes, key=lambda pair: pair[0] * pair[1])
                    facts["largest texture"] = f"{largest[0]}x{largest[1]}"
                if found.get("clips"):
                    facts["clips"] = ", ".join(
                        f"{clip} ({seconds:.2f} s)" for clip, seconds in found["clips"]
                    )
                if found.get("decoders"):
                    facts["compressed"] = ", ".join(found["decoders"])
                if found.get("extensions"):
                    facts["extensions"] = ", ".join(found["extensions"])
                if found.get("generator"):
                    facts["made by"] = found["generator"]
                summary = f"{name}: {found['triangles']:,} triangles"
                if found["clip_count"]:
                    summary += f", {found['clip_count']} clip(s)"
            else:
                summary = f"{name}: {megabytes:.1f} MB {found.get('format', '')}"
            run_result.publish(summary=summary, counts=counts, facts=facts, node_id=node_id)
        except Exception as error:
            logger.debug("Three Load Model published no report (%s)", error)
