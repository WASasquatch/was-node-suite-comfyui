"""Write a batch of images as one animated WebP, frames encoded in parallel."""

from __future__ import annotations

from comfy_api.latest import io, ui

from ...modules import log
from ...modules.interface import file_report
from ...modules.io import rooted
from ...modules.media import webp_anim
from ...modules.util import sandbox
from .image_save import PLACEHOLDER_PREFIX, subfolder_of

logger = log.get_logger("nodes.io")

#: Effort names offered on the method widget, as the encoder's 0 to 6 scale.
METHODS = {"default": 4, "fastest": 0, "slowest": 6}


class FastSaveAnimatedWEBP(io.ComfyNode):
    """Save Animated WEBP with every frame encoded on its own thread."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASFastSaveAnimatedWEBP",
            display_name="Fast Save Animated WEBP",
            search_aliases=[
                "WASFastSaveAnimatedWEBP",
                "Fast Save Animated WEBP",
                "Save Animated WEBP",
                "animated webp",
                "webp animation",
            ],
            category="WAS Suite/IO",
            description=(
                "Write a batch of images as one looping animated WebP, as core Save Animated "
                "WEBP does, with the frames encoded side by side on every core. Lossless "
                "frames keep their exact pixels. Each frame is stored whole, so a lossless "
                "file is a little larger than core's."
            ),
            inputs=[
                io.Image.Input("images", tooltip="The frames of the animation, in order."),
                io.Combo.Input(
                    "root",
                    options=rooted.options(),
                    tooltip=(
                        "Which folder the file lands in: ComfyUI's own 'output' or 'temp', "
                        "or any folder added under paths.allow_write in config.yaml."
                    ),
                ),
                io.String.Input(
                    "filename_prefix",
                    default="ComfyUI",
                    tooltip="The file's name before its number, such as `renders/walk` for a subfolder.",
                ),
                io.Float.Input(
                    "fps", default=6.0, min=0.01, max=1000.0, step=0.01,
                    tooltip="Frames per second. `24` plays as film, `6` as a slow loop.",
                ),
                io.Boolean.Input(
                    "lossless", default=True,
                    tooltip="`true` keeps every pixel exactly; `false` stores smaller, softer frames.",
                ),
                io.Int.Input(
                    "quality", default=80, min=0, max=100,
                    tooltip="`0` to `100`. Picture quality when lossy, compression effort when lossless.",
                ),
                io.Combo.Input(
                    "method", options=list(METHODS),
                    tooltip="`fastest` writes quickly and larger, `slowest` spends longer for a smaller file.",
                ),
            ],
            outputs=[
                io.Image.Output(display_name="images", tooltip="The frames, passed through unchanged."),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, images, root=rooted.DEFAULT, filename_prefix="ComfyUI", fps=6.0,
                lossless=True, quality=80, method="default") -> io.NodeOutput:
        """Encode the frames and write the file.

        Args:
            images: ``(frames, height, width, channels)`` images.
            root: Which permitted folder the file lands in.
            filename_prefix: Name part before the counter, a subfolder allowed.
            fps: Frames per second.
            lossless: Whether frames are stored losslessly.
            quality: Lossy quality or lossless effort, 0 to 100.
            method: A key of :data:`METHODS`.

        Returns:
            The frames, with the written file as the node's preview.

        Raises:
            PathNotAllowed: The chosen root is not a folder this pack may write to.
        """
        import folder_paths

        wanted = (filename_prefix or "").replace("\\", "/")
        below, _, leaf = wanted.rpartition("/")
        base = str(rooted.destination(root, below))
        folder, name, counter, _, _ = sandbox.save_image_path(
            leaf or PLACEHOLDER_PREFIX, base, images[0].shape[1], images[0].shape[0]
        )
        destination = sandbox.resolve_write(folder)
        file = f"{name}_{counter:05}_.webp"
        target = sandbox.resolve_write_file(destination, file)

        frames = [ui.ImageSaveHelper._convert_tensor_to_pil(image) for image in images]
        exif = ui.ImageSaveHelper._create_webp_metadata(frames[0], cls)
        data = webp_anim.encode(
            frames,
            int(1000.0 / fps),
            lossless=lossless,
            quality=quality,
            method=METHODS.get(method, METHODS["default"]),
            exif=exif.tobytes() if hasattr(exif, "tobytes") else bytes(exif or b""),
        )
        target.write_bytes(data)
        logger.info("animated WebP saved to: %s", target)

        file_report.publish(
            [str(target)],
            kind="webp",
            folder=str(destination),
            facts={"frames": len(frames), "fps": fps, "lossless": "yes" if lossless else "no"},
        )
        subfolder = subfolder_of(str(destination), folder_paths.get_output_directory())
        if subfolder is None:
            return io.NodeOutput(images)
        saved = ui.SavedResult(file, subfolder, io.FolderType.output)
        return io.NodeOutput(images, ui=ui.SavedImages([saved], is_animated=len(frames) > 1))
