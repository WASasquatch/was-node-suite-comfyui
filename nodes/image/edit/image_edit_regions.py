"""Find everything in a picture with SAM 3 and pick what to edit on the node."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules import log
from ....modules.compat.types import EDIT_JOBS
from ....modules.image import edit_regions
from ....modules.interface import edit_regions as edit_regions_panel
from ....modules.interface import preview, run_result

NODE = "Image Edit Regions"

logger = log.get_logger("nodes.image.edit")


class ImageEditRegions(io.ComfyNode):
    """Find the things in a picture and hold the edits picked on them."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASImageEditRegions",
            display_name="Image Edit Regions",
            search_aliases=[
                "WASImageEditRegions",
                "Image Edit Regions",
                "segment everything",
                "click to edit",
                "sam3",
                "regional edit",
                "edit jobs",
            ],
            category="WAS Suite/Image/Edit",
            is_output_node=True,
            description=(
                "Find everything in a picture with SAM 3 and pick what to change on the node: click a "
                "thing, then edit, recolour, replace or remove it with a prompt. The picked edits go "
                "to Image Edit Crop as jobs, each with its own mask and prompt."
            ),
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip="The picture to edit. A batch is read from its first picture.",
                ),
                io.Model.Input(
                    "model",
                    tooltip="SAM 3, from Load Checkpoint with sam3.1_multiplex_fp16.safetensors.",
                ),
                io.Clip.Input(
                    "clip",
                    tooltip="SAM 3's text encoder, the CLIP from the same Load Checkpoint.",
                ),
                io.Combo.Input(
                    "edit_model", options=list(edit_regions.REFERENCES), default="MiniMax H3 Image",
                    tooltip=(
                        "How prompts name the picture: MiniMax H3 Image = <Picture 1>; other = "
                        "\"the picture\"."
                    ),
                ),
                io.String.Input(
                    "edits", default="",
                    tooltip=(
                        "The edits picked on the panel, as JSON such as `{\"jobs\": []}`. Empty = nothing "
                        "to edit yet: the run finds the things in the picture and skips the stages after it."
                    ),
                ),
                io.Float.Input(
                    "threshold", default=0.5, min=0.05, max=0.95, step=0.01, advanced=True,
                    tooltip="Least confidence a find needs: 0.3 = more things, some wrong; 0.6 = fewer, surer.",
                ),
                io.String.Input(
                    "names", default="", advanced=True,
                    tooltip=(
                        "More things to look for, separated by commas, added to the built-in list of "
                        "about 170, such as `lantern, parasol`."
                    ),
                ),
            ],
            outputs=[
                EDIT_JOBS.Output(
                    display_name="jobs",
                    tooltip="Every picked edit with its mask and prompt, for Image Edit Crop's jobs input.",
                ),
                io.Mask.Output(
                    display_name="masks",
                    tooltip="One mask per picked edit, at the picture's size.",
                ),
                io.Int.Output(
                    display_name="count",
                    tooltip="How many edits were picked.",
                ),
            ],
        )

    @classmethod
    def execute(cls, image, model, clip, edit_model, edits, threshold, names) -> io.NodeOutput:
        """Find the things in the picture and build the picked edits.

        Args:
            image: The picture, ``(batch, height, width, channels)``.
            model: SAM 3 as a MODEL.
            clip: SAM 3's text encoder.
            edit_model: A key of ``REFERENCES``.
            edits: The panel's JSON.
            threshold: Least confidence a find needs.
            names: Extra things to look for.

        Returns:
            The jobs, their masks and their count; the jobs and masks are skipped with no edits.
        """
        import torch
        from comfy_execution.graph_utils import ExecutionBlocker

        if edit_model not in edit_regions.REFERENCES:
            raise ValueError(f"{NODE} does not know the edit model `{str(edit_model)[:40]}`. Pick one from the menu.")
        picture = image[0]
        height, width = picture.shape[:2]
        looked_for = edit_regions.names_from(names)
        segments = edit_regions.find(model, clip, picture, looked_for, float(threshold), NODE)
        picked = edit_regions.read_edits(edits, NODE)
        jobs, notes = edit_regions.build_jobs(
            model, picture, segments, picked, edit_regions.REFERENCES[edit_model], NODE,
        )
        preview.publish(image[:1])
        layers, size = edit_regions.label_layers(segments, width, height)
        edit_regions_panel.publish(segments, layers, size, width, height, notes)
        facts = {f"note {number}": note for number, note in enumerate(notes, 1)}
        try:
            run_result.publish(
                status=run_result.WARNING if notes else run_result.OK,
                summary=f"{len(segments)} thing(s) found, {len(jobs.jobs)} edit(s) picked",
                counts={"found": len(segments), "edits": len(jobs.jobs)},
                facts=facts or None,
            )
        except Exception as error:
            logger.debug("%s published no report (%s)", NODE, error)
        if not jobs.jobs:
            logger.info("%s found %d thing(s); no edit is picked, so the stages after it are skipped", NODE, len(segments))
            return io.NodeOutput(ExecutionBlocker(None), ExecutionBlocker(None), 0)
        masks = torch.stack([job.mask for job in jobs.jobs])
        return io.NodeOutput(jobs, masks, len(jobs.jobs))
