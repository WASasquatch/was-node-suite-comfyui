"""Writing a sampled MiniMax H3 repair window back into the clip it was cut from."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.latent import h3_repair


class H3RepairSplice(io.ComfyNode):
    """Write the rows and sound a repair window redrew back into its clip."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASH3RepairSplice",
            display_name="H3 Repair Splice",
            search_aliases=[
                "WASH3RepairSplice",
                "H3 Repair Splice",
                "minimax h3 repair",
                "write back repair",
                "fix frames",
            ],
            category="WAS Suite/Latent/Video",
            description=(
                "Write a sampled H3 Repair Window back into the clip it was cut from. Only the "
                "frames and sound the window redrew are replaced; every other latent row and "
                "audio step of the clip is carried through unchanged, along with its record of "
                "segments and scene cuts. The report names the frames and seconds that changed. "
                "Decode the result as any finished clip, or repair another stretch of it."
            ),
            inputs=[
                io.Latent.Input(
                    "latent",
                    tooltip="The clip the window was cut from, the same latent wired into H3 Repair Window.",
                ),
                io.Latent.Input(
                    "sampled",
                    tooltip="What the sampler returned for the window H3 Repair Window opened.",
                ),
            ],
            outputs=[
                io.Latent.Output(
                    display_name="latent",
                    tooltip="The repaired clip, for a decode or another repair.",
                ),
                io.String.Output(
                    display_name="report",
                    tooltip=(
                        "Which frames and seconds changed and at what strength, as `picture "
                        "frames 68-118 (2.83-4.96 s, tokens 20-34) at 1.00, sound kept`."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, latent, sampled) -> io.NodeOutput:
        """Write the redrawn rows and steps back and report them.

        Raises:
            ValueError: Either latent is not an H3 joint latent, the window carries no repair
                record, or it was cut from another clip.
        """
        repaired = h3_repair.splice(latent, sampled)
        return io.NodeOutput(repaired, h3_repair.describe(sampled[h3_repair.REPAIR_KEY]))
