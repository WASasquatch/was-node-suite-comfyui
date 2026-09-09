"""The values inside a latent, as numbers a graph can branch on."""

from __future__ import annotations

import torch
from comfy_api.latest import io, ui

from ...modules.compat import limits
from ...modules.compat.sockets import require_input
from ...modules.compat.types import NUMBER
from ...modules.logic.switch_index import OUT_OF_RANGE

#: The node's display name, spelled as its title bar spells it, for every message it raises.
NODE = "Latent Statistics"

#: Measure the whole batch together rather than one latent of it.


class LatentStatistics(io.ComfyNode):
    """Measure a latent's values and report whether any of them is nan or inf."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLatentStatistics",
            display_name="Latent Statistics",
            search_aliases=[
                "WASLatentStatistics",
                "Latent Statistics",
                "latent mean",
                "latent std",
                "nan check",
                "black image",
                "degenerate latent",
                "measure latent",
            ],
            category="WAS Suite/Latent",
            description=(
                "Measure the values inside a latent rather than its size: the mean, the "
                "spread, the range, and whether anything in it is nan or inf. A sampler "
                "that diverges answers a latent of nan or of huge values, which decodes to "
                "a black, grey or garbled picture with nothing saying why. Wire is_finite "
                "into a gate to catch that before the VAE decode. Measure the whole batch "
                "at once, or one latent of it by index."
            ),
            inputs=[
                io.Latent.Input(
                    "samples",
                    tooltip=(
                        "The latent to measure, from a sampler, a VAE Encode or an Empty "
                        "Latent Image. It is read, never changed. A video latent shaped "
                        "[batch, channels, frames, height, width] is measured the same way."
                    ),
                ),
                io.Combo.Input(
                    "scope",
                    options=["whole batch", "one latent"],
                    default="whole batch",
                    tooltip=(
                        "What to measure. `whole batch` answers one set of figures for every "
                        "latent together; `one latent` measures the one the index picks."
                    ),
                ),
                io.MultiType.Input(
                    io.Int.Input(
                        "index",
                        default=0,
                        min=-limits.max_resolution(),
                        max=limits.max_resolution(),
                        step=1,
                    ),
                    [io.Int, NUMBER, io.Float],
                    tooltip=(
                        "Which latent to measure, read only when scope is `one latent`. "
                        "Counts from 0, and negatives count from the end: -1 = last, -2 the "
                        "one before it. A decimal is truncated: 2.7 = 2."
                    ),
                ),
                io.Combo.Input(
                    "out_of_range",
                    options=list(OUT_OF_RANGE),
                    default="error",
                    tooltip=(
                        "Index outside 0..batch_size-1, which index -1 never reaches. With "
                        "3 latents and index 4: `wrap` = latent 1, `clamp` = latent 2, "
                        "`error` stops the prompt and names the batch size."
                    ),
                ),
            ],
            outputs=[
                io.Float.Output(
                    display_name="mean",
                    tooltip=(
                        "Average of every value measured. A denoised latent sits near 0.0 "
                        "and drifts with the content, so compare it between runs of one "
                        "workflow rather than against a fixed figure. nan when any value "
                        "measured is nan."
                    ),
                ),
                io.Float.Output(
                    display_name="std",
                    tooltip=(
                        "Spread of the values around the mean, over all of them rather than "
                        "a sample. Around 1.0 for fresh noise. 0.0 for an Empty Latent "
                        "Image, which is all zeros, and for a result that has collapsed to "
                        "a flat block."
                    ),
                ),
                io.Float.Output(
                    display_name="min",
                    tooltip=(
                        "Smallest value measured. With max it gives the range: an ordinary "
                        "SD latent stays inside roughly -10 to 10, and a much wider range "
                        "is a sampler running away."
                    ),
                ),
                io.Float.Output(
                    display_name="max",
                    tooltip=(
                        "Largest value measured. inf where the sampler overflowed, which is "
                        "also what turns is_finite false."
                    ),
                ),
                io.Float.Output(
                    display_name="absolute_mean",
                    tooltip=(
                        "Average of the values with their signs dropped, so positives and "
                        "negatives cancelling out cannot hide a strong latent. Near 0.0 "
                        "means an empty or collapsed latent whatever mean says."
                    ),
                ),
                io.Int.Output(
                    display_name="batch_size",
                    tooltip=(
                        "How many latents the batch holds, whatever index was set. Wire it "
                        "into a loop's iterations to walk the batch one latent at a time."
                    ),
                ),
                io.Int.Output(
                    display_name="channels",
                    tooltip=(
                        "Channels each latent carries: 4 for SD1.5 and SDXL, 16 for SD3, "
                        "Flux and Wan. Worth testing before a latent is handed to a "
                        "different model than the one that made it."
                    ),
                ),
                io.Int.Output(
                    display_name="height",
                    tooltip=(
                        "Latent rows, an eighth of the decoded pixel height: 64 here decodes "
                        "to 512 pixels. 0 for a latent with no rows, such as an audio one."
                    ),
                ),
                io.Int.Output(
                    display_name="width",
                    tooltip=(
                        "Latent columns, an eighth of the decoded pixel width: 64 here "
                        "decodes to 512 pixels."
                    ),
                ),
                io.Boolean.Output(
                    display_name="is_finite",
                    tooltip=(
                        "false where any value measured is nan or inf, which is what a "
                        "diverging sampler leaves behind. Wire it into Any Gate so a broken "
                        "run stops before the VAE decode instead of saving a black frame."
                    ),
                ),
                io.String.Output(
                    display_name="summary",
                    tooltip=(
                        "Every figure on one line, as `index=all  batch_size=1  "
                        "channels=4  ...  non_finite=0`. For a log, a console print, or "
                        "burning into a frame with Image Draw Text."
                    ),
                ),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls, samples, scope="whole batch", index=0, out_of_range="error"
    ) -> io.NodeOutput:
        """Measure the whole batch, or one latent of it.

        Args:
            samples: The LATENT to read.
            scope: Measure every latent together, or the one the index picks.
            index: Which latent to measure, counting from 0, negatives from the end.
            out_of_range: ``wrap``, ``clamp`` or ``error``.

        Returns:
            The five figures, the four sizes, whether every value is finite, and the lot
            as one line of text.

        Raises:
            ValueError: Nothing is connected to samples, the value carries no tensor, the
                batch holds nothing, or index is outside it and ``out_of_range`` is
                ``error``.
        """
        require_input(
            samples,
            NODE,
            "samples",
            "latent",
            "latent source such as a sampler, VAE Encode or Empty Latent Image",
            "LATENT",
        )
        tensor = cls.tensor_of(samples)
        batch_size, channels, height, width = cls.sizes(tensor)
        if batch_size == 0 or tensor.numel() == 0:
            raise ValueError(
                f"{NODE} was given a latent holding no values, so there is nothing to "
                f"measure. It arrived as batch_size={batch_size}, channels={channels}, "
                f"height={height}, width={width}, and each of those has to be 1 or more. "
                f"Check the node feeding samples."
            )

        if scope == "whole batch":
            measured, label = tensor, "all"
        else:
            # Negatives count from the end, as everywhere else in the pack.
            wanted = int(index)
            position = cls.resolve(
                wanted + batch_size if wanted < 0 else wanted, batch_size, out_of_range
            )
            measured, label = tensor[position], str(position)

        values = measured.detach().reshape(-1).to(torch.float32)
        non_finite = int((~torch.isfinite(values)).sum())
        mean = float(values.mean())
        std = float(values.std(unbiased=False))
        smallest = float(values.min())
        largest = float(values.max())
        absolute_mean = float(values.abs().mean())
        summary = (
            f"index={label}  batch_size={batch_size}  channels={channels}  "
            f"height={height}  width={width}  mean={mean:.6f}  std={std:.6f}  "
            f"min={smallest:.6f}  max={largest:.6f}  absolute_mean={absolute_mean:.6f}  "
            f"non_finite={non_finite}"
        )
        return io.NodeOutput(
            mean,
            std,
            smallest,
            largest,
            absolute_mean,
            batch_size,
            channels,
            height,
            width,
            non_finite == 0,
            summary,
            ui=ui.PreviewText(summary),
        )

    @staticmethod
    def tensor_of(samples) -> torch.Tensor:
        """The tensor a LATENT carries.

        Args:
            samples: Whatever arrived on the samples input.

        Returns:
            The tensor under the mapping's ``samples`` key. A latent packing several
            streams answers the first of them.

        Raises:
            ValueError: The value is not a latent, or its ``samples`` key holds something
                other than a tensor.
        """
        tensor = samples.get("samples") if isinstance(samples, dict) else None
        if getattr(tensor, "is_nested", False):
            streams = list(tensor.unbind())
            tensor = streams[0] if streams else None
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(
                f"{NODE} was given something on samples that is not a latent: a LATENT is a "
                f"mapping carrying its values under 'samples'. Wire a sampler, a VAE Encode "
                f"or an Empty Latent Image into this node."
            )
        return tensor

    @staticmethod
    def sizes(tensor: torch.Tensor) -> tuple[int, int, int, int]:
        """The four sizes a latent's shape reports.

        Args:
            tensor: A latent's tensor, ``[B, C, H, W]`` or ``[B, C, T, H, W]``.

        Returns:
            ``(batch_size, channels, height, width)``, each 0 where the shape has no such
            axis.
        """
        shape = tuple(int(size) for size in tensor.shape)
        # The last two axes past batch and channels are the rows and columns, frame axis or not.
        spatial = shape[2:]
        return (
            shape[0] if shape else 0,
            shape[1] if len(shape) > 1 else 0,
            spatial[-2] if len(spatial) > 1 else 0,
            spatial[-1] if spatial else 0,
        )

    @staticmethod
    def resolve(index: int, count: int, out_of_range: str) -> int:
        """Turn a requested index into a latent's position in the batch.

        Args:
            index: The requested position, counting from 0.
            count: How many latents the batch holds, which is one or more.
            out_of_range: One of :data:`modules.logic.switch_index.OUT_OF_RANGE`.

        Returns:
            A position from 0 to ``count - 1``.

        Raises:
            ValueError: The index is outside the batch and ``out_of_range`` is ``error``.
        """
        if 0 <= index < count:
            return index
        if out_of_range == "wrap":
            return index % count
        if out_of_range == "clamp":
            return 0 if index < 0 else count - 1
        raise ValueError(
            f"{NODE} was asked for latent {index} of a batch holding {count}, numbered 0 to "
            f"{count - 1}. Set index to -1 to measure the whole batch, or set out_of_range "
            f"to wrap or clamp."
        )
