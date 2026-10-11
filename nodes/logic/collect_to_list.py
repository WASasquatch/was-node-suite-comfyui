"""Gather every value a fan-out produced back onto one wire."""

from __future__ import annotations

from comfy_api.latest import io

NODE_NAME = "Collect to List"

#: What a run that collected nothing reports.
EMPTY = (
    f"{NODE_NAME} was handed nothing to collect. Whatever is wired into value produced an "
    f"empty list, so there is no value, no count and no text to answer with. Check that the "
    f"node feeding it found something: an archive holding no readable entries and a range of "
    f"zero steps both produce an empty list."
)

#: The delimiter spelling that stands for a newline.
NEWLINE = "\\n"


def first(values, fallback: str) -> str:
    """The first entry of a widget's value, which arrives as a list.

    Args:
        values: Whatever the socket delivered.
        fallback: Answer for a socket that delivered nothing.

    Returns:
        The entry as a string.
    """
    if isinstance(values, (list, tuple)):
        values = values[0] if values else None
    return fallback if values is None else str(values)


def as_text(value) -> str:
    """One collected value written out for the joined string.

    Args:
        value: Whatever one run of the fan-out put on the socket.

    Returns:
        Text and numbers as they are, and anything else as its kind and size, such as
        ``IMAGE 5x512x512x3``.
    """
    from ...modules.logic.describe import describe_value

    if value is None:
        return ""
    if isinstance(value, (bool, int, float, str)):
        return str(value)
    described = describe_value(value)
    shape = described["shape"]
    return f"{described['type_name']} {shape}" if shape else described["type_name"]


class CollectToList(io.ComfyNode):
    """Join everything a fan-out produced into one value, a count and one string."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        template = io.MatchType.Template("collect_to_list")
        return io.Schema(
            node_id="WASCollectToList",
            display_name=NODE_NAME,
            search_aliases=[
                "WASCollectToList",
                NODE_NAME,
                "collect",
                "gather",
                "fan in",
                "list to batch",
                "join runs",
            ],
            category="WAS Suite/Logic/Loop",
            description=(
                "Gather every run of a fan-out, such as Number Range, Zip Open or Load Text "
                "Files From Zip, back onto one wire. Images, masks and latents join into one "
                "batch for a video encoder or a single save; anything else arrives as a list. "
                "count says how many were gathered and joined writes them as one string."
            ),
            inputs=[
                io.MatchType.Input(
                    "value",
                    template=template,
                    tooltip=(
                        "The last node of the fan-out, wired here once. Whatever connects, "
                        "IMAGE, LATENT, STRING or a model, sets the type the value output "
                        "carries. A source that ran once gathers into a collection of one."
                    ),
                ),
                io.String.Input(
                    "delimiter",
                    default=", ",
                    tooltip=(
                        "Placed between the entries in joined. ', ' builds a comma-separated "
                        "caption; \\n puts each entry on its own line; empty runs them "
                        "together with nothing between. It changes neither value nor count."
                    ),
                ),
            ],
            outputs=[
                io.MatchType.Output(
                    template=template,
                    display_name="value",
                    tooltip=(
                        "Everything gathered, as one value. Images, masks and latents join "
                        "into a single batch, so 5 runs of one image give a 5 image batch; "
                        "anything else arrives as a list of 5. The socket carries the type "
                        "that was wired into value."
                    ),
                ),
                io.Int.Output(
                    display_name="count",
                    tooltip=(
                        "How many are in value. 5 runs of one image = 5; 5 runs of a 4 frame "
                        "batch = 20, joined into one batch; 5 runs of text = 5. Feed it to a "
                        "batch index or an iteration count."
                    ),
                ),
                io.String.Output(
                    display_name="joined",
                    tooltip=(
                        "Every gathered value written out as text, separated by delimiter. "
                        "Text and numbers appear as they are, an image, mask or latent as "
                        "its kind and size, 'IMAGE 1x512x512x3'. Save it to log what a run "
                        "gathered."
                    ),
                ),
            ],
            is_input_list=True,
        )

    @classmethod
    def execute(cls, value=None, delimiter=None) -> io.NodeOutput:
        """Join the collected values, count them and write them out.

        Args:
            value: Every value the fan-out produced, in run order.
            delimiter: The joined string's separator, as the list this node receives.

        Returns:
            The joined value, how many are in it, and the values as one string.

        Raises:
            ValueError: The fan-out produced nothing.
        """
        from ...modules.compat.lists import as_list, require_values
        from ...modules.logic import loop_accumulate

        values = require_values(as_list(value), EMPTY)
        separator = first(delimiter, ", ")
        if separator == NEWLINE:
            separator = "\n"

        collected, _kind = loop_accumulate.finalize(values, True, values[-1])
        frames = loop_accumulate.frame_count(collected)
        count = len(values) if frames is None else int(frames)
        return io.NodeOutput(collected, count, separator.join(as_text(item) for item in values))
