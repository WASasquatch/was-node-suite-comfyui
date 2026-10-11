"""Truncate up to four strings to a character or word count."""

from __future__ import annotations

from comfy_api.latest import io


#: Input ids in socket order, the first box being the unsuffixed one.
SLOTS = ("text", *(f"text_{letter}" for letter in "bcdefghijklmnopqrstuvwx"))


class TextStringTruncate(io.ComfyNode):
    """Cut each connected string down to ``truncate_to`` characters or words."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Text String Truncate",
            display_name="Text String Truncate",
            search_aliases=["Text String Truncate", "truncate", "trim", "shorten", "clip text"],
            category="WAS Suite/Text/Operations",
            description=(
                "Truncate up to four strings to a number of characters or words, keeping "
                "either the beginning or the end."
            ),
            inputs=[
                io.Combo.Input(
                    "truncate_by",
                    options=["characters", "words"],
                    tooltip=(
                        "What truncate_to counts. `characters` counts single characters, "
                        "spaces included. `words` counts whitespace-separated words and "
                        "rejoins them with one space each, so line breaks and runs of "
                        "spaces in the text collapse."
                    ),
                ),
                io.Combo.Input(
                    "truncate_from",
                    options=["end", "beginning"],
                    tooltip=(
                        "Which end of the text is kept. `end` keeps the tail and throws the "
                        "start away, so 'a long prompt' truncated to 6 characters becomes "
                        "'prompt'. `beginning` keeps the head, giving 'a long'."
                    ),
                ),
                io.Int.Input(
                    "truncate_to",
                    default=10,
                    min=-99999999,
                    max=99999999,
                    step=1,
                    tooltip=(
                        "How much to keep, in characters or words. A negative value counts "
                        "what to remove: -10 with `end` keeps all but the last 10; with "
                        "`beginning`, only the last 10. 0 with `end` keeps everything; with "
                        "`beginning`, nothing."
                    ),
                ),
                io.String.Input(
                    "text",
                    multiline=True,
                    placeholder="Eg: a long caption",
                    tooltip=(
                        "First text to shorten; STRING. Cut per truncate_by, "
                        "truncate_from and truncate_to; leaves on TEXT."
                    ),
                ),
                io.String.Input(
                    "text_b",
                    multiline=True,
                    optional=True,
                    placeholder="Eg: a second caption",
                    tooltip=(
                        "Second text; STRING. Same settings, leaves on TEXT_B."
                    ),
                ),
                io.String.Input(
                    "text_c",
                    multiline=True,
                    optional=True,
                    placeholder="Eg: a third caption",
                    tooltip=(
                        "Third text; STRING. Same settings, leaves on TEXT_C."
                    ),
                ),
                io.String.Input(
                    "text_d",
                    multiline=True,
                    optional=True,
                    placeholder="Eg: a fourth caption",
                    tooltip=(
                        "Fourth text; STRING. Same settings, leaves on TEXT_D."
                    ),
                ),
                io.String.Input(
                    "text_e",
                    multiline=True,
                    optional=True,
                    placeholder="Eg: a fourth caption",
                    tooltip=(
                        "Text 5, emitted on its own output. Tokens are substituted."
                    ),
                ),
                io.String.Input(
                    "text_f",
                    multiline=True,
                    optional=True,
                    placeholder="Eg: a fourth caption",
                    tooltip=(
                        "Text 6, emitted on its own output. Tokens are substituted."
                    ),
                ),
                io.String.Input(
                    "text_g",
                    multiline=True,
                    optional=True,
                    placeholder="Eg: a fourth caption",
                    tooltip=(
                        "Text 7, emitted on its own output. Tokens are substituted."
                    ),
                ),
                io.String.Input(
                    "text_h",
                    multiline=True,
                    optional=True,
                    placeholder="Eg: a fourth caption",
                    tooltip=(
                        "Text 8, emitted on its own output. Tokens are substituted."
                    ),
                ),
                io.String.Input(
                    "text_i",
                    multiline=True,
                    optional=True,
                    placeholder="Eg: a fourth caption",
                    tooltip=(
                        "Text 9, emitted on its own output. Tokens are substituted."
                    ),
                ),
                io.String.Input(
                    "text_j",
                    multiline=True,
                    optional=True,
                    placeholder="Eg: a fourth caption",
                    tooltip=(
                        "Text 10, emitted on its own output. Tokens are substituted."
                    ),
                ),
                io.String.Input(
                    "text_k",
                    multiline=True,
                    optional=True,
                    placeholder="Eg: a fourth caption",
                    tooltip=(
                        "Text 11, emitted on its own output. Tokens are substituted."
                    ),
                ),
                io.String.Input(
                    "text_l",
                    multiline=True,
                    optional=True,
                    placeholder="Eg: a fourth caption",
                    tooltip=(
                        "Text 12, emitted on its own output. Tokens are substituted."
                    ),
                ),
                io.String.Input(
                    "text_m",
                    multiline=True,
                    optional=True,
                    placeholder="Eg: a fourth caption",
                    tooltip=(
                        "Text 13, emitted on its own output. Tokens are substituted."
                    ),
                ),
                io.String.Input(
                    "text_n",
                    multiline=True,
                    optional=True,
                    placeholder="Eg: a fourth caption",
                    tooltip=(
                        "Text 14, emitted on its own output. Tokens are substituted."
                    ),
                ),
                io.String.Input(
                    "text_o",
                    multiline=True,
                    optional=True,
                    placeholder="Eg: a fourth caption",
                    tooltip=(
                        "Text 15, emitted on its own output. Tokens are substituted."
                    ),
                ),
                io.String.Input(
                    "text_p",
                    multiline=True,
                    optional=True,
                    placeholder="Eg: a fourth caption",
                    tooltip=(
                        "Text 16, emitted on its own output. Tokens are substituted."
                    ),
                ),
                io.String.Input(
                    "text_q",
                    multiline=True,
                    optional=True,
                    placeholder="Eg: a fourth caption",
                    tooltip=(
                        "Text 17, emitted on its own output. Tokens are substituted."
                    ),
                ),
                io.String.Input(
                    "text_r",
                    multiline=True,
                    optional=True,
                    placeholder="Eg: a fourth caption",
                    tooltip=(
                        "Text 18, emitted on its own output. Tokens are substituted."
                    ),
                ),
                io.String.Input(
                    "text_s",
                    multiline=True,
                    optional=True,
                    placeholder="Eg: a fourth caption",
                    tooltip=(
                        "Text 19, emitted on its own output. Tokens are substituted."
                    ),
                ),
                io.String.Input(
                    "text_t",
                    multiline=True,
                    optional=True,
                    placeholder="Eg: a fourth caption",
                    tooltip=(
                        "Text 20, emitted on its own output. Tokens are substituted."
                    ),
                ),
                io.String.Input(
                    "text_u",
                    multiline=True,
                    optional=True,
                    placeholder="Eg: a fourth caption",
                    tooltip=(
                        "Text 21, emitted on its own output. Tokens are substituted."
                    ),
                ),
                io.String.Input(
                    "text_v",
                    multiline=True,
                    optional=True,
                    placeholder="Eg: a fourth caption",
                    tooltip=(
                        "Text 22, emitted on its own output. Tokens are substituted."
                    ),
                ),
                io.String.Input(
                    "text_w",
                    multiline=True,
                    optional=True,
                    placeholder="Eg: a fourth caption",
                    tooltip=(
                        "Text 23, emitted on its own output. Tokens are substituted."
                    ),
                ),
                io.String.Input(
                    "text_x",
                    multiline=True,
                    optional=True,
                    placeholder="Eg: a fourth caption",
                    tooltip=(
                        "Text 24, emitted on its own output. Tokens are substituted."
                    ),
                ),
            ],
            outputs=[
                io.String.Output(display_name="TEXT", tooltip="The shortened text."),
                io.String.Output(display_name="TEXT_B", tooltip="The shortened text_b."),
                io.String.Output(display_name="TEXT_C", tooltip="The shortened text_c."),
                io.String.Output(display_name="TEXT_D", tooltip="The shortened text_d."),
                io.String.Output(display_name="TEXT_E", tooltip="Text 5, with its tokens substituted."),
                io.String.Output(display_name="TEXT_F", tooltip="Text 6, with its tokens substituted."),
                io.String.Output(display_name="TEXT_G", tooltip="Text 7, with its tokens substituted."),
                io.String.Output(display_name="TEXT_H", tooltip="Text 8, with its tokens substituted."),
                io.String.Output(display_name="TEXT_I", tooltip="Text 9, with its tokens substituted."),
                io.String.Output(display_name="TEXT_J", tooltip="Text 10, with its tokens substituted."),
                io.String.Output(display_name="TEXT_K", tooltip="Text 11, with its tokens substituted."),
                io.String.Output(display_name="TEXT_L", tooltip="Text 12, with its tokens substituted."),
                io.String.Output(display_name="TEXT_M", tooltip="Text 13, with its tokens substituted."),
                io.String.Output(display_name="TEXT_N", tooltip="Text 14, with its tokens substituted."),
                io.String.Output(display_name="TEXT_O", tooltip="Text 15, with its tokens substituted."),
                io.String.Output(display_name="TEXT_P", tooltip="Text 16, with its tokens substituted."),
                io.String.Output(display_name="TEXT_Q", tooltip="Text 17, with its tokens substituted."),
                io.String.Output(display_name="TEXT_R", tooltip="Text 18, with its tokens substituted."),
                io.String.Output(display_name="TEXT_S", tooltip="Text 19, with its tokens substituted."),
                io.String.Output(display_name="TEXT_T", tooltip="Text 20, with its tokens substituted."),
                io.String.Output(display_name="TEXT_U", tooltip="Text 21, with its tokens substituted."),
                io.String.Output(display_name="TEXT_V", tooltip="Text 22, with its tokens substituted."),
                io.String.Output(display_name="TEXT_W", tooltip="Text 23, with its tokens substituted."),
                io.String.Output(display_name="TEXT_X", tooltip="Text 24, with its tokens substituted."),
            ],
        )

    @classmethod
    def execute(
        cls, text, truncate_by, truncate_from, truncate_to, **extra
    ) -> io.NodeOutput:
        values = [text, *(extra.get(name, "") for name in SLOTS[1:])]
        return io.NodeOutput(
            *(cls.truncate(value, truncate_to, truncate_from, truncate_by) for value in values)
        )

    @staticmethod
    def truncate(string, max_length, mode="end", truncate_by="characters"):
        """One string, cut to ``max_length`` units from the end named by ``mode``.

        Args:
            string: The text to cut.
            max_length: Units to keep, or units to drop when negative.
            mode: ``beginning`` or ``end``, which end of the string survives.
            truncate_by: ``characters`` or ``words``.

        Returns:
            The truncated string. A ``max_length`` of 0 with ``mode`` ``end`` keeps the
            whole string.
        """
        if truncate_by == "characters":
            if mode == "beginning":
                return string[:max_length] if max_length >= 0 else string[max_length:]
            return string[-max_length:] if max_length >= 0 else string[:max_length]
        words = string.split()
        if mode == "beginning":
            return " ".join(words[:max_length]) if max_length >= 0 else " ".join(words[max_length:])
        return " ".join(words[-max_length:]) if max_length >= 0 else " ".join(words[:max_length])
