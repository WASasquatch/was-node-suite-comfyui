"""Mix conditionings together with a choice of blend formulas."""

from __future__ import annotations

import math

import torch
from comfy_api.latest import io

from ....modules.compat.sockets import require_input

REQUIRES = "extras"

#: Blend formulas, in the order the widget offers them.
BLENDING_MODES = [
    "add",
    "bislerp",
    "cosine interp",
    "cuberp",
    "difference",
    "exclusion",
    "hslerp",
    "inject",
    "lerp",
    "random",
    "slerp",
    "subtract",
]


def normalize(tensor, target_min=None, target_max=None):
    """Rescale a tensor into a range.

    Args:
        tensor: The tensor to rescale.
        target_min: Value the smallest element becomes. ``None`` keeps the tensor's own
            minimum, which leaves the range where it is.
        target_max: Value the largest element becomes. ``None`` keeps the tensor's own
            maximum.

    Returns:
        The rescaled tensor.
    """
    min_val = tensor.min()
    max_val = tensor.max()

    if target_min is None:
        target_min = min_val
    if target_max is None:
        target_max = max_val

    normalized = (tensor - min_val) / (max_val - min_val)
    return normalized * (target_max - target_min) + target_min


#: Each formula takes the two tensors and a factor, and returns the mixed tensor.
BLEND_FUNCTIONS = {
    # Weighted sum with the factor split between the two, so the total stays put.
    "add": lambda a, b, t: (a * t + b * (1 - t)),
    # Straight linear interpolation.
    "bislerp": lambda a, b, t: (a * (1 - t) + b * t),
    # Linear interpolation with the factor eased by a cosine at both ends.
    "cosine interp": lambda a, b, t: (a + b - (a - b) * torch.cos(t * torch.tensor(math.pi))) / 2,
    # Linear interpolation with the factor eased by a cubic at both ends.
    "cuberp": lambda a, b, t: a + (b - a) * (3 * t**2 - 2 * t**3),
    # How far apart the two are, scaled by the factor.
    "difference": lambda a, b, t: (abs(a - b) * t),
    # The two combined with agreement cancelled out, scaled by the factor.
    "exclusion": lambda a, b, t: ((a + b - 2 * a * b) * t),
    # Linear interpolation that reverses which tensor leads once the factor passes half.
    "hslerp": lambda a, b, t: (a * (1 - t) + b * t) if t < 0.5 else (a * t + b * (1 - t)),
    # The second added on top of the first rather than mixed into it.
    "inject": lambda a, b, t: (a + b * t),
    # Straight linear interpolation.
    "lerp": lambda a, b, t: (a * (1 - t) + b * t),
    # The second scaled by fresh random noise before it is mixed in.
    "random": lambda a, b, t: (a + (torch.rand_like(b) * b - a) * t),
    # Straight linear interpolation.
    "slerp": lambda a, b, t: (a * (1 - t) + b * t),
    # The second taken away from the first, both scaled by the factor.
    "subtract": lambda a, b, t: (a * t - b * t),
}


def pooled_output(conditioning, socket):
    """Read a conditioning's pooled output.

    Args:
        conditioning: The value that arrived on the socket.
        socket: The input's name, spelled as the schema spells it.

    Returns:
        The pooled output tensor.

    Raises:
        ValueError: The conditioning carries no pooled output.
    """
    pooled = conditioning[0][1].get("pooled_output")
    if pooled is None:
        raise ValueError(
            f"Conditioning (Blend) has no pooled output on its {socket} input. Encode that "
            "prompt with a text encoder that produces one, such as SDXL's or Flux's."
        )
    return pooled


#: Every slot the series grows through, in the order they are blended. conditioning_a and
#: conditioning_b are always drawn; the rest are revealed by web/was_growing_sockets.js as each
#: one is wired.
SLOT_NAMES = tuple(f"conditioning_{letter}" for letter in "abcdefghijklmnopqrstuvwxyz")


class WASConditioningBlend(io.ComfyNode):
    """Blend any number of conditionings into one by a choice of formulas."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASConditioningBlend",
            display_name="Conditioning (Blend)",
            search_aliases=[
                "WASConditioningBlend",
                "Conditioning (Blend)",
                "prompt mix",
                "blend prompts",
                "conditioning average",
            ],
            category="WAS Suite/Conditioning",
            description=(
                "Mix any number of encoded prompts into one by a choice of twelve formulas, "
                "giving a subject genuinely between them rather than a picture containing "
                "both. Each further slot blends onto the result so far. Every prompt needs a "
                "pooled output, such as SDXL's or Flux's."
            ),
            inputs=[
                io.Conditioning.Input(
                    "conditioning_a",
                    tooltip=(
                        "The prompt blended away from. Only its first entry is read, so feed "
                        "it a plain text encode rather than a combined or scheduled "
                        "conditioning."
                    ),
                ),
                io.Conditioning.Input(
                    "conditioning_b",
                    tooltip=(
                        "The prompt blended towards. Only its first entry is read, as with "
                        "conditioning_a."
                    ),
                ),
                io.Combo.Input(
                    "blending_mode",
                    options=BLENDING_MODES,
                    tooltip=(
                        "`lerp`, `slerp`, `bislerp` = straight mix; `cosine interp`, `cuberp` "
                        "= ends held longer; `add` = balances the pair; `inject` = newer "
                        "prompt on top; `difference`, `exclusion` = only what they disagree "
                        "on; `random` = varies per element, uses seed."
                    ),
                ),
                io.Float.Input(
                    "blending_strength",
                    default=0.5,
                    min=-10.0,
                    max=10.0,
                    step=0.001,
                    tooltip=(
                        "How far each blend leans: 0.5 = even mix; with `lerp` and the straight "
                        "mixes, 0.0 = the next prompt, 1.0 = the result so far, reversed for "
                        "`add` and `cosine interp`. Outside 0 to 1 pushes past either prompt."
                    ),
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    tooltip=(
                        "Seed for the `random` blending mode, so a run can be repeated. 0 "
                        "leaves the random source as it was, which makes `random` differ from "
                        "run to run. Every other mode ignores this."
                    ),
                ),
                io.Conditioning.Input(
                    "conditioning_c",
                    optional=True,
                    tooltip=(
                        "A further prompt, blended onto the result of the ones before it "
                        "with the same mode and strength. The interface reveals the next "
                        "slot as this one is filled."
                    ),
                ),
                io.Conditioning.Input(
                    "conditioning_d",
                    optional=True,
                    tooltip=(
                        "A further prompt, blended onto the result of the ones before it "
                        "with the same mode and strength. The interface reveals the next "
                        "slot as this one is filled."
                    ),
                ),
                io.Conditioning.Input(
                    "conditioning_e",
                    optional=True,
                    tooltip=(
                        "A further prompt, blended onto the result of the ones before it "
                        "with the same mode and strength. The interface reveals the next "
                        "slot as this one is filled."
                    ),
                ),
                io.Conditioning.Input(
                    "conditioning_f",
                    optional=True,
                    tooltip=(
                        "A further prompt, blended onto the result of the ones before it "
                        "with the same mode and strength. The interface reveals the next "
                        "slot as this one is filled."
                    ),
                ),
                io.Conditioning.Input(
                    "conditioning_g",
                    optional=True,
                    tooltip=(
                        "A further prompt, blended onto the result of the ones before it "
                        "with the same mode and strength. The interface reveals the next "
                        "slot as this one is filled."
                    ),
                ),
                io.Conditioning.Input(
                    "conditioning_h",
                    optional=True,
                    tooltip=(
                        "A further prompt, blended onto the result of the ones before it "
                        "with the same mode and strength. The interface reveals the next "
                        "slot as this one is filled."
                    ),
                ),
                io.Conditioning.Input(
                    "conditioning_i",
                    optional=True,
                    tooltip=(
                        "A further prompt, blended onto the result of the ones before it "
                        "with the same mode and strength. The interface reveals the next "
                        "slot as this one is filled."
                    ),
                ),
                io.Conditioning.Input(
                    "conditioning_j",
                    optional=True,
                    tooltip=(
                        "A further prompt, blended onto the result of the ones before it "
                        "with the same mode and strength. The interface reveals the next "
                        "slot as this one is filled."
                    ),
                ),
                io.Conditioning.Input(
                    "conditioning_k",
                    optional=True,
                    tooltip=(
                        "A further prompt, blended onto the result of the ones before it "
                        "with the same mode and strength. The interface reveals the next "
                        "slot as this one is filled."
                    ),
                ),
                io.Conditioning.Input(
                    "conditioning_l",
                    optional=True,
                    tooltip=(
                        "A further prompt, blended onto the result of the ones before it "
                        "with the same mode and strength. The interface reveals the next "
                        "slot as this one is filled."
                    ),
                ),
                io.Conditioning.Input(
                    "conditioning_m",
                    optional=True,
                    tooltip=(
                        "A further prompt, blended onto the result of the ones before it "
                        "with the same mode and strength. The interface reveals the next "
                        "slot as this one is filled."
                    ),
                ),
                io.Conditioning.Input(
                    "conditioning_n",
                    optional=True,
                    tooltip=(
                        "A further prompt, blended onto the result of the ones before it "
                        "with the same mode and strength. The interface reveals the next "
                        "slot as this one is filled."
                    ),
                ),
                io.Conditioning.Input(
                    "conditioning_o",
                    optional=True,
                    tooltip=(
                        "A further prompt, blended onto the result of the ones before it "
                        "with the same mode and strength. The interface reveals the next "
                        "slot as this one is filled."
                    ),
                ),
                io.Conditioning.Input(
                    "conditioning_p",
                    optional=True,
                    tooltip=(
                        "A further prompt, blended onto the result of the ones before it "
                        "with the same mode and strength. The interface reveals the next "
                        "slot as this one is filled."
                    ),
                ),
                io.Conditioning.Input(
                    "conditioning_q",
                    optional=True,
                    tooltip=(
                        "A further prompt, blended onto the result of the ones before it "
                        "with the same mode and strength. The interface reveals the next "
                        "slot as this one is filled."
                    ),
                ),
                io.Conditioning.Input(
                    "conditioning_r",
                    optional=True,
                    tooltip=(
                        "A further prompt, blended onto the result of the ones before it "
                        "with the same mode and strength. The interface reveals the next "
                        "slot as this one is filled."
                    ),
                ),
                io.Conditioning.Input(
                    "conditioning_s",
                    optional=True,
                    tooltip=(
                        "A further prompt, blended onto the result of the ones before it "
                        "with the same mode and strength. The interface reveals the next "
                        "slot as this one is filled."
                    ),
                ),
                io.Conditioning.Input(
                    "conditioning_t",
                    optional=True,
                    tooltip=(
                        "A further prompt, blended onto the result of the ones before it "
                        "with the same mode and strength. The interface reveals the next "
                        "slot as this one is filled."
                    ),
                ),
                io.Conditioning.Input(
                    "conditioning_u",
                    optional=True,
                    tooltip=(
                        "A further prompt, blended onto the result of the ones before it "
                        "with the same mode and strength. The interface reveals the next "
                        "slot as this one is filled."
                    ),
                ),
                io.Conditioning.Input(
                    "conditioning_v",
                    optional=True,
                    tooltip=(
                        "A further prompt, blended onto the result of the ones before it "
                        "with the same mode and strength. The interface reveals the next "
                        "slot as this one is filled."
                    ),
                ),
                io.Conditioning.Input(
                    "conditioning_w",
                    optional=True,
                    tooltip=(
                        "A further prompt, blended onto the result of the ones before it "
                        "with the same mode and strength. The interface reveals the next "
                        "slot as this one is filled."
                    ),
                ),
                io.Conditioning.Input(
                    "conditioning_x",
                    optional=True,
                    tooltip=(
                        "A further prompt, blended onto the result of the ones before it "
                        "with the same mode and strength. The interface reveals the next "
                        "slot as this one is filled."
                    ),
                ),
                io.Conditioning.Input(
                    "conditioning_y",
                    optional=True,
                    tooltip=(
                        "A further prompt, blended onto the result of the ones before it "
                        "with the same mode and strength. The interface reveals the next "
                        "slot as this one is filled."
                    ),
                ),
                io.Conditioning.Input(
                    "conditioning_z",
                    optional=True,
                    tooltip=(
                        "A further prompt, blended onto the result of the ones before it "
                        "with the same mode and strength. The interface reveals the next "
                        "slot as this one is filled."
                    ),
                ),
            ],
            outputs=[
                io.Conditioning.Output(
                    display_name="conditioning",
                    tooltip=(
                        "The blended prompt, as a one-entry conditioning carrying the mixed "
                        "embedding and pooled output. Anything else the inputs carried, an "
                        "area, a mask, a control hint, is not passed on."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(
        cls, conditioning_a, conditioning_b, blending_mode, blending_strength, seed, **extra
    ) -> io.NodeOutput:
        """Blend every connected prompt together, in slot order.

        Args:
            conditioning_a: The prompt blended away from.
            conditioning_b: The prompt blended towards.
            blending_mode: Which formula combines a pair.
            blending_strength: How strongly each blend leans, as the mode reads it.
            seed: Seed for the ``random`` mode.
            extra: The optional ``conditioning_c`` to ``conditioning_z`` slots, connected or
                not.

        Returns:
            One conditioning holding the blended embedding and pooled output.

        Raises:
            ValueError: A required conditioning input is empty, or a connected one carries no
                pooled output.
        """
        for value, socket in (
            (conditioning_a, "conditioning_a"),
            (conditioning_b, "conditioning_b"),
        ):
            require_input(
                value,
                "Conditioning (Blend)",
                socket,
                "conditioning",
                "CLIP Text Encode",
                "CONDITIONING",
            )

        if seed > 0:
            torch.manual_seed(seed)

        blend = BLEND_FUNCTIONS[blending_mode]
        factor = 1 - blending_strength

        embedding = conditioning_a[0][0].clone()
        pooled = pooled_output(conditioning_a, "conditioning_a").clone()
        for name in SLOT_NAMES[1:]:
            following = conditioning_b if name == "conditioning_b" else extra.get(name)
            if following is None:
                continue
            embedding = normalize(blend(embedding, following[0][0].clone(), factor))
            pooled = normalize(blend(pooled, pooled_output(following, name).clone(), factor))

        return io.NodeOutput([[embedding, {"pooled_output": pooled}]])
