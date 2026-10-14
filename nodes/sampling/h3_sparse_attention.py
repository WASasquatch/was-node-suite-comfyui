"""Block-sparse attention for MiniMax H3, set layer by layer."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.sampling import h3_sparse_attention as h3_sparse

logger = log.get_logger("nodes.h3_sparse_attention")

NODE_NAME = "H3 Sparse Attention"

#: Tooltip on the keep_percent of both methods that keep a fixed share of blocks.
KEEP_TIP = (
    "Share of key blocks each sparse layer keeps exactly: `10` = default, the share FastH3 was "
    "trained at; higher is closer to dense and slower."
)

#: The selection methods, each with the one setting it reads.
METHOD_OPTIONS = [
    io.DynamicCombo.Option(key="sol-attn", inputs=[
        io.Float.Input(
            "tau",
            default=1.0,
            min=0.0,
            max=4.0,
            step=0.05,
            tooltip=(
                "Sparsity of each sparse layer: `0` = every block exact, in INT8; `1.0` = "
                "default, about 16% of key blocks exact; `1.5` = about 7%; `2.0` = about 3%, "
                "fastest and softest."
            ),
        ),
    ]),
    io.DynamicCombo.Option(key="sla", inputs=[
        io.Float.Input("keep_percent", default=10.0, min=0.5, max=95.0, step=0.5, tooltip=KEEP_TIP),
    ]),
    io.DynamicCombo.Option(key="vsa", inputs=[
        io.Float.Input("keep_percent", default=10.0, min=0.5, max=95.0, step=0.5, tooltip=KEEP_TIP),
    ]),
]

#: The layer choices: the presets, then the typed plan.
LAYER_OPTIONS = [
    io.DynamicCombo.Option(key="balanced", inputs=[]),
    io.DynamicCombo.Option(key="careful", inputs=[]),
    io.DynamicCombo.Option(key="fastest", inputs=[]),
    io.DynamicCombo.Option(key=h3_sparse.MANUAL, inputs=[
        io.String.Input(
            "layer_plan",
            default=h3_sparse.SENSITIVE,
            tooltip=(
                "Layers 0 to 49 set apart: `39-42, 49` = those dense; `0-3: 0.5` = layers 0 to 3 "
                "at tau 0.5, or keep 0.5% under sla and vsa; `20: dense` = one dense. Later "
                "entries win; empty = every layer sparse."
            ),
        ),
    ]),
]


class H3SparseAttention(io.ComfyNode):
    """Make a MiniMax H3 model's attention block-sparse, layer by layer."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASH3SparseAttention",
            display_name=NODE_NAME,
            search_aliases=[
                "WASH3SparseAttention", NODE_NAME,
                "sparse attention",
                "sol-attn",
                "sol attn",
                "sla",
                "vsa",
                "block sparse attention",
                "minimax h3",
                "low vram",
            ],
            category="WAS Suite/Sampling",
            description=(
                "Run MiniMax H3's attention block-sparse with Sol-Attn, SLA or VSA, faster and "
                "lighter on VRAM, while chosen layers stay dense on the model's own attention "
                "backend. The layers H3's video suffers most for stay dense by default. Sparse "
                "layers compute in INT8, which softens detail."
            ),
            inputs=[
                io.Model.Input("model", tooltip="The MiniMax H3 model, after any LoRA."),
                io.DynamicCombo.Input(
                    "method",
                    options=METHOD_OPTIONS,
                    display_name="method",
                    tooltip=(
                        "`sol-attn` = default, an adaptive threshold per head, for any H3 "
                        "weights; `sla` = a fixed share of blocks, for SLA-distilled LoRAs; "
                        "`vsa` = video cubes plus a learned coarse branch, for FastH3 weights."
                    ),
                ),
                io.DynamicCombo.Input(
                    "layers",
                    options=LAYER_OPTIONS,
                    display_name="layers",
                    tooltip=(
                        "`balanced` = default, layers 39-42 and 49 dense, 2.3x faster steps "
                        "with sol-attn tau 1 on an RTX 4090; `careful` = 0-1, 37-43 and 49 "
                        "dense, about 2x; `fastest` = every layer sparse, 2.7x, softest; "
                        "`manual` = type layer_plan."
                    ),
                ),
                io.Float.Input(
                    "start_percent",
                    default=0.2,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "Point in the schedule sparse attention starts: `0.2` = default, the "
                        "first fifth of the steps dense; `0` = every step sparse."
                    ),
                ),
                io.Float.Input(
                    "end_percent",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Point in the schedule sparse attention stops: `1.0` = the last step.",
                ),
                io.Combo.Input(
                    "sink_conditioning",
                    options=h3_sparse.SINKS,
                    default=h3_sparse.SINKS[0],
                    advanced=True,
                    tooltip=(
                        "sol-attn and sla: `exact_kv_and_rows` = default, the prompt, reference "
                        "and sound rows read exactly and the sound rows run dense; `exact_kv` = "
                        "read exactly only; `off` = routed like the video."
                    ),
                ),
                io.Int.Input(
                    "extra_tokens",
                    default=256,
                    min=0,
                    max=256,
                    step=64,
                    advanced=True,
                    tooltip=(
                        "sol-attn and sla: top-scoring tokens each query block reads beyond its "
                        "chosen blocks: `256` = default, closest to dense; `0` = none, fastest."
                    ),
                ),
                io.Int.Input(
                    "min_tokens",
                    default=12288,
                    min=0,
                    max=1 << 20,
                    step=512,
                    advanced=True,
                    tooltip=(
                        "Calls with fewer tokens run dense: `12288` = default. 1344x768 at 124 "
                        "frames is about 38000 tokens; short clips run as fast dense."
                    ),
                ),
            ],
            outputs=[
                io.Model.Output(
                    display_name="model",
                    tooltip="The model with the planned layers sparse, for a guider or sampler.",
                ),
                io.String.Output(
                    display_name="plan",
                    tooltip="The method, and which layers run dense and which sparse, at what setting.",
                ),
            ],
        )

    @classmethod
    def execute(cls, model, method=None, layers=None, start_percent=0.2, end_percent=1.0,
                sink_conditioning=h3_sparse.SINKS[0], extra_tokens=256,
                min_tokens=12288) -> io.NodeOutput:
        """Patch the planned layers.

        Raises:
            ValueError: The model is not MiniMax H3, or layer_plan cannot be read.
        """
        method = method if isinstance(method, dict) else {"method": method or "sol-attn"}
        chosen = method.get("method", "sol-attn")
        value = float(method.get("tau", 1.0) if chosen == "sol-attn" else method.get("keep_percent", 10.0))
        layers = layers if isinstance(layers, dict) else {"layers": layers or "balanced"}
        choice = layers.get("layers", "balanced")
        if choice == h3_sparse.MANUAL:
            text = layers.get("layer_plan", h3_sparse.SENSITIVE)
        else:
            text = h3_sparse.PRESETS.get(choice, h3_sparse.SENSITIVE)
        plan = h3_sparse.parse_plan(text, value, len(h3_sparse.blocks_of(model)), chosen)
        patched = h3_sparse.sparse(model, plan, chosen, start_percent, end_percent,
                                   sink_conditioning, extra_tokens, min_tokens)
        report = f"{choice}, {h3_sparse.describe(plan, chosen)}"
        problem = h3_sparse.kernel_problem()
        if problem is not None:
            report = f"{report}. Every layer runs dense here: {problem}."
            logger.warning("%s: %s; every layer runs dense", NODE_NAME, problem)
        return io.NodeOutput(patched, report)
