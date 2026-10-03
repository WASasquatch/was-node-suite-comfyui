"""EMA-VFI frame interpolation model loading."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.types import EMA_VFI_MODEL
from ...modules.model import frame_interpolation

#: Shown in the checkpoint list when the model folder holds none, so the widget has
#: something to draw.
NO_CHECKPOINT = "put a checkpoint in models/EMA-VFI"


class EMAVFIModelLoader(io.ComfyNode):
    """Build the interpolation network EMA-VFI Frame Interpolation runs on."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        found = frame_interpolation.offered()
        return io.Schema(
            node_id="WASEMAVFIModelLoader",
            display_name="EMA-VFI Video Model Loader",
            search_aliases=[
                "WASEMAVFIModelLoader",
                "EMA-VFI Video Model Loader",
                "EMA-VFI Model Loader",
                "EMA-VFI",
                "frame interpolation",
                "interpolate frames",
                "slow motion",
            ],
            category="WAS Suite/Loaders",
            description=(
                "Build an EMA-VFI network for EMA-VFI Frame Interpolation. The network is "
                "kept for the life of the process, so one loader can feed several nodes "
                "without building it again. The weights are not bundled: with "
                "features.network on the checkpoint is fetched on first use, and with it "
                "off put one in ComfyUI/models/EMA-VFI and restart so it appears in the "
                "list."
            ),
            inputs=[
                io.Combo.Input(
                    "checkpoint",
                    options=found or [NO_CHECKPOINT],
                    tooltip=(
                        "Which EMA-VFI weights to build. The 'small' files are faster and "
                        "less accurate; the '_t' files can land anywhere between two frames "
                        "and are the ones a multiplier above 2 needs. A name not yet on disk "
                        "is fetched on the first run that needs it."
                    ),
                ),
            ],
            outputs=[
                EMA_VFI_MODEL.Output(
                    display_name="ema_vfi_model",
                    tooltip=(
                        "The built network, for the ema_vfi_model input of EMA-VFI Frame "
                        "Interpolation."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, checkpoint) -> io.NodeOutput:
        """Build the network for ``checkpoint``.

        Raises:
            ValueError: No checkpoint is on disk and none can be fetched.
            ModelUnavailable: The chosen checkpoint is not on disk.
        """
        if checkpoint == NO_CHECKPOINT or not checkpoint:
            raise ValueError(
                "EMA-VFI Model Loader has no weights to build. Either set "
                f"{frame_interpolation.NETWORK_FEATURE}: true in config.yaml and let the "
                "first run fetch one, or download a checkpoint from "
                f"{frame_interpolation.DOWNLOAD_PAGE} into ComfyUI/models/EMA-VFI. Either "
                "way, restart ComfyUI so the list is rebuilt."
            )
        built = frame_interpolation.backend(checkpoint)
        return io.NodeOutput(frame_interpolation.Network(backend=built, name=checkpoint))
