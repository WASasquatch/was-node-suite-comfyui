"""
Canvas Parser for WAS Content Viewer.

Handles both:
- INPUT: IMAGE tensors -> saves to input files (persistent), returns canvas marker JSON for frontend
- OUTPUT: Canvas composite (base64 PNG) -> converts to IMAGE tensor for backend

"""

import base64
import io
import os
import json
import uuid
import hashlib

from .base_parser import BaseParser


class CanvasParser(BaseParser):
    """Canvas parser for IMAGE tensor input and composite output."""

    PARSER_NAME = "canvas"
    PARSER_PRIORITY = 100  # High priority - check before generic parsers

    # Markers for content identification
    CANVAS_MARKER = "$WAS_CANVAS$"
    OUTPUT_MARKER = "$WAS_CANVAS_OUTPUT$"
    CANVAS_TYPE = "canvas_composer"

    @classmethod
    def detect_input(cls, content) -> bool:
        """Check if content contains IMAGE tensors that should be displayed in canvas view."""
        if content is None:
            return False

        items = content if isinstance(content, (list, tuple)) else [content]

        for item in items:
            if cls._is_image_tensor(item):
                return True
        return False

    @classmethod
    def handle_input(cls, content, logger=None) -> dict:
        """
        Process IMAGE tensors and prepare them for canvas view display.

        Saves tensors as temp PNG files and returns canvas marker JSON.
        """

        items = content if isinstance(content, (list, tuple)) else [content]

        session_id = str(uuid.uuid4())[:8]
        image_files = []

        for item in items:
            if cls._is_image_tensor(item):
                files = cls._tensor_to_input_files(item, session_id, logger)
                image_files.extend(files)

        if not image_files:
            return None

        canvas_data = {
            "type": "canvas_composer",
            "images": image_files,
            "session_id": session_id,
            "count": len(image_files),
        }

        display_content = cls.CANVAS_MARKER + json.dumps(canvas_data)
        content_hash = f"canvas_{session_id}_{len(image_files)}"

        if logger:
            logger.info(
                f"[Canvas Parser] Processed {len(image_files)} images for canvas view"
            )

        return {
            "display_content": display_content,
            "output_values": list(items),
            "content_hash": content_hash,
        }

    @classmethod
    def detect_output(cls, content: str) -> bool:
        """Check if content is canvas composite output (base64 image)."""
        if not isinstance(content, str):
            return False
        return content.startswith(cls.OUTPUT_MARKER)

    @classmethod
    def parse_output(cls, content: str, logger=None) -> dict:
        """Parse canvas composite output and convert to IMAGE tensor."""
        import torch
        import numpy as np
        from PIL import Image

        base64_data = content[len(cls.OUTPUT_MARKER) :]

        if base64_data.startswith("data:"):
            base64_data = base64_data.split(",", 1)[1]

        try:
            image_bytes = base64.b64decode(base64_data)
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")

            img_array = np.array(pil_img).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array).unsqueeze(0)

            if logger:
                logger.info(
                    f"[Canvas Parser] Converted composite to IMAGE tensor: {img_tensor.shape}"
                )

            return {
                "output_values": [img_tensor],
                "display_text": f"Canvas Output: {pil_img.width}x{pil_img.height} RGBA",
                "content_hash": f"canvas_output_{pil_img.width}x{pil_img.height}_{hash(base64_data[:100]) & 0xFFFFFFFF}",
            }

        except Exception as e:
            if logger:
                logger.error(f"[Canvas Parser] Failed to convert composite: {e}")
            return None

    @staticmethod
    def _is_image_tensor(item) -> bool:
        """Check if item is an IMAGE tensor (torch tensor with shape [B,H,W,C] or [H,W,C])."""
        if item is None:
            return False
        if hasattr(item, "shape") and hasattr(item, "cpu"):
            shape = item.shape
            if len(shape) == 4 and shape[-1] in (1, 3, 4):
                return True
            if len(shape) == 3 and shape[-1] in (1, 3, 4):
                return True
        return False

    @staticmethod
    def _tensor_to_input_files(tensor, session_id, logger=None):
        """Convert IMAGE tensor to input PNG files (persists across restarts)."""
        import numpy as np
        from PIL import Image
        import folder_paths

        input_dir = folder_paths.get_input_directory()
        subdir = f"was_viewer_{session_id}"
        full_subdir = os.path.join(input_dir, subdir)
        os.makedirs(full_subdir, exist_ok=True)

        files = []

        if len(tensor.shape) == 3:
            tensor = tensor.unsqueeze(0)

        batch_size = tensor.shape[0]

        for idx in range(batch_size):
            img_array = tensor[idx].cpu().numpy()
            img_array = (img_array * 255).astype(np.uint8)

            if img_array.shape[-1] == 4:
                pil_img = Image.fromarray(img_array, mode="RGBA")
                pil_img = CanvasParser._trim_transparency(pil_img)
            elif img_array.shape[-1] == 3:
                pil_img = Image.fromarray(img_array, mode="RGB")
            else:
                pil_img = Image.fromarray(img_array)

            img_hash = hashlib.md5(img_array.tobytes(), usedforsecurity=False).hexdigest()[:12]
            filename = f"{idx:04d}_{img_hash}.png"
            filepath = os.path.join(full_subdir, filename)

            if not os.path.exists(filepath):
                pil_img.save(filepath, format="PNG")
                if logger:
                    logger.debug(f"[Canvas Parser] Saved: {filepath}")

            files.append({"filename": filename, "subfolder": subdir, "type": "input"})

        return files

    @staticmethod
    def _trim_transparency(pil_img, padding=0):
        """Trim transparent pixels from image edges."""
        import numpy as np

        if pil_img.mode != "RGBA":
            return pil_img

        alpha = np.array(pil_img.split()[-1])
        rows = np.any(alpha > 0, axis=1)
        cols = np.any(alpha > 0, axis=0)

        if not rows.any() or not cols.any():
            return pil_img

        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        rmin = max(0, rmin - padding)
        rmax = min(pil_img.height - 1, rmax + padding)
        cmin = max(0, cmin - padding)
        cmax = min(pil_img.width - 1, cmax + padding)

        return pil_img.crop((cmin, rmin, cmax + 1, rmax + 1))

    @classmethod
    def detect_state(cls, state_data: dict) -> bool:
        """Check if state data contains canvas dataUrl."""
        return isinstance(state_data, dict) and "dataUrl" in state_data

    @classmethod
    def parse_state(cls, state_data: dict, logger=None) -> dict:
        """
        Parse canvas state containing composed image dataUrl.

        Returns:
            dict with keys: image (IMAGE tensor), mask (MASK tensor), width, height,
                           display_content (JSON for UI)
            or None if invalid
        """
        import torch
        import numpy as np
        from PIL import Image

        if not isinstance(state_data, dict) or "dataUrl" not in state_data:
            return None

        try:
            data_url = state_data["dataUrl"]
            if "," in data_url:
                b64_data = data_url.split(",", 1)[1]
            else:
                b64_data = data_url

            img_bytes = base64.b64decode(b64_data)
            pil_img = Image.open(io.BytesIO(img_bytes))

            if pil_img.mode == "RGBA":
                alpha_channel = pil_img.split()[3]
                alpha_array = np.array(alpha_channel).astype(np.float32) / 255.0
                alpha_mask = torch.from_numpy(alpha_array).unsqueeze(0)
                rgb_img = Image.new("RGB", pil_img.size, (0, 0, 0))
                rgb_img.paste(pil_img, mask=alpha_channel)
                pil_img = rgb_img
            else:
                alpha_mask = torch.ones((1, pil_img.height, pil_img.width))
                if pil_img.mode != "RGB":
                    pil_img = pil_img.convert("RGB")

            img_array = np.array(pil_img).astype(np.float32) / 255.0
            composed_image = torch.from_numpy(img_array).unsqueeze(0)

            if logger:
                logger.info(
                    f"[Canvas Parser] Loaded composed image from state: {pil_img.width}x{pil_img.height}"
                )

            canvas_data = {
                "type": cls.CANVAS_TYPE,
                "images": [],
                "count": 0,
                "has_output": True,
            }

            return {
                "image": composed_image,
                "mask": alpha_mask,
                "width": pil_img.width,
                "height": pil_img.height,
                "display_content": json.dumps(canvas_data),
                "content_hash": "composed",
            }

        except Exception as e:
            if logger:
                logger.error(f"[Canvas Parser] Error decoding state: {e}")
            return None

    @classmethod
    def detect_display_content(cls, content) -> bool:
        """Check if content contains IMAGE tensors for display."""
        if content is None:
            return False
        items = content if isinstance(content, (list, tuple)) else [content]
        for item in items:
            if cls._is_image_tensor(item):
                return True
        return False

    @classmethod
    def prepare_display(cls, content, logger=None) -> dict:
        """
        Convert IMAGE tensors to base64 data URLs for canvas display.

        Returns:
            dict with keys: display_content (JSON string), count, content_hash
        """
        import numpy as np
        from PIL import Image

        images = content if isinstance(content, (list, tuple)) else [content]
        base64_images = []

        try:
            for img_batch in images:
                if img_batch is None:
                    continue

                if hasattr(img_batch, "shape"):
                    if len(img_batch.shape) == 4:
                        for i in range(img_batch.shape[0]):
                            img_array = (
                                img_batch[i].cpu().numpy()
                                if hasattr(img_batch[i], "cpu")
                                else img_batch[i]
                            )
                            img_array = (img_array * 255).astype(np.uint8)
                            pil_img = Image.fromarray(img_array)

                            buffer = io.BytesIO()
                            pil_img.save(buffer, format="PNG")
                            b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
                            base64_images.append(f"data:image/png;base64,{b64}")
                    elif len(img_batch.shape) == 3:
                        img_array = (
                            img_batch.cpu().numpy()
                            if hasattr(img_batch, "cpu")
                            else img_batch
                        )
                        img_array = (img_array * 255).astype(np.uint8)
                        pil_img = Image.fromarray(img_array)

                        buffer = io.BytesIO()
                        pil_img.save(buffer, format="PNG")
                        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
                        base64_images.append(f"data:image/png;base64,{b64}")
        except Exception as e:
            if logger:
                logger.error(f"[Canvas Parser] Error converting images: {e}")

        canvas_data = {
            "type": cls.CANVAS_TYPE,
            "images": base64_images,
            "count": len(base64_images),
        }

        if logger:
            logger.info(f"[Canvas Parser] Processed {len(base64_images)} images")

        return {
            "display_content": json.dumps(canvas_data),
            "count": len(base64_images),
            "content_hash": str(len(base64_images)),
        }

    @classmethod
    def get_default_outputs(cls, content, output_types: list, logger=None) -> tuple:
        """
        Get default IMAGE and MASK outputs based on input content.

        Returns:
            tuple matching output_types, or None if not applicable
        """
        import torch

        if not output_types or set(output_types) != {"IMAGE", "MASK"}:
            return None

        images = content if isinstance(content, (list, tuple)) else [content]

        if images and len(images) > 0 and images[0] is not None:
            output_image = images[0]
            if hasattr(images[0], "shape") and len(images[0].shape) >= 3:
                h, w = images[0].shape[1], images[0].shape[2]
                output_mask = torch.ones((1, h, w))
            else:
                output_mask = torch.ones((1, 64, 64))
        else:
            output_image = torch.zeros((1, 64, 64, 3))
            output_mask = torch.ones((1, 64, 64))

        if output_types[0] == "IMAGE":
            return (output_image, output_mask)
        else:
            return (output_mask, output_image)
