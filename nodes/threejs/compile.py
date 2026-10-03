"""A Three.js scene written out as a page that runs on its own."""

from __future__ import annotations

import os
from pathlib import Path

from comfy_api.latest import io

from ...modules.compat.types import THREE_APP
from ...modules.io import rooted
from ...modules.log import get_logger
from ...modules.threejs import compile as bundler
from ...modules.threejs.spec import refuse_script, require_spec
from ...modules.util import filenames, sandbox

REQUIRES = "threejs"

logger = get_logger("nodes.threejs")

SUFFIX = ".zip"


class ThreeCompile(io.ComfyNode):
    """Write the scene out as a self-contained archive."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASThreeCompile",
            display_name="Three Compile",
            search_aliases=[
                "WASThreeCompile",
                "Three Compile",
                "export scene",
                "standalone",
                "web page",
            ],
            category="WAS Suite/Three",
            description=(
                "Write the scene out as a zip holding a web page that runs on its own. Unpack "
                "it and open index.html: the scene draws, and the camera can be orbited, with "
                "no ComfyUI and nothing fetched over the network. Three.js, the scene "
                "description and every texture go in the archive, and each texture address is "
                "rewritten to point at its copy. A scene using Three Custom Geometry, Custom "
                "Material, Custom Object, Custom Update or Script Module carries that "
                "JavaScript into the page, so the archive is code as well as data, and is "
                "refused unless threejs.allow_scripts is on in config.yaml."
            ),
            is_output_node=True,
            inputs=[
                THREE_APP.Input(
                    "app",
                    tooltip="The scene, camera and renderer settings, from Three App.",
                ),
                io.String.Input(
                    "filename_prefix",
                    default="three_scene",
                    multiline=False,
                    tooltip=(
                        "Leading part of the file name, as `three_scene`. A `/` in it makes "
                        "sub-folders under the chosen root."
                    ),
                ),
                io.Combo.Input(
                    "root",
                    options=rooted.options(),
                    default=rooted.DEFAULT,
                    tooltip=(
                        "Which of ComfyUI's folders the archive is written into, as `output` "
                        "or `temp`."
                    ),
                ),
                io.String.Input(
                    "title",
                    default="Three.js scene",
                    multiline=False,
                    tooltip="Title the page carries, shown in the browser tab. `Three.js scene` by default.",
                ),
            ],
            outputs=[
                io.String.Output(
                    display_name="file",
                    tooltip="Where the archive was written, as a path.",
                ),
                io.Int.Output(
                    display_name="entries",
                    tooltip="How many files went into the archive, textures included.",
                ),
            ],
        )

    @classmethod
    def execute(cls, app, filename_prefix, root, title) -> io.NodeOutput:
        """Pack the scene and write it.

        Raises:
            ValueError: ``app`` is not an app descriptor, or carries javascript while
                ``threejs.allow_scripts`` is off.
            FileNotFoundError: A file the page needs is missing from the pack.
            PathNotAllowed: The name resolved outside every permitted write root.
            OSError: The archive could not be written.
        """
        require_spec(app, "app")
        refuse_script(app, "Three Compile")

        web_root = Path(__file__).resolve().parents[2] / "web"
        body, names = bundler.bundle(app, str(title), web_root)

        below, _, leaf = (filename_prefix or "").replace("\\", "/").rpartition("/")
        directory = rooted.destination(root, below)
        if not directory.exists():
            logger.warning("the path `%s` doesn't exist! Creating it...", directory)
            os.makedirs(directory, exist_ok=True)

        filename = filenames.generate_filename(directory, leaf or "three_scene", "_", 5, SUFFIX, "")
        target = sandbox.resolve_write_file(directory, filename)
        target.write_bytes(body)

        logger.info(
            "Three Compile wrote %s: %s, %.1f KB",
            target, bundler.written_files(names), len(body) / 1024.0,
        )
        return io.NodeOutput(str(target), len(names))
