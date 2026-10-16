"""A Three.js scene packed as a page that runs on its own.

Every texture goes in the archive and its address is rewritten to point at the copy.
"""

from __future__ import annotations

__all__ = ["PAGE_NAME", "bundle", "written_files"]

import copy
import html
import json
import re
import zipfile
from pathlib import Path

from ..log import get_logger
from ..interface import three_asset

logger = get_logger("threejs.compile")

#: The file a reader opens.
PAGE_NAME = "index.html"

#: Where the pictures and models go inside the archive.
TEXTURE_DIR = "textures"
MODEL_DIR = "models"

#: The folder and suffix each held content type is written under.
SUFFIXES = {
    "image/png": (TEXTURE_DIR, ".png"),
    "image/jpeg": (TEXTURE_DIR, ".jpg"),
    "image/bmp": (TEXTURE_DIR, ".bmp"),
    "image/gif": (TEXTURE_DIR, ".gif"),
    "image/webp": (TEXTURE_DIR, ".webp"),
    "image/x-tga": (TEXTURE_DIR, ".tga"),
    "application/octet-stream": (MODEL_DIR, ".bin"),
    "model/gltf-binary": (MODEL_DIR, ".glb"),
    "model/gltf+json": (MODEL_DIR, ".gltf"),
    "text/plain": (MODEL_DIR, ".obj"),
    "model/mtl": (MODEL_DIR, ".mtl"),
    "model/stl": (MODEL_DIR, ".stl"),
    "model/vnd.collada+xml": (MODEL_DIR, ".dae"),
    "model/fbx": (MODEL_DIR, ".fbx"),
    "model/ply": (MODEL_DIR, ".ply"),
    "model/3mf": (MODEL_DIR, ".3mf"),
}

#: Files copied out of the pack. The archive mirrors ``web/`` so the relative import inside
#: ``runtime.js`` resolves in the unpacked folder exactly as it does in the pack.
CARRIED = (
    "vendor/three/three.module.js",
    "vendor/three/three.core.js",
    "vendor/three/LICENSE",
    "vendor/three/loaders/GLTFLoader.js",
    "vendor/three/loaders/OBJLoader.js",
    "vendor/three/loaders/STLLoader.js",
    "vendor/three/utils/BufferGeometryUtils.js",
    "vendor/three/utils/SkeletonUtils.js",
    "threejs/runtime.js",
)

#: Model suffix -> the further files its loader needs. Only the ones a scene actually uses
#: are written, which keeps an archive holding one .glb from carrying every parser.
BY_FORMAT = {
    "dae": (
        "vendor/three/loaders/ColladaLoader.js",
        "vendor/three/loaders/TGALoader.js",
        "vendor/three/loaders/collada/ColladaParser.js",
        "vendor/three/loaders/collada/ColladaComposer.js",
    ),
    "fbx": (
        "vendor/three/loaders/FBXLoader.js",
        "vendor/three/libs/fflate.module.js",
        "vendor/three/curves/NURBSCurve.js",
        "vendor/three/curves/NURBSUtils.js",
    ),
    "ply": ("vendor/three/loaders/PLYLoader.js",),
    "obj": ("vendor/three/loaders/MTLLoader.js",),
    "3mf": (
        "vendor/three/loaders/3MFLoader.js",
        "vendor/three/libs/fflate.module.js",
    ),
}

#: Files every effect chain needs, and then the ones each kind of effect needs on top.
EFFECT_BASE = (
    "vendor/three/postprocessing/EffectComposer.js",
    "vendor/three/postprocessing/Pass.js",
    "vendor/three/postprocessing/MaskPass.js",
    "vendor/three/postprocessing/ShaderPass.js",
    "vendor/three/postprocessing/RenderPass.js",
    "vendor/three/postprocessing/OutputPass.js",
    "vendor/three/shaders/CopyShader.js",
    "vendor/three/shaders/OutputShader.js",
)

#: Environment source -> the file its loader needs. An environment read from a picture off the
#: wire needs nothing beyond three itself.
#: What an area light is shaded from. Large, so only a scene holding one carries them.
AREA_LIGHT = (
    "vendor/three/lights/RectAreaLightUniformsLib.js",
    "vendor/three/lights/RectAreaLightTexturesLib.js",
)

BY_ENVIRONMENT = {
    "studio room": ("vendor/three/environments/RoomEnvironment.js",),
    "hdr": ("vendor/three/loaders/HDRLoader.js",),
    "exr": ("vendor/three/loaders/EXRLoader.js",),
}

BY_EFFECT = {
    "Bloom": (
        "vendor/three/postprocessing/UnrealBloomPass.js",
        "vendor/three/shaders/LuminosityHighPassShader.js",
    ),
    "DepthOfField": (
        "vendor/three/postprocessing/BokehPass.js",
        "vendor/three/shaders/BokehShader.js",
    ),
    "Antialias": (
        "vendor/three/postprocessing/SMAAPass.js",
        "vendor/three/shaders/SMAAShader.js",
    ),
}


#: A CSS colour the page's background may be: ``#`` and 3 to 8 hex digits, or a plain name.
COLOUR = re.compile(r"#[0-9a-fA-F]{3,8}|[a-zA-Z]{3,20}")

PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  html, body {{ margin: 0; height: 100%; background: {background}; overflow: hidden; }}
  canvas {{ display: block; width: 100%; height: 100%; }}
</style>
</head>
<body>
<canvas id="stage"></canvas>
<script type="module">
import * as THREE from "./vendor/three/three.module.js";
import {{ createOrbitControls, createRuntime, toneMappingConstant }} from "./threejs/runtime.js";

const app = await (await fetch("./scene.json")).json();
const canvas = document.getElementById("stage");
const runtime = createRuntime(canvas, null, {{ network: {network} }});

const renderer = new THREE.WebGLRenderer({{
    canvas,
    antialias: app.params?.antialias !== false && !app.deps?.effects,
    alpha: app.deps?.scene?.params?.backgroundMode === "transparent",
}});
runtime.ctx.renderer = renderer;
runtime.ctx.timelineSeconds = Number(app.params?.loopSeconds) || 4;
runtime.ctx.duration = runtime.ctx.timelineSeconds;
runtime.ctx.timeOrigin = 0;
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, Number(app.params?.pixelRatioLimit) || 2));
renderer.shadowMap.enabled = app.params?.shadows !== false;
renderer.toneMapping = toneMappingConstant(app.params?.toneMapping);
renderer.toneMappingExposure = Number(app.params?.exposure) || 1;

runtime.ctx.shadowMapSize = Number(app.params?.shadowMapSize) || 2048;
const scene = await runtime.buildScene(app.deps?.scene);
const camera = runtime.buildCamera(app.deps?.camera, window.innerWidth / window.innerHeight);
runtime.attachCameraTrack(camera, app.deps?.camera, scene);
const composer = await runtime.createComposer(app.deps?.effects ?? null, {{
    renderer, scene, camera,
    width: Math.max(2, window.innerWidth), height: Math.max(2, window.innerHeight),
}});
const controls = app.params?.orbitControls ? createOrbitControls(camera, canvas) : null;
if (controls) {{
    controls.autoRotate = Boolean(app.params?.autoRotate);
    controls.autoRotateSpeed = Number(app.params?.autoRotateSpeed) || 1;
    controls.sync();
}}

const resize = () => {{
    const width = Math.max(2, window.innerWidth);
    const height = Math.max(2, window.innerHeight);
    renderer.setSize(width, height, false);
    composer?.setSize(width, height);
    runtime.updateCameraAspect(camera, width / height);
}};
window.addEventListener("resize", resize);
resize();

let lastTime = null;
renderer.setAnimationLoop((milliseconds) => {{
    const time = milliseconds * 0.001;
    if (lastTime === null) lastTime = time;
    const delta = Math.min(0.1, Math.max(0, time - lastTime));
    lastTime = time;
    for (const update of runtime.ctx.updateFunctions) update({{ time, delta, ctx: runtime.ctx }});
    controls?.update();
    runtime.updateShaderUniforms(time, window.innerWidth, window.innerHeight);
    if (composer) composer.render();
    else renderer.render(scene, camera);
}});
</script>
</body>
</html>
"""


def _localise(spec, textures: dict[str, bytes]):
    """Rewrite one descriptor tree so every texture points at a file in the archive.

    Args:
        spec: A descriptor, or any value inside one.
        textures: Filled in with the archive name of each texture against its bytes.

    Returns:
        The descriptor with texture addresses replaced by archive-relative names.
    """
    if isinstance(spec, list):
        return [_localise(item, textures) for item in spec]
    if not isinstance(spec, dict):
        return spec

    out = {key: _localise(value, textures) for key, value in spec.items()}
    params = out.get("params")
    if not isinstance(params, dict):
        return out

    address = params.get("url")
    if isinstance(address, str) and address:
        moved = _archived(address, textures)
        if moved is not None:
            params["url"] = moved

    # A model names its pictures beside itself, and those go in the archive too.
    resources = params.get("resources")
    if isinstance(resources, dict) and resources:
        params["resources"] = {
            name: (_archived(where, textures) or where)
            for name, where in resources.items()
        }
    return out


def _archived(address: str, textures: dict[str, bytes]) -> str | None:
    """Put one held asset in the archive and answer its name there.

    Args:
        address: The address the browser would otherwise fetch.
        textures: Filled in with the archive name against the bytes.

    Returns:
        The archive-relative name, or ``None`` where the address is not a held asset or the
        asset is no longer held.
    """
    if not address.startswith(three_asset.ROUTE):
        return None
    key = address.partition("key=")[2]
    entry = three_asset.entry_for(key)
    if entry is None:
        logger.warning(
            "an asset is no longer held under key %s, so the page will be missing it; "
            "queue the graph again before compiling",
            key,
        )
        return None
    body, content_type = entry
    folder, suffix = SUFFIXES.get(content_type, (TEXTURE_DIR, ".png"))
    name = f"{folder}/{key}{suffix}"
    textures[name] = body
    return f"./{name}"


def _loaders_for(app: dict) -> tuple[str, ...]:
    """The further loader files a descriptor's model formats need.

    Args:
        app: The app descriptor, after localising.

    Returns:
        Paths under ``web``, each once, in a stable order.
    """
    formats: set[str] = set()
    lights: set[str] = set()
    effects: set[str] = set()
    environments: set[str] = set()

    def walk(node) -> None:
        if isinstance(node, dict):
            if node.get("type") == "ModelFile":
                formats.add(str(node.get("params", {}).get("format", "")).lower())
            if node.get("kind") == "effect":
                effects.add(str(node.get("type", "")))
            if node.get("type") == "AreaLight":
                lights.add("area")
            if node.get("kind") == "environment":
                params = node.get("params", {})
                source = str(params.get("source", ""))
                environments.add(
                    str(params.get("format", "")).lower() if source == "file" else source
                )
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(app)
    wanted: list[str] = []
    for suffix in sorted(formats):
        for name in BY_FORMAT.get(suffix, ()):
            if name not in wanted:
                wanted.append(name)
    if effects:
        for name in EFFECT_BASE:
            if name not in wanted:
                wanted.append(name)
        for kind in sorted(effects):
            for name in BY_EFFECT.get(kind, ()):
                if name not in wanted:
                    wanted.append(name)
    for source in sorted(environments):
        for name in BY_ENVIRONMENT.get(source, ()):
            if name not in wanted:
                wanted.append(name)
    if "area" in lights:
        for name in AREA_LIGHT:
            if name not in wanted:
                wanted.append(name)
    return tuple(wanted)


def bundle(app: dict, title: str, web_root: Path) -> tuple[bytes, list[str]]:
    """Build the archive holding the page and everything it needs.

    Args:
        app: The app descriptor to pack.
        title: The page's title.
        web_root: The pack's ``web`` directory, which the carried files are read from.

    Returns:
        The archive's bytes and the names inside it, in the order they were written.

    Raises:
        FileNotFoundError: A file the page needs is missing from the pack.
    """
    from ..model import network_enabled

    textures: dict[str, bytes] = {}
    localised = _localise(copy.deepcopy(app), textures)
    background = localised.get("deps", {}).get("scene", {}).get("params", {}).get(
        "background", "#111111"
    )
    if not isinstance(background, str) or not COLOUR.fullmatch(background.strip()):
        background = "#111111"

    import io as _io

    buffer = _io.BytesIO()
    names: list[str] = []
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        page = PAGE.format(
            title=html.escape(title or "Three.js scene"),
            background=background.strip(),
            network="true" if network_enabled() else "false",
        )
        archive.writestr(PAGE_NAME, page)
        names.append(PAGE_NAME)

        archive.writestr("scene.json", json.dumps(localised, separators=(",", ":")))
        names.append("scene.json")

        for name in (*CARRIED, *_loaders_for(localised)):
            path = web_root / name
            if not path.is_file():
                raise FileNotFoundError(
                    f"{path} is missing from the pack, so the page would not run. The Three.js "
                    f"files are vendored under web/vendor/three; reinstall the pack if they "
                    f"have been removed."
                )
            archive.writestr(name, path.read_bytes())
            names.append(name)

        for name, body in sorted(textures.items()):
            archive.writestr(name, body)
            names.append(name)

    return buffer.getvalue(), names


def written_files(names: list[str]) -> str:
    """A short account of what went in, for the log.

    Args:
        names: The archive's entry names.

    Returns:
        One line naming the counts.
    """
    pictures = sum(1 for name in names if name.startswith(f"{TEXTURE_DIR}/"))
    meshes = sum(1 for name in names if name.startswith(f"{MODEL_DIR}/"))
    return f"{len(names)} file(s), {pictures} texture(s), {meshes} model(s)"
