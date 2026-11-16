# three.js, vendored

WebGL renderer, scene graph, loaders and maths used by the Three.js nodes.

- Upstream: <https://github.com/mrdoob/three.js>
- Version r185, npm `three@0.185.0`
- Licence: MIT, retained verbatim in `LICENSE` beside this file

## Files taken from upstream

| File | Upstream path | Changed |
|---|---|---|
| `controls/TransformControls.mjs` | `examples/jsm/controls/TransformControls.js` | renamed to `.mjs`, import specifiers |
| `curves/NURBSCurve.mjs` | `examples/jsm/curves/NURBSCurve.js` | renamed to `.mjs`, import specifiers |
| `curves/NURBSUtils.mjs` | `examples/jsm/curves/NURBSUtils.js` | renamed to `.mjs`, import specifiers |
| `environments/RoomEnvironment.mjs` | `examples/jsm/environments/RoomEnvironment.js` | renamed to `.mjs`, import specifiers |
| `libs/draco/LICENSE` | Draco's own `LICENSE` | no |
| `libs/draco/gltf/draco_decoder.wasm` | `examples/jsm/libs/draco/gltf/draco_decoder.wasm` | no |
| `libs/draco/gltf/draco_wasm_wrapper.txt` | `examples/jsm/libs/draco/gltf/draco_wasm_wrapper.js` | renamed to `.txt` |
| `libs/fflate.module.mjs` | `examples/jsm/libs/fflate.module.js` | renamed to `.mjs` |
| `libs/meshopt_decoder.LICENSE.md` | meshoptimizer's own `LICENSE.md` | no |
| `libs/meshopt_decoder.module.mjs` | `examples/jsm/libs/meshopt_decoder.module.js` | renamed to `.mjs` |
| `lights/RectAreaLightTexturesLib.mjs` | `examples/jsm/lights/RectAreaLightTexturesLib.js` | renamed to `.mjs`, import specifiers |
| `lights/RectAreaLightUniformsLib.mjs` | `examples/jsm/lights/RectAreaLightUniformsLib.js` | renamed to `.mjs`, import specifiers |
| `loaders/3MFLoader.mjs` | `examples/jsm/loaders/3MFLoader.js` | renamed to `.mjs`, import specifiers |
| `loaders/ColladaLoader.mjs` | `examples/jsm/loaders/ColladaLoader.js` | renamed to `.mjs`, import specifiers |
| `loaders/DRACOLoader.mjs` | `examples/jsm/loaders/DRACOLoader.js` | renamed to `.mjs`, import specifiers |
| `loaders/EXRLoader.mjs` | `examples/jsm/loaders/EXRLoader.js` | renamed to `.mjs`, import specifiers |
| `loaders/FBXLoader.mjs` | `examples/jsm/loaders/FBXLoader.js` | renamed to `.mjs`, import specifiers |
| `loaders/GLTFLoader.mjs` | `examples/jsm/loaders/GLTFLoader.js` | renamed to `.mjs`, import specifiers |
| `loaders/HDRLoader.mjs` | `examples/jsm/loaders/HDRLoader.js` | renamed to `.mjs`, import specifiers |
| `loaders/MTLLoader.mjs` | `examples/jsm/loaders/MTLLoader.js` | renamed to `.mjs`, import specifiers |
| `loaders/OBJLoader.mjs` | `examples/jsm/loaders/OBJLoader.js` | renamed to `.mjs`, import specifiers |
| `loaders/PLYLoader.mjs` | `examples/jsm/loaders/PLYLoader.js` | renamed to `.mjs`, import specifiers |
| `loaders/STLLoader.mjs` | `examples/jsm/loaders/STLLoader.js` | renamed to `.mjs`, import specifiers |
| `loaders/TGALoader.mjs` | `examples/jsm/loaders/TGALoader.js` | renamed to `.mjs`, import specifiers |
| `loaders/collada/ColladaComposer.mjs` | `examples/jsm/loaders/collada/ColladaComposer.js` | renamed to `.mjs`, import specifiers |
| `loaders/collada/ColladaParser.mjs` | `examples/jsm/loaders/collada/ColladaParser.js` | renamed to `.mjs`, import specifiers |
| `postprocessing/BokehPass.mjs` | `examples/jsm/postprocessing/BokehPass.js` | renamed to `.mjs`, import specifiers |
| `postprocessing/EffectComposer.mjs` | `examples/jsm/postprocessing/EffectComposer.js` | renamed to `.mjs`, import specifiers |
| `postprocessing/MaskPass.mjs` | `examples/jsm/postprocessing/MaskPass.js` | renamed to `.mjs`, import specifiers |
| `postprocessing/OutputPass.mjs` | `examples/jsm/postprocessing/OutputPass.js` | renamed to `.mjs`, import specifiers |
| `postprocessing/Pass.mjs` | `examples/jsm/postprocessing/Pass.js` | renamed to `.mjs`, import specifiers |
| `postprocessing/RenderPass.mjs` | `examples/jsm/postprocessing/RenderPass.js` | renamed to `.mjs`, import specifiers |
| `postprocessing/SMAAPass.mjs` | `examples/jsm/postprocessing/SMAAPass.js` | renamed to `.mjs`, import specifiers |
| `postprocessing/ShaderPass.mjs` | `examples/jsm/postprocessing/ShaderPass.js` | renamed to `.mjs`, import specifiers |
| `postprocessing/UnrealBloomPass.mjs` | `examples/jsm/postprocessing/UnrealBloomPass.js` | renamed to `.mjs`, import specifiers |
| `shaders/BokehShader.mjs` | `examples/jsm/shaders/BokehShader.js` | renamed to `.mjs`, import specifiers |
| `shaders/CopyShader.mjs` | `examples/jsm/shaders/CopyShader.js` | renamed to `.mjs`, import specifiers |
| `shaders/LuminosityHighPassShader.mjs` | `examples/jsm/shaders/LuminosityHighPassShader.js` | renamed to `.mjs`, import specifiers |
| `shaders/OutputShader.mjs` | `examples/jsm/shaders/OutputShader.js` | renamed to `.mjs`, import specifiers |
| `shaders/SMAAShader.mjs` | `examples/jsm/shaders/SMAAShader.js` | renamed to `.mjs`, import specifiers |
| `three.core.mjs` | `build/three.core.js` | renamed to `.mjs` |
| `three.module.mjs` | `build/three.module.js` | renamed to `.mjs`, import specifiers |
| `utils/BufferGeometryUtils.mjs` | `examples/jsm/utils/BufferGeometryUtils.js` | renamed to `.mjs`, import specifiers |
| `utils/SkeletonUtils.mjs` | `examples/jsm/utils/SkeletonUtils.js` | renamed to `.mjs`, import specifiers |

Every upstream `.js` module is kept here as `.mjs`. Every file marked "import specifiers" imports
`three`, `three/addons/...` or a relative `.js` path upstream, and each of those specifiers is
rewritten to the relative path of the `.mjs` copy in this directory, which is why the upstream
`examples/jsm/` layout is mirrored here rather than flattened. Nothing else in any file is
altered, and a file marked "renamed to `.mjs`" alone is byte for byte the published file.

`three.module.mjs` re-exports `three.core.mjs`, so both files are required and both are loaded
from this directory. Nothing is fetched over the network at runtime.

## fflate, bundled inside three.js

`libs/fflate.module.mjs` is fflate 0.8.2 by Arjun Barrett, MIT, carried inside the three.js
distribution and taken with it. Its licence and copyright are in the banner at the top of the
file. `FBXLoader` and `3MFLoader` read compressed data through it.

- Upstream: <https://github.com/101arrowz/fflate>

## meshoptimizer, bundled inside three.js

`libs/meshopt_decoder.module.mjs` is the meshoptimizer 1.1 decoder by Arseny Kapoulkine, MIT,
carried inside the three.js distribution and taken with it. Its licence is
`libs/meshopt_decoder.LICENSE.md`. `GLTFLoader` reads `EXT_meshopt_compression` models through
it. Its WebAssembly is embedded in the file.

- Upstream: <https://github.com/zeux/meshoptimizer>

## Draco, bundled inside three.js

`libs/draco/gltf/` is Google's Draco decoder, the glTF build three.js r185 carries, Apache
License 2.0, retained verbatim in `libs/draco/LICENSE`. `DRACOLoader` fetches the wrapper and the
WebAssembly from this folder to read `KHR_draco_mesh_compression` models. The wrapper is byte for
byte the published file, renamed to `.txt`. The asm.js fallback build is not taken.

- Upstream: <https://github.com/google/draco>

## Not taken

`examples/jsm/controls/OrbitControls.js` is not taken. Camera control is `createOrbitControls`
in `web/threejs/runtime.mjs`.
