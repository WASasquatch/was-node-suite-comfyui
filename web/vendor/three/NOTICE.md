# three.js, vendored

WebGL renderer, scene graph, loaders and maths used by the Three.js nodes.

- Upstream: <https://github.com/mrdoob/three.js>
- Version r185, npm `three@0.185.0`
- Licence: MIT, retained verbatim in `LICENSE` beside this file

## Files taken from upstream

| File | Upstream path | Changed |
|---|---|---|
| `controls/TransformControls.js` | `examples/jsm/controls/TransformControls.js` | import specifiers |
| `curves/NURBSCurve.js` | `examples/jsm/curves/NURBSCurve.js` | import specifiers |
| `curves/NURBSUtils.js` | `examples/jsm/curves/NURBSUtils.js` | import specifiers |
| `environments/RoomEnvironment.js` | `examples/jsm/environments/RoomEnvironment.js` | import specifiers |
| `libs/draco/LICENSE` | Draco's own `LICENSE` | no |
| `libs/draco/gltf/draco_decoder.wasm` | `examples/jsm/libs/draco/gltf/draco_decoder.wasm` | no |
| `libs/draco/gltf/draco_wasm_wrapper.txt` | `examples/jsm/libs/draco/gltf/draco_wasm_wrapper.js` | renamed to `.txt` |
| `libs/fflate.module.js` | `examples/jsm/libs/fflate.module.js` | no |
| `libs/meshopt_decoder.LICENSE.md` | meshoptimizer's own `LICENSE.md` | no |
| `libs/meshopt_decoder.module.js` | `examples/jsm/libs/meshopt_decoder.module.js` | no |
| `lights/RectAreaLightTexturesLib.js` | `examples/jsm/lights/RectAreaLightTexturesLib.js` | import specifiers |
| `lights/RectAreaLightUniformsLib.js` | `examples/jsm/lights/RectAreaLightUniformsLib.js` | import specifiers |
| `loaders/3MFLoader.js` | `examples/jsm/loaders/3MFLoader.js` | import specifiers |
| `loaders/ColladaLoader.js` | `examples/jsm/loaders/ColladaLoader.js` | import specifiers |
| `loaders/DRACOLoader.js` | `examples/jsm/loaders/DRACOLoader.js` | import specifiers |
| `loaders/EXRLoader.js` | `examples/jsm/loaders/EXRLoader.js` | import specifiers |
| `loaders/FBXLoader.js` | `examples/jsm/loaders/FBXLoader.js` | import specifiers |
| `loaders/GLTFLoader.js` | `examples/jsm/loaders/GLTFLoader.js` | import specifiers |
| `loaders/HDRLoader.js` | `examples/jsm/loaders/HDRLoader.js` | import specifiers |
| `loaders/MTLLoader.js` | `examples/jsm/loaders/MTLLoader.js` | import specifiers |
| `loaders/OBJLoader.js` | `examples/jsm/loaders/OBJLoader.js` | import specifiers |
| `loaders/PLYLoader.js` | `examples/jsm/loaders/PLYLoader.js` | import specifiers |
| `loaders/STLLoader.js` | `examples/jsm/loaders/STLLoader.js` | import specifiers |
| `loaders/TGALoader.js` | `examples/jsm/loaders/TGALoader.js` | import specifiers |
| `loaders/collada/ColladaComposer.js` | `examples/jsm/loaders/collada/ColladaComposer.js` | import specifiers |
| `loaders/collada/ColladaParser.js` | `examples/jsm/loaders/collada/ColladaParser.js` | import specifiers |
| `postprocessing/BokehPass.js` | `examples/jsm/postprocessing/BokehPass.js` | import specifiers |
| `postprocessing/EffectComposer.js` | `examples/jsm/postprocessing/EffectComposer.js` | import specifiers |
| `postprocessing/MaskPass.js` | `examples/jsm/postprocessing/MaskPass.js` | import specifiers |
| `postprocessing/OutputPass.js` | `examples/jsm/postprocessing/OutputPass.js` | import specifiers |
| `postprocessing/Pass.js` | `examples/jsm/postprocessing/Pass.js` | import specifiers |
| `postprocessing/RenderPass.js` | `examples/jsm/postprocessing/RenderPass.js` | import specifiers |
| `postprocessing/SMAAPass.js` | `examples/jsm/postprocessing/SMAAPass.js` | import specifiers |
| `postprocessing/ShaderPass.js` | `examples/jsm/postprocessing/ShaderPass.js` | import specifiers |
| `postprocessing/UnrealBloomPass.js` | `examples/jsm/postprocessing/UnrealBloomPass.js` | import specifiers |
| `shaders/BokehShader.js` | `examples/jsm/shaders/BokehShader.js` | import specifiers |
| `shaders/CopyShader.js` | `examples/jsm/shaders/CopyShader.js` | import specifiers |
| `shaders/LuminosityHighPassShader.js` | `examples/jsm/shaders/LuminosityHighPassShader.js` | import specifiers |
| `shaders/OutputShader.js` | `examples/jsm/shaders/OutputShader.js` | import specifiers |
| `shaders/SMAAShader.js` | `examples/jsm/shaders/SMAAShader.js` | import specifiers |
| `three.core.js` | `build/three.core.js` | no |
| `three.module.js` | `build/three.module.js` | no |
| `utils/BufferGeometryUtils.js` | `examples/jsm/utils/BufferGeometryUtils.js` | import specifiers |
| `utils/SkeletonUtils.js` | `examples/jsm/utils/SkeletonUtils.js` | import specifiers |

Every file marked "import specifiers" imports `three` and `three/addons/...` upstream. Each of
those specifiers is rewritten to the relative path of the copy in this directory, which is why
the upstream `examples/jsm/` layout is mirrored here rather than flattened. Nothing else in any
file is altered.

`three.module.js` re-exports `three.core.js`, so both files are required and both are loaded
from this directory. Nothing is fetched over the network at runtime.

## fflate, bundled inside three.js

`libs/fflate.module.js` is fflate 0.8.2 by Arjun Barrett, MIT, carried inside the three.js
distribution and taken with it. Its licence and copyright are in the banner at the top of the
file. `FBXLoader` and `3MFLoader` read compressed data through it.

- Upstream: <https://github.com/101arrowz/fflate>

## meshoptimizer, bundled inside three.js

`libs/meshopt_decoder.module.js` is the meshoptimizer 1.1 decoder by Arseny Kapoulkine, MIT,
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
in `web/threejs/runtime.js`.
