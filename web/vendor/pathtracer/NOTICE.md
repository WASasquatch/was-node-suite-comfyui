# three-gpu-pathtracer, vendored

GPU path tracer and BVH builder used by the Three Path Trace Render node.

- Upstream: <https://github.com/gkjohnson/three-gpu-pathtracer>
- Version 0.0.24, npm `three-gpu-pathtracer@0.0.24`
- Licence: MIT, retained verbatim in `LICENSE.pathtracer` beside this file

## three-mesh-bvh, its dependency

The path tracer traces against a bounding volume hierarchy built by a separate package, taken
with it.

- Upstream: <https://github.com/gkjohnson/three-mesh-bvh>
- Version 0.9.14, npm `three-mesh-bvh@0.9.14`
- Licence: MIT, retained verbatim in `LICENSE.mesh-bvh` beside this file

## Files taken from upstream

| File | Upstream path | Changed |
|---|---|---|
| `index.module.mjs` | `three-gpu-pathtracer@0.0.24/build/index.module.js` | renamed to `.mjs`, import specifiers |
| `mesh-bvh.mjs` | `three-mesh-bvh@0.9.14/build/index.module.js` | renamed to `.mjs`, import specifiers |

Both are the published single-file builds, kept here as `.mjs`. Each imports the bare
specifiers `three`, `three-mesh-bvh` and `three/examples/jsm/postprocessing/Pass.js`, and each
of those is rewritten to the relative path of the `.mjs` copy in `web/vendor/`. Nothing else in
either file is altered. Nothing is fetched over the network at runtime.

`xatlas-web` is listed upstream as a peer dependency. Neither build imports it, and the
lightmap baker that uses it is not taken, so it is not present.

## Not taken

The upstream `src/` tree, its examples and its test fixtures are not taken. The two builds
above carry everything the node loads, and both are loaded on demand rather than with the
rest of the pack.
