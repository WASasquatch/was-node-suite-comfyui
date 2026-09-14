# Third-party code in the pack

The suite is MIT, see [`LICENSE`](../LICENSE). Everything bundled with it keeps its own licence
text in the folder it ships in, and nothing here is fetched from the internet while ComfyUI
runs.

| Component | Licence and copyright | What it is here for | Licence text |
|---|---|---|---|
| [HugeRTE](https://hugerte.org/) 1.0.12 | MIT, Ephox Corporation DBA Tiny Technologies, Inc., and the HugeRTE contributors | The editor **Rich Text Editor** draws on the node, in [`web/vendor/hugerte/`](../web/vendor/hugerte/) | [`license.txt`](../web/vendor/hugerte/license.txt) |
| ⤷ [DOMPurify](https://github.com/cure53/DOMPurify) 3.4.11, compiled into two HugeRTE files | MPL-2.0 **or** Apache-2.0, Dr.-Ing. Mario Heiderich, Cure53 and other contributors | The sanitiser HugeRTE's `xss_sanitization` runs | [`LICENSE-MPL-2.0.txt`](../web/vendor/hugerte/LICENSE-MPL-2.0.txt), [`LICENSE-APACHE-2.0.txt`](../web/vendor/hugerte/LICENSE-APACHE-2.0.txt) |
| [three.js](https://threejs.org/) r185 | MIT, three.js authors | The renderer, scene graph and loaders the Three.js nodes draw with, in [`web/vendor/three/`](../web/vendor/three/). The nodes are off until `features.threejs` is on | [`LICENSE`](../web/vendor/three/LICENSE), [`NOTICE.md`](../web/vendor/three/NOTICE.md) |
| [three-gpu-pathtracer](https://github.com/gkjohnson/three-gpu-pathtracer) 0.0.24 | MIT, Garrett Johnson | The tracer **Three Path Trace Render** renders with, in [`web/vendor/pathtracer/`](../web/vendor/pathtracer/). Loaded only when a frame is traced | [`LICENSE.pathtracer`](../web/vendor/pathtracer/LICENSE.pathtracer), [`NOTICE.md`](../web/vendor/pathtracer/NOTICE.md) |
| ⤷ [three-mesh-bvh](https://github.com/gkjohnson/three-mesh-bvh) 0.9.14 | MIT, Garrett Johnson | The hierarchy the tracer traces against, taken with it | [`LICENSE.mesh-bvh`](../web/vendor/pathtracer/LICENSE.mesh-bvh) |
| [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) | MIT, Shiqi Yu | The face detector **Image Crop Face (YuNet)** uses, as weights converted to safetensors in [`modules/data/models/`](../modules/data/models/) | [`models/LICENSE`](../modules/data/models/LICENSE) |
| [Marigold IID](https://huggingface.co/prs-eth/marigold-iid-appearance-v1-1) | Open RAIL++-M, PRS ETH Zürich | The two token empty prompt embedding the intrinsic maps on **Power Preprocessor** condition on, read out of the checkpoints' own text encoder into [`modules/data/models/`](../modules/data/models/). The checkpoints themselves are downloaded, not bundled | [`models/LICENSE`](../modules/data/models/LICENSE) |
| [EMA-VFI](https://github.com/MCG-NJU/EMA-VFI) | Apache-2.0, Zhang, Zhu, Wang, Chen, Wu and Wang | The network **EMA-VFI Frame Interpolation** runs, vendored in [`modules/vendor/ema_vfi/`](../modules/vendor/ema_vfi/). No weights are bundled | [`LICENSE`](../modules/vendor/ema_vfi/LICENSE), [`NOTICE.md`](../modules/vendor/ema_vfi/NOTICE.md) |
| [Prism.js](https://prismjs.com/) | MIT, Lea Verou | Syntax highlighting in the **Content Viewer** | [`code_scripts/LICENSE`](../web/viewer/views/code_scripts/LICENSE) |
| [KaTeX](https://katex.org/) 0.16.9 | MIT, Khan Academy and other contributors | Maths in the viewer's Markdown view | [`markdown_scripts/LICENSE`](../web/viewer/views/markdown_scripts/LICENSE) |
| [Mermaid](https://mermaid.js.org/) 10.9.5 | MIT, Knut Sveidqvist and contributors | Diagrams in the viewer's Markdown view | [`markdown_scripts/LICENSE`](../web/viewer/views/markdown_scripts/LICENSE) |
| ⤷ [DOMPurify](https://github.com/cure53/DOMPurify) 3.2.4, bundled into `mermaid.min.txt` | MPL-2.0 **or** Apache-2.0, Dr.-Ing. Mario Heiderich, Cure53 and other contributors | Mermaid's own sanitising of the SVG it builds | [`LICENSE-MPL-2.0.txt`](../web/viewer/views/markdown_scripts/LICENSE-MPL-2.0.txt), [`LICENSE-APACHE-2.0.txt`](../web/viewer/views/markdown_scripts/LICENSE-APACHE-2.0.txt) |
| [KaTeX fonts](https://katex.org/) | SIL Open Font License 1.1, Design Science, Inc. and Khan Academy | The six `.woff2` faces KaTeX sets maths in, in [`web/viewer/fonts/`](../web/viewer/fonts/) | [`fonts/LICENSE`](../web/viewer/fonts/LICENSE) |
| [DejaVu](https://github.com/dejavu-fonts/dejavu-fonts) 2.37 | Bitstream Vera and Arev, Bitstream Inc. and Tavmjong Bah, with the DejaVu changes in the public domain | Four faces **Image Draw Text** renders with, in [`modules/data/fonts/dejavu/`](../modules/data/fonts/dejavu/) | [`dejavu/LICENSE`](../modules/data/fonts/dejavu/LICENSE), [`AUTHORS`](../modules/data/fonts/dejavu/AUTHORS) |
| [Liberation](https://github.com/liberationfonts/liberation-fonts) 2.1.5 | SIL Open Font License 1.1, Red Hat, Inc., with digitized data copyright Google Corporation | Four faces metric-compatible with Arial, Times New Roman and Courier New, in [`modules/data/fonts/liberation/`](../modules/data/fonts/liberation/) | [`liberation/LICENSE`](../modules/data/fonts/liberation/LICENSE), [`AUTHORS`](../modules/data/fonts/liberation/AUTHORS) |

Seven of the components are MIT in their own code, and two of those compile in a copy of
DOMPurify, which is dual licensed `MPL-2.0 OR Apache-2.0`. Of the rest, one is Apache-2.0, two
are under the SIL Open Font License, one is Bitstream Vera with Arev, and one is Open RAIL++-M.
All of those permit redistribution. MPL-2.0 is the only copyleft licence here and its conditions
are file level, reaching the two files DOMPurify is compiled into and nothing beside them;
complying with Apache-2.0 instead satisfies the same component. Open RAIL++-M covers 8 KB of
precomputed numbers rather than any model, and is the only licence here carrying use
restrictions. The two font licences ask that the licence and copyright travel with the files,
which is what the `LICENSE` and `AUTHORS` beside each family are, and the OFL reserves the
Liberation names against modified copies. These are unmodified.

Counting every file rather than every component, the pack carries 128 third-party files, 26.5 MB
of the repository: browser libraries, fonts, the eight face cascades, two sets of weights and
one network.

## Verifying a copy

`web/vendor/MANIFEST.json` records a SHA-256 for every file under `web/vendor/`, with the npm
package and version each directory was taken from. Fetch that package and compare digests to
confirm a copy is unmodified.

## Where DOMPurify sits

DOMPurify is not a file of its own. It is compiled into three files, each carrying DOMPurify's own
`@license` line, and each copy runs from that comment to the end of the sanitiser it introduces:

| File | Whole file | The DOMPurify copy in it | Share |
|---|---|---|---|
| `web/vendor/hugerte/hugerte.min.mjs` | 472,604 bytes | 58,147 bytes, at offsets 184,972 to 243,119 | 12.3% |
| `web/vendor/hugerte/themes/silver/theme.min.mjs` | 448,049 bytes | 58,144 bytes, at offsets 103,327 to 161,471 | 13.0% |
| `web/viewer/views/markdown_scripts/mermaid.min.txt` | 3,338,725 bytes | 22,720 bytes, at offsets 106,243 to 128,963 | 0.7% |

That is 116,291 bytes of MPL-2.0 or Apache-2.0 in the HugeRTE pair, 8.6 per cent of the
1,353,824 bytes of HugeRTE the pack ships. Everything else in `web/vendor/hugerte/` is MIT.
Upstream declares the choice as `(MPL-2.0 OR Apache-2.0)`, so satisfying either is enough. Both
texts ship in both directories with their SHA-256 beside them.

## The rest of the bundles

**`mermaid.min.txt` is Mermaid's whole dependency tree** in one file: js-yaml 4.1.0, cytoscape.js
with its cose-bilkent layout, a second copy of KaTeX at 0.16.11, and four small helpers. All MIT,
each with its notice intact and listed in
[the licence file beside it](../web/viewer/views/markdown_scripts/LICENSE). The KaTeX version in
the table above is the standalone copy the viewer loads.

**HugeRTE is a fork of TinyMCE**, taken while TinyMCE 6 was MIT and kept MIT after TinyMCE 7
moved to GPLv2 or commercial. A Tiny Technologies copyright appears in an MIT file. The pack
keeps 19 of the 217 published files, bytes unchanged;
[`web/vendor/hugerte/README.md`](../web/vendor/hugerte/README.md) lists what was taken. Prism.js
states no version in the files that ship.

**The eight Haar and LBP cascades** in
[`modules/data/cascades/`](../modules/data/cascades/) that **Image Crop Face** reads:
six are OpenCV's own under the Intel Open Source Computer Vision Library licence,
`haarcascade_upperbody.xml` is copyright Hannes Kruppa and Bernt Schiele of ETH Zurich under the
same BSD-style terms, and `lbpcascade_animeface.xml` is MIT, copyright nagadomi. Each carries its
full licence text inside the `.xml`.

**The YuNet face model** ships with the pack in [`modules/data/models/`](../modules/data/models/),
as safetensors converted from the ONNX release, with the MIT text beside it in
[`LICENSE`](../modules/data/models/LICENSE). Nothing is downloaded to use it.

**The Marigold empty prompt embedding** ships beside it as
`marigold_empty_prompt.safetensors`, 8,280 bytes holding one 1 x 2 x 1024 single precision
tensor. It is what the Marigold checkpoints' own CLIP text encoder answers for the empty
prompt, identical in both repositories, so the intrinsic maps need neither that 649 MB
encoder nor its tokeniser. Open RAIL++-M, which the
Marigold repositories release their weights under, claims no rights in output generated with
the model, and its use restrictions reach any use of that output.

The software licences here place no conditions on what you make with the suite. Documents,
captions and images you produce are yours. Open RAIL++-M is the exception noted above: it claims
no rights in output, and its use restrictions still reach any use of that output, so they reach
the intrinsic maps **Power Preprocessor** answers with.
