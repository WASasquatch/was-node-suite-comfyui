# KaTeX and Mermaid, vendored

Maths typesetting and diagrams for the Content Viewer's markdown view. The files are read as
text and inlined into the view's sandboxed frame.

| Package | Version | Licence |
|---|---|---|
| KaTeX, <https://github.com/KaTeX/KaTeX> | npm `katex@0.16.9` | MIT, in `LICENSE` |
| Mermaid, <https://github.com/mermaid-js/mermaid> | npm `mermaid@10.9.5` | MIT, in `LICENSE` |
| DOMPurify 3.2.4, compiled into Mermaid's bundle | as bundled by `mermaid@10.9.5` | `MPL-2.0 OR Apache-2.0`, in `LICENSE-MPL-2.0.txt` and `LICENSE-APACHE-2.0.txt` |

## Files taken from upstream

| File | Upstream file | Changed |
|---|---|---|
| `katex.min.txt` | `katex/dist/katex.min.js` | no |
| `katex-auto-render.min.txt` | `katex/dist/contrib/auto-render.min.js` | no |
| `katex-with-fonts.min.css.txt` | `katex/dist/katex.min.css` | each of its 20 `@font-face` rules names its `.woff2` font as a `data:` URL holding that font's bytes; the rest of the stylesheet is unchanged |
| `mermaid.min.txt` | `mermaid/dist/mermaid.min.js` | no |

Files marked unchanged are byte for byte the published file, renamed to `.txt`.
