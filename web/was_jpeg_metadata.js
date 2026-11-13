/**
 * Frontend extension to parse JPEG EXIF workflows on drag-and-drop.
 *
 * Intercepts dropped JPEG images, parses the EXIF APP1 segment for workflow
 * and prompt metadata matching the WebP EXIF layout, and loads them into ComfyUI.
 */

import { app } from "../../scripts/app.js";

const EXT_NAME = "WASNodeSuite.JpegMetadata";

// Matches JSON strings OR non-finite tokens (NaN, Infinity, -Infinity) outside strings
const NON_FINITE_REGEX = /"(?:\\.|[^"\\])*"|(?<![\w.-])(-?Infinity|NaN)(?![\w.])/g;

/**
 * Parse JSON text, converting unquoted NaN and Infinity tokens to null.
 *
 * @param {string} text - The JSON text to parse.
 * @returns {object|null} The parsed JSON value.
 */
function parseJsonWithNonFinite(text) {
  try {
    return JSON.parse(text);
  } catch {
    const sanitized = text.replace(NON_FINITE_REGEX, (match, nonFinite) =>
      nonFinite ? "null" : match
    );
    return JSON.parse(sanitized);
  }
}

app.registerExtension({
  name: EXT_NAME,
  async setup(app) {
    const originalHandleFile = app.handleFile;
    app.handleFile = async function (file, ...args) {
      let parsedWorkflow = null;
      let parsedPrompt = null;

      if (
        file &&
        (file.type === "image/jpeg" ||
          file.name?.toLowerCase().endsWith(".jpg") ||
          file.name?.toLowerCase().endsWith(".jpeg"))
      ) {
        try {
          // Read the first 128 KB.
          const sliceSize = Math.min(file.size || 0, 128 * 1024);
          const arrayBuffer = await file.slice(0, sliceSize).arrayBuffer();
          const dataView = new DataView(arrayBuffer);

          if (dataView.byteLength >= 4 && dataView.getUint16(0) === 0xffd8) {
            let offset = 2;
            let workflowStr = null;
            let promptStr = null;

            while (offset < dataView.byteLength) {
              if (dataView.getUint8(offset) !== 0xff) break;
              const marker = dataView.getUint16(offset);
              // Stop walk at SOS (Start of Scan) or EOI
              if (marker === 0xffda || marker === 0xffd9) break;
              // Skip standalone markers without length (RST0..RST7, TEM)
              if ((marker >= 0xffd0 && marker <= 0xffd7) || marker === 0xff01) {
                offset += 2;
                continue;
              }
              if (offset + 4 > dataView.byteLength) break;
              const length = dataView.getUint16(offset + 2);
              if (length < 2) break;

              if (marker === 0xffe1 && offset + 10 <= dataView.byteLength) {
                const header = String.fromCharCode(
                  ...new Uint8Array(arrayBuffer, offset + 4, 6)
                );
                if (header === "Exif\0\0") {
                  const tiffOffset = offset + 10;
                  if (tiffOffset + 8 <= dataView.byteLength) {
                    const endian = dataView.getUint16(tiffOffset);
                    if (endian === 0x4949 || endian === 0x4d4d) {
                      const littleEndian = endian === 0x4949; // "II" vs "MM"
                      if (dataView.getUint16(tiffOffset + 2, littleEndian) === 0x002a) {
                        const ifdOffset = dataView.getUint32(tiffOffset + 4, littleEndian);
                        const ifdStart = tiffOffset + ifdOffset;
                        if (ifdStart + 2 <= dataView.byteLength) {
                          const numTags = dataView.getUint16(ifdStart, littleEndian);
                          for (let i = 0; i < numTags; i++) {
                            const tagStart = ifdStart + 2 + i * 12;
                            if (tagStart + 12 > dataView.byteLength) break;

                            const tagType = dataView.getUint16(tagStart + 2, littleEndian);
                            const tagCount = dataView.getUint32(tagStart + 4, littleEndian);
                            if (tagCount === 0) continue;

                            const typeSize =
                              tagType === 3
                                ? 2
                                : tagType === 4
                                ? 4
                                : tagType === 5 || tagType === 12
                                ? 8
                                : 1;
                            const totalBytes = tagCount * typeSize;
                            const valOffset =
                              totalBytes <= 4
                                ? tagStart + 8
                                : tiffOffset +
                                  dataView.getUint32(tagStart + 8, littleEndian);

                            if (
                              valOffset >= 0 &&
                              valOffset + totalBytes <= dataView.byteLength
                            ) {
                              let len = totalBytes;
                              if (
                                tagType === 2 &&
                                len > 0 &&
                                dataView.getUint8(valOffset + len - 1) === 0
                              ) {
                                len -= 1;
                              }
                              if (len > 0) {
                                const strBytes = new Uint8Array(
                                  arrayBuffer,
                                  valOffset,
                                  len
                                );
                                const strValue = new TextDecoder("utf-8").decode(
                                  strBytes
                                );
                                if (
                                  strValue.startsWith("workflow:") ||
                                  strValue.startsWith("Workflow:")
                                ) {
                                  workflowStr = strValue.substring(9);
                                } else if (
                                  strValue.startsWith("prompt:") ||
                                  strValue.startsWith("Prompt:")
                                ) {
                                  promptStr = strValue.substring(7);
                                }
                              }
                            }
                          }
                        }
                      }
                    }
                  }
                  break;
                }
              }
              offset += length + 2;
            }

            if (workflowStr) {
              try {
                const parsed = parseJsonWithNonFinite(workflowStr);
                if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
                  parsedWorkflow = parsed;
                }
              } catch (err) {
                console.error(`[${EXT_NAME}] Failed to parse workflow:`, err);
              }
            }

            // Fall back to prompt if workflow was missing or failed to parse
            if (!parsedWorkflow && promptStr) {
              try {
                const parsed = parseJsonWithNonFinite(promptStr);
                if (parsed && typeof parsed === "object") {
                  parsedPrompt = parsed;
                }
              } catch (err) {
                console.error(`[${EXT_NAME}] Failed to parse prompt:`, err);
              }
            }
          }
        } catch (e) {
          console.error(`[${EXT_NAME}] Error parsing JPEG EXIF for workflow:`, e);
        }
      }

      const fileName = file?.name ? file.name.replace(/\.\w+$/, "") : "Workflow";
      const source = args[0] || "file_drop";
      const options = args[1];

      if (parsedWorkflow) {
        await app.loadGraphData(parsedWorkflow, true, true, fileName, {
          openSource: source,
          deferWarnings: options?.deferWarnings,
        });
        return;
      }

      if (parsedPrompt && app.isApiJson?.(parsedPrompt)) {
        try {
          await app.loadApiJson(parsedPrompt, fileName, {
            deferWarnings: options?.deferWarnings,
          });
        } catch (err) {
          console.error(`[${EXT_NAME}] Failed to load API prompt:`, err);
          app.showErrorOnFileLoad?.(file);
        }
        return;
      }

      return await originalHandleFile?.apply(this, [file, ...args]);
    };
  },
});
