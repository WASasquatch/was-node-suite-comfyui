/**
 * Frontend extension to parse JPEG EXIF workflows on drag-and-drop.
 *
 * Intercepts dropped JPEG images, parses the EXIF APP1 segment for tags
 * 0x010E (Workflow) and 0x010F (Prompt), and loads them into ComfyUI.
 */

import { app } from "../../scripts/app.js";

const EXT_NAME = "WASNodeSuite.JpegMetadata";

app.registerExtension({
  name: EXT_NAME,
  async setup(app) {
    const originalHandleFile = app.handleFile;
    app.handleFile = async function (file, ...args) {
      if (
        file &&
        (file.type === "image/jpeg" ||
          file.name?.toLowerCase().endsWith(".jpg") ||
          file.name?.toLowerCase().endsWith(".jpeg"))
      ) {
        try {
          const arrayBuffer = await file.arrayBuffer();
          const dataView = new DataView(arrayBuffer);

          if (dataView.getUint16(0) === 0xffd8) {
            let offset = 2;
            while (offset < dataView.byteLength) {
              const marker = dataView.getUint16(offset);
              const length = dataView.getUint16(offset + 2);

              if (marker === 0xffe1) {
                const header = String.fromCharCode(
                  ...new Uint8Array(arrayBuffer, offset + 4, 6)
                );
                if (header === "Exif\0\0") {
                  const tiffOffset = offset + 10;
                  const littleEndian = dataView.getUint16(tiffOffset) === 0x4949; // "II"

                  const readUint16 = (off) => dataView.getUint16(off, littleEndian);
                  const readUint32 = (off) => dataView.getUint32(off, littleEndian);

                  const ifdOffset = readUint32(tiffOffset + 4);
                  const ifdStart = tiffOffset + ifdOffset;

                  const numTags = readUint16(ifdStart);
                  let workflow = null;
                  let prompt = null;

                  for (let i = 0; i < numTags; i++) {
                    const tagStart = ifdStart + 2 + i * 12;
                    const tagId = readUint16(tagStart);
                    const tagCount = readUint32(tagStart + 4);

                    if (tagId === 0x010e || tagId === 0x010f) {
                      const valueOffset =
                        tagCount > 4 ? readUint32(tagStart + 8) : tagStart + 8;
                      const strBytes = new Uint8Array(
                        arrayBuffer,
                        tiffOffset + valueOffset,
                        tagCount - 1
                      );
                      const strValue = new TextDecoder("utf-8").decode(strBytes);

                      if (strValue.startsWith("Workflow:")) {
                        workflow = strValue.substring(9);
                      } else if (strValue.startsWith("Prompt:")) {
                        prompt = strValue.substring(7);
                      }
                    }
                  }

                  if (workflow) {
                    await app.loadGraphData(JSON.parse(workflow));
                    return;
                  } else if (prompt) {
                    await app.loadApiJson(JSON.parse(prompt));
                    return;
                  }
                  break;
                }
              }
              offset += length + 2;
            }
          }
        } catch (e) {
          console.error(`[${EXT_NAME}] Error parsing JPEG EXIF for workflow:`, e);
        }
      }
      return await originalHandleFile?.apply(this, [file, ...args]);
    };
  },
});
