/**
 * The audio a panel draws and plays, read from the WAV sides a node wrote and held in the page.
 *
 * A held clip is one Float32Array per channel with coarser minimum and maximum summaries.
 */

import { addressOf, keptEntry } from "./compare_media.js";

const LOG_NAME = "WASNodeSuite.AudioMedia";

// Largest WAV file read into the page: a side's samples plus its header.
const MAX_FILE_BYTES = 96 * 1024 * 1024 + 4096;

// Channels a side holds at most.
const MAX_CHANNELS = 8;

/**
 * Whether an entry names an audio side the pack kept, the only kind a panel loads.
 *
 * @param {object} entry - A side's `filename` and `type`.
 * @returns {boolean} True for a kept WAV.
 */
export function keptAudio(entry) {
  return keptEntry(entry) && entry.filename.endsWith(".wav");
}

// Bytes of samples held across every panel before the least recently used idle clip is let go.
const HELD_BYTES = 768 * 1024 * 1024;

// Samples one summary value covers, finest first.
export const SUMMARY_STEPS = [64, 1024, 16384];

// `address -> {promise, clip, bytes, users, used}`.
const held = new Map();

/**
 * Read a WAV file's samples.
 *
 * @param {ArrayBuffer} buffer - The file.
 * @returns {{rate: number, channels: Float32Array[], length: number}} One array per channel.
 * @throws {Error} The bytes are not a WAV of 16, 24 or 32 bit PCM or 32 bit float.
 */
export function readWav(buffer) {
  const view = new DataView(buffer);
  const tag = (at) => String.fromCharCode(view.getUint8(at), view.getUint8(at + 1),
    view.getUint8(at + 2), view.getUint8(at + 3));
  if (buffer.byteLength < 12 || tag(0) !== "RIFF" || tag(8) !== "WAVE") {
    throw new Error("not a WAV file");
  }
  let format = null;
  let data = null;
  let at = 12;
  while (at + 8 <= buffer.byteLength) {
    const id = tag(at);
    const size = view.getUint32(at + 4, true);
    const body = at + 8;
    if (id === "fmt " && size >= 16) {
      format = {
        code: view.getUint16(body, true),
        channels: view.getUint16(body + 2, true),
        rate: view.getUint32(body + 4, true),
        bits: view.getUint16(body + 14, true),
      };
      if (format.code === 0xfffe && size >= 26) format.code = view.getUint16(body + 24, true);
    } else if (id === "data") {
      data = { offset: body, size: Math.min(size, buffer.byteLength - body) };
    }
    at = body + size + (size % 2);
  }
  if (!format || !data || !(format.channels > 0) || !(format.rate > 0)) {
    throw new Error("the WAV file names no format or no samples");
  }
  const { channels, bits, code } = format;
  const floating = code === 3 && bits === 32;
  if (!floating && !(code === 1 && (bits === 16 || bits === 24 || bits === 32))) {
    throw new Error(`a WAV of format ${code} at ${bits} bits is not read here`);
  }
  if (channels > MAX_CHANNELS) throw new Error(`a WAV of ${channels} channels is not read here`);
  const width = bits / 8;
  const frames = Math.floor(data.size / (width * channels));
  const out = Array.from({ length: channels }, () => new Float32Array(frames));
  if (floating && data.offset % 4 === 0) {
    const all = new Float32Array(buffer, data.offset, frames * channels);
    for (let frame = 0, index = 0; frame < frames; frame += 1) {
      for (let channel = 0; channel < channels; channel += 1, index += 1) out[channel][frame] = all[index];
    }
  } else {
    let cursor = data.offset;
    for (let frame = 0; frame < frames; frame += 1) {
      for (let channel = 0; channel < channels; channel += 1) {
        let value;
        if (floating) value = view.getFloat32(cursor, true);
        else if (bits === 16) value = view.getInt16(cursor, true) / 32768;
        else if (bits === 24) {
          const low = view.getUint16(cursor, true);
          const high = view.getInt8(cursor + 2);
          value = ((high << 16) | low) / 8388608;
        } else value = view.getInt32(cursor, true) / 2147483648;
        out[channel][frame] = value;
        cursor += width;
      }
    }
  }
  return { rate: format.rate, channels: out, length: frames };
}

/**
 * Minimum and maximum of every run of samples, at each of `SUMMARY_STEPS`.
 *
 * @param {Float32Array[]} channels - One array per channel.
 * @returns {Array<{step: number, mins: Float32Array[], maxs: Float32Array[]}>} Finest first.
 */
export function summarise(channels) {
  const levels = [];
  let source = null;
  for (const step of SUMMARY_STEPS) {
    const mins = [];
    const maxs = [];
    for (let channel = 0; channel < channels.length; channel += 1) {
      const count = Math.ceil(channels[channel].length / step);
      const low = new Float32Array(count);
      const high = new Float32Array(count);
      if (!source) {
        const samples = channels[channel];
        for (let bucket = 0; bucket < count; bucket += 1) {
          let min = Infinity;
          let max = -Infinity;
          const end = Math.min(samples.length, (bucket + 1) * step);
          for (let index = bucket * step; index < end; index += 1) {
            const value = samples[index];
            if (value < min) min = value;
            if (value > max) max = value;
          }
          low[bucket] = min;
          high[bucket] = max;
        }
      } else {
        const ratio = step / source.step;
        const fineMin = source.mins[channel];
        const fineMax = source.maxs[channel];
        for (let bucket = 0; bucket < count; bucket += 1) {
          let min = Infinity;
          let max = -Infinity;
          const end = Math.min(fineMin.length, (bucket + 1) * ratio);
          for (let index = bucket * ratio; index < end; index += 1) {
            if (fineMin[index] < min) min = fineMin[index];
            if (fineMax[index] > max) max = fineMax[index];
          }
          low[bucket] = min;
          high[bucket] = max;
        }
      }
      mins.push(low);
      maxs.push(high);
    }
    source = { step, mins, maxs };
    levels.push(source);
  }
  return levels;
}

/**
 * The lowest and highest sample of one channel over a span, read from the coarsest summary that
 * still resolves it.
 *
 * @param {object} clip - A held clip.
 * @param {number} channel - Channel index.
 * @param {number} first - First sample, inclusive.
 * @param {number} last - Last sample, exclusive.
 * @returns {[number, number]} `[min, max]`, `[0, 0]` for an empty span.
 */
export function spanRange(clip, channel, first, last) {
  const lo = Math.max(0, Math.floor(first));
  const hi = Math.min(clip.length, Math.ceil(last));
  if (hi <= lo) return [0, 0];
  let level = null;
  for (const candidate of clip.summary) {
    if (candidate.step * 2 <= hi - lo) level = candidate;
  }
  let min = Infinity;
  let max = -Infinity;
  const samples = clip.channels[channel];
  if (!level) {
    for (let index = lo; index < hi; index += 1) {
      if (samples[index] < min) min = samples[index];
      if (samples[index] > max) max = samples[index];
    }
    return [min, max];
  }
  const step = level.step;
  const startBucket = Math.ceil(lo / step);
  const endBucket = Math.floor(hi / step);
  for (let index = lo; index < Math.min(hi, startBucket * step); index += 1) {
    if (samples[index] < min) min = samples[index];
    if (samples[index] > max) max = samples[index];
  }
  const mins = level.mins[channel];
  const maxs = level.maxs[channel];
  for (let bucket = startBucket; bucket < endBucket; bucket += 1) {
    if (mins[bucket] < min) min = mins[bucket];
    if (maxs[bucket] > max) max = maxs[bucket];
  }
  for (let index = Math.max(lo, endBucket * step); index < hi; index += 1) {
    if (samples[index] < min) min = samples[index];
    if (samples[index] > max) max = samples[index];
  }
  return [min, max];
}

/** Let go of the least recently used idle clips until the page holds under `HELD_BYTES`. */
function trim() {
  let total = 0;
  for (const entry of held.values()) total += entry.bytes;
  if (total <= HELD_BYTES) return;
  const idle = [...held.entries()]
    .filter(([, entry]) => entry.users === 0 && entry.clip)
    .sort((a, b) => a[1].used - b[1].used);
  for (const [address, entry] of idle) {
    if (total <= HELD_BYTES) break;
    held.delete(address);
    total -= entry.bytes;
  }
}

/**
 * Read one written side into the page.
 *
 * @param {object} entry - A side's `ui` entry: `filename`, `subfolder`, `type`, `exact` and
 *   `detail`.
 * @param {string} address - Where it is served from.
 * @returns {Promise<object|null>} The clip, or null when it could not be read.
 */
async function fetchClip(entry, address) {
  const answer = await fetch(address);
  const size = Number(answer.headers.get("content-length") ?? 0);
  if (answer.status !== 200 || size > MAX_FILE_BYTES) {
    answer.body?.cancel?.();
    return null;
  }
  const buffer = await answer.arrayBuffer();
  if (buffer.byteLength > MAX_FILE_BYTES) return null;
  const decoded = readWav(buffer);
  return {
    address,
    rate: decoded.rate,
    channels: decoded.channels,
    length: decoded.length,
    duration: decoded.length / decoded.rate,
    exact: entry?.exact === true,
    detail: typeof entry?.detail === "string" ? entry.detail : "",
    summary: summarise(decoded.channels),
  };
}

/**
 * Hold one side's clip in the page for a panel that draws it.
 *
 * @param {object} entry - A side's `ui` entry.
 * @returns {{address: string, promise: Promise<object|null>}} The address to release and the
 *   clip, null when it could not be read. A failed read is forgotten so a later call reads again.
 */
export function holdAudio(entry) {
  const address = keptAudio(entry) ? addressOf(entry) : "";
  if (!address) return { address: "", promise: Promise.resolve(null) };
  let record = held.get(address);
  if (!record) {
    record = { promise: null, clip: null, bytes: 0, users: 0, used: 0 };
    held.set(address, record);
    const current = record;
    record.promise = fetchClip(entry, address)
      .then((clip) => {
        if (!clip) {
          if (held.get(address) === current) held.delete(address);
          return null;
        }
        current.clip = clip;
        current.bytes = clip.length * clip.channels.length * 4 * 1.1;
        trim();
        return clip;
      })
      .catch((error) => {
        console.warn(`[${LOG_NAME}] A panel's audio could not be read:`, error);
        if (held.get(address) === current) held.delete(address);
        return null;
      });
  }
  record.users += 1;
  record.used = performance.now();
  return { address, promise: record.promise };
}

/**
 * Stop drawing a clip one `holdAudio` call asked for.
 *
 * @param {string} address - The address `holdAudio` answered.
 * @returns {void}
 */
export function releaseAudio(address) {
  const record = held.get(address);
  if (!record) return;
  record.users = Math.max(0, record.users - 1);
  record.used = performance.now();
  trim();
}

/**
 * A time as the panel prints it.
 *
 * @param {number} seconds - The time.
 * @param {number} [places] - Decimal places of the seconds.
 * @returns {string} `0:04.250`, or `1:02:04.250` past an hour.
 */
export function formatTime(seconds, places = 3) {
  const value = Math.max(0, Number(seconds) || 0);
  const hours = Math.floor(value / 3600);
  const minutes = Math.floor((value % 3600) / 60);
  const rest = (value % 60).toFixed(places).padStart(places ? places + 3 : 2, "0");
  return hours ? `${hours}:${String(minutes).padStart(2, "0")}:${rest}` : `${minutes}:${rest}`;
}
