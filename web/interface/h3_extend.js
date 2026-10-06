/**
 * MiniMax H3 frame arithmetic: frames run 17k+5, tokens 5k+2.
 *
 * `timeline` follows H3 Extend Append: a return cuts the clip at its last whole clip and
 * appends the window less its rejoined rows; a row the nodes refuse is null.
 */

/** Video frames a second. */
export const FPS = 24;

/** Audio latent frames a second. */
export const AUDIO_LATENT_FPS = 40;

/** Frames one clip of latent tokens covers. */
export const CLIP_FRAMES = 17;

/** The lead frames every sequence opens with. */
export const CLIP_LEAD = 5;

/** Latent tokens per clip. */
export const CLIP_TOKENS = 5;

/** The tokens a sequence opens with. */
export const TOKEN_LEAD = 2;

/** A cut referencing the clip's last frames as a video. */
export const REFERENCE_VIDEO = "reference (video)";

/** A cut referencing stills sampled across the whole clip as pictures. */
export const REFERENCE_SAMPLE = "reference (sample)";

/** Frames a video reference reaches back at least. */
export const REFERENCE_FRAMES = 56;

/** A cut to new picture with the soundtrack carried across it. */
export const AUDIO_CARRY = "carry (audio only)";

/** A cut to new picture referencing the clip's last frames, with the soundtrack carried across. */
export const AUDIO_REFERENCE = "carry (audio) + reference (video)";

/** How a segment continues from the one before it, in menu order. */
export const CONTINUITY = Object.freeze([
  "carry", "refresh", "handoff", REFERENCE_VIDEO, REFERENCE_SAMPLE, "cut", AUDIO_CARRY,
  AUDIO_REFERENCE,
]);

/** Continuities that carry the soundtrack across a cut to new picture. */
export const BRIDGING = Object.freeze([AUDIO_CARRY, AUDIO_REFERENCE]);

/** How a row's sound follows the scene before: as its transition does, carried, or fresh. */
export const SOUNDS = Object.freeze(["auto", "carry", "fresh"]);

/** Picture transitions that cut to a new scene, which a carried sound bridges. */
export const CUT_PICTURES = Object.freeze(["cut", "handoff", REFERENCE_VIDEO, REFERENCE_SAMPLE]);

/**
 * What a transition draws once its sound choice is applied.
 *
 * @param {string} continuity - The row's continuity.
 * @param {string} [sound="auto"] - An entry of `SOUNDS`.
 * @returns {{picture: string, bridged: boolean, fresh: boolean}} The picture transition, whether
 *   the scene before's sound carries across a cut, and whether a carried shot's sound starts fresh.
 */
export function resolvedTransition(continuity, sound = "auto") {
  const named = continuity === "reference" ? REFERENCE_VIDEO : String(continuity);
  const picture = named === AUDIO_CARRY ? "cut" : named === AUDIO_REFERENCE ? REFERENCE_VIDEO : named;
  if (sound === "fresh") return { picture, bridged: false, fresh: picture === "carry" || picture === "refresh" };
  const bridged = BRIDGING.includes(named) || (sound === "carry" && CUT_PICTURES.includes(picture));
  return { picture, bridged, fresh: false };
}

/**
 * The continuity a row's window is sized as once its sound choice is applied.
 *
 * @param {string} continuity - The row's continuity.
 * @param {string} [sound="auto"] - An entry of `SOUNDS`.
 * @returns {string} `AUDIO_CARRY` for a bridged cut, else the picture transition.
 */
export function geometryOf(continuity, sound = "auto") {
  const { picture, bridged } = resolvedTransition(continuity, sound);
  return bridged ? AUDIO_CARRY : picture;
}

/** Continuities that open a fresh scene and carry no frames. */
export const CUT_LIKE = Object.freeze(["cut", "handoff", REFERENCE_VIDEO, REFERENCE_SAMPLE]);

/** What a row may choose for how it continues from the row before it. */
export const ROW_CONTINUITY = Object.freeze([...CONTINUITY]);

/** A row's model choice that picks between the wired models by what the row carries. */
export const AUTO_MODEL = "auto";

/** What a row may choose for the model its segment is sampled with. */
export const MODEL_CHOICES = Object.freeze([AUTO_MODEL, "fl2va", "ref2va"]);

/** Which of the shared header and footer a row takes, the first being the default. */
export const WRAPS = Object.freeze(["both", "header only", "footer only", "neither"]);

/** A row's source that continues from the segment before it. */
export const PREVIOUS_SOURCE = -1;

/** Prompt rows a node offers, row 1 being the clip. */
export const MAX_ROWS = 24;

/** Every part an asset may play, in menu order. */
export const ROLES = Object.freeze([
  "first frame", "last frame", "keyframe", "reference picture", "reference clip",
  "reference audio",
]);

/** Roles that pin frames of the segment. */
export const PINNING = Object.freeze(ROLES.slice(0, 3));

/** Roles the segment's prompt references. */
export const REFERENCING = Object.freeze(ROLES.slice(3));

/** The segment number that puts an asset on every segment. */
export const EVERY_SEGMENT = 0;

/** Frames a clip is read up to when no length is asked for. */
export const LONGEST_CLIP = 362;

// The whole part of a number, with the sign kept.
const whole = Math.trunc;

/**
 * Round half to even.
 *
 * @param {number} value - Value to round.
 * @returns {number} The nearest whole number, a tie going to the even one.
 */
function roundHalfEven(value) {
  const floor = Math.floor(value);
  const rest = value - floor;
  if (rest > 0.5) return floor + 1;
  if (rest < 0.5) return floor;
  return floor % 2 === 0 ? floor : floor + 1;
}

/**
 * The remainder of a division, with the sign of the divisor.
 *
 * @param {number} value - Value to reduce.
 * @param {number} modulus - Positive modulus.
 * @returns {number} The remainder, 0 up to the modulus.
 */
function floorMod(value, modulus) {
  return ((value % modulus) + modulus) % modulus;
}

/**
 * The whole number of clips closest to a frame count, halves rounding up.
 *
 * @param {number} frames - Frames asked for.
 * @param {number} [least=0] - Fewest clips answered.
 * @returns {number} A clip count.
 */
export function nearestClips(frames, least = 0) {
  return Math.max(whole(least), Math.floor((2 * whole(frames) + CLIP_FRAMES) / (2 * CLIP_FRAMES)));
}

/**
 * The longest guide length at or below a frame count, never under the lead.
 *
 * @param {number} frames - Frames available.
 * @returns {number} A count on the 17k+5 grid, at least `CLIP_LEAD`.
 */
export function floorOverlap(frames) {
  return CLIP_LEAD + Math.floor(Math.max(0, whole(frames) - CLIP_LEAD) / CLIP_FRAMES) * CLIP_FRAMES;
}

/**
 * The guide length closest to a frame count, never under the lead.
 *
 * @param {number} frames - Frames asked for.
 * @returns {number} A count on the 17k+5 grid, at least `CLIP_LEAD`.
 */
export function snapOverlap(frames) {
  return CLIP_LEAD + nearestClips(Math.max(0, whole(frames) - CLIP_LEAD)) * CLIP_FRAMES;
}

/**
 * The clip length closest to a frame count.
 *
 * @param {number} frames - Frames asked for.
 * @returns {number} A count on the 17k+5 grid, at least `CLIP_LEAD`.
 */
export function snapClip(frames) {
  return snapOverlap(frames);
}

/**
 * The extension closest to a frame count, never under one clip.
 *
 * @param {number} frames - Frames asked for.
 * @returns {number} A positive multiple of `CLIP_FRAMES`.
 */
export function snapExtension(frames) {
  return nearestClips(frames, 1) * CLIP_FRAMES;
}

/**
 * Latent tokens a frame count occupies.
 *
 * @param {number} frames - A frame count on the 17k+5 grid.
 * @returns {number} The token count, 5k+2.
 */
export function tokensFor(frames) {
  const count = whole(frames);
  if (count <= CLIP_LEAD) return TOKEN_LEAD;
  return Math.floor((count - CLIP_LEAD) / CLIP_FRAMES) * CLIP_TOKENS + TOKEN_LEAD;
}

/**
 * Frames a token count covers, the inverse of `tokensFor`.
 *
 * @param {number} tokens - A token count on the 5k+2 grid.
 * @returns {number} The frame count.
 */
export function framesFor(tokens) {
  const count = whole(tokens);
  if (count <= TOKEN_LEAD) return CLIP_LEAD;
  return Math.floor((count - TOKEN_LEAD) / CLIP_TOKENS) * CLIP_FRAMES + CLIP_LEAD;
}

/**
 * Where a clip ends ahead of a cut, on the last whole clip it holds.
 *
 * @param {number} tokens - Rows the clip holds.
 * @returns {{rows: number, frames: number}} What is kept; a clip under one whole clip is kept entire.
 */
export function cutPoint(tokens) {
  const count = whole(tokens);
  const kept = count - floorMod(count, CLIP_TOKENS);
  if (kept <= 0) return { rows: count, frames: framesFor(count) };
  return { rows: kept, frames: (kept / CLIP_TOKENS) * CLIP_FRAMES };
}

/**
 * The last point at or before a token count where a sampled clip can end.
 *
 * @param {number} tokens - Latent rows.
 * @returns {number} The largest 5k+2 count not above `tokens`, or `tokens` under one lead.
 */
export function clipEnd(tokens) {
  const count = whole(tokens);
  if (count < TOKEN_LEAD) return count;
  return count - floorMod(count - TOKEN_LEAD, CLIP_TOKENS);
}

/**
 * Latent rows a segment end covers, on either the clip or the cut grid.
 *
 * @param {number} frames - A recorded segment end in frames.
 * @returns {number} The rows those frames decode from.
 */
export function seenRows(frames) {
  const count = whole(frames);
  if (floorMod(count, CLIP_FRAMES) === 0) return Math.floor(count / CLIP_FRAMES) * CLIP_TOKENS;
  return tokensFor(count);
}

/**
 * Window rows a return drops so it opens on the first frame not yet shown.
 *
 * @param {number} seen - Rows of the earlier segment shown before the cut away.
 * @param {number} rows - Rows of the clip the window carries from.
 * @param {number} head - Rows the window carries.
 * @returns {number} A multiple of `CLIP_TOKENS`.
 */
export function rejoinSkip(seen, rows, head) {
  const ahead = Math.max(0, whole(seen) - (whole(rows) - whole(head)));
  return Math.floor((2 * ahead + CLIP_TOKENS) / (2 * CLIP_TOKENS)) * CLIP_TOKENS;
}

/**
 * The sound bridge closest to a frame count, in whole clips and never under one.
 *
 * @param {number} frames - Frames asked for.
 * @returns {number} A positive multiple of `CLIP_FRAMES`.
 */
export function bridgeFrames(frames) {
  return nearestClips(Math.max(1, whole(frames)), 1) * CLIP_FRAMES;
}

/**
 * Audio latent frames a video frame count covers.
 *
 * @param {number} frames - A video frame count.
 * @returns {number} The audio latent length.
 */
export function audioSpan(frames) {
  return roundHalfEven((whole(frames) / FPS) * AUDIO_LATENT_FPS);
}

/**
 * Frames a length in seconds covers, before snapping.
 *
 * @param {number} seconds - A length in seconds.
 * @returns {number} A frame count at `FPS`, at least one frame.
 */
export function framesOf(seconds) {
  return Math.max(1, roundHalfEven(Number(seconds) * FPS));
}

/**
 * How long a frame count runs for.
 *
 * @param {number} frames - A frame count.
 * @returns {number} The length in seconds, to two decimals.
 */
export function durationOf(frames) {
  return roundHalfEven((whole(frames) / FPS) * 100) / 100;
}

/**
 * The overlap a segment may carry, never more than the segment itself holds.
 *
 * @param {number} frames - The segment's frame count.
 * @param {number} overlap - Frames asked to carry, 0 for a cut.
 * @returns {number} 0 for a cut, otherwise a count on the guide grid.
 */
export function snapOverlapFor(frames, overlap) {
  if (whole(overlap) <= 0) return 0;
  return Math.min(snapOverlap(overlap), floorOverlap(frames));
}

/**
 * New frames a segment adds so its window runs the clip length closest to `frames`.
 *
 * @param {number} frames - Frames the segment's window is asked to run, carried frames included.
 * @param {number} [overlap=0] - Frames asked to carry, 0 for a cut.
 * @param {string} [continuity="carry"] - The row's continuity.
 * @returns {number} A positive multiple of `CLIP_FRAMES`.
 */
export function snapSegment(frames, overlap = 0, continuity = CONTINUITY[0]) {
  const carried = CUT_LIKE.includes(continuity) || whole(overlap) <= 0
    ? CLIP_LEAD
    : snapOverlapFor(frames, overlap);
  return Math.max(CLIP_FRAMES, snapClip(frames) - carried);
}

/**
 * The window a row is sampled in, and how much of its start is not new.
 *
 * @param {number} frames - Frames the row's window is asked to run.
 * @param {number} overlap - Frames asked to carry, 0 for a cut.
 * @param {string} continuity - The row's continuity.
 * @param {boolean} [opening=false] - True for the row that opens the clip.
 * @returns {{head: number, window: number}} Frames carried or bridged at the start, and the whole window.
 */
export function windowOf(frames, overlap, continuity, opening = false) {
  if (opening) return { head: 0, window: snapClip(frames) };
  if (BRIDGING.includes(continuity)) {
    // The sound bridge is whole clips of the snapped overlap, one clip at least.
    const bridge = bridgeFrames(snapOverlapFor(frames, overlap));
    return { head: bridge, window: snapClip(bridge + snapSegment(frames, overlap, continuity)) };
  }
  if (CUT_LIKE.includes(continuity) || whole(overlap) <= 0) {
    return { head: 0, window: snapSegment(frames, overlap, continuity) + CLIP_LEAD };
  }
  const carried = snapOverlapFor(frames, overlap);
  return { head: carried, window: carried + snapSegment(frames, overlap, continuity) };
}

/**
 * The finished segment a segment continues from.
 *
 * @param {number} source - Negative counts back from this segment, positive names one from 1, 0 the one before.
 * @param {number} index - This segment's number, from 0.
 * @returns {number|null} A segment number from 0, or null where the source names no finished segment.
 */
export function resolvedSource(source, index) {
  const given = whole(source);
  const at = whole(index);
  const picked = given === 0 ? at - 1 : given < 0 ? at + given : given - 1;
  return picked >= 0 && picked < at ? picked : null;
}

/**
 * Frames a pinned clip covers, one for a still.
 *
 * @param {number} count - Frames the asset holds.
 * @returns {number} 1 under five frames, otherwise the longest 17k+5 count at or below it.
 */
export function guideLength(count) {
  const frames = whole(count);
  return frames < CLIP_LEAD ? 1 : floorOverlap(frames);
}

/**
 * Where a keyframe lands in the window its segment is sampled in.
 *
 * @param {number} frame - The asset's frame, from 0 in the new frames or back from the end when negative.
 * @param {number} head - Frames at the start of the window that are carried rather than new.
 * @param {number} window - Frames the whole window holds.
 * @param {number} [guide=1] - Frames the asset covers from where it lands.
 * @returns {number|null} A frame index into the window, or null where it would land outside it.
 */
export function resolvedIndex(frame, head, window, guide = 1) {
  const at = whole(frame);
  const span = whole(window);
  const covers = Math.max(1, whole(guide));
  const index = at >= 0 ? whole(head) + at : span + at;
  return index < 0 || index + covers > span ? null : index;
}

/**
 * Whether a segment cuts the clip it follows, which drops that clip's last lead frames.
 *
 * @param {string} continuity - The segment's continuity.
 * @param {number} overlap - Frames it asks to carry.
 * @param {number} source - Its source, as the row holds it.
 * @param {number} index - Its number, from 0.
 * @returns {boolean} True for a cut, a reference, a handoff, a sound bridge or a return.
 */
export function cutsInto(continuity, overlap, source, index) {
  if (whole(index) <= 0) return false;
  let named = String(continuity);
  if (named === "reference") named = REFERENCE_VIDEO;
  if (BRIDGING.includes(named) || CUT_LIKE.includes(named) || whole(overlap) <= 0) return true;
  const picked = resolvedSource(source, index);
  return picked !== null && picked !== whole(index) - 1;
}

/**
 * Where a closing frame lands in its window.
 *
 * @param {number} window - Frames the window holds.
 * @param {number} guide - Frames the closing asset covers.
 * @param {boolean} cutAfter - Whether the next segment cuts this one.
 * @returns {number} The last place the frame survives the join, never before frame 0.
 */
export function closingIndex(window, guide, cutAfter) {
  return Math.max(0, whole(window) - Math.max(1, whole(guide)) - (cutAfter ? CLIP_LEAD : 0));
}

/**
 * One prompt row's values as the node reads them.
 *
 * @param {object} row - The row as given.
 * @returns {{seconds: number, overlap: number, continuity: string, source: number}} The row.
 */
function readRow(row) {
  const given = row || {};
  const named = given.continuity === "reference" ? REFERENCE_VIDEO : given.continuity;
  const choice = ROW_CONTINUITY.includes(named) ? named : CONTINUITY[0];
  const source = given.source === undefined || given.source === null
    ? PREVIOUS_SOURCE
    : whole(Number(given.source));
  return {
    seconds: Number(given.seconds) || 0,
    overlap: whole(Number(given.overlap) || 0),
    continuity: choice,
    sound: SOUNDS.includes(given.sound) ? given.sound : SOUNDS[0],
    source,
  };
}

/**
 * The clip cut back to the end of one of its segments.
 *
 * @param {object} clip - The clip so far, with its segment ends and trimmed tails.
 * @param {number} index - Segment number, from 0.
 * @returns {object} The clip as it stood when that segment ended.
 */
function untilSegment(clip, index) {
  const ends = clip.ends;
  if (index === ends.length - 1) return clip;
  const end = ends[index];
  const tail = clip.tails.get(index);
  if (tail !== undefined && floorMod(end, CLIP_FRAMES) === 0) {
    const rows = Math.min(clip.tokens, Math.floor(end / CLIP_FRAMES) * CLIP_TOKENS);
    const tokens = rows + tail.rows;
    return {
      tokens,
      audio: Math.min(clip.audio, audioSpan(end)) + tail.audio,
      ends: [...ends.slice(0, index), framesFor(tokens)],
      tails: clip.tails,
    };
  }
  const tokens = clipEnd(tokensFor(end));
  const frames = framesFor(tokens);
  return {
    tokens: Math.min(clip.tokens, tokens),
    audio: Math.min(clip.audio, audioSpan(frames)),
    ends: [...ends.slice(0, index), frames],
    tails: clip.tails,
  };
}

/**
 * The window one row samples, as H3 Extend Window opens it.
 *
 * @param {object|null} clip - The clip so far, null ahead of the opening row.
 * @param {number} index - The row, from 0.
 * @param {number} length - The row's snapped new frames, the whole clip for row 0.
 * @param {number} carried - The row's snapped overlap, 0 for a cut.
 * @param {string} choice - The row's continuity.
 * @param {number} source - The row's source, as the widget holds it.
 * @returns {object|null} The pass, or null where the node refuses the row.
 */
function openWindow(clip, index, length, carried, choice, source) {
  if (index <= 0) {
    return { window: length, overlap: 0, bridge: null, rejoin: null, source: null };
  }
  const continuity = choice;
  if (BRIDGING.includes(continuity)) {
    const bridge = bridgeFrames(carried);
    const window = snapClip(bridge + snapExtension(length));
    const kept = cutPoint(clip.tokens).frames;
    if (Math.min(clip.audio, audioSpan(kept)) < audioSpan(bridge)) return null;
    return { window, overlap: bridge, bridge, rejoin: null, source: index - 1 };
  }
  const picked = resolvedSource(source, index);
  if (picked === null) return null;
  let from = clip;
  let seen = 0;
  const earlier = picked !== index - 1;
  if (earlier) {
    seen = picked < clip.ends.length ? seenRows(clip.ends[picked]) : 0;
    from = untilSegment(clip, picked);
  }
  if (continuity === REFERENCE_SAMPLE || carried <= 0 || continuity === "cut") {
    return { window: snapClip(length), overlap: 0, bridge: null, rejoin: null, source: picked };
  }
  const overlap = snapOverlap(carried);
  const extension = snapExtension(length);
  const head = tokensFor(overlap);
  if (from.tokens < head) return null;
  if (continuity === "handoff" || continuity === REFERENCE_VIDEO) {
    return { window: snapClip(extension), overlap: 0, bridge: null, rejoin: null, source: picked };
  }
  const rejoin = earlier ? rejoinSkip(seen, from.tokens, head) : null;
  return { window: overlap + extension, overlap, bridge: null, rejoin, source: picked };
}

/**
 * The clip with one sampled window joined on, as H3 Extend Append joins it.
 *
 * @param {object|null} clip - The clip so far, null ahead of the opening row.
 * @param {number} index - The row, from 0.
 * @param {object} pass - The window from `openWindow`.
 * @returns {object} The longer clip, with `kept` naming the frame a cut ended on, or null.
 */
function appendWindow(clip, index, pass) {
  const rows = tokensFor(pass.window);
  const sound = audioSpan(pass.window);
  if (index <= 0) {
    return { tokens: rows, audio: sound, ends: [framesFor(rows)], tails: new Map(), kept: null };
  }
  const ends = clip.ends.length ? [...clip.ends] : [framesFor(clip.tokens)];
  const tails = new Map(clip.tails);
  if (pass.bridge === null && pass.overlap > 0 && pass.rejoin === null) {
    const overlap = snapOverlap(pass.overlap);
    const held = Math.min(tokensFor(overlap), clip.tokens, rows);
    const span = Math.min(audioSpan(overlap), clip.audio, sound);
    const tokens = clip.tokens + rows - held;
    ends.push(framesFor(tokens));
    return { tokens, audio: clip.audio + sound - span, ends, tails, kept: null };
  }
  const cut = cutPoint(clip.tokens);
  const keptAudio = Math.min(clip.audio, audioSpan(cut.frames));
  let tokens;
  let audio;
  if (pass.bridge !== null) {
    const span = Math.min(audioSpan(pass.bridge), sound);
    tokens = cut.rows + rows - Math.floor(pass.bridge / CLIP_FRAMES) * CLIP_TOKENS;
    audio = keptAudio + sound - span;
  } else if (pass.overlap <= 0) {
    tokens = cut.rows + rows;
    audio = keptAudio + sound;
  } else {
    const start = Math.max(0, Math.min(pass.rejoin, Math.floor((rows - 1) / CLIP_TOKENS) * CLIP_TOKENS));
    const dropped = Math.floor(start / CLIP_TOKENS) * CLIP_FRAMES;
    tokens = cut.rows + rows - start;
    audio = keptAudio + sound - Math.min(sound, audioSpan(dropped));
  }
  if (clip.tokens - cut.rows > 0) {
    tails.set(ends.length - 1, { rows: clip.tokens - cut.rows, audio: clip.audio - keptAudio });
  }
  ends[ends.length - 1] = Math.min(ends[ends.length - 1], cut.frames);
  ends.push(framesFor(tokens));
  return { tokens, audio, ends, tails, kept: cut.frames };
}

/**
 * Where each row's new picture lands in the finished clip.
 *
 * @param {Array<{seconds: number, overlap: number, continuity: string, source: number}>} rows -
 *   One entry per prompt row, in order.
 * @returns {Array<object|null>} One entry per row, in frames, ending on null at a row the nodes refuse.
 */
export function timeline(rows) {
  const entries = [];
  let clip = null;
  for (let index = 0; index < rows.length; index += 1) {
    const row = readRow(rows[index]);
    const frames = framesOf(row.seconds);
    const opening = index === 0;
    const shape = geometryOf(row.continuity, row.sound);
    const { head, window } = windowOf(frames, row.overlap, shape, opening);
    const length = opening ? snapClip(frames) : snapSegment(frames, row.overlap, shape);
    const carried = opening ? 0 : snapOverlapFor(frames, row.overlap);
    const pass = openWindow(clip, index, length, carried, shape, row.source);
    if (pass === null) {
      entries.push(null);
      break;
    }
    const before = clip === null ? 0 : framesFor(clip.tokens);
    clip = appendWindow(clip, index, pass);
    const end = clip.ends[clip.ends.length - 1];
    const start = clip.kept === null ? before : clip.kept;
    entries.push({
      frames,
      head,
      window,
      newFrames: end - start,
      start,
      end,
      trimmed: before - start,
      source: pass.source,
    });
  }
  return entries;
}
