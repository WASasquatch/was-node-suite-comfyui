/**
 * Playing held clips through the page's one audio context, one player sounding at a time.
 *
 * A player plays from a time, loops a span or plays pieces back to back, and reports its
 * position and level.
 */

const LOG_NAME = "WASNodeSuite.AudioEngine";

// Seconds a start or a stop is ramped over, so neither clicks.
const RAMP = 0.006;

// Samples the level meter reads per channel.
const METER_SIZE = 2048;

let context = null;
let sounding = null;
const buffers = new WeakMap();

/**
 * The page's audio context, made on the first play.
 *
 * @returns {AudioContext|null} The context, or null where the browser offers none.
 */
function audioContext() {
  if (!context) {
    const Maker = window.AudioContext || window.webkitAudioContext;
    if (!Maker) return null;
    context = new Maker();
  }
  if (context.state === "suspended") context.resume().catch(() => {});
  return context;
}

/**
 * The playable buffer of a held clip, made once per clip.
 *
 * @param {AudioContext} ctx - The page's context.
 * @param {object} clip - A clip from `holdAudio`.
 * @returns {AudioBuffer} The buffer, at the clip's own rate.
 */
function bufferOf(ctx, clip) {
  let buffer = buffers.get(clip);
  if (!buffer) {
    buffer = ctx.createBuffer(clip.channels.length, Math.max(1, clip.length), clip.rate);
    clip.channels.forEach((samples, channel) => buffer.copyToChannel(samples, channel));
    buffers.set(clip, buffer);
  }
  return buffer;
}

/**
 * Build a player.
 *
 * @param {{onState?: (playing: boolean) => void}} [options] - `onState` hears every start and stop.
 * @returns {object} `play`, `playPieces`, `stop`, `playing`, `position`, `levels` and `dispose`.
 */
export function createPlayer(options = {}) {
  let nodes = [];
  let timeline = [];
  let clipPlaying = null;
  let looping = null;
  let pausedAt = 0;
  let analysers = [];
  let output = null;
  let ended = 0;

  const announce = (playing) => {
    try {
      options.onState?.(playing);
    } catch (error) {
      console.error(`[${LOG_NAME}] A player state handler failed:`, error);
    }
  };

  const teardown = () => {
    const ctx = context;
    for (const node of nodes) {
      try {
        if (node.gain && ctx) {
          node.gain.gain.cancelScheduledValues(ctx.currentTime);
          node.gain.gain.setTargetAtTime(0, ctx.currentTime, RAMP / 3);
        }
        node.source.onended = null;
        node.source.stop(ctx ? ctx.currentTime + RAMP : 0);
      } catch (error) {
        console.debug(`[${LOG_NAME}] A source had already stopped:`, error);
      }
    }
    nodes = [];
    clearTimeout(ended);
  };

  const route = (ctx, channels) => {
    if (output && analysers.length === channels) return output;
    output?.disconnect();
    output = ctx.createGain();
    output.connect(ctx.destination);
    const splitter = ctx.createChannelSplitter(channels);
    output.connect(splitter);
    analysers = Array.from({ length: channels }, (_, channel) => {
      const analyser = ctx.createAnalyser();
      analyser.fftSize = METER_SIZE;
      splitter.connect(analyser, channel);
      return analyser;
    });
    return output;
  };

  /**
   * Play pieces of a clip back to back.
   *
   * @param {object} clip - A held clip.
   * @param {Array<{from: number, to: number, fadeIn?: number, fadeOut?: number}>} pieces - Spans in
   *   seconds of the clip, each with optional fades in seconds.
   * @returns {boolean} True when it started.
   */
  function playPieces(clip, pieces) {
    const ctx = audioContext();
    if (!ctx || !clip || !pieces?.length) return false;
    if (sounding && sounding !== api) sounding.stop();
    teardown();
    const buffer = bufferOf(ctx, clip);
    const into = route(ctx, buffer.numberOfChannels);
    let at = ctx.currentTime + 0.03;
    timeline = [];
    for (const piece of pieces) {
      const from = Math.max(0, Math.min(clip.duration, piece.from));
      const to = Math.max(from, Math.min(clip.duration, piece.to));
      const span = to - from;
      if (span <= 0) continue;
      const source = ctx.createBufferSource();
      source.buffer = buffer;
      const gain = ctx.createGain();
      source.connect(gain).connect(into);
      const rise = Math.min(span / 2, Math.max(RAMP, piece.fadeIn || 0));
      const fall = Math.min(span / 2, Math.max(RAMP, piece.fadeOut || 0));
      gain.gain.setValueAtTime(0, at);
      gain.gain.linearRampToValueAtTime(1, at + rise);
      gain.gain.setValueAtTime(1, at + span - fall);
      gain.gain.linearRampToValueAtTime(0, at + span);
      source.start(at, from, span);
      nodes.push({ source, gain });
      timeline.push({ at, from, span });
      at += span;
    }
    if (!timeline.length) return false;
    clipPlaying = clip;
    looping = null;
    sounding = api;
    const total = (at - ctx.currentTime) * 1000;
    ended = setTimeout(() => finish(), total + 60);
    announce(true);
    return true;
  }

  /**
   * Play a clip from a time, optionally looping a span.
   *
   * @param {object} clip - A held clip.
   * @param {{from?: number, loop?: [number, number]|null}} [how] - Where to start, and the span to
   *   loop, in seconds.
   * @returns {boolean} True when it started.
   */
  function play(clip, how = {}) {
    const ctx = audioContext();
    if (!ctx || !clip) return false;
    if (sounding && sounding !== api) sounding.stop();
    teardown();
    const buffer = bufferOf(ctx, clip);
    const into = route(ctx, buffer.numberOfChannels);
    const loop = how.loop && how.loop[1] - how.loop[0] > 0.01 ? how.loop : null;
    let from = Math.max(0, Math.min(clip.duration, how.from ?? 0));
    if (loop && (from < loop[0] || from >= loop[1])) from = loop[0];
    if (!loop && from >= clip.duration - 0.001) from = 0;
    const source = ctx.createBufferSource();
    source.buffer = buffer;
    const gain = ctx.createGain();
    source.connect(gain).connect(into);
    const at = ctx.currentTime + 0.02;
    gain.gain.setValueAtTime(0, at);
    gain.gain.linearRampToValueAtTime(1, at + RAMP);
    if (loop) {
      source.loop = true;
      source.loopStart = loop[0];
      source.loopEnd = loop[1];
      source.start(at, from);
    } else {
      source.start(at, from);
      source.onended = () => finish();
    }
    nodes = [{ source, gain }];
    timeline = [{ at, from, span: loop ? Infinity : clip.duration - from }];
    clipPlaying = clip;
    looping = loop;
    sounding = api;
    announce(true);
    return true;
  }

  function finish() {
    if (!clipPlaying) return;
    const last = timeline[timeline.length - 1];
    pausedAt = last ? Math.min(clipPlaying.duration, last.from + last.span) : 0;
    teardown();
    clipPlaying = null;
    if (sounding === api) sounding = null;
    announce(false);
  }

  /**
   * Stop playing, keeping the position.
   *
   * @returns {void}
   */
  function stop() {
    if (!clipPlaying) return;
    pausedAt = position();
    teardown();
    clipPlaying = null;
    if (sounding === api) sounding = null;
    announce(false);
  }

  /**
   * Where playback is, in seconds of the clip.
   *
   * @returns {number} The playing position, or where it last stopped.
   */
  function position() {
    if (!clipPlaying || !context) return pausedAt;
    const now = context.currentTime;
    const first = timeline[0];
    if (!first || now < first.at) return first ? first.from : pausedAt;
    for (const piece of timeline) {
      if (now < piece.at + piece.span) {
        const into = now - piece.at;
        if (looping) {
          const span = looping[1] - looping[0];
          const reached = piece.from + into;
          return reached < looping[1] ? reached : looping[0] + ((reached - looping[0]) % span);
        }
        return piece.from + into;
      }
    }
    const last = timeline[timeline.length - 1];
    return last.from + last.span;
  }

  /**
   * The level of each channel over the last few milliseconds.
   *
   * @returns {Array<{peak: number, rms: number}>} Linear levels, one per channel; empty when idle.
   */
  function levels() {
    if (!clipPlaying) return [];
    const block = new Float32Array(METER_SIZE);
    return analysers.map((analyser) => {
      analyser.getFloatTimeDomainData(block);
      let peak = 0;
      let sum = 0;
      for (const value of block) {
        const size = Math.abs(value);
        if (size > peak) peak = size;
        sum += value * value;
      }
      return { peak, rms: Math.sqrt(sum / block.length) };
    });
  }

  function dispose() {
    stop();
    output?.disconnect();
    output = null;
    analysers = [];
  }

  const api = {
    play,
    playPieces,
    stop,
    position,
    levels,
    dispose,
    playing: () => Boolean(clipPlaying),
    clip: () => clipPlaying,
    seek: (seconds) => { pausedAt = Math.max(0, seconds); },
  };
  return api;
}
