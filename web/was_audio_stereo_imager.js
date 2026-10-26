/**
 * The Audio Stereo Imager panel: a vectorscope and a correlation meter of the side shown, read
 * from its samples, following the playhead while it plays.
 */

import { app } from "../../scripts/app.js";
import { createGraphSection } from "./interface/audio_graph.js";
import { drawLabel, mountAudioPanel } from "./interface/audio_view.js";

const EXT_NAME = "WASNodeSuite.AudioStereoImager";
const NODE_ID = "WASAudioStereoImager";

// Samples drawn around the playhead while playing, and at most across a whole clip at rest.
const LIVE_SAMPLES = 4096;
const STILL_SAMPLES = 24000;

/** Correlation of two channels over a span, 1 for mono. */
function correlation(left, right, first, last) {
  let ll = 0, rr = 0, lr = 0;
  for (let index = first; index < last; index += 1) {
    ll += left[index] * left[index];
    rr += right[index] * right[index];
    lr += left[index] * right[index];
  }
  const spread = Math.sqrt(ll * rr);
  return spread > 0 ? lr / spread : 1;
}

function scopeSection(node) {
  let frame = 0;
  const section = createGraphSection({
    height: 170,
    hint: "Vectorscope: a vertical line is mono, a wide cloud is wide stereo, a horizontal line cancels in mono",
    draw(ctx, size, theme) {
      const panel = node.__wasAudioPanel;
      const held = panel?.clip();
      const frameBox = { width: size.width, height: size.height, top: 0, theme };
      const meterHeight = 16;
      const scope = Math.min(size.width, size.height - meterHeight - 6);
      const cx = size.width / 2;
      const cy = (size.height - meterHeight - 6) / 2;
      const radius = scope / 2 - 6;
      ctx.strokeStyle = theme.border;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(cx - radius, cy);
      ctx.lineTo(cx + radius, cy);
      ctx.moveTo(cx, cy - radius);
      ctx.lineTo(cx, cy + radius);
      ctx.moveTo(cx - radius * 0.7, cy - radius * 0.7);
      ctx.lineTo(cx + radius * 0.7, cy + radius * 0.7);
      ctx.moveTo(cx + radius * 0.7, cy - radius * 0.7);
      ctx.lineTo(cx - radius * 0.7, cy + radius * 0.7);
      ctx.globalAlpha = 0.5;
      ctx.stroke();
      ctx.globalAlpha = 1;
      if (!held) {
        drawLabel(ctx, frameBox, "run to see the image", cx - 50, cy, { muted: true });
        return;
      }
      const left = held.channels[0];
      const right = held.channels[1] ?? held.channels[0];
      const playing = panel.player.playing();
      const at = Math.round(panel.playhead() * held.rate);
      const first = playing ? Math.max(0, at - LIVE_SAMPLES) : 0;
      const last = playing ? Math.min(held.length, at) : held.length;
      const count = Math.max(1, last - first);
      const step = playing ? 1 : Math.max(1, Math.floor(count / STILL_SAMPLES));
      let loudest = 1e-9;
      for (let index = first; index < last; index += step) loudest = Math.max(loudest, Math.abs(left[index]), Math.abs(right[index]));
      const gain = (radius / loudest) * 0.7;
      ctx.fillStyle = theme.accent;
      ctx.globalAlpha = playing ? 0.5 : 0.25;
      for (let index = first; index < last; index += step) {
        const side = (left[index] - right[index]) * Math.SQRT1_2;
        const mid = (left[index] + right[index]) * Math.SQRT1_2;
        ctx.fillRect(cx + side * gain - 0.75, cy - mid * gain - 0.75, 1.5, 1.5);
      }
      ctx.globalAlpha = 1;
      drawLabel(ctx, frameBox, "M", cx + 4, cy - radius + 8, { muted: true });
      drawLabel(ctx, frameBox, "L", cx - radius * 0.7 - 14, cy - radius * 0.7 + 4, { muted: true });
      drawLabel(ctx, frameBox, "R", cx + radius * 0.7 + 2, cy - radius * 0.7 + 4, { muted: true });
      const value = correlation(left, right, first, last);
      const barTop = size.height - meterHeight;
      const barLeft = 30;
      const barWidth = size.width - barLeft - 46;
      ctx.fillStyle = theme.bg;
      ctx.fillRect(barLeft, barTop + 4, barWidth, meterHeight - 8);
      const mark = barLeft + ((value + 1) / 2) * barWidth;
      ctx.fillStyle = value < 0 ? theme.error : value < 0.3 ? theme.warning : theme.success;
      ctx.fillRect(Math.min(mark, barLeft + barWidth / 2), barTop + 4, Math.abs(mark - (barLeft + barWidth / 2)), meterHeight - 8);
      ctx.fillStyle = theme.fg;
      ctx.fillRect(mark - 1, barTop + 2, 2, meterHeight - 4);
      drawLabel(ctx, frameBox, "-1", 2, barTop + meterHeight / 2, { muted: true });
      drawLabel(ctx, frameBox, `+1  ${value.toFixed(2)}`, size.width - 2, barTop + meterHeight / 2, { align: "right" });
    },
  });
  const loop = () => {
    frame = 0;
    section.repaint();
    if (node.__wasAudioPanel?.player.playing()) frame = requestAnimationFrame(loop);
  };
  return {
    element: section.element,
    update: () => section.repaint(),
    follow: () => {
      if (!frame) frame = requestAnimationFrame(loop);
    },
    dispose: () => {
      if (frame) cancelAnimationFrame(frame);
      section.dispose();
    },
  };
}

function imagerFigures(report) {
  const facts = report?.stereo;
  if (!facts) return undefined;
  return `correlation ${facts.correlation_in.toFixed(2)} to ${facts.correlation_out.toFixed(2)} · `
    + `width ${facts.width_in.toFixed(2)} to ${facts.width_out.toFixed(2)}`;
}

app.registerExtension({
  name: EXT_NAME,
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_ID) return;
    const proto = nodeType.prototype;
    if (proto.__was_audio_stereo_imager_wrapped) return;
    proto.__was_audio_stereo_imager_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      try {
        const node = this;
        let scope = null;
        const built = mountAudioPanel(this, {
          initial: "out",
          height: 380,
          figures: imagerFigures,
          extra: () => {
            scope = scopeSection(node);
            return scope;
          },
          onPlay: () => scope?.follow(),
        });
        if (built) node.__wasAudioPanel = built.panel;
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to build the stereo panel:`, error);
      }
      return result;
    };
  },
});
