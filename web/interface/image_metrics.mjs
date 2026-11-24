/**
 * What a picture is, and how two of them differ, measured from the pixels.
 *
 * Pure arithmetic over 8-bit channels, with no DOM beyond the canvas used to read pixels out of
 * an image.
 */

const CHANNELS = ["r", "g", "b"];

// SSIM's own constants, from Wang et al. `L` is the dynamic range of an 8-bit channel.
const SSIM_L = 255;
const SSIM_K1 = 0.01;
const SSIM_K2 = 0.03;

// The window SSIM is measured over: an 11 wide gaussian of sigma 1.5, as the paper specifies.
const SSIM_RADIUS = 5;
const SSIM_SIGMA = 1.5;

// Bins per channel in a histogram, unless a caller asks for another number. 64 is enough to see
// a clipped highlight or a lifted black without becoming noise at the width a node is drawn.
export const HISTOGRAM_BINS = 64;

/**
 * Read an image's pixels, optionally resampled to another size.
 *
 * @param {CanvasImageSource} image - The decoded picture.
 * @param {number} width - Width to read at.
 * @param {number} height - Height to read at.
 * @returns {Uint8ClampedArray|null} RGBA bytes, or null when the pixels cannot be read.
 */
export function readPixels(image, width, height) {
  const w = Math.max(1, Math.round(width));
  const h = Math.max(1, Math.round(height));
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) return null;
  ctx.drawImage(image, 0, 0, w, h);
  try {
    return ctx.getImageData(0, 0, w, h).data;
  } catch (error) {
    // A picture from another origin taints the canvas and cannot be read back. The pack serves
    // its own previews, so this is a misconfiguration rather than something to work around.
    console.error("[WASNodeSuite.ImageMetrics] The pixels could not be read:", error);
    return null;
  }
}

/**
 * Rec. 601 luma, which is what SSIM and the sharpness figure are measured on.
 *
 * @param {Uint8ClampedArray} rgba - RGBA bytes.
 * @returns {Float32Array} One luma sample per pixel.
 */
export function luma(rgba) {
  const out = new Float32Array(rgba.length / 4);
  for (let i = 0, p = 0; i < rgba.length; i += 4, p += 1) {
    out[p] = 0.299 * rgba[i] + 0.587 * rgba[i + 1] + 0.114 * rgba[i + 2];
  }
  return out;
}

/**
 * A per-channel histogram.
 *
 * @param {Uint8ClampedArray} rgba - RGBA bytes.
 * @param {number} [bins] - How many bins per channel.
 * @returns {{r: Float64Array, g: Float64Array, b: Float64Array, peak: number}} Counts per
 *   channel and the largest count in any of them, for scaling a plot.
 */
export function histogram(rgba, bins = HISTOGRAM_BINS) {
  const count = Math.max(2, Math.round(bins));
  const out = { r: new Float64Array(count), g: new Float64Array(count), b: new Float64Array(count) };
  const scale = count / 256;
  for (let i = 0; i < rgba.length; i += 4) {
    out.r[Math.min(count - 1, (rgba[i] * scale) | 0)] += 1;
    out.g[Math.min(count - 1, (rgba[i + 1] * scale) | 0)] += 1;
    out.b[Math.min(count - 1, (rgba[i + 2] * scale) | 0)] += 1;
  }
  let peak = 0;
  for (const channel of CHANNELS) {
    for (const value of out[channel]) if (value > peak) peak = value;
  }
  out.peak = peak;
  return out;
}

/**
 * One channel of sRGB as CIE L*a*b*, under D65.
 *
 * @param {number} r - Red, 0 to 255.
 * @param {number} g - Green, 0 to 255.
 * @param {number} b - Blue, 0 to 255.
 * @returns {number[]} L, a and b.
 */
function toLab(r, g, b) {
  const linear = (value) => {
    const v = value / 255;
    return v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4;
  };
  const rl = linear(r), gl = linear(g), bl = linear(b);
  // sRGB to XYZ, then normalised against the D65 white point.
  const x = (0.4124564 * rl + 0.3575761 * gl + 0.1804375 * bl) / 0.95047;
  const y = 0.2126729 * rl + 0.7151522 * gl + 0.0721750 * bl;
  const z = (0.0193339 * rl + 0.1191920 * gl + 0.9503041 * bl) / 1.08883;
  const f = (t) => (t > 0.008856451679035631 ? Math.cbrt(t) : 7.787037037037035 * t + 16 / 116);
  const fx = f(x), fy = f(y), fz = f(z);
  return [116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)];
}

/**
 * Blur one plane with a separable gaussian, clamping at the edges.
 *
 * @param {Float32Array} plane - The samples.
 * @param {number} width - Plane width.
 * @param {number} height - Plane height.
 * @param {number[]} kernel - The 1D kernel, already normalised.
 * @returns {Float32Array} The blurred plane.
 */
function blur(plane, width, height, kernel) {
  const radius = (kernel.length - 1) / 2;
  const pass = new Float32Array(plane.length);
  const out = new Float32Array(plane.length);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      let sum = 0;
      for (let k = -radius; k <= radius; k += 1) {
        const sx = Math.min(width - 1, Math.max(0, x + k));
        sum += plane[y * width + sx] * kernel[k + radius];
      }
      pass[y * width + x] = sum;
    }
  }
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      let sum = 0;
      for (let k = -radius; k <= radius; k += 1) {
        const sy = Math.min(height - 1, Math.max(0, y + k));
        sum += pass[sy * width + x] * kernel[k + radius];
      }
      out[y * width + x] = sum;
    }
  }
  return out;
}

/**
 * The gaussian SSIM is windowed with.
 *
 * @param {number} radius - Half the window, so the window is `2 * radius + 1` wide.
 * @param {number} sigma - Standard deviation.
 * @returns {number[]} The normalised kernel.
 */
function gaussian(radius, sigma) {
  const kernel = [];
  let total = 0;
  for (let i = -radius; i <= radius; i += 1) {
    const value = Math.exp(-(i * i) / (2 * sigma * sigma));
    kernel.push(value);
    total += value;
  }
  return kernel.map((value) => value / total);
}

/**
 * Structural similarity between two luma planes.
 *
 * @param {Float32Array} a - The first plane.
 * @param {Float32Array} b - The second, the same size.
 * @param {number} width - Plane width.
 * @param {number} height - Plane height.
 * @returns {number} The mean SSIM, 1 for identical planes.
 */
export function ssim(a, b, width, height) {
  const kernel = gaussian(SSIM_RADIUS, SSIM_SIGMA);
  const aa = new Float32Array(a.length);
  const bb = new Float32Array(a.length);
  const ab = new Float32Array(a.length);
  for (let i = 0; i < a.length; i += 1) {
    aa[i] = a[i] * a[i];
    bb[i] = b[i] * b[i];
    ab[i] = a[i] * b[i];
  }
  const muA = blur(a, width, height, kernel);
  const muB = blur(b, width, height, kernel);
  const sAA = blur(aa, width, height, kernel);
  const sBB = blur(bb, width, height, kernel);
  const sAB = blur(ab, width, height, kernel);

  const c1 = (SSIM_K1 * SSIM_L) ** 2;
  const c2 = (SSIM_K2 * SSIM_L) ** 2;
  let total = 0;
  for (let i = 0; i < a.length; i += 1) {
    const ma = muA[i], mb = muB[i];
    const va = sAA[i] - ma * ma;
    const vb = sBB[i] - mb * mb;
    const cov = sAB[i] - ma * mb;
    total += (((2 * ma * mb + c1) * (2 * cov + c2))
      / ((ma * ma + mb * mb + c1) * (va + vb + c2)));
  }
  return total / a.length;
}

/**
 * What one picture is, on its own.
 *
 * @param {Uint8ClampedArray} rgba - RGBA bytes.
 * @param {number} width - Picture width.
 * @param {number} height - Picture height.
 * @param {number} [bins] - Histogram bins per channel.
 * @returns {object} Per-channel mean and standard deviation, the luma range, the entropy of the
 *   luma distribution in bits, a sharpness figure, and the histogram.
 */
export function describeOne(rgba, width, height, bins = HISTOGRAM_BINS) {
  const pixels = rgba.length / 4;
  const sums = [0, 0, 0];
  const squares = [0, 0, 0];
  for (let i = 0; i < rgba.length; i += 4) {
    for (let c = 0; c < 3; c += 1) {
      const value = rgba[i + c];
      sums[c] += value;
      squares[c] += value * value;
    }
  }
  const mean = {};
  const deviation = {};
  CHANNELS.forEach((name, c) => {
    mean[name] = sums[c] / pixels;
    deviation[name] = Math.sqrt(Math.max(0, squares[c] / pixels - mean[name] ** 2));
  });

  const plane = luma(rgba);
  let low = 255, high = 0, lumaSum = 0;
  const fine = new Float64Array(256);
  for (let i = 0; i < plane.length; i += 1) {
    const value = plane[i];
    if (value < low) low = value;
    if (value > high) high = value;
    lumaSum += value;
    fine[Math.min(255, value | 0)] += 1;
  }
  let entropy = 0;
  for (const count of fine) {
    if (count > 0) {
      const p = count / plane.length;
      entropy -= p * Math.log2(p);
    }
  }

  // Mean absolute laplacian: higher on a picture with more edge energy, which is the useful
  // half of "is this one sharper than that one" without claiming to be a focus measure.
  let edges = 0;
  let counted = 0;
  for (let y = 1; y < height - 1; y += 1) {
    for (let x = 1; x < width - 1; x += 1) {
      const i = y * width + x;
      edges += Math.abs(4 * plane[i] - plane[i - 1] - plane[i + 1]
        - plane[i - width] - plane[i + width]);
      counted += 1;
    }
  }

  return {
    width, height, pixels,
    mean, deviation,
    lumaMean: lumaSum / plane.length,
    lumaMin: low, lumaMax: high,
    entropy,
    sharpness: counted ? edges / counted : 0,
    histogram: histogram(rgba, bins),
  };
}

/**
 * How two pictures differ.
 *
 * @param {Uint8ClampedArray} a - RGBA bytes of the first.
 * @param {Uint8ClampedArray} b - RGBA bytes of the second, the same size.
 * @param {number} width - Width of both.
 * @param {number} height - Height of both.
 * @returns {{rmse: number, mae: number, psnr: number, deltaE: number, ssim: number}} The
 *   differences, with `psnr` in decibels and `Infinity` where the two are identical.
 */
export function comparePair(a, b, width, height) {
  const pixels = a.length / 4;
  let squared = 0;
  let absolute = 0;
  let labTotal = 0;
  for (let i = 0; i < a.length; i += 4) {
    for (let c = 0; c < 3; c += 1) {
      const delta = a[i + c] - b[i + c];
      squared += delta * delta;
      absolute += Math.abs(delta);
    }
    const [l1, a1, b1] = toLab(a[i], a[i + 1], a[i + 2]);
    const [l2, a2, b2] = toLab(b[i], b[i + 1], b[i + 2]);
    labTotal += Math.sqrt((l1 - l2) ** 2 + (a1 - a2) ** 2 + (b1 - b2) ** 2);
  }
  const mse = squared / (pixels * 3);
  return {
    rmse: Math.sqrt(mse),
    mae: absolute / (pixels * 3),
    psnr: mse === 0 ? Infinity : 10 * Math.log10((SSIM_L * SSIM_L) / mse),
    deltaE: labTotal / pixels,
    ssim: ssim(luma(a), luma(b), width, height),
  };
}
