// stereogram.js
// Core texture-mapped Single Image Stereogram (SIS) generator.
//
// Algorithm: the classic Thimbleby–Inglis–Witten separation method
// ("Displaying 3D Images: Algorithms for Single Image Random Dot Stereograms").
//
// Depth convention: White (Z=1) = near (pops toward viewer); black = far.
//
// RULES (measured from the reference stereograms in references/):
//   - Background separation = 13% of output width (~7.6 repeats across).
//   - Depth strength mu = 0.36.
//
// TEXTURE: the pattern canvas is used only as a COLOR SOURCE. Its palette is
// extracted and re-laid as random dots at non-grid positions, so the lookup
// canvas has NO internal periodicity. The only autocorrelation peak in the
// output is the one introduced by the stereo algorithm (at the separation s0
// for the background, smaller inside raised regions). This matches how the
// reference stereograms work and prevents false/inverted fusion.

export const SEP_FRACTION = 0.13;
export const DEPTH_MU = 0.36;

/**
 * Generate a stereogram into an output canvas.
 *
 * @param {CanvasImageSource & {width:number,height:number}} patternCanvas - color source
 * @param {CanvasImageSource & {width:number,height:number}} depthCanvas   - grayscale depth map
 * @param {HTMLCanvasElement} outCanvas - destination (resized in place)
 * @param {Object} opts
 * @param {number}  [opts.width=800]
 * @param {number}  [opts.height=600]
 * @param {number}  [opts.patternRepeats=1] - for uploaded: copies per band; for generated: dot-size
 * @param {boolean} [opts.aperiodicTexture=false] - true = generated (random dots); false = uploaded (use actual texture)
 * @param {boolean} [opts.invert=false]
 * @param {boolean} [opts.popIn=false]
 */
export function generateStereogram(patternCanvas, depthCanvas, outCanvas, opts = {}) {
  const width  = Math.max(1, Math.round(opts.width  || 800));
  const height = Math.max(1, Math.round(opts.height || 600));
  const mu     = opts.mu != null ? Math.max(0.05, Math.min(0.6, Number(opts.mu))) : DEPTH_MU;
  const eyeSep = Math.max(4, Math.round(2 * SEP_FRACTION * width));
  const reps      = Math.max(1, Math.round(opts.patternRepeats || 1));
  const aperiodic = !!opts.aperiodicTexture;
  const invert = !!opts.invert;
  const popIn  = !!opts.popIn;
  const blurSigma  = opts.depthBlur  != null ? Number(opts.depthBlur)  : 2.5;
  const borderPx   = opts.borderPx   != null ? Math.round(opts.borderPx) : 0;

  // --- Depth map -> normalized Z [0,1] per pixel ----------------------------
  const depthData = sampleToImageData(depthCanvas, width, height);
  let depth = new Float32Array(width * height);
  {
    const d = depthData.data;
    for (let i = 0, p = 0; i < depth.length; i++, p += 4) {
      let z = (0.299 * d[p] + 0.587 * d[p + 1] + 0.114 * d[p + 2]) / 255;
      if (invert) z = 1 - z;
      if (popIn)  z = 1 - z;
      depth[i] = z;
    }
  }

  // Gaussian blur on depth: smooths abrupt period-change boundaries that cause
  // the eye to lock onto a false stereo phase before the correct one.
  if (blurSigma > 0) {
    depth = gaussianBlurDepth(depth, width, height, blurSigma);
  }

  // Background border: force outer ring to Z=0 (background) so the eye always
  // has a large flat anchor region to fuse on before moving to the subject.
  if (borderPx > 0) {
    const bx = Math.min(borderPx, Math.floor(width  / 2));
    const by = Math.min(borderPx, Math.floor(height / 2));
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        if (x < bx || x >= width - bx || y < by || y >= height - by) {
          depth[y * width + x] = 0;
        }
      }
    }
  }

  // --- Pattern lookup canvas ------------------------------------------------
  // For aperiodic (generated): full-size dot canvas — no tiling, no seed period.
  //   Dots 2–4 px radius give the visual system coherent blobs to lock onto,
  //   unlike 1-px salt-and-pepper noise from per-pixel random colors.
  // For uploaded: band-locked tile at s0 (unchanged).
  const s0        = Math.max(2, separation(0, mu, eyeSep));
  const patLookup = aperiodic
    ? buildAperiodicDotTexture(patternCanvas, width, height)
    : buildPatternLookup(patternCanvas, width, height, s0, reps, false);
  const pat = patLookup.data;

  // --- Stereogram output ----------------------------------------------------
  outCanvas.width  = width;
  outCanvas.height = height;
  const octx   = outCanvas.getContext('2d');
  const outImg = octx.createImageData(width, height);
  const out    = outImg.data;

  const same = new Int32Array(width);

  for (let y = 0; y < height; y++) {
    const row = y * width;

    for (let x = 0; x < width; x++) same[x] = x;

    for (let x = 0; x < width; x++) {
      const z   = depth[row + x];
      const sep = separation(z, mu, eyeSep);

      const left  = x - (sep >> 1);
      const right = left + sep;

      if (left >= 0 && right < width) {
        let visible = true;
        let t = 1;
        do {
          const zt = z + (2 * (2 - mu * z) * t) / (mu * eyeSep);
          const xl = x - t, xr = x + t;
          visible = (xl < 0 || depth[row + xl] < zt) &&
                    (xr >= width || depth[row + xr] < zt);
          t++;
          if (zt >= 1) break;
        } while (visible);

        if (visible) same[left] = right;
      }
    }

    // Resolve right -> left: constrained pixels copy from their linked partner;
    // unconstrained pixels seed from the aperiodic lookup.
    for (let x = width - 1; x >= 0; x--) {
      const oi = (row + x) << 2;
      if (same[x] === x) {
        const pi = (row + x) << 2;
        out[oi]     = pat[pi];
        out[oi + 1] = pat[pi + 1];
        out[oi + 2] = pat[pi + 2];
      } else {
        const si = (row + same[x]) << 2;
        out[oi]     = out[si];
        out[oi + 1] = out[si + 1];
        out[oi + 2] = out[si + 2];
      }
      out[oi + 3] = 255;
    }
  }

  octx.putImageData(outImg, 0, 0);
  return outCanvas;
}

// --- helpers ---------------------------------------------------------------

function separation(z, mu, eyeSep) {
  return Math.round(((1 - mu * z) * eyeSep) / (2 - mu * z));
}

function sampleToImageData(src, w, h) {
  const c   = document.createElement('canvas');
  c.width   = w;
  c.height  = h;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(src, 0, 0, w, h);
  return ctx.getImageData(0, 0, w, h);
}

/**
 * Build the pattern lookup canvas.
 *
 * Two paths:
 *  aperiodic=true  (Generate Pattern): extract palette, render random fine dots.
 *                  Only the stereo algorithm's links create periodicity → clean fusion.
 *  aperiodic=false (uploaded image):   tile the actual texture, band-width locked to
 *                  s0 so the ONLY horizontal period is the stereo separation (no ghosting).
 *                  `reps` = how many copies of the pattern fit within one band.
 */
function buildPatternLookup(patternCanvas, w, h, period, reps, aperiodic) {
  const c   = document.createElement('canvas');
  c.width   = w;
  c.height  = h;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  const s0  = Math.max(1, Math.round(period));

  if (aperiodic) {
    // --- Generated pattern: aperiodic random-dot strip tiled in X only -------
    const palette = samplePalette(patternCanvas, 500);
    const strip   = aperiodicStrip(palette, s0, h, reps);
    ctx.fillStyle = ctx.createPattern(strip, 'repeat-x');
    ctx.fillRect(0, 0, w, h);

  } else {
    // --- Uploaded pattern: tile the real texture, band-locked to s0 ----------
    // The tile is exactly s0 wide so the pattern's own horizontal period equals
    // the stereo separation — no competing peaks, no ghosting.
    const copyW = s0 / reps;
    const copyH = Math.max(1, Math.round((patternCanvas.height * copyW) / patternCanvas.width));

    const tile  = document.createElement('canvas');
    tile.width  = s0;
    tile.height = copyH;
    const tctx  = tile.getContext('2d');
    tctx.imageSmoothingEnabled = true;
    for (let i = 0; i < reps; i++) {
      tctx.drawImage(patternCanvas, i * copyW, 0, copyW, copyH);
    }

    ctx.fillStyle = ctx.createPattern(tile, 'repeat');
    ctx.fillRect(0, 0, w, h);
  }

  return ctx.getImageData(0, 0, w, h);
}

/**
 * Render random colored dots at uniformly random positions onto a (w × h) canvas.
 * No grid, no internal period — only the color palette comes from the source pattern.
 */
function aperiodicStrip(palette, w, h, reps) {
  const c   = document.createElement('canvas');
  c.width   = w;
  c.height  = h;
  const ctx = c.getContext('2d');

  // Background: darkest color in palette
  const bg = palette.reduce((a, b) => (a[0]+a[1]+a[2]) < (b[0]+b[1]+b[2]) ? a : b);
  ctx.fillStyle = `rgb(${bg[0]},${bg[1]},${bg[2]})`;
  ctx.fillRect(0, 0, w, h);

  // Dot radius: stay close to reference noise grain (1–3 px).
  // reps=1 → up to 3px; reps=6 → up to 1.5px (floor). Higher reps = finer.
  const maxR = Math.max(1.5, 3 / reps);
  const minR = 1;
  const avgR = (minR + maxR) / 2;
  const count = Math.round((w * h) / (Math.PI * avgR * avgR * 1.6));

  for (let i = 0; i < count; i++) {
    const col = palette[Math.floor(Math.random() * palette.length)];
    ctx.fillStyle = `rgb(${col[0]},${col[1]},${col[2]})`;
    ctx.beginPath();
    ctx.arc(
      Math.random() * w,
      Math.random() * h,
      minR + Math.random() * (maxR - minR),
      0, Math.PI * 2
    );
    ctx.fill();
  }
  return c;
}

/**
 * Separable Gaussian blur on a Float32Array depth field (values 0..1).
 * Smooths abrupt period-change edges to reduce false stereo-phase locking.
 */
function gaussianBlurDepth(data, w, h, sigma) {
  const radius = Math.ceil(sigma * 2.5);
  const kernel = [];
  let ksum = 0;
  for (let i = -radius; i <= radius; i++) {
    const v = Math.exp(-(i * i) / (2 * sigma * sigma));
    kernel.push(v);
    ksum += v;
  }
  for (let i = 0; i < kernel.length; i++) kernel[i] /= ksum;

  // Horizontal pass → tmp
  const tmp = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let acc = 0;
      for (let k = -radius; k <= radius; k++) {
        const xi = Math.min(w - 1, Math.max(0, x + k));
        acc += data[y * w + xi] * kernel[k + radius];
      }
      tmp[y * w + x] = acc;
    }
  }

  // Vertical pass → out
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let acc = 0;
      for (let k = -radius; k <= radius; k++) {
        const yi = Math.min(h - 1, Math.max(0, y + k));
        acc += tmp[yi * w + x] * kernel[k + radius];
      }
      out[y * w + x] = acc;
    }
  }
  return out;
}

/**
 * Build a full-size (w × h) aperiodic dot texture from the pattern's palette.
 * Dots are 2–4 px radius — large enough for the visual system to lock onto,
 * small enough not to create internal structure.  No tiling, so no seed period.
 */
function buildAperiodicDotTexture(patternCanvas, w, h) {
  const palette = samplePalette(patternCanvas, 500);
  const c   = document.createElement('canvas');
  c.width   = w;
  c.height  = h;
  const ctx = c.getContext('2d', { willReadFrequently: true });

  const bg = palette.reduce((a, b) => (a[0] + a[1] + a[2]) < (b[0] + b[1] + b[2]) ? a : b);
  ctx.fillStyle = `rgb(${bg[0]},${bg[1]},${bg[2]})`;
  ctx.fillRect(0, 0, w, h);

  const minR = 1, maxR = 2.5;
  const avgR = (minR + maxR) / 2;
  const count = Math.round((w * h) / (Math.PI * avgR * avgR * 2.5));

  for (let i = 0; i < count; i++) {
    const col = palette[Math.floor(Math.random() * palette.length)];
    ctx.fillStyle = `rgb(${col[0]},${col[1]},${col[2]})`;
    ctx.beginPath();
    ctx.arc(Math.random() * w, Math.random() * h, minR + Math.random() * (maxR - minR), 0, Math.PI * 2);
    ctx.fill();
  }

  return ctx.getImageData(0, 0, w, h);
}

/**
 * Sample n colors from src as [r,g,b] triples.
 * Used to extract the color palette for the aperiodic dot texture.
 */
export function samplePalette(src, n) {
  const c   = document.createElement('canvas');
  const f   = Math.min(1, Math.sqrt(n / (src.width * src.height + 1)));
  c.width   = Math.max(1, Math.round(src.width  * f));
  c.height  = Math.max(1, Math.round(src.height * f));
  c.getContext('2d').drawImage(src, 0, 0, c.width, c.height);
  const d   = c.getContext('2d', { willReadFrequently: true })
               .getImageData(0, 0, c.width, c.height).data;
  const out = [];
  const stride = Math.max(1, Math.floor(d.length / 4 / n));
  for (let i = 0; i < d.length; i += stride * 4) {
    out.push([d[i], d[i + 1], d[i + 2]]);
  }
  return out.length ? out : [[180, 100, 40]];
}
