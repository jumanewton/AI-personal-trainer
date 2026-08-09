/**
 * pose.js — Canvas skeleton renderer for AI Personal Trainer replay viewer.
 *
 * Draws a colour-coded MediaPipe pose skeleton from normalised landmark data
 * loaded from the session CSV.
 *
 * Exported function (used by index.html):
 *   drawPose(canvas, row)   — renders a single frame from a CSV row object
 */

// ─────────────────────────────────────────────────────────────────────────────
// MediaPipe landmark connections (33-point model)
// ─────────────────────────────────────────────────────────────────────────────
const POSE_CONNECTIONS = [
  // Face
  [0, 1], [1, 2], [2, 3], [3, 7],
  [0, 4], [4, 5], [5, 6], [6, 8],
  [9, 10],
  // Left arm
  [11, 13], [13, 15], [15, 17], [15, 19], [15, 21], [17, 19],
  // Right arm
  [12, 14], [14, 16], [16, 18], [16, 20], [16, 22], [18, 20],
  // Torso
  [11, 12], [11, 23], [12, 24], [23, 24],
  // Left leg
  [23, 25], [25, 27], [27, 29], [27, 31], [29, 31],
  // Right leg
  [24, 26], [26, 28], [28, 30], [28, 32], [30, 32],
];

// Colour by body segment (index set)
const SEG = {
  face:      new Set([0,1,2,3,4,5,6,7,8,9,10]),
  leftArm:   new Set([11,13,15,17,19,21]),
  rightArm:  new Set([12,14,16,18,20,22]),
  torso:     new Set([11,12,23,24]),
  leftLeg:   new Set([23,25,27,29,31]),
  rightLeg:  new Set([24,26,28,30,32]),
};

const SEG_COLORS = {
  face:     "#94a3b8",
  leftArm:  "#38bdf8",
  rightArm:  "#818cf8",
  torso:    "#fbbf24",
  leftLeg:  "#34d399",
  rightLeg: "#a78bfa",
};

function segmentColor(a, b) {
  for (const [seg, set] of Object.entries(SEG)) {
    if (set.has(a) || set.has(b)) return SEG_COLORS[seg];
  }
  return "#6b6b80";
}

function jointColor(idx) {
  for (const [seg, set] of Object.entries(SEG)) {
    if (set.has(idx)) return SEG_COLORS[seg];
  }
  return "#6b6b80";
}

// ─────────────────────────────────────────────────────────────────────────────
// Parse landmarks from a CSV row object
// ─────────────────────────────────────────────────────────────────────────────
function parseLandmarks(row) {
  const pts = [];
  for (let i = 0; i < 33; i++) {
    const x = parseFloat(row[`img_${i}_x`]);
    const y = parseFloat(row[`img_${i}_y`]);
    if (!isNaN(x) && !isNaN(y)) {
      pts.push({ x, y });
    } else {
      pts.push(null);
    }
  }
  return pts;
}

// ─────────────────────────────────────────────────────────────────────────────
// Main draw function
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Render a single pose frame onto the given canvas element.
 *
 * @param {HTMLCanvasElement} canvas
 * @param {Object} row  — a single row object from the session CSV (parsed to numbers)
 */
function drawPose(canvas, row) {
  // Size canvas to its display size
  const rect = canvas.getBoundingClientRect();
  const dpr  = window.devicePixelRatio || 1;
  const w    = rect.width  || canvas.parentElement.clientWidth;
  const h    = rect.height || canvas.parentElement.clientHeight;

  if (canvas.width !== w * dpr) {
    canvas.width  = w * dpr;
    canvas.height = h * dpr;
    canvas.style.width  = w + "px";
    canvas.style.height = h + "px";
  }

  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  // ── Background ────────────────────────────────────────────────────────
  ctx.clearRect(0, 0, w, h);

  // Radial glow background
  const grad = ctx.createRadialGradient(w/2, h*0.55, 0, w/2, h*0.55, w * 0.55);
  grad.addColorStop(0,   "rgba(0, 212, 255, 0.04)");
  grad.addColorStop(0.5, "rgba(124, 58, 237, 0.02)");
  grad.addColorStop(1,   "rgba(9, 9, 15, 0)");
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, w, h);

  // ── Landmarks ─────────────────────────────────────────────────────────
  const pts = parseLandmarks(row);
  if (!pts.every(p => p === null)) {
    // Map normalised [0,1] coords into canvas with margin
    const margin = 40;
    const cw = w - margin * 2;
    const ch = h - margin * 2;

    function toCanvasX(nx) { return margin + nx * cw; }
    function toCanvasY(ny) { return margin + ny * ch; }

    // ── Connections ─────────────────────────────────────────────────────
    POSE_CONNECTIONS.forEach(([a, b]) => {
      const pa = pts[a];
      const pb = pts[b];
      if (!pa || !pb) return;

      const cx1 = toCanvasX(pa.x), cy1 = toCanvasY(pa.y);
      const cx2 = toCanvasX(pb.x), cy2 = toCanvasY(pb.y);

      const color = segmentColor(a, b);
      ctx.save();
      ctx.beginPath();
      ctx.moveTo(cx1, cy1);
      ctx.lineTo(cx2, cy2);
      ctx.strokeStyle = color;
      ctx.lineWidth   = 2.5;
      ctx.globalAlpha = 0.85;
      ctx.lineCap     = "round";
      ctx.stroke();
      ctx.restore();
    });

    // ── Joints ──────────────────────────────────────────────────────────
    pts.forEach((pt, idx) => {
      if (!pt) return;
      const cx = toCanvasX(pt.x);
      const cy = toCanvasY(pt.y);

      const color = jointColor(idx);
      const r = SEG.face.has(idx) ? 3 : 5;

      // Glow
      ctx.save();
      ctx.beginPath();
      ctx.arc(cx, cy, r + 4, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.globalAlpha = 0.12;
      ctx.fill();
      ctx.restore();

      // Circle
      ctx.save();
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.globalAlpha = 0.95;
      ctx.fill();
      ctx.restore();

      // White outline
      ctx.save();
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.strokeStyle = "rgba(255,255,255,0.25)";
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.restore();
    });

    // ── Key angle annotations ────────────────────────────────────────────
    drawAngleAnnotation(ctx, row, pts, toCanvasX, toCanvasY);
  } else {
    // No person detected in this frame
    ctx.fillStyle = "#6b6b80";
    ctx.font = "14px Outfit, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("No pose detected", w / 2, h / 2);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Angle annotation helpers
// ─────────────────────────────────────────────────────────────────────────────

function drawAngleAnnotation(ctx, row, pts, toX, toY) {
  const score = parseFloat(row.form_score) || 100;
  const textColor = score < 60 ? "#ef4444" : score < 80 ? "#f59e0b" : "#22c55e";

  const annotations = [
    { lm: 25, key: "angle_left_knee",  label: "Knee L" },
    { lm: 26, key: "angle_right_knee", label: "Knee R" },
    { lm: 23, key: "angle_left_hip",   label: "Hip" },
  ];

  annotations.forEach(({ lm, key, label }) => {
    const pt = pts[lm];
    const val = parseFloat(row[key]);
    if (!pt || isNaN(val)) return;

    const cx = toX(pt.x) + 10;
    const cy = toY(pt.y) - 6;

    ctx.save();
    ctx.font = "bold 11px Outfit, sans-serif";
    ctx.fillStyle = textColor;
    ctx.globalAlpha = 0.9;
    ctx.fillText(`${val.toFixed(0)}°`, cx, cy);
    ctx.restore();
  });
}
