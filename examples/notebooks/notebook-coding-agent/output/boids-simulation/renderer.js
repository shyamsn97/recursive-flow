export function render(ctx, boids) {
  // Slight alpha clear for motion trails
  const canvas = ctx.canvas;
  const w = canvas.width;
  const h = canvas.height;

  // Draw a translucent black rectangle to produce motion trails.
  // Alpha tuned for a visible but short trail; adjust if needed.
  ctx.fillStyle = 'rgba(0,0,0,0.12)';
  ctx.fillRect(0, 0, w, h);

  // Local references for speed
  const n = (typeof boids.n === 'number') ? boids.n : boids.x.length;
  const xs = boids.x, ys = boids.y, vxs = boids.vx, vys = boids.vy;
  const cols = boids.colors;

  // Size of boid in pixels (nose length). 6..8 recommended.
  const s = 7;
  const backFactor = 0.6 * s;    // how far back the base is from the position
  const sideFactor = 0.45 * s;   // half-width of the base

  // Render each boid as a filled triangle oriented along velocity
  for (let i = 0; i < n; i++) {
    const x = xs[i];
    const y = ys[i];
    const vx = vxs[i];
    const vy = vys[i];

    // Normalize velocity to get orientation. Fallback to up vector if zero-speed.
    let len = Math.hypot(vx, vy);
    let ux, uy;
    if (len > 1e-6) {
      ux = vx / len;
      uy = vy / len;
    } else {
      ux = 0;
      uy = -1;
    }

    // Perpendicular vector
    const px = -uy;
    const py = ux;

    // Compute triangle points
    const nx = x + ux * s;
    const ny = y + uy * s;

    const lx = x - ux * backFactor + px * sideFactor;
    const ly = y - uy * backFactor + py * sideFactor;

    const rx = x - ux * backFactor - px * sideFactor;
    const ry = y - uy * backFactor - py * sideFactor;

    // Draw one boid path and fill with its color
    ctx.fillStyle = cols[i];
    ctx.beginPath();
    ctx.moveTo(nx, ny);
    ctx.lineTo(lx, ly);
    ctx.lineTo(rx, ry);
    ctx.closePath();
    ctx.fill();
  }
}
