export const SPEED = 240;

const MAX_TURN_RATE = Math.PI * 2.0; // radians per second (clamped by dt)
const PERCEPTION = 48; // neighbor radius in px
const SEP_WEIGHT = 1.6;
const ALIGN_WEIGHT = 1.0;
const COH_WEIGHT = 1.0;

function clamp(v, a, b) {
  return v < a ? a : v > b ? b : v;
}

function angDiff(a, b) {
  // smallest signed difference a->b
  let d = b - a;
  while (d <= -Math.PI) d += Math.PI * 2;
  while (d > Math.PI) d -= Math.PI * 2;
  return d;
}

export class Boid {
  constructor(x, y, angle, hue) {
    this.x = x || 0;
     this.y = y || 0;
    this.angle = (typeof angle === 'number') ? angle : 0; // radians
    this.hue = (typeof hue === 'number') ? hue : 0;
  }

  update(dt, neighbors, width, height) {
    // neighbors: array of objects with x,y,angle (we only need positions and angle)
    let sx = 0, sy = 0; // separation vector (away)
    let ax = 0, ay = 0; // alignment (sum of headings)
    let cx = 0, cy = 0; // cohesion (sum of positions)
    let count = 0;
    const px = this.x, py = this.y;
    for (let i = 0; i < neighbors.length; ++i) {
      const nb = neighbors[i];
      if (!nb) continue;
      // compute toroidal dx/dy (wrap-aware) to account for world edges
      let dx = nb.x - px;
      let dy = nb.y - py;
      // shortest wrap for dx
      if (width) {
        if (dx > width / 2) dx -= width;
        else if (dx < -width / 2) dx += width;
      }
      if (height) {
        if (dy > height / 2) dy -= height;
        else if (dy < -height / 2) dy += height;
      }
      const dist2 = dx * dx + dy * dy;
    const r2 = PERCEPTION * PERCEPTION;
      if (dist2 > 0 && dist2 <= r2) {
        const dist = Math.sqrt(dist2);
        // separation: away from neighbor, stronger when closer
        sx += -dx / (dist + 1e-6) / (dist + 1e-6);
        sy += -dy / (dist + 1e-6) / (dist + 1e-6);
        // alignment: neighbor heading as unit vector
        const ha = nb.angle || 0;
        ax += Math.cos(ha);
        ay += Math.sin(ha);
        // cohesion
        cx += dx;
        cy += dy;
        count += 1;
      }
    }

    // combine steering
    let steerX = 0, steerY = 0;
    if (count > 0) {
      // average alignment
      ax /= count; ay /= count;
      // average cohesion vector (toward average position)
      cx /= count; cy /= count;
      // separation already summed as weighted by inverse distance
      steerX = SEP_WEIGHT * sx + ALIGN_WEIGHT * ax + COH_WEIGHT * cx;
      steerY = SEP_WEIGHT * sy + ALIGN_WEIGHT * ay + COH_WEIGHT * cy;
      // if steering is zero, skip
      const mag = Math.hypot(steerX, steerY);
      if (mag > 1e-6) {
        steerX /= mag; steerY /= mag; // normalize desired direction
      }
    }

    // desired angle (if no neighbors steer stays same direction)
    let desiredAngle = this.angle;
    if (Math.abs(steerX) > 1e-6 || Math.abs(steerY) > 1e-6) {
      desiredAngle = Math.atan2(steerY, steerX);
    }

    // limit turn rate
    const diff = angDiff(this.angle, desiredAngle);
    const maxTurn = MAX_TURN_RATE * dt;
    const clamped = clamp(diff, -maxTurn, maxTurn);
    this.angle += clamped;
    // normalize angle to [-PI,PI)
    while (this.angle <= -Math.PI) this.angle += Math.PI * 2;
    while (this.angle > Math.PI) this.angle -= Math.PI * 2;

    // move at constant speed
    const vx = Math.cos(this.angle) * SPEED;
    const vy = Math.sin(this.angle) * SPEED;
    this.x += vx * dt;
    this.y += vy * dt;

    // wrap/toroidal world
    if (width) {
      if (this.x < 0) this.x += width;
      else if (this.x >= width) this.x -= width;
    }
    if (height) {
      if (this.y < 0) this.y += height;
      else if (this.y >= height) this.y -= height;
    }
  }

  draw(ctx) {
    // draw a small triangle oriented by angle
    const size = 5; // triangle size ~4-6 px
    ctx.save();
    ctx.translate(this.x, this.y);
    ctx.rotate(this.angle);
    ctx.beginPath();
    // point forward
    ctx.moveTo(size, 0);
    ctx.lineTo(-size * 0.6, size * 0.7);
    ctx.lineTo(-size * 0.6, -size * 0.7);
    ctx.closePath();
    ctx.fillStyle = `hsl(${this.hue}, 80%, 60%)`;
    ctx.fill();
    // optional subtle stroke for contrast
    ctx.lineWidth = 0.6;
    ctx.strokeStyle = 'rgba(0,0,0,0.15)';
    ctx.stroke();
    ctx.restore();
  }
}
