import { Vec2 } from './vec2.js';

export class Boid {
  constructor(x, y, opts = {}) {
    this.position = new Vec2(x, y);
    const baseSpeed = opts.speed ?? (180 + Math.random() * 100);
    this.velocity = Vec2.randomUnit().mul(baseSpeed);
    this.acceleration = new Vec2(0, 0);

    this.maxSpeed = opts.maxSpeed ?? (220 + Math.random() * 80);
    this.maxForce = opts.maxForce ?? 220; // max steering accel (px/s^2)

    this.size = opts.size ?? 3.5;
    this.hue = Math.floor(Math.random() * 360);

    this.sepRadius = opts.sepRadius ?? 24;
    this.neighRadius = opts.neighRadius ?? 64;
  }

  applyForce(force) { this.acceleration.add(force); }

  flock(neighbors) {
    const pos = this.position;
    let steerSep = new Vec2(0, 0);
    let steerAli = new Vec2(0, 0);
    let steerCoh = new Vec2(0, 0);
    let countSep = 0, countAli = 0, countCoh = 0;

    const nr2 = this.neighRadius * this.neighRadius;
    const sr2 = this.sepRadius * this.sepRadius;

    for (const other of neighbors) {
      if (other === this) continue;
      const dx = other.position.x - pos.x;
      const dy = other.position.y - pos.y;
      const d2 = dx*dx + dy*dy;

      if (d2 < nr2) {
        // alignment & cohesion
        steerAli.add(other.velocity);
        steerCoh.add(other.position);
        countAli++; countCoh++;
      }

      if (d2 < sr2 && d2 > 1e-6) {
        // separation (inverse square falloff)
        const inv = 1 / d2;
        steerSep.add(new Vec2(-dx * inv, -dy * inv));
        countSep++;
      }
    }

    if (countAli > 0) {
      steerAli.div(countAli);
      steerAli.normalize().mul(this.maxSpeed);
      steerAli.sub(this.velocity).limit(this.maxForce);
    }
    if (countCoh > 0) {
      steerCoh.div(countCoh);
      steerCoh.sub(this.position);
      steerCoh.normalize().mul(this.maxSpeed);
      steerCoh.sub(this.velocity).limit(this.maxForce);
    }
    if (countSep > 0) {
      steerSep.div(countSep);
      steerSep.normalize().mul(this.maxSpeed);
      steerSep.sub(this.velocity).limit(this.maxForce * 1.2);
    }

    const wSep = 1.8, wAli = 0.8, wCoh = 0.6;

    this.applyForce(steerSep.mul(wSep));
    this.applyForce(steerAli.mul(wAli));
    this.applyForce(steerCoh.mul(wCoh));

    // small jitter to avoid perfect symmetry
    const jitter = Vec2.randomUnit().mul(10);
    this.applyForce(jitter);
  }

  update(dt, width, height) {
    // integrate motion
    this.acceleration.limit(this.maxForce);
    this.velocity.add(this.acceleration.clone().mul(dt)).limit(this.maxSpeed);
    this.position.add(this.velocity.clone().mul(dt));
    this.acceleration.set(0, 0);

    // toroidal wrap
    if (this.position.x < 0) this.position.x += width;
    else if (this.position.x >= width) this.position.x -= width;
    if (this.position.y < 0) this.position.y += height;
    else if (this.position.y >= height) this.position.y -= height;

    // animate hue a bit
    this.hue = (this.hue + 12 * dt) % 360;
  }
}
