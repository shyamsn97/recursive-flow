import { Boid, SPEED } from './boid.js'

const VISION_RADIUS = 45;
const VISION_RADIUS_SQ = VISION_RADIUS * VISION_RADIUS;

export class Simulation {
  constructor(canvas, count) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.width = canvas.width;
    this.height = canvas.height;

    // Boids array
    this.boids = [];

    // Preallocate neighbor lists to minimize allocations across frames.
    // We'll expand this._neighborLists if count grows.
    this._neighborLists = [];

    // Initialize boids with random positions, random angles, and random hues.
    for (let i = 0; i < count; i++) {
      const x = Math.random() * this.width;
      const y = Math.random() * this.height;
      // initial speed based on SPEED and random angle
      const angle = Math.random() * Math.PI * 2;
      const vx = Math.cos(angle) * SPEED;
      const vy = Math.sin(angle) * SPEED;
      const hue = Math.floor(Math.random() * 360);
      this.boids.push(new Boid(x, y, vx, vy, hue));
      this._neighborLists[i] = this._neighborLists[i] || [];
      this._neighborLists[i].length = 0;
    }
  }

  update(dt) {
    const boids = this.boids;
    const n = boids.length;

    // Ensure neighbor lists array length matches boids length
    if (this._neighborLists.length < n) {
      for (let i = this._neighborLists.length; i < n; i++) {
        this._neighborLists[i] = [];
      }
    }

    // Clear existing neighbor lists in-place to avoid reallocations.
    for (let i = 0; i < n; i++) {
      this._neighborLists[i].length = 0;
    }

    // Naive O(N^2) neighbor search using squared distance.
    for (let i = 0; i < n; i++) {
      const bi = boids[i];
      const xi = bi.x, yi = bi.y;
      for (let j = i + 1; j < n; j++) {
        const bj = boids[j];
        const dx = xi - bj.x;
        const dy = yi - bj.y;
        const dist2 = dx * dx + dy * dy;
        if (dist2 <= VISION_RADIUS_SQ) {
          this._neighborLists[i].push(bj);
          this._neighborLists[j].push(bi);
        }
      }
    }

    // Update each boid, passing its neighbors and the world size.
    const w = this.width, h = this.height;
    for (let i = 0; i < n; i++) {
      boids[i].update(dt, this._neighborLists[i], w, h);
    }
  }

  draw() {
    const ctx = this.ctx;
    const w = this.width, h = this.height;

    // Slight fade for motion trails; can be fully transparent if desired.
    ctx.save();
    ctx.fillStyle = 'rgba(0,0,0,0.08)';
    ctx.fillRect(0, 0, w, h);
    ctx.restore();

    // Draw each boid (assumes Boid.draw(ctx) is implemented).
    const boids = this.boids;
    for (let i = 0; i < boids.length; i++) {
      boids[i].draw(ctx);
    }
  }
}
