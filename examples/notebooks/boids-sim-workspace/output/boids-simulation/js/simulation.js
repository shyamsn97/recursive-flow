import { Boid } from './boid.js';

export class Simulation {
  constructor(width, height, count = 400) {
    this.width = width;
    this.height = height;
    this.boids = [];
    this.neighRadius = 64;
    this.cellSize = this.neighRadius;
    this.grid = new Map();
    this.populate(count);
  }

  setSize(w, h) { this.width = w; this.height = h; }

  clearGrid() { this.grid.clear(); }

  _cellKey(cx, cy) { return cx + ',' + cy; }

  _insertToGrid(boid) {
    const cx = Math.floor(boid.position.x / this.cellSize);
    const cy = Math.floor(boid.position.y / this.cellSize);
    const key = this._cellKey(cx, cy);
    let bucket = this.grid.get(key);
    if (!bucket) { bucket = []; this.grid.set(key, bucket); }
    bucket.push(boid);
  }

  _nearby(boid) {
    const cx = Math.floor(boid.position.x / this.cellSize);
    const cy = Math.floor(boid.position.y / this.cellSize);
    const neighbors = [];
    for (let ox = -1; ox <= 1; ox++) {
      for (let oy = -1; oy <= 1; oy++) {
        const key = this._cellKey(cx + ox, cy + oy);
        const bucket = this.grid.get(key);
        if (bucket) {
          for (const other of bucket) neighbors.push(other);
        }
      }
    }
    return neighbors;
  }

  populate(count) {
    this.boids.length = 0;
    for (let i = 0; i < count; i++) {
      const x = Math.random() * this.width;
      const y = Math.random() * this.height;
      const b = new Boid(x, y, {
        maxSpeed: 220 + Math.random() * 100,
        maxForce: 240,
        size: 3 + Math.random() * 2,
        sepRadius: 22 + Math.random() * 8,
        neighRadius: this.neighRadius
      });
      // start with a good initial speed
      const speedBoost = 0.6 + Math.random() * 0.8;
      b.velocity.normalize().mul(b.maxSpeed * speedBoost);
      this.boids.push(b);
    }
  }

  update(dt) {
    // rebuild spatial grid
    this.clearGrid();
    for (const b of this.boids) this._insertToGrid(b);

    // apply flocking using local neighborhoods
    for (const b of this.boids) {
      const neighbors = this._nearby(b);
      b.flock(neighbors);
    }

    // integrate
    for (const b of this.boids) {
      b.update(dt, this.width, this.height);
    }
  }
}
