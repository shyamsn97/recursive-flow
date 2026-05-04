// boids.js — ES module implementing hyper-fast Boids with spatial hashing.
// Exports: class Boids

export class Boids {
  constructor(width, height, n) {
    this.width = Math.max(1, width | 0);
    this.height = Math.max(1, height | 0);
    this.n = n | 0;

    // Simulation parameters (tuned for stable, fast motion with thousands of boids)
    this.cellSize = 32;                // ~32 px grid cells
    this.neighborRadius = 32;          // perception radius
    this.separationRadius = 16;        // stronger push when very close
    this.maxSpeed = 220;               // pixels/second (180..240 recommended)
    this.alignWeight = 0.05;
    this.cohesionWeight = 0.01;
    this.separationWeight = 0.15;
    this.maxDt = 0.05;                 // clamp dt to handle spikes
    this._eps = 1e-6;

    this.neighborRadius2 = this.neighborRadius * this.neighborRadius;
    this.separationRadius2 = this.separationRadius * this.separationRadius;

    // Boid state (typed arrays for performance)
    this.x = new Float32Array(this.n);
    this.y = new Float32Array(this.n);
    this.vx = new Float32Array(this.n);
    this.vy = new Float32Array(this.n);

    // Grid (spatial hash: linked lists via head/next)
    this._setupGrid();

    // Colors precomputed (CSS strings)
    this.colors = new Array(this.n);
    const golden = 137.50776405003785; // golden angle in degrees for nice distribution
    for (let i = 0; i < this.n; i++) {
      const h = Math.floor((i * golden) % 360);
      this.colors[i] = `hsl(${h},100%,60%)`;
    }

    // Initialize positions and fast random velocities
    for (let i = 0; i < this.n; i++) {
      this.x[i] = Math.random() * this.width;
      this.y[i] = Math.random() * this.height;
      const a = Math.random() * Math.PI * 2;
      const speed = this.maxSpeed * (0.6 + 0.4 * Math.random()); // 60%..100% of max
      this.vx[i] = Math.cos(a) * speed;
      this.vy[i] = Math.sin(a) * speed;
    }
  }

  // Internal: setup or resize the grid arrays based on current width/height
  _setupGrid() {
    this.gridW = Math.max(1, Math.ceil(this.width / this.cellSize) | 0);
    this.gridH = Math.max(1, Math.ceil(this.height / this.cellSize) | 0);
    const cells = this.gridW * this.gridH;
    if (!this.head || this.head.length !== cells) {
      this.head = new Int32Array(cells);
    }
    if (!this.next || this.next.length !== this.n) {
      this.next = new Int32Array(this.n);
    }
  }

  // Internal: rebuild spatial hash for current positions
  _rebuildGrid() {
    const head = this.head;
    head.fill(-1); // reset all cell heads to empty
    const next = this.next;
    const x = this.x, y = this.y;
    const cs = this.cellSize;
    const gw = this.gridW, gh = this.gridH;

    for (let i = 0; i < this.n; i++) {
      let cx = (x[i] / cs) | 0;
      let cy = (y[i] / cs) | 0;
      // clamp (positions are wrapped within [0,w/h) so these are safe)
      if (cx < 0) cx = 0; else if (cx >= gw) cx = gw - 1;
      if (cy < 0) cy = 0; else if (cy >= gh) cy = gh - 1;
      const c = cy * gw + cx;
      next[i] = head[c];
      head[c] = i;
    }
  }

  update(dt) {
    if (!isFinite(dt) || dt <= 0) return;
    if (dt > this.maxDt) dt = this.maxDt;

    // 1) Rebuild spatial grid
    this._rebuildGrid();

    // 2) Neighbor loops and steering
    const n = this.n;
    const x = this.x, y = this.y, vx = this.vx, vy = this.vy;
    const head = this.head, next = this.next;
    const gw = this.gridW, gh = this.gridH;
    const cs = this.cellSize;

    const w = this.width, h = this.height;
    const halfW = 0.5 * w, halfH = 0.5 * h;

    const nr2 = this.neighborRadius2;
    const sr2 = this.separationRadius2;

    const alignW = this.alignWeight;
    const cohW = this.cohesionWeight;
    const sepW = this.separationWeight;

    const maxSpeed = this.maxSpeed;
    const maxSpeed2 = maxSpeed * maxSpeed;
    const eps = this._eps;

    for (let i = 0; i < n; i++) {
      // Determine cell coordinates for boid i
      let cxi = (x[i] / cs) | 0;
      let cyi = (y[i] / cs) | 0;
      if (cxi < 0) cxi = 0; else if (cxi >= gw) cxi = gw - 1;
      if (cyi < 0) cyi = 0; else if (cyi >= gh) cyi = gh - 1;

      let sumVx = 0, sumVy = 0, cntN = 0;
      let sumDX = 0, sumDY = 0; // for cohesion (relative to i using toroidal min image)
      let repelX = 0, repelY = 0;

      // scan 3x3 neighbor cells with wrap-around
      for (let oy = -1; oy <= 1; oy++) {
        let nyc = cyi + oy;
        if (nyc < 0) nyc += gh;
        else if (nyc >= gh) nyc -= gh;

        for (let ox = -1; ox <= 1; ox++) {
          let nxc = cxi + ox;
          if (nxc < 0) nxc += gw;
          else if (nxc >= gw) nxc -= gw;

          let c = nyc * gw + nxc;
          for (let j = head[c]; j !== -1; j = next[j]) {
            if (j === i) continue;

            // Toroidal minimum image distance
            let dx = x[j] - x[i];
            if (dx > halfW) dx -= w;
            else if (dx < -halfW) dx += w;

            let dy = y[j] - y[i];
            if (dy > halfH) dy -= h;
            else if (dy < -halfH) dy += h;

            const d2 = dx * dx + dy * dy;
            if (d2 <= nr2) {
              sumVx += vx[j];
              sumVy += vy[j];
              sumDX += dx;
              sumDY += dy;
              cntN++;

              if (d2 <= sr2) {
                const inv = 1.0 / (d2 + eps);
                // Push away from neighbor (away vector is -dx, -dy)
                repelX -= dx * inv;
                repelY -= dy * inv;
              }
            }
          }
        }
      }

      // Compute steering acceleration
      let ax = 0.0, ay = 0.0;

      if (cntN > 0) {
        // Alignment: steer toward average neighbor velocity
        const avx = sumVx / cntN;
        const avy = sumVy / cntN;
        ax += (avx - vx[i]) * alignW;
        ay += (avy - vy[i]) * alignW;

        // Cohesion: steer toward center of neighbors (relative vector)
        const cdx = (sumDX / cntN);
        const cdy = (sumDY / cntN);
        ax += cdx * cohW;
        ay += cdy * cohW;
      }

      // Separation: strong repulsion when too close
      ax += repelX * sepW;
      ay += repelY * sepW;

      // Update velocity (no per-step allocations)
      let nvx = vx[i] + ax;
      let nvy = vy[i] + ay;

      // Clamp speed
      const s2 = nvx * nvx + nvy * nvy;
      if (s2 > maxSpeed2) {
        const s = Math.sqrt(s2);
        const k = maxSpeed / (s + eps);
        nvx *= k; nvy *= k;
      }
      vx[i] = nvx;
      vy[i] = nvy;
    }

    // 3) Integrate positions with dt and wrap around edges (toroidal)
    for (let i = 0; i < n; i++) {
      let xi = x[i] + vx[i] * dt;
      let yi = y[i] + vy[i] * dt;

      if (xi < 0) xi += w;
      else if (xi >= w) xi -= w;

      if (yi < 0) yi += h;
      else if (yi >= h) yi -= h;

      x[i] = xi;
      y[i] = yi;
    }
  }

  resize(width, height) {
    const w = Math.max(1, width | 0);
    const h = Math.max(1, height | 0);
    if (w === this.width && h === this.height) return;
    this.width = w;
    this.height = h;
    this._setupGrid(); // grid dims may change; head array resized
    // Positions already wrapped; on next update grid will be rebuilt
  }
}
