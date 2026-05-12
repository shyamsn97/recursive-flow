// Auto-generated Simulation class per contract
// Provides: export class Simulation { constructor(ctx, canvas) {} update(dt) {} draw() {} }
// Optional helpers: start(), stop(), resize(), add/remove entities, background clearing, stats.

export class Simulation {
  constructor(ctx, canvas) {
    if (!ctx || !canvas) throw new Error("Simulation requires ctx and canvas");
    this.ctx = ctx;
    this.canvas = canvas;

    // Simulation state
    this.entities = []; // e.g., Boid instances or drawables with update(dt) and draw(ctx)
    this.running = false;
    this._raf = null;

    // Timing
    this._last = 0;
    this.time = 0;     // accumulated time in seconds
    this.dt = 0;       // last frame dt in seconds
    this.fps = 0;      // instantaneous fps
    this._fpsAlpha = 0.15; // smoothing for fps
    this.speed = 220; // default speed constant reference

    // Background
    this.clearColor = "rgba(0,0,0,0.15)"; // trail effect; set to 'transparent' for no clear
    this.autoClear = true;

    // Sizing
    this.fitToParent = true;
    this.devicePixelRatio = (globalThis.devicePixelRatio || 1);

    // Input (optional)
    this.mouse = { x: 0, y: 0, down: false };
    this._bindEvents();

    // Initial size
    this.resize();
  }

  // Public: advance simulation by dt (seconds)
  update(dt) {
    this.dt = dt;
    this.time += dt;

    // Update entities in-place; guard against dynamic mutation
    const list = this.entities.slice();
    for (let i = 0; i < list.length; i++) {
      const e = list[i];
      if (!e) continue;
      if (typeof e.update === "function") {
        e.update(dt, this.entities, this.width, this.height);
      }
    }
  }

  // Public: render current frame
  draw() {
    const ctx = this.ctx;
    if (!ctx) return;

    if (this.autoClear) {
      if (this.clearColor && this.clearColor !== "transparent") {
        ctx.save();
        ctx.globalCompositeOperation = "source-over";
        ctx.fillStyle = this.clearColor;
        ctx.fillRect(0, 0, this.width, this.height);
        ctx.restore();
      } else {
        ctx.clearRect(0, 0, this.width, this.height);
      }
    }

    const list = this.entities;
    for (let i = 0; i < list.length; i++) {
      const e = list[i];
      if (!e) continue;
      if (typeof e.draw === "function") {
        e.draw(ctx);
      }
    }
  }

  // Start RAF loop
  start() {
    if (this.running) return;
    this.running = true;
    this._last = performance.now();
    const tick = (t) => {
      if (!this.running) return;
      const now = t || performance.now();
      let dt = Math.max(0, (now - this._last) / 1000);
      // Clamp dt to avoid huge jumps when tab was inactive
      if (dt > 0.1) dt = 0.1;
      this._last = now;

      // Smooth fps
      const instFPS = dt > 0 ? (1 / dt) : 0;
      this.fps = this.fps ? (this._fpsAlpha * instFPS + (1 - this._fpsAlpha) * this.fps) : instFPS;

      this.update(dt);
      this.draw();
      this._raf = requestAnimationFrame(tick);
    };
    this._raf = requestAnimationFrame(tick);
  }

  // Stop RAF loop
  stop() {
    this.running = false;
    if (this._raf != null) {
      cancelAnimationFrame(this._raf);
      this._raf = null;
    }
  }

  // Resize canvas to match its client size (or parent)
  resize() {
    const dpr = this.devicePixelRatio || 1;
    const target = (this.fitToParent && this.canvas.parentElement) ? this.canvas.parentElement : this.canvas;

    const cssW = Math.floor(target.clientWidth || this.canvas.width || 300);
    const cssH = Math.floor(target.clientHeight || this.canvas.height || 150);

    const w = Math.max(1, Math.floor(cssW * dpr));
    const h = Math.max(1, Math.floor(cssH * dpr));

    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }

    // Maintain CSS size to avoid layout shifts
    this.canvas.style.width = cssW + "px";
    this.canvas.style.height = cssH + "px";

    this.width = w;
    this.height = h;

    // Set up default drawing state
    const ctx = this.ctx;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.scale(dpr, dpr);
  }

  // Manage entities
  add(entity) {
    if (!entity) return;
    this.entities.push(entity);
    return entity;
  }

  remove(entity) {
    const i = this.entities.indexOf(entity);
    if (i >= 0) this.entities.splice(i, 1);
  }

  clear() {
    this.entities.length = 0;
  }

  // Internal: input bindings
  _bindEvents() {
    const c = this.canvas;
    this._onResize = () => this.resize();
    this._onMouseMove = (e) => {
      const rect = c.getBoundingClientRect();
      this.mouse.x = (e.clientX - rect.left);
      this.mouse.y = (e.clientY - rect.top);
    };
    this._onMouseDown = () => { this.mouse.down = true; };
    this._onMouseUp = () => { this.mouse.down = false; };
    window.addEventListener("resize", this._onResize);
    c.addEventListener("mousemove", this._onMouseMove);
    c.addEventListener("mousedown", this._onMouseDown);
    window.addEventListener("mouseup", this._onMouseUp);
  }

  // Cleanup listeners and stop loop
  destroy() {
    this.stop();
    window.removeEventListener("resize", this._onResize);
    this.canvas.removeEventListener("mousemove", this._onMouseMove);
    this.canvas.removeEventListener("mousedown", this._onMouseDown);
    window.removeEventListener("mouseup", this._onMouseUp);
  }
}
