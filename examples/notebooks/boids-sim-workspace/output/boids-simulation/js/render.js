export class Renderer {
  constructor(canvas, simulation) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.simulation = simulation;
    this.dpr = Math.max(1, window.devicePixelRatio || 1);
    this.bg = 'rgba(0, 0, 0, 0.2)'; // motion blur trail
    this.resize();
    window.addEventListener('resize', () => this.resize(), { passive: true });
  }

  resize() {
    const cssW = this.canvas.clientWidth || window.innerWidth;
    const cssH = this.canvas.clientHeight || window.innerHeight;
    this.dpr = Math.max(1, window.devicePixelRatio || 1);
    this.canvas.width = Math.round(cssW * this.dpr);
    this.canvas.height = Math.round(cssH * this.dpr);
    this.ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    this.simulation.setSize(cssW, cssH);

    // hard clear
    this.ctx.fillStyle = '#000';
    this.ctx.fillRect(0, 0, cssW, cssH);
  }

  drawBoid(b) {
    const ctx = this.ctx;
    const angle = Math.atan2(b.velocity.y, b.velocity.x);
    const size = 4 + b.size; // base size
    const len = size * 3.0;
    const w = size * 1.3;

    ctx.save();
    ctx.translate(b.position.x, b.position.y);
    ctx.rotate(angle);

    ctx.beginPath();
    ctx.moveTo(len, 0);
    ctx.lineTo(-len * 0.5, w);
    ctx.lineTo(-len * 0.3, 0);
    ctx.lineTo(-len * 0.5, -w);
    ctx.closePath();

    const light = 55 + Math.min(40, (Math.hypot(b.velocity.x, b.velocity.y) / b.maxSpeed) * 35);
    ctx.fillStyle = `hsl(${b.hue | 0}, 85%, ${light}%)`;
    ctx.strokeStyle = `hsla(${b.hue | 0}, 95%, 80%, 0.6)`;
    ctx.lineWidth = 0.6;
    ctx.fill();
    ctx.stroke();

    ctx.restore();
  }

  renderFrame() {
    const w = this.canvas.clientWidth || window.innerWidth;
    const h = this.canvas.clientHeight || window.innerHeight;

    // fade previous frame for trails
    this.ctx.fillStyle = this.bg;
    this.ctx.fillRect(0, 0, w, h);

    for (const b of this.simulation.boids) {
      this.drawBoid(b);
    }
  }
}
