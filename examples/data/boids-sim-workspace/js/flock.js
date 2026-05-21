import Boid from './boid.js';

/*
 Flock manager
 - constructor(width,height)
 - populate(n)
 - updateAll(ctx)
 - resize(width,height)
 - getCount()
 
 This implementation is defensive about Boid API:
 - Attempts to construct Boid with (x,y), falls back to no-arg.
 - Calls b.update(width,height,boids) if available, otherwise b.update().
 - Calls b.draw(ctx) if available, otherwise b.render(ctx) if available.
*/

export class Flock {
  constructor(width, height) {
    this.width = typeof width === 'number' ? width : 0;
    this.height = typeof height === 'number' ? height : 0;
    this.boids = [];
  }

  populate(n) {
    const count = Math.max(0, Math.floor(n) || 0);
    for (let i = 0; i < count; i++) {
      const x = Math.random() * this.width;
      const y = Math.random() * this.height;
      let b;
      // Try to create a Boid with position, but fall back if constructor differs.
      try {
        b = new Boid(x, y);
      } catch (e) {
        b = new Boid();
        // Try to set position if possible
        try {
          if (typeof b.setPosition === 'function') {
            b.setPosition(x, y);
          } else if ('x' in b && 'y' in b) {
            b.x = x;
            b.y = y;
          }
        } catch (_) {
          // ignore if setting position not supported
        }
      }
      this.boids.push(b);
    }
  }

  updateAll(ctx) {
    // Update each boid and then draw/render it on provided context if available.
    for (const b of this.boids) {
      if (b && typeof b.update === 'function') {
        try {
          // Preferred signature: update(width, height, boids)
          b.update(this.width, this.height, this.boids);
        } catch (err) {
          // Fallback to no-arg update
          try { b.update(); } catch (_) {}
        }
      }
      // Drawing: prefer draw(ctx), fallback to render(ctx)
      if (ctx && typeof b.draw === 'function') {
        try { b.draw(ctx); } catch (_) {}
      } else if (ctx && typeof b.render === 'function') {
        try { b.render(ctx); } catch (_) {}
      }
    }
  }

  resize(width, height) {
    this.width = width;
    this.height = height;
  }

  getCount() {
    return this.boids.length;
  }
}
