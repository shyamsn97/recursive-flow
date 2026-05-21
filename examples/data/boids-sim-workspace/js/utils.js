// js/utils.js
// ES6 module exporting Vector class and randomColor function.
// No side effects.

export class Vector {
  constructor(x = 0, y = 0) {
    this.x = x;
    this.y = y;
  }

  // Add another vector to this vector (mutates, returns this)
  add(v) {
    this.x += v.x;
    this.y += v.y;
    return this;
  }

  // Subtract another vector from this vector (mutates, returns this)
  sub(v) {
    this.x -= v.x;
    this.y -= v.y;
    return this;
  }

  // Multiply this vector by a scalar (mutates, returns this)
  mult(n) {
    this.x *= n;
    this.y *= n;
    return this;
  }

  // Divide this vector by a scalar (mutates, returns this)
  div(n) {
    if (n !== 0) {
      this.x /= n;
      this.y /= n;
    }
    return this;
  }

  // Magnitude (length) of the vector
  mag() {
    return Math.hypot(this.x, this.y);
  }

  // Set magnitude to n (mutates, returns this)
  setMag(n) {
    return this.normalize().mult(n);
  }

  // Limit magnitude to max (mutates, returns this)
  limit(max) {
    if (this.mag() > max) {
      this.setMag(max);
    }
    return this;
  }

  // Normalize to unit vector (mutates, returns this)
  normalize() {
    const m = this.mag();
    if (m !== 0) {
      this.div(m);
    }
    return this;
  }

  // Heading (angle) in radians
  heading() {
    return Math.atan2(this.y, this.x);
  }

  // Distance to another vector
  dist(v) {
    const dx = this.x - v.x;
    const dy = this.y - v.y;
    return Math.hypot(dx, dy);
  }

  // Return a copy of this vector
  copy() {
    return new Vector(this.x, this.y);
  }

  // Return a random unit vector (static)
  static random2D() {
    const angle = Math.random() * Math.PI * 2;
    return new Vector(Math.cos(angle), Math.sin(angle));
  }
}

// Return a bright HSL color string. Bright colors typically have high saturation
// and mid-to-high lightness. No side effects.
export function randomColor() {
  const h = Math.floor(Math.random() * 360); // hue: 0-359
  const s = 85 + Math.floor(Math.random() * 15); // saturation: 85-99%
  const l = 50 + Math.floor(Math.random() * 11); // lightness: 50-60%
  return `hsl(${h}, ${s}%, ${l}%)`;
}
