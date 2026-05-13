// 2D vector utility
export class Vec2 {
  constructor(x = 0, y = 0) { this.x = x; this.y = y; }
  set(x, y) { this.x = x; this.y = y; return this; }
  add(v) { this.x += v.x; this.y += v.y; return this; }
  sub(v) { this.x -= v.x; this.y -= v.y; return this; }
  mul(s) { this.x *= s; this.y *= s; return this; }
  div(s) { this.x /= s; this.y /= s; return this; }
  clone() { return new Vec2(this.x, this.y); }
  length() { return Math.hypot(this.x, this.y); }
  len2() { return this.x*this.x + this.y*this.y; }
  normalize() { const l = this.length() || 1; this.x /= l; this.y /= l; return this; }
  limit(max) {
    const l2 = this.len2();
    if (l2 > max*max) {
      const l = Math.sqrt(l2);
      this.x = this.x / l * max;
      this.y = this.y / l * max;
    }
    return this;
  }
  distance(v) { const dx = this.x - v.x, dy = this.y - v.y; return Math.hypot(dx, dy); }
  heading() { return Math.atan2(this.y, this.x); }

  static fromAngle(a, m = 1) { return new Vec2(Math.cos(a) * m, Math.sin(a) * m); }
  static randomUnit() { return Vec2.fromAngle(Math.random() * Math.PI * 2, 1); }
}
