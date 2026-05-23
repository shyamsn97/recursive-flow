(function (global) {
  "use strict";

  // Boid constructor: position (x, y), velocity (vx, vy)
  function Boid(x, y, vx, vy) {
if (typeof window !== 'undefined') { window.Boid = Boid; } else if (typeof self !== 'undefined') { self.Boid = Boid; }
    this.x = x || 0;
    this.y = y || 0;
    this.vx = vx || 0;
    this.vy = vy || 0;

    // Cache for draw to avoid repeated string allocations
    this._lastHue = -1;
    this._color = "hsl(0,100%,50%)";
  }

  // Integrate velocity with acceleration, clamp speed, and wrap around edges.
  Boid.prototype.step = function (ax, ay, maxSpeed, width, height, dt) {
    dt = (dt === undefined ? 1 : dt);
    // Update velocity with acceleration
    this.vx += ax * dt;
    this.vy += ay * dt;

    // Clamp speed if needed
    var vx = this.vx;
    var vy = this.vy;
    var s2 = vx * vx + vy * vy;
    if (maxSpeed > 0) {
      var m2 = maxSpeed * maxSpeed;
      if (s2 > m2) {
        var inv = maxSpeed / Math.sqrt(s2);
        vx *= inv;
        vy *= inv;
        this.vx = vx;
        this.vy = vy;
      }
    }

    // Integrate position
    this.x += this.vx * dt;
    this.y += this.vy * dt;

    // Wrap around edges (torus)
    var w = width, h = height;
    var x = this.x, y = this.y;
    if (x < 0) x += w;
    else if (x >= w) x -= w;
    if (y < 0) y += h;
    else if (y >= h) y -= h;
    this.x = x;
    this.y = y;
  };

  // Draw a triangle oriented along velocity; fill with HSL using provided hue.
  // Avoid allocations in the hot path.
  Boid.prototype.draw = function (ctx, hue, size) {
    // Cache color string if hue changed
    if (hue !== this._lastHue) {
      // Build once per hue change
      this._color = "hsl(" + hue + ",100%,50%)";
      this._lastHue = hue;
    }
    ctx.fillStyle = this._color;

    // Compute forward unit vector from velocity
    var vx = this.vx, vy = this.vy;
    var len2 = vx * vx + vy * vy;
    var ux, uy;
    if (len2 > 1e-8) {
      var invLen = 1 / Math.sqrt(len2);
      ux = vx * invLen;
      uy = vy * invLen;
    } else {
      // Default orientation if near-zero velocity
      ux = 1; uy = 0;
    }

    // Triangle in local (forward/right) coordinates:
    // tip:     ( L,  0)
    // back-L:  (-B, -H)
    // back-R:  (-B,  H)
    var L = size * 2.0;    // nose length
    var B = size * 0.9;    // back offset
    var H = size * 0.7;    // half width

    var x = this.x, y = this.y;

    // Rotate and translate using basis [ux -uy; uy ux]
    var x1 = x + L * ux;
    var y1 = y + L * uy;

    var x2 = x + (-B) * ux - (-H) * uy; // -B*ux + H*uy
    var y2 = y + (-B) * uy + (-H) * ux; // -B*uy - H*ux
    x2 = x - B * ux + H * uy;
    y2 = y - B * uy - H * ux;

    var x3 = x + (-B) * ux - ( H) * uy; // -B*ux - H*uy
    var y3 = y + (-B) * uy + ( H) * ux; // -B*uy + H*ux
    x3 = x - B * ux - H * uy;
    y3 = y - B * uy + H * ux;

    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.lineTo(x3, y3);
    ctx.closePath();
    ctx.fill();
  };

  // Create a boid at a random position with random direction and speed in [speedMin, speedMax]
  Boid.createRandom = function (width, height, speedMin, speedMax) {
    var x = Math.random() * width;
    var y = Math.random() * height;

    var angle = Math.random() * (Math.PI * 2);
    var speed = speedMin + Math.random() * Math.max(0, (speedMax - speedMin));
    var vx = Math.cos(angle) * speed;
    var vy = Math.sin(angle) * speed;

    return new Boid(x, y, vx, vy);
  };

  // Expose globally
  global.Boid = Boid;

})(typeof window !== "undefined" ? window : (typeof globalThis !== "undefined" ? globalThis : this));