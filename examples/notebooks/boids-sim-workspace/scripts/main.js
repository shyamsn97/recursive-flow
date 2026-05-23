;(function () {
  'use strict';

  // Configuration
  var NUM_BOIDS = 420; // ~400 fast-moving boids
  var CELL_SIZE = 50;  // ~perception radius (45-55 px)
  var SEP_RADIUS = 22;
  var ALIGN_RADIUS = 50;
  var COH_RADIUS = 55;

  var WEIGHT_SEP = 1.7;    // strong, short radius
  var WEIGHT_ALIGN = 0.6;  // medium
  var WEIGHT_COH = 0.6;    // medium

  var MAX_FORCE = 120;     // clamp steering (px/s^2)
  var MAX_SPEED = 180;     // clamp in Boid.step (px/s)

  var BODY_LEN = 10;       // triangle length in CSS px
  var BODY_WID = 6;        // triangle width in CSS px

  var SPEED_MIN = 90;      // initial speed range (px/s)
  var SPEED_MAX = 170;

  var canvas, ctx;
  var dpr = 1;
  var widthCSS = 0;
  var heightCSS = 0;

  var boids = [];
  var timeOffset = 0; // for rainbow cycling

  function setupCanvas() {
    canvas = document.getElementById('boidsCanvas');
    if (!canvas) return;
    ctx = canvas.getContext('2d', { alpha: false });

    function resize() {
      dpr = Math.max(1, window.devicePixelRatio || 1);
      // Use the canvas CSS size (it should fill the viewport via CSS)
      var w = Math.max(1, Math.round(canvas.clientWidth || window.innerWidth));
      var h = Math.max(1, Math.round(canvas.clientHeight || window.innerHeight));
      widthCSS = w;
      heightCSS = h;

      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);

      // Keep simulation in CSS pixels; scale drawing by dpr
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    window.addEventListener('resize', resize);
    resize();
  }

  function initBoids() {
    boids.length = 0;
    for (var i = 0; i < NUM_BOIDS; i++) {
      var x = Math.random() * widthCSS;
      var y = Math.random() * heightCSS;
      var b = Boid.createRandom(widthCSS, heightCSS, SPEED_MIN, SPEED_MAX);
      boids.push(b);
    }
  }

  // Build uniform spatial grid for neighbor search
  function buildGrid() {
    // Map "cx,cy" -> array of boid indices
    var grid = new Map();
    var inv = 1 / CELL_SIZE;
    for (var i = 0; i < boids.length; i++) {
      var b = boids[i];
      var cx = Math.floor(b.x * inv);
      var cy = Math.floor(b.y * inv);
      var key = cx + ',' + cy;
      var arr = grid.get(key);
      if (arr) {
        arr.push(i);
      } else {
        grid.set(key, [i]);
      }
    }
    return grid;
  }

  function limit(vec, max) {
    var vx = vec.x, vy = vec.y;
    var mag2 = vx * vx + vy * vy;
    var max2 = max * max;
    if (mag2 > max2 && mag2 > 0) {
      var s = max / Math.sqrt(mag2);
      vec.x = vx * s;
      vec.y = vy * s;
    }
  }

  function steerTowards(fromVX, fromVY, desiredX, desiredY, maxForce) {
    // steering = desired - current
    var sx = desiredX - fromVX;
    var sy = desiredY - fromVY;
    var mag2 = sx * sx + sy * sy;
    if (mag2 > 0) {
      var mag = Math.sqrt(mag2);
      if (mag > maxForce) {
        sx = (sx / mag) * maxForce;
        sy = (sy / mag) * maxForce;
      }
    }
    return { x: sx, y: sy };
  }

  function computeAcceleration(i, grid) {
    var b = boids[i];

    var cellX = Math.floor(b.x / CELL_SIZE);
    var cellY = Math.floor(b.y / CELL_SIZE);

    var alignCount = 0;
    var cohCount = 0;
    var sepCount = 0;

    var sumVX = 0, sumVY = 0;       // for alignment
    var sumPX = 0, sumPY = 0;       // for cohesion
    var sumSepX = 0, sumSepY = 0;   // for separation

    var maxCheckRadius = COH_RADIUS; // largest perception

    for (var oy = -1; oy <= 1; oy++) {
      for (var ox = -1; ox <= 1; ox++) {
        var nx = cellX + ox;
        var ny = cellY + oy;
        var key = nx + ',' + ny;
        var arr = grid.get(key);
        if (!arr) continue;

        for (var k = 0; k < arr.length; k++) {
          var j = arr[k];
          if (j === i) continue;
          var other = boids[j];

          var dx = other.x - b.x;
          var dy = other.y - b.y;
          var d2 = dx * dx + dy * dy;
          if (d2 === 0) continue;

          // Within broadest radius?
          if (d2 <= maxCheckRadius * maxCheckRadius) {
            var d = Math.sqrt(d2);

            // Alignment
            if (d <= ALIGN_RADIUS) {
              sumVX += other.vx;
              sumVY += other.vy;
              alignCount++;
            }

            // Cohesion
            if (d <= COH_RADIUS) {
              sumPX += other.x;
              sumPY += other.y;
              cohCount++;
            }

            // Separation (strong, short radius), weight by inverse square
            if (d <= SEP_RADIUS) {
              var invd2 = 1 / d2;
              sumSepX += (b.x - other.x) * invd2;
              sumSepY += (b.y - other.y) * invd2;
              sepCount++;
            }
          }
        }
      }
    }

    // Compute steering components
    var steerX = 0;
    var steerY = 0;

    // Separation steering: aim in direction of sumSep, scaled to max speed first
    if (sepCount > 0) {
      var sepX = sumSepX / sepCount;
      var sepY = sumSepY / sepCount;
      var sepMag2 = sepX * sepX + sepY * sepY;
      if (sepMag2 > 0) {
        var sepMag = Math.sqrt(sepMag2);
        var desX = (sepX / sepMag) * MAX_SPEED;
        var desY = (sepY / sepMag) * MAX_SPEED;
        var sepSteer = steerTowards(b.vx, b.vy, desX, desY, MAX_FORCE);
        steerX += WEIGHT_SEP * sepSteer.x;
        steerY += WEIGHT_SEP * sepSteer.y;
      }
    }

    // Alignment steering: match average neighbor velocity
    if (alignCount > 0) {
      var avgVX = sumVX / alignCount;
      var avgVY = sumVY / alignCount;
      var avgMag2 = avgVX * avgVX + avgVY * avgVY;
      if (avgMag2 > 0) {
        var avgMag = Math.sqrt(avgMag2);
        var desX = (avgVX / avgMag) * MAX_SPEED;
        var desY = (avgVY / avgMag) * MAX_SPEED;
        var alignSteer = steerTowards(b.vx, b.vy, desX, desY, MAX_FORCE);
        steerX += WEIGHT_ALIGN * alignSteer.x;
        steerY += WEIGHT_ALIGN * alignSteer.y;
      }
    }

    // Cohesion steering: move towards average neighbor position
    if (cohCount > 0) {
      var centerX = sumPX / cohCount;
      var centerY = sumPY / cohCount;
      var toCX = centerX - b.x;
      var toCY = centerY - b.y;
      var toCMag2 = toCX * toCX + toCY * toCY;
      if (toCMag2 > 0) {
        var toCMag = Math.sqrt(toCMag2);
        var desX = (toCX / toCMag) * MAX_SPEED;
        var desY = (toCY / toCMag) * MAX_SPEED;
        var cohSteer = steerTowards(b.vx, b.vy, desX, desY, MAX_FORCE);
        steerX += WEIGHT_COH * cohSteer.x;
        steerY += WEIGHT_COH * cohSteer.y;
      }
    }

    // Clamp total steering to max force
    var acc = { x: steerX, y: steerY };
    limit(acc, MAX_FORCE);
    return acc;
  }

  function wrapAround(b) {
    if (b.x < 0) b.x += widthCSS;
    else if (b.x >= widthCSS) b.x -= widthCSS;
    if (b.y < 0) b.y += heightCSS;
    else if (b.y >= heightCSS) b.y -= heightCSS;
  }

  function drawBoid(i, b) {
    var angle = Math.atan2(b.vy, b.vx) || 0;

    // Rainbow hue cycling
    var hue = (timeOffset + i * (360 / NUM_BOIDS)) % 360;

    ctx.save();
    ctx.translate(b.x, b.y);
    ctx.rotate(angle);

    ctx.beginPath();
    ctx.moveTo(BODY_LEN, 0);
    ctx.lineTo(-BODY_LEN * 0.7, BODY_WID * 0.5);
    ctx.lineTo(-BODY_LEN * 0.7, -BODY_WID * 0.5);
    ctx.closePath();

    ctx.fillStyle = 'hsl(' + hue + ', 100%, 60%)';
    ctx.fill();
    ctx.restore();
  }

  function frame(dt) {
    // dt in seconds; cap to avoid huge jumps when tab resumes
    if (!isFinite(dt) || dt <= 0) dt = 0.016;
    dt = Math.min(dt, 0.05);

    // Re-apply DPR transform each frame to be safe after resizes/zoom
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    // Clear to black (no trails)
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, widthCSS, heightCSS);

    // Build spatial grid
    var grid = buildGrid();

    // Update physics
    for (var i = 0; i < boids.length; i++) {
      var acc = computeAcceleration(i, grid);
      boids[i].step(acc.x, acc.y, MAX_SPEED, widthCSS, heightCSS, dt);
      wrapAround(boids[i]);
    }

    // Draw boids
    for (var j = 0; j < boids.length; j++) {
      drawBoid(j, boids[j]);
    }
  }

  function start() {
    setupCanvas();
    if (!ctx) return;
    initBoids();

    var last = performance.now();
    function loop(now) {
      var dt = (now - last) / 1000;
      last = now;

      timeOffset = (timeOffset + dt * 60) % 360; // hue shift speed
      frame(dt);
      requestAnimationFrame(loop);
    }
    requestAnimationFrame(loop);
  }

  if (document.readyState === 'complete' || document.readyState === 'interactive') {
    start();
  } else {
    window.addEventListener('DOMContentLoaded', start, { once: true });
  }
})();