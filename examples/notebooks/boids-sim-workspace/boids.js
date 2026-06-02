(function () {
  function start() {
    var canvas = document.getElementById('boids');
    if (!canvas) return;
    var ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Simulation constants
    var COUNT = 250;
    var ALIGN_RADIUS = 50;
    var COHESION_RADIUS = 50;
    var SEPARATION_RADIUS = 20;
    var MAX_SPEED = 2.5;
    var MAX_FORCE = 0.05;

    // World size in CSS pixels (drawing units after transform)
    var viewW = 0;
    var viewH = 0;
    var halfW = 0;
    var halfH = 0;

    var boids = [];

    function resize() {
      var dpr = Math.max(1, Math.min(3, window.devicePixelRatio || 1));
      viewW = window.innerWidth || document.documentElement.clientWidth || 800;
      viewH = window.innerHeight || document.documentElement.clientHeight || 600;
      halfW = viewW * 0.5;
      halfH = viewH * 0.5;

      canvas.width = Math.floor(viewW * dpr);
      canvas.height = Math.floor(viewH * dpr);
      // Reset transform then scale so we can use CSS pixels for all math/drawing
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    function randRange(a, b) {
      return a + Math.random() * (b - a);
    }

    function limitVec(x, y, max) {
      var m2 = x * x + y * y;
      if (m2 > max * max) {
        var m = Math.sqrt(m2);
        var s = max / (m || 1);
        return [x * s, y * s];
      }
      return [x, y];
    }

    function initBoids() {
      boids.length = 0;
      for (var i = 0; i < COUNT; i++) {
        var angle = Math.random() * Math.PI * 2;
        var speed = randRange(MAX_SPEED * 0.4, MAX_SPEED);
        boids.push({
          x: Math.random() * viewW,
          y: Math.random() * viewH,
          vx: Math.cos(angle) * speed,
          vy: Math.sin(angle) * speed,
          hue: Math.floor(Math.random() * 360)
        });
      }
    }

    function wrap(b) {
      if (b.x < 0) b.x += viewW;
      else if (b.x >= viewW) b.x -= viewW;
      if (b.y < 0) b.y += viewH;
      else if (b.y >= viewH) b.y -= viewH;
    }

    function update() {
      // Compute accelerations for all boids
      var axArr = new Array(boids.length);
      var ayArr = new Array(boids.length);

      for (var i = 0; i < boids.length; i++) {
        var b = boids[i];

        var sumAlignX = 0, sumAlignY = 0, countAlign = 0;
        var sumCohX = 0, sumCohY = 0, countCoh = 0;
        var sumSepX = 0, sumSepY = 0;

        for (var j = 0; j < boids.length; j++) {
          if (i === j) continue;
          var o = boids[j];

          // Compute minimal image distance (torus)
          var dx = o.x - b.x;
          var dy = o.y - b.y;

          if (dx > halfW) dx -= viewW;
          else if (dx < -halfW) dx += viewW;
          if (dy > halfH) dy -= viewH;
          else if (dy < -halfH) dy += viewH;

          var d2 = dx * dx + dy * dy;

          // Alignment and Cohesion within same radius
          if (d2 < ALIGN_RADIUS * ALIGN_RADIUS) {
            sumAlignX += o.vx;
            sumAlignY += o.vy;
            countAlign++;
          }
          if (d2 < COHESION_RADIUS * COHESION_RADIUS) {
            // Accumulate neighbor positions accounting for wrap
            sumCohX += b.x + dx;
            sumCohY += b.y + dy;
            countCoh++;
          }
          if (d2 < SEPARATION_RADIUS * SEPARATION_RADIUS && d2 > 0) {
            var d = Math.sqrt(d2);
            // Weight more strongly the closer the neighbor is
            var inv = 1 / (d || 1);
            // Direction away from neighbor
            sumSepX -= dx * inv;
            sumSepY -= dy * inv;
          }
        }

        // Alignment: steer towards average heading
        var steerAx = 0, steerAy = 0;
        if (countAlign > 0) {
          var avgVx = sumAlignX / countAlign;
          var avgVy = sumAlignY / countAlign;
          var desiredA = normalizeTo(avgVx, avgVy, MAX_SPEED);
          var steerA = subtract(desiredA[0], desiredA[1], b.vx, b.vy);
          var limitedA = limitVec(steerA[0], steerA[1], MAX_FORCE);
          steerAx = limitedA[0];
          steerAy = limitedA[1];
        }

        // Cohesion: steer toward center of mass
        var steerCx = 0, steerCy = 0;
        if (countCoh > 0) {
          var centerX = sumCohX / countCoh;
          var centerY = sumCohY / countCoh;
          var toCenterX = centerX - b.x;
          var toCenterY = centerY - b.y;
          var desiredC = normalizeTo(toCenterX, toCenterY, MAX_SPEED);
          var steerC = subtract(desiredC[0], desiredC[1], b.vx, b.vy);
          var limitedC = limitVec(steerC[0], steerC[1], MAX_FORCE);
          steerCx = limitedC[0];
          steerCy = limitedC[1];
        }

        // Separation: steer away from neighbors
        var steerSx = 0, steerSy = 0;
        if (sumSepX !== 0 || sumSepY !== 0) {
          var desiredS = normalizeTo(sumSepX, sumSepY, MAX_SPEED);
          var steerS = subtract(desiredS[0], desiredS[1], b.vx, b.vy);
          var limitedS = limitVec(steerS[0], steerS[1], MAX_FORCE);
          steerSx = limitedS[0];
          steerSy = limitedS[1];
        }

        // Weight contributions
        var weightAlign = 1.0;
        var weightCoh = 0.9;
        var weightSep = 1.5;

        var ax = steerAx * weightAlign + steerCx * weightCoh + steerSx * weightSep;
        var ay = steerAy * weightAlign + steerCy * weightCoh + steerSy * weightSep;

        axArr[i] = ax;
        ayArr[i] = ay;
      }

      // Integrate velocities and positions
      for (var k = 0; k < boids.length; k++) {
        var bk = boids[k];
        bk.vx += axArr[k];
        bk.vy += ayArr[k];

        var limitedV = limitVec(bk.vx, bk.vy, MAX_SPEED);
        bk.vx = limitedV[0];
        bk.vy = limitedV[1];

        bk.x += bk.vx;
        bk.y += bk.vy;

        wrap(bk);
      }
    }

    function normalizeTo(x, y, mag) {
      var m = Math.sqrt(x * x + y * y);
      if (m === 0) return [0, 0];
      var s = mag / m;
      return [x * s, y * s];
    }

    function subtract(ax, ay, bx, by) {
      return [ax - bx, ay - by];
    }

    function draw() {
      ctx.clearRect(0, 0, viewW, viewH);

      var size = 8; // triangle length
      for (var i = 0; i < boids.length; i++) {
        var b = boids[i];
        var angle = Math.atan2(b.vy, b.vx) || 0;

        ctx.save();
        ctx.translate(b.x, b.y);
        ctx.rotate(angle);
        ctx.fillStyle = 'hsl(' + b.hue + ', 80%, 60%)';

        ctx.beginPath();
        ctx.moveTo(size, 0);
        ctx.lineTo(-size * 0.6, size * 0.5);
        ctx.lineTo(-size * 0.6, -size * 0.5);
        ctx.closePath();
        ctx.fill();
        ctx.restore();
      }
    }

    function tick() {
      update();
      draw();
      requestAnimationFrame(tick);
    }

    // Initialize
    resize();
    initBoids();
    window.addEventListener('resize', function () {
      resize();
    }, { passive: true });
    window.addEventListener('orientationchange', function () {
      // Delay resize to allow viewport to settle
      setTimeout(resize, 50);
    }, { passive: true });

    requestAnimationFrame(tick);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', start, { once: true });
  } else {
    start();
  }
})();