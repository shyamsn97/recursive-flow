import { Simulation } from './sim.js';

(function () {
  const COUNT = 400; // ~ hundreds of boids

  function setupCanvasForDPR(canvas) {
    const dpr = Math.max(1, window.devicePixelRatio || 1);
    const cssW = Math.max(1, Math.floor(window.innerWidth));
    const cssH = Math.max(1, Math.floor(window.innerHeight));
    canvas.style.width = cssW + 'px';
    canvas.style.height = cssH + 'px';
    canvas.width = Math.floor(cssW * dpr);
    canvas.height = Math.floor(cssH * dpr);
    const ctx = canvas.getContext('2d');
    // setTransform to an absolute scale so repeated calls don't accumulate
    if (ctx && typeof ctx.setTransform === 'function') {
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
  }

  function init() {
    const canvas = document.getElementById('c');
    if (!canvas) return;

    // initial size
    setupCanvasForDPR(canvas);

    // create simulation with the requested signature
    const sim = new Simulation(canvas, COUNT);

    // Resize handler updates canvas size; Simulation reads canvas size each frame.
    const onResize = () => {
      setupCanvasForDPR(canvas);
      // keep simulation running; no need to recreate sim
    };
    window.addEventListener('resize', onResize, { passive: true });
    window.addEventListener('orientationchange', onResize, { passive: true });

    // RAF loop with dt in seconds, capped to 0.033
    let last = performance.now();
    function loop(now) {
      let dt = (now - last) / 1000;
      last = now;
      if (!isFinite(dt) || dt <= 0) dt = 0;
      if (dt > 0.033) dt = 0.033;
      if (typeof sim.update === 'function') sim.update(dt);
      if (typeof sim.draw === 'function') sim.draw();
      requestAnimationFrame(loop);
    }

    // start loop
    last = performance.now();
    requestAnimationFrame(loop);
  }

  // Run on load
  if (document.readyState === 'complete') {
    init();
  } else {
    window.addEventListener('load', init, { once: true });
  }
})();
