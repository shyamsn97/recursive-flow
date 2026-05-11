(function(){
  // Core namespace and utilities
  var G = window.Game || {};
  window.Game = G;

  G.WIDTH = 800;
  G.HEIGHT = 450;
  G.GRAVITY = 2200; // px/s^2
  G.colors = {
    sky: "#87ceeb",
    bg: "#9adaff",
    ground: "#5a3e2b",
    platform: "#7a513b",
    player: "#1e90ff",
    coin: "#ffd700",
    enemy: "#cc3333",
    text: "#101010",
    hud: "#ffffff"
  };

  G.clamp = function(v, a, b){ return Math.max(a, Math.min(b, v)); };
  G.rectsIntersect = function(a, b){
    return a.x < b.x + b.w && a.x + a.w > b.x && a.y < b.y + b.h && a.y + a.h > b.y;
  };
  G.randRange = function(min, max){ return Math.random() * (max - min) + min; };

  // Draw background helper
  G.drawSky = function(ctx, w, h){
    // simple vertical gradient sky
    var g = ctx.createLinearGradient(0, 0, 0, h);
    g.addColorStop(0, "#bde9ff");
    g.addColorStop(1, G.colors.sky);
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, w, h);
  };
})();