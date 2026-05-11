(function(global){
  const GameNS = global.GameNS = global.GameNS || {};
  const clamp = (v,min,max) => Math.max(min, Math.min(max, v));

  function rectsOverlap(a,b){
    return !(a.x + a.w <= b.x || b.x + b.w <= a.x || a.y + a.h <= b.y || b.y + b.h <= a.y);
  }

  class Game {
    constructor(canvas){
      this.canvas = canvas;
      this.ctx = canvas.getContext('2d');
      this.running = false;

      this.level = null;
      this.player = null;
      this.enemies = [];
      this.coins = [];

      this.cameraX = 0;
      this.score = 0;
      this.lives = 3;

      this.lastTime = 0;
      this.accum = 0;
      this.dt = 1/60; // fixed update step
    }

    start(){
      GameNS.Input.init(this.canvas);

      this.resetLevel();
      this.running = true;
      this.lastTime = performance.now();
      const loop = (t) => {
        if (!this.running) return;
        const elapsed = Math.min(0.05, (t - this.lastTime) / 1000);
        this.lastTime = t;
        this.accum += elapsed;
        while (this.accum >= this.dt) {
          this.update(this.dt);
          this.accum -= this.dt;
        }
        this.draw(this.ctx);
        requestAnimationFrame(loop);
      };
      requestAnimationFrame(loop);
    }

    resetLevel(){
      this.level = new GameNS.Level();
      const sp = this.level.getSpawn();
      this.player = new GameNS.Player(sp.x, sp.y);
      this.enemies = this.level.getEnemyDefs().map(e => new GameNS.Enemy(e.x, e.y, e.range));
      this.coins = this.level.getCoinPositions().map(c => new GameNS.Coin(c.x, c.y));
      this.cameraX = 0;
      this.score = 0;
      this.lives = 3;
    }

    addScore(n){ this.score += n; }

    update(dt){
      // Player update with input
      this.player.update(dt, this.level, GameNS.Input);

      // Coin collection
      for (let c of this.coins){
        if (c.collected) continue;
        if (rectsOverlap(this.player.getAABB(), c.getAABB())){
          c.collected = true;
          this.addScore(1);
        }
      }

      // Enemies
      for (let e of this.enemies){
        e.update(dt, this.level);
        if (rectsOverlap(this.player.getAABB(), e.getAABB())){
          // Simple hit: lose a life and respawn to last spawn point
          this.lives -= 1;
          const sp = this.level.getSpawn();
          this.player.x = sp.x; this.player.y = sp.y;
          this.player.vx = 0; this.player.vy = 0;
          this.cameraX = clamp(this.player.x - this.canvas.width*0.35, 0, this.level.getWorldWidth() - this.canvas.width);
          if (this.lives <= 0){
            this.resetLevel();
            break;
          }
        }
      }

      // Camera follows player with lookahead
      const target = this.player.x - this.canvas.width * 0.35;
      const worldMax = this.level.getWorldWidth() - this.canvas.width;
      this.cameraX = clamp(this.cameraX + (target - this.cameraX) * 0.1, 0, Math.max(0, worldMax));
    }

    draw(ctx){
      const W = this.canvas.width, H = this.canvas.height;

      // Sky gradient background already from CSS; here add parallax hills
      ctx.clearRect(0,0,W,H);
      // Parallax layers
      const cam = this.cameraX;
      ctx.fillStyle = '#a6d8ff';
      ctx.fillRect(0,0,W,H*0.6);
      ctx.fillStyle = '#e8f7ff';
      ctx.fillRect(0,H*0.6,W,H*0.4);

      // Distant hills
      ctx.fillStyle = '#8cc0ff';
      for (let i=0;i<6;i++){
        const hx = Math.floor(- (cam*0.3 % 400) + i*400);
        ctx.beginPath();
        ctx.ellipse(hx+150, H*0.7, 220, 80, 0, 0, Math.PI*2);
        ctx.fill();
      }

      // Platforms (ground)
      for (let p of this.level.getPlatforms()){
        const x = Math.floor(p.x - this.cameraX);
        const y = Math.floor(p.y);
        ctx.fillStyle = '#5c6d7d';
        ctx.fillRect(x, y, p.w, p.h);
        ctx.fillStyle = '#364552';
        ctx.fillRect(x, y, p.w, 8);
      }

      // Coins
      for (let c of this.coins) c.draw(ctx, this.cameraX);

      // Enemies
      for (let e of this.enemies) e.draw(ctx, this.cameraX);

      // Player
      this.player.draw(ctx, this.cameraX);

      // HUD
      ctx.fillStyle = 'rgba(0,0,0,0.35)';
      ctx.fillRect(12, 10, 180, 56);
      ctx.fillStyle = '#fff';
      ctx.font = '16px system-ui, sans-serif';
      ctx.fillText('Score: ' + this.score, 24, 34);
      ctx.fillText('Lives: ' + this.lives, 24, 56);

      // Instructions
      ctx.fillStyle = 'rgba(0,0,0,0.35)';
      ctx.fillRect(W-300, 10, 288, 56);
      ctx.fillStyle = '#fff';
      ctx.fillText('Arrows / A D to move', W-288, 34);
      ctx.fillText('W / Up / Space to jump', W-288, 56);
    }
  }

  GameNS.Game = Game;
})(window);
