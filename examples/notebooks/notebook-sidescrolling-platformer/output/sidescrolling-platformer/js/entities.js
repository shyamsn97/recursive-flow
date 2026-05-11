(function(global){
  const GameNS = global.GameNS = global.GameNS || {};

  function aabbOverlap(a, b){
    return !(a.x + a.w <= b.x || b.x + b.w <= a.x || a.y + a.h <= b.y || b.y + b.h <= a.y);
  }

  function resolveCollisions(box, vel, platforms){
    // Move along X then Y, resolving against static rect platforms
    // Returns {onGround: boolean}
    let onGround = false;

    // Horizontal sweep
    box.x += vel.x;
    for (let p of platforms){
      if (aabbOverlap(box, p)){
        if (vel.x > 0) box.x = p.x - box.w;
        else if (vel.x < 0) box.x = p.x + p.w;
        vel.x = 0;
      }
    }

    // Vertical sweep
    box.y += vel.y;
    for (let p of platforms){
      if (aabbOverlap(box, p)){
        if (vel.y > 0) {
          box.y = p.y - box.h;
          onGround = true;
        } else if (vel.y < 0) {
          box.y = p.y + p.h;
        }
        vel.y = 0;
      }
    }
    return { onGround };
  }

  class Player {
    constructor(x, y){
      this.x = x; this.y = y;
      this.w = 28; this.h = 38;
      this.vx = 0; this.vy = 0;
      this.speed = 230;   // px/s
      this.jump = 430;    // initial jump velocity
      this.gravity = 1200;// px/s^2
      this.maxFall = 900;
      this.color = '#222';
      this.onGround = false;
      this.facing = 1; // 1 right, -1 left
    }
    getAABB(){
      return { x: this.x, y: this.y, w: this.w, h: this.h };
    }
    update(dt, level, input){
      const left  = input.isDown('Left');
      const right = input.isDown('Right');
      const up    = input.isDown('Up');

      // Horizontal input
      const ax = (right ? 1 : 0) - (left ? 1 : 0);
      this.vx = ax * this.speed;

      if (ax !== 0) this.facing = ax;

      // Gravity
      this.vy += this.gravity * dt;
      if (this.vy > this.maxFall) this.vy = this.maxFall;

      // Jump
      if (up && this.onGround) {
        this.vy = -this.jump;
      }

      // Collisions
      const box = this.getAABB();
      const res = resolveCollisions(box, {x: this.vx * dt, y: this.vy * dt}, level.getPlatforms());
      this.onGround = res.onGround;

      // Commit
      this.x = box.x; this.y = box.y;
    }
    draw(ctx, cameraX){
      const x = Math.floor(this.x - cameraX);
      const y = Math.floor(this.y);
      // Body
      ctx.fillStyle = '#333';
      ctx.fillRect(x, y, this.w, this.h);
      // Face
      ctx.fillStyle = '#ffd24a';
      ctx.fillRect(x+4, y+4, this.w-8, this.h/2 - 6);
      // Eye
      ctx.fillStyle = '#000';
      ctx.fillRect(x + (this.facing>0 ? this.w-12 : 6), y+10, 4, 4);
      // Feet
      ctx.fillStyle = '#111';
      ctx.fillRect(x+4, y+this.h-6, this.w-8, 6);
    }
  }

  class Enemy {
    constructor(x, y, range){
      this.x = x; this.y = y;
      this.baseX = x; this.range = range || 100;
      this.w = 30; this.h = 30;
      this.vx = 80; this.vy = 0;
      this.gravity = 1200;
      this.maxFall = 900;
      this.dir = 1;
      this.color = '#f25f5c';
    }
    getAABB(){ return { x: this.x, y: this.y, w: this.w, h: this.h }; }
    update(dt, level){
      // Patrol horizontally within [baseX - range, baseX + range]
      const minX = this.baseX - this.range;
      const maxX = this.baseX + this.range;
      if (this.x < minX) this.dir = 1;
      if (this.x > maxX) this.dir = -1;
      this.vx = this.dir * 80;

      // Gravity
      this.vy += this.gravity * dt;
      if (this.vy > this.maxFall) this.vy = this.maxFall;

      // Collisions
      const box = this.getAABB();
      const res = (function(){
        // Use same resolver but restrict X movement if we hit a wall: flip direction.
        const beforeX = box.x;
        const out = {
          onGround: false
        };
        // Horizontal
        box.x += this.vx * dt;
        for (let p of level.getPlatforms()){
          if (!(box.x + box.w <= p.x || p.x + p.w <= box.x || box.y + box.h <= p.y || p.y + p.h <= box.y)) {
            if (this.vx > 0) box.x = p.x - box.w;
            else if (this.vx < 0) box.x = p.x + p.w;
            this.vx = -this.vx; // bounce/flip
            this.dir = -this.dir;
          }
        }
        // Vertical
        box.y += this.vy * dt;
        for (let p of level.getPlatforms()){
          if (!(box.x + box.w <= p.x || p.x + p.w <= box.x || box.y + box.h <= p.y || p.y + p.h <= box.y)) {
            if (this.vy > 0) { box.y = p.y - box.h; out.onGround = true; }
            else if (this.vy < 0) { box.y = p.y + p.h; }
            this.vy = 0;
          }
        }
        return out;
      }).call(this);
      // Commit
      this.x = box.x; this.y = box.y;
    }
    draw(ctx, cameraX){
      const x = Math.floor(this.x - cameraX);
      const y = Math.floor(this.y);
      ctx.fillStyle = this.color;
      ctx.fillRect(x, y, this.w, this.h);
      ctx.fillStyle = '#000';
      ctx.fillRect(x+6, y+8, 6, 6);
      ctx.fillRect(x+this.w-12, y+8, 6, 6);
      ctx.fillStyle = '#fff';
      ctx.fillRect(x+8, y+10, 2, 2);
      ctx.fillRect(x+this.w-10, y+10, 2, 2);
    }
  }

  class Coin {
    constructor(x, y){
      this.x = x; this.y = y;
      this.r = 8;
      this.collected = false;
      this.t = 0;
    }
    getAABB(){ return { x: this.x - this.r, y: this.y - this.r, w: this.r*2, h: this.r*2 }; }
    draw(ctx, cameraX){
      if (this.collected) return;
      this.t += 0.07;
      const bob = Math.sin(this.t) * 2;
      const x = Math.floor(this.x - cameraX);
      const y = Math.floor(this.y + bob);
      ctx.beginPath();
      ctx.arc(x, y, this.r, 0, Math.PI*2);
      ctx.closePath();
      ctx.fillStyle = '#ffd24a';
      ctx.fill();
      ctx.strokeStyle = '#caa836';
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  }

  GameNS.Player = Player;
  GameNS.Enemy = Enemy;
  GameNS.Coin = Coin;
})(window);
