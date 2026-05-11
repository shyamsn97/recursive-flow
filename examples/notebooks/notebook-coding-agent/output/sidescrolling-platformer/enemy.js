(function(){
  var G = window.Game;
  function Enemy(x, y, leftBound, rightBound){
    this.x = x; this.y = y;
    this.w = 36; this.h = 36;
    this.vx = 100; this.vy = 0;
    this.leftBound = leftBound; this.rightBound = rightBound;
    this.alive = true;
  }
  Enemy.prototype.rect = function(){ return {x:this.x, y:this.y, w:this.w, h:this.h}; };
  Enemy.prototype.update = function(dt, level){
    if (!this.alive) return;
    // Patrol bounds
    if (this.x < this.leftBound){ this.x = this.leftBound; this.vx = Math.abs(this.vx); }
    if (this.x + this.w > this.rightBound){ this.x = this.rightBound - this.w; this.vx = -Math.abs(this.vx); }

    this.vy += G.GRAVITY * dt * 0.9;
    // Horizontal
    this.x += this.vx * dt;
    var r = this.rect();
    var hits = level.getCollisions(r);
    for (var i=0;i<hits.length;i++){
      var p = hits[i];
      if (this.vx > 0) this.x = p.x - this.w; else this.x = p.x + p.w;
      this.vx = -this.vx; // bounce off
      r = this.rect();
    }
    // Vertical
    this.y += this.vy * dt;
    r = this.rect();
    hits = level.getCollisions(r);
    for (var j=0;j<hits.length;j++){
      var q = hits[j];
      if (this.vy > 0){ this.y = q.y - this.h; this.vy = 0; }
      else if (this.vy < 0){ this.y = q.y + q.h; this.vy = 0; }
      r = this.rect();
    }
    if (this.y + this.h > level.worldHeight){
      this.y = level.worldHeight - this.h; this.vy = 0;
    }
  };
  Enemy.prototype.draw = function(ctx, camX){
    if (!this.alive) return;
    ctx.fillStyle = G.colors.enemy;
    ctx.fillRect(Math.floor(this.x - camX), Math.floor(this.y), this.w, this.h);
  };
  Enemy.prototype.collideWithPlayer = function(player){
    if (!this.alive) return false;
    if (!Game.rectsIntersect(this.rect(), player.rect())) return false;
    var playerBottom = player.y + player.h;
    var enemyTop = this.y;
    if (player.vy > 0 && (playerBottom - enemyTop) < 18){
      // stomp
      this.alive = false;
      player.vy = -player.jumpSpeed * 0.6;
      return "stomp";
    }
    return "hit";
  };
  G.Enemy = Enemy;
})();