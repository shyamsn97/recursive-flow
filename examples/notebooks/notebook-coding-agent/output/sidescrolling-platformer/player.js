(function(){
  var G = window.Game;
  function Player(x, y){
    this.x = x; this.y = y;
    this.w = 32; this.h = 42;
    this.vx = 0; this.vy = 0;
    this.maxSpeed = 320;
    this.accel = 2400;
    this.jumpSpeed = 760;
    this.onGround = false;
    this.facing = 1;
  }
  Player.prototype.rect = function(){ return {x:this.x, y:this.y, w:this.w, h:this.h}; };
  Player.prototype.reset = function(x, y){
    this.x = x; this.y = y;
    this.vx = 0; this.vy = 0;
    this.onGround = false;
  };
  Player.prototype.update = function(dt, level, input){
    var left  = input.isDown("ArrowLeft") || input.isDown("KeyA");
    var right = input.isDown("ArrowRight") || input.isDown("KeyD");
    var jump  = input.isDown("Space") || input.isDown("ArrowUp") || input.isDown("KeyW");

    if (left && !right) { this.vx -= this.accel * dt; this.facing = -1; }
    else if (right && !left) { this.vx += this.accel * dt; this.facing = 1; }
    else { this.vx *= 0.86; if (Math.abs(this.vx) < 1) this.vx = 0; }

    if (this.onGround && jump){
      this.vy = -this.jumpSpeed;
      this.onGround = false;
    }

    this.vy += G.GRAVITY * dt;

    // Clamp horizontal speed
    if (this.vx > this.maxSpeed) this.vx = this.maxSpeed;
    if (this.vx < -this.maxSpeed) this.vx = -this.maxSpeed;

    // Horizontal move + collide
    this.x += this.vx * dt;
    var r = this.rect();
    var hits = level.getCollisions(r);
    for (var i=0;i<hits.length;i++){
      var p = hits[i];
      if (this.vx > 0) this.x = p.x - this.w;
      else if (this.vx < 0) this.x = p.x + p.w;
      this.vx = 0;
      r = this.rect();
    }

    // Vertical move + collide
    this.y += this.vy * dt;
    r = this.rect();
    hits = level.getCollisions(r);
    this.onGround = false;
    for (var j=0;j<hits.length;j++){
      var q = hits[j];
      if (this.vy > 0){ // falling
        this.y = q.y - this.h;
        this.vy = 0;
        this.onGround = true;
      }else if (this.vy < 0){ // jumping up
        this.y = q.y + q.h;
        this.vy = 0;
      }
      r = this.rect();
    }

    // World bounds
    if (this.x < 0){ this.x = 0; this.vx = 0; }
    if (this.x + this.w > level.worldWidth){ this.x = level.worldWidth - this.w; this.vx = 0; }
    if (this.y + this.h > level.worldHeight){
      this.y = level.worldHeight - this.h;
      this.vy = 0; this.onGround = true;
    }
  };
  Player.prototype.draw = function(ctx, camX){
    ctx.fillStyle = G.colors.player;
    var sx = Math.floor(this.x - camX), sy = Math.floor(this.y);
    ctx.fillRect(sx, sy, this.w, this.h);
    // eyes
    ctx.fillStyle = "#fff";
    var ex = sx + (this.facing > 0 ? this.w - 10 : 6);
    ctx.fillRect(ex, sy+10, 4, 4);
  };
  G.Player = Player;
})();