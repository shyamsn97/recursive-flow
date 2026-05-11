(function(){
  var G = window.Game;

  function Level(){
    this.platforms = [];
    this.coins = [];
    this.enemies = [];
    this.worldWidth = 3200;
    this.worldHeight = 900;

    // Build terrain
    // Ground segments
    var groundY = 400;
    this.addPlatform(0, groundY, this.worldWidth, this.worldHeight - groundY);

    // Floating platforms (x, y, w, h)
    var plats = [
      [300, 320, 140, 18],
      [520, 280, 140, 18],
      [780, 260, 160, 18],
      [1050, 300, 160, 18],
      [1320, 250, 160, 18],
      [1600, 230, 200, 18],
      [1850, 300, 160, 18],
      [2100, 270, 160, 18],
      [2400, 240, 200, 18],
      [2700, 300, 160, 18]
    ];
    for (var i=0;i<plats.length;i++){
      var p = plats[i];
      this.addPlatform(p[0], p[1], p[2], p[3]);
    }

    // Coins on platforms
    for (var k=0;k<plats.length;k++){
      var px = plats[k][0], py = plats[k][1];
      for (var c=0;c<3;c++){
        this.coins.push(new G.Coin(px + 25 + c*45, py - 14));
      }
    }
    // Coins along ground
    for (var gx=150; gx<this.worldWidth; gx+= 350){
      this.coins.push(new G.Coin(gx, groundY - 14));
    }

    // Enemies patrolling near some platforms
    this.enemies.push(new G.Enemy(360, groundY - 36, 300, 600));
    this.enemies.push(new G.Enemy(820, groundY - 36, 760, 980));
    this.enemies.push(new G.Enemy(1350, groundY - 36, 1280, 1500));
    this.enemies.push(new G.Enemy(1900, groundY - 36, 1800, 2100));
    this.enemies.push(new G.Enemy(2500, groundY - 36, 2400, 2650));
    this.enemies.push(new G.Enemy(2850, groundY - 36, 2780, 3000));

    this.spawnX = 40;
    this.spawnY = groundY - 42;
  }

  Level.prototype.addPlatform = function(x, y, w, h){
    this.platforms.push({x:x, y:y, w:w, h:h});
  };

  Level.prototype.getCollisions = function(rect){
    var hits = [];
    for (var i=0;i<this.platforms.length;i++){
      var p = this.platforms[i];
      if (Game.rectsIntersect(rect, p)) hits.push(p);
    }
    return hits;
  };

  Level.prototype.draw = function(ctx, camX){
    // Platforms and ground
    for (var i=0;i<this.platforms.length;i++){
      var p = this.platforms[i];
      var sx = Math.floor(p.x - camX);
      if (sx + p.w < -5 || sx > G.WIDTH + 5) continue;
      ctx.fillStyle = (p.h > 50 ? G.colors.ground : G.colors.platform);
      ctx.fillRect(sx, Math.floor(p.y), p.w, p.h);
      // Top highlight
      if (p.h <= 50){
        ctx.fillStyle = "rgba(255,255,255,0.08)";
        ctx.fillRect(sx, Math.floor(p.y), p.w, 3);
      }
    }
  };

  G.Level = Level;
})();