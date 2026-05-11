(function(){
  var G = window.Game;

  var canvas = document.getElementById("game");
  var ctx = canvas.getContext("2d");

  G.Input.init();

  var level = new G.Level();
  var player = new G.Player(level.spawnX, level.spawnY);

  var cameraX = 0;
  var last = performance.now();
  var acc = 0;

  var score = 0;
  var lives = 3;
  var gameOver = false;

  function resetGame(){
    level = new G.Level();
    player = new G.Player(level.spawnX, level.spawnY);
    cameraX = 0;
    score = 0;
    lives = 3;
    gameOver = false;
  }

  function update(dt){
    if (gameOver){
      // Allow restart
      if (G.Input.isDown("KeyR")) resetGame();
      return;
    }

    // Clamp dt (avoid large steps)
    if (dt > 0.033) dt = 0.033;

    player.update(dt, level, G.Input);

    // Coins
    for (var i=0;i<level.coins.length;i++){
      if (level.coins[i].update(player)) score += 1;
    }

    // Enemies
    for (var e=0;e<level.enemies.length;e++){
      var en = level.enemies[e];
      en.update(dt, level);
      var res = en.collideWithPlayer(player);
      if (res === "hit"){
        lives -= 1;
        player.reset(level.spawnX, level.spawnY);
        if (lives <= 0){
          gameOver = true;
        }
      }
    }

    // Camera follows player
    cameraX = G.clamp(player.x + player.w/2 - G.WIDTH/2, 0, level.worldWidth - G.WIDTH);
  }

  function draw(t){
    // Sky
    G.drawSky(ctx, canvas.width, canvas.height);

    // Midground hills (parallax)
    ctx.fillStyle = G.colors.bg;
    var hillX = - (cameraX * 0.3 % 600);
    for (var i=-1;i<5;i++){
      ctx.beginPath();
      ctx.moveTo(hillX + i*600, G.HEIGHT);
      ctx.quadraticCurveTo(hillX + i*600 + 150, 340, hillX + i*600 + 300, G.HEIGHT);
      ctx.fill();
    }

    // World
    level.draw(ctx, cameraX);

    // Coins
    for (var i=0;i<level.coins.length;i++){
      level.coins[i].draw(ctx, cameraX, t*0.001);
    }
    // Enemies
    for (var e=0;e<level.enemies.length;e++){
      level.enemies[e].draw(ctx, cameraX);
    }

    // Player
    player.draw(ctx, cameraX);

    // HUD
    ctx.fillStyle = "rgba(0,0,0,0.4)";
    ctx.fillRect(8, 8, 210, 60);
    ctx.fillStyle = G.colors.hud;
    ctx.font = "16px sans-serif";
    ctx.textBaseline = "top";
    ctx.fillText("Score: " + score + " / " + level.coins.length, 16, 16);
    ctx.fillText("Lives: " + lives, 16, 38);

    if (score >= level.coins.length && !gameOver){
      ctx.fillStyle = "rgba(0,0,0,0.5)";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = "#fff";
      ctx.font = "24px sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText("You collected all coins! Press R to restart", canvas.width/2, canvas.height/2);
    }

    if (gameOver){
      ctx.fillStyle = "rgba(0,0,0,0.6)";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = "#fff";
      ctx.font = "28px sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText("Game Over — Press R to restart", canvas.width/2, canvas.height/2);
    }
  }

  function frame(now){
    var dt = (now - last) / 1000;
    last = now;
    update(dt);
    draw(now);
    requestAnimationFrame(frame);
  }

  requestAnimationFrame(frame);
})();