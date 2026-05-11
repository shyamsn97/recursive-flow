(function(){
  var G = window.Game;
  function Coin(x, y){
    this.x = x; this.y = y;
    this.r = 8;
    this.collected = false;
    this.pulse = Math.random()*Math.PI*2;
  }
  Coin.prototype.rect = function(){ return {x:this.x - this.r, y:this.y - this.r, w:this.r*2, h:this.r*2}; };
  Coin.prototype.update = function(player){
    if (this.collected) return false;
    if (G.rectsIntersect(this.rect(), player.rect())){
      this.collected = true;
      return true;
    }
    return false;
  };
  Coin.prototype.draw = function(ctx, camX, t){
    if (this.collected) return;
    var bob = Math.sin(t*6 + this.pulse) * 2;
    ctx.fillStyle = G.colors.coin;
    ctx.beginPath();
    ctx.arc(Math.floor(this.x - camX), Math.floor(this.y + bob), this.r, 0, Math.PI*2);
    ctx.fill();
    ctx.strokeStyle = "#ccad00";
    ctx.lineWidth = 2;
    ctx.stroke();
  };
  G.Coin = Coin;
})();