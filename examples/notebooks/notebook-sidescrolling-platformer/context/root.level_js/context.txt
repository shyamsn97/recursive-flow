(function(global){
  const GameNS = global.GameNS = global.GameNS || {};

  class Level {
    constructor(){
      // Platforms are static rectangles: {x,y,w,h}
      // Build a long level width with some platforms.
      this.platforms = [];
      const groundY = 480; // relative to 540 canvas height
      const segW = 800;
      const segments = 6; // ~4800px wide
      for (let i=0;i<segments;i++){
        this.platforms.push({ x: i*segW, y: groundY, w: segW, h: 60 });
      }
      // Some elevated platforms
      this.platforms.push({ x: 300, y: 390, w: 140, h: 20 });
      this.platforms.push({ x: 520, y: 330, w: 120, h: 20 });
      this.platforms.push({ x: 860, y: 420, w: 120, h: 20 });
      this.platforms.push({ x: 1200, y: 360, w: 150, h: 20 });
      this.platforms.push({ x: 1600, y: 300, w: 160, h: 20 });
      this.platforms.push({ x: 2100, y: 380, w: 140, h: 20 });
      this.platforms.push({ x: 2500, y: 340, w: 140, h: 20 });
      this.platforms.push({ x: 3000, y: 420, w: 120, h: 20 });
      this.platforms.push({ x: 3450, y: 360, w: 180, h: 20 });
      this.platforms.push({ x: 3900, y: 300, w: 160, h: 20 });

      this.spawn = { x: 40, y: 200 };

      // Coin and enemy placements (world coordinates)
      this.coins = [
        {x: 350, y: 360}, {x: 580, y: 300}, {x: 900, y: 390}, {x: 1260, y: 330},
        {x: 1650, y: 270}, {x: 2120, y: 350}, {x: 2540, y: 310}, {x: 3030, y: 390},
        {x: 3490, y: 330}, {x: 3950, y: 270}
      ];
      this.enemies = [
        {x: 700, y: 450, range: 120},
        {x: 1450, y: 450, range: 140},
        {x: 2300, y: 450, range: 160},
        {x: 3200, y: 450, range: 120}
      ];
    }
    getPlatforms(){ return this.platforms; }
    getCoinPositions(){ return this.coins; }
    getEnemyDefs(){ return this.enemies; }
    getSpawn(){ return this.spawn; }
    getWorldWidth(){
      let maxR = 0;
      for (let p of this.platforms) maxR = Math.max(maxR, p.x + p.w);
      return Math.max(maxR, 4200);
    }
  }

  GameNS.Level = Level;
})(window);
