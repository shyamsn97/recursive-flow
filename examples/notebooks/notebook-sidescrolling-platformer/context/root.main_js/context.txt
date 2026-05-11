(function(global){
  window.addEventListener('load', function(){
    const canvas = document.getElementById('game');
    if (!canvas) {
      console.error('Missing #game canvas');
      return;
    }
    const GameClass = (window.GameNS && window.GameNS.Game) ? window.GameNS.Game : null;
    if (!GameClass) {
      console.error('Game class not found on window.GameNS');
      return;
    }
    const game = new GameClass(canvas);
    if (game && game.start) game.start();
  });
})(window);
