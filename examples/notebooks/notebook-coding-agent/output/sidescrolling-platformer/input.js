(function(){
  var G = window.Game;
  var keys = {};
  var capture = new Set(["ArrowLeft","ArrowRight","ArrowUp","ArrowDown","Space","KeyA","KeyD","KeyW","KeyR"]);
  function kd(e){ if(capture.has(e.code)) e.preventDefault(); keys[e.code] = true; }
  function ku(e){ if(capture.has(e.code)) e.preventDefault(); keys[e.code] = false; }
  G.Input = {
    init: function(){
      window.addEventListener("keydown", kd, {passive:false});
      window.addEventListener("keyup", ku, {passive:false});
    },
    isDown: function(code){ return !!keys[code]; }
  };
})();