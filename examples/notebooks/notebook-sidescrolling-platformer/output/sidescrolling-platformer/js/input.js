(function(global){
  const GameNS = global.GameNS = global.GameNS || {};

  const pressed = new Set();
  const blockKeys = new Set(['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','Space']);

  function onDown(e){
    pressed.add(e.code);
    if (blockKeys.has(e.code)) e.preventDefault();
  }
  function onUp(e){
    pressed.delete(e.code);
    if (blockKeys.has(e.code)) e.preventDefault();
  }

  GameNS.Input = {};
  GameNS.Input.init = function(target){
    // target is a DOM element (canvas or window) to focus on click for keys to work on mobile/desktop
    window.addEventListener('keydown', onDown, {passive:false});
    window.addEventListener('keyup', onUp, {passive:false});
    if (target && target.focus) {
      target.tabIndex = 0;
      target.addEventListener('click', function(){ target.focus(); });
    }
  };
  GameNS.Input.isDown = function(code){
    // Accept both e.code values and simple aliases
    if (code === 'Left')  return pressed.has('ArrowLeft')  || pressed.has('KeyA');
    if (code === 'Right') return pressed.has('ArrowRight') || pressed.has('KeyD');
    if (code === 'Up')    return pressed.has('ArrowUp')    || pressed.has('KeyW') || pressed.has('Space');
    return pressed.has(code);
  };
})(window);
