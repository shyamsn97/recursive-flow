// Utility helpers for Boids simulation (ESM, no DOM deps)
export function randRange(min, max) {
  return min + Math.random() * (max - min);
}
export function clamp(x, lo, hi) {
  return Math.max(lo, Math.min(hi, x));
}
export function wrap(value, max) {
  // Proper positive modulo wrap to [0, max)
  if (max <= 0) return 0;
  let v = value % max;
  if (v < 0) v += max;
  return v;
}
export function hsl(h, s, l) {
  // Comma syntax is widely supported
  return `hsl(${h}, ${s}%, ${l}%)`;
}
export function len(vx, vy) {
  return Math.hypot(vx, vy);
}
export function limit(vx, vy, max) {
  const m = Math.hypot(vx, vy);
  if (m === 0 || m <= max) return [vx, vy];
  const k = max / m;
  return [vx * k, vy * k];
}
