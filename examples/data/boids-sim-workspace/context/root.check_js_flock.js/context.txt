// js/flock.js
import Boid from './boid.js';

export function createFlock(count, width, height){
  const boids = [];
  for (let i=0;i<count;i++){
    const x = Math.random() * width;
    const y = Math.random() * height;
    const b = new Boid(x,y,i);
    boids.push(b);
  }
  return boids;
}
