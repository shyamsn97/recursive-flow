// js/boid.js
export default class Boid {
  constructor(x, y, id=0) {
    this.pos = {x: x, y: y};
    const angle = Math.random() * Math.PI * 2;
    const speed = 2 + Math.random() * 2;
    this.vel = {x: Math.cos(angle) * speed, y: Math.sin(angle) * speed};
    this.acc = {x: 0, y: 0};
    this.maxForce = 0.05;
    this.maxSpeed = 4;
    this.id = id;
  }

  applyForce(f){
    this.acc.x += f.x;
    this.acc.y += f.y;
  }

  // Vector helpers
  static add(a, b){ return {x: a.x + b.x, y: a.y + b.y}; }
  static sub(a, b){ return {x: a.x - b.x, y: a.y - b.y}; }
  static mult(v, s){ return {x: v.x * s, y: v.y * s}; }
  static div(v, s){ return {x: v.x / s, y: v.y / s}; }
  static mag(v){ return Math.hypot(v.x, v.y); }
  static setMag(v, m){
    const mag = Boid.mag(v) || 1;
    return {x: v.x / mag * m, y: v.y / mag * m};
  }
  static limit(v, max){
    const m = Boid.mag(v);
    if (m > max) return Boid.setMag(v, max);
    return v;
  }

  update(){
    this.vel.x += this.acc.x;
    this.vel.y += this.acc.y;
    this.vel = Boid.limit(this.vel, this.maxSpeed);
    this.pos.x += this.vel.x;
    this.pos.y += this.vel.y;
    this.acc.x = 0;
    this.acc.y = 0;
  }

  edges(width, height){
    if (this.pos.x > width) this.pos.x = 0;
    if (this.pos.x < 0) this.pos.x = width;
    if (this.pos.y > height) this.pos.y = 0;
    if (this.pos.y < 0) this.pos.y = height;
  }

  // Behaviors: alignment, cohesion, separation
  align(boids, perception=50){
    let steering = {x:0,y:0}, total=0;
    for (const other of boids){
      const d = Math.hypot(this.pos.x - other.pos.x, this.pos.y - other.pos.y);
      if (other !== this && d < perception){
        steering.x += other.vel.x;
        steering.y += other.vel.y;
        total++;
      }
    }
    if (total > 0){
      steering = Boid.div(steering, total);
      steering = Boid.setMag(steering, this.maxSpeed);
      steering = Boid.sub(steering, this.vel);
      steering = Boid.limit(steering, this.maxForce);
    }
    return steering;
  }

  cohesion(boids, perception=60){
    let steering = {x:0,y:0}, total=0;
    for (const other of boids){
      const d = Math.hypot(this.pos.x - other.pos.x, this.pos.y - other.pos.y);
      if (other !== this && d < perception){
        steering.x += other.pos.x;
        steering.y += other.pos.y;
        total++;
      }
    }
    if (total > 0){
      steering = Boid.div(steering, total);
      steering = Boid.sub(steering, this.pos);
      steering = Boid.setMag(steering, this.maxSpeed);
      steering = Boid.sub(steering, this.vel);
      steering = Boid.limit(steering, this.maxForce);
    }
    return steering;
  }

  separation(boids, perception=30){
    let steering = {x:0,y:0}, total=0;
    for (const other of boids){
      const d = Math.hypot(this.pos.x - other.pos.x, this.pos.y - other.pos.y);
      if (other !== this && d < perception){
        let diff = Boid.sub(this.pos, other.pos);
        if (d !== 0) diff = Boid.div(diff, d); // weight by distance
        steering.x += diff.x;
        steering.y += diff.y;
        total++;
      }
    }
    if (total > 0){
      steering = Boid.div(steering, total);
      steering = Boid.setMag(steering, this.maxSpeed);
      steering = Boid.sub(steering, this.vel);
      steering = Boid.limit(steering, this.maxForce * 1.5);
    }
    return steering;
  }

  flock(boids, weights = {align:1.0, cohesion:1.0, separation:1.5}){
    const a = this.align(boids, 50);
    const c = this.cohesion(boids, 60);
    const s = this.separation(boids, 28);
    this.applyForce(Boid.mult(a, weights.align));
    this.applyForce(Boid.mult(c, weights.cohesion));
    this.applyForce(Boid.mult(s, weights.separation));
  }

  // draw as colorful triangle oriented along velocity
  draw(ctx, hue=180, size=6){
    const angle = Math.atan2(this.vel.y, this.vel.x);
    ctx.save();
    ctx.translate(this.pos.x, this.pos.y);
    ctx.rotate(angle);
    ctx.beginPath();
    ctx.moveTo(size, 0);
    ctx.lineTo(-size * 0.6, size * 0.6);
    ctx.lineTo(-size * 0.6, -size * 0.6);
    ctx.closePath();
    ctx.fillStyle = `hsl(${hue} 95% 55%)`;
    ctx.fill();
    ctx.restore();
  }
}
