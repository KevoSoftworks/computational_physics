import numpy as np

class Particle:
	def __init__(self, r, v, m=1, k=1):
		self.r = r
		self.v = v

		self.m = m
		self.k = k

	def update(self, F, dt):
		self.v += 0.5*F(self.k, self.r)/self.m*dt
		self.r += self.v*dt
		self.v += 0.5*F(self.k, self.r)/self.m*dt

class ParticleManager:
	def __init__(self, m=1, k=1):
		self.m = m
		self.k = k

	def generate(self, N, rlim=(-1, 1), vlim=(-1, 1), Ndim=1):
		p = []
		rpool = np.random.rand(N, Ndim) * (rlim[1] - rlim[0]) + rlim[0]
		vpool = np.random.rand(N, Ndim) * (vlim[1] - vlim[0]) + vlim[0]
		for i in range(N):
			p.append(Particle(rpool[i], vpool[i], self.m, self.k))

		return p

class Grid:
	def __init__(self, particles, dt, force=lambda k, r: 0):
		self.particles = particles
		self.dt = dt
		self.ticks = 0
		self.force = force

	def update(self):
		for p in self.particles:
			p.update(self.force, self.dt)
		
		self.ticks += 1