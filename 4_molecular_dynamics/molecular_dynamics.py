import numpy as np
from functools import reduce

class Particle:
	def __init__(self, r, v, m=1, k=1):
		self.r = r
		self.v = v

		self.m = m
		self.k = k

	def update(self, F, dt, *args):
		self.v += 0.5*F.calc(self.k, self.r, args[0].r, args[1].r)/self.m*dt
		self.r += self.v*dt
		self.v += 0.5*F.calc(self.k, self.r, args[0].r, args[1].r)/self.m*dt

class ParticleManager:
	def __init__(self, m=1, k=1):
		self.m = m
		self.k = k

	def generate(self, N, rlim=(-1, 1), vlim=(-1, 1), Ndim=1, ret=None):
		p = []
		rpool = np.random.rand(N, Ndim) * (rlim[1] - rlim[0]) + rlim[0]
		vpool = np.random.rand(N, Ndim) * (vlim[1] - vlim[0]) + vlim[0]
		for i in range(N):
			p.append(Particle(rpool[i], vpool[i], self.m, self.k))

		self.p = p

		if ret is None:
			return p
		
		return self

	def remove_translation(self, p=None, ret=None):
		if p is None:
			p = self.p

		m = np.sum([i.m for i in p])
		mv = np.sum([i.m*i.v for i in p])

		for pa in p:
			pa.v -= mv / m

		self.p = p

		if ret is None:
			return p
		
		return self


class Force:
	NONE = 0
	SPRING_SIMPLE = 1
	SPRING_NN = 2

	def __init__(self, expr = lambda *args: 0):
		self.expr = expr
		self.type = Force.NONE

	def spring_simple(self):
		self.expr = lambda k, r, *args: -k*r
		self.type = Force.SPRING_SIMPLE
		return self

	def spring_nearest_neighbours(self):
		self.expr = lambda k, r, r1, r2: -k*(2*r - r1 - r2)
		self.type = Force.SPRING_NN
		return self
	
	def calc(self, *args):
		return self.expr(*args)

class Grid:
	def __init__(self, particles, dt, force=Force()):
		self.particles = particles
		self.dt = dt
		self.ticks = 0
		self.force = force

	def __getitem__(self, index):
		if(type(index) is int):
			index %= len(self.particles)

		return self.particles[index]
	
	def __len__(self):
		return len(self.particles)

	def update(self):
		for i, p in enumerate(self.particles):
			p.update(self.force, self.dt, self[i-1], self[i+1])
		
		self.ticks += 1