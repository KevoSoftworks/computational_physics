import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from functools import reduce
from itertools import product
from copy import deepcopy

class Particle:
	def __init__(self, r, v, m=1, k=1):
		self.r = r
		self.v = v

		self.m = m
		self.k = k

		self.bounds = None

	def update(self, F, dt, *args):
		self.v += 0.5*F.calc(self.k, self.r, args[0], args[1], self.bounds)/self.m*dt
		self.r += self.v*dt
		
		if self.bounds is not None:
			self.r = ((self.r - self.bounds[0]) % (self.bounds[1] - self.bounds[0])) + self.bounds[0]
			if any(self.r > 1) or any(self.r < 0):
				print("FUCK")
			if self.r[0] < 0:
				print("FUUUU")

		self.v += 0.5*F.calc(self.k, self.r, args[0], args[1], self.bounds)/self.m*dt
	
	def set_bounds(self, lim):
		self.bounds = lim
		return self

	def get_closest(self, p):
		if self.bounds is None:
			return p.r

		ri = deepcopy(self.r) - self.bounds[0]
		rj = deepcopy(p.r) - self.bounds[0]
		L = self.bounds[1] - self.bounds[0]

		ret = []
		
		for ci, cj in zip(ri, rj):
			coords = np.array([cj - L, cj, cj + L])
			index = np.argmin(np.abs(coords - ci))
			ret.append(coords[index])

		return np.array(ret) + self.bounds[0]

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

	def generate_grid(self, N, rlim=(0, 1), vlim=(-1, 1), Ndim=3, ret=None):
		def divisors(n, dim):
			if dim == 1:
				return [n]
			for i in range(int(n/2)+1, 1, -1):
				if n % i == 0:
					res = divisors(int(n / i), dim-1)
					if res is not False:
						return [i, *res]
			return False

		p = []

		count = divisors(N, Ndim)

		spaces = [np.linspace(*rlim, count[i], endpoint=False) for i in range(Ndim)]
		rpool = list(product(*spaces))
		vpool = np.random.rand(N, Ndim) * (vlim[1] - vlim[0]) + vlim[0]
		for i in range(N):
			p.append(Particle(np.array(rpool[i]), vpool[i], self.m, self.k).set_bounds(rlim))

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
	LENNARD_JONES = 3

	def __init__(self, expr = lambda *args: 0):
		self.expr = expr
		self.type = Force.NONE

	def spring_simple(self):
		self.expr = lambda k, r, *args: -k*r
		self.type = Force.SPRING_SIMPLE
		return self

	def spring_nearest_neighbours(self):
		self.expr = lambda k, r, r1, r2, *args: -k*(2*r - r1.r - r2.r)
		self.type = Force.SPRING_NN
		return self

	def lennard_jones(self, rc):
		#self.expr = lambda k, r, *args: 24/r * (1/r**6 - 2/r**12) if r <= rc else 0
		def lj(k, _, pi, pj, *args):
			r = pi.get_closest(pj)
			r_hat = r / np.linalg.norm(r)
			dist = np.sqrt(np.sum(r**2))

			val = 24/dist * (1/(dist**6) - 2/(dist**12)) if dist <= rc else 0
			val *= r_hat

			if np.isnan(val[:]).any():
				print("Fuck")
			print(val)
			return val

		self.expr = lj
		self.type = Force.LENNARD_JONES
		return self
	
	def calc(self, *args):
		return self.expr(*args)

class Grid:
	def __init__(self, particles, dt, force=Force()):
		self.particles = particles
		self.dt = dt
		self.ticks = 0
		self.force = force

		self.history = []

	def __getitem__(self, index):
		if(type(index) is int):
			index %= len(self.particles)

		return self.particles[index]
	
	def __len__(self):
		return len(self.particles)

	def update(self):
		for i, p in enumerate(self.particles):
			p.update(self.force, self.dt, self[i-1], self[i+1], self.particles)

		self.history.append([j.r for j in self[:]])
		
		self.ticks += 1

	def animate(self, lim):
		fig = plt.figure()
		ax = fig.add_subplot(111, projection="3d")
		ax.set_xlim3d(lim)
		ax.set_ylim3d(lim)
		ax.set_zlim3d(lim)
		scatter = ax.scatter([p[0] for p in self.history[0]], [p[1] for p in self.history[0]], [p[2] for p in self.history[0]])

		def anim(frame, grid):
			scatter._offsets3d([p[0] for p in self.history[frame]], [p[1] for p in self.history[frame]], [p[2] for p in self.history[frame]])
			return scatter

		#handler = FuncAnimation(fig, anim, frames=len(self.history), fargs=self, repeat=False, blit=False, interval=50)
		#return handler