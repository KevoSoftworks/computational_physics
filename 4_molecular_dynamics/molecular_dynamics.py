import numpy as np
import matplotlib.pyplot as plt
from uuid import uuid4 as uuid
from matplotlib.animation import FuncAnimation
from functools import reduce
from itertools import product
from copy import deepcopy

class Particle:
	def __init__(self, r, v, m=1, k=1):
		self.id = uuid()

		self.r = r
		self.v = v

		self.m = m
		self.k = k

		self.bounds = None

	def update(self, F, dt, *args):
		#0: p(i-1), 1: p(i+1), 2: all p
		self.v += 0.5*F.calc(self.k, self, args[0], args[1], args[2])/self.m*dt
		self.r += self.v*dt
		
		if self.bounds is not None:
			#self.r = ((self.r - self.bounds[0]) % (self.bounds[1] - self.bounds[0])) + self.bounds[0]
			self.r %= self.bounds[1]

		self.v += 0.5*F.calc(self.k, self, args[0], args[1], args[2])/self.m*dt
	
	def set_bounds(self, lim):
		self.bounds = lim
		return self

	def get_closest(self, p):
		if self.bounds is None:
			return p.r

		ri = self.r #deepcopy(self.r) - self.bounds[0]
		rj = p.r #deepcopy(p.r) - self.bounds[0]
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
		p = []
		count = np.array([int(np.ceil(N ** (1/Ndim))) for _ in range(Ndim)])

		spaces = [np.linspace(*rlim, count[i], endpoint=False) for i in range(Ndim)]
		rpool = np.array(list(product(*spaces)))
		vpool = np.random.rand(N, Ndim) * (vlim[1] - vlim[0]) + vlim[0]
		for i in range(N):
			p.append(Particle(rpool[i], vpool[i], self.m, self.k).set_bounds(rlim))

		self.p = p

		if ret is None:
			return p
		
		return self

	def remove_translation(self, p=None, ret=None):
		if p is None:
			p = self.p

		m = np.sum([i.m for i in p])
		mv = np.sum([i.m*i.v for i in p], axis=0)

		for pa in p:
			pa.v -= mv / m

		self.p = p

		if ret is None:
			return p
		
		return self

	def scale(self, T, ret=None):
		K = np.array([0.5*i.m*i.v**2 for i in self.p])
		cur_T = np.mean(2/len(K) * K)

		ratio = T / cur_T

		for i in self.p:
			i.v *= np.sqrt(ratio)
		
		if ret is None:
			return self.p
		
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
		self.expr = lambda k, r, *args: -k*r.r
		self.type = Force.SPRING_SIMPLE
		return self

	def spring_nearest_neighbours(self):
		self.expr = lambda k, r, r1, r2, *args: -k*(2*r.r - r1.r - r2.r)
		self.type = Force.SPRING_NN
		return self

	def lennard_jones(self, rc):
		def lj(k, pi, _1, _2, pall):
			"""r = np.array([pi.get_closest(pj) for pj in pall if pi.id != pj.id])
			dist = 1/np.sqrt(np.sum((r - pi.r) ** 2, axis=1))
			dir = (r - pi.r) / np.linalg.norm(r - pi.r)

			dir = dir[dist <= rc]
			dist = 1/dist[dist <= rc]

			f = np.repeat(24*dist*(dist**6 - 2*dist**12), 3).reshape(np.shape(dir))

			tot1 = np.sum(f*dir, axis=0)
			return tot1"""

			tot = 0
			for pj in pall:
				if pi.id == pj.id:
					continue

				r = pi.get_closest(pj)
				dist = np.sqrt(np.sum((r - pi.r)**2))
				dir = (r - pi.r) / np.linalg.norm(r - pi.r)

				#if dist < 0.5:
				#	print("extreme")

				if(dist == 0):
					print(f"{dist}, {r}")
					print(f"{pi.id}, {pi.r}")
					print(f"{pj.id}, {pj.r}")
					raise Exception("FUCK!")

				val = 24/dist * (1/(dist**6) - 2/(dist**12)) if dist <= rc else 0
				val *= dir

				tot += val

			#if np.sum(tot) != np.sum(tot1):
			#	raise Exception(f"{tot} != {tot1}")

			#if np.sum(tot) > 100:
				#print("Fuck...")

			return tot

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

	@property
	def E(self):
		#TODO
		K = np.sum([0.5*p.m*p.v**2 for p in self[:]])
		V = []
		for p in self[:]:
			r = np.array([p.get_closest(pj) for pj in self[:] if p.id != pj.id])
			dist = 1/np.sqrt(np.sum((r - p.r) ** 2, axis=1))

	def update(self):
		part = deepcopy(self.particles)
		for i, p in enumerate(self.particles):
			p.update(self.force, self.dt, self[i-1], self[i+1], part)

		self.history.append([deepcopy(j.r) for j in self.particles])
		
		self.ticks += 1

	def animate(self, lim):
		fig = plt.figure()
		ax = fig.add_subplot(111, projection="3d")
		ax.set_xlim3d(lim)
		ax.set_xlabel("x")
		ax.set_ylim3d(lim)
		ax.set_ylabel("y")
		ax.set_zlim3d(lim)
		ax.set_zlabel("z")
		scatter, = ax.plot(np.array([p[0] for p in self.history[0]]), np.array([p[1] for p in self.history[0]]), np.array([p[2] for p in self.history[0]]), marker="o", linestyle="")

		def anim(frame, hist):
			scatter.set_xdata(hist[frame][:,0])
			scatter.set_ydata(hist[frame][:,1])
			scatter.set_3d_properties(hist[frame][:,2])
			return scatter,

		handler = FuncAnimation(fig, anim, frames=len(self.history), fargs=[np.array(self.history)], repeat=False, blit=False, interval=1000/30)
		return handler