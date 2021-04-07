import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from functools import reduce
from itertools import product
from copy import deepcopy

class ParticleManager:
	def __init__(self, N, rlim=(-1, 1), vlim=(-1, 1), Ndim=1, m=1, k=1, as_grid=False, rho=None):
		self._pressure_component = None
		self._bins = None
		self._hist_rdy = False

		self.N = N
		self.Ndim = Ndim
		self.m = m
		self.k = k

		self.bounds = rlim
		self.rho = rho

		if as_grid and rho is not None:
			self._generate_grid(N, rho, vlim, Ndim)
		else:
			self._generate(N, rlim, vlim, Ndim)

	def _generate(self, N, rlim, vlim, Ndim):
		self.r = np.random.rand(N, Ndim) * (rlim[1] - rlim[0]) + rlim[0]
		self.v = np.random.rand(N, Ndim) * (vlim[1] - vlim[0]) + vlim[0]
		self.a = np.zeros((N, Ndim))

	def _generate_grid(self, N, rho, vlim=(-1, 1), Ndim=3, ret=None):
		rlim = (0, (N/rho)**(1/3))
		self.bounds = rlim

		size = int(np.ceil(N ** (1/Ndim)))
		axis, step = np.linspace(*rlim, size, endpoint=False, retstep=True)
		grid = np.stack(np.meshgrid(axis, axis, axis)).reshape((Ndim, size**Ndim)).T

		self.r = grid[:N] + step / 2 + np.random.uniform(-0.1, 0.1, (N, 3))
		self.v = np.random.rand(N, Ndim) * (vlim[1] - vlim[0]) + vlim[0]

	def init_hist(self, bins):
		self._bins = bins
		self._hist = np.zeros(bins, dtype=np.int32)

		return self

	def remove_translation(self):
		self.v -= np.sum(self.m * self.v, axis=0) / (self.N * self.m)

		return self

	def scale(self, T):
		ratio = T / self.T

		self.v *= np.sqrt(ratio)

		return self

	def set_initial_a(self, force):
		if force.type == Force.LENNARD_JONES:
			f, _ = force.calc(self)
		else:
			f = force.calc(self)
		self.a = f / self.m

		return self

	def verlet(self, force, dt):
		pot = None

		self.v += 0.5 * self.a * dt
		self.r += self.v * dt

		if force.type == Force.LENNARD_JONES:
			self.r %= self.bounds[1]
			f, pot = force.calc(self)
		else:
			f = force.calc(self)

		self.a = f / self.m
		self.v += 0.5 * self.a * dt

		return pot

	@property
	def K(self):
		return 0.5 * np.sum(self.m * self.v**2)

	@property
	def T(self):
		return 2/(3*self.N) * self.K	# eq. 3.2 of Ercolessi

	@property
	def V(self):
		if self.Ndim < 3:
			raise Exception(f"Volume cannot be determined for {self.Ndim} dimensions, need at least 3.")

		return (self.bounds[1] - self.bounds[0]) ** self.Ndim

	@property
	def P(self):
		if self.rho is None:
			raise Exception("No density defined, cannot compute pressure")

		if self._pressure_component is None:
			raise Exception("No pressure component set by the Force class. Please run the force computation.")

		return self.rho * self.k * self.T + self._pressure_component / (3 * self.V)


class Force:
	NONE = 0
	SPRING_SIMPLE = 1
	SPRING_NN = 2
	LENNARD_JONES = 3

	def __init__(self, expr = lambda *args: 0):
		self.expr = expr
		self.type = Force.NONE

	def spring_simple(self):
		self.expr = lambda pm: -pm.k * pm.r
		self.type = Force.SPRING_SIMPLE
		return self

	def spring_nearest_neighbours(self):
		self.expr = lambda pm: -pm.k*(2*pm.r - np.roll(pm.r, -1, axis=0) - np.roll(pm.r, 1, axis=0))
		self.type = Force.SPRING_NN
		return self

	def lennard_jones(self, rc):
		def compute_component(dist):
			inv_dist = 1 / dist

			return -24 * inv_dist * (inv_dist ** 6 - 2*inv_dist**12)

		def compute_potential(dist):
			inv_dist = 1 / dist

			return 4 * (inv_dist ** 12 - inv_dist ** 6)

		def lj(pm):
			# This code is the most unoptimised piece of crap that I have ever written.
			# However, it works, which is most important in the goal of solving the problems.
			# Because I have already spent a year trying to get this working, now that it
			# does, I am not changing it, in fear of breaking it again. 
			total_force = np.zeros(pm.r.shape)
			total_pot = 0
			total_pressure = 0

			pot_rc = compute_potential(rc)

			for i in range(pm.r.shape[0]):
				ri = pm.r[i]
				for j in range(i + 1, pm.r.shape[0]):
					rj = pm.r[j]

					diff = ri - rj
					diff -= np.sign(diff) * pm.bounds[1] * (np.abs(diff) // (pm.bounds[1] / 2))
					dist = np.linalg.norm(diff)

					if pm._bins is not None and pm._hist_rdy:
						hist, _ = np.histogram(dist, bins=pm._bins, range=(0, pm.bounds[1] / 2))
						pm._hist += hist

					if dist <= rc:
						tmp = compute_component(dist) * (diff / dist)

						total_pressure += tmp @ diff
						total_pot += (compute_potential(dist) - pot_rc)

						total_force[i] += tmp
						total_force[j] -= tmp

			# Hacky solution, but otherwise I need to change much more code
			pm._pressure_component = total_pressure
			return total_force, total_pot

		self.expr = lj
		self.type = Force.LENNARD_JONES
		return self
	
	def calc(self, *args):
		return self.expr(*args)

class Grid:
	def __init__(self, particle_manager, dt, force=Force()):
		self.pm = particle_manager
		self.dt = dt
		self.ticks = 0
		self.force = force

		self.history = []
		self.history.append(np.copy(self.pm.r))

	def __len__(self):
		return self.pm.r.shape[0]

	def update(self):
		pot = self.pm.verlet(self.force, self.dt)
		self.history.append(np.copy(self.pm.r))
		
		self.ticks += 1

		return pot

	def animate(self, lim):
		fig = plt.figure()
		ax = fig.add_subplot(111, projection="3d")
		ax.set_xlim3d(lim)
		ax.set_xlabel("x")
		ax.set_ylim3d(lim)
		ax.set_ylabel("y")
		ax.set_zlim3d(lim)
		ax.set_zlabel("z")
		scatter, = ax.plot(self.history[0][:, 0], self.history[0][:, 1], self.history[0][:, 2], marker="o", linestyle="")

		def anim(frame, hist):
			scatter.set_xdata(hist[frame][:,0])
			scatter.set_ydata(hist[frame][:,1])
			scatter.set_3d_properties(hist[frame][:,2])
			return scatter,

		handler = FuncAnimation(fig, anim, frames=len(self.history), fargs=[self.history], repeat=False, blit=False, interval=1000/30)
		return handler