import numpy as np
import matplotlib.pyplot as plt

from scipy.special import genlaguerre

from enum import Enum
from abc import ABC, abstractmethod

class GridType(Enum):
	LINEAR = 0
	LOG = 1

class Grid:
	def __init__(self, points, start = 0.0, end = 10.0, type=GridType.LINEAR, t1=0):
		self.start = start
		self.end = end
		self.points = points
		self.type = type
		self.t1 = t1

		self.grid, self.step = np.linspace(self.start, self.end, self.points, \
									endpoint=True, retstep=True)

	def __add__(self, num):
		return self.grid + num

	def __sub__(self, num):
		return self.grid - num
	
	def __mul__(self, num):
		return self.grid * num

	def __truediv__(self, num):
		return self.grid / num

	def __pow__(self, power):
		return self.grid ** 2

	def __repr__(self):
		return self.grid

	def __lt__(self, comp):
		return self.grid < comp

	def __gt__(self, comp):
		return self.grid > comp

	def __le__(self, comp):
		return self.grid <= comp

	def __ge__(self, comp):
		return self.grid >= comp

	def __eq__(self, comp):
		return self.grid == comp

	def __ne__(self, comp):
		return self.grid != comp
	
	def __len__(self):
		return self.points

	def __getitem__(self, index):
		return self.grid[index]

class Potential(ABC):
	@abstractmethod
	def outerTurningPoint(self, lambda_, **kwargs):
		pass

	@abstractmethod
	def W(self, rho):
		pass
	
	@abstractmethod
	def analytical(self, rho):
		pass

class HarmonicPotential(Potential):
	def __init__(self):
		pass

	def outerTurningPoint(self, lambda_, l = 0, **kwargs):
		if l == 0:
			return 2*np.sqrt(lambda_)
		else:
			raise NotImplementedError(f"Outer turning point cannot be determined for l={self.l}")
	
	def W(self, rho, l = 0, **kwargs):
		# Special case when l = 0
		if l == 0:
			return 0.25 * rho**2
		
		return (l * (l + 1)) / rho**2 + 0.25 * rho**2

	def analytical(self, rho, l = 0, k = 0):
		poly = genlaguerre(k, l + 0.5)
		return rho**(l+1)*np.exp(-0.25*rho**2)*poly(0.5*rho**2)

class CoulombPotential(Potential):
	def __init__(self):
		pass

	def outerTurningPoint(self, lambda_, l=0, **kwargs):
		if l == 0:
			return -2 / lambda_
		
		return (np.sqrt(lambda_ * l * (l + 1) + 1) - 1) / lambda_
	
	def W(self, rho, l=0, **kwargs):
		# Special case when l = 0
		if l == 0:
			return -2 / rho

		return (l * (l + 1)) / rho**2 - 2 / rho
	
	def analytical(self, rho, l=0, k=0):
		raise NotImplementedError

class CoulombPotentialLog(CoulombPotential):	# As one wise Zoz once said: "Fuck!"
	def W(self, u, l=0, t1=0, **kwargs):
		if l == 0:
			return -2 / (np.exp(u) - t1)

		return (l * (l + 1)) / (np.exp(u) - t1)**2 - 2 / (np.exp(u) - t1)
		


class WaveFunction:
	def __init__(self, rho, potential, l = 0, k = 0):
		self.rho = rho
		self.potential = potential
		self.l = l
		self.k = k

		self.wave = np.zeros(len(rho))

	def __sub__(self, item):
		if isinstance(item, WaveFunction):
			return self.wave - item.wave
		return self.wave - item

	def __repr__(self):
		return self.wave
	
	def __len__(self):
		return len(self.wave)
	
	def __getitem__(self, index):
		return self.wave[index]

	def propagate(self, lambda_, numerov=False):
		# A bit of a hacky workaround, but track the latest lambda for F(),
		# and whether we used numerov to calculate the solution
		self.lambda_ = lambda_
		self.numerov = numerov

		# Set the initial values
		self.wave[0:2] = \
			self.rho[0:2] ** (self.l + 1) if self.rho.type == GridType.LINEAR \
			else np.exp(self.rho[0:2]) ** (self.l + 1)
		self.wave[-2:] = \
			np.exp(-self.rho[-2:] * np.sqrt(abs(lambda_))) if self.rho.type == GridType.LINEAR \
			else np.exp(-np.exp(self.rho[-2:]) * np.sqrt(abs(lambda_)))

		index = self._otpIndex(lambda_)

		self._propagateForward(index, lambda_, numerov)
		crossval = self.wave[index]
		self._propagateBackward(index, lambda_, numerov)

		print(self.wave)

		# Ensure continuity
		self.wave[index:] = crossval / self.wave[index] * self.wave[index:]
		self.wave[index] = crossval

		return self
	
	def normalise(self):
		factor = \
			(np.linalg.norm(self.wave) * np.sqrt(self.rho.step)) if self.rho.type == GridType.LINEAR \
			else (np.linalg.norm(self.wave * np.exp(self.rho)) * np.sqrt(self.rho.step))

		self.wave /= factor

		return self
	
	def analytical(self):
		for i in range(len(self.wave)):
			self.wave[i] = self.potential.analytical(self.rho[i])

		return self
	
	def error(self, wf):
		return np.abs(self - wf)

	def F(self):
		index = self._otpIndex(self.lambda_)

		# Forward
		fwd = np.array([self._propagateSingle(\
				(self.rho[i], self.rho[i-1], self.rho[i-2]), \
				(self.wave[i-1], self.wave[i-2]), \
				self.lambda_, self.numerov) \
			for i in range(index - 1, index + 2)])

		# Backward (note: the first element has the highest i)
		bwd = np.array([self._propagateSingle(\
				(self.rho[i], self.rho[i+1], self.rho[i+2]), \
				(self.wave[i+1], self.wave[i+2]), \
				self.lambda_, self.numerov) \
			for i in range(index + 1, index - 2, -1)])

		# Ensure continuity with the actual wave function
		fwd *= self.wave[index] / fwd[1]
		bwd *= self.wave[index] / bwd[1]
		
		return (bwd[0] - bwd[-1]) - (fwd[-1] - fwd[0])

	
	def _otpIndex(self, lambda_):
		otp = self.potential.outerTurningPoint(lambda_, l = self.l, k = self.k)
		if self.rho.type == GridType.LOG:
			otp = np.log(otp)

		return np.where(self.rho >= otp)[0][0]

	def _propagateForward(self, index, lambda_, numerov):
		iter = range(2, index + 1)

		for i in iter:
			self.wave[i] = self._propagateSingle(\
							(self.rho[i], self.rho[i-1], self.rho[i-2]), \
							(self.wave[i-1], self.wave[i-2]), \
							lambda_, numerov)

	def _propagateBackward(self, index, lambda_, numerov):
		iter = range(len(self.wave) - 3, index - 1, -1)

		for i in iter:
			self.wave[i] = self._propagateSingle(\
							(self.rho[i], self.rho[i+1], self.rho[i+2]), \
							(self.wave[i+1], self.wave[i+2]), \
							lambda_, numerov)
	
	def _propagateSingle(self, r, w, lambda_, numerov):
		if self.rho.type == GridType.LINEAR:
			q = lambda x: 1 - self.rho.step**2 / 12 * (self.potential.W(x) - lambda_)
		else:
			q = lambda x: 1 - self.rho.step**2 / 12 * ((self.potential.W(x, self.rho.t1) - lambda_\
				* np.exp(2*x)) + 0.25)

		if numerov:
			g = (self.potential.W(r[1], l = self.l, k = self.k) - lambda_) if self.rho.type == GridType.LINEAR \
				else (self.potential.W(r[1], l = self.l, k = self.k) - lambda_)*np.exp(2*r[1]) + 0.25

			return 1/q(r[0]) * ( \
				self.rho.step**2 * g * w[0] \
				+ 2*w[0]*q(r[1]) \
				- w[1]*q(r[2]))
		else:
			return 2 * w[0] - w[1] \
				+ self.rho.step**2 * (self.potential.W(r[1], l = self.l, k = self.k) - lambda_) \
				* w[0]


class Solver:
	def __init__(self, grid, potential, error=1E-4, numerov=True, l=0):
		self.grid = grid
		self.pot = potential
		self.wf = WaveFunction(self.grid, self.pot, l)
		self.error = error
		self.numerov = numerov

	def __getitem__(self, index):
		pass

	def bisect(self, left, right, fullreturn=False):
		# Since we are using recursion, we'll have Python crash itself once we
		# reach a recursion limit. Fun times!
		self.wf.propagate(left, self.numerov)
		self.wf.normalise()

		Fleft = self.wf.F()

		lambda_ = (left + right) / 2.
		self.wf.propagate(lambda_, self.numerov)
		self.wf.normalise()

		F = self.wf.F()
		err = (right - left) / 2

		if np.sign(F) == np.sign(Fleft):
			left = lambda_
		else:
			right = lambda_

		if F == 0 or (err < self.error and abs(((left + right) / 2) - lambda_) < self.error):
			if fullreturn:
				return [[lambda_], [F]]
			else:
				return [lambda_, F]
		
		if fullreturn:
			res = self.bisect(left, right, fullreturn)
			return [[lambda_, *res[0]], [F, *res[1]]]
		else:
			return self.bisect(left, right, fullreturn)
	
	def search(self, guess, fullreturn=False, damping=1):
		self.wf.propagate(guess, self.numerov)
		self.wf.normalise() # This ensures that A = 1
		otp = self.wf._otpIndex(guess)
		f = self.wf.F() * damping

		lambda_ = guess - f / (2*self.grid.step) * self.wf[otp]

		if abs(f) < self.error and abs(guess - lambda_) < self.error:
			# The previous iteration was accurate enough
			if fullreturn:
				return [[guess], [f]]
			else:
				return [guess, f]

		res = self.search(lambda_, fullreturn)
		if fullreturn:
			return [[guess, *res[0]], [f, *res[1]]]
		else:
			return res



def plot(grid, ana, *args, err_index=0, title=None, legend=(), scatter=False):
	# Setup plot
	w, h = plt.figaspect(2.0)
	plt.figure(figsize=(h,h))

	# Plot the wavefunctions
	if isinstance(ana, WaveFunction):
		plt.subplot2grid((3, 1), (0, 0), rowspan=2)
	else:
		plt.subplot2grid((2, 1), (0, 0), rowspan=2)

	labels = [legend[i] if i < len(legend) else f"Computed solution ({i+1})" for i in range(len(grid))]

	for i, wf in enumerate(args):
		if scatter:
			plt.scatter(grid, wf, label=labels[i])
		else:
			plt.plot(grid, wf, label=labels[i])
	
	if isinstance(ana, WaveFunction):
		plt.plot(grid, ana, 'k', linewidth = 0.5, label="Analytical solution")

	if title == None:
		plt.title(f"Wavefunctions for grid n = {len(grid)}")
	else:
		plt.title(title)

	plt.xlabel(r"$\rho$")
	plt.ylabel(r"$\zeta$")
	plt.legend()
	plt.grid()

	# Plot the error (if it is provided)
	if isinstance(ana, WaveFunction):
		plt.subplot2grid((3, 1), (2, 0))

		if isinstance(err_index, tuple):
			for i in err_index:
				plt.plot(ana.error(args[i]), label=f"Error in {labels[i]}")
		else:
			plt.plot(ana.error(args[err_index]), label=f"Error in {labels[err_index]}")

		plt.title(f"Error between analytical and computed solutions")
		plt.xlabel(r"$\rho$")
		plt.ylabel(r"$\Delta\zeta$")
		plt.ticklabel_format(style="sci", axis="y", scilimits=(0,0))
		plt.legend()
		plt.grid()

	plt.tight_layout()

if __name__ == "__main__":
	print("This script can only be imported.")