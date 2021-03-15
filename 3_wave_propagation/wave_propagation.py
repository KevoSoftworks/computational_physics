from functools import cached_property, reduce

import numpy as np

class Matrix:
	def __init__(self, potential, lambda_):
		self.potential = potential
		self.lambda_ = lambda_
		self.midpoint = False
	
	def eta(self, y):
		if y < self.potential.start or y >= self.potential.end or not self.midpoint:
			return np.sqrt(self.potential[y] - self.lambda_ + 0j)

		return np.sqrt(self.potential[y + (self.potential.step / 2.)] - self.lambda_ + 0j)
	
	def flush(self):
		pass

	def transmission(self, dir="right"):
		raise NotImplementedError(f"Transmission not implemented for type {type(self)}")

	def reflection(self, dir="right"):
		raise NotImplementedError(f"Transmission not implemented for type {type(self)}")

	def T_wkb(self, D):
		prefactor = np.abs(self.eta(self.potential.y[-1])/self.eta(self.potential.y[0]))
		dE = [self.potential[y] - self.lambda_ for y in self.potential.y[2:-2]]

		return prefactor * D(self.lambda_) * \
			np.exp(-2 * np.sum(self.potential.step * np.sqrt(dE)))

class TransferMatrix(Matrix):
	DTYPE = np.cdouble

	@cached_property
	def M(self):
		p = lambda prev, cur: 1/(2*prev) * np.array([ \
			[prev + cur, prev - cur], \
			[prev - cur, prev + cur] \
		], dtype=TransferMatrix.DTYPE)

		q = lambda cur: np.array([ \
			[np.exp(cur * self.potential.step), 0], \
			[0, np.exp(-cur * self.potential.step)] \
		], dtype=TransferMatrix.DTYPE)

		product = np.array([[1, 0], [0, 1]], dtype=TransferMatrix.DTYPE)

		for y in self.potential.y[0:-1]:
			eta_prev = self.eta(y - self.potential.step)
			eta_cur = self.eta(y)

			product = product @ p(eta_prev, eta_cur) @ q(eta_cur)

		eta_prev = eta_cur
		eta_cur = self.eta(self.potential.end)
				
		return np.array([ \
			[np.exp(-self.eta(self.potential.start - self.potential.step) * self.potential.start), 0], \
			[0, np.exp(self.eta(self.potential.start - self.potential.step) * self.potential.start)] \
		], dtype=TransferMatrix.DTYPE) \
		@ product \
		@ p(eta_prev, eta_cur) \
		@ np.array([ \
			[np.exp(self.eta(self.potential.end) * self.potential.end), 0], \
			[0, np.exp(-self.eta(self.potential.end) * self.potential.end)] \
		], dtype=TransferMatrix.DTYPE)
	
	def flush(self):
		del self.M
	
	def transmission(self, dir="right"):
		# The definitions of right and left are reversed, unfortunately.
		if dir == "right":
			return 1/(np.abs(self.M[0, 0])**2)
		
		return np.abs(self.M[1, 1] - self.M[1, 0] * self.M[0, 1] / self.M[0, 0]) ** 2
	
	def reflection(self, *args):
		return np.abs(self.M[1, 0])**2 / np.abs(self.M[0, 0])**2

class ScatterMatrix(Matrix):
	@cached_property
	def S(self):
		class SubMatrix:
			def __init__(self, arr):
				self.mat = arr

			def __matmul__(self, other):
				a = self.mat
				b = other.mat
				return SubMatrix(np.array([
					[a[0,0] + a[0,1]*b[0,0]*(1 - a[1,1]*b[0,0])**(-1)*a[1,0], a[0,1]*(1 - b[0,0]*a[1,1])**(-1)*b[0,1]],
					[b[1,0]*(1 - a[1,1]*b[0,0])**(-1)*a[1,0], b[1,1] + b[1,0]*(1 - a[1,1]*b[0,0])**(-1)*a[1,1]*b[0,1]]
				]))
		
		def pre(prev, cur, nom=None):
			if nom is None:
				return (prev - cur) / (prev + cur)
			
			return (2*nom) / (prev + cur)
		
		m = []

		for y in self.potential.y:
			prev = self.eta(y - self.potential.step)
			cur = self.eta(y)

			m.append(SubMatrix(np.array([
				[pre(prev, cur) * np.exp(2*prev*y), pre(prev, cur, cur) * np.exp((prev-cur)*y)],
				[pre(prev, cur, prev) * np.exp((prev-cur)*y), -pre(prev, cur) * np.exp(-2*cur*y)]
			])))

		return reduce(lambda a, b: a @ b, m).mat
	
	def flush(self):
		del self.S

	def transmission(self, dir="right"):
		prefactor = np.abs(self.eta(self.potential.end + self.potential.step) \
			/ self.eta(self.potential.start - self.potential.step))

		if dir == "right":
			return prefactor * np.abs(self.S[1, 0])**2
		
		return prefactor * np.abs(self.S[0, 1])**2
	
	def reflection(self, *args):
		return np.abs(self.S[0, 0])**2


class Potential:
	def __init__(self, start, end, len, expr = None, outside_range = (lambda x: 0, lambda x: 0)):
		self.start = start
		self.end = end
		self.len = len

		self.pot = np.zeros(self.len)
		self.y, self.step = np.linspace(self.start, self.end, self.len, retstep=True)

		if type(expr) == int or type(expr) == float:
			self.pot = expr * np.ones(self.len)
		elif callable(expr):
			self.pot = expr
		else:
			raise NotImplementedError(f"Unsupported expression type {type(expr)}")

		if [callable(i) for i in outside_range] != [True, True]:
			raise ValueError("Both values in `outside_range` must be callable")

		self.outside_range = outside_range

	def __getitem__(self, y):
		if y < self.start:
			return self.outside_range[0](y)
		elif y >= self.end:
			return self.outside_range[1](y)

		if callable(self.pot):
			return self.pot(y)

		index = np.where(self.y == y)[0][0]
		return self.pot[index]

def T_ana(lambda_, a = 1):
	k = np.sqrt(lambda_)
	eta = np.sqrt(1 - lambda_)

	return (1 + ((k**2 + eta**2) / (2*k*eta))**2 * np.sinh(eta * a) ** 2) ** -1