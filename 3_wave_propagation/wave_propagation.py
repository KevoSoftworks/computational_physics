from functools import cached_property

import numpy as np

class Matrix:
	def __init__(self, potential, lambda_):
		self.potential = potential
		self.lambda_ = lambda_
	
	def eta(self, y):
		return np.sqrt(self.potential[y + self.potential.step / 2.] - self.lambda_ + 0j)

	def transmission(self, dir="right"):
		raise NotImplementedError(f"Transmission not implemented for type {type(self)}")

	def reflection(self, dir="right"):
		raise NotImplementedError(f"Transmission not implemented for type {type(self)}")

class TransferMatrix(Matrix):
	@cached_property
	def M(self):
		p = lambda prev, cur: 1/(2*prev) * np.array([ \
			[prev + cur, prev - cur], \
			[prev - cur, prev + cur] \
		])

		q = lambda cur: np.array([ \
			[np.exp(cur * self.potential.step), 0], \
			[0, np.exp(-cur * self.potential.step)] \
		])

		product = np.array([[1, 0], [0, 1]])

		for y in self.potential.y[1:-1]:
			eta_prev = self.eta(y - self.potential.step)
			eta_cur = self.eta(y)

			product = product @ p(eta_prev, eta_cur) @ q(eta_cur)

		eta_prev = self.eta(self.potential.end - self.potential.step)
		eta_cur = self.eta(self.potential.end)
				
		return np.array([ \
			[np.exp(-self.eta(self.potential.start) * (self.potential.start + self.potential.step)), 0], \
			[0, np.exp(self.eta(self.potential.start) * (self.potential.start + self.potential.step))] \
		]) @ product \
		@ p(eta_prev, eta_cur) @ np.array([ \
			[np.exp(self.eta(self.potential.end) * self.potential.end), 0], \
			[0, np.exp(-self.eta(self.potential.end) * self.potential.end)] \
		])
	
	def flush(self):
		del self.M
	
	def transmission(self, dir="right"):
		if dir == "right":
			return 1/(np.abs(self.M[0, 0])**2)
		
		return np.abs(self.M[1, 1] - (self.M[1, 0] * self.M[0, 1]) / self.M[0, 0]) ** 2
	
	def reflection(self, *args):
		return np.abs(self.M[1, 0])**2 / np.abs(self.M[0, 0])**2

class Potential:
	def __init__(self, start, end, len, expr = None, outside_range = (lambda x: 0, lambda x: 0)):
		self.start = start
		self.end = end
		self.len = len

		self.pot = np.zeros(self.len)
		self.y = np.linspace(self.start, self.end, self.len)

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
		elif y > self.end:
			return self.outside_range[1](y)

		if callable(self.pot):
			return self.pot(y)

		index = np.where(self.y == y)[0][0]
		return self.pot[index]

	@property
	def step(self):
		return self.y[1] - self.y[0]	#Hacky, but it works

def T_ana(lambda_, a = 1):
	k = np.sqrt(lambda_)
	eta = np.sqrt(1 - lambda_)

	return (1 + ((k**2 + eta**2) / (2*k*eta))**2 * np.sinh(eta * a) ** 2) ** -1