import copy

import numpy as np

class Lattice:
	def __init__(self, L, B, beta = 1):
		self.L = L
		self.B = B
		self.beta = beta

		self.lattice = np.random.randint(2, size=(L, L))
		self.lattice = self.lattice * 2 - 1

		self.E = -self.B * self.M

	@property
	def N(self):
		return self.L**2

	@property
	def Z(self):
		return np.sum(np.exp(-self.beta * self.E))
	
	@property
	def M(self):
		return np.sum(self.lattice)

	# This function should be called from the "new" lattice S'
	def delta_e(self, lattice):
		return self.B * (lattice.M - self.M)

	def flip(self, x = -1, y = -1):
		if x == -1 or y == -1:
			coord = np.floor(Probability.random(2)*self.L).astype(int)
		else:
			coord = (x, y)

		self.lattice[coord[0]][coord[1]] *= -1

		return self

class Probability:
	ENTROPY_LEN = 1000
	ENTROPY_INDEX = ENTROPY_LEN + 1
	ENTROPY = None

	@staticmethod
	def random(count):
		if Probability.ENTROPY_INDEX + count >= Probability.ENTROPY_LEN \
		or Probability.ENTROPY is None:
			Probability.ENTROPY = np.random.rand(Probability.ENTROPY_LEN)
			Probability.ENTROPY_INDEX = 0
		
		gen = Probability.ENTROPY[Probability.ENTROPY_INDEX:Probability.ENTROPY_INDEX+count]
		Probability.ENTROPY_INDEX += count

		return gen

	@staticmethod
	def get(dE, beta = 1):
		return min(1, np.exp(-beta * dE))

	@staticmethod
	def accept(dE, beta = 1):
		r = np.random.rand()
		return r < Probability.get(dE, beta)


class Solver:
	def __init__(self, lattice):
		self.lattice = lattice

	def iterate(self, iterations = -1):
		i = 0
		while i < iterations or iterations == -1:
			tmp = copy.deepcopy(self.lattice).flip()
			dE = tmp.delta_e(self.lattice)

			if Probability.accept(dE, self.lattice.beta):
				self.lattice = tmp
				self.lattice.E += dE
				#self.lattice.M is a function and needn't be updated
			
			i += 1

			yield self.lattice
	
	# Solve using large steps L**2
	def fullsolve(self, iterations, ret_val = ("<m>")):
		gen = self.iterate()
		M = []
		ret = {}

		for _ in range(iterations):
			for _ in range(self.lattice.N):
				lat = next(gen)
			
			M.append(lat.M)
		
		M = np.array(M)
		
		if "<m>" in ret_val:
			ret["<m>"] = 1/iterations * np.sum(M) / self.lattice.N

		if "M" in ret_val:
			ret["M"] = M

		if "<M>" in ret_val:
			ret["<M>"] = 1/iterations * np.sum(M)

		if "chi" in ret_val:
			exp_M = 1/iterations * np.sum(M)
			exp_M2 = 1/iterations * np.sum(M**2)

			ret["chi"] = self.lattice.beta * (exp_M2 - exp_M**2)

		return ret