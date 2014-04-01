from collections import namedtuple
from pprint import pprint

import scipy.special
import numpy as np
from numpy import arange, eye, linalg

jn_zeros = lambda x, y: scipy.special.jn_zeros(int(x), int(y))
jn = lambda x, y: scipy.special.jn(int(x), float(y))


# TODO seems that the default branch is ok
# def my_sqrt(x):
# 	if x > 0:
# 		return sqrt(CC(x))
# 	else:
# 		return -sqrt(CC(x))


ScatteringResult = namedtuple('ScatteringResult', ['wf', 'T'])

class DeltaCylinderScattering:
	def __init__(self, R, RR, uu, m, maxn):
		self.R = R
		self.RR = RR
		self.uu = uu
		self.m = m
		self.maxn = maxn

		self.jzeros = jn_zeros(m, maxn)

		## !!! n scope interference
		self.phis = [self.get_phi_function(n) for n in range(1, self.maxn + 1)]
		self.phi_energies = [(self.jzeros[n - 1] / self.RR)^2 for n in range(1, self.maxn + 1)]

		print("Transversal mode energies:")
		print(self.phi_energies)

	# phis are orthonormalized w.r.t. to weight function r
	def get_phi_function(self, n):
		def fun(r):
			return CC(sqrt(2.0) / (self.RR * jn(abs(self.m) + 1, self.jzeros[n - 1])) * jn(self.m, self.jzeros[n - 1] * r / self.RR))
		return fun

	def compute_scattering_full(self, energy):
		res = []
		for i, phiE in enumerate(self.phi_energies):
			if phiE < energy:
				res.append(self.compute_scattering(i + 1, energy))
		T = 0.0
		for r in res:
			T += r.T

		print("Energy = {}, T = {}".format(energy, T))
		return T

	def compute_scattering(self, n, energy, verbose = False):
		if verbose:
			print("-------------------")
			print("Energy: {}".format(energy))
		kks = [sqrt(CC(energy - phiE)) for phiE in self.phi_energies]
		integrals = [[numerical_integral(lambda r: r * self.phis[i](r) * self.phis[j](r), 0, self.R)[0] for j in range(self.maxn)] for i in range(self.maxn)]

		A = np.zeros((self.maxn, self.maxn), dtype = np.complex64)
		B = np.zeros(self.maxn, dtype = np.complex64)
		for i in range(1, self.maxn + 1):
			A[i - 1, i - 1] += 2 * kks[i - 1]
			for j in range(1, self.maxn + 1):
				A[i - 1, j - 1] -= self.uu / I * integrals[i - 1][j - 1]
			B[i - 1] = self.uu / I * integrals[n - 1][i - 1]

		Rs = linalg.solve(A, B)
		Ts = np.zeros(self.maxn, dtype = np.complex64)

		for i in range(1, self.maxn + 1):
			if i == n:
				Ts[i - 1] = Rs[i - 1] + 1
			else:
				Ts[i - 1] = Rs[i - 1]

		if verbose:
			print("Rs:")
			print(Rs)
			print("Ts:")
			print(Ts)

		## calculation of total transmission coefficient
		T = 0.0
		for i, t in enumerate(Ts):
			if self.phi_energies[i] < energy: # open channel
				T += (kks[i] / kks[n - 1] * t * t.conj()).real
		if verbose:
			print("Total transmission coefficient:")
			print(T)
		##


		def psi1(z, r):
			res = CC(0)
			res += CC(exp(I * kks[n - 1] * z) * self.phis[n - 1](r))
			for i in range(1, self.maxn + 1):
				res += CC(Rs[i - 1] * exp(-I * kks[i - 1] * z) * self.phis[i - 1](r))
			return res

		def psi2(z, r):
			res = CC(0)
			for i in range(1, self.maxn + 1):
				res += CC(Ts[i - 1] * exp(I * kks[i - 1] * z) * self.phis[i - 1](r))
			return res

		def psi(z, r):
			if z < 0:
				return psi1(z, r)
			else:
				return psi2(z, r)
		if verbose:
			print("-------------------")
		return ScatteringResult(wf = psi, T = T)

def test():
	R = 1.0
	RR = 5.0
	uu = -0.4
	m = 0
	maxn = 5

	dcs = DeltaCylinderScattering(R, RR, uu, m, maxn)

	n = 1
	energy = 1.5
	
	res = dcs.compute_scattering(n, energy)
	T = res.T
	wf = res.wf
	pf = lambda z, r: wf(z, r).norm()

	d = 20.0

	print(wf(0, R / 2))
	print(wf(0, R / 2))
	print(wf(0, R))
	print(wf(0, R))

	plot(lambda en: dcs.compute_scattering_full(en).T, (0.5, 9.0), plot_points = 100).show()

	# plot(lambda r: pf(0, r), (0.0, RR)).show()
	# plot3d(pf, (-d, d), (0.0, RR), axes_labels = ['z', 'r'], plot_points = 200).show(viewer = 'jmol')

	# p = density_plot(pf, (-d, d), (0.0, RR), cmap = 'jet', axes_labels = ['z', 'r'], aspect_ratio = 3, plot_points = 200)
	# p.save("plot.png")
	# p.show()
	input()


test()
