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

	# phis are orthonormalized w.r.t. to weight function r
	def get_phi_function(self, n):
		def fun(r):
			return CC(sqrt(2.0) / (self.RR * jn(abs(self.m) + 1, self.jzeros[n - 1])) * jn(self.m, self.jzeros[n - 1] * r / self.RR))
		return fun

	def get_wavefunction(self, n, energy):
		kks = [sqrt(CC(energy - phiE)) for phiE in self.phi_energies]
		integrals = [[numerical_integral(lambda r: r * self.phis[i](r) * self.phis[j](r), 0, self.R)[0] for j in range(self.maxn)] for i in range(self.maxn)]
		raise NotImplementedError

		A = np.zeros((self.maxn, self.maxn), dtype = np.complex64)
		B = np.zeros(self.maxn, dtype = np.complex64)
		for i in range(1, self.maxn + 1):
			A[i - 1, i - 1] += 2 * kks[i - 1]
			for j in range(1, self.maxn + 1):
				A[i - 1, j - 1] -= self.uu / I * integrals[i - 1][j - 1]
			B[i - 1] = uu / I * integrals[n - 1][i - 1]

		Rs = linalg.solve(A, B)
		Ts = np.zeros(maxn, dtype = np.complex64)

		for i in range(1, self.maxn + 1):
			if i == n:
				Ts[i - 1] = Rs[i - 1] + 1
			else:
				Ts[i - 1] = Rs[i - 1]

		print("Rs:")
		print(Rs)
		print("Ts:")
		print(Ts)

		## calculation of total transmission coefficient
		T = 0.0
		for i, t in enumerate(Ts):
			if self.phi_energies[i] < energy: # open channel
				T += (kks[i] / kks[n] * t * t.conj()).real
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

		return psi

def test():
	R = 1.0
	RR = 2.0
	m = 0
	d = 5.0
	maxn = 5
	uu = -0.4
	energy = 8.0
	k = sqrt(energy)

	zeros = jn_zeros(m, maxn)
	def get_phi_function(n):
		def fun(r):
			return CC(sqrt(2.0) / (RR * jn(abs(m) + 1, zeros[n - 1])) * jn(m, zeros[n - 1] * r / RR))
		return fun

	phis = [get_phi_function(n) for n in range(1, maxn + 1)]
	phi_energies = [(zeros[n - 1] / RR)^2 for n in range(1, maxn + 1)]
	print("Transversal modes energies:")
	print(phi_energies)

	kks = [sqrt(CC(energy - phiE)) for phiE in phi_energies]
	print("kks:")
	print(kks)

	# mode1 = get_phi_function(1)
	# print("Modes 1 and 2 inner product:")
	# print(numerical_integral(lambda r: r * mode1(r) * mode1(r), 0.0, RR))[0]
	# mode2 = get_phi_function(2)
	# print("Modes 1 and 2 inner product:")
	# print(numerical_integral(lambda r: r * mode1(r) * mode2(r), 0.0, RR))[0]

	integrals = [[numerical_integral(lambda r: r * phis[i](r) * phis[j](r), 0, R)[0] for j in range(maxn)] for i in range(maxn)]

	# print("Integrals:")
	# pprint(integrals)

	n = 1
	A = np.zeros((maxn, maxn), dtype = np.complex64)
	B = np.zeros(maxn, dtype = np.complex64)
	for i in range(1, maxn + 1):
		A[i - 1, i - 1] += 2 * kks[i - 1]
		for j in range(1, maxn + 1):
			A[i - 1, j - 1] -= uu / I * integrals[i - 1][j - 1]
		B[i - 1] = uu / I * integrals[n - 1][i - 1]
		# if i == n:
		# 	A[i - 1, i - 1] += 2 * kks[n - 1]
		# 	for j in range(1, maxn + 1):
		# 		A[i - 1, j - 1] -= uu / I * integrals[j - 1][n - 1]
		# 	B[i - 1] = - uu / I * integrals[n - 1][n - 1]
		# else:
		# 	for j in range(1, maxn + 1):
		# 		A[i - 1][j - 1] += integrals[j - 1][i - 1]
		# 	B[i - 1] = integrals[n - 1][n - 1]

	Rs = linalg.solve(A, B)
	Ts = np.zeros(maxn, dtype = np.complex64)

	for i in range(1, maxn + 1):
		if i == n:
			Ts[i - 1] = Rs[i - 1] + 1
		else:
			Ts[i - 1] = Rs[i - 1]

	print("Rs:")
	print(Rs)
	print("Ts:")
	print(Ts)

	## calculation of total transmission coefficient
	T = 0.0
	for i, t in enumerate(Ts):
		if phi_energies[i] < energy:			
			T += (kks[i] / kks[n] * t * t.conj()).real
	print("Total transmission coefficient:")
	print(T)
	##


	def psi1(z, r):
		res = CC(0)
		res += CC(exp(I * kks[n - 1] * z) * phis[n - 1](r))
		for i in range(1, maxn + 1):
			res += CC(Rs[i - 1] * exp(-I * kks[i - 1] * z) * phis[i - 1](r))
		return res

	def psi2(z, r):
		res = CC(0)
		for i in range(1, maxn + 1):
			res += CC(Ts[i - 1] * exp(I * kks[i - 1] * z) * phis[i - 1](r))
		return res

	def psi(z, r):
		if z < 0:
			return psi1(z, r)
		else:
			return psi2(z, r)

	pf = lambda z, r: psi(z, r).norm()

	print(psi1(0, R / 2))
	print(psi2(0, R / 2))
	print(psi1(0, R))
	print(psi2(0, R))
	print(psi2(d, R))

	# plot(lambda r: pf(0, r), (0.0, RR)).show()
	# plot3d(pf, (-d, d), (0.0, RR), axes_labels = ['z', 'r'], plot_points = 100).show(viewer = 'jmol')

	p = density_plot(pf, (-d, d), (0.0, RR), cmap = 'jet', axes_labels = ['z', 'r'], aspect_ratio = 3, plot_points = 200)
	p.save("plot.png")
	p.show()
	input()


test()
