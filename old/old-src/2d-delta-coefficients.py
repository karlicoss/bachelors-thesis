# This file was *autogenerated* from the file 2d-delta-coefficients.sage.
from sage.all_cmdline import *   # import sage library
_sage_const_300 = Integer(300); _sage_const_2 = Integer(2); _sage_const_1 = Integer(1); _sage_const_2p0 = RealNumber('2.0'); _sage_const_20p0 = RealNumber('20.0'); _sage_const_5 = Integer(5); _sage_const_1p0 = RealNumber('1.0'); _sage_const_5p0 = RealNumber('5.0'); _sage_const_0p65 = RealNumber('0.65'); _sage_const_0p1 = RealNumber('0.1'); _sage_const_0p0 = RealNumber('0.0'); _sage_const_0 = Integer(0); _sage_const_0p19 = RealNumber('0.19'); _sage_const_0p4 = RealNumber('0.4'); _sage_const_20 = Integer(20); _sage_const_4p0 = RealNumber('4.0'); _sage_const_3p0 = RealNumber('3.0'); _sage_const_10p0 = RealNumber('10.0')
from collections import namedtuple
from pprint import pprint

import scipy.constants as sc
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


# TODO
# hbar = 1.0
hbar = sc.hbar


# df: [0, RR] -> {0, 1}, returns if there is a delta at point (0, r)
# intf: f1, f2 -> float
class PiecewiseDeltaCylinderScattering:
	def __init__(self, mu, RR, uu, intf, m, maxn):
		self.mu = mu

		self.RR = RR
		self.uu = uu

		self.intf = intf

		self.m = m
		self.maxn = maxn

		self.jzeros = jn_zeros(m, maxn)

		## !!! n scope interference
		self.phis = [self.get_phi_function(n) for n in range(_sage_const_1 , self.maxn + _sage_const_1 )]
		self.phi_energies = [hbar ** _sage_const_2  / (_sage_const_2  * self.mu) * (self.jzeros[n - _sage_const_1 ] / self.RR)**_sage_const_2  for n in range(_sage_const_1 , self.maxn + _sage_const_1 )]

	# phis are orthonormalized w.r.t. to weight function r
	def get_phi_function(self, n):
		coeff = sqrt(_sage_const_2p0 ) / (self.RR * jn(abs(self.m) + _sage_const_1 , self.jzeros[n - _sage_const_1 ]))
		print(type(coeff))
		def fun(r):
			return CC(coeff * jn(self.m, self.jzeros[n - _sage_const_1 ] * r / self.RR))
		return fun

	def compute_scattering_full(self, energy):
		res = []
		for i, phiE in enumerate(self.phi_energies):
			if phiE < energy:
				res.append(self.compute_scattering(i + _sage_const_1 , energy))
		T = _sage_const_0p0 
		for r in res:
			T += r.T

		print("Energy = {}, T = {}".format(energy, T))
		return T

	def compute_scattering(self, n, energy, verbose = False):
		if verbose:
			print("-------------------")
			print("Energy: {}".format(energy / sc.eV)) # TODO eV
		kks = [sqrt(CC(_sage_const_2  * self.mu * (energy - phiE))) / hbar for phiE in self.phi_energies]

		if verbose:
			print("Wavevectors:")
			print(kks)

		integrals = [[None for j in range(self.maxn)] for i in range(self.maxn)]
		for i in range(self.maxn):
			for j in range(i, self.maxn):
				integrals[i][j] = self.intf(self.phis[i], self.phis[j])
				integrals[j][i] = integrals[i][j]

		if verbose:
			print("Integrals:")
			pprint(integrals)

		A = np.zeros((self.maxn, self.maxn), dtype = np.complex64)
		B = np.zeros(self.maxn, dtype = np.complex64)
		for i in range(_sage_const_1 , self.maxn + _sage_const_1 ):
			A[i - _sage_const_1 , i - _sage_const_1 ] += _sage_const_2  * kks[i - _sage_const_1 ]
			for j in range(_sage_const_1 , self.maxn + _sage_const_1 ):
				A[i - _sage_const_1 , j - _sage_const_1 ] -= _sage_const_2  * self.mu / hbar ** _sage_const_2  * self.uu / I * integrals[i - _sage_const_1 ][j - _sage_const_1 ]
			B[i - _sage_const_1 ] = _sage_const_2  * self.mu / hbar ** _sage_const_2  * self.uu / I * integrals[n - _sage_const_1 ][i - _sage_const_1 ]

		Rs = linalg.solve(A, B)
		Ts = np.zeros(self.maxn, dtype = np.complex64)

		for i in range(_sage_const_1 , self.maxn + _sage_const_1 ):
			if i == n:
				Ts[i - _sage_const_1 ] = Rs[i - _sage_const_1 ] + _sage_const_1 
			else:
				Ts[i - _sage_const_1 ] = Rs[i - _sage_const_1 ]

		if verbose:
			print("Rs:")
			print(Rs)
			print("Ts:")
			print(Ts)

		## calculation of total transmission coefficient
		T = _sage_const_0p0 
		for i, t in enumerate(Ts):
			if self.phi_energies[i] < energy: # open channel
				T += (kks[i] / kks[n - _sage_const_1 ] * t * t.conj()).real
		if verbose:
			print("Total transmission coefficient:")
			print(T)
		##


		def psi1(z, r):
			res = CC(_sage_const_0 )
			res += CC(exp(I * kks[n - _sage_const_1 ] * z) * self.phis[n - _sage_const_1 ](r))
			for i in range(_sage_const_1 , self.maxn + _sage_const_1 ):
				res += CC(Rs[i - _sage_const_1 ] * exp(-I * kks[i - _sage_const_1 ] * z) * self.phis[i - _sage_const_1 ](r))
			return res

		def psi2(z, r):
			res = CC(_sage_const_0 )
			for i in range(_sage_const_1 , self.maxn + _sage_const_1 ):
				res += CC(Ts[i - _sage_const_1 ] * exp(I * kks[i - _sage_const_1 ] * z) * self.phis[i - _sage_const_1 ](r))
			return res

		def psi(z, r):
			if z < _sage_const_0 :
				return psi1(z, r)
			else:
				return psi2(z, r)
		if verbose:
			print("-------------------")
		return ScatteringResult(wf = psi, T = T)

def test_cylinder():
	R = _sage_const_1p0  * sc.nano
	RR = _sage_const_5p0  * sc.nano
	uu = -_sage_const_0p4  * sc.nano * sc.eV
	m = _sage_const_0 
	mu = _sage_const_0p19  * sc.m_e # mass
	maxn = _sage_const_2 

	intf = lambda f, g: numerical_integral(lambda r: r * f(r) * g(r), _sage_const_0 , R)[_sage_const_0 ]
	dcs = PiecewiseDeltaCylinderScattering(mu, RR, uu, intf, m, maxn)

	print("Transversal mode energies:")
	print([en / sc.eV for en in dcs.phi_energies])

	n = _sage_const_1 
	# energy = dcs.phi_energies[1] - 1e-17 * sc.eV
	# energy = 0.24 * sc.eV
	energy = _sage_const_1p0  * sc.eV
	
	for i in range(_sage_const_20 ):
		res = dcs.compute_scattering(n, energy, verbose = False)
	T = res.T
	print(T)
	wf = res.wf
	pf = lambda z, r: wf(z, r).norm()

	d = _sage_const_20p0  * sc.nano

	print(wf(_sage_const_0 , R / _sage_const_2 ))
	print(wf(_sage_const_0 , R / _sage_const_2 ))
	print(wf(_sage_const_0 , R))
	print(wf(_sage_const_0 , R))

	ll = _sage_const_0p0 
	rr = _sage_const_1p0 
	step = _sage_const_0p1 
	left = ll * sc.eV
	right = rr * sc.eV
	# plot(lambda en: dcs.compute_scattering_full(en),
	# 	(left, right),
	# 	plot_points = 300,
	# 	axes_labels = ['E, eV', 'T'],
	# 	ticks = [[a for a in arange(left, right, step * sc.eV)], [a for a in arange(0.0, 5.0, 1.0)]],
	# 	tick_formatter = [["{:.1f} eV".format(a) for a in arange(ll, rr, step)], ["{:.1f}".format(a) for a in arange(0.0, 5.0, 1.0)]]).show()

	# plot(lambda r: pf(0, r), (0.0, RR)).show()
	# plot3d(pf, (-d, d), (0.0, RR), axes_labels = ['z', 'r'], plot_points = 200).show(viewer = 'jmol')

	# p = density_plot(pf, (-d, d), (0.0, RR), cmap = 'jet', axes_labels = ['z', 'r'], aspect_ratio = 3, plot_points = 200)
	# p.save("plot.png")
	# p.show()
	# input()

def test_slit():
	R = _sage_const_0p1 
	RR = _sage_const_3p0 
	uu = _sage_const_1p0 
	mu = _sage_const_1p0 
	m = _sage_const_0 
	maxn = _sage_const_5 


	intf = lambda f, g: numerical_integral(lambda r: r * f(r) * g(r), R, RR)[_sage_const_0 ]
	dcs = PiecewiseDeltaCylinderScattering(mu, RR, uu, intf, m, maxn)

	n = _sage_const_1 
	energy = _sage_const_2p0 
	
	res = dcs.compute_scattering(n, energy)
	T = res.T
	wf = res.wf
	pf = lambda z, r: wf(z, r).norm()

	d = _sage_const_10p0 

	print(wf(_sage_const_0 , R / _sage_const_2 ))
	print(wf(_sage_const_0 , R / _sage_const_2 ))
	print(wf(_sage_const_0 , R))
	print(wf(_sage_const_0 , R))

	plot(lambda en: dcs.compute_scattering_full(en).T, (_sage_const_0p65 , _sage_const_4p0 ), plot_points = _sage_const_300 ).show()

	# plot(lambda r: pf(0, r), (0.0, RR)).show()
	# plot3d(pf, (-d, d), (0.0, RR), axes_labels = ['z', 'r'], plot_points = 200).show(viewer = 'jmol')

	# p = density_plot(pf, (-d, d), (0.0, RR), cmap = 'jet', axes_labels = ['z', 'r'], aspect_ratio = 3, plot_points = 200)
	p.save("plot.png")
	p.show()
	input()

test_cylinder()
# test_slit()
