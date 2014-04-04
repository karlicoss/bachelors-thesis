from collections import namedtuple
from pprint import pprint

import numpy as np
from numpy import arange, eye, linalg
from numpy import complex_ as complex # TODO not sure if that's ok
from numpy import sqrt, exp

import pylab as pl
import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm, Normalize

import scipy as sp
import scipy.integrate
import scipy.constants as sc
import scipy.special


def cnorm(z):
	return (z * np.conj(z)).real

I = complex(1j) # TODO not sure if ok


jn_zeros = lambda x, y: scipy.special.jn_zeros(x, int(y))
jn = lambda x, y: scipy.special.jn(x, float(y))

ScatteringResult = namedtuple('ScatteringResult', ['wf', 'T'])


# TODO
# hbar = 1.0
hbar = sc.hbar


def integrate_complex(f, a, b, **kwargs):
	real_integrand = scipy.integrate.quad(lambda x: sp.real(f(x)), a, b, **kwargs)
	imag_integrand = scipy.integrate.quad(lambda x: sp.imag(f(x)), a, b, **kwargs)
	return (real_integrand[0] + I * imag_integrand[0], real_integrand[1] + I * imag_integrand[1])

# df: [0, RR] -> {0, 1}, returns if there is a delta at point (0, r)
# intf: f1, f2 -> float
# m : int
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
		self.phis = [self.get_phi_function(n) for n in range(1, self.maxn + 1)]
		self.phi_energies = [hbar ** 2 / (2 * self.mu) * (self.jzeros[n - 1] / self.RR) ** 2 for n in range(1, self.maxn + 1)]

	# phis are orthonormalized w.r.t. to weight function r
	# n: int
	def get_phi_function(self, n):
		coeff = sqrt(2.0) / (self.RR * jn(abs(self.m) + 1, self.jzeros[n - 1]))
		def fun(r):
			return complex(coeff * jn(self.m, self.jzeros[n - 1] * r / self.RR))
		return fun

	def compute_scattering_full(self, energy):
		res = []
		for i, phiE in enumerate(self.phi_energies):
			if phiE < energy:
				res.append(self.compute_scattering(i + 1, energy))
		T = 0.0
		for r in res:
			T += r.T

		print("Energy = {} eV, T = {}".format(energy / sc.eV, T))
		return T

	def compute_scattering(self, n, energy, verbose = False):
		assert(energy > self.phi_energies[n - 1])

		if verbose:
			print("-------------------")
			print("Energy: {} eV".format(energy / sc.eV)) # TODO eV
		kks = [sqrt(complex(2 * self.mu * (energy - phiE))) / hbar for phiE in self.phi_energies]

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
		for i in range(1, self.maxn + 1):
			A[i - 1, i - 1] += 2 * kks[i - 1]
			for j in range(1, self.maxn + 1):
				A[i - 1, j - 1] -= 2 * self.mu / hbar ** 2 * self.uu / I * integrals[i - 1][j - 1]
			B[i - 1] = 2 * self.mu / hbar ** 2 * self.uu / I * integrals[n - 1][i - 1]

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
				T += (kks[i] / kks[n - 1] * cnorm(t)).real
		if verbose:
			print("Total transmission coefficient:")
			print(T)
		##


		def psi1(z, r):
			res = complex(0.0)
			res += exp(I * kks[n - 1] * z) * self.phis[n - 1](r)
			for i in range(1, self.maxn + 1):
				res += Rs[i - 1] * exp(-I * kks[i - 1] * z) * self.phis[i - 1](r)
			return res

		def psi2(z, r):
			res = complex(0.0)
			for i in range(1, self.maxn + 1):
				res += Ts[i - 1] * exp(I * kks[i - 1] * z) * self.phis[i - 1](r)
			return res

		def psi(z, r):
			if z < 0:
				return psi1(z, r)
			else:
				return psi2(z, r)
		if verbose:
			print("-------------------")
		return ScatteringResult(wf = psi, T = T)


def plot_transmission(dcs, left, right, step):
	xs = arange(left, right, step)
	ys = list(map(lambda en: dcs.compute_scattering_full(en), xs))

	fig = plt.figure(figsize = (15, 10), dpi = 500)
	ax = fig.add_subplot(111) # , aspect = 'equal'
	
	ax.vlines(dcs.phi_energies, 0.0, dcs.maxn)

	xticks = arange(left, right, 0.1 * sc.eV)
	xlabels = ["{:.1f}".format(t / sc.eV) for t in xticks]
	ax.set_xlim(left, right)
	ax.set_xticks(xticks)
	ax.set_xticklabels(xlabels)
	ax.set_xlabel("E, eV")

	yticks = arange(0.0, dcs.maxn, 1.0)
	ylabels = ["{:.1f}".format(t) for t in yticks]
	ax.set_ylim(0.0, dcs.maxn)
	ax.set_yticks(yticks)
	ax.set_yticklabels(ylabels)
	ax.set_ylabel("T")

	cax = ax.plot(xs, ys)
	ax.set_title("Transmission coefficient, m = {}".format(dcs.m))

	fig.savefig("transmission.png")

def plot_wavefunction(dcs, n, energy, fname = "plot.png"):
	res = dcs.compute_scattering(n, energy, verbose = False)		

	T = res.T
	wf = res.wf
	pf = lambda z, r: cnorm(wf(z, r))
	vpf = np.vectorize(pf)

	print("Total transmission: {}".format(T))


	d = 40.0 * sc.nano

	dd = 0.05 * sc.nano
	dr = 0.05 * sc.nano
	x, y = np.mgrid[slice(-d, d + dd, dd), slice(0.0, dcs.RR + dr, dr)]
	z = vpf(x, y)

	z_min, z_max = np.abs(z).min(), np.abs(z).max()

	print(z_min)
	print(z_max)

	fig = plt.figure(figsize = (15, 10), dpi = 500)
	ax = fig.add_subplot(211) # , aspect = 'equal'

	ax.set_title("m = {}, n = {}, E = {:.2f} eV, T = {:.2f}".format(dcs.m, n, energy / sc.eV, T))

	xticks = arange(-d, d, 10 * sc.nano)
	xlabels = ["{:.1f}".format(t / sc.nano) for t in xticks]
	ax.set_xlim(-d, d)
	ax.set_xticks(xticks)
	ax.set_xticklabels(xlabels)

	yticks = arange(0.0, dcs.RR, sc.nano)
	ylabels = ["{:.1f}".format(t / sc.nano) for t in yticks]
	ax.set_ylim(0.0, dcs.RR)
	ax.set_yticks(yticks)
	ax.set_yticklabels(ylabels)

	cax = ax.pcolor(x, y, z, cmap = 'gnuplot', norm = Normalize(z_min, z_max))
	
	cbar = fig.colorbar(cax)

	bx = fig.add_subplot(212)
	bx.set_title("Wavefunction at z = 0")

	rr = arange(0.0, dcs.RR, 0.01 * sc.nano)
	ww = list(map(lambda r: pf(0, r), rr))
	bx.plot(rr, ww)

	fig.savefig(fname)

def test_cylinder(R, RR, uu, m, mu, maxn):
	intf = lambda f, g: integrate_complex(lambda r: r * f(r) * g(r), 0, R)[0]
	dcs = PiecewiseDeltaCylinderScattering(mu, RR, uu, intf, m, maxn)

	print("Transversal mode energies:")
	print([en / sc.eV for en in dcs.phi_energies])

	# energy = 0.2179999 * sc.eV
	energy = 0.244 * sc.eV
	n = 1
	plot_wavefunction(dcs, n, energy)


	# plot_transmission(dcs, 0.0 * sc.eV, 1.0 * sc.eV, 0.001 * sc.eV)

	#######
	# n = 1
	# for ee in arange(0.05, 1.0, 0.01):
	# 	print("Plotting for energy = {:.2f} eV".format(ee))
	# 	energy = ee * sc.eV
	# 	plot_wavefunction(n, energy, "{:.2f}".format(ee))
	# #######




def test_slit():
	R = 2.0 * sc.nano
	RR = 3.0 * sc.nano
	uu = -0.1 * sc.nano * sc.eV
	mu = 0.19 * sc.m_e
	m = 0
	maxn = 5


	intf = lambda f, g: integrate_complex(lambda r: r * f(r) * g(r), R, RR)[0]
	dcs = PiecewiseDeltaCylinderScattering(mu, RR, uu, intf, m, maxn)

	print("Transversal mode energies:")
	print([en / sc.eV for en in dcs.phi_energies])

	# plot_transmission(dcs, 0.0 * sc.eV, 1.0 * sc.eV, 0.001 * sc.eV)

	n = 1
	energy = 0.676999 * sc.eV
	plot_wavefunction(dcs, n, energy)


def main():
	R = 1.0 * sc.nano
	RR = 5.0 * sc.nano
	uu = -0.4 * sc.nano * sc.eV
	m = 0
	mu = 0.19 * sc.m_e # mass
	maxn = 5
	test_cylinder(R, RR, uu, m, mu, maxn)

main()