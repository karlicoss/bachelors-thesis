import scipy.special

jn_zeros = lambda x, y: scipy.special.jn_zeros(int(x), int(y))
# jn = lambda x, y: scipy.special.jn(int(x), float(y))
def jn(x, y):
	# print(y)
	return scipy.special.jn(int(x), float(y))


class CylinderDeltaScattering:
	# energy: wave full energy
	# R : wire radius
	# m : magnetic quantum number
	# aa : delta well "strength"
	def __init__(self, energy, R, m, aa, maxn = 100):
		self.energy = energy
		self.R = R
		self.m = m
		self.zeros = jn_zeros(self.m, maxn)
		self.aa = aa

	def get_theta_function(self):
		def fun(theta):
			return CC(exp(I * self.m * theta) / (2 * pi))
		return fun

	# n >= 1
	def get_phi_energy(self, n):
		return (self.zeros[n - 1] / self.R)^2
	# n >= 1
	def get_phi_function(self, n):
		def fun(r):
			# print("point")
			return CC(sqrt(2.0) / (self.R * jn(abs(self.m) + 1, self.zeros[n - 1])) * jn(self.m, self.zeros[n - 1] * r / self.R))
		return fun

	# scattering from the left, incident amplitude is 1.0
	def get_varphi_function(self, n):
		# en = self.energy - self.get_phi_energy(n)
		# assert(en > 0)
		# def fun(z):
		# 	return CC(exp(I * sqrt(en) * z))
		# return fun
		en = self.energy - self.get_phi_energy(n)
		k = sqrt(abs(en))
		if en > 0:			
			RR = -(self.aa / (self.aa + 2 * I * k)) # reflection amplitude
			TT = 1 - self.aa / (self.aa + 2 * I * k)# transmisstion amplitude
			def fun(z):
				# print("point2")
				if z < 0:
					return CC(exp(I * k * z) + RR * exp(-I * k * z))
				else:
					return CC(TT * exp(I * k * z))
			return fun
		else:
			def fun(z):
				# print("point2")
				if z < 0:
					return exp(k * z)
				else:
					return exp(-k * z)
			return fun
			# raise ValueError("energy should be greater than zero")

def alala():
	energy = 0.232
	m = 0
	n = 1
	R = 5.0
	d = 10.0
	aa = 1.0

	cs = CylinderDeltaScattering(energy, R, m, aa)
	print(cs.get_phi_energy(n))
	theta = cs.get_theta_function()
	phi = cs.get_phi_function(n)
	varphi = cs.get_varphi_function(n)
	wf = lambda z, r: phi(r) * varphi(z)
	rwf = lambda z, r: wf(z, r).real()
	iwf = lambda z, r: wf(z, r).imag()
	pf = lambda z, r: wf(z, r).norm()
	# wf = lambda t, r, z: theta(t) * phi(r) * varphi(z)
	# rwf = lambda t, r, z: wf(t, r, z).real()
	# pf = lambda t, r, z: wf(t, r, z).norm()
	# t, r, z = var('r t z')
	# T = Cylindrical('height', ['radius', 'azimuth'])

	p = density_plot(pf, (-d, d), (0.0, R), cmap = 'jet', axes_labels = ['d', 'r'], aspect_ratio = 1, plot_points = 100)
	p.save("plot.png")
	p.show()

	# f(t, r)
	# def to_cartesian(f):
		# def fun(x, y):
			# return f(atan2(y, x), sqrt(x^2 + y^2))
		# return fun

	# f = to_cartesian(lambda t, r: rwf(t, r, 0.0))

	# T.transform(radius = r, azimuth = t, height = z)
	# print(rwf(0.0, 0.0, 0.0))
	# plot3d(lambda t, r: pf(t, r, 0.0), (-pi, pi), (0.01, R)).show()
	# plot(varphi, 0.0, 1.0).show()
	# print(pf(0.0, 0.0))
	# print(pf(0.0, d))
	# plot3d(rwf, (0.0, R), (0.0, d), axes_labels = ['r', 'z']).show(viewer = 'jmol')

alala()

# sphere().show()

input()