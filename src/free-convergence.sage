from numpy import arange
from scipy.optimize import newton, brentq, minimize_scalar
from sage.functions.other import psi1

### Some common functions ###

def test_orthonormality(a, wf1, wf2):
	return numerical_integral(lambda x: wf1(x) * wf2(x).conjugate(), 0.0, a)[0]

# normalizes the functions f on the interval (0, a)
# also makes functions pure real
# TODO convert to get_normalization
def normalize(f, a):
	return lambda x: A * f(x)
###

def get_normalization(f, a):
	return 1.0 / sqrt(numerical_integral(lambda x: f(x).norm(), 0.0, a)[0])

def make_real(f):
	def rotate(c):
		if c.arg() > 0:
			return c.abs()
		else:
			return -c.abs()


	return lambda x: rotate(f(x))

class FreeParticle:
	# r_A is R-matrix parameter A
	# r_B is R-matrix paramater B
	# n is the number of eigenstates to compute
	def __init__(self, r_A, r_B, n):
		self.r_A = r_A
		self.r_B = r_B
		self.n = n
		states, values = self.find_eigens()
		self.eigenstates = states
		self.eigenvalues = values

	# Solution of -d^2 psi/dx^2 = - k^2 psi(x)
	def get_wavefunction_negative(self, k):
		A = 1.0 / sqrt(sinh(2 * self.r_A * k) / k - 2 * self.r_A)
		wf = lambda x: CC(exp(- k * x) - exp(k * x)) # -2 * sinh(k * x)
		nwf = lambda x: A * wf(x)
		return make_real(nwf)

	# Solution of -d^2 psi/dx^2 = 0
	def get_wavefunction_zero(self, k):
		A = 1.0 / sqrt(self.r_A^3 / 3)
		wf = lambda x: CC(x)
		nwf = lambda x: A * wf(x)
		return make_real(nwf)

	# Solution of -d^2 pdi/dx^2 = k^2 psi(x)
	def get_wavefunction_positive(self, k):
		A = sqrt(1.0 / (4 * (self.r_A / 2 - sin(2 * k * self.r_A) / (4 * k)))) # -2 * I * sin(k * x)
		wf = lambda x: CC(exp(- I * k * x) - exp(I * k * x))
		nwf = lambda x: A * wf(x)
		return make_real(nwf)

	def find_eigens(self):
		states = []
		values = []

		b = self.r_B # TODO
		a = self.r_A # TODO

		eqn = lambda x: x * coth(x) - b
		eqp = lambda x: x * cot(x) - b

		if b > 1.0:
			root = brentq(eqn, 0.01, 10.0 * abs(b)) # TODO investigate right boundary
			root /= a
			wf = self.get_wavefunction_negative(root)
			states.append(wf)
			values.append(-root ** 2)
		elif b == 1.0:
			root = 0.0
			root /= a
			wf = self.get_wavefunction_zero(root)
			states.append(wf)
			values.append(0.0)
		else: # b < 1.0
			root = brentq(eqp, 0.01, pi - 0.01) # TODO 0.01
			root /= a
			wf = self.get_wavefunction_positive(root)
			states.append(wf)
			values.append(root ** 2)

		i = 1
		while len(states) < self.n:
			root = brentq(eqp, pi * i + 0.01, pi * (i + 1) - 0.01) # TODO 0.01
			root /= a
			wf = self.get_wavefunction_positive(root)
			states.append(wf)
			values.append(root ** 2)
			i += 1
		return states, values

# assuming f is decreasing with x increasing and f(x0) > 0
def find_right_interval(f, x0, alpha):
	x = x0
	while True: 
		fx = f(x)
		if fx < 0:
			return x
		while f(x + alpha) > fx:
			alpha /= 2
		x += alpha

def find_left_interval(f, x0, alpha):
	x = x0
	while True:
		# print("x = {}".format(x))
		fx = f(x)
		if fx > 0:
			return x
		while f(x - alpha) < fx:
			alpha /= 2
		x -= alpha


# V(x) = -1.0 delta(x - 1.0)
class DeltaPotential:
	def __init__(self, r_A, r_B, n):
		self.r_A = r_A
		self.r_B = r_B
		self.n = n
		self.dd = 1.0
		self.aa = 1.0
		states, values = self.find_eigens()
		self.eigenstates = states
		self.eigenvalues = values

	def get_wavefunction_negative(self, k):
		B = (0.5 - 0.5 * exp(2 * k) + k) / k
		C = -1.0 + (0.5 - 0.5 * exp(-2 * k)) / k
		def wf(x):
			if x < self.dd:
				return CC(exp(-k * x) - exp(k * x))
			else:
				return CC(B * exp(-k * x) + C * exp(k * x))
		A = get_normalization(wf, self.r_A)
		nwf = lambda x: A * wf(x)
		return make_real(nwf)

	def get_wavefunction_zero(self, k):
		B = 0.0
		C = 1.0
		def wf(x):
			if x < self.dd:
				return CC(x)
			else:
				return CC(B * x + C)
		A = get_normalization(wf, self.r_A)
		nwf = lambda x: A * wf(x)
		return make_real(nwf)

	def get_wavefunction_positive(self, k):
		B = (I * (-1 + exp(2 * I * k) - 2 * I * k)) / (2 * k)
		C = -((exp(-2 * I * k) * (-I + I * exp(2 * I * k) + 2 * exp(2 * I * k) * k)) / (2 * k))
		def wf(x):
			if x < self.dd:
				return CC(exp(-I * k * x) - exp(I * k * x))
			else:
				return CC(B * exp(-I * k * x) + C * exp(I * k * x))
		A = get_normalization(wf, self.r_A)
		nwf = lambda x: A * wf(x)
		return make_real(nwf)

	# TODO Assumes a = 2.0!!!!
	def find_eigens(self):
		states = []
		values = []

		eqn = lambda k: (2 * k * (k + coth(k) * (-1 + k * coth(k)))) / (-1 + 2 * k * coth(k)) - self.r_B
		eqp = lambda k: (2 * k * (-k + cot(k) * (-1 + k * cot(k)))) / (-1 + 2 * k * cot(k)) - self.r_B


		if self.r_B > 0.0:
			root = newton(eqn, 0.01) # TODO investigate constant
			wf = self.get_wavefunction_negative(root)
			states.append(root)
			values.append(-root ** 2)
		elif self.r_B == 0.0:
			root = 0.0
			wf = self.get_wavefunction_zero(root)
			states.append(wf)
			values.append(0.0)
		else: # self.r_B < 0.0
			# root = newton(eqp, 0.01) # TODO constant
			root = brentq(eqp, 0.01, find_right_interval(eqp, 0.01, 0.01))
			wf = self.get_wavefunction_positive(root)
			states.append(wf)
			values.append(root ** 2)

		i = 1
		while len(states) < self.n:
			root = None
			if i % 2 == 1:
				# root = newton(eqp, (i + 1) / 2 * pi - 0.1) # TODO 0.01
				right = (i + 1) / 2 * pi - 0.01
				root = brentq(eqp, find_left_interval(eqp, right, 0.01), right)
			else:
				left = i / 2 * pi + 0.01
				root = brentq(eqp, left, find_right_interval(eqp, left, 0.01))
			print("Found root: {} : {}".format(root, eqp(root)))
			wf = self.get_wavefunction_positive(root)
			states.append(wf)
			values.append(root ** 2)
			i += 1
		return states, values


# u'(a) / u(a) = X, where u(a) = e^{-ikx} + U e^{ikx}
def get_U(p, energy, X):
	k = sqrt(energy)
	U = exp(-2 * I * k * p.r_A) * (I * k + X) / (I * k - X)
	return CC(U)


def compute_phase_shift(p, energy, debug = False):
	def get_R(p, energy):
		s = 0.0
		# print(energies)
		for es, ee in zip(p.eigenstates, p.eigenvalues):
			s += es(p.r_A) ** 2 / (ee - energy) / p.r_A
			# print("i = {}, R = {}".format(i, s))
		return s

	R = get_R(p, energy)
	# Rub = get_R_upper_bound(a, b, energy, n)
	X = (1 + R * p.r_B) / (R * p.r_A)
	U = get_U(p, energy, X)
	if debug:
		print("a = {}, b = {}, n = {}, R = {}, U = {}, phase shift = {}".format(p.r_A, p.r_B, p.n, R, U, U.arg()))
	return U.arg()

def plot_phase_shift(p, left, right):
	p = plot(lambda energy: compute_phase_shift(p, energy), left, right)
	p.save("plot-delta.png")


def compute_R_upper_bound(p, energy):
	k = sqrt(energy)
	s = 0.0
	rest0 = 0.0
	n0 = None
	for i, (es, ee) in enumerate(zip(p.eigenstates, p.eigenvalues)):
		val = float(es(p.r_A) ** 2 / (ee - energy) / p.r_A)
		if (pi / 2 + pi * i) / p.r_A  < k:
			s += val
		else:
			if n0 is None:
				n0 = i
			rest0 += val


	X = p.r_A * k / pi
	rest = float(1 / pi ** 2 * 1 / X * (psi1(n0 + 0.5 + X) - psi1(n0 + 0.5 - X)))
	print("partial sum = {}".format(s))
	print("rest0 = {}".format(rest0))
	print("rest = {}".format(rest))



# TODO refactor
def get_partial_shifts(a, b, energy, count):
	shifts = []
	s = 0.0
	eigens = find_eigens(a, b, count)
	for cnt, (elambda, wflambda) in enumerate(eigens):
		s += wflambda(a) * wflambda(a) / (elambda - energy) / a
		X = (1 + s * b) / (s * a)
		U = get_U(a, energy, X)
		shifts.append((cnt + 1, U.argument()))
	return shifts


# a = 10.0
# b = -2.0 # 2.0
# n = 50
# energy = 10.0

# a = 2.5

# p = FreeParticle(a, b, n)
# print(compute_phase_shift(p, energy, debug = True))

# p = FreeParticle(a, b, n)

# p = DeltaPotential(a, b, n)

# compute_phase_shift(p, energy, debug = True)
# compute_R_upper_bound(p, energy)
# plot_phase_shift(p, 0.1, 100.0)

def test(n):
	b = -2.0 # 2.0
	energy = 10.0

	l = []
	for a in arange(0.5, 50.0, 0.1):
		p = FreeParticle(a, b * a, n)
		l.append((a, compute_phase_shift(p, energy, debug = True)))
	plot = list_plot(l, plotjoined = True, ymin = -pi, ymax = pi)
	plot.save("bbbbb{}.png".format(n))

# test(25)
# test(50)
test(100)
# test(200)

# wf1 = get_wavefunction(a, 11.17270)
# wf2 = get_wavefunction(a, 14.27635)
# print(numerical_integral(lambda x: wf1(x).conjugate() * wf2(x), 0.0, a))



def run_plots():
	bs = list(arange(-20.0, 20.0, 1.0))
	energies = list(arange(0.5, 200.0, 1.0))

	for energy in energies:
		cnt = len(bs)
		plots = sum([list_plot(get_partial_shifts(a, b, energy, n), plotjoined = True, rgbcolor = hue(0.0, 1.0, 1.0 / cnt * i)) for i, b in enumerate(bs)])
		plots.save("output/plot-free-energy-{:.2f}.png".format(energy))

# eigens = find_eigens(a, b, n)

# for i, (wn, _) in enumerate(eigens):
	# print("i = {}, Wavenumber: {}, asymptotic: {}".format(i, wn, float(pi / 2 + pi * i)))

def adsffssf():
	for i, wf1 in enumerate(fp.eigenstates):
		for j, wf2 in enumerate(fp.eigenstates):
			print("i = {}, j = {}:   {}".format(i, j, test_orthonormality(a, wf1, wf2)))
		print("-------------")

# print("First {} wavenumbers: {}".format(n, str(find_wavenumbers(a, b, n))))