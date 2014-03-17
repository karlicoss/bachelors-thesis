from numpy import arange

### Some common functions ###

def test_orthonormality(a, wf1, wf2):
	return numerical_integral(lambda x: wf1(x) * wf2(x).conjugate(), 0.0, a)[0]


def make_real(c):
	if c.arg() > 0:
		return c.abs()
	else:
		return -c.abs()

# normalizes the functions f on the interval (0, a)
# also makes functions pure real
def normalize(f, a):
	A = 1.0 / sqrt(numerical_integral(lambda x: f(x).norm(), 0.0, a)[0])
	return lambda x: A * make_real(f(x))
###

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
	# TODO normalize numerically maybe?
	def get_wavefunction_negative(self, k):
		a = self.r_A
		A = 1.0 / sqrt(sinh(2 * a * k) / k - 2 * a)
		def wf(x):
			return CC(A * (exp(- k * x) - exp(k * x))) # -2 * sinh(k * x)
		return wf

	# Solution of -d^2 psi/dx^2 = 0
	def get_wavefunction_zero(self, k):
		a = self.r_A
		A = 1.0 / sqrt(a^3 / 3)
		def wf(x):
			return CC(A * x)
		return wf

	# Solution of -d^2 pdi/dx^2 = k^2 psi(x)
	def get_wavefunction_positive(self, k):
		a = self.r_A
		# A = sqrt(1.0 / (4 * (a / 2 - sin(2 * k * a) / (4 * k))))
			# return CC(A * I * (exp(- I * k * x) - exp(I * k * x))) # -2 * I * sin(k * x)
		def wf(x):
			return CC(exp(- I * k * x) - exp(I * k * x))
		nwf = normalize(wf, a)
		return nwf

	def find_eigens(self):
		states = []
		values = []

		b = self.r_B # TODO
		a = self.r_A # TODO

		if b > 1.0:
			root = find_root(lambda x: x * coth(x) - b, 0.01, 10.0 * abs(b)) # TODO investigate right boundary
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
			root = find_root(lambda x: x * cot(x) - b, 0.01, pi - 0.01) # TODO 0.01
			root /= a
			wf = self.get_wavefunction_positive(root)
			states.append(wf)
			values.append(root ** 2)

		i = 1
		while len(states) < self.n:
			root = find_root(lambda x: x * cot(x) - b, pi * i + 0.01, pi * (i + 1) - 0.01) # TODO 0.01
			root /= a
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


def compute_phase_shift(p, energy):
	def get_R(p, energy):
		s = 0.0
		# print(energies)
		for es, ee in zip(p.eigenstates, p.eigenvalues):
			s += es(p.r_A) ** 2 / (ee - energy) / p.r_A
			# print("i = {}, R = {}".format(i, s))
		return s

	R = get_R(p, energy)
	# Rub = get_R_upper_bound(a, b, energy, n)
	X = (1 + R * b) / (R * a)
	U = get_U(p, energy, X)
	print("n = {}, R = {}, U = {}, phase shift = {}".format(n, R, U, U.arg()))


# TODO refactor
def get_R_upper_bound(a, b, energy, count):
	s = 0.0
	k = sqrt(energy)
	eigens = find_eigens(a, b, count)
	i = 0
	for (elambda, wflambda) in eigens:
		if pi / 2 + pi * i  < k:
			s += wflambda(a) * wflambda(a) / (elambda - energy) / a
			i += 1
		else:
			break
	from sage.functions.other import psi1
	print(i)
	rest = - 1.0 / (pi * k) * (-psi1(0.5 + i + k / pi) + psi1(0.5 + i - k / pi))
	print(float(rest))
	return float(s + rest)



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


a = 1.0
b = -2.0 # 2.0
n = 10
energy = 5.0

fp = FreeParticle(a, b, n)

compute_phase_shift(fp, energy)




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

for i, wf1 in enumerate(fp.eigenstates):
	for j, wf2 in enumerate(fp.eigenstates):
		print("i = {}, j = {}:   {}".format(i, j, test_orthonormality(a, wf1, wf2)))
	print("-------------")


# print("First {} wavenumbers: {}".format(n, str(find_wavenumbers(a, b, n))))