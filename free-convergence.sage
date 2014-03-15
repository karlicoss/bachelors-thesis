from numpy import arange

# Solution of -d^2 psi/dx^2 = - k^2 psi(x)
def get_wavefunction_negative(a, k):
	A = 1.0 / sqrt(sinh(2 * a * k) / k - 2 * a)
	def wf(x):
		return CC(A * (exp(- k * x) - exp(k * x))) # -2 * sinh(k * x)
	return wf

# Solution of -d^2 psi/dx^2 = 0
def get_wavefunction_zero(a, k):
	A = 1.0 / sqrt(a^3 / 3)
	def wf(x):
		return CC(A * x)
	return wf


# TODO wavefunctions are valid only for positive energies!!!
# They are orthogonal as eigenstates of the Hamiltonian with non-degenerate spectrum
# Solution of -d^2 pdi/dx^2 = k^2 psi(x)
def get_wavefunction_positive(a, k):
	A = sqrt(1.0 / (4 * (a / 2 - sin(2 * k * a) / (4 * k))))
	def wf(x):
		# we make wavefunction real by multiplying by I
		return CC(A * I * (exp(- I * k * x) - exp(I * k * x))) # -2 * I * sin(k * x)
	return wf

def test_orthonormality(a, wf1, wf2):
	return numerical_integral(lambda x: wf1(x) * wf2(x).conjugate(), 0.0, a)[0]

# print(test_normality(1.0, get_wavefunction_negative(1.0, 2.0)))
# print(test_normality(1.0, get_wavefunction_zero(1.0, 2.0)))
# print(test_normality(1.0, get_wavefunction_positive(1.0, 2.0)))


# a = 1.0 !!!
def find_eigens(a, b, count):
	res = []

	if b > 1.0:
		root = find_root(lambda x: x * coth(x) - b, 0.01, 10.0 * abs(b)) # TODO investigate right boundary
		wf = get_wavefunction_negative(a, root)
		res.append((-root ** 2, wf))
	elif b == 1.0:
		root = 0.0
		wf = get_wavefunction_zero(a, root)
		res.append((root ** 2, wf))
	else: # b < 1.0
		root = find_root(lambda x: x * cot(x) - b, 0.01, pi - 0.01) # TODO 0.01
		wf = get_wavefunction_positive(a, root)
		res.append((root ** 2, wf))

	for i in range(1, count + 1):
		root = find_root(lambda x: x * cot(x) - b, pi * i + 0.01, pi * (i + 1) - 0.01) # TODO 0.01
		wf = get_wavefunction_positive(a, root)
		res.append((root ** 2, wf))
	return res

def get_R(a, b, energy, count):
	s = 0.0
	eigens = find_eigens(a, b, count)
	# print(energies)
	k0, wf0 = eigens[0]
	print(k0)
	print(wf0(a) * wf0(a))

	for elambda, wflambda in eigens:
		s += wflambda(a) * wflambda(a) / (elambda - energy)
	s /= a
	return s

# u'(a) / u(a) = X, where u(a) = e^{-ikx} + U e^{ikx}
def get_U(a, energy, X):
	k = sqrt(energy)
	U = exp(-2 * I * k * a) * (I * k + X) / (I * k - X)
	return CC(U)


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
b = 3.0 # 2.0
energy = 1.0

# wf1 = get_wavefunction(a, 11.17270)
# wf2 = get_wavefunction(a, 14.27635)
# print(numerical_integral(lambda x: wf1(x).conjugate() * wf2(x), 0.0, a))


n = 50

bs = list(arange(-20.0, 20.0, 0.5))
energies = list(arange(0.5, 30.0, 0.5))

for energy in energies:
	cnt = len(bs)
	plots = sum([list_plot(get_partial_shifts(a, b, energy, n), plotjoined = True, rgbcolor = hue(0.0, 1.0, 1.0 / cnt * i)) for i, b in enumerate(bs)])
	plots.save("output/plot-free-energy-{:.2f}.png".format(energy))

# eigens = find_eigens(a, b, n)

# for i, (wn, _) in enumerate(eigens):
	# print("i = {}, Wavenumber: {}, asymptotic: {}".format(i, wn, float(pi / 2 + pi * i)))

# for i, (_, wf1) in enumerate(eigens):
# 	for j, (_, wf2) in enumerate(eigens):
# 		print("i = {}, j = {}:   {}".format(i, j, test_orthonormality(a, wf1, wf2)))
# 	print("-------------")


# print("First {} wavenumbers: {}".format(n, str(find_wavenumbers(a, b, n))))

# R = get_R(a, b, energy, n)
# X = (1 + R * b) / (R * a)
# U = get_U(a, energy, X)
# print("n = {}, R = {}, U = {}, phase shift = {}".format(n, R, U, U.arg()))