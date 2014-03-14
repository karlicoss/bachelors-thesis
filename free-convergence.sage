
# TODO wavefunctions are valid only for positive energies!!!
# They are orthogonal as eigenstates of the Hamiltonian with non-degenerate spectrum
def get_wavefunction(a, k):
	A = sqrt(1.0 / (4 * (a / 2 - sin(2 * k * a) / (4 * k))))
	def wf(x):
		# we make wavefunction real by multiplying by I
		return CC(A * I * (exp(- I * k * x) - exp(I * k * x))) # -2 * I * sin(k * x)
	return wf


# a = 1.0 !!!
def find_wavenumbers(a, b, count):
	roots = []

	if b > 1.0:
		root = find_root(lambda x: x * coth(x) - b, 0.01, 10.0 * abs(b)) # todo investigate right boundary
		roots.append(root)
	elif b == 1.0:
		roots.append(0.0)
	else: # b < 1.0
		root = find_root(lambda x: x * cot(x) - b, 0.01, pi)
		roots.append(root)

	for i in range(1, count + 1):
		x = pi / 2 + pi * i
		root = find_root(lambda x: x * cot(x) - b, pi * i, pi * (i + 1))
		roots.append(root)
	return roots

def get_R(a, b, energy, count):
	k = sqrt(energy)
	s = 0.0
	wavenumbers = find_wavenumbers(a, b, count)
	# print(energies)
	for klambda in wavenumbers:
		wlambda = get_wavefunction(a, klambda)
		s += wlambda(a) * wlambda(a) / (klambda ** 2 - k ** 2)
	s /= a
	return s

# u'(a) / u(a) = X, where u(a) = e^{-ikx} + U e^{ikx}
def get_U(a, energy, X):
	k = sqrt(energy)
	U = exp(-2 * I * k * a) * (I * k + X) / (I * k - X)
	return CC(U)


a = 1.0
b = -2.0 # 2.0
energy = 1.0

# wf1 = get_wavefunction(a, 11.17270)
# wf2 = get_wavefunction(a, 14.27635)
# print(numerical_integral(lambda x: wf1(x).conjugate() * wf2(x), 0.0, a))


n = 200

for i, wn in enumerate(find_wavenumbers(a, b, n)):
	print("Wavenumber: {}, asymptotic: {}".format(wn, float(pi / 2 + pi * i)))

# print("First {} wavenumbers: {}".format(n, str(find_wavenumbers(a, b, n))))

R = get_R(a, b, energy, n)
X = (1 + R * b) / (R * a)
U = get_U(a, energy, X)
print("n = {}, R = {}, U = {}, phase shift = {}".format(n, R, U, U.arg()))

# for i in range(1, n):
# 	R = get_R(a, b, energy, i)
# 	X = (1 + R * b) / (R * a)
# 	U = get_U(a, energy, X)
# 	print("i = {}, R = {}, U = {}, phase shift = {}".format(i, R, U, U.arg()))