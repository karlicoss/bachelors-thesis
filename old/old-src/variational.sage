from numpy import linalg

V_0 = 3.1
L = 1.0
    

def V(x):
    assert(x >= 0)
    if x < L:
        return V_0
    else:
        return 0.0


# TODO does not take L into account
def alalal(energy):
    k_1 = sqrt(energy - V_0)
    k_2 = sqrt(energy)
#    var('A')
#    var('D')
#    psi1(x) = A * e^(-i * k_1 * x) - A * e^(i * k_1 * x)
#    psi1d = psi1.derivative()
#    
#    psi2(x) =     e^(-i * k_2 * x) + D * e^(i * k_2 * x)
#    psi2d = psi2.derivative()
#    
#    print(psi1(L))
#    print(psi2(L))
#    
#    print(psi1d(L))
#    print(psi2d(L))
#    
#    sols = solve([psi1(L) == psi2(L), psi1d(L) == psi2d(L)], A, D)
#    print(sols)
    # psi_2(x) = C * e^(-i k_2 x) + D e^(i k_2 x)
    # psi_1(x) = A (e^(-i k_1 x) - e^(i k_2 x))
        
    var('A')
    var('D')
    sols = solve(
    [A * (e^(-k_1 * I * L) - e^(k_1 * I * L))           - D * e^(k_2 * I * L)           == e^(-k_2 * I * L),
     A * k_1 * L * (e^(-k_1 * I * L) + e^(k_1 * I * L)) + D * k_2 * L * e^(k_2 * I * L) == k_2 * L * e^(-k_2 * I * L)],
    A, D,
    solution_dict = True)

    assert(len(sols) == 1)
    # TODO assert that |D| = 1.0
    Aval = CC(sols[0][A])
    Dval = CC(sols[0][D])
    return (Aval, Dval)
    # return Dval.argument()

def shiftExact(energy):
    _, D = alalal(energy)
    return D.argument()

def wavefunctionExact(energy):
    k_1 = sqrt(energy - V_0)
    k_2 = sqrt(energy)
    A, D = alalal(energy)
    def wf(x):
        if x < 0:
            return 0.0
        elif x < L:
            return A * (e^(-k_1 * I * x) - e^(k_1 * I * x))
        else:
            return e^(-k_2 * I * x) + D * e^(k_2 * I * x)
    return wf

class RmatrixSolver:
    def __init__(self, energe, a, base, debug = False):
        self.energy = energy
        self.a = a
        self.base = base
        self.debug = debug
        self.k_1 = sqrt(energy - V_0) # TODO refactor
        self.k_2 = sqrt(energy) # TODO refactor

        n = len(base)
            
        def getC():
            c = matrix(CC, n, n)
        #    print(c.parent())
            for i in range(n):
                for j in range(n):
                    fi = base[i]
                    fj = base[j]
                    
                    # <fi | -d^2/dx^2 + V(x) + L - E | fj>
                    # 1.
                    dfj = fj.derivative(x, 1)
                    d2fj = fj.derivative(x, 2)
                    p1, _ = numerical_integral(lambda x: fi(x) * (-d2fj(x)), 0.0, a)
                    if debug:
                        print("p1 = {}".format(p1))
                    # 2.
                    p2, _ = numerical_integral(lambda x: fi(x) * V(x) * fj(x), 0.0, a)
                    if debug:
                        print("p2 = {}".format(p2))
                    # 3.
                    p3 = fi(a) * dfj(a)
                    if debug:
                        print("p3 = {}".format(p3))
                    # 4.
                    p4, _ = numerical_integral(lambda x: fi(x) * (- energy) * fj(x), 0.0, a)
                    if debug:
                        print("p4 = {}".format(p4))
                    res = p1 + p2 + p3 + p4
                    if debug:
                        print(res)
                    c[i, j] = res
            return c
                
        def getrhs():
            v = vector(CC, n)
            for i in range(n):
                fi = base[i]
                v[i] = fi(a)
            return v

        # psi(a) / psi'(a)
        def R():
            C = getC()
            rhs = getrhs()
            # print(C)
            # print(rhs)
            
            coeff = vector(linalg.solve(C, rhs))
            # these are actually c_i / psi'^E(a)
            self.coeff = coeff
            return sum([coeff[i] * base[i](a) for i in range(n)])
     
        k = sqrt(energy)
        r = R()
        U = CC(e^(-2 * I * k * a) * (I * k * r + 1) / (I * k * r - 1)) # TODO ????????
        self.U = U
        self.shift = U.argument()
        self.coeff = vector([c * self.wavefunction_Ed(a) for c in self.coeff])

        # TODO !!!!! assuming only one coefficient
        

    def wavefunction_I(self, x):
        return sum(c * f(x) for c, f in zip(self.coeff, self.base))

    def wavefunction_E(self, x):
        return e^(-I * self.k_2 * x) + self.U * e^(I * self.k_2 * x)

    def wavefunction_Ed(self, x):
        return -I * self.k_2 * e^(-I * self.k_2 * x) + self.U * I * self.k_2 * e^(I * self.k_2 * x)

    def get_wavefunction(self):
        def wf(x):
            if x <= 0:
                return 0
            elif x <= a:
                return self.wavefunction_I(x)
            else:
                return self.wavefunction_E(x)
        return wf

def plotWavefunctions(wf, fname):
    xmin = 0.0
    xmax = 10.0
    ymin = -3.0
    ymax = 3.0
    preal = plot(lambda x: wf(x).real(), xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax, color = 'red')
    pimag = plot(lambda x: wf(x).imag(), xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax, color = 'blue')
    pprob = plot(lambda x: wf(x).abs(),  xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax, color = 'black')
    (preal + pimag + pprob).save(fname)


energy = 20.0


print("Shift using exact calculation: {}".format(shiftExact(energy))) 
plotWavefunctions(wavefunctionExact(energy), "exact.png")


def ga(b):
    f(x) = x * exp(-x^2 / b^2)
    return f

def sins(p):
    f(x) = sin(p * x)
    return f


gaussian_base = [ga(i) for i in range(1, 50)]
sin_base = [sins(i) for i in range(1, 50)]

# TODO TEST
# a = 10.0
# L = 5.0
# V_0 = 11.0
# energy = 20.0
# C_11 should be -0.93

# print("-------")
a = 5.0
rms = RmatrixSolver(energy, a, gaussian_base[:20 ], debug = False)

print("Coefficients: {}".format(rms.coeff))
print("Shift using R-matrix: {}".format(rms.shift))

print(rms.wavefunction_E(a))
print(rms.wavefunction_I(a))

plotWavefunctions(rms.get_wavefunction(), "r-matrix.png")