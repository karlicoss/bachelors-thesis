import numpy as np
import matplotlib.pyplot as plt

width = 100
step = 10
L = int(width // step)

en = 0.1
kk = np.sqrt(en)


wf = np.zeros(L, dtype=np.complex)
wf[0] = 1
wf[1] = wf[0] * np.exp(1j * kk * step)

for i in range(2, L):
    wf[i] = (2 - step ** 2 * en) * wf[i - 1] - wf[i - 2]

# pf = np.square(np.absolute(wf))
# print(pf)

plt.plot(wf)
plt.show()
# print(pf)
