import numpy as np
import matplotlib.pyplot as plt

def Heaviside(x):
    return np.heaviside(x, 1)

x = np.linspace(-2, 2, 1000)
y = [Heaviside(i) for i in x]

plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('H(x)')
plt.title('Graph of Heaviside Step Function')
plt.grid(True)
# plt.show()
plt.savefig('H(x)')