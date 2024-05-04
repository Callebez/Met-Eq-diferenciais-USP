import numpy as np
import matplotlib.pyplot as plt 

def polarToCartesian(r, theta):
    return [r * np.cos(theta), r * np.sin(theta)]


def analyticalSol(r, theta, n):
    val = 0
    for i in range(1, n):
        val += 1.0/ (2.0 * i - 1.0) * r ** (2.0 * i - 1.0) * np.sin((2.0 * i - 1.0) * theta)
    return (400.0 / np.pi ) * val

n = 5
nr = n
nth = n
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
# ax = fig.add_subplot()

r = np.linspace(0, 1, nr)
theta = np.linspace(0, np.pi, nth)

R, T = np.meshgrid(r, theta)
sol = analyticalSol(R, T, 100)
ax.plot_surface(T, R, sol)

h = 1.0/((n+1))
def createA(n):
    A = np.diag(-4 * np.ones(n*n), 0) + np.diag(np.ones((n)*(n)-1), 1)+ np.diag(np.ones((n)*(n)-1), -1)
    A += np.diag(np.ones(n*n-n), n) + np.diag(np.ones(n*n-n), -n)
    return A/h**2

b=np.zeros(n*n)
b[n*n-n-1:] = -100.0/h**2

# b = b.reshape(n*n)
print(b)
x = np.linalg.solve(createA(n),b)
# # print(x[0:n])
# x[0:n] = 0
X = x.reshape((n,n))

# # print(X)
ax.plot_surface(T, R, X)
# print()
plt.show()