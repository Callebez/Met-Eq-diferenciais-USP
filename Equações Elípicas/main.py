import numpy as np
import matplotlib.pyplot as plt 

def mapCircle(r, theta):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return [x,y]
def analyticalSol(r, theta, n):
    val = 0
    for i in range(1, n):
        val += 1.0/ (2.0 * i - 1.0) * r ** (2.0 * i - 1.0) * np.sin((2.0 * i - 1.0) * theta)
    return (400.0 / np.pi ) * val



nr = 100
nth = 100
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
r = np.linspace(0, 1, nr)
theta = np.linspace(0, np.pi, nth)
R, T = np.meshgrid(r,theta)
sol = analyticalSol(R,T, 100)
X, Y = R * np.cos(T), R * np.sin(T)
ax.plot_surface(X, Y, sol)
plt.show()