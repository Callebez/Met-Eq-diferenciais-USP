import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import *
from scipy.sparse import linalg
from scipy.integrate import odeint, solve_ivp
n = 10**2
nx = n
ny = n
Lx = 1
Ly = 1
def laplacian(n):
    dx = 1.0 / (n+1)
    I = diags([np.ones(n-1),np.ones(n-1)],[-1,1])
    B = eye(n)
    A = diags([np.ones(n-1),-4*np.ones(n),np.ones(n-1)], [-1,0,1])
    C = (kron(I, B) + kron(B, A))/dx**2
    return C
def heat_source(x,y):
    xs = 1.0 / 2.0 
    ys = 1.0 / 2.0
    w = 0.2
    return np.exp(-((x-xs)**2+(y-ys)**2)/w**2)

m = 10
x = np.linspace(0, Lx, m + 2)
y = np.linspace(0, Ly, m + 2)
X, Y = np.meshgrid(x, y)
S = heat_source(X,Y)
S = S[1:-1, 1:-1].flatten()

def heating_plate(Q,t):
    n = int(np.sqrt(len(Q)))
    L  = laplacian(n)
    return csr_matrix.dot(L, Q)+ S 

fig = plt.figure()
ax = fig.add_subplot( projection='3d')
# ax.plot_surface(X,Y,S)
# plt.show()


Q = np.zeros((m+2,m+2))
q_new = np.zeros_like(Q)

Lx, Ly = 1.0, 1.0  # Domain size
nx, ny = m, m  # Number of control volumes in x and y directions
dx, dy = Lx / (nx+1), Ly / (ny+1)  # Control volume si
alpha = 1.0
dt = 0.25 * min(dx**2, dy**2) / alpha  # Time step size (CFL condition)
T = 0.1
Nt = int(T / dt)  # Number of time steps

# dt = 0.01 
for n in range(Nt):
    # print(n)
    q_flat = Q[1:-1, 1:-1].flatten()
    
    # Solve for the new time step
    aux = heating_plate(q_flat, n*dt)
    q_flat_new = q_flat + dt * (aux)
    q_flat_new = q_flat + 0.5 * dt * ((aux) + heating_plate(q_flat_new, n*dt))
    
    # Reshape back to 2D array
    q_new[1:-1, 1:-1] = q_flat_new.reshape((m, m))
    
    # Apply Neumann boundary conditions (zero gradient)
    q_new[0, :] = q_new[1, :]         # Bottom boundary
    q_new[-1, :] = q_new[-2, :]       # Top boundary
    q_new[:, 0] = q_new[:, 1]         # Left boundary
    q_new[:, -1] = q_new[:, -2]       # Right boundary
    
    # Update q
    Q[:, :] = q_new[:, :]
    # if n % 1000 == 0:
        # label = "it" + str(n)
        # ax.plot_surface(X,Y, Q,label=label)
    # print(Q.shape)
ax.plot_surface(X,Y, Q,label="final")

plt.legend()
plt.show()
# heating_plate(n,Q,S)  
# x = np.linspace(0, Lx, n + 2)
# y = np.linspace(0, Ly, n + 2)
# X, Y = np.meshgrid(x, y)
# F = heat_source(X,Y)
# # F.reshape
# C = laplacian(n)
# F = F[1:-1,1:-1].reshape(n*n)

