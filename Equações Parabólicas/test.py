import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import *
from scipy.sparse import bmat, diags, eye
from scipy.sparse.linalg import spsolve

# Parameters
n = 20
Lx, Ly = 1.0, 1.0  # Domain size
nx, ny = n, n  # Number of control volumes in x and y directions
dx, dy = Lx / (nx + 1), Ly / (ny + 1)  # Control volume sizes
alpha = 1.0  # Diffusion coefficient
dt = 0.25 * min(dx**2, dy**2) / alpha  # Time step size (CFL condition)
T = 0.24  # Total time
Nt = int(T / dt)  # Number of time steps

# Define the source location and g(t)
x_s, y_s = 0.5, 0.5
def g(t):
    return 2.0 if t < 0.25 else 0.0

# Create grid
x = np.linspace(0, Lx, nx + 2)
y = np.linspace(0, Ly, ny + 2)
X, Y = np.meshgrid(x, y)

# Initialize the solution array
q = np.zeros((nx + 2, ny + 2))  # Including ghost cells
q_new = np.zeros_like(q)

# Find the indices of the source location
i_s = int(x_s / dx) + 1
j_s = int(y_s / dy) + 1

# Create the sparse matrix
def laplacian(n):
    dx = 1.0 / (n+1)
    I = diags([np.ones(n-1),np.ones(n-1)],[-1,1])
    B = eye(n)
    A = diags([np.ones(n-1),-4*np.ones(n),np.ones(n-1)], [-1,0,1])
    C = (kron(I, B) + kron(B, A))/dx**2
    return C

A = laplacian(nx)
I = eye(nx * ny)

# Implicit Euler matrix
implicit_matrix = I - dt * alpha * A

# Time-stepping loop
for n in range(Nt):
    t = n * dt
    q_flat = q[1:-1, 1:-1].flatten()
    
    # Define the source term for the current time step
    S_flat = np.zeros_like(q_flat)
    if t < 0.25:
        index = (i_s - 1) * ny + (j_s - 1)
        S_flat[index] = g(t)
    
    # Right-hand side of the implicit Euler equation
    b = q_flat + dt * S_flat
    
    # Solve the linear system
    q_flat_new = spsolve(implicit_matrix, b)
    
    # Reshape back to 2D array
    q_new[1:-1, 1:-1] = q_flat_new.reshape((nx, ny))
    
    # Apply Neumann boundary conditions (zero gradient)
    q_new[0, :] = q_new[1, :]         # Bottom boundary
    q_new[-1, :] = q_new[-2, :]       # Top boundary
    q_new[:, 0] = q_new[:, 1]         # Left boundary
    q_new[:, -1] = q_new[:, -2]       # Right boundary
    
    # Update q
    q[:, :] = q_new[:, :]

# Plot the final solution
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(X, Y, q)
# plt.imshow(q[1:-1, 1:-1].T, extent=[0, Lx, 0, Ly], origin='lower', aspect='auto', cmap='viridis')
# plt.colorbar(label='q')
plt.title('Solution of the PDE at T = {:.2f}'.format(T))
plt.xlabel('x')
plt.ylabel('y')
plt.show()
