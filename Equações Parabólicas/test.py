import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import bmat, diags, eye
from scipy.sparse.linalg import spsolve
from matplotlib import cm

# Parameters
Lx, Ly = 1.0, 1.0  # Domain size
nx, ny = 50, 50  # Number of control volumes in x and y directions
dx, dy = Lx / (nx+1), Ly / (ny+1)  # Control volume sizes
alpha = 1.0  # Diffusion coefficient
dt = 0.25 * min(dx**2, dy**2) / alpha  # Time step size (CFL condition)
T = 1.2  # Total time
Nt = int(T / dt)  # Number of time steps

def Tmatrix(nx, ny):
    dx = 1.0 / (nx + 1)
    dy = 1.0 / (ny + 1)
    main_diag  = (-2.0 * (1.0 / dx**2 + 1.0 / dy**2)) * np.ones(nx)
    upper_diag = (1.0 / dx**2) * np.ones(nx-1)
    lower_diag = (1.0 / dy**2) * np.ones(nx-1)
    return diags([lower_diag, main_diag, upper_diag], [-1, 0, 1])

def create_sparse_block_tridiagonal_matrix(nx, ny):
    I_sparse = eye(nx)
    D_sparse = Tmatrix(nx, ny)
    
    dx = 1.0 / (nx + 1)
    dy = 1.0 / (ny + 1)
    blocks = [[None] * ny for _ in range(ny)]
    
    for i in range(ny):
        blocks[i][i] = D_sparse  # Main diagonal
        if i > 0:
            blocks[i][i - 1] = (1.0 / dy**2) * I_sparse  # Lower diagonal
        if i < ny - 1:
            blocks[i][i + 1] = (1.0 / dy**2) * I_sparse  # Upper diagonal
    
    tridiagonal_matrix = bmat(blocks, format='csr')
    
    return tridiagonal_matrix

def heat_source(x, y):
    xs, ys = 0.5, 0.5
    w = 0.2
    return np.exp(-((x - xs)**2 + (y - ys)**2) / w**2)

# Create grid
x = np.linspace(0, Lx, nx + 2)
y = np.linspace(0, Ly, ny + 2)
X, Y = np.meshgrid(x, y)

# Initialize the solution array
q = np.zeros((nx + 2, ny + 2))  # Including ghost cells
q_new = np.zeros_like(q)

# Source term (Gaussian distribution)
S = heat_source(X, Y)

# Convert 2D arrays to 1D vectors for sparse matrix operations
S_flat = S[1:-1, 1:-1].flatten()

# Create the sparse matrix
A = create_sparse_block_tridiagonal_matrix(nx, ny)
print(Nt)
# Time-stepping loop
for n in range(Nt):
    # print(n)
    q_flat = q[1:-1, 1:-1].flatten()
    
    # Solve for the new time step
    q_flat_new = q_flat + dt * (alpha * A.dot(q_flat) + S_flat)
    
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
plt.imshow(q[1:-1, 1:-1].T, extent=[0, Lx, 0, Ly], origin='lower', aspect='auto', cmap='viridis')
plt.colorbar(label='q')
plt.title('Solution of the PDE at T = {:.2f}'.format(T))
plt.xlabel('x')
plt.ylabel('y')

# fig = plt.figure()
# ax1 = fig.add_subplot(121, projection='3d')
# ax = fig.add_subplot()
# ax1.plot_surface(X,Y, q.T)
# CS = ax.contourf(X, Y, q)
# cbar = fig.colorbar(CS)
plt.show()
