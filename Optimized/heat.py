from matplotlib import cm
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.sparse import kron, diags, eye, issparse, csc_matrix
import scipy.sparse.linalg as la 
import math
def laplacian(n):
    d = 1 / ( n + 1)
    main_diag = -2 * np.ones(n)  
    off_diag =  np.ones(n-1)  
    return diags([off_diag, main_diag, off_diag], [-1, 0, 1],shape=(n,n))/d**2

def heat_source_uniform(x,y):
    xs = 1.0 / 2.0 
    ys = 1.0 / 2.0
    w = 0.2
    return np.exp(-((x-xs)**2+(y-ys)**2)/w**2)
def g(t):
    return 2.0 if t < 0.25 else 0.0

def laplacian_Dirichlet(n, d, k):
    main_diag = -2 * np.ones(n) 
    main_diag[0] = -d**2#/k
    main_diag[-1] = -d**2#/k
    lower_diag =  np.ones(n-1)  
    upper_diag =  np.ones(n-1)  
    upper_diag[0] = 0.0
    lower_diag[-1] = 0.0

    return diags([lower_diag, main_diag, upper_diag], [-1, 0, 1],shape=(n,n))/d**2

def heat_equation_heterogeneous(nx, ny, Lx, Ly, time, dt):
    
    Ix = eye(nx+2)
    Iy = eye(ny+2)
   
    dx, dy = Lx / (nx+1), Ly / (ny+1)  # Control volume 
    alpha = 1.0
    # dt = 0.25 * min(dx**2, dy**2) / alpha  # Time step size (CFL condition)
    T = time
    Nt = int(T / dt)  # Number of time steps

    Laplacian_x = laplacian_Dirichlet(nx+2, dx, dt)
    Laplacian_y = laplacian_Dirichlet(nx+2, dx, dt)
    
    Dx = kron(Iy, Laplacian_x)
    Dy = kron(Laplacian_y, Ix)
    Ax = eye((nx+2)* (ny+2)) -  dt * Dx
    Ay = eye((nx+2)* (ny+2)) -  dt * Dy
    # A = eye((nx+2)* (ny+2)) - dt * (Dx + Dy)
  
    x = np.linspace(0, Lx, nx + 2)
    y = np.linspace(0, Ly, ny + 2)
    X, Y = np.meshgrid(x, y)
    S = heat_source_uniform(X, Y)
    S[:,0] = 0.0
    S[:,-1] = 0.0
    S[-1,:] = 0.0
    S[0,:] = 0.0
    S = S.flatten()
    Q = np.zeros((nx+2)*(ny+2))
    # Q[0::nx+2]  = - 1                                          # Right boundary
    # Q[nx+1::nx+2] = - 1                                        # Left boundary
    # Q[:nx+2]  = 1./ np.pi * np.sin(np.pi * x)                  # Bottom boundary
    # Q[-(nx+2):]  = 1./ (3 * np.pi) * np.sin(3 * np.pi * x) + 1 # Top boundary
    for i in range(Nt):
        
        Q[0::nx+2]  = - 1                                          # Right boundary
        Q[nx+1::nx+2] = - 1                                        # Left boundary
        Q[:nx+2]  = 1./ np.pi * np.sin(np.pi * x)                  # Bottom boundary
        Q[-(nx+2):]  = 1./ (3 * np.pi) * np.sin(3 * np.pi * x) + 1 # Top boundary
        # Q = la.spsolve(A, Q +  dt * S )
        # Solve for the new time step
        q_flat_new_aux = la.spsolve(Ax, Q + 0.5 * dt * S)
        q_flat_new = la.spsolve(Ay, q_flat_new_aux + 0.5 * dt * S)

        Q = q_flat_new

    Q = Q.reshape((nx+2, ny+2))
    return X, Y, Q

def laplacian_Neumann(n, d, dt):
    main_diag = -2 * np.ones(n) 
    main_diag[0] =  - 2.0 * d**2
    main_diag[-1] = - 2.0 * d**2
    lower_diag =  np.ones(n-1)  
    upper_diag =  np.ones(n-1)  
    upper_diag[0] =  2.0 * d**2
    lower_diag[-1] =  2.0 * d**2
    return diags([lower_diag, main_diag, upper_diag], [-1, 0, 1], shape=(n,n))/d**2

def heat_equation_uniformNeumann(nx, ny, Lx, Ly, time, dt):
    
    Ix = eye(nx+2)
    Iy = eye(ny+2)
   
    dx, dy = Lx / (nx+1), Ly / (ny+1)  # Control volume 

    Nt = int(time / dt)  # Number of time steps

    Laplacian_x = laplacian_Neumann(nx + 2, dx, dt)
    Laplacian_y = laplacian_Neumann(ny + 2, dy, dt)
    
    Dx = kron(Iy, Laplacian_x)
    Dy = kron(Laplacian_y, Ix)
    # Dy = kron(Ix, Laplacian_y)

    Ax = eye((nx+2)* (ny+2)) - dt * Dx
    Ay = eye((nx+2)* (ny+2)) - dt * Dy
    A = eye((nx+2)* (ny+2)) - dt * (Dx + Dy)
    # print(eye((nx+2)* (ny+2)).toarray())
    x = np.linspace(0, Lx, nx + 2)
    y = np.linspace(0, Ly, ny + 2)
    X, Y = np.meshgrid(x, y)
    S = heat_source_uniform(X, Y)
    
    S = S.flatten()    
    Q = np.zeros((nx+2)*(ny+2))

    print(dt, time, Nt)
    for i in range(Nt):     
        # Solve for the new time step
        
        ### Using a coupled laplacian      
        # Q = la.spsolve(A, Q + dt * S)
        ### Uncoupled Laplacian 
        q_new_aux = la.spsolve(Ax, Q + dt / 2 * S)
        Q = la.spsolve(Ay, q_new_aux + dt / 2 * S)

    Q = Q.reshape((nx+2, ny+2))
    return X, Y, Q
def heat_equation_deltaNeumann(nx, ny, Lx, Ly, time, dt):
    
    Ix = eye(nx+2)
    Iy = eye(ny+2)
   
    dx, dy = Lx / (nx+1), Ly / (ny+1)  # Control volume 
    alpha = 1.0
    # dt = 0.25 * min(dx**2, dy**2) / alpha  # Time step size 
   
    T = time
    Nt = int(T / dt)  # Number of time steps

    Laplacian_x = laplacian_Neumann(nx + 2, dx, dt)
    Laplacian_y = laplacian_Neumann(ny + 2, dy, dt)
    # print(aplacian_x.toarray())
    Dx = kron(Iy, Laplacian_x)
    Dy = kron(Laplacian_y, Ix)
    # Dy = kron(Ix, Laplacian_y)

    Ax = eye((nx+2)* (ny+2)) - dt * Dx
    Ay = eye((nx+2)* (ny+2)) - dt * Dy
    A = eye((nx+2)* (ny+2)) - dt * (Dx + Dy)
    
    x = np.linspace(0, Lx, nx + 2)
    y = np.linspace(0, Ly, ny + 2)
    X, Y = np.meshgrid(x, y)
    # S = heat_source_uniform(X, Y)

    # S = S.flatten()    
    Q = np.zeros((nx+2)*(ny+2))
    xs, ys = 0.5, 0.5
    i_s = int(xs / dx) + 1
    j_s = int(ys / dy) + 1
    
    for i in range(Nt):     
        
        # Solve for the new time step
         
        S = np.zeros_like(Q)
        t = i * dt
        if t < 0.25:
            index = (i_s - 1) * nx + (j_s - 1)
            S[index] = g(t)

        ### Using a coupled laplacian      
        # Q = la.spsolve(A, Q + dt * S)
      
        # Q = Q_new
        ### Uncoupled Laplacian     
        q_new_aux = la.spsolve(Ax, Q + dt / 2.0 * S)
        # S = np.zeros_like(Q)
        # t = i * dt
        # if t < 0.25:
        #     index = (i_s - 1) * nx + (j_s - 1)
        #     S[index] = g(t)
        Q = la.spsolve(Ay, q_new_aux + dt / 2.0 * S)
        
    Q = Q.reshape((nx+2, ny+2))
    return X, Y, Q

def spatialError(max_power, time_iterations, func):
    # dt = 1./45.
    dt = 5e-3
    n = 3**max_power
    m = math.ceil(n/2)
    total_time = dt*time_iterations
    X, Y, Q = func(n, n, 1.0, 1.0, total_time, dt)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(X,Y,Q)
    plt.show()

    # print(Q[m, m], X[m, m])
    refValue = Q[m, m]
    erro = []
    h = []
    for i in range(1,max_power):
        n = 3**i
        m = math.ceil(n/2)
        h.append(1/(n+1))
        # total_time = 0.25*(1/(n+1))**2*time_iterations
        X, Y, Q = func(n,n, 1.0, 1.0, total_time, dt)
        erro.append(np.abs(Q[m,m] - refValue))
    return h, erro
def temporalError(max_power, total_time, func):
    dt_min = np.power(5., -max_power)
    n = 18
    m = math.ceil(n/2)
    # total_time = dt*time_iterations
    X, Y, Q = func(n, n, 1.0, 1.0, total_time, dt_min)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(X,Y,Q)
    plt.show()

    # print(Q[m, m], X[m, m])
    refValue = Q[m, m]
    erro = []
    h = []
    for i in range(2, max_power):
        dt = np.power(5., -i)
        # total_time = dt * time_iterations
        h.append(dt)
        # total_time = 0.25*(1/(n+1))**2*time_iterations
        X, Y, Q = func(n,n, 1.0, 1.0, total_time, dt)
        erro.append(np.abs(Q[m,m] - refValue))
    return h, erro

nx = 81
ny = 81
dt = 0.001
Lx = 1.0
Ly = 1.0
# X, Y, Q = heat_equation_deltaNeumann(nx, ny, Lx, Ly, .251, dt)
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.plot_surface(X,Y,Q)
# plt.show()

# X1, Y1, Q1 = heat_equation_heterogeneous(nx, ny, Lx, Ly, 5.0)
h, error = temporalError(7, 0.2, heat_equation_uniformNeumann)
# h, error = spatialError(7, 10, heat_equation_uniformNeumann)
plt.title("Convergência para fonte uniforme \n e condições de Dirichlet ($h = 0.05$)")
plt.loglog(h, error, '*')
plt.loglog(h, h, label=r"h")
plt.loglog(h,np.power(h, 2), label=r"h^2")
# plt.loglog(h,np.power(h, 3), label=r"h^3")
plt.xlabel(r"$log(dt)$")
plt.ylabel(r"$Log(|E|)$ no ponto $(0.5,0.5)$")
plt.legend()
plt.grid()
plt.show()

# fig = plt.figure(figsize=plt.figaspect(0.5))
# fig.suptitle("Distribuição de calor uniforme \n Condições de fronteira de Dirichlet")
# # =============
# # First subplot
# # =============
# # set up the Axes for the first plot
# ax = fig.add_subplot(projection='3d')
# # ax.set_title("t = 0.25")
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# # ax.set_zticks([])
# # plot a 3D surface like in the example mplot3d/surface3d_demo
# surf = ax.plot_surface(X, Y, Q, rstride=1, cstride=1, cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)
# # ax.set_zlim(-1.01, 1.01)
# fig.colorbar(surf, shrink=0.5, aspect=10)

# # # ==============
# # # Second subplot
# # # ==============
# # # set up the Axes for the second plot
# # ax = fig.add_subplot(1, 2, 2, projection='3d')
# # ax.set_title("t = 0.26")
# # ax.set_xlabel("X")
# # ax.set_zticks([])
# # ax.set_ylabel("Y")
# # # plot a 3D wireframe like in the example mplot3d/wire3d_demo
# # surf = ax.plot_surface(X1, Y1, Q1, rstride=1, cstride=1, cmap=cm.coolwarm,
# #                        linewidth=0, antialiased=False)
# # fig.colorbar(surf, shrink=0.5, aspect=10)

# plt.show()



