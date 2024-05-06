import numpy as np 
import matplotlib.pyplot as plt 
from scipy.sparse import *
from scipy.sparse.linalg import *
from matplotlib import cm

def matprint(mat, fmt="g"):
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")



def Tmatrix(n):
    dr = 1.0 / (nr) 
    dt = np.pi / (nt)
    Tmain_diag  = ([-2.0 * (1.0/dr**2 + 1.0/( (x*dr)** 2 * dt**2)) for x in range(1,nr)])
    Tupper_diag = ([(1.0/dr**2 + 1./( 2 * (x*dr) * dr)) for x in range(1,nr-1)])
    Tlower_diag = ([(1.0/dr**2 - 1./( 2 * ((x+1)*dr) * dr)) for x in range(1,nr-1)])

    return diags([Tmain_diag, Tupper_diag, Tlower_diag],[0,1,-1])

def Ablocks(nr, nt):
    dr = 1.0 / (nr) 
    dt = np.pi / (nt)
    Gupper_diag = np.array([1.0/((x * dr)**2 * dt**2) for x in range(1,nr)])
    Glower_diag = np.array([1.0/(((x+1) * dr)**2 * dt**2) for x in range(0,nr-1)])
    v_upper = np.array([])
    v_upper = np.append(v_upper, Gupper_diag)
    
    v_lower = np.array([])
    v_lower = np.append(v_lower, Glower_diag)

    T = Tmatrix(nr)
    A = block_diag((T,T))
    for i in range(nt-3):
        A = block_diag((A,T))
        v_upper = np.append(v_upper, Gupper_diag)
        v_lower = np.append(v_lower, Glower_diag)

    G = diags([v_lower, v_upper], [-nr+1, nr-1]) # verificar se os dois g são o mesmo.

    return A+G

def analyticalSol(r, theta, n):
    val = 0
    for i in range(1, n):
        val += 1.0/ (2.0 * i - 1.0) * r ** (2.0 * i - 1.0) * np.sin((2.0 * i - 1.0) * theta)
    return (400.0 / np.pi ) * val



# A = Ablocks(nr,nt)
# # matprint(A.toarray())
# b = np.zeros(nt-1)
# b[-1] = -100./dr**2 * (1.0 + 1./(2.*(nr)))
# v = np.array([])

# for i in range(nt-1):
#     v = np.append(v, b)
# # v[-1] = 0
# # v[n-1] = 0
# v = v.reshape((nr-1,nt-1))
# # matprint(v)
# x = spsolve(A,v.reshape((nr-1)*(nt-1)))
# X = x.reshape((nr-1,nt-1))
# # matprint(X)
# Y = np.zeros((nr+1,nt+1))
# Y[1:nr,1:nr] = X
# Y[1:nr, -1] = 100




# A = sol-Y
# A.reshape(len(sol)**2)
# L2 = [x**2 for x in A]
# L2 = np.sqrt(np.sum(L2))
# print(L2)
# # print()
# plt.legend()
# plt.show()

def resolve(numR, numT):
    dr = (1.0 / numR)
    dt = (np.pi / numT)
    A = Ablocks(numR,numT)
    b = np.zeros(numT-1)
    b[-1] = -100./dr**2 * (1.0 + 1./(2.*(numR+1)))
    v = np.array([])

    for i in range(numT-1):
        v = np.append(v, b)
    x = spsolve(A,v)
    X = x.reshape((numR-1,numT-1))
    Y = np.zeros((numR+1,numT+1))
    Y[1:numR,1:numR] = X
    Y[1:numR, -1] = 100
    return Y
# n = 500
# nr = n
# nt = n
# r = np.linspace(0, 1, nr+1)
# theta = np.linspace(0, np.pi, nt+1)
# R, T = np.meshgrid(r,theta)
# sol =  analyticalSol(R, T, 1000)
# boundary = 100.0*np.ones(nr+1)
# boundary[0] = 0
# boundary[-1] = 0
# gibbsError = boundary-sol[:,-1]
# plt.plot(r, gibbsError)
# plt.grid()
# plt.show()
# print(sol[:,nt])
h = []
l2 = []
# for i in range(2,9):
#     n = int(2**i)
#     nt = n
#     nr = n
#     r = np.linspace(0, 1, nr+1)
#     theta = np.linspace(0, np.pi, nt+1)
#     R, T = np.meshgrid(r,theta)
#     sol =  analyticalSol(R, T, 1000)
#     X = resolve(nt, nr)
#     erro = sol[1:nr-2,1:nt-2]-X[1:nr-2,1:nt-2]
#     erro = [x**2 for x in erro.reshape(len(erro)**2)]
#     erro = np.sqrt(np.sum(erro))
#     h.append(1/nr)
#     l2.append(erro)
i = 5
n = 20
nt = n
nr = n
r = np.linspace(0, 1, nr+1)
theta = np.linspace(0, np.pi, nt+1)
R, T = np.meshgrid(r,theta)
sol =  analyticalSol(R, T, 200)
X = resolve(nt, nr)

plt.rcParams['text.usetex'] = True
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# ax = fig.add_subplot()

# ax.contour(R, T, X-sol,label="erro")
# CS = ax1.contourf(R, T, X)
ax.set_title("Equação de Laplace em coordenadas Polares",fontsize=16)
ax.plot_surface(R, T, X,label="Solução por diferenças finitas")
ax.plot_surface(R, T, sol,label="Solução analítica")
# plt.subplot(131)
CS = ax.contourf(R, T, X-sol, zdir='z', offset=-2,cmap=cm.hot)
cbar = fig.colorbar(CS)
cbar.ax.set_ylabel('Diferença entre solução analítica e por diferenças finitas')

ax.set_xlabel(r'r',fontsize=12)
ax.set_ylabel(r'$\theta$',fontsize=12)

# ax.plot_surface(R, T, sol)

# ax.clabel(CS, inline=True, fontsize=10)
# ax1.set_title('Simplest default with labels')
# ax.plot_surface(R,T, X)

# ax.plot_surface(R, T, sol)

# plt.loglog(h,l2, "o" , "erro norma L2")
# plt.loglog(h, np.power(h,2),label="h^2")
# plt.loglog(h, h, label="h")
plt.legend()
plt.grid()
plt.show()
