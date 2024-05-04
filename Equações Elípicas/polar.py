import numpy as np 
import matplotlib.pyplot as plt 
from scipy.sparse import diags, dia_matrix, block_diag, bmat, hstack, vstack
from scipy.sparse.linalg import spsolve
import pprint
n = 100
dr = 1.0/n
dp = np.pi/n
ri = np.linspace(dr, 1,n,True)


def Tmatrix(n):
    ri = np.linspace(dr,1,n)
    T_main_diag = -2.0* (1. / dr**2  + 1./(ri ** 2 * dp**2))
    T_upper_diag = 1./dr**2 + 1./(2 * ri[:n-1] * dr)
    T_lower_diag = 1./dr**2 - 1./(2 * ri[1:n] * dr)
    data = [T_main_diag, T_upper_diag , T_lower_diag]
    Ts = diags(data,[0,1,-1])
    return Ts
def Gmatrix(n, i):
    ri = np.linspace(dr,1,n)
    G = diags(1.0/(ri**2*dp**2), i)
    return G

def matprint(mat, fmt="g"):
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")

ri = np.linspace(dr,1,n)

G_diag = 1.0/(ri**2 * dp**2)

m = n - 1
T = Tmatrix(n)
A = block_diag((T,T))
for i in range(m-1):
    A = block_diag((A,T))

v = np.array([])
for i in range(m):
    v = np.append(v, G_diag)
# print(v)

Gsup = diags(v,n)
Ginf = diags(v, - n)
# matprint(Gsup.toarray())
# matprint(Ginf.toarray())

B = A+Gsup+Ginf
# matprint(B.toarray())

b = np.zeros(n)
b[-1] = -100./(dp**2)

v1 = np.array(b)
for i in range(m):
    v1 = np.append(v1,b)
# print(len(v1))

B = B.tocsc()
x = spsolve(B,v1)
# print(len(x))

nr = n
nth = n
r = np.linspace(0, 1, nr)
theta = np.linspace(0, np.pi, nth)

X = x.reshape((n,n))

R, T = np.meshgrid(r, theta)
fig = plt.figure()
# ax = fig.add_subplot()

ax = fig.add_subplot(projection='3d')
ax.plot_surface(R, T, X)
print()
plt.show()