import numpy as np 
import matplotlib.pyplot as plt
from scipy.sparse import *
from scipy.sparse.linalg import *
from matplotlib import cm

def Tmatrix(nx,ny):
    dx = 1./(nx+1)
    dy = 1./(ny+1)
    print(dx)
    main_diag  = (-2.0 * (1.0 / dx**2 + 1.0/ dy**2)) * np.ones(nx)
    upper_diag = (1.0 / dx**2) * np.ones(nx-1)
    lower_diag = (1.0 / dy**2) * np.ones(nx-1)
    return diags([lower_diag, main_diag, upper_diag], [-1,0,1])
    
def create_sparse_block_tridiagonal_matrix(nx, ny):
    """
    Create a sparse block tridiagonal matrix with given I, D matrices and size.
    
    Parameters:
    I (ndarray): Identity matrix (block).
    D (ndarray): Square matrix (block).
    size (int): Number of block rows (and columns) in the tridiagonal matrix.
    
    Returns:
    scipy.sparse.csr_matrix: Sparse block tridiagonal matrix.
    """
    # Convert I and D to sparse matrices
    I_sparse = np.eye(nx)
    D_sparse = Tmatrix(nx, ny)
    
    dx = 1./(nx+1)
    dy = 1./(ny+1)
    # Create lists to hold the blocks
    blocks = [[None]*ny for _ in range(ny)]
    
    # Fill the diagonal and off-diagonal blocks
    for i in range(ny):
        blocks[i][i] = D_sparse  # Main diagonal
        if i > 0:
            blocks[i][i-1] = (1.0 / dx**2) * I_sparse  # Lower diagonal
        if i < ny - 1:
            blocks[i][i+1] = (1.0 / dy**2) * I_sparse  # Upper diagonal
    
    # Create the block matrix using scipy.sparse.bmat
    tridiagonal_matrix = bmat(blocks, format='csr')
    
    return tridiagonal_matrix

def heat_source(x,y):
    xs = 1.0 / 2.0 
    ys = 1.0 / 2.0
    w = 0.2
    return np.exp(-((x-xs)**2+(y-ys)**2)/w**2)
nx = 3
ny = 4

sparse_block_tridiagonal_matrix = create_sparse_block_tridiagonal_matrix(nx, ny)

print(sparse_block_tridiagonal_matrix.toarray())

    
