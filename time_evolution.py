import numpy as np

def expanded_timeop(hamiltonian: np.ndarray, t: float) -> np.ndarray:
    """
    Returns expaned time evolution operator
    """
    
    return expandbasis(timeop(hamiltonian, t))

def timeop(hamiltonian: np.ndarray, t: float) -> np.ndarray:
    """time evolution operator
    
    Args:
        hamiltonian: hamiltonian of the cavity array in the cavity-emitter basis
        t: time
    Returns the time evolution operator in the cavity-emitter basis
    """
    
    eig_vals, eig_vecs = np.linalg.eig(hamiltonian)
    #time evolution operator in the diagonal basis
    U = np.exp(-1j*t*eig_vals)
    U = np.diag(U)
    #change back to cavity-emitter basis
    if np.allclose(hamiltonian, np.transpose(np.conj(hamiltonian))): #if H is hermitian
        U = eig_vecs @ U @ eig_vecs.T
    else: #must find inverse
        U = eig_vecs @ U @ np.linalg.inv(eig_vecs)
    return U

def expandbasis(a: np.ndarray) -> np.ndarray:
    """Expands 'a' to 2^n or 'qubit' basis for single photon.
    Basis vectors from c1 X e1 X e2 X...X c2 X e1 X...  i.e. the 'qubit' basis
    where X is kronecker product
    
    Args:
        a: square matrix to be expanded 
    Returns:
        a in the expanded qubit basis
    """
    
    #assert square matrix
    assert a.shape[0] == a.shape[1], "matrix is not square"
    
    n = a.shape[0]
    #dimension of expanded basis
    dim = 2**n
    
    #fill in expanded matrix
    a_expanded = np.eye(dim, dtype='complex')
    for i in range(n):
        row = 1 << (n-1-i)
        for j in range(n):
            col = 1 << (n-1-j)
            a_expanded[row][col] = a[i][j]
    
    return a_expanded
