import numpy as np
from scipy import sparse

import constants 

# Random stuf

def random_complexe_matrix(dimn : int, dimm : int):
  re = np.random.rand(dimn,dimm) # partie rééle
  im = np.random.rand(dimn,dimm) # partie complexe
  return re + 1j*im

# Maths

def normalisation(v : np.array) :
  return v/np.linalg.norm(v)

def trace(A : np.array) :
  return np.einsum("ii->",A)

def partial_trace_A(rho : np.array) :
  return rho[0:2,0:2] + rho[2:4,2:4]

def dot(A : np.array,B : np.array) :
  return np.einsum("ik,kj->ij",A,B)

def norm2(v : np.array) :
  return np.einsum("ij,ij->",v.conj(), v).real

# phy

def ground_state(H : np.array) : # calcul de l'énérgie et du vecteur propre associé au ground state
  eigenValues, eigenVector = sparse.linalg.eigsh(H, k=1, which="SA")  # k=1 cause we need one vector, "SA" for smallest eigenvalue
  return eigenValues, eigenVector/np.linalg.norm(eigenVector)
  """eigenValues, eigenVectors = np.linalg.eigh(H)
  i = np.argmin(eigenValues)
  eigenVector = eigenVectors[:,i]
  return eigenValues[i], eigenVector/np.linalg.norm(eigenVector)"""

def compute_rho(v : np.array) :
  return np.outer(v,np.conjugate(v))
  
# SVD

def svd(A : np.array) : # return U, s, Vh
  return np.linalg.svd(A, full_matrices=False)

def sparse_svd(A : np.array, k: int) :
  return sparse.linalg.svds(A, k)

def mps_tensor(A : np.array, B :np.array):
  return np.einsum('sk,kS->sS', A, B) #return psi

def mps_norm(A : np.array, B :np.array):
  return np.einsum('sk,kS,sl,lS->', A.conj(), B.conj(), A, B).real

# Measurement

def measurements_averageEnergy(H : np.array, rho : np.array) : # <H>
  return np.trace(H @ rho)

def measurements_entropy(rho : np.array) : # S
  eigenValues = np.abs(np.linalg.eigvals(rho)) # this avoids numerical instability issues : log(0)
  eigenValues = eigenValues[eigenValues > 1e-12]
  return  -np.sum(eigenValues * np.log(eigenValues))

def measurements_entropy_SVD(s : np.array) : # S en utilisant les valeurs singulière
  s2 = np.abs(s)**2
  s2 = s2[s2 > 0] # this avoids numerical instability issues : log(0)
  return  float(-np.sum(s2 * np.log(s2)))

def measurements_purity(rho : np.array) : # P
  return np.trace(rho @ rho)