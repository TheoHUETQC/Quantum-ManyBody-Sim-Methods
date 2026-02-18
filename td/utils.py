import numpy as np
from scipy import sparse

import constants 

# Random stuf

def random_complexe_matrix(dimn : int, dimm : int) -> np.ndarray:
  re = np.random.rand(dimn,dimm) # partie rééle
  im = np.random.rand(dimn,dimm) # partie complexe
  return re + 1j*im

# Maths

def normalisation(v : np.ndarray) -> np.ndarray:
  return v/np.linalg.norm(v)

def trace(A : np.ndarray) -> float:
  return np.einsum("ii->",A)

def partial_trace_A(rho : np.ndarray) -> np.ndarray:
  return rho[0:2,0:2] + rho[2:4,2:4]

def dot(A : np.ndarray,B : np.ndarray) -> np.ndarray:
  return np.einsum("ik,kj->ij",A,B)

def norm2(v : np.ndarray) -> float:
  return np.einsum("ij,ij->",v.conj(), v).real

# phy

def ground_state(H : np.ndarray) -> tuple[np.ndarray, np.ndarray] : # calcul de l'énérgie et du vecteur propre associé au ground state
  eigenValues, eigenVector = sparse.linalg.eigsh(H, k=1, which="SA")  # k=1 cause we need one vector, "SA" for smallest eigenvalue
  return eigenValues, eigenVector/np.linalg.norm(eigenVector)
  """eigenValues, eigenVectors = np.linalg.eigh(H)
  i = np.argmin(eigenValues)
  eigenVector = eigenVectors[:,i]
  return eigenValues[i], eigenVector/np.linalg.norm(eigenVector)"""

def compute_rho(v : np.ndarray) -> np.ndarray:
  return np.outer(v,np.conjugate(v))
  
# SVD

def svd(A : np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] : # return U, s, Vh
  return np.linalg.svd(A, full_matrices=False)

def sparse_svd(A : np.ndarray, k: int) :
  return sparse.linalg.svds(A, k)

def mps_tensor(A : np.ndarray, B :np.ndarray) -> np.ndarray :
  return np.einsum('sk,kS->sS', A, B) #return psi

def mps_norm(A : np.ndarray, B :np.ndarray) -> float :
  return np.einsum('sk,kS,sl,lS->', A.conj(), B.conj(), A, B).real

# Measurement

def measurements_averageEnergy(H : np.ndarray, rho : np.ndarray) -> float : # <H>
  return float(np.trace(H @ rho))

def measurements_entropy(rho : np.ndarray) -> float: # S
  eigenValues = np.abs(np.linalg.eigvals(rho)) # this avoids numerical instability issues : log(0)
  eigenValues = eigenValues[eigenValues > 1e-12]
  return  float(-np.sum(eigenValues * np.log(eigenValues)))

def measurements_entropy_SVD(s : np.ndarray) -> float: # S en utilisant les valeurs singulière
  s2 = np.abs(s)**2
  s2 = s2[s2 > 0] # this avoids numerical instability issues : log(0)
  return  float(-np.sum(s2 * np.log(s2)))

def measurements_purity(rho : np.ndarray) -> float: # P
  return float(np.trace(rho @ rho))