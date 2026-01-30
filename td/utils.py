import numpy as np
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
  eigenValues, eigenVectors = np.linalg.eigh(H)
  i = np.argwhere(eigenValues == np.min(eigenValues))[0][0]
  eigenVector = eigenVectors[:,i]
  return eigenValues[i], eigenVector/np.linalg.norm(eigenVector)

def compute_rho(v : np.array) :
  return np.outer(v,np.conjugate(v))
  
# SVD

def svd(A : np.array) :
  return np.linalg.svd(A, full_matrices=False) # return U, s, Vh

def mps_tensor(A : np.array, B :np.array):
  return np.einsum('sk,kS->sS', A, B) #return psi

def mps_norm(A : np.array, B :np.array):
  return np.einsum('sk,kS,sl,lS->', A.conj(), B.conj(), A, B).real

# Measurement

def measurements_averageEnergy(H : np.array, rho : np.array) : # <H>
  return np.trace(H @ rho)

def measurements_entropy(rho : np.array) : # S
  eigenValues = np.abs(np.linalg.eigvals(rho)) + constants.epsilon  # with epsilon = 1e-8, this avoids numerical instability issues
  return  -np.sum(eigenValues * np.log(eigenValues))

def measurements_entropy_SVD(s : np.array) : # S en utilisant les valeurs singulière
  s2 = np.abs(s)**2 + constants.epsilon  # with epsilon = 1e-8, this avoids numerical instability issues
  return  -np.sum(s2 * np.log(s2))

def measurements_purity(rho : np.array) : # P
  return np.trace(rho @ rho)

