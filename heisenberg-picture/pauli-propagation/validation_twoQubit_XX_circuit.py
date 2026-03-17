import numpy as np
from scipy.linalg import expm

# parameter
dt = 0.1
n_runs = 5

# init matrix
I = np.array([[1, 0], 
              [0, 1]], dtype=complex)

Z = np.array([[1, 0], 
              [0, -1]], dtype=complex)

Y = np.array([[0, -1j], 
              [1j, 0]], dtype=complex)

X = np.array([[0, 1], 
              [1, 0]], dtype=complex)

Z1 = np.kron(Z, I)
H = np.kron(X, X)

U = expm(-1j * dt * H/2) # U = e^{i * dt * H /2}
U_dagger = U.conj().T # dagger


Z1_t = Z1.copy()
for _ in range(n_runs):
    Z1_t = U_dagger @ Z1_t @ U # Z1(t) = U^\dagger Z1(t-dt) U

pauli_propagation_result = 0.47943 * np.kron(Y,X) + 0.87758 * np.kron(Z,I)

print("Is the analytical calculation close to the Pauli propagation result? ", np.isclose(pauli_propagation_result, Z1_t))
print("with an error of : ", np.linalg.norm(pauli_propagation_result - Z1_t))