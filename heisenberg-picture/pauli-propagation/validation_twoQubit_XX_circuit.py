import numpy as np
from scipy.linalg import expm

#------- parameter -------#

dt = 0.1
n_runs = 150

#------- init matrix -------#

I = np.array([[1, 0],
              [0, 1]], dtype=complex)

Z = np.array([[1, 0],
              [0, -1]], dtype=complex)

Y = np.array([[0, -1j],
              [1j, 0]], dtype=complex)

X = np.array([[0, 1],
              [1, 0]], dtype=complex)

#------- Analytic data -------#

Z1 = np.kron(Z, I)
H = np.kron(X, X)
U = expm(-1j * dt * H/2) # U = e^{i * dt * H /2}
U_dagger = U.conj().T # dagger

Z1_t_analytic = Z1.copy()
for _ in range(n_runs):
    Z1_t_analytic = U_dagger @ Z1_t_analytic @ U # Z1(t) = U^\dagger Z1(t-dt) U

#------- Pauli Propagation data -------#

pauli_propagation_result = {"5" : 0.47943 * np.kron(Y,X) +
                                  0.87758 * np.kron(Z,I),
                            "150":0.65029 * np.kron(Y,X) +
                                  -0.75969 * np.kron(Z,I)
                            }
Z1_t_paupi_propagation = pauli_propagation_result[str(n_runs)]

#------- result print -------#

print("Is the analytical calculation close to the Pauli propagation result? ", np.all(np.isclose(Z1_t_paupi_propagation, Z1_t_analytic)))
#- Is the analytical calculation close to the Pauli propagation result?  True

print("with an error of : ", np.linalg.norm(Z1_t_paupi_propagation - Z1_t_analytic))
#- with an error of :  6.007022392374302e-06