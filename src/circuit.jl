module circuit
export local_to_global_matrices, matrix_to_gate, haar_unitary, random_circuit, applynoiselayer

using ITensors, ITensorMPS
using PauliPropagation
using LinearAlgebra

#------------ matrix 4x4 -> matrix 2^Nx2^N ------------
function local_to_global_matrices(U::Matrix, pair::Tuple{Int64, Int64}, nqubits::Int64)
  i,j = pair
  U_global = [1.0 + 0.0im;;]
  k = 1
  while k <= nqubits
      if k == i
          # Si on arrive à l'indice de la paire on applique la porte U (4x4)
          U_global = kron(U_global, U)
          k += 2  # On saute le qubit suivant puisqu'il est inclus dans U
      else
          U_global = kron(U_global, I(2))
          k += 1
      end
  end
  return U_global
end

#------------ matrix 4x4 -> gate for 3 methods ------------
function matrix_to_gate(U::Matrix, pair::Tuple{Int64, Int64}, nqubits::Int64)
  # need to be 4x4 and unitary
  @assert U * U' ≈ U' * U ≈ I(4)

  # --- Pauli Propagation ---
  U_ptm = calculateptm(U)
  U_pp = TransferMapGate(U_ptm, pair)

  # --- MPO ---
  i, j = pair
  s1, s2 = sites[i], sites[j]
  U_mpo = itensor(U, s2', s1', s2, s1) # itensor(U, s1', s2', s1, s2)

  # --- Exact ---
  U_exact = local_to_global_matrices(U, pair, nqubits)

  return U_pp, U_mpo, U_exact
end

#------------ Random unitary matrix ------------
function haar_unitary(n::Int64)
    # 1. Generation of Z
    Z = randn(ComplexF64, n, n)

    # 2. QR Decomposition
    F = qr(Z)
    Q = Matrix(F.Q)
    R = F.R

    # 3. Phase correction
    d = diag(R)
    ph = d ./ abs.(d)

    U = Q * Diagonal(ph)
    return U
end

#------------ Random Circuit for 3 methods ------------
function random_circuit(nqubits::Integer, nlayers::Integer; topology=nothing)
    circuit_pp::Vector{Gate} = []
    circuit_mpo::Vector{Vector{ITensor}} = []
    circuit_exact::Vector{Vector{Matrix}} = []

    if isnothing(topology)
        topology = bricklayertopology(nqubits; periodic=false)
    end

    for _ in 1:nlayers
      layer_mpo::Vector{ITensor} = []
      layer_exact::Vector{Matrix} = []
      for pair in topology
          U = haar_unitary(4)
          # U = [0 0 0 1; 0 0 1 0; 0 1 0 0; 1 0 0 0] # XX gate for test
          U_pp, U_mpo, U_exact = matrix_to_gate(U, pair, nqubits)
          push!(circuit_pp, U_pp)
          push!(layer_mpo, U_mpo)
          push!(layer_exact, U_exact)
      end
      push!(circuit_mpo, layer_mpo)
      push!(circuit_exact, layer_exact)
    end
    return circuit_pp, circuit_mpo, circuit_exact
end

#------------ Noise layer ------------
function applynoiselayer(psum::PauliSum;depol_strength=0.02, dephase_strength=0.02, noise_level=1.0)
    for (pstr, coeff) in psum
        set!(psum, pstr,
            coeff*(1-noise_level*depol_strength)^countweight(pstr)*(1-noise_level*dephase_strength)^countxy(pstr))
    end
end

end # module