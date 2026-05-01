module circuit
export local_to_global_matrices, matrix_to_gate, mpo_compute_MFIM_circuit, pp_TFIM_circuit, haar_unitary, random_circuit

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
function matrix_to_gate(U::Matrix, pair::Tuple{Int64, Int64}, nqubits::Int64, sites)
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

# ======================  MFIM Circuit ======================  

# ------------ Hamiltonien local h_{j,j+1} ------------
function build_two_site_MFIM_hamiltonian(j::Int64, N::Int64, g::Float64=0.5, h::Float64=0.5)
  s1 = sites[j]
  s2 = sites[j + 1]

  # Terme d'interaction ZZ
  hj = op("X", s1) * op("X", s2)

  # Gestion du champ pour le site de gauche (s1)
  # Si on est au bord gauche (j=1), on met le champ complet, sinon la moitié
  g1 = (j == 1) ? g : g/2
  h1 = (j == 1) ? h : h/2
  hj += g1 * (op("X", s1) * op("Id", s2)) + h1 * (op("Z", s1) * op("Id", s2))

  # Gestion du champ pour le site de droite (s2)
  # Si on est au bord droit (j=N-1), on met le champ complet, sinon la moitié
  g2 = (j == N - 1) ? g : g/2
  h2 = (j == N - 1) ? h : h/2
  hj += g2 * (op("Id", s1) * op("X", s2)) + h2 * (op("Id", s1) * op("Z", s2))

  return hj
end

# ------------ Construction des gates TEBD ------------
function build_TEBD_gates_hj_MFIM(τ::Float64, N::Int64, g::Float64=0.5, h::Float64=0.5)
  gates_odd = ITensor[]
  gates_even = ITensor[]
  for j in 1:(N-1)
      hj = build_two_site_MFIM_hamiltonian(j, N, g, h)
      if isodd(j)
        Gj = exp(-im * τ / 2 * hj)
        push!(gates_odd, Gj)
      else
        Gj = exp(-im * τ * hj)
        push!(gates_even, Gj)
      end
  end
  return gates_odd, gates_even
end

# ------------ MFIM Circuit for mpo ------------
function mpo_compute_MFIM_circuit(Nqubits::Int64, nlayers::Int64, τ::Float64, g::Float64=0.5, h::Float64=0.5)
  gates_odd, gates_even = build_TEBD_gates_hj_MFIM(τ, Nqubits, g, h)

  # On crée une seule liste ordonnée pour un pas de Trotter complet
  # Ordre : Odd (τ/2) -> Even (τ) -> Odd (τ/2)
  one_step_layer = vcat(gates_odd, gates_even, gates_odd)

  return [one_step_layer for _ in 1:nlayers]
end

# ======================  TFIM Circuit ====================== 

# ------------ TFIM Circuit for Pauli propagation ------------
function pp_TFIM_circuit(nqubits::Integer, nlayers::Integer; topology=nothing)
    circuit::Vector{Gate} = []

    if isnothing(topology)
      topology = bricklayertopology(nqubits; periodic=false)
    end

    for _ in 1:nlayers
      rxxlayer!(circuit, topology)
    end
    return circuit
end

# ------------ TFIM Circuit for exact method ------------
function exact_circuit_TFIM(nqubits::Int64, dt::Float64, nlayers::Int64)
    """
    H = ∑XᵢXⱼ
    """
    Id = ComplexF64[1 0; 0 1]
    X  = ComplexF64[0 1; 1 0]

    topology = [(i, i+1) for i in 1:(nqubits-1)]

    dim = 2^nqubits
    H = zeros(ComplexF64, dim, dim)

    for (qubit_i, qubit_j) in topology

        term = [1.0 + 0.0im;;] 
        
        for k in 1:nqubits
            if k == qubit_i || k == qubit_j
                term = kron(term, X)
            else
                term = kron(term, Id)
            end
        end
        H += term
    end

    U = exp(-1im * dt * H / 2)

    layer = [U]
    circuit_exact = Vector{Vector{Matrix}}()
    
    for _ in 1:nlayers
        push!(circuit_exact, layer)
    end

    return circuit_exact
end

# ======================  Random Circuit ======================  

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
    sites = ITensors.siteinds("Qubit", nqubits)
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
          U_pp, U_mpo, U_exact = matrix_to_gate(U, pair, nqubits, sites)
          push!(circuit_pp, U_pp)
          push!(layer_mpo, U_mpo)
          push!(layer_exact, U_exact)
      end
      push!(circuit_mpo, layer_mpo)
      push!(circuit_exact, layer_exact)
    end
    return circuit_pp, circuit_mpo, circuit_exact, sites
end

end # module