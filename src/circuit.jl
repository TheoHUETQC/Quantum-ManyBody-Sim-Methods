module circuit
export local_to_global_matrices, matrix_to_gate, mpo_MFIM_circuit, pp_MFIM_circuit, exact_circuit_MFIM, haar_unitary, random_circuit

using ITensors, ITensorMPS
using PauliPropagation
using LinearAlgebra

#------------ matrix 4x4 -> matrix 2^Nx2^N ------------
function local_to_global_matrices(U::Matrix, pair::Tuple{Int64, Int64}, nqubits::Int64)::Matrix{ComplexF64}
  raw"""
  local_to_global_matrices(U::Matrix, pair::Tuple{Int64, Int64}, nqubits::Int64)::Matrix{ComplexF64}

  Embeds a local two-qubit operator `U` into the global Hilbert space of a system containing `nqubits`.

  This function utilizes Kronecker products to extend a local operator (typically a $4 \\times 4$ matrix acting on two qubits) into a global operator of dimensions $2^{nqubits} \\times 2^{nqubits}$. It places the local operator `U` at the specified qubit indices, applying the identity matrix to all other sites.

  # Arguments
  - `U::Matrix{ComplexF64}`: The local operator (matrix) acting on the qubit pair.
  - `pair::Tuple{Int64, Int64}`: A tuple indicating the two consecutive target qubit indices (e.g., `(i, i+1)`). 
  - `nqubits::Int64`: The total number of qubits in the system.

  # Returns
  - `Matrix{ComplexF64}`: The resulting global operator matrix in the full Hilbert space.

  # Throws
  - `AssertionError`: If the indices in `pair` are not consecutive (i.e., if `j != i + 1`).
  """
  i,j = pair
  @assert j == i+1

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
function matrix_to_gate(U::Matrix, pair::Tuple{Int64, Int64}, nqubits::Int64, sites::Vector{<:Index})::Tuple{Gate, ITensor, Matrix{ComplexF64}}
  raw"""
  matrix_to_gate(U::Matrix, pair::Tuple{Int64, Int64}, nqubits::Int64, sites::Vector{<:Index})::Tuple{Gate, ITensor, Matrix{ComplexF64}}

  Converts a local $4 \\times 4$ unitary matrix into three distinct representations compatible with Pauli Propagation, MPO (ITensors), and Exact simulation methodologies.

  This utility function acts as a factory, preparing a two-qubit gate `U` for multiple simulation backends:
  - **Pauli Propagation**: Calculates the Pauli Transfer Matrix (PTM) and wraps it into a `TransferMapGate`.
  - **MPO**: Converts the dense matrix into an `ITensor` compatible with `ITensors.jl`, indexed by the provided `sites`.
  - **Exact**: Embeds the local operator into a global $2^{nqubits} \\times 2^{nqubits}$ matrix using Kronecker products.

  # Arguments
  - `U::Matrix{ComplexF64}`: The local $4 \\times 4$ unitary matrix to convert.
  - `pair::Tuple{Int64, Int64}`: The qubit indices on which the gate acts.
  - `nqubits::Int64`: The total number of qubits in the system.
  - `sites::Vector{<:Index}`: The ITensors `SiteSet` or vector of indices corresponding to the physical system.

  # Returns
  - `Tuple{Gate, ITensor, Matrix{ComplexF64}}`: A tuple containing the gate in `PauliPropagation` format, the MPO `ITensor` format, and the global dense `Matrix` format respectively.

  # Throws
  - `AssertionError`: If `U` is not unitary (i.e., $U U^\dagger \neq I$).
  """
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
function build_two_site_MFIM_hamiltonian(sites, j::Int64, N::Int64, g::Float64=0.5, h::Float64=0.5)::ITensor
  raw"""
  build_two_site_MFIM_hamiltonian(sites, j::Int64, N::Int64, g::Float64=0.5, h::Float64=0.5)::ITensor

  Constructs the local two-site Hamiltonian tensor for the Mixed Field Ising Model (MFIM) at bond `j` for an MPS-based simulation.

  The function calculates the interaction term $X_j X_{j+1}$ and distributes the local magnetic field terms (transverse field `g` and longitudinal field `h`) across the sites. To ensure the global Hamiltonian is correctly represented, fields are applied in full at the chain boundaries and split evenly (halved) between adjacent sites for internal bonds.

  # Arguments
  - `sites`: The site indices collection (typically an ITensors.jl `SiteSet`).
  - `j::Int64`: The current bond index representing the interaction between site `j` and `j+1`.
  - `N::Int64`: The total number of sites in the system.
  - `g::Float64=0.5`: The strength of the transverse magnetic field ($X$ term).
  - `h::Float64=0.5`: The longitudinal magnetic field strength ($Z$ term).

  # Returns
  - `ITensor`: The two-site operator representing the local Hamiltonian contribution for the bond `(j, j+1)`.
  """
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
function build_TEBD_gates_hj_MFIM(sites, τ::Float64, N::Int64, g::Float64=0.5, h::Float64=0.5)::Tuple{Vector{ITensor}, Vector{ITensor}}
  raw"""
  build_TEBD_gates_hj_MFIM(sites, τ::Float64, N::Int64, g::Float64=0.5, h::Float64=0.5)::Tuple{Vector{ITensor}, Vector{ITensor}}

  Prepares the unitary evolution gates (Trotter gates) for the Mixed Field Ising Model (MFIM) using the Time-Evolving Block Decimation (TEBD) method.

  This function iterates over the bonds of the lattice, constructs the local Hamiltonian terms using `build_two_site_MFIM_hamiltonian`, and computes the corresponding unitary gates $U = e^{-i \cdot \\tau \cdot H}$. The gates are partitioned into an "odd" set and an "even" set to facilitate the standard checkerboard decomposition required for TEBD evolution.

  # Arguments
  - `sites::ITensors.SiteSet`: The site indices collection corresponding to the physical system.
  - `τ::Float64`: The time step parameter for the evolution.
  - `N::Int64`: The total number of sites in the system.
  - `g::Float64=0.5`: The strength of the transverse magnetic field ($X$ term).
  - `h::Float64=0.5`: The strength of the longitudinal magnetic field ($Z$ term).

  # Returns
  - `Tuple{Vector{ITensor}, Vector{ITensor}}`: A tuple containing two vectors of gates (`gates_odd`, `gates_even`). Each element is an `ITensor` representing the unitary gate for a specific bond.
  """
  gates_odd = ITensor[]
  gates_even = ITensor[]
  for j in 1:(N-1)
      hj = build_two_site_MFIM_hamiltonian(sites, j, N, g, h)
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
  sites = ITensors.siteinds("S=1/2", Nqubits)
  gates_odd, gates_even = build_TEBD_gates_hj_MFIM(sites, τ, Nqubits, g, h)

  # On crée une seule liste ordonnée pour un pas de Trotter complet
  # Ordre : Odd (τ/2) -> Even (τ) -> Odd (τ/2)
  one_step_layer = vcat(gates_odd, gates_even, gates_odd)

  return [one_step_layer for _ in 1:nlayers], sites
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
function haar_unitary(n::Int64)::Matrix{ComplexF64}
  raw"""
  haar_unitary(n::Int64)::Matrix{ComplexF64}

  Generates a random unitary matrix sampled according to the Haar measure on the unitary group $U(n)$.

  This function implements the standard construction (Mezzadri algorithm): it generates a random complex Gaussian matrix, performs a QR decomposition, and applies a phase correction to the columns of $Q$ to ensure the resulting matrix is distributed uniformly according to the Haar measure.

  # Arguments
  - `n::Int64`: The dimension of the square matrix to be generated.

  # Returns
  - `Matrix{ComplexF64}`: A unitary matrix of size $n \\times n$ satisfying $U \\cdot U^\\dagger = I$.
  """
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
function random_circuit(
  nqubits::Integer, 
  nlayers::Integer; 
  topology::Union{Vector{Tuple{Int64, Int64}},Nothing}=nothing
  )::Tuple{Vector{Gate}, Vector{Vector{ITensor}}, Vector{Vector{Matrix}}, Vector{<:Index}}
  raw"""
  random_circuit(nqubits::Integer, nlayers::Integer; topology::Union{Vector{Tuple{Int64, Int64}},Nothing}=nothing)::Tuple{Vector{Gate}, Vector{Vector{ITensor}}, Vector{Vector{Matrix}}, Vector{<:Index}}

  Generates a random quantum circuit in three simultaneous representations: Pauli Propagation, MPO (ITensors), and Exact matrix multiplication.

  This function acts as a cross-validation tool. For each layer and topology connection, it generates a random $4 \\times 4$ Haar unitary gate and converts it into the three required formats. This allows for rigorous comparison and validation of different simulation methodologies (Pauli Propagation vs. MPO vs. Exact) using the exact same unitary transformations.

  # Arguments
  - `nqubits::Integer`: The total number of qubits in the system.
  - `nlayers::Integer`: The number of layers (depth) of the circuit.
  - `topology::Union{Vector{Tuple{Int64, Int64}}, Nothing}=nothing`: The connectivity graph for the gates. If `nothing`, a non-periodic bricklayer topology is used.

  # Returns
  - `Tuple{Vector{Gate}, Vector{Vector{ITensor}}, Vector{Vector{Matrix}}, Vector{<:Index}}`: A tuple containing:
      1. `Vector{Gate}`: The circuit formatted for `PauliPropagation.jl`.
      2. `Vector{Vector{ITensor}}`: The circuit formatted as layers of `ITensor`s for MPO simulations.
      3. `Vector{Vector{Matrix}}`: The circuit formatted as layers of global dense `Matrix` objects for exact simulation.
      4. `Vector{<:Index}`: The `SiteSet` indices used for the ITensor representation.
  """
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