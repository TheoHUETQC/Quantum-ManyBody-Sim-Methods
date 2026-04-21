module exact_functions
export overlap, operator_entropy, propagate_layerbylayer, circuit_TFIM, get_Zi

using LinearAlgebra

#------------ Overlap with a state psi ------------
function overlap(O::Matrix, ψ::Vector{Float64})::Float64
    return real(ψ' * O * ψ)
end

#------------ Operator Entropy ------------
function operator_entropy(O::Matrix, bond::Int)::Float64
    dim_total = size(O, 1)
    nqubits = Int(log2(dim_total))
    
    if bond < 1 || bond >= nqubits # on ne peut pas couper au bond 1 (pas de lien à gauche)
        return 0.0
    end

    # dL = dimension à gauche, dR = dimension à droite
    dL = 2^bond
    dR = 2^(nqubits - bond)

    O_tensor = reshape(O, dR, dL, dR, dL)
    O_perm = permutedims(O_tensor, (2, 4, 1, 3))
    M_schmidt = reshape(O_perm, dL^2, dR^2)
    s = svdvals(M_schmidt)

    prob = s.^2 / sum(s.^2)
    entropy = -sum(p * log(p + 1e-15) for p in prob)

    return entropy
end

#------------ Propagate Layer by layer ------------ 
function propagate_layerbylayer(    
    circuit::Vector{Vector{Matrix}},
    observable::Matrix;
    bond::Union{Int, Nothing}=nothing,
    ψ0::Union{Vector{Float64}, Nothing}=nothing)

    t = time()
    dim = size(observable, 1)
    nlayers = length(circuit)

    entropies, norms, overlaps = Float64[], Float64[], Float64[]

    if bond === nothing
      N = Int(log2(dim)) # dim = 2^N
      bond = N ÷ 2 +1 # L'entropie est mesurée au milieu de la chaine par defaut
    end
    
    if ψ0 === nothing
        ψ0 = append!([1.],[0. for _ in 2:dim]) # |0> state
    end

    norm0 = norm(observable)
    push!(entropies, operator_entropy(observable, bond))
    push!(overlaps, overlap(observable, ψ0))
    push!(norms, norm0)

    current = copy(observable)
    for (layer_idx, layer) in enumerate(reverse(circuit))
        for gate in reverse(layer)
            current = gate' * current * gate
        end
        push!(entropies, operator_entropy(current, bond))
        push!(overlaps, overlap(current, ψ0))
        push!(norms, norm(current))

        if layer_idx % max(1, nlayers ÷ 10)==0
            println("layer : $layer_idx /$nlayers complete")
        end
    end

    result = Dict("S" => entropies, "norm" => norms, "overlap" => overlaps)

    elapsed_time = time() - t
    println("Time taken by ext.propagate_layerbylayer: ", elapsed_time, " seconds")

    return current, result
end

#------------ Circuit and observable for test ------------ 
function circuit_TFIM(nqubits::Int64, dt::Float64, nlayers::Int64)
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

function get_Zi(nqubits::Int64, i::Int64)
    """
    Zᵢ = I..IZI..I with Z at the index i
    """
    Id = ComplexF64[1 0; 0 1]
    Z  = ComplexF64[1 0; 0 -1]

    obs = [1.0 + 0.0im;;] 

    for k in 1:nqubits
        if k == i
            obs = kron(obs, Z)
        else
            obs = kron(obs, Id)
        end
    end
    
    return obs
end

end # module