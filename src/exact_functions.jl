module exact_functions
export overlap, operator_entropy, propagate_layerbylayer, circuit_TFIM, get_Zi

using LinearAlgebra

#------------ Overlap with a state psi ------------
function overlap(O::Matrix, ψ::Vector{Float64})::Float64
    """
    overlap(O::Matrix, ψ::Vector{Float64})::Float64

    Compute the expectation value $\langle \psi | \hat{O} | \psi \rangle$ of a dense matrix observable for a given state vector.

    ### Arguments

    * `O`: The matrix representation of the operator.
    * `ψ`: The state vector $\psi$.

    ### Returns

    * The real-valued expectation value as a `Float64`.

    ### Notes

    This function performs the matrix-vector multiplication $\langle \psi | \hat{O} | \psi \rangle$. It returns the real part of the result, which is appropriate for physical observables represented by Hermitian matrices.
    """
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
    ψ0::Union{Vector{Float64}, Nothing}=nothing,
    disable_print::Bool=false
    )

    t = time()
    nlayers = length(circuit)

    entropies, norms, overlaps = Float64[], Float64[], Float64[]

    if bond != nothing
      push!(entropies, operator_entropy(observable, bond))
    end
    if ψ0 != nothing
      push!(overlaps, overlap(observable, ψ0))
    end
    push!(norms, norm(observable))

    current = copy(observable)
    for (layer_idx, layer) in enumerate(reverse(circuit))
        for gate in reverse(layer)
          current = gate' * current * gate
        end
        if bond != nothing
          push!(entropies, operator_entropy(current, bond))
        end
        if ψ0 != nothing
          push!(overlaps, overlap(current, ψ0))
        end
        push!(norms, norm(current))

        if layer_idx % max(1, nlayers ÷ 10)==0 && !disable_print
            println("layer : $layer_idx /$nlayers complete")
        end
    end

    elapsed_time = time() - t
    println("Time taken by ext.propagate_layerbylayer: ", elapsed_time, " seconds")

    result = Dict("norm" => norms, "S" => entropies, "overlap" => overlaps, "time" => elapsed_time)

    return current, result
end

#------------ Observable for test ------------ 

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