module exact_functions
export overlap, operator_entropy, propagate_layerbylayer

using LinearAlgebra

#------------ Overlap with a state psi ------------
function overlap(O::Matrix, ψ::Vector{Float64})::Float64
    return real(ψ' * O * ψ)
end

#------------ Operator Entropy ------------
function operator_entropy(O::Matrix, bond::Int)::Float64
    if bond == 1 # on ne peut pas couper au bond 1 (pas de lien à gauche)
        return 0.0
    end
    return 1.
end

#------------ Propagate Layer by layer ------------ 
function propagate_layerbylayer(    
    circuit::Vector{Vector{Matrix}},
    observable::Matrix;
    bond::Union{Int, Nothing}=nothing,
    ψ0::Union{Vector{Float64}, Nothing}=nothing)

    t = time()
    dim = sqrt(length(observable))
    nlayers = length(circuit)

    entropies, norms, overlaps = Float64[], Float64[], Float64[]

    if bond === nothing
      N = Int(log(dim) ÷ log(2)) # dim = 2^N
      bond = N ÷ 2 # L'entropie est mesurée au milieu de la chaine par defaut
    end
    
    if ψ0 === nothing
        ψ0 = append!([1],[0 for _ in 2:dim]) # |0> state
    end

    norm0 = norm(observable)
    push!(entropies, operator_entropy(observable, bond))
    push!(overlaps, overlap(observable, ψ0))
    push!(norms, norm0)

    current = copy(observable)
    for (layer_idx, layer) in enumerate(reverse(circuit))
        for gate in layer
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

end # module