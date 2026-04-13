module mpo_functions
export overlap, operator_entropy, propagate_layerbylayer

using ITensors, ITensorMPS

#------------ Overlap with a state psi ------------
function overlap(O::MPO, ψ::MPS)::Float64
  return real(inner(ψ, O, ψ))
end

#------------ Operator Entropy ------------
function compute_entropy(s)::Float64
  p = s.^2 / sum(s.^2)
  p = p[p .> 1e-18]
  return -sum(p .* log.(p))
end

function operator_entropy(O::MPO, bond::Int)::Float64
    if bond == 1 # on ne peut pas couper au bond 1 (pas de lien à gauche)
        return 0.0
    end
    
    O_ortho = orthogonalize(O, bond)
  
    l_link = linkind(O_ortho, bond-1)
    s_inds = siteinds(O_ortho, bond)
    
    # SVD, on sépare le lien gauche et les indices physiques
    U, S, V = svd(O_ortho[bond], l_link, s_inds...)
    
    s_vals = diag(S)
    return compute_entropy(s_vals)
end

#------------ Propagate Layer by layer ------------ 
function propagate_layerbylayer(
    circuit::Vector{Vector{ITensor}},
    observable::MPO;
    cutoff::Float64=1e-8,
    bond::Union{Int, Nothing}=nothing,
    ψ0::Union{MPS, Nothing}=nothing
    )
  t0 = time()
  nlayers = length(circuit)
  
  maxbonds, entropies, norms, overlaps = Int[], Float64[], Float64[], Float64[]

  current = copy(observable)
  adjoint_circuit = [dag.(layer) for layer in circuit]
  
  sites_mps = siteinds(first, current)

  N = length(sites_mps)
  if bond === nothing 
    bond = N ÷ 2 # L'entropie est mesurée au milieu de la chaine par defaut
  end

  if ψ0 === nothing
    init_state = ["Up" for _ in 1:N] # "Dn" pour down
    ψ0 = MPS(sites_mps, init_state)
  end

  norm0 = norm(observable)  
  push!(overlaps, overlap(observable, ψ0))
  for (layer_idx, layer) in enumerate(adjoint_circuit)
    current = apply(layer, current; apply_dag=true, cutoff=cutoff) # apply(U,0,apply_dag=true) fait U O U+ donc on applique d'abord le dag a layer pour avoir +U O U
    
    maxbond_temp = maxlinkdim(current)

    push!(maxbonds, maxbond_temp)
    push!(entropies, operator_entropy(current, bond))
    push!(overlaps, overlap(current, ψ0))

    if layer_idx % (nlayers ÷ 10)==0
        norm_temp = norm(current)
        push!(norms, norm_temp)

        if !isapprox(norm_temp, norm0; rtol=1e-3) # on verifie que la norme ne diverge pas trop
          println("layer : $layer_idx /$nlayers Break cause norm = $norm_temp ≠ $norm0")
          break
        end

        println("layer : $layer_idx /$nlayers complete")
    end
  end

  elapsed_time = time() - t0
  println("Time taken by mpo_functions.propagate_layerbylayer: ", elapsed_time, " seconds")

  result = Dict("maxbond" => maxbonds, "S" => entropies, "norm" => norms, "overlap" => overlaps)

  return current, result
end

end # module