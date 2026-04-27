module mpo_functions
export compute_matrix, overlap, operator_entropy, propagate_layerbylayer

using ITensors, ITensorMPS

#------------ MPO -> Matrix ------------
function compute_matrix(mpo::MPO, sites)
    nqubits = length(sites)

    Zitensor = ITensor(1.)
    for i = 1:nqubits
      Zitensor *= mpo[i]
    end

    row_indices = [sites[i]' for i in reverse(1:nqubits)]
    col_indices = [sites[i]  for i in reverse(1:nqubits)]
    
    tensor_array = Array(Zitensor, row_indices..., col_indices...)

    dim = 2^nqubits
    return reshape(tensor_array, dim, dim)
end

#------------ Overlap with a state psi ------------
function overlap(O::MPO, ψ::MPS)::Float64
  return real(inner(ψ', O, ψ))
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
function tensor_dag(U::ITensor)::ITensor
  indices = inds(U)
  N = length(indices) ÷ 2

  U_dag = U
  for i in 1:N 
    U_dag = swapind(U_dag, indices[i], indices[i+N])
  end
  return U_dag 
end

function propagate_layerbylayer(
    circuit::Vector{Vector{ITensor}},
    observable::MPO;
    cutoff::Float64=1e-8,
    maxdim::Int=200,
    bond::Union{Int, Nothing}=nothing,
    ψ0::Union{MPS, Nothing}=nothing
    )
  t0 = time()
  nlayers = length(circuit)
  
  maxlink, entropies, norms, overlaps = Int[], Float64[], Float64[], Float64[]

  current = copy(observable)
  heinseberg_circuit = [reverse(tensor_dag.(layer)) for layer in reverse(circuit)]
  
  sites_mps = [siteind(current, i; plev=0) for i in 1:length(current)]

  N = length(sites_mps)
  if bond === nothing 
    bond = N ÷ 2 +1 # L'entropie est mesurée au milieu de la chaine par defaut
  end
  
  if ψ0 === nothing
    init_state = ["Up" for _ in 1:N] # "Dn" pour down
    ψ0 = MPS(sites_mps, init_state)
  end

  norm0 = norm(observable)
  push!(maxlink, maxlinkdim(observable))
  push!(entropies, operator_entropy(observable, bond))
  push!(overlaps, overlap(observable, ψ0))
  push!(norms, norm0)

  for (layer_idx, layer) in enumerate(heinseberg_circuit)
    current = apply(layer, current; apply_dag=true, cutoff=cutoff, maxdim=maxdim) # apply(U,0,apply_dag=true) fait U O U+ donc on applique d'abord le dag a layer pour avoir +U O U
    norm_temp = norm(current)
    current *= (norm0/norm_temp) # pour conserver la norm malgres les troncations

    push!(maxlink, maxlinkdim(current))
    push!(entropies, operator_entropy(current, bond))
    push!(overlaps, overlap(current, ψ0))
    push!(norms, norm_temp)

    if layer_idx % max(1, nlayers ÷ 10)==0
        println("layer : $layer_idx /$nlayers complete")
    end
  end

  elapsed_time = time() - t0
  println("Time taken by mpo_functions.propagate_layerbylayer: ", elapsed_time, " seconds")

  result = Dict("maxlink" => maxlink, "S" => entropies, "norm" => norms, "overlap" => overlaps, "time"=> elapsed_time)

  return current, result
end


#  MPS(A,sites;cutoff=cutoff,maxdim=maxdim)

end # module