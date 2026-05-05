module mpo_functions
export compute_matrix, overlap, operator_entropy, applynoiselayer, propagate_layerbylayer, find_truncations

using ITensors, ITensorMPS

#------------ MPO -> Matrix ------------
# need to fix this function
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

#------------ Noise layer ------------
function apply_depolarizing_noise(O::MPO, sites_to_noise::Vector{<:Index}, lambda::Float64; cutoff::Float64=1e-8, maxdim::Int=200)
    O_noisy = copy(O)
    
    c0 = 1.0 - (3.0 * lambda / 4.0)
    c1 = lambda / 4.0
    
    # On applique le bruit site par site
    for s in sites_to_noise
        X = op("X", s)
        Y = op("Y", s)
        Z = op("Z", s)
        
        # apply_dag=true fait X*O*X, Y*O*Y, Z*O*Z car Pauli est hermitien
        O_X = apply([X], O_noisy; apply_dag=true, cutoff=cutoff)
        O_Y = apply([Y], O_noisy; apply_dag=true, cutoff=cutoff)
        O_Z = apply([Z], O_noisy; apply_dag=true, cutoff=cutoff)
        
        # +(A, B; kwargs...) fait A+B avec truncation
        O_new = +(c0 * O_noisy, c1 * O_X; cutoff=cutoff, maxdim=maxdim)
        O_new = +(O_new, c1 * O_Y; cutoff=cutoff, maxdim=maxdim)
        O_new = +(O_new, c1 * O_Z; cutoff=cutoff, maxdim=maxdim)
        
        O_noisy = O_new
    end
    return O_noisy
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
    bond::Union{Int, Nothing}=nothing, # for the Entropy
    ψ0::Union{MPS, Nothing}=nothing, # for the Overlap
    γ::Float64=0., # for the Noise
    disable_print::Bool=false
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
    if !(γ == 0.)
	    current = apply_depolarizing_noise(current, sites_mps, γ; cutoff, maxdim)
    end
    current *= (norm0/norm(current)) # pour conserver la norm malgres les troncations

    push!(maxlink, maxlinkdim(current))
    push!(entropies, operator_entropy(current, bond))
    push!(overlaps, overlap(current, ψ0))
    push!(norms, norm(current))

    if layer_idx % max(1, nlayers ÷ 10)==0 && !disable_print
        println("layer : $layer_idx /$nlayers complete")
    end
  end

  elapsed_time = time() - t0
  println("Time taken by mpo_functions.propagate_layerbylayer: ", elapsed_time, " seconds")

  result = Dict("maxlink" => maxlink, "S" => entropies, "norm" => norms, "overlap" => overlaps, "time"=> elapsed_time)

  return current, result
end

#------------ Find optimal truncations ------------ 
function find_truncations(
  tolerance::Float64, 
  circuit::Vector{Vector{ITensor}}, 
  observable::MPO,;
  bond::Union{Int, Nothing}=nothing, # for the Entropy
  ψ0::Union{MPS, Nothing}=nothing, # for the Overlap
  γ::Float64=0. # for the Noise
  )::Tuple{Int64, Float64}

  dim = 2^length(observable)
  nlayers = length(circuit)

  smaller_cutoff = 1e-10
  
  println("----- Max Dim TEST -----")
  maxdim = 8
  
  overlap_before = fill(Inf, (nlayers+1))
  is_close = false
  while !is_close
    println("--- Max weight = $maxdim, Min abs coeff = $smaller_cutoff ---")
    mpo, result =  propagate_layerbylayer(circuit, observable; cutoff=smaller_cutoff, maxdim, bond, ψ0, γ, disable_print=true)
    overlap = result["overlap"]
    #entropy = result["S"]
    isclose_overlap = isapprox(overlap, overlap_before; rtol=tolerance)
    #isclose_matrix = isapprox(entropy, entropy_before; rtol=tolerance)
    #isclose_entropy = isapprox(mpo, mpo_before; rtol=tolerance)

    is_close = isclose_overlap #&& isclose_matrix && isclose_entropy
    if is_close
      break
    end

    overlap_before = overlap
    maxdim += 3

    if maxdim > dim
      maxdim = dim
      break
    end
  end

  println("----- Cutoff TEST -----")
  cutoff_power = -2

  overlap_before = fill(Inf, (nlayers+1))
  is_close = false
  while !is_close
    println("--- Max weight = $maxdim, Min abs coeff = 1e$cutoff_power ---")
    cutoff = 10^float(cutoff_power)
    mpo, result =  propagate_layerbylayer(circuit, observable; cutoff, maxdim, bond, ψ0, γ, disable_print=true)
    overlap = result["overlap"]
    #entropy = result["S"]
    isclose_overlap = isapprox(overlap, overlap_before; rtol=tolerance)
    #isclose_matrix = isapprox(entropy, entropy_before; rtol=tolerance)
    #isclose_entropy = isapprox(mpo, mpo_before; rtol=tolerance)

    is_close = isclose_overlap #&& isclose_matrix && isclose_entropy
    if is_close
      break
    end

    overlap_before = overlap
    cutoff_power -= 2

    if cutoff <= smaller_cutoff
      return maxdim, smaller_cutoff
    end
  end
  cutoff = 10^float(cutoff_power)
  return maxdim, cutoff
end

#  MPS(A,sites;cutoff=cutoff,maxdim=maxdim)

end # module