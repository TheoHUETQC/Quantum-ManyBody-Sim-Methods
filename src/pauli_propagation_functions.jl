module pauli_propagation_functions
export decode_pauli, compute_matrix, overlap, pauli_norm, shannon_entropy, renyi_entropy, applynoiselayer, propagate_layerbylayer

using PauliPropagation
using PauliPropagation: Xmat, Ymat, Zmat
using LinearAlgebra

#------------ Decode Pauli string ------------
function decode_pauli(pauli_string, num_qubits::Int)
    mapping = Dict(0b00 => "I", 0b01 => "X", 0b10 => "Y", 0b11 => "Z")
    
    res = ""
    for i in 0:(num_qubits - 1)
        bits = (pauli_string >> (2 * i)) & 0b11
        res *= mapping[bits]
    end
    return res
end

#------------ PauliSum -> Matrix ------------
#countweight(pstr)
function compute_matrix(observable::Union{PauliSum, PauliString})
  if typeof(observable) <: PauliString{<:Unsigned, Float64}
    observable = PauliSum(observable)
  end
  nqubits = observable.nqubits
  mapping = Dict('I' => I(2), 'X' => Xmat, 'Y' => Ymat, 'Z' => Zmat)

  result = zeros(ComplexF64, 2^nqubits, 2^nqubits)
  for (pauli_string, coeff) in observable
      string = decode_pauli(pauli_string, nqubits)
      result_string = [1.0 + 0.0im;;]
      for op in string
        result_string = kron(result_string, mapping[op])
      end
      result += coeff * result_string
  end
  return result
end

#------------ overlap ------------ 
function overlap(observable::Union{PauliSum, PauliString}, ψ)::Float64 
  if typeof(observable) <: PauliString{<:Unsigned, Float64}
    observable = PauliSum(observable)
  end

  nqubits = observable.nqubits
  mapping = Dict('I' => I(nqubits), 'X' => Xmat, 'Y' => Ymat, 'Z' => Zmat)
  
  state0 = append!([1],[0 for _ in 2:(2^nqubits)])
  if ψ == state0
    return overlapwithzero(observable)
  end 

  result = 0.0
  for (pauli_string, coeff) in observable
      string = decode_pauli(pauli_string, nqubits)
      result_string = [1.0 + 0.0im;;]
      for op in string
        result_string = kron(result_string, mapping[op])
      end
      result += coeff * ψ' * result_string * ψ
  end
  return result
end

#------------ Pauli Norm ------------ 
function pauli_norm(observable::Union{PauliSum, PauliString})::Float64
    if typeof(observable) <: PauliString{<:Unsigned, Float64}
      observable = PauliSum(observable)
    end
    return sum(((P, c),) -> abs(c)^2, observable; init=0.0)
end

#------------ Pauli Entropy ------------ 
function shannon_entropy(observable::Union{PauliSum, PauliString})::Float64
  if typeof(observable) <: PauliString{<:Unsigned, Float64}
    observable = PauliSum(observable)
  end
  return sum(((P, c),) -> c != 0 ? -abs(c)^2 * log(abs(c)^2) : 0.0, observable; init=0.0)
end

function renyi_entropy(observable::Union{PauliSum, PauliString{}}; k::Int64=2)::Float64
  """
  renyi_entropy(observable, k=2)

  Calculate the Rényi entropy of order `k` for the Pauli weight distribution of an `observable`.

  This metric quantifies the "spread" or complexity of an operator in the Pauli basis. 
  An entropy of 0 indicates that the observable is a single Pauli string, while higher 
  values indicate that the observable is fragmented into many terms.

  # Arguments
  - `observable::Union{PauliSum, PauliString}`: The operator to analyze.
  - `k::Real`: The order of the Rényi entropy (default is 2).

  # Returns
  - `Float64`: The entropy value M_k = frac{1}{1-k} log sum_P pi(P)^k, where pi(P) is the normalized weight of each Pauli string.
  """
  if typeof(observable) <: PauliString{<:Unsigned, Float64}
    observable = PauliSum(observable)
  end

  if k == 1 # L'entropie de Shannon (k=1) est la limite de Rényi
    return shannon_entropy(observable)
  end

  norm_sq = pauli_norm(observable) # compute \sum_P |c_P|²  
  if norm_sq == 0
    return 0.
  end

  sum_coeffs_2k = sum(p -> abs(p.second)^(2*k), observable; init=0.0) # p = (P, c_P)
  zeta_k = sum_coeffs_2k / (norm_sq^k)

  return log(zeta_k) / (1-k)
end

#------------ Noise layer ------------
function applynoiselayer(psum::PauliSum;depol_strength=0.02, dephase_strength=0.02, noise_level=1.0)
    """
    applynoiselayer(psum::PauliSum; depol_strength=0.02, dephase_strength=0.02, noise_level=1.0)

    Applies a noise layer to a `PauliSum` by attenuating its coefficients in-place.

    The function simulates decoherence by scaling each Pauli string's coefficient based on its weight 
    and the specific types of operators it contains.

    # Arguments
    - `psum::PauliSum`: The collection of Pauli operators to be modified.
    - `depol_strength`: The base rate of depolarizing noise (affects all non-identity operators).
    - `dephase_strength`: The base rate of dephasing noise (specifically affects X and Y operators).
    - `noise_level`: A global multiplier to scale the overall noise intensity.

    # Mathematical Model
    For each Pauli string, the coefficient is updated as:
    `coeff *= (1 - noise_level * depol_strength)^weight * (1 - noise_level * dephase_strength)^xy_count`
    """
    for (pstr, coeff) in psum
        set!(psum, pstr,
            coeff*(1-noise_level*depol_strength)^countweight(pstr)*(1-noise_level*dephase_strength)^countxy(pstr))
    end
end

#------------ Propagate Layer by layer ------------ 
function get_layer(
  layer_idx::Integer,
  ngate_bylayer::Integer, 
  circuit::Union{Gate, Vector{Gate}}, 
  parameters::Union{Nothing, Vector{Float64}}
  )
  first_gate_idx = ((layer_idx-1)*ngate_bylayer)+1; last_gate_idx = (layer_idx * ngate_bylayer)
  layer_gates = circuit[first_gate_idx:last_gate_idx]

  if parameters === nothing
    parameter = nothing
  else
    parameter = parameters[first_gate_idx:last_gate_idx]
  end
  return layer_gates, parameter
end

function propagate_1layer(
  layer_gates::Union{Gate, Vector{Gate}}, 
  current::Union{PauliSum, PauliString},
  parameter::Union{Nothing, Vector{Float64}};
  max_weight::Union{Nothing, Integer}=nothing, 
  min_abs_coeff::Float64=0.,
  γ::Float64=0., # for the Noise
  )::PauliSum

  if max_weight === nothing
    max_weight = current.nqubits 
  end

  current = propagate(layer_gates, current, parameter; max_weight, min_abs_coeff)
  if !(γ == 0.)
	  applynoiselayer(current; depol_strength=1, dephase_strength=0, noise_level=γ)
  end
  current /= sqrt(pauli_norm(current)) # on divise par la norm pour que \sum |c_\alpha|²=1 malgres les troncations
  return current
end

function propagate_layerbylayer(
  circuit::Union{Gate, Vector{Gate}}, 
  observable::Union{PauliSum, PauliString}, 
  nlayers::Int64, 
  parameters::Union{Nothing, Vector{Float64}}=nothing; 
  max_weight::Union{Integer,Nothing}=nothing, 
  min_abs_coeff::Float64=0.,
  k::Int64=2, # for the Entropy
  ψ0::Union{Vector{Float64}, Nothing}=nothing, # for the Overlap
  γ::Float64=0., # for the Noise
  disable_print::Bool=false
)

  t = time()
  nqubits = observable.nqubits
  ngate_bylayer = size(circuit,1) ÷ nlayers

  overlaps, entropy, norm = Float64[], Float64[], Float64[]

  if max_weight === nothing
    max_weight = nqubits 
  end
  if ψ0 === nothing
    ψ0 = append!([1],[0 for _ in 2:(2^nqubits)]) # |0> state
  end

  current = PauliPropagation.PauliSum(observable)
  push!(overlaps, overlap(current, ψ0))
  push!(entropy, renyi_entropy(current; k))
  push!(norm, pauli_norm(current))
  
  for layer_idx in nlayers:-1:1 # pour propager on a besoin de donner les couches dans le sens inverse /!\ (Heisenberg picture)
    layer_gates, parameter = get_layer(layer_idx, ngate_bylayer, circuit, parameters)
    current = propagate_1layer(layer_gates, current, parameter; max_weight, min_abs_coeff, γ)

    push!(norm, pauli_norm(current))
    push!(overlaps, overlap(current, ψ0))
    push!(entropy, renyi_entropy(current; k))

    step=nlayers-layer_idx+1
    if step % max(1, nlayers÷10)==0 && !disable_print
      println("layer : ",step,"/",nlayers," complete")
    end
  end

  elapsed_time = time() - t
  println("Time taken by pp.propagate_layerbylayer: ", elapsed_time, " seconds")

  result = Dict("overlap" => overlaps, "S" => entropy, "norm" => norm, "time" => elapsed_time)
 
  return current, result
end

#------------ Find optimal truncations ------------ 
function find_truncations(
  tolerance::Float64, 
  circuit::Union{Gate, Vector{Gate}}, 
  observable::Union{PauliSum, PauliString}, 
  nlayers::Int64, 
  parameters::Union{Nothing, Vector{Float64}}=nothing;
  γ::Float64=0., # for the Noise
  disable_print::Bool=false
  )::Tuple{Int64, Float64}

  nqubits = observable.nqubits
  ngate_bylayer = size(circuit,1) ÷ nlayers
  ψ0 = append!([1],[0 for _ in 2:(2^nqubits)]) # |0> state
  
  smaller_min_abs_coeff = 1e-13
  
  heisenberg_circuit, parameter_list = Vector{Gate}[], Union{Float64,Nothing}[]
  for layer_idx in nlayers:-1:1
    layer_gates, parameter = get_layer(layer_idx, ngate_bylayer, circuit, parameters)
    push!(heisenberg_circuit, layer_gates)
    push!(parameter_list, parameter)
  end

  # ----- Max weight TEST -----
  max_weight = 2 #+1 for the first test

  overlaps = fill(Inf, nlayers)

  isclose_overlap = false
  while !isclose_overlap
    overlaps_before = overlaps
    max_weight += max(1, nqubits÷10)

    if max_weight >= nqubits
      max_weight = nqubits
      break
    end

    current = observable
    overlaps = Float64[]
    for i in 1:nlayers
      current = propagate_1layer(heisenberg_circuit[i], current, parameter_list[i]; max_weight, min_abs_coeff=smaller_min_abs_coeff, γ)
      push!(overlaps, overlap(current, ψ0))
    end
    isclose_overlap = isapprox(overlaps, overlaps_before; rtol=tolerance)
  end

  # ----- Min abs coeff TEST -----
  min_abs_coeff_power = -1 #-1 for the first test

  overlaps = fill(Inf, nlayers)
  isclose_overlap = false
  while !isclose_overlap
    overlaps_before = overlaps
    min_abs_coeff_power -= 1
    min_abs_coeff = 10^float(min_abs_coeff_power)

    if min_abs_coeff <= smaller_min_abs_coeff
      if !disable_print
        println("Optimal truncations find : (Max weight=$max_weight, Min abs coeff=1e$smaller_min_abs_coeff)")
      end
      return max_weight, smaller_min_abs_coeff
    end

    current = observable
    overlaps = Float64[]
    for i in 1:nlayers
      current = propagate_1layer(heisenberg_circuit[i], current, parameter_list[i]; max_weight, min_abs_coeff, γ)
      push!(overlaps, overlap(current, ψ0))
    end
    isclose_overlap = isapprox(overlaps, overlaps_before; rtol=tolerance)
  end
  min_abs_coeff = 10^float(min_abs_coeff_power)
  if !disable_print
    println("Optimal truncations find : (Max weight=$max_weight, Min abs coeff=1e$min_abs_coeff_power)")
  end
  return max_weight, min_abs_coeff
end

end # module