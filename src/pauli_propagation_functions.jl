module pauli_propagation_functions
export decode_pauli, compute_matrix, overlap, pauli_norm, pauli_entropy, operator_entropy, propagate_layerbylayer

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
function compute_matrix(observable::Union{PauliSum, PauliString})
  if typeof(observable) == PauliString{UInt8, Float64}
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
function overlap2(observable::Union{PauliSum, PauliString}, ψ)::Float64
  state0 = append!([1],[0 for _ in 2:(2^observable.nqubits)])
  if ψ == state0
    return overlapwithzero(observable)
  end 
  matrix = compute_matrix(observable)
  return real(ψ' * matrix * ψ)
end

# faster than overlap2
function overlap(observable::Union{PauliSum, PauliString}, ψ)::Float64 
  if typeof(observable) == PauliString{UInt8, Float64}
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
function pauli_norm(observable::Union{PauliSum, PauliString})
    if typeof(observable) == PauliString{UInt8, Float64}
      observable = PauliSum(observable)
    end
    return sum(((P, c),) -> abs(c)^2, observable; init=0.0)
end

#------------ Pauli Entropy ------------ 
function pauli_entropy(pauli_sum::PauliSum)
    return sum(((P, c),) -> c != 0 ? -abs(c)^2 * log(abs(c)^2) : 0.0, pauli_sum; init=0.0)
end

#------------ Operator Entropy ------------
function operator_entropy_matrix(O::Matrix, bond::Int)::Float64
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

function operator_entropy(observable::Union{PauliSum, PauliString}, bond::Int)::Float64
    matrix = compute_matrix(observable)
    entropy = operator_entropy_matrix(matrix, bond)
  return entropy
end

#------------ Propagate Layer by layer ------------ 
function propagate_layerbylayer(
  circuit, 
  observable::Union{PauliSum, PauliString}, 
  nlayers::Int64, 
  parameters=nothing; 
  max_weight::Integer, 
  min_abs_coeff::Float64,
  ψ0::Union{Vector{Float64}, Nothing}=nothing)

  t1 = time()
  ngate_bylayer = size(circuit,1) ÷ nlayers

  overlaps, entropy, norm = Float64[], Float64[], Float64[]

  if ψ0 === nothing
      ψ0 = append!([1],[0 for _ in 2:(2^observable.nqubits)]) # |0> state
  end

  current = PauliPropagation.PauliSum(observable)
  push!(overlaps, overlap(current, ψ0))
  push!(entropy, pauli_entropy(current))
  push!(norm, pauli_norm(current))
  
  for i in nlayers:-1:1 # pour propager on a besoin de donner les couches dans le sens inverse /!\
    first_gate_idx = ((i-1)*ngate_bylayer)+1; last_gate_idx = (i * ngate_bylayer)
    layer_gates = circuit[first_gate_idx:last_gate_idx]

    if parameters === nothing
        parameter = nothing
    else
        parameter = parameters[first_gate_idx:last_gate_idx]
    end
    current = propagate(layer_gates, current, parameter; max_weight, min_abs_coeff)

    current /= sqrt(pauli_norm(current)) # on divise par la norm pour que \sum |c_\alpha|²=1 malgres les troncations

    push!(norm, pauli_norm(current))
    push!(overlaps, overlap(current, ψ0))
    push!(entropy, pauli_entropy(current))

    j=nlayers-i+1
    if j % max(1, nlayers÷10)==0
      println("layer : ",j,"/",nlayers," complete")
    end
  end

  elapsed_time = time() - t1
  println("Time taken by pp.propagate_layerbylayer: ", elapsed_time, " seconds")

  result = Dict("overlap" => overlaps, "S" => entropy, "norm" => norm, "time" => elapsed_time)
 
  return current, result
end

end # module