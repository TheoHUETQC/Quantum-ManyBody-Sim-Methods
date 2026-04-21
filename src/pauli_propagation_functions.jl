module pauli_propagation_functions
export decode_pauli, compute_matrix, overlap, pauli_norm, pauli_entropy, propagate_layerbylayer

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
      result_string = 1.
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
      result_string = 1.
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

    push!(overlaps, overlap(current, ψ0))
    push!(entropy, pauli_entropy(current))

    j=nlayers-i+1
    if j % max(1, nlayers÷10)==0
      norm_temp = pauli_norm(current)
      push!(norm, norm_temp)
      if !(norm_temp ≈ 1)
        println("layer : ", j,"/", nlayers," Break cause pauli norm = ", norm_temp, " ≠ 1")
        break
      end
      println("layer : ",j,"/",nlayers," complete")
    end
  end

  result = Dict("overlap" => overlaps, "S" => entropy, "norm" => norm)

  elapsed_time = time() - t1
  println("Time taken by pp.propagate_layerbylayer: ", elapsed_time, " seconds")
  return current, result
end

end # module