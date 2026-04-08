module pauli_propagation_functions
export pauli_norm, pauli_entropy, propagate_layerbylayer

using PauliPropagation

#------------ Pauli Norm ------------ 
function pauli_norm(pauli_sum::PauliSum)
    return sum(((P, c),) -> abs(c)^2, pauli_sum; init=0.0)
end

#------------ Pauli Entropy ------------ 
function pauli_entropy(pauli_sum::PauliSum)
    return sum(((P, c),) -> c != 0 ? -abs(c)^2 * log(abs(c)^2) : 0.0, pauli_sum; init=0.0)
end

#------------ Propagate Layer by layer ------------ 
function propagate_layerbylayer(circuit, observable::PauliString, nlayers::Int64, parameters=nothing; max_weight::Integer, min_abs_coeff::Float64)
  t1 = time()
  ngate_bylayer = size(circuit,1) ÷ nlayers

  overlap, entropy, norm = Float64[], Float64[], Float64[]
  current = observable

  for i in nlayers:-1:1 # pour propager on a besoin de donner les couches dans le sens inverse /!\
    first_gate_idx = ((i-1)*ngate_bylayer)+1; last_gate_idx = (i * ngate_bylayer)
    layer_gates = circuit[first_gate_idx:last_gate_idx]

    if parameters == nothing
        parameter = nothing
    else
        parameter = parameters[first_gate_idx:last_gate_idx]
    end
    current = propagate(layer_gates, current, parameter; max_weight, min_abs_coeff)

    push!(overlap, overlapwithzero(current))
    push!(entropy, pauli_entropy(current))

    j=nlayers-i
    if j % (nlayers÷10)==0
      norm_temp = pauli_norm(current)
      push!(norm, norm_temp)
      if !(norm_temp ≈ 1)
        println("layer : ", j,"/", nlayers," Break cause pauli norm = ", norm_temp, " ≠ 1")
        break
      end
      println("layer : ",j,"/",nlayers," complete")
    end
  end

  result = Dict("overlap" => overlap, "S" => entropy, "norm" => norm)

  elapsed_time = time() - t1
  println("Time taken by propagate_layerbylayer: ", elapsed_time, " seconds")
  return current, result
end

end # module