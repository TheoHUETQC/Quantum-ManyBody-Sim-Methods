# --- Import ---
using Pkg; Pkg.add("LaTeXStrings"); Pkg.add("PauliPropagation"); Pkg.add("ITensors"); Pkg.add("ITensorMPS"); Pkg.add("DataFrames"); Pkg.add("CSV")

# Pauli Propagation
using PauliPropagation

include("../src/pauli_propagation_functions.jl")
import .pauli_propagation_functions as pp

# MPO
using ITensors, ITensorMPS

include("../src/matrix_product_operator_functions.jl")
import .mpo_functions as mpo

# exact
include("../src/exact_functions.jl")
import .exact_functions as ext

# other
using DataFrames, CSV

include("../src/circuit.jl")
import .circuit as ct

include("../src/utils.jl")
import .utils as us

import Statistics: mean

# --- Parameters ---
run = parse(Int64, ARGS[1])
path = joinpath("results", "run_$run")
mkpath(path)

nlayers = 40

# Observable : Z_i
i = 2

# qubits list
Ns = [5, 6, 7, 8, 9]

# Gamma list = lambda_list/Nqubits
lambda_list = 0:0.05:0.4

power_list = 2:6
maxdim_list = 2 .^ (power_list)
maxsize_list = 4 .^(power_list)
cutoff = 1e-7

for nqubits in Ns
  println("------------- nqubits=$nqubits -------------")
  # define the circuit
  circuit_pp, circuit_mpo, circuit_exact, sites = ct.random_circuit(nqubits, nlayers)

  # define initial state |0> state
  ψ0_exact = append!([1.],[0. for _ in 2:(2^nqubits)])

  ψ0_pp = ψ0_exact

  ψ0_mps = MPS(sites, "0")

  # define observable Z_i = I...IZI...I
  Z_i_exact = ext.get_Zi(nqubits, i)

  Z_i_pp = PauliString(nqubits, :Z, i)

  ops = ["Id" for n in 1:nqubits]
  ops[i] = "Z"
  Z_i_mpo = MPO(sites, ops)

  # --- PROPAGATION ---
  error_mpo_dict = Dict("N_qubits" => nqubits, "maxdim" => maxdim_list)
  error_pp_dict = Dict("N_qubits" => nqubits, "maxsize" => maxsize_list)

  for lambda in lambda_list
    gamma = lambda/nqubits
    println("---------- gamma=$lambda / $nqubits ----------")

    error_pp_list, error_mpo_list = Vector{Float64}[], Vector{Float64}[]

    for j in 1:length(maxdim_list)

      println("------- gamma=$lambda / $nqubits, Exact -------")
      Zi_t_exact, result_exact = ext.propagate_layerbylayer(circuit_exact, Z_i_exact; ψ0=ψ0_exact, γ=gamma)
      overlap_exact = result_exact["overlap"]

      max_size = maxsize_list[j]
      maxdim = maxdim_list[j]

      println("------- gamma=$lambda / $nqubits, Pauli, max_size=$max_size -------")
      psum, result_pp = pp.propagate_layerbylayer(circuit_pp, Z_i_pp, nlayers; max_size, min_abs_coeff=cutoff, ψ0=ψ0_pp, γ=gamma)

      println("------- gamma=$lambda / $nqubits, MPO, maxdim=$maxdim -------")
      Z_it_mpo, result_mpo = mpo.propagate_layerbylayer(circuit_mpo, Z_i_mpo; cutoff, maxdim, ψ0=ψ0_mps, γ=gamma)

      # --- Save Data ---
      push!(error_pp_list, @. abs(overlap_exact - result_pp["overlap"]))
      push!(error_mpo_list, @. abs(overlap_exact - result_mpo["overlap"]))
    end
    error_mpo_dict["error gammaN=$lambda"] = error_mpo_list
    error_pp_dict["error gammaN=$lambda"] = error_pp_list
  end

  # --- Save Data ---
  complete_path_error_mpo = joinpath(path, "results_N_$(nqubits)-error_mpo.csv")
  complete_path_error_pp = joinpath(path, "results_N_$(nqubits)-error_pp.csv")

  error_mpo_df = DataFrame(error_mpo_dict)
  error_pp_df = DataFrame(error_pp_dict)

  CSV.write(complete_path_error_mpo, error_mpo_df)
  CSV.write(complete_path_error_pp, error_pp_df)
end