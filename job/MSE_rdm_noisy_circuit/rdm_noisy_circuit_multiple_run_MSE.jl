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

# Observable : Z_i
i = 2

# qubits list
Ns = [7, 9, 11]

# Gamma list = lambda_list/Nqubits
lambda_list = 0:0.05:0.4

normalize = true

for nqubits in Ns
  nlayers = nqubits

  power_list = (nqubits-5):(nqubits-1)
  maxdim_list = 4 .^ (power_list)
  maxsize_list = 4 .^(power_list)
  println("------------- nqubits=$nqubits -------------")
  # define the circuit
  circuit_rdm = ct.random_circuit(nqubits, nlayers; separateOddEvenLayer=true)

  # define initial state |0> state
  ψ0_exact = append!([1.],[0. for _ in 2:(2^nqubits)])

  ψ0_pp = ψ0_exact

  ψ0_mps = MPS(circuit_rdm["sites"], "0")

  # define observable Z_i = I...IZI...I
  Z_i_exact = ext.get_Zi(nqubits, i)

  Z_i_pp = PauliString(nqubits, :Z, i)

  ops = ["Id" for n in 1:nqubits]
  ops[i] = "Z"
  Z_i_mpo = MPO(circuit_rdm["sites"], ops)

  # --- PROPAGATION ---
  maxdim_list_for_csv = copy(maxdim_list)
  maxsize_list_for_csv = copy(maxsize_list)
  push!(maxdim_list_for_csv, 4^nqubits)
  push!(maxsize_list_for_csv, 4^nqubits)
  error_mpo_dict = Dict("N_qubits" => nqubits, "maxdim" => maxdim_list_for_csv)
  error_pp_dict = Dict("N_qubits" => nqubits, "maxsize" => maxsize_list_for_csv)

  for lambda in lambda_list
    gamma = lambda/nqubits
    println("---------- gamma=$lambda / $nqubits ----------")

    error_pp_list, error_mpo_list = Vector{Float64}[], Vector{Float64}[]
    exact_value_sq = Vector{Float64}[]

    println("------- gamma=$lambda / $nqubits, Exact -------")
    Zi_t_exact, result_exact = ext.propagate_layerbylayer(circuit_rdm["exact"], Z_i_exact; ψ0=ψ0_exact, γ=gamma, normalize)
    overlap_exact = result_exact["overlap"]

    for (maxdim, max_size) in zip(maxdim_list, maxsize_list)
      println("------- gamma=$lambda / $nqubits, Pauli, max_size=$max_size -------")
      psum, result_pp = pp.propagate_layerbylayer(circuit_rdm["pauli"], Z_i_pp, nlayers*2; max_size, ψ0=ψ0_pp, γ=gamma, normalize)

      println("------- gamma=$lambda / $nqubits, MPO, maxdim=$maxdim -------")
      Z_it_mpo, result_mpo = mpo.propagate_layerbylayer(circuit_rdm["mpo"], Z_i_mpo; maxdim, ψ0=ψ0_mps, γ=gamma, normalize)

      # --- Save Data ---
      push!(error_pp_list, @. abs(overlap_exact - result_pp["overlap"])^2)
      push!(error_mpo_list, @. abs(overlap_exact - result_mpo["overlap"])^2)
      push!(exact_value_sq, overlap_exact.^2)
    end
    push!(error_pp_list, @. abs(overlap_exact - overlap_exact)^2)
    push!(error_mpo_list, @. abs(overlap_exact - overlap_exact)^2)

    error_mpo_dict["sq error, gammaN=$lambda"] = error_mpo_list
    error_pp_dict["sq error, gammaN=$lambda"] = error_pp_list
    error_pp_dict["exact value sq, gammaN=$lambda"] = exact_value_sq # pour l'erreur relative
    error_mpo_dict["exact value sq, gammaN=$lambda"] = exact_value_sq
  end

  # --- Save Data ---
  complete_path_error_mpo = joinpath(path, "results_N_$(nqubits)-MSE_mpo.csv")
  complete_path_error_pp = joinpath(path, "results_N_$(nqubits)-MSE_pp.csv")

  error_mpo_df = DataFrame(error_mpo_dict)
  error_pp_df = DataFrame(error_pp_dict)

  CSV.write(complete_path_error_mpo, error_mpo_df)
  CSV.write(complete_path_error_pp, error_pp_df)
end