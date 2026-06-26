# --- Import ---
using Pkg; Pkg.add("PauliPropagation"); Pkg.add("DataFrames"); Pkg.add("CSV"); Pkg.add("ITensors"); Pkg.add("ITensorMPS")

# Pauli Propagation
using PauliPropagation

include("../src/pauli_propagation_functions.jl")
import .pauli_propagation_functions as pp

# MPO Propagation
using ITensors, ITensorMPS

include("../src/matrix_product_operator_functions.jl")
import .mpo_functions as mpo

# other
using DataFrames, CSV

include("../src/circuit.jl")
import .circuit as ct

# --- Parameters ---
run = parse(Int64, ARGS[1])
path = joinpath("results", "run_$run")
mkpath(path)

# Observable : Z_i
i = 2

# qubits list
Ns = [7, 9, 11]

# Gamma list = lambda_list/Nqubits
lambda_list = 0:0.02:0.36

# Entropy : M(k) or S(k)
k=2

for nqubits in Ns
  println("------------- nqubits=$nqubits -------------")
  # define the circuit
  nlayers = 4*nqubits
  circuit_rdm = ct.random_circuit(nqubits, 2*nqubits; separateOddEvenLayer=true, exact=false)

  # define the observable
  Z_i_pp = PauliString(nqubits, :Z, i) # I...IZI...I
  ops = ["Id" for n in 1:nqubits]
  ops[i] = "Z"
  Z_i_mpo = MPO(circuit_rdm["sites"], ops)

  # --- PROPAGATION ---
  results_OSE_dict = Dict("N_qubits" => nqubits, "Layer" => 0:nlayers)
  results_OE_dict = Dict("N_qubits" => nqubits, "Layer" => 0:nlayers)

  for lambda in lambda_list
    gamma = lambda/nqubits
    println("---------- gamma=$lambda / $nqubits ----------")
    psum_tf, result_pp = pp.propagate_layerbylayer(circuit_rdm["pauli"], Z_i_pp, nlayers; γ=gamma, k, normalize=true)
    mpo_tf, result_mpo = mpo.propagate_layerbylayer(circuit_rdm["mpo"], Z_i_mpo; γ=gamma, bond=nqubits÷2, k, normalize=true)

    # --- Save Data ---
    results_OSE_dict["gammaN=$lambda"] = result_pp["S"]
    results_OE_dict["gammaN=$lambda"] = result_mpo["S"]
  end

  complete_path_OSE = joinpath(path, "results_N_$(nqubits)-OSE.csv")
  complete_path_OE = joinpath(path, "results_N_$(nqubits)-OE.csv")

  results_OSE_df = DataFrame(results_OSE_dict)
  results_OE_df = DataFrame(results_OE_dict)

  CSV.write(complete_path_OSE, results_OSE_df)
  CSV.write(complete_path_OE, results_OE_df)
end