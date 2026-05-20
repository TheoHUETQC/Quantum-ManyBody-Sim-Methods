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

# other
using DataFrames, CSV

include("../src/utils.jl")
import .utils as us

include("../src/circuit.jl")
import .circuit as ct

# --- Parameters ---
run = parse(Int64, ARGS[1])
path = joinpath("results", "run_$run")
mkpath(path)

nlayers = 40
tolerance = 1e-1

# Observable : Z_i
i = 2

# qubits list
Ns = [5, 6, 7, 8, 9, 10]

# Gamma list
"""x = Array(-5:2)
lambda_val = @. 2^float(x)
gamma_list = lambda_val/nqubits"""
gamma_list = 0:0.02:0.4


for nqubits in Ns
  # entropy parameter
  k=2
  bond = nqubits ÷ 2 # we divide the system in the middle
  
  println("------------- nqubits=$nqubits -------------")
  # define the circuit
  circuit_pp, circuit_mpo, _, sites = ct.random_circuit(nqubits, nlayers)

  Z_i_pp = PauliString(nqubits, :Z, i) # I...IZI...I

  ops = ["Id" for n in 1:nqubits]
  ops[i] = "Z"
  Z_i_mpo = MPO(sites, ops)

  # --- PROPAGATION ---
  results_Renyi_entropy_dict = Dict("N_qubits" => nqubits, "Layer" => 0:nlayers)
  results_Operator_Entanglement_dict = copy(results_Renyi_entropy_dict)

  for gamma in gamma_list
    println("---------- gamma=$gamma ----------")
    println("------- Pauli -------")
    (max_weight, min_abs_coeff) = pp.find_truncations(tolerance, circuit_pp, Z_i_pp, nlayers; γ=gamma)
    psum, result_pp = pp.propagate_layerbylayer(circuit_pp, Z_i_pp, nlayers; max_weight, min_abs_coeff, γ=gamma, k)

    println("------- MPO -------")
    (maxdim, cutoff) = mpo.find_truncations(tolerance, circuit_mpo, Z_i_mpo; γ=gamma)
    Z_it_mpo, result_mpo = mpo.propagate_layerbylayer(circuit_mpo, Z_i_mpo; cutoff, maxdim, bond, γ=gamma)

    results_Renyi_entropy_dict["gamma=$gamma"] = result_pp["S"]
    results_Operator_Entanglement_dict["gamma=$gamma"] = result_mpo["S"]
  end

  # --- Save Data ---
  complete_path_Renyi_entropy = joinpath(path, "results_N_$(nqubits)-Renyi_entropy.csv")
  complete_path_Operator_Entanglement = joinpath(path, "results_N_$(nqubits)-Operator_Entanglement.csv")
  
  results_Renyi_entropy = DataFrame(results_Renyi_entropy_dict)
  results_Operator_Entanglement = DataFrame(results_Operator_Entanglement_dict)

  CSV.write(complete_path_Renyi_entropy, results_Renyi_entropy)
  CSV.write(complete_path_Operator_Entanglement, results_Operator_Entanglement)
end