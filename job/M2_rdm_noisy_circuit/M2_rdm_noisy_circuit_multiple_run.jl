# --- Import ---
using Pkg; Pkg.add("PauliPropagation"); Pkg.add("DataFrames"); Pkg.add("CSV")

# Pauli Propagation
using PauliPropagation

include("../src/pauli_propagation_functions.jl")
import .pauli_propagation_functions as pp

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

# Entropy : M(k)
k=2

for nqubits in Ns
  println("------------- nqubits=$nqubits -------------")
  # define the circuit
  nlayers = 4*nqubits
  circuit_pp, _, _, _ = ct.random_circuit(nqubits, nlayers)

  # define the observable
  Z_i_pp = PauliString(nqubits, :Z, i) # I...IZI...I

  # --- PROPAGATION ---
  results_Renyi_entropy_dict = Dict("N_qubits" => nqubits, "Layer" => 0:nlayers)

  for lambda in lambda_list
    gamma = lambda/nqubits
    println("---------- gamma=$lambda / $nqubits ----------")
    psum, result_pp = pp.propagate_layerbylayer(circuit_pp, Z_i_pp, nlayers; γ=gamma, k, normalize=false)

    # --- Save Data ---
    results_Renyi_entropy_dict["gammaN=$lambda"] = result_pp["S"]
  end

  complete_path_Renyi_entropy = joinpath(path, "results_N_$(nqubits)-Renyi_entropy.csv")
  results_Renyi_entropy = DataFrame(results_Renyi_entropy_dict)
  CSV.write(complete_path_Renyi_entropy, results_Renyi_entropy)
end