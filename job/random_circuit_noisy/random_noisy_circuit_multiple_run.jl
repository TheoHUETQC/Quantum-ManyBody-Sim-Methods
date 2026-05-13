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
nlayers = 40
tolerance = 1e-1

# Observable : Z_i
i = 2

# qubits list
Ns = [5, 7, 9, 10]

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
  for gamma in gamma_list
    println("---------- gamma=$gamma ----------")
    println("------- Pauli -------")
    (max_weight, min_abs_coeff) = pp.find_truncations(tolerance, circuit_pp, Z_i_pp, nlayers; γ=gamma)
    psum, result_pp = pp.propagate_layerbylayer(circuit_pp, Z_i_pp, nlayers; max_weight, min_abs_coeff, γ=gamma, k)

    println("------- MPO -------")
    (maxdim, cutoff) = mpo.find_truncations(tolerance, circuit_mpo, Z_i_mpo; γ=gamma)
    Z_it_mpo, result_mpo = mpo.propagate_layerbylayer(circuit_mpo, Z_i_mpo; cutoff, maxdim, bond, γ=gamma)

    # --- Save Data ---
    output_file = "results_N_$nqubits-gamma_$(replace(string(gamma), "." => "_")).csv"
    
    path = joinpath("run_$run", "N_$nqubits")
    mkpath(path)

    complete_path = joinpath(path, output_file)

    results = DataFrame(
        Gamma = gamma,
        N_qubits = nqubits,
        Layer = 0:nlayers,
        Renyi_entropy = result_pp["S"],
        Operator_Entanglement = result_mpo["S"]
    )

    CSV.write(complete_path, results)
  end
end