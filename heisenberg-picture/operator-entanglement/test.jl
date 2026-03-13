using ITensors, ITensorMPS

# nombre de spins
N = 6

# creation des sites
sites = ITensors.siteinds("S=1/2", N)

# constante dans l'hamiltonien
g = 0.5
h = 0.5

# constante de numérisation 
cutoff = 1e-8
τ = 0.1


# Hamiltonien local h_{j,j+1}
function build_two_site_hamiltonian(j::Int64)
  s1 = sites[j]
  s2 = sites[j + 1]
  hj =
     op("X", s1) * op("X", s2) +
     (g/2) * (op("X", s1) * op("Id", s2) + op("Id", s1) * op("X", s2)) +
     (h/2) * (op("Z", s1) * op("Id", s2) + op("Id", s1) * op("Z", s2))
  return hj
end

# Construction des gates TEBD
function build_gates(τ::Float64)::Tuple{Vector{ITensor}, Vector{ITensor}}
  gates_odd = ITensor[]
  gates_even = ITensor[]
  for j in 1:(N - 1)
      hj = build_two_site_hamiltonian(j)
      Gj = exp(-im * τ / 2 * hj) # gate Trotter ordre 2
      if isodd(j)
        push!(gates_odd, Gj)
      else
        push!(gates_even, Gj)
      end
  end
  return gates_odd, gates_even
end

# Application U† O U pour une liste de gates
function apply_heisenberg_step(O::MPO, gates::Vector{ITensor})::MPO
  O = apply(gates, O; cutoff=cutoff)
  O = apply(dag.(reverse(gates)), O; cutoff=cutoff)
  return O
end

# Evolution temporelle
function simulate_operator_evolution(O::MPO, steps::Int64, τ::Float64=0.01)::MPO
  gates_odd, gates_even = build_gates(τ)

  for step in 1:steps
    O = apply_heisenberg_step(O,gates_odd)
    O = apply_heisenberg_step(O,gates_even)
    O = apply_heisenberg_step(O,gates_odd)
  end
  return O
end


# construction MPO identite
IdMPO = MPO(sites, "Id")

# Simulation
steps = 20
Ot = simulate_operator_evolution(IdMPO, steps, τ)

println("Max bond dimension = ", maxlinkdim(Ot))
println("\nIdentity distance = ", norm(Ot - IdMPO))