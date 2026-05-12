module utils
export average_data, overlap_with_Zi_plot, complexity_plot, truncation_plot, print_methods_error

using Plots
using LinearAlgebra
import Statistics: mean

function average_data(data::Vector{Float64}, windowsize::Int)
    n = length(data)
    if windowsize > n || windowsize <= 0
        error("The window size must be between 1 and the size of the data.")
    end

    output_length = n - windowsize + 1 # on peut pas prendre les extremitées

    result = zeros(Float64, output_length)
    
    for i in 1:output_length
        window_temp = @views data[i : i + windowsize - 1] # views évite de copier les données en mémoire lors de la découpe
        
        # average
        result[i] = sum(window_temp) / windowsize
    end
    
    return result
end

function overlap_with_Zi_plot(
  nqubits::Int64,
  overlap_pp::Vector{Float64},
  overlap_mpo::Vector{Float64},
  overlap_exact::Vector{Float64},
  i::Int64,
  max_weight::Int64,
  min_abs_coeff::Float64,
  maxdim::Int64,
  cutoff::Float64
)
  p = plot(title="<Z_$i>(t) for n qubits=$nqubits, \n Pauli : max weight=$max_weight, min abs coeff=$min_abs_coeff, \n MPO : Maxdim=$maxdim, Cutoff=$cutoff", xlabel="layers", ylabel="overlap with the state |0>")
  plot!(p, overlap_pp, label="Pauli")
  plot!(p, overlap_mpo, label="MPO")
  plot!(p, overlap_exact, label="Exact", line = (1, :dash), color=:black)
  plot!(legend=:outerbottom, legendcolumns=2)
  display(p)
end 

function complexity_plot(
  Nqubits_list::Union{UnitRange{Int64}, Vector{Int64}},
  nlayers::Int64,
  times_exact::Vector{Float64},
  times_pp::Vector{Float64},
  times_mpo::Vector{Float64};
  logscale::Bool=false
)
  if logscale
    plot(Nqubits_list, [times_exact, times_pp, times_mpo],
      label = ["Exact method" "Pauli Propagation" "MPO"],
      yaxis = :log10,                # Passage en échelle semi-log
      marker = [:circle :square :square],     # Ajout de points pour voir les mesures réelles
      markersize = 5,
      lw = 2,                        # Épaisseur de ligne
      xlabel = "Number of qubits (N)",
      ylabel = "Computation time (seconds) log scale",
      title = "Scaling of complexity : Exact vs Pauli vs MPO, \nnlayers=$nlayers",
      subtitle = "TFIM Circuit",
      legend = :topleft,
      grid = :both,
      minorgrid = true
    )
  else
    plot(Nqubits_list, [times_exact, times_pp, times_mpo],
      label = ["Exact method" "Pauli Propagation" "MPO"],
      marker = [:circle :square :square],     # Ajout de points pour voir les mesures réelles
      markersize = 5,
      lw = 2,                        # Épaisseur de ligne
      xlabel = "Number of qubits (N)",
      ylabel = "Computation time (seconds) log scale",
      title = "Scaling of complexity : Exact vs Pauli vs MPO, \nnlayers=$nlayers",
      subtitle = "TFIM Circuit",
      legend = :topleft,
      grid = :both,
      minorgrid = true
    )
  end
end

function truncation_plot(
  parameter,
  truncations::Vector{Tuple{Int64, Float64}};
  method::Union{String, Nothing}=nothing,
  title::String="",
  parameterName::String=""
  )
  @assert length(parameter) == length(truncations) # parameter and truncations need to be the same size
  if method == "MPO"
    ylabel1 = "Max Dimension"
    ylabel2 = "Cutoff (log scale)"
  elseif method == "Pauli"
    ylabel1 = "Max Weight"
    ylabel2 = "Min Abs Coefficent (log scale)"
  else
    ylabel1 = "Int"
    ylabel2 = "Float (log scale)"
  end

  valeurs_int = first.(truncations)
  valeurs_float = last.(truncations)

  p1 = plot(parameter, valeurs_int, 
      title = title, 
      ylabel = ylabel1, 
      xlabel = parameterName,
      marker = :circle, 
      legend = false)

  p2 = plot(parameter, valeurs_float, 
      title = "Method : $method",
      ylabel = ylabel2, 
      xlabel = parameterName,
      marker = :square, 
      yaxis = :log10, 
      color = :red, 
      legend = false)

  plot(p1, p2, layout = (1, 2), size = (800, 400))
end

function print_methods_error(
  Ot_exact::Matrix, Ot_pauli_propagation_matrix::Matrix, Ot_mpo_matrix::Matrix, # the matrix
  overlap_exact::Vector{Float64}, overlap_pp::Vector{Float64}, overlap_mpo::Vector{Float64}, # the overlap
  error_tolerance::Float64; operatorName::String="O"
  )
  # Matrix Comparison
  pauli_matrix_max_diff = mean(abs.(Ot_exact .- Ot_pauli_propagation_matrix))
  mpo_matrix_max_diff = mean(abs.(Ot_exact .- Ot_mpo_matrix))
  pauli_matrix_close = pauli_matrix_max_diff < error_tolerance
  mpo_matrix_close = mpo_matrix_max_diff < error_tolerance

  println("--- Matrix Errors ---")
  println("$(operatorName)(t_final) Pauli vs Exact close (tol=$error_tolerance)? ", pauli_matrix_close)
  println("$(operatorName)(t_final) MPO vs Exact close (tol=$error_tolerance)? ", mpo_matrix_close)
  println("Mean matrix error (Pauli): ", pauli_matrix_max_diff)
  println("Mean matrix error (MPO): ", mpo_matrix_max_diff)

  # Overlap Comparison
  pauli_overlap_max_diff = maximum(abs.(overlap_pp .- overlap_exact))
  mpo_overlap_max_diff = maximum(abs.(overlap_mpo .- overlap_exact))
  pauli_overlap_close = pauli_overlap_max_diff < error_tolerance
  mpo_overlap_close = mpo_overlap_max_diff < error_tolerance

  println("\n--- Overlap Curve Errors ---")
  println("Overlap Pauli vs Exact close (tol=$error_tolerance)? ", pauli_overlap_close)
  println("Overlap MPO vs Exact close (tol=$error_tolerance)? ", mpo_overlap_close)
  println("Max overlap error (Pauli): ", pauli_overlap_max_diff)
  println("Max overlap error (MPO): ", mpo_overlap_max_diff)
end

end # Module