module utils
export average_data, truncation_plot

using Plots

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

function truncation_plot(
  parameter,
  truncations::Vector{Tuple{Int64, Float64}};
  method::Union{String, Nothing}=nothing,
  title::String="",
  parameterName::String=""
  )
  @assert length(parameter) == length(truncations) # parameter and truncations need to be the same size
  if method == "MPO"
    ylabel1 = "Max Weight"
    ylabel2 = "Min Abs Coefficent (log scale)"
  elseif method == "Pauli"
    ylabel1 = "Max Dimension"
    ylabel2 = "Cutoff (log scale)"
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

end # Module