"""
    function plot_vector_fields(vecf::Function, data::Matrix{Float64},
            p::Vector{Float64})
"""
function plot_vector_fields(vecf::Function, data::Matrix{Float64},
        p::Vector{Float64})

    arrow_end = zeros(size(data));
    for i = 1:1:size(data, 2)
        arrow_end[:, i] = vecf(data[:, i], estp, 0);
    end

    norm_arrow_end = arrow_end ./ maximum(abs.(arrow_end));

    outfig = quiver(data[1, :], data[2, :],
        quiver = (norm_arrow_end[1, :], norm_arrow_end[2, :]));

    return outfig;
end
