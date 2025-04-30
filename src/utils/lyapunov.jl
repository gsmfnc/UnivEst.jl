"""
    function get_lyapunov_derivative_values(sys::controlled_system,
            data::Matrix{Float64}, p::Vector{Float64})
    function get_lyapunov_derivative_values(sys::controlled_system,
            data::Matrix{Float64}, p::Vector{Float64}, P::Matrix{Float64})
"""
function get_lyapunov_derivative_values(sys::controlled_system,
        data::Matrix{Float64}, p::Vector{Float64})

    dV_values = zeros(size(data, 2));
    f = sys.f;

    for i = 1:1:length(dV_values)
        x = data[:, i];
        dx = f(x, p, 0.0);
        dV_values[i] = x' * dx;
    end

    return dV_values;
end
function get_lyapunov_derivative_values(sys::controlled_system,
        data::Matrix{Float64}, p::Vector{Float64}, P::Matrix{Float64})

    dV_values = zeros(size(data, 2));
    f = sys.f;

    for i = 1:1:length(dV_values)
        x = data[:, i];
        dx = f(x, p, 0.0);
        dV_values[i] = x' * P * dx;
    end

    return dV_values;
end

"""
    plot_lyapunov_derivative_values_2d(test_points, dV_values)
"""
function plot_lyapunov_derivative_values_2d(test_points, dV_values)
    min_dv_val = minimum(dV_values);
    if dV_values[1] > 0
        marker_color = RGB(1, 0, 0);
    else
        tmp = 1 - dV_values[1] / min_dv_val;
        marker_color = RGB(tmp, tmp, tmp)
    end
    p = scatter([test_points[1, 1]], [test_points[2, 1]], c=marker_color,
        legend = false);

    for i = 2:1:length(dV_values)
        if dV_values[i] > 0
            marker_color = RGB(1, 0, 0);
        else
            tmp = 1 - dV_values[i] / min_dv_val;
            marker_color = RGB(tmp, tmp, tmp)
        end
        p = scatter!([test_points[1, i]], [test_points[2, i]], c=marker_color,
            legend = false);
    end

    return p;
end

"""
    plot_lyapunov_derivative_values_3d(test_points, dV_values)
"""
function plot_lyapunov_derivative_values_3d(test_points, dV_values)
    min_dv_val = minimum(dV_values);
    if dV_values[1] > 0
        marker_color = RGB(1, 0, 0);
    else
        tmp = 1 - dV_values[1] / min_dv_val;
        marker_color = RGB(tmp, tmp, tmp)
    end
    p = scatter3d([test_points[1, 1]], [test_points[2, 1]],
        [test_points[3, 1]], c=marker_color, legend = false,
        xlabel="x", ylabel="z", zlabel="theta");

    for i = 2:1:length(dV_values)
        if dV_values[i] > 0
            marker_color = RGB(1, 0, 0);
        else
            tmp = 1 - dV_values[i] / min_dv_val;
            marker_color = RGB(tmp, tmp, tmp)
        end
        p = scatter3d!([test_points[1, i]], [test_points[2, i]],
            [test_points[3, i]], c=marker_color, legend = false);
    end

    return p;
end
