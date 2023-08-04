"""
    struct system_obs
        phi::Function

        t0::Float64
        ts::Float64
        tf::Float64

        u0::Vector
        p::Vector

        tolerances::Vector{Float64}
    end

Letting n = length(u0), it represents the dynamical system
    dx_1/dt = x_2
    .
    .
    .
    dx_{n-1}/dt = x_n
    dx_n/dt = phi(x, p, t)

When getting the solution of the system, it will start integrating from t0 to
tf and sampling with ts. 'tolerances' is a vector such that
reltol = tolerances[1] and abstol = tolerances[2].
"""
struct system_obs
    phi::Function

    t0::Float64
    ts::Float64
    tf::Float64

    u0::Vector
    p::Vector

    tolerances::Vector{Float64}
end
