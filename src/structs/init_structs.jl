############################INIT_SYSTEM#########################################

"""
    init_system_obs(phi::Function, u0::Vector{Float64}, p::Vector{Float64};
        t0::Float64 = 0.0, tf::Float64 = 10.0, ts::Float64 = 1e-02,
        reltol::Float64 = 1e-8, abstol::Float64 = 1e-8)

Initialize a struct of system_obs type.
"""
function init_system_obs(phi::Function, u0::Vector{Float64}, p::Vector{Float64};
        t0::Float64 = 0.0, tf::Float64 = 10.0, ts::Float64 = 1e-02,
        reltol::Float64 = 1e-8, abstol::Float64 = 1e-8)
    sys = system_obs(phi, t0, ts, tf, u0, p, [reltol, abstol]);
    return sys;
end
