"""
    init_system(f::Function, h::Function, u0::Vector{Float64};
        p::Vector{Float64} = [0.0], p_h::Vector{Float64} = [0.0],
        t0::Float64 = 0.0, tf::Float64 = 10.0, ts::Float64 = 1e-02,
        reltol::Float64 = 1e-8, abstol::Float64 = 1e-8)
    init_system(f::Function, h::Function, obs_map::Function,
        u0::Vector{Float64};
        p::Vector{Float64} = [0.0], p_h::Vector{Float64} = [0.0],
        t0::Float64 = 0.0, tf::Float64 = 10.0, ts::Float64 = 1e-02,
        reltol::Float64 = 1e-8, abstol::Float64 = 1e-8)

Initializes a struct of system type.
"""
function init_system(f::Function, h::Function, u0::Vector{Float64};
        p::Vector{Float64} = [0.0], p_h::Vector{Float64} = [0.0],
        t0::Float64 = 0.0, tf::Float64 = 10.0, ts::Float64 = 1e-02,
        reltol::Float64 = 1e-8, abstol::Float64 = 1e-8)
    blank(a, b, c) = 0;
    sys = system(f, h, blank, t0, ts, tf, u0, p, p_h, [reltol, abstol]);
    return sys;
end
function init_system(f::Function, h::Function, obs_map::Function,
        u0::Vector{Float64};
        p::Vector{Float64} = [0.0], p_h::Vector{Float64} = [0.0],
        t0::Float64 = 0.0, tf::Float64 = 10.0, ts::Float64 = 1e-02,
        reltol::Float64 = 1e-8, abstol::Float64 = 1e-8)
    sys = system(f, h, obs_map, t0, ts, tf, u0, p, p_h, [reltol, abstol]);
    return sys;
end

"""
    init_system_obs(phi::Function, u0::Vector{Float64};
        p::Vector{Float64} = [0.0], t0::Float64 = 0.0, tf::Float64 = 10.0,
        ts::Float64 = 1e-02, reltol::Float64 = 1e-8, abstol::Float64 = 1e-8)

Initializes a struct of system_obs type.
"""
function init_system_obs(phi::Function, u0::Vector{Float64};
        p::Vector{Float64} = [0.0], t0::Float64 = 0.0, tf::Float64 = 10.0,
        ts::Float64 = 1e-02, reltol::Float64 = 1e-8, abstol::Float64 = 1e-8)
    blank(a, b, c) = 0;
    sys = system_obs(phi, t0, ts, tf, u0, p, [reltol, abstol], blank);
    return sys;
end
function init_system_obs(phi::Function, obs_map::Function, u0::Vector{Float64};
        p::Vector{Float64} = [0.0], t0::Float64 = 0.0, tf::Float64 = 10.0,
        ts::Float64 = 1e-02, reltol::Float64 = 1e-8, abstol::Float64 = 1e-8)
    sys = system_obs(phi, t0, ts, tf, u0, p, [reltol, abstol], obs_map);
    return sys;
end

"""
    init_forward_kinematics(fd_kin::Function;
        t0::Float64 = 0.0, tf::Float64 = 10.0, ts::Float64 = 1e-02)

Initializes a struct of forward_kinematics type.
"""
function init_forward_kinematics(fd_kin::Function;
        t0::Float64 = 0.0, tf::Float64 = 10.0, ts::Float64 = 1e-02)
    sys = forward_kinematics(fd_kin, t0, ts, tf);
    return sys;
end

"""
    init_periodical_signal(;
        t0::Float64 = 0.0, tf::Float64 = 10.0, ts::Float64 = 1e-02)

Initializes a struct of periodical_signal type.
"""
function init_periodical_signal(;
        t0::Float64 = 0.0, tf::Float64 = 10.0, ts::Float64 = 1e-02)
    s(b, a, w, p, t) = sum(a .* sin.(w * t .+ p)) + b
    sys = periodical_signal(s, t0, ts, tf);
    return sys;
end
