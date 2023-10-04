################################################################################
#########################EXPORTED STRUCTS#######################################
################################################################################
"""
    struct system_obs
        phi::Function

        t0::Float64
        ts::Float64
        tf::Float64

        u0::Vector
        p::Vector

        tolerances::Vector{Float64}

        obs_map::Function
    end

Letting n = length(u0), it represents the dynamical system
    dx_1/dt = x_2
    .
    .
    .
    dx_{n-1}/dt = x_n
    dx_n/dt = phi(x, p, t)

When requesting the solution of the system, it will start integrating from t0 to
tf and sampling with ts. 'tolerances' is a vector such that
reltol = tolerances[1] and abstol = tolerances[2].
'obs_map' ...
"""
struct system_obs
    phi::Function

    t0::Float64
    ts::Float64
    tf::Float64

    u0::Vector
    p::Vector

    tolerances::Vector{Float64}

    obs_map::Function
end

"""
    struct system
        f::Function
        h::Function
        obs_map::Function

        t0::Float64
        ts::Float64
        tf::Float64

        u0::Vector
        p::Vector
        p_h::Vector

        tolerances::Vector{Float64}
    end

It represents the dynamical system dx/dt = f(x, p, t), y = h(x, p_h, t).

When requesting the solution of the system, it will start integrating from t0 to
tf and sampling with ts.
'tolerances' is a vector such that reltol = tolerances[1] and
abstol = tolerances[2].
'obs_map' ...
"""
struct system
    f::Function
    h::Function
    obs_map::Function

    t0::Float64
    ts::Float64
    tf::Float64

    u0::Vector
    p::Vector
    p_h::Vector

    tolerances::Vector{Float64}
end

"""
    struct forward_kinematics
        forward_kinematics::Function

        t0::Float64
        ts::Float64
        tf::Float64
    end

Represents the forward kinematics of some manipulator.
'forward_kinematics' has to be a function of three arguments: (_, p, t).
The first argument, however, has to be unused.
"""
struct forward_kinematics
    forward_kinematics::Function

   t0::Float64
    ts::Float64
    tf::Float64
end

"""
    struct periodical_signal
        s::Function

        t0::Float64
        ts::Float64
        tf::Float64
    end
"""
struct periodical_signal
    s::Function

    t0::Float64
    ts::Float64
    tf::Float64
end

################################################################################
##############################NOT EXPORTED######################################
################################################################################
"""
    struct sys_training_env
        f::Function
        h::Function
        obs_map::Function

        n::Int

        t0::Float64
        tf_tr::Float64
        ts::Float64
        tolerances::Vector{Float64}

        d_samples::Int

        data::Matrix{Float64}

        batch_indexes::Vector{Int}

        mxs::Matrix{Float64}
    end
"""
struct sys_training_env
    f::Function
    h::Function
    obs_map::Function

    n::Int

    t0::Float64
    tf_tr::Float64
    ts::Float64
    tolerances::Vector{Float64}

    d_samples::Int

    data::Matrix{Float64}

    batch_indexes::Vector{Int}

    mxs::Matrix{Float64}
end

"""
    struct sysobs_training_env
        f::Function
        obs_map::Function
        n::Int

        t0::Float64
        tf_tr::Float64
        ts::Float64
        tolerances::Vector{Float64}

        d_samples::Int

        data::Matrix{Float64}
        u0::Vector{Float64}
        fixed_ic::Bool
    end
"""
struct sysobs_training_env
    f::Function
    obs_map::Function
    n::Int

    t0::Float64
    tf_tr::Float64
    ts::Float64
    tolerances::Vector{Float64}

    d_samples::Int

    data::Matrix{Float64}
    u0::Vector{Float64}
    fixed_ic::Bool
end

"""
    struct fk_training_env
        fd_kin::forward_kinematics

        tf_train::Float64
        data::Matrix{Float64}
    end
"""
struct fk_training_env
    fd_kin::forward_kinematics

    tf_train::Float64
    data::Matrix{Float64}
end

"""
    struct freq_training_env
        s::periodical_signal

        t0::Float64
        ts::Float64
        tf_tr::Float64
        y_samples::Matrix{Float64}
    end
"""
struct freq_training_env
    s::periodical_signal

    t0::Float64
    ts::Float64
    tf_tr::Float64
    y_samples::Matrix{Float64}
end

"""
    struct grad_inv_env
        dynamics::Function
        n::Int
        m::Int
        t0::Float64
        tf::Float64
        ts::Float64
        u0::Vector
        tolerances::Vector
    end
"""
struct grad_inv_env
    dynamics::Function
    n::Int
    m::Int
    t0::Float64
    tf::Float64
    ts::Float64
    u0::Vector
    tolerances::Vector
end

"""
    struct gain_training_env
        f::Function

        n::Int

        t0::Float64
        tf::Float64
        ts::Float64
        tolerances::Vector{Float64}

        d_samples::Int

        ics::Matrix{Float64}
    end
"""
struct gain_training_env
    f::Function

    n::Int

    t0::Float64
    tf::Float64
    ts::Float64
    tolerances::Vector{Float64}

    d_samples::Int

    ics::Matrix{Float64}

    hgo_type::Int
    S::Vector
    coeffs::Vector
    gain_type::Int
end

"""
    struct pretraining_env
        data::Matrix{Float64}
        g::Function
        Lfmh::Function
    end
"""
struct pretraining_env
    data::Matrix{Float64}
    g::Function
    Lfmh::Function
    obs_map::Function
end

"""
    struct inverse_env
        data::Matrix{Float64}
        g::Function
        Lfmh::Function
    end
"""
struct inverse_env
    data::Matrix{Float64}
    N3::Function
    obs_map::Function
end
