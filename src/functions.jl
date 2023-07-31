struct training_parameters
    t0::Float64
    ts::Float64
    tf::Float64
    phi::Function
    u0::Vector
    p::Vector
    f::Function
    O::Function

    y_samples::Vector{Float64}
    opt
    tolerances::Vector{Float64}
    tf_tr::Float64
    data::Matrix{Float64}
    M::Int
    d_time::Float64

    add_loss::Function
end

struct gain_training_parameters
    t0::Float64
    ts::Float64
    tf::Float64

    phi::Function

    p::Vector{Float64}
    u0::Vector{Float64}
    hu0::Matrix{Float64}

    hgo_type::Int
    gain_type::Int
    coeffs::Vector{Float64}
    disturbance::Function

    opt
    tolerances::Vector{Float64}
    dynamics::Function

    add_loss::Function
end

struct ctrl_training_parameters
    alpha::Float64
    beta::Float64
    gamma::Float64

    data::Matrix{Float64}
    f::Function
    u::Function

    tolerances::Vector{Float64}
end

struct dt_training_parameters
    f::Function
    h::Function
    n::Int64

    opt
    y_samples::Matrix{Float64}

    add_loss::Function
end

#EXPORTED FUNCTIONS#############################################################

function fft_plot(x::Vector{Float64}, ts::Float64, tfinal::Float64)
    Fs = Int(round(1 / ts));
    L = Int(round(tfinal * Fs));
    t = 0:ts:(tfinal - ts);

    n = nextpow(2, L);

    Y = fft(x);

    P2 = abs.(Y / L);
    P1 = P2[1:(Int(round(n/2)) + 1)];
    P1[2:(end - 1)] = 2 * P1[2:(end - 1)];

    freqs = 0:Fs/n:Fs/2-Fs/n;
    amps = P1[1:Int(round(n / 2))];

    p1 = plot(freqs, amps);

    p1, freqs, amps
end

function find_peaks_infos(fft_amps::Vector{Float64},
        fft_freqs::StepRangeLen{Float64, Base.TwicePrecision{Float64},
        Base.TwicePrecision{Float64}}, n::Int)
    peaks_indx = findlocalmaxima(fft_amps);

    tmp = sortperm(fft_amps[peaks_indx])[(end - n + 1):end];
    tmp2 = peaks_indx[tmp];
    freqs = 2 * pi * fft_freqs[tmp2];
    amps = fft_amps[tmp2];
    append!(amps, fft_amps[1]);
    phases = zeros(length(freqs));

    return freqs, amps, phases
end

function set_env_parameter(env::training_parameters, param::String,
        new_value::Any)
    training_parameters_labels = ["t0", "ts", "tf", "phi", "u0", "p", "f", "O",
        "y_samples", "opt", "tolerances", "tf_tr", "data", "M", "d_time",
        "add_loss"];

    arguments_array = [env.t0, env.ts, env.tf, env.phi, env.u0, env.p, env.f,
        env.O, env.y_samples, env.opt, env.tolerances, env.tf_tr, env.data,
        env.M, env.d_time, env.add_loss];
    for i = 1:1:length(training_parameters_labels)
        if training_parameters_labels[i] == param
            arguments_array[i] = new_value;
        end
    end

    a = arguments_array;
    new_env = training_parameters(a[1], a[2], a[3], a[4], a[5], a[6], a[7],
        a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16]);
    return new_env;
end

function set_gain_env_parameter(env::gain_training_parameters, param::String,
        new_value::Any)
    training_parameters_labels = ["t0", "ts", "tf", "phi", "u0", "p", "hu0",
        "hgo_type", "gain_type", "coeffs", "disturbance", "opt", "tolerances",
        "dynamics", "add_loss"];

    arguments_array = [env.t0, env.ts, env.tf, env.phi, env.p, env.u0,
        env.hu0, env.hgo_type, env.gain_type, env.coeffs, env.disturbance,
        env.opt, env.tolerances, env.dynamics, env.add_loss];

    for i = 1:1:length(training_parameters_labels)
        if training_parameters_labels[i] == param
            arguments_array[i] = new_value;
        end
    end

    a = arguments_array;
    new_env = gain_training_parameters(a[1], a[2], a[3], a[4], a[5], a[6], a[7],
        a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15]);
    return new_env;
end

function set_ctrl_env_parameter(env::ctrl_training_parameters, param::String,
        new_value::Any)
    training_parameters_labels = ["alpha", "beta", "gamma", "data", "f", "u",
        "tolerances"];

    arguments_array = [env.alpha, env.beta, env.gamma, env.data, env.f, env.u,
        env.tolerances];

    for i = 1:1:length(training_parameters_labels)
        if training_parameters_labels[i] == param
            arguments_array[i] = new_value;
        end
    end

    a = arguments_array;
    new_env = ctrl_training_parameters(a[1], a[2], a[3], a[4], a[5], a[6],
        a[7]);
    return new_env;
end

function set_dt_env_parameter(env::dt_training_parameters, param::String,
        new_value::Any)
    training_parameters_labels = ["f", "h", "n", "opt", "y_samples",
        "add_loss"];

    arguments_array = [env.f, env.h, env.n, env.opt, env.y_samples,
        env.add_loss];

    for i = 1:1:length(training_parameters_labels)
        if training_parameters_labels[i] == param
            arguments_array[i] = new_value;
        end
    end

    a = arguments_array;
    new_env = dt_training_parameters(a[1], a[2], a[3], a[4], a[5], a[6]);
    return new_env;
end

function init_env(phi::Function, data::Matrix{Float64}, t0::Float64,
        tf::Float64, ts::Float64, opt;
        reltol::Float64 = 1e-8, abstol::Float64 = 1e-8)

    blank(a, b, c) = [];
    blank2(env, u, p, u0) = 0;
    env = training_parameters(t0, ts, tf, phi, [0.0], [0.0], blank, blank,
        [0.0], opt, [reltol, abstol], 0.0, data, 0, 0.0, blank2);

    return env;
end

function init_env(phi::Function, y_samples::Vector{Float64},
        t0::Float64, tf::Float64, ts::Float64, opt, n, l;
        reltol::Float64 = 1e-8, abstol::Float64 = 1e-8)

    blank(a, b, c) = [];
    blank2(env, u, p, u0) = 0;
    env = training_parameters(t0, ts, tf, phi, randn(n), randn(l), blank, blank,
        y_samples, opt, [reltol, abstol], 0.0, zeros(1, 1), 0, 0.0, blank2);

    return env;
end

function init_env(phi::Function, data::Matrix{Float64},
        t0::Float64, tf::Float64, ts::Float64, opt, n, l;
        reltol::Float64 = 1e-8, abstol::Float64 = 1e-8)

    blank(a, b, c) = [];
    blank2(env, u, p, u0) = 0;
    env = training_parameters(t0, ts, tf, phi, randn(n), randn(l), blank, blank,
        data[1, :], opt, [reltol, abstol], 0.0, data, 0, 0.0, blank2);

    return env;
end

function init_env(phi::Function, u0::Vector{Float64}, p::Vector{Float64},
        t0::Float64, tf::Float64, ts::Float64, opt;
        reltol::Float64 = 1e-8, abstol::Float64 = 1e-8)

    y_samples = get_output_samples(phi, u0, p, t0, tf, ts, [reltol, abstol]);
    blank(a, b, c) = [];
    blank2(env, u, p, u0) = 0;
    env = training_parameters(t0, ts, tf, phi, u0, p, blank, blank,
        y_samples, opt, [reltol, abstol], 0.0, zeros(1, 1), 0, 0.0, blank2);

    return env;
end

function init_env(f::Function, O::Function, y_samples::Vector{Float64},
        t0::Float64, tf::Float64, ts::Float64, opt, n, l;
        reltol::Float64 = 1e-8, abstol::Float64 = 1e-8)

    blank(a, b, c) = [];
    blank2(env, u, p, u0) = 0;
    env = training_parameters(t0, ts, tf, blank, randn(n), randn(l), f, O,
        y_samples, opt, [reltol, abstol], 0.0, zeros(1, 1), 0, 0.0, blank2);

    return env;
end

function init_env(f::Function, O::Function, u0::Vector{Float64},
        p::Vector{Float64}, t0::Float64, tf::Float64, ts::Float64, opt;
        reltol::Float64 = 1e-8, abstol::Float64 = 1e-8)

    y_samples = get_output_samples(f, O, u0, p, t0, tf, ts, [reltol, abstol]);
    blank(a, b, c) = [];
    blank2(env, u, p, u0) = 0;
    env = training_parameters(t0, ts, tf, blank, u0, p, f, O, y_samples, opt,
        [reltol, abstol], 0.0, zeros(1, 1), 0, 0.0, blank2);

    return env;
end

function init_env(y_samples::Vector{Float64}, t0::Float64, tf::Float64,
        ts::Float64, opt; reltol::Float64 = 1e-8, abstol::Float64 = 1e-8)

    blank(a, b, c) = [];
    blank2(env, u, p, u0) = 0;
    env = training_parameters(t0, ts, tf, blank, [0.0], [0.0], blank, blank,
        y_samples, opt, [reltol, abstol], 0.0, zeros(1, 1), 0, 0.0, blank2);

    return env;
end

function init_gain_env(phi::Function, u0::Vector{Float64}, hp::Vector{Float64},
        hu0::Matrix{Float64}, t0::Float64, tf::Float64, ts::Float64, opt;
        reltol::Float64 = 1e-8, abstol::Float64 = 1e-8)

    hgo_type = -1;
    gain_type = -1;
    coeffs = [];
    data = [];
    blank(p) = 0;
    blank2(u, p, t) = [];
    d(t) = 0;
    env = gain_training_parameters(t0, ts, tf, phi, hp, u0, hu0,
        hgo_type, gain_type, coeffs, d, opt, [reltol, abstol], blank2, blank);
end

function init_ctrl_env(alpha::Float64, beta::Float64, gamma::Float64,
        data::Matrix{Float64}, f::Function, u::Function;
        reltol::Float64 = 1e-8, abstol::Float64 = 1e-8)

    env = ctrl_training_parameters(alpha, beta, gamma, data, f, u,
        [reltol, abstol]);
end

function init_dt_env(f::Function, h::Function, n::Int64, opt,
        y_samples::Matrix{Float64})

    blank2(env, u, p, u0) = 0;
    env = dt_training_parameters(f, h, n, opt, y_samples, blank2);

    return env;
end

function get_sys_solution(env::training_parameters)
    if env.f(env.u0, env.p, 0) == []
        f = get_system_dynamics(env.phi, env.u0, env.p);
    else
        f = env.f;
    end
    sol = get_sol(f, env.u0, env.p, env.t0, env.tf, env.ts, env.tolerances)
end

function get_sys_solution(env::training_parameters, p::Vector{Float64})
    if env.f(env.u0, env.p, 0) == []
        f = get_system_dynamics(env.phi, env.u0, env.p);
    else
        f = env.f;
    end
    sol = get_sol(f, env.u0, p, env.t0, env.tf, env.ts, env.tolerances)
end

function get_sys_solution(env::training_parameters, p::Vector{Float64},
        u0::Vector{Float64})
    if env.f(env.u0, env.p, 0) == []
        f = get_system_dynamics(env.phi, env.u0, env.p);
    else
        f = env.f;
    end
    sol = get_sol(f, u0, p, env.t0, env.tf, env.ts, env.tolerances)
end

function get_lyapunov_derivative_values(env::ctrl_training_parameters,
        data::Matrix{Float64}, p::Vector{Float64})

    dV_values = zeros(size(data, 2));
    f = env.f;

    for i = 1:1:length(dV_values)
        x = data[:, i];
        dx = f(x, p);
        dV_values[i] = x' * dx;
    end

    return dV_values;
end

function get_ctrl_sol(env::ctrl_training_parameters, estp::Vector{Float64},
        u0::Vector{Float64}, tf::Float64)
    f(x, p, t) = env.f(x, p);
    prob = ODEProblem(f, u0, (0.0, tf));
    sol = solve(prob, p = estp, abstol = env.tolerances[1],
        reltol = env.tolerances[2]);

    u_vals = zeros(1, size(sol, 2));
    for i = 1:1:length(u_vals)
        u_vals[i] = env.u(sol[:, i], estp);
    end

    return sol.t, sol[:, :], u_vals
end

function get_dt_sys_sol(env::dt_training_parameters, estp::Vector{Float64},
        u0::Vector{Float64}, steps::Int64)
    sol = zeros(length(u0), steps);
    sol[:, 1] = u0;
    for i = 2:1:steps
        sol[:, i] = get_dt_step(env.f, sol[:, i - 1], estp, i);
    end

    return sol;
end

function get_dt_sys_sol(f::Function, estp::Vector{Float64},
        u0::Vector{Float64}, steps::Int64)
    sol = zeros(length(u0), steps);
    sol[:, 1] = u0;
    for i = 2:1:steps
        sol[:, i] = get_dt_step(f, sol[:, i - 1], estp, i);
    end

    return sol;
end

################################################################################

function get_system_dynamics(phi::Function, u0::Vector{Float64},
        p::Vector{Float64})
    n = length(u0);
    A = zeros(n, n);
    for i = 1:1:(n - 1)
        A[i, i + 1] = 1;
    end
    B = zeros(n, 1);
    B[n] = 1;

    f(u, p, t) = vec(A * u + B * phi(u, p, t));
    return f;
end

function get_sol(f, u0::Vector{Float64}, p::Vector{Float64}, t0::Float64,
        tf::Float64, ts::Float64, tolerances::Vector{Float64})
    prob = ODEProblem(f, u0, (t0, tf), saveat = range(t0, tf,
        length = Int(round((tf - t0) / ts))));
    sol = solve(prob, p = p, abstol = tolerances[1], reltol = tolerances[2],
        sensealg = InterpolatingAdjoint(autojacvec =
            ZygoteVJP(allow_nothing = true)));
    sol[:, :]
end

function get_output_samples(phi::Function, u0::Vector{Float64},
        p::Vector{Float64}, t0::Float64, tf::Float64, ts::Float64,
        tolerances::Vector{Float64})
    f = get_system_dynamics(phi, u0, p);

    sol = get_sol(f, u0, p, t0, tf, ts, tolerances);
    sol[1, :]
end

function get_output_samples(f::Function, O::Function, u0::Vector{Float64},
        p::Vector{Float64}, t0::Float64, tf::Float64, ts::Float64,
        tolerances::Vector{Float64})

    sol = get_sol(f, u0, p, t0, tf, ts, tolerances);
    sol_obs = get_obs_sol(sol, O, u0, p, t0, ts, tf);
    return sol_obs[1, :];
end

function get_obs_sol(sol::Matrix{Float64}, O::Function, u0::Vector{Float64},
        p::Vector{Float64}, t0::Float64, ts::Float64, tf::Float64)
    sol_obs = zeros(length(O(u0, p, 0)), size(sol, 2));
    t = t0:ts:tf;
    for i = 1:1:(length(t) - 1)
        sol_obs[:, i] = O(sol[:, i], p, t[i]);
    end
    return sol_obs;
end

function get_freq_sol(freqs::Vector{Float64}, phases::Vector{Float64},
        amps::Vector{Float64}, bias::Float64, t::Float64)
    return sum(amps .* sin.(freqs * t .+ phases)) + bias;
end

function get_dt_step(f, ut0::Vector{Float64}, p::Vector{Float64}, i::Int64)
    return f(ut0, p, i)
end

function findlocalmaxima(signal::Vector{Float64})
    inds = Int[]
    if signal[2] > signal[3]
        push!(inds, 2)
    end
    if length(signal) > 1
        for i = 3:(length(signal) - 1)
            if signal[i - 1] < signal[i] > signal[i + 1]
                push!(inds, i)
            end
        end
    end
    inds
end
