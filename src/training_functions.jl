#EXPORTED FUNCTIONS#############################################################

function training_routine(env::training_parameters, tfs::Vector{Float64},
        estp::Vector{Float64}, its::Int)
    global SUPPENV
    SUPPENV = env;

    hu0 = estp[1:length(SUPPENV.u0)];
    hp = estp[(length(SUPPENV.u0) + 1):end];
    if SUPPENV.f(hu0, hp, 0) == []
        SUPPENV = set_env_parameter(SUPPENV, "f",
            get_system_dynamics(SUPPENV.phi, hu0, hp));
    end

    @time begin
    for i = 1:1:length(tfs)
        SUPPENV = set_env_parameter(SUPPENV, "tf_tr", tfs[i]);

        pinit = Lux.ComponentArray(estp);
        adtype = Optimization.AutoZygote();
        optf = Optimization.OptimizationFunction((x, p) -> loss(x), adtype);
        optprob = Optimization.OptimizationProblem(optf, pinit);

        @time begin
        res = Optimization.solve(optprob,
                                SUPPENV.opt,
                                maxiters = its);
        end
        estp = res.u;
        println(estp)
    end
    end
    return estp;
end

function freq_training_routine(env::training_parameters, F::Float64,
        tf::Float64, estp::Vector{Float64}, its::Int;
        adam_param::Vector{Float64} = [0.0, 0.0],
        window_size::Int = 0)
    freqs, phases, amps, bias =
        freq_training(env, F, tf, estp, its, [0, 0], adam_param, window_size);
    return freqs, phases, amps, bias;
end

function freq_training_routine(env::training_parameters, F::Float64,
        tf::Float64, estp::Vector{Float64}, its::Vector{Int};
        adam_param::Vector{Float64} = [0.0, 0.0],
        window_size::Int = 0)
    freqs, phases, amps, bias =
        freq_training(env, F, tf, estp, 0, its, adam_param, window_size);
    return freqs, phases, amps, bias;
end

function alg_training_routine(env::training_parameters, tfs::Vector{Float64},
        estp::Vector{Float64}, its::Int)
    global SUPPENV
    SUPPENV = env

    @time begin
    for i = 1:1:length(tfs)
        SUPPENV = set_env_parameter(SUPPENV, "tf_tr", tfs[i]);

        pinit = Lux.ComponentArray(estp);
        adtype = Optimization.AutoZygote();
        optf = Optimization.OptimizationFunction((x, p) -> loss_alg(x),
            adtype);
        optprob = Optimization.OptimizationProblem(optf, pinit);

        @time begin
        res = Optimization.solve(optprob,
                                SUPPENV.opt,
                                maxiters = its);
        end
        estp = res.u;
        println(estp)
    end
    end
    return estp;
end

function gain_training_routine(env::gain_training_parameters, hgo_type::Int,
                gain_type::Int, W0::Vector{Float64}, its::Int)
    global SUPPENV;
    SUPPENV = env;

    SUPPENV = set_gain_env_parameter(SUPPENV, "hgo_type", hgo_type);
    SUPPENV = set_gain_env_parameter(SUPPENV, "gain_type", gain_type);

    n = length(SUPPENV.u0);
    dynamics = get_system_dynamics(SUPPENV.phi, SUPPENV.u0, SUPPENV.p);
    hgo = get_hgo_dynamics();
    f(u, p, t) = [
        dynamics(u[1:n], SUPPENV.p, t)
        hgo(u[(n + 1):end], u[1] + SUPPENV.disturbance(t), p, SUPPENV.p, t)
    ];

    SUPPENV = set_gain_env_parameter(SUPPENV, "dynamics", f);

    @time begin
    pinit = Lux.ComponentArray(W0);
    adtype = Optimization.AutoZygote();
    optf = Optimization.OptimizationFunction((x, p) -> loss_gain(x), adtype);
    optprob = Optimization.OptimizationProblem(optf, pinit);
    res = Optimization.solve(optprob, SUPPENV.opt, maxiters = its);
    end

    return res.u;
end

function ctrl_training_routine(env::ctrl_training_parameters,
        p0::Vector{Float64})
    global SUPPENV;
    SUPPENV = env;

    res = Optim.optimize(loss_ctrl, p0);
    estp = res.minimizer;

    return estp;
end

################################################################################

function loss(p)
    hu0 = p[1:length(SUPPENV.u0)];
    hp = p[(length(SUPPENV.u0) + 1):end];

    sol = get_sol(SUPPENV.f, hu0, hp, SUPPENV.t0,
        SUPPENV.tf_tr, SUPPENV.ts, SUPPENV.tolerances; SUPPENV.ode_kwargs...);

    d_samples = Int(round((SUPPENV.d_time - SUPPENV.t0) / SUPPENV.ts)) + 1;

    vloss = 0;
    if SUPPENV.M <= 0
        if SUPPENV.O(hu0, hp, 0) == []
            for i = d_samples:1:size(sol, 2)
                vloss = vloss + abs(SUPPENV.y_samples[i] - sol[1, i]);
            end
        else
            for i = d_samples:1:size(sol, 2)
                tmp = SUPPENV.O(sol[:, i], hp, i * SUPPENV.ts);
                vloss = vloss + abs(SUPPENV.y_samples[i] - tmp[1]);
            end
        end
    else
        if SUPPENV.O(hu0, hp, 0) == []
            for i = d_samples:1:size(sol, 2)
                vloss = vloss +
                    sum(abs.(SUPPENV.data[1:SUPPENV.M, i] -
                        sol[1:SUPPENV.M, i]));
            end
        else
            for i = d_samples:1:size(sol, 2)
                vloss = vloss +
                    sum(abs.(SUPPENV.data[1:SUPPENV.M, i] -
                        SUPPENV.O(sol[1:SUPPENV.M, i], hp, i * SUPPENV.ts)));
            end
        end
    end

    vloss = vloss + SUPPENV.add_loss(SUPPENV, sol, hp, hu0);

    return vloss;
end

function loss_alg(p)
    d_samples = Int(round((SUPPENV.d_time - SUPPENV.t0) / SUPPENV.ts)) + 1;
    N = Int(round((SUPPENV.tf_tr - SUPPENV.t0) / SUPPENV.ts)) + 1;

    vloss = 0;
    for i = d_samples:1:N
        vloss = vloss + sum(abs.(SUPPENV.data[i, :] -
            SUPPENV.phi(0, p, (i - 1) * SUPPENV.ts)));
    end

    return vloss;
end

function freq_training(env::training_parameters, F::Float64,
        tf::Float64, estp::Vector{Float64}, its::Int, max_iters::Vector{Int},
        adam_param::Vector{Float64}, window_size::Int)
    global SUPPENV
    SUPPENV = env;

    if max_iters == [0, 0]
        max_iters[1] = its;
        max_iters[2] = its;
    end

    N = Int(floor(tf / F));
    tf_tr = F;
    @time begin
    for i = 1:1:N
        SUPPENV = set_env_parameter(SUPPENV, "tf_tr", tf_tr);

        if adam_param != [0.0, 0.0]
            adam_p = adam_param[1] - (adam_param[1] - adam_param[2]) /
                (N - 1) * (i - 1);
            SUPPENV = set_env_parameter(SUPPENV, "opt", ADAM(adam_p));
        end

        if window_size != 0
            if i > window_size
                SUPPENV = set_env_parameter(SUPPENV, "t0",
                    (i - window_size) * F);
            end
        end

        pinit = Lux.ComponentArray(estp);
        adtype = Optimization.AutoZygote();
        optf = Optimization.OptimizationFunction((x, p) -> loss_freq(x),
            adtype);
        optprob = Optimization.OptimizationProblem(optf, pinit);

        its_adam = max_iters[1] - Int(round(
            (max_iters[1] - max_iters[2]) / (N - 1) * (i - 1)));

        @time begin
        res = Optimization.solve(optprob,
                                SUPPENV.opt,
                                maxiters = its_adam);
        end
        estp = res.u;

        tf_tr = tf_tr + F;
    end
    end

    n = Int(floor(length(estp) - 1) / 3);

    freqs = estp[1:n];
    phases = estp[(n + 1):(2 * n)];
    amps = estp[(2 * n + 1):(3 * n)];
    bias = estp[end];

    return freqs, phases, amps, bias;
end

function loss_freq(p)
    n = Int(floor(length(p) - 1) / 3);

    freqs = p[1:n];
    phases = p[(n + 1):(2 * n)];
    amps = p[(2 * n + 1):(3 * n)];
    bias = p[end];

    d_samples = Int(round(SUPPENV.t0 / SUPPENV.ts)) + 1;
    N = Int(round(SUPPENV.tf_tr / SUPPENV.ts)) + 1;

    vloss = 0;
    for i = d_samples:1:N
        vloss = vloss + abs(SUPPENV.y_samples[i] -
            get_freq_sol(freqs, phases, amps, bias, (i - 1) * SUPPENV.ts));
    end

    return vloss;
end

function loss_gain(p)
    n = length(SUPPENV.u0);

    vloss = 0;
    for i = 1:1:size(SUPPENV.hu0, 2)
        sol = get_sol(SUPPENV.dynamics, [SUPPENV.u0; SUPPENV.hu0[:, i]], p,
            SUPPENV.t0, SUPPENV.tf, SUPPENV.ts, SUPPENV.tolerances;
            SUPPENV.ode_kwargs...);
        x = sol[1:n, :];
        hx = sol[(n + 1):end, :];
        vloss = vloss + sum(abs, x - hx);
    end

    vloss = vloss + SUPPENV.add_loss(p);

    return vloss;
end

function loss_ctrl(p)
    alpha = SUPPENV.alpha;
    beta = SUPPENV.beta;
    gamma = SUPPENV.gamma;

    data = SUPPENV.data;
    f = SUPPENV.f;
    u = SUPPENV.u;

    l = 0;
    for i = 1:1:size(data, 2)
        x = data[:, i];
        dx = f(x, p);
        l = l + tanh(alpha * (x' * dx + beta)) + gamma * abs(u(x, p));
    end

    return l;
end
