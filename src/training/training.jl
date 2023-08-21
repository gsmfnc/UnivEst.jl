################################################################################
#######################EXPORTED FUNCTIONS#######################################
################################################################################
"""
    fd_kin_training(fd_kin::forward_kinematics, data::Matrix{Float64},
        tfs::Vector{Float64}, its::Int, estp::Vector{Float64};
        opt::Function = Adam(1e-02))
"""
function fd_kin_training(fd_kin::forward_kinematics, data::Matrix{Float64},
        tfs::Vector{Float64}, its::Int, estp::Vector{Float64};
        opt = Adam(1e-02))
    global SUPPENV
    SUPPENV = fk_training_env(fd_kin, 0.0, data);

    @time begin
    for i = 1:1:length(tfs)
        SUPPENV = set_fk_training_val(SUPPENV, "tf_train", tfs[i]);

        pinit = ComponentArray(estp);
        adtype = Optimization.AutoZygote();
        optf = Optimization.OptimizationFunction((x, p) -> loss_fd(x),
            adtype);
        optprob = Optimization.OptimizationProblem(optf, pinit);

        @time begin
        res = Optimization.solve(optprob,
                                opt,
                                maxiters = its);
        end
        estp = res.u;
        println(estp)
    end
    end
    return estp;
end

"""
    periodical_signal_training(s::periodical_signal, data::Matrix{Float64},
        tf::Float64, its::Int, hbias::Float64, hamps::Vector{Float64},
        hphases::Vector{Float64}, hpuls::Vector{Float64};
        window_size::Float64 = 0.0, varying_iters::Vector{Int} = [0, 0],
        opt = Adam(1e-02), max_window_number::Int = 0,
        varying_adam_p::Vector{Float64} = [])
"""
function periodical_signal_training(s::periodical_signal, data::Matrix{Float64},
        tf::Float64, its::Int; nu::Int = 0,
        hbias::Float64 = 0.0, hamps::Vector = [],
        hphases::Vector = [], hpuls::Vector = [],
        window_size::Float64 = 0.0, varying_iters::Vector{Int} = [0, 0],
        opt = Adam(1e-02), max_window_number::Int = 0,
        varying_adam_p::Vector = [])

    if nu == 0 && (length(hamps) == 0 || length(hphases == 0) ||
            length(hpuls) == 0)
        return
    end

    if hbias == 0.0
        hbias = randn(1)[1] * 1e-03;
    end
    if length(hamps) == 0
        hamps = randn(nu) * 1e-02;
    end
    if length(hphases) == 0
        hphases = randn(nu) * 1e-02;
    end
    if length(hpuls) == 0
        hpuls = randn(nu) * 1e-02;
    end

    global SUPPENV
    SUPPENV = freq_training_env(s, s.t0, s.ts, tf, data);

    estp = vcat(hpuls, hphases, hamps, hbias);

    if varying_iters == [0, 0]
        varying_iters[1] = its;
        varying_iters[2] = its;
    end

    if window_size == 0.0
        F = tf;
    else
        F = window_size;
    end

    new_opt = opt;

    N = Int(floor(tf / F));
    tf_tr = F;

    @time begin
    for i = 1:1:N
        SUPPENV = set_freq_training_val(SUPPENV, "tf_tr", tf_tr);

        if length(varying_adam_p) > 0
            adam_p = varying_adam_p[1] -
                (varying_adam_p[1] - varying_adam_p[2]) / (N - 1) * (i - 1);
            new_opt = ADAM(adam_p);
        end

        if max_window_number != 0
            if i > window_size
                SUPPENV = set_freq_training_val(SUPPENV, "t0",
                    (i - window_size) * F);
            end
        end

        pinit = ComponentArray(estp);
        adtype = Optimization.AutoZygote();
        optf = Optimization.OptimizationFunction((x, p) -> loss_freq(x),
            adtype);
        optprob = Optimization.OptimizationProblem(optf, pinit);

        its_adam = varying_iters[1] - Int(round(
            (varying_iters[1] - varying_iters[2]) / (N - 1) * (i - 1)));

        @time begin
        res = Optimization.solve(optprob,
                                new_opt,
                                maxiters = its_adam);
        end
        estp = res.u;

        tf_tr = tf_tr + F;
    end
    end

    n = Int(floor(length(estp) - 1) / 3);

    puls = estp[1:n];
    phases = estp[(n + 1):(2 * n)];
    amps = estp[(2 * n + 1):(3 * n)];
    bias = estp[end];

    return bias, amps, phases, puls;
end
