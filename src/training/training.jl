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

    open("TRAININGSAVE.CSV", "w") do io
        writedlm(io, "_")
        writedlm(io, estp')
    end

    @time begin
    for i = 1:1:length(tfs)
        SUPPENV = set_fk_training_val(SUPPENV, "tf_train", tfs[i]);
        estp = optimize_loss(estp, loss_fd, opt, its);
    end
    end
    return estp;
end

"""
    periodical_signal_training(s::periodical_signal, data::Matrix{Float64},
        tf::Float64; its::Int = 100, nu::Int = 0,
        hbias::Float64 = 0.0, hamps::Vector = [],
        hphases::Vector = [], hpuls::Vector = [],
        window_size::Float64 = 0.0, varying_iters::Vector{Int} = [0, 0],
        opt = Adam(1e-02), max_window_number::Int = 0,
        varying_adam_p::Vector = [], save::Bool = false, t0::Float64 = 0.0)
"""
function periodical_signal_training(s::periodical_signal, data::Matrix{Float64},
        tf::Float64; its::Int = 100, nu::Int = 0,
        hbias::Float64 = 0.0, hamps::Vector = [],
        hphases::Vector = [], hpuls::Vector = [],
        window_size::Float64 = 0.0, varying_iters::Vector{Int} = [0, 0],
        opt = Adam(1e-02), max_window_number::Int = 0,
        varying_adam_p::Vector = [], save::Bool = false, t0::Float64 = 0.0)

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
    SUPPENV = freq_training_env(s, t0 + s.t0, s.ts, tf, data);

    estp = vcat(hpuls, hphases, hamps, hbias);

    open("TRAININGSAVE.CSV", "w") do io
        writedlm(io, "_")
        writedlm(io, estp')
    end

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
    tf_tr = SUPPENV.t0 + F;

    if save
        times = zeros(N, 1);
        estps = zeros(length(estp), N);
    end

    @time begin
    for i = 1:1:N
        SUPPENV = set_freq_training_val(SUPPENV, "tf_tr", tf_tr);

        if length(varying_adam_p) > 0
            adam_p = varying_adam_p[1] -
                (varying_adam_p[1] - varying_adam_p[2]) / (N - 1) * (i - 1);
            new_opt = ADAM(adam_p);
        end

        if max_window_number != 0
            if i > max_window_number
                SUPPENV = set_freq_training_val(SUPPENV, "t0",
                    t0 + (i - max_window_number) * F);
            end
        end

        its_adam = varying_iters[1] - Int(round(
            (varying_iters[1] - varying_iters[2]) / (N - 1) * (i - 1)));
        if save
            times[i] = @elapsed estp =
                optimize_loss(estp, loss_freq, new_opt, its_adam)
            estps[:, i] = estp;
        else
            estp = optimize_loss(estp, loss_freq, new_opt, its_adam)
        end

        tf_tr = tf_tr + F;
    end
    end

    if save
        puls, phases, amps, bias = find_infos_from_estp(estp);
        return puls, phases, amps, bias, estps, times;
    else
        return find_infos_from_estp(estp);
    end
end

"""
    find_infos_from_estp(estp::Vector{Float64})

Returns pulsations, amplitudes, phases and bias starting from the vector 'estp'
in the file TRAININGSAVE.CSV.
"""
function find_infos_from_estp(estp::Vector{Float64})
    n = Int(floor(length(estp) - 1) / 3);

    puls = estp[1:n];
    phases = estp[(n + 1):(2 * n)];
    amps = estp[(2 * n + 1):(3 * n)];
    bias = estp[end];

    return puls, phases, amps, bias
end

################################################################################
##############################NOT EXPORTED######################################
################################################################################
"""
    optimize_loss(estp::Vector{Float64}, loss_func::Function, opt,
        its::Int)
"""
function optimize_loss(estp::Vector{Float64}, loss_func::Function, opt,
        its::Int)
    pinit = ComponentArray(estp);
    adtype = Optimization.AutoZygote();
    optf = Optimization.OptimizationFunction((x, p) -> loss_func(x),
        adtype);
    optprob = Optimization.OptimizationProblem(optf, pinit);

    @time begin
    res = Optimization.solve(optprob,
                            opt,
                            maxiters = its);
    end
    estp = res.u;
    open("TRAININGSAVE.CSV", "a") do io
        writedlm(io, "_")
        writedlm(io, estp')
    end
    return estp
end
