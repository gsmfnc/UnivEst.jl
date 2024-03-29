################################################################################
#######################EXPORTED FUNCTIONS#######################################
################################################################################
"""
    gradient_inversion_algorithm(sys::system, sysobs::system_obs,
        K::Function, obs_map::Function,
        gain_params::Vector{Float64}, tfin::Float64, its::Int;
        callback::Bool = false, gain_type::Int = UnivEst.TIMEVARYING_GAIN,
        hgo_type::Int = UnivEst.CLASSICALHGO, coeffs::Vector = [],
        estp0::Vector = [], opt = Adam(1e-02))
"""
function gradient_inversion_algorithm(sys::system, sysobs::system_obs,
        K::Function, obs_map::Function, d::Function,
        gain_params::Vector{Float64}, tfin::Float64, its::Int;
        callback::Bool = false, gain_type::Int = UnivEst.TIMEVARYING_GAIN,
        hgo_type::Int = UnivEst.CLASSICALHGO, coeffs::Vector = [],
        estp0::Vector = [], opt = Adam(1e-02))
    if hgo_type != UnivEst.CLASSICALHGO
        println("hgo_type = UnivEst.CLASSICALHGO: not implemented yet");
    end
    if gain_type != UnivEst.TIMEVARYING_GAIN
        println("gain_type = UnivEst.TIMEVARYING_GAIN: not implemented yet");
    end

    global SUPPENV
    n = length(sys.u0);
    m = length(sysobs.u0);
    gain_func = get_gain_func(gain_type);
    phi(u, t) = sysobs.phi(u, sysobs.p, t);
    hgo = get_hgo_dynamics(hgo_type, m, coeffs, gain_func, phi);
    grad_alg(x, xi, p, t) = (sign(t - p[2]) + 1) / 2 * p[1] * K(x, t) *
        (xi - obs_map(x, t));
    dynamics(u, p, t) = [
        sys.f(u[1:n], sys.p, t)
        hgo(u[(n + 1):(n + m)], u[1] + d(t), p[3:end], t)
        grad_alg(u[(n + m + 1):end], u[(n + 1):(n + m)], p, t)
    ];
    SUPPENV = grad_inv_env(dynamics, n, m, sys.t0, tfin, sys.ts, sys.u0,
        sys.tolerances);

    estp = vcat(estp0, gain_params);
    if length(estp0) == 0
        estp = vcat([1e02, 1.0], gain_params);
    end

    if !callback
        estp = optimize_loss(estp, loss_invalg, opt, its);
    else
        estp = optimize_loss(estp, loss_invalg, opt, its, callback = callback,
            callbackfunc = gain_callback);
    end

    return estp;
end

"""
    inverse_training(N3::Function, obs_map::Function, its::Int;
        opt = Adam(1e-02), estp0::Vector = [], data = [], Nsamples::Int = 100)
"""
function inverse_training(N3::Function, obs_map::Function, its::Int, N::Int,
        n::Int;
        opt = Adam(1e-02), estp0::Vector = [], data = [], Nsamples::Int = 100)
    if length(data) == 0
        samples = randn(n, Nsamples);
    else
        samples = data;
    end

    global SUPPENV
    SUPPENV = inverse_env(data, N3, obs_map);

    estp = estp0;
    if length(estp0) == 0
        estp = randn(N * n + N * n + n * n) / n;
    end

    mean_err = loss_inv(estp);
    println("initial mean error:", mean_err)

    estp = optimize_loss(estp, loss_inv, opt, its);

    mean_err = loss_inv(estp);
    println("mean error:", mean_err)

    return estp;
end

"""
    pretraining(g::Function, Lfmh::Function, obs_map::Function, its::Int,
        n::Int;
        opt = Adam(1e-02), estp0::Vector = [], data = [],
        Nsamples::Int = 100)
"""
function pretraining(g::Function, Lfmh::Function, obs_map::Function, its::Int,
        N::Int, n::Int;
        opt = Adam(1e-02), estp0::Vector = [], data = [],
        Nsamples::Int = 100)

    if length(data) == 0
        samples = randn(n, Nsamples);
    else
        samples = data;
    end

    global SUPPENV
    SUPPENV = pretraining_env(samples, g, Lfmh, obs_map);

    estp = estp0;
    if length(estp0) == 0
        estp = randn(N + N * n + n) / n;
    end

    mean_err = loss_pre(estp);
    println("initial mean error:", mean_err)

    estp = optimize_loss(estp, loss_pre, opt, its);

    mean_err = loss_pre(estp);
    println("mean error:", mean_err)

    return estp;
end

"""
    gain_training(sys::system_obs, tfin::Float64, its::Int, d::Function,
        ics::Matrix{Float64};
        estW0::Vector = [], opt = Adam(1e-01),
        dtime::Float64 = 0.0, save::Bool = false,
        callback::Bool = false, gain_type::Int = UnivEst.TIMEVARYING_GAIN,
        hgo_type::Int = UnivEst.CLASSICALHGO,
        coeffs::Vector = [], S::Vector = [])
"""
function gain_training(sys::system_obs, tfin::Float64, its::Int, d::Function,
        ics::Matrix{Float64};
        estW0::Vector = [], opt = Adam(1e-01),
        dtime::Float64 = 0.0, save::Bool = false,
        callback::Bool = false, gain_type::Int = UnivEst.TIMEVARYING_GAIN,
        hgo_type::Int = UnivEst.CLASSICALHGO,
        coeffs::Vector = [], S::Vector = [])
    d_samples = Int(round((dtime - sys.t0) / sys.ts)) + 1;

    n = length(sys.u0);
    if hgo_type == UnivEst.MIN_CASCADE && length(S) < n
        println("A valid vector of saturations S must be supplied.");
        return;
    end

    global SUPPENV
    f = get_system_dynamics(sys.phi, sys.u0, sys.p);
    gain_func = get_gain_func(gain_type);
    phi(u, t) = sys.phi(u, sys.p, t);
    if hgo_type == UnivEst.CLASSICALHGO
        hgo = get_hgo_dynamics(hgo_type, n, coeffs, gain_func, phi);
        dynamics1(u, p, t) = [
            f(u[1:n], sys.p, t)
            hgo(u[(n + 1):end], u[1] + d(t), p, t)
        ];
        SUPPENV = gain_training_env(dynamics1, n, sys.t0, tfin, sys.ts,
            sys.tolerances, d_samples, ics, hgo_type, S, coeffs, gain_type);
    else
        hgo = get_min_cascade_dynamics(n, coeffs, gain_func, S, phi);
        dynamics2(u, p, t) = [
            f(u[1:n], sys.p, t)
            hgo(u[(n + 1):end], u[1] + d(t), p, t)
        ];
        SUPPENV = gain_training_env(dynamics2, n, sys.t0, tfin, sys.ts,
            sys.tolerances, d_samples, ics, hgo_type, S, coeffs, gain_type);
    end

    if length(estW0) == 0
        if gain_type == UnivEst.INCREASING_GAIN
            estp = [50.0, 0.0];
        end
        if gain_type == UnivEst.DECREASING_GAIN
            estp = [50.0, 48.0, 5.0];
        end
        if gain_type == UnivEst.TIMEVARYING_GAIN
            estp = [50.0, 0.0, 48.0, 5.0];
        end
    else
        estp = estW0;
    end

    open("TRAININGSAVE.CSV", "w") do io
        writedlm(io, "_")
        writedlm(io, estp')
    end

    if !callback
        estp = optimize_loss(estp, loss_gain, opt, its);
    else
        estp = optimize_loss(estp, loss_gain, opt, its, callback = callback,
            callbackfunc = gain_callback);
    end

    return estp;
end

"""
    sys_training(sys::system, data::Matrix{Float64}, tfs::Vector{Float64},
        its::Int;
        estu0::Vector = [], estp0::Vector = [], opt = Adam(1e-02),
        dtime::Float64 = 0.0, save::Bool = false,
        callback::Bool = false, batch_size::Float64 = 0.0, batch_no::Int = 0)
"""
function sys_training(sys::system, data::Matrix{Float64}, tfs::Vector{Float64},
        its::Int;
        estu0::Vector = [], estp0::Vector = [], opt = Adam(1e-02),
        dtime::Float64 = 0.0, save::Bool = false,
        callback::Bool = false, batch_size::Float64 = 0.0, batch_no::Int = 0)
    if batch_size == 0.0 && batch_no == 0
        lossf = loss_sys;
		batch_indexes = [0];
    else
        lossf = batch_loss_sys;
		maxN = Int(round(tfs[1] / sys.ts));
		batch_size_i = Int(round(batch_size / sys.ts));
		indx = rand(1:(maxN - batch_size_i), 1, batch_no);

		batch_indexes = indx[1]:(indx[1] + batch_size_i - 1);
		for i = 2:length(indx)
			batch_indexes = vcat(batch_indexes,
				indx[i]:(indx[i] + batch_size_i - 1));
		end
		batch_indexes = sort(batch_indexes);
    end

    d_samples = Int(round((dtime - sys.t0) / sys.ts)) + 1;
    mxs = maximum(abs, data[d_samples:end, :], dims = 1);

    global SUPPENV
    SUPPENV = sys_training_env(sys.f, sys.h, sys.obs_map, length(sys.u0),
        sys.t0, sys.tf, sys.ts, sys.tolerances, d_samples, data,
		vec(batch_indexes), mxs);

    if length(estu0) == 0
        estu0 = vec(randn(1, length(sys.u0)) * 1e-02);
    end
    if length(estp0) == 0
        estp0 = vec(randn(1, length(sys.p)) * 1e-02);
    end

    estp = vcat(estu0, estp0);

    open("TRAININGSAVE.CSV", "w") do io
        writedlm(io, "_")
        writedlm(io, estp')
    end

    N = length(tfs);
    if save
        times = zeros(N, 1);
        estps = zeros(length(estp), N);
    end

    for i = 1:1:length(tfs)
        SUPPENV = set_sys_training_env_val(SUPPENV, "tf_tr", tfs[i]);

        if save
            if !callback
                times[i] = @elapsed estp =
                    optimize_loss(estp, lossf, opt, its)
            else
                times[i] = @elapsed estp =
                    optimize_loss(estp, lossf, opt, its, callback = callback)
            end
            estps[:, i] = estp;
        else
            if !callback
                estp = optimize_loss(estp, lossf, opt, its);
            else
                estp = optimize_loss(estp, lossf, opt, its,
                    callback = callback);
            end
        end
    end

    if save
        return estp[1:length(sys.u0)], estp[(length(sys.u0) + 1):end], times,
            estps;
    else
        return estp[1:length(sys.u0)], estp[(length(sys.u0) + 1):end];
    end
end

"""
    sysobs_training(sys::system_obs, data::Matrix{Float64},
        tfs::Vector{Float64}, its::Int;
        estu0::Vector = [], estp0::Vector = [], opt = Adam(1e-02),
        dtime::Float64 = 0.0, save::Bool = false, callback::Bool = false,
        fixed_ic::Bool = false)
"""
function sysobs_training(sys::system_obs, data::Matrix{Float64},
        tfs::Vector{Float64}, its::Int;
        estu0::Vector = [], estp0::Vector = [], opt = Adam(1e-02),
        dtime::Float64 = 0.0, save::Bool = false, callback::Bool = false,
        fixed_ic::Bool = false)

    if length(estu0) == 0
        estu0 = vec(randn(1, length(sys.u0)) * 1e-02);
    end
    if length(estp0) == 0
        estp0 = vec(randn(1, length(sys.p)) * 1e-02);
    end

    estp = vcat(estu0, estp0);

    global SUPPENV
    f = get_system_dynamics(sys.phi, sys.u0, sys.p);
    d_samples = Int(round((dtime - sys.t0) / sys.ts)) + 1;
    SUPPENV = sysobs_training_env(f, sys.obs_map, length(sys.u0), sys.t0,
        sys.tf, sys.ts, sys.tolerances, d_samples, data, vec(estu0), fixed_ic);

    open("TRAININGSAVE.CSV", "w") do io
        writedlm(io, "_");
        writedlm(io, estp');
    end

    N = length(tfs);
    if save
        times = zeros(N, 1);
        estps = zeros(length(estp), N);
    end

    for i = 1:1:length(tfs)
        SUPPENV = set_sysobs_training_env_val(SUPPENV, "tf_tr", tfs[i]);

        if save
            if !callback
                times[i] = @elapsed estp =
                    optimize_loss(estp, loss_sysobs, opt, its);
            else
                times[i] = @elapsed estp =
                    optimize_loss(estp, loss_sysobs, opt, its,
                        callback = callback);
            end
            estps[:, i] = estp;
        else
            if !callback
                estp = optimize_loss(estp, loss_sysobs, opt, its);
            else
                estp = optimize_loss(estp, loss_sysobs, opt, its,
                    callback = callback);
            end
        end
    end

    if save
        return estp[1:length(sys.u0)], estp[(length(sys.u0) + 1):end], times,
            estps;
    else
        return estp[1:length(sys.u0)], estp[(length(sys.u0) + 1):end];
    end
end

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
        its::Int;
        callback::Bool = false, callbackfunc::Function = default_callback)
    pinit = ComponentArray(estp);
    adtype = Optimization.AutoZygote();
    optf = Optimization.OptimizationFunction((x, p) -> loss_func(x),
        adtype);
    optprob = Optimization.OptimizationProblem(optf, pinit);

    if !callback
        @time begin
        res = Optimization.solve(optprob,
                                opt,
                                maxiters = its);
        end
    else
        @time begin
        res = Optimization.solve(optprob,
                                opt,
                                maxiters = its,
                                callback = callbackfunc);
        end
    end
    estp = res.u;

    open("TRAININGSAVE.CSV", "a") do io
        writedlm(io, "_")
        writedlm(io, estp')
    end

    return estp
end

"""
    default_callback(p, l, pred)
"""
function default_callback(p, l, pred)
    println("p = ", p)
    println("loss = ", l)
    plt = plot(SUPPENV.data[:, 1])
    plt = plot!(pred[1, :])
    display(plt)

    return false;
end

"""
    gain_callback(p, l, pred)
"""
function gain_callback(p, l, pred)
    println("p = ", p)
    println("loss = ", l)
    #plt = plot(pred[1:SUPPENV.n, :]')
    #plt = plot!(pred[(SUPPENV.n + 1):(2 * SUPPENV.n), :]')
    #display(plt)

    return false;
end
