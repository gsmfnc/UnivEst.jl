################################################################################
###############################EXPORTED FUNCTIONS###############################
################################################################################

function test_hgo(epsilon, phi, p, hphi, hp, u0, hu0, t0, tf, ts;
        hgo_type = UnivEst.CLASSICALHGO, coeffs = [], abstol = 1e-8,
        reltol = 1e-8)
    n = length(u0);
    L = length(p);

    f = get_system_dynamics(phi, u0, p);
    hgo = get_hgo_dynamics(hgo_type, n - 1, coeffs, epsilon, hphi);

    full_dynamics(u, theta, t) = [
        f(u[1:n], theta[1:L], t)
        hgo(u[(n + 1):end], u[1], theta[(L + 1):end], t)
    ];
    tolerances = [abstol, reltol];
    sol = get_sol(full_dynamics, vcat(u0, hu0), vcat(p, hp), t0, tf, ts,
        tolerances);
    x_sol = sol[1:n, :];
    hx_sol = sol[(n + 1):end, :];
    return x_sol, hx_sol;
end

function estimate_time_derivatives(env::training_parameters, hgo_type::Int,
        epsilon::Vector{Float64}, N::Int; k::Vector = [], Ts::Float64 = 0.0)
    hgo = get_hgo_dynamics(hgo_type, N, k, epsilon);

    if Ts != 0.0 && env.u0 == [] && env.p == []
        println("You cannot choose a different sampling time.")
        return
    end

    if Ts == 0.0
        ts = env.ts;
        y_samples = env.y_samples;
    else
        ts = Ts;
        if env.phi(env.u0, env.p, 0) != []
            y_samples = get_output_samples(env.phi, env.u0, env.p, env.t0,
                env.tf, ts, env.tolerances);
        else
            if env.f(env.u0, env.p, 0) != []
                y_samples = get_output_samples(env.f, env.O, env.u0, env.p,
                    env.t0, env.tf, env.ts, env.tolerances);
            end
        end
    end

    global HGO
    HGO = hgo_infos(hgo, y_samples, env.t0, ts)

    m = 0;
    if hgo_type == UnivEst.CLASSICALHGO
        m = N + 1;
    end
    if hgo_type == UnivEst.M_CASCADE
        m = 2 * N + 1;
    end
    if hgo_type == UnivEst.CASCADE
        m = 2 * N;
    end

    prob = ODEProblem(estimate_time_derivatives_dynamics,
        zeros(m), (env.t0, env.tf), saveat = range(env.t0, env.tf,
            length = Int(round((env.tf - env.t0) / ts))));
    sol = solve(prob, abstol = env.tolerances[1], reltol = env.tolerances[2],
        maxiters = 1e8);
    downsampling(env, sol[:, :], ts)
end

function extract_estimates(data::Matrix{Float64}, hgo_type::Int)
    if hgo_type == UnivEst.M_CASCADE
        N1 = Int(ceil(size(data, 1) / 2));
        N2 = size(data, 2);

        ndata1 = zeros(N1, N2);
        for i = 1:1:N2
            for j = 1:1:N1
                ndata1[j, i] = data[1 + 2 * (j - 1), i];
            end
        end

        ndata2 = zeros(N1, N2);
        for i = 1:1:N2
            for j = 1:1:(N1 - 1)
                ndata2[j, i] = data[1 + 2 * (j - 1), i];
            end
            ndata2[end, i] = data[1 + 2 * (N1 - 2) + 1, i];
        end

        ndata3 = zeros(N1, N2);
        for i = 1:1:N2
            ndata3[1, i] = data[1, i];
            for j = 1:1:(N1 - 1)
                ndata3[j + 1, i] = data[2 + 2 * (j - 1), i];
            end
        end

        ndata4 = zeros(N1, N2);
        for i = 1:1:N2
            ndata4[1, i] = data[1, i];
            for j = 1:1:(N1 - 2)
                ndata4[j + 1, i] = data[2 + 2 * (j - 1), i];
            end
            ndata4[end, i] = data[2 + 2 * (N1 - 2) + 1, i];
        end
        return [ndata1, ndata2, ndata3, ndata4]
    end
    if hgo_type == UnivEst.CASCADE
        N1 = Int(size(data, 1) / 2) + 1;
        N2 = size(data, 2);

        ndata1 = zeros(N1, N2);
        for i = 1:1:N2
            for j = 1:1:(N1 - 1)
                ndata1[j, i] = data[1 + 2 * (j - 1), i];
            end
            ndata1[end, i] = data[1 + 2 * (N1 - 2) + 1, i];
        end

        ndata2 = zeros(N1, N2);
        for i = 1:1:N2
            ndata2[1, i] = data[1, i];
            for j = 1:1:(N1 - 1)
                ndata2[j + 1, i] = data[2 + 2 * (j - 1), i];
            end
        end

        return [ndata1, ndata2]
    end
    return []
end

function test_timevarying_hgo(env, W, u0, hu0, t0, tf, ts, hgo_type, gain_type;
        abstol = 1e-08, reltol = 1e-8)
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
    sol = get_sol(f, [u0; hu0], W, t0, tf, ts, [abstol, reltol]);
    x = sol[1:n, :];
    hx = sol[(n + 1):end, :];

    return x, hx;
end

function plot_gain(W, gain_type)
    gain = get_timevarying_gain(gain_type);
    times = 0.0:1e-02:(W[end] * 5.0);

    gain_vec = zeros(length(times), 1);
    for i = 1:1:length(times)
        gain_vec[i] = gain(W, times[i]);
    end

    return plot(times, gain_vec);
end

################################################################################
################################################################################
################################################################################

function downsampling(env::training_parameters, data::Matrix{Float64},
        ts::Float64)
    N1 = Int(round((env.tf - env.t0) / ts));
    N2 = Int(round((env.tf - env.t0) / env.ts));
    N = Int(round(N1 / N2));
    dataset = zeros(size(data, 1), N2);
    for i = 1:1:N2
        dataset[:, i] = data[:, (i - 1) * N + 1];
    end
    return dataset
end

function estimate_time_derivatives_dynamics(du, u, p, t)
    y = HGO.samples[min(length(HGO.samples),
        Int(floor((t - HGO.t0) / HGO.ts)) + 1)];
    du .= HGO.dynamics(u, y);
end

struct hgo_infos
    dynamics::Function
    samples::Vector{Float64}
    t0::Float64
    ts::Float64
end

function get_hgo_dynamics(hgo_type::Int, n::Int, coeffs::Vector,
        epsilon::Float64)
    return get_hgo_dynamics(hgo_type, n, coeffs, [epsilon]);
end

function get_hgo_dynamics(hgo_type::Int, n::Int, coeffs::Vector,
        epsilon::Vector{Float64})
    if hgo_type == UnivEst.CLASSICALHGO
        A, B = get_hgo_matrices(hgo_type, n, coeffs, epsilon);
        hgo(u, y) = vec(A * u + B * y);
        return hgo
    end
    if hgo_type == UnivEst.M_CASCADE
        A, B = get_hgo_matrices(hgo_type, n, coeffs, epsilon);
        cascade(u, y) = vec(A * u + B * y);
        return cascade;
    end
    if hgo_type == UnivEst.CASCADE
        A, B = get_hgo_matrices(hgo_type, n, coeffs, epsilon);
        astmar(u, y) = vec(A * u + B * y);
        return astmar;
    end
end

function get_hgo_dynamics(hgo_type::Int, n::Int, coeffs::Vector,
        epsilon::Float64, phi::Function)
    return get_hgo_dynamics(hgo_type, n, coeffs, [epsilon], phi);
end

function get_hgo_dynamics(hgo_type::Int, n::Int, coeffs::Vector,
        epsilon::Vector{Float64}, phi::Function)
#remove everything, call get_hgo_matrices and only leave the function def
    if hgo_type == UnivEst.CLASSICALHGO
        A, B = get_hgo_matrices(hgo_type, n, coeffs, epsilon);
        H = zeros(N, 1);
        H[end] = 1;
        hgo(u, y, p, t) = vec(A * u + B * y + H * phi(u, p, t));
        return hgo
    end
    if hgo_type == UnivEst.CASCADE
        A, B = get_hgo_matrices(hgo_type, n, coeffs, epsilon);
        H = zeros(tmp_ind, 1);
        H[end] = 1;
        astmar(u, y, p, t) = vec(A * u + B * y + H * phi(u, p, t));
        return astmar;
    end
end

sigma(x) = exp(x) / (1 + exp(x));
function get_timevarying_gain(gain_type::Int)
    if gain_type == UnivEst.TIMEVARYING_GAIN
        gain1(W, t) = W[1] * sigma(t - W[2]) - W[3] * sigma(t - W[4]);
        return gain1;
    end
    if gain_type == UnivEst.INCREASING_GAIN
        gain2(W, t) = W[1] * sigma(t - W[2]);
        return gain2;
    end
    if gain_type == UnivEst.DECREASING_GAIN
        gain3(W, t) = W[1] - W[2] * sigma(t - W[3]);
        return gain3;
    end
end

function get_hgo_dynamics()
#call get_hgo_matrices
    N = length(SUPPENV.u0);
    if SUPPENV.hgo_type == UnivEst.CLASSICALHGO
        if length(SUPPENV.coeffs) == 0
            COEFFS_TABLE = [
                [1.414, 1.0],
                [2.0, 2.0, 1.0],
                [2.613, 3.414, 2.613, 1.000],
                [3.236, 5.240, 5.236, 3.326, 1.000],
                [3.86, 7.46, 9.14, 7.46, 3.86, 1.0],
                [4.49, 10.10, 14.59, 14.59, 10.10, 4.49, 1.0],
                [5.13, 13.14, 21.85, 25.69, 21.85, 13.14, 5.13, 1.0]
            ];
            k = COEFFS_TABLE[N - 1];
        else
            k = SUPPENV.coeffs;
        end
        A1 = zeros(N, 1);
        A2 = zeros(N, N);
        for i = 1:1:N
            A1[i] = - k[i];
            if i + 1 <= N
                A2[i, i + 1] = 1;
            end
        end
        B = - A1[:, 1];
        H = zeros(N, 1);
        H[end] = 1;

        gain = get_timevarying_gain(SUPPENV.gain_type);
        gain_vec(W, t) = [gain(W, t).^i for i in [1:N]];

        hgo(u, y, W, p, t) = vec(A1 .* gain_vec(W, t)[1] * u[1] + A2 * u +
            B .* gain_vec(W, t)[1] * y + H * SUPPENV.phi(u, p, t));

        return hgo
    end
end

function get_hgo_matrices(hgo_type::Int, n::Int, coeffs::Vector,
        epsilon::Vector{Float64})
    N = n + 1;
    if hgo_type == UnivEst.CLASSICALHGO
        if length(coeffs) == 0
            COEFFS_TABLE = [
                [1.414, 1.0],
                [2.0, 2.0, 1.0],
                [2.613, 3.414, 2.613, 1.000],
                [3.236, 5.240, 5.236, 3.326, 1.000],
                [3.86, 7.46, 9.14, 7.46, 3.86, 1.0],
                [4.49, 10.10, 14.59, 14.59, 10.10, 4.49, 1.0],
                [5.13, 13.14, 21.85, 25.69, 21.85, 13.14, 5.13, 1.0]
            ];
            k = COEFFS_TABLE[N - 1];
        else
            k = coeffs;
        end
        A = zeros(N, N);
        for i = 1:1:N
            A[i, 1] = - k[i] * (epsilon[1]^-1)^i;
            if i + 1 <= N
                A[i, i + 1] = 1;
            end
        end
        B = - A[:, 1];

        return A, B
    end
    if hgo_type == UnivEst.M_CASCADE
        tmp_ind = 2 * (N - 1);
        A = zeros(tmp_ind + 1, tmp_ind + 1);
        for i = 1:1:Int(round((tmp_ind / 2)))
            A[(i - 1) * 2 + 1, (i - 1) * 2 + 1] = - 2 * epsilon[i]^-1;
            A[(i - 1) * 2 + 1, (i - 1) * 2 + 2] = 1;
            A[i * 2, (i - 1) * 2 + 1] = - epsilon[i]^-2;
            if i > 1
                A[(i - 1) * 2 + 1, (i - 1) * 2] = 2 * epsilon[i]^-1;
                A[i * 2, (i - 1) * 2] = epsilon[i]^-2;
            end
        end
        B = - A[:, 1];
        A[end, end - 1] = epsilon[N]^-1;
        A[end, end] = - epsilon[N]^-1;

        return A, B;
    end
    if hgo_type == UnivEst.CASCADE
        if length(coeffs) == 0
            COEFFS_TABLE = [
                [1.414, 1.0],
                [5.0, 10.0, 5.0, 2.4],
                [7.0, 28.0, 7.0, 9.0, 7.0, 2.85714],
            ];
            k = COEFFS_TABLE[N - 1];
        else
            k = coeffs;
        end

        tmp_ind = 2 * (N - 1);
        A = zeros(tmp_ind, tmp_ind);
        for i = 1:1:Int(round((tmp_ind / 2)))
            A[(i - 1) * 2 + 1, (i - 1) * 2 + 1] =
                - k[(i - 1) * 2 + 1] * epsilon[1]^-1;
            A[(i - 1) * 2 + 1, (i - 1) * 2 + 2] = 1;
            A[i * 2, (i - 1) * 2 + 1] =
                - k[(i - 1) * 2 + 2] * epsilon[1]^-2;
            if i < Int(round((tmp_ind / 2)))
                A[i * 2, (i - 1) * 2 + 3] = 1;
            end
            if i > 1
                A[(i - 1) * 2 + 1, (i - 1) * 2] =
                    k[(i - 1) * 2 + 1] * epsilon[1]^-1;
                A[i * 2, (i - 1) * 2] =
                    k[(i - 1) * 2 + 2] * epsilon[1]^-2;
            end
        end
        B = - A[:, 1];

        return A, B;
    end
end
