###############################################################################
#############################EXPORTED FUNCTIONS#################################
################################################################################

"""
    bode_hgo(hgo_type::Int, n::Int, epsilon::Vector{Float64};
        coeffs = [])

Returns mag, phase and frequencies of bode diagram related to the standard
(hgo_type = UnivEst.CLASSICALHGO), the m-cascade of second-order high gain
observers (hgo_type = UnivEst.M_CASCADE) or the cascade of high-gain observers
(hgo_type = UnivEst.CASCADE), between input noise and state estimation vector.
'n' is the number of time-derivatives that the observer estimates.
'epsilon' is a vector such that the inverse of its entries are the gains of the
observer.
'coeffs' are the k coefficients.
"""
function bode_hgo(hgo_type::Int, n::Int, epsilon::Vector{Float64};
        coeffs = [])
    A, B = get_hgo_matrices(hgo_type, n, coeffs, epsilon);

    N = size(A, 1);
    C = zeros(N, N);
    for i = 1:1:N
        C[i, i] = 1;
    end
    D = 0;

    ss_obs = ss(A, B, C, D);
    return bode(ss_obs)
end

"""
    estimate_t_derivatives(sys::system_obs, hgo_type::Int, n::Int,
        epsilon::Vector{Float64};
        coeffs::Vector = [], gamma::Float64 = 0.0)
    estimate_t_derivatives(sys::system_obs, hgo_type::Int, n::Int,
        epsilon::Vector{Float64}, d::Function;
        coeffs::Vector = [], gamma::Float64 = 0.0)
    estimate_t_derivatives(sys::system_obs, hgo_type::Int, n::Int,
        epsilon::Vector{Float64}, d::Function, d_sys::Function,
        d_sys_u0::Vector{Float64};
        coeffs::Vector = [], gamma::Float64 = 0.0)

    estimate_t_derivatives(sys::system, hgo_type::Int, n::Int,
        epsilon::Vector{Float64};
        coeffs::Vector = [], gamma::Float64 = 0.0)
    estimate_t_derivatives(sys::system, hgo_type::Int, n::Int,
        epsilon::Vector{Float64}, d::Function;
        coeffs::Vector = [], gamma::Float64 = 0.0)
    estimate_t_derivatives(sys::system, hgo_type::Int, n::Int,
        epsilon::Vector{Float64}, d::Function, d_sys::Function,
        d_sys_u0::Vector{Float64};
        coeffs::Vector = [], gamma::Float64 = 0.0)

Estimates time derivatives with output corrupted by additive noise (given by
the function d).
'd' has to be a function with one argument.
'd', 'd_sys' and 'd_sys_u0' can be used to generate a disturbance that is given
by a dynamical system d_sys from the initial condition d_sys_u0 and d is the
output function (d and d_sys must be functions with one and two arguments
respectively, e.g., d(u), d_sys(u, t)).
Returns the system sys solution, the estimated derivatives and the state of
d_sys.
"""
function estimate_t_derivatives(sys::system_obs, hgo_type::Int, n::Int,
        epsilon::Vector{Float64};
        coeffs::Vector = [], gamma::Float64 = 0.0)
    d(t) = 0.0;
    return estimate_t_derivatives(sys, hgo_type, n, epsilon, d,
        coeffs = coeffs, gamma = gamma);
end
function estimate_t_derivatives(sys::system_obs, hgo_type::Int, n::Int,
        epsilon::Vector{Float64}, d::Function;
        coeffs::Vector = [], gamma::Float64 = 0.0)
    N = n + 1;

    sys_m = length(sys.u0);
    m = 0;
    if hgo_type == UnivEst.CLASSICALHGO
        m = n + 1;
    end
    if hgo_type == UnivEst.M_CASCADE
        m = 2 * n + 1;
    end
    if hgo_type == UnivEst.CASCADE
        m = 2 * n;
    end

    hgo = get_hgo_dynamics(hgo_type, n, coeffs, epsilon);
    f = get_system_dynamics(sys.phi, sys.u0, sys.p);
    if gamma == 0.0
        u0 = [sys.u0; vec(zeros(m, 1))];
        dynamics1(u, p, t) = [
            f(u[1:sys_m], p, t)
            hgo(u[(sys_m + 1):end], u[1] + d(t))
        ];
        sol = get_sol(dynamics1, u0, sys.p, sys.t0, sys.tf, sys.ts,
            sys.tolerances);
        return sol[1:sys_m, :], sol[(sys_m + 1):end, :];
    else
        u0 = [sys.u0; vec(zeros(m, 1)); 0.0];
        dynamics2(u, p, t) = [
            f(u[1:sys_m], p, t)
            hgo(u[(sys_m + 1):(end - 1)], u[end])
            - gamma * (u[end] - u[1] - d(t))
        ];
        sol = get_sol(dynamics2, u0, sys.p, sys.t0, sys.tf, sys.ts,
            sys.tolerances);
        return sol[1:sys_m, :], sol[(sys_m + 1):(end - 1), :];
    end
end
function estimate_t_derivatives(sys::system_obs, hgo_type::Int, n::Int,
        epsilon::Vector{Float64}, d::Function, d_sys::Function,
        d_sys_u0::Vector{Float64};
        coeffs::Vector = [], gamma::Float64 = 0.0)
    N = n + 1;
    sys_m = length(sys.u0);
    m = 0;
    if hgo_type == UnivEst.CLASSICALHGO
        m = n + 1;
    end
    if hgo_type == UnivEst.M_CASCADE
        m = 2 * n + 1;
    end
    if hgo_type == UnivEst.CASCADE
        m = 2 * n;
    end

    hgo = get_hgo_dynamics(hgo_type, n, coeffs, epsilon);
    f = get_system_dynamics(sys.phi, sys.u0, sys.p);
    if gamma == 0.0
        u0 = [sys.u0; vec(zeros(m, 1)); d_sys_u0];
        dynamics1(u, p, t) = [
            f(u[1:sys_m], p, t)
            hgo(u[(sys_m + 1):(sys_m + m)], u[1] + d(u[(sys_m + m + 1):end]))
            d_sys(u[(sys_m + m + 1):end], t)
        ];
        sol = get_sol(dynamics1, u0, sys.p, sys.t0, sys.tf, sys.ts,
            sys.tolerances);
        return sol[1:sys_m, :], sol[(sys_m + 1):(sys_m + m), :],
            sol[(sys_m + m + 1):end, :]
    else
        u0 = [sys.u0; vec(zeros(m, 1)); d_sys_u0; 0.0];
        dynamics2(u, p, t) = [
            f(u[1:sys_m], p, t)
            hgo(u[(sys_m + 1):(sys_m + m)], u[end])
            d_sys(u[(sys_m + m + 1):(end - 1)], t)
            - gamma * (u[end] - u[1] - d(u[(sys_m + m + 1):(end - 1)]))
        ];
        sol = get_sol(dynamics2, u0, sys.p, sys.t0, sys.tf, sys.ts,
            sys.tolerances);
        return sol[1:sys_m, :], sol[(sys_m + 1):(sys_m + m), :],
            sol[(sys_m + m + 1:(end - 1)), :]
    end
end
function estimate_t_derivatives(sys::system, hgo_type::Int, n::Int,
        epsilon::Vector{Float64};
        coeffs::Vector = [], gamma::Float64 = 0.0)
    d(t) = 0.0;
    return estimate_t_derivatives(sys, hgo_type, n, epsilon, d,
        coeffs = coeffs, gamma = gamma);
end
function estimate_t_derivatives(sys::system, hgo_type::Int, n::Int,
        epsilon::Vector{Float64}, d::Function;
        coeffs::Vector = [], gamma::Float64 = 0.0)
    N = n + 1;

    sys_m = length(sys.u0);
    m = 0;
    if hgo_type == UnivEst.CLASSICALHGO
        m = n + 1;
    end
    if hgo_type == UnivEst.M_CASCADE
        m = 2 * n + 1;
    end
    if hgo_type == UnivEst.CASCADE
        m = 2 * n;
    end

    hgo = get_hgo_dynamics(hgo_type, n, coeffs, epsilon);
    if gamma == 0.0
        dynamics1(u, p, t) = [
            sys.f(u[1:sys_m], p, t)
            hgo(u[(sys_m + 1):end], sys.h(u[1:sys_m], p, t) + d(t))
        ];
        u0 = [sys.u0; vec(zeros(m, 1))];
        sol = get_sol(dynamics1, u0, sys.p, sys.t0, sys.tf, sys.ts,
            sys.tolerances);
        return sol[1:sys_m, :], sol[(sys_m + 1):end, :];
    else
        dynamics2(u, p, t) = [
            sys.f(u[1:sys_m], p, t)
            hgo(u[(sys_m + 1):(end - 1)], u[end])
            - gamma * (u[end] - sys.h(u[1:sys_m], p, t) - d(t))
        ];
        u0 = [sys.u0; vec(zeros(m, 1)); 0.0];
        sol = get_sol(dynamics2, u0, sys.p, sys.t0, sys.tf, sys.ts,
            sys.tolerances);
        return sol[1:sys_m, :], sol[(sys_m + 1):(end - 1), :];
    end
end
function estimate_t_derivatives(sys::system, hgo_type::Int, n::Int,
        epsilon::Vector{Float64}, d::Function, d_sys::Function,
        d_sys_u0::Vector{Float64};
        coeffs::Vector = [], gamma::Float64 = 0.0)
    N = n + 1;
    sys_m = length(sys.u0);
    m = 0;
    if hgo_type == UnivEst.CLASSICALHGO
        m = n + 1;
    end
    if hgo_type == UnivEst.M_CASCADE
        m = 2 * n + 1;
    end
    if hgo_type == UnivEst.CASCADE
        m = 2 * n;
    end

    hgo = get_hgo_dynamics(hgo_type, n, coeffs, epsilon);
    if gamma == 0.0
        dynamics1(u, p, t) = [
            sys.f(u[1:sys_m], p, t)
            hgo(u[(sys_m + 1):(sys_m + m)], sys.h(u[1:sys_m], p, t) +
                d(u[(sys_m + m + 1):end]))
            d_sys(u[(sys_m + m + 1):end], t)
        ];
        u0 = [sys.u0; vec(zeros(m, 1)); d_sys_u0];
        sol = get_sol(dynamics1, u0, sys.p, sys.t0, sys.tf, sys.ts,
            sys.tolerances);
        return sol[1:sys_m, :], sol[(sys_m + 1):(sys_m + m), :],
            sol[(sys_m + m + 1:end), :]
    else
        dynamics2(u, p, t) = [
            sys.f(u[1:sys_m], p, t)
            hgo(u[(sys_m + 1):(sys_m + m)], u[end])
            d_sys(u[(sys_m + m + 1):(end - 1)], t)
            - gamma * (u[end] - sys.h(u[1:sys_m], p, t) -
                d(u[(sys_m + m + 1):(end - 1)]));
        ];
        u0 = [sys.u0; vec(zeros(m, 1)); d_sys_u0; 0.0];
        sol = get_sol(dynamics2, u0, sys.p, sys.t0, sys.tf, sys.ts,
            sys.tolerances);
        return sol[1:sys_m, :], sol[(sys_m + 1):(sys_m + m), :],
            sol[(sys_m + m + 1:(end - 1)), :]
    end
end

"""
    get_hgo_matrices(hgo_type::Int, n::Int, coeffs::Vector,
        epsilon::Vector{Float64})

Returns the matrices A and B related to the standard high-gain observer
(hgo_type = UnivEst.CLASSICALHGO), the m-cascade of second-order high gain
observers (hgo_type = UnivEst.M_CASCADE) or the cascade of high-gain observers
(hgo_type = UnivEst.CASCADE).
'n' is the number of time-derivatives that the observer estimates.
'coeffs' are the k coefficients.
'epsilon' is the parameter such that its inverse is the gain of the observer.
"""
function get_hgo_matrices(hgo_type::Int, n::Int, coeffs::Vector,
        epsilon::Vector{Float64})
    N = n + 1;
    if hgo_type == UnivEst.CLASSICALHGO
        k = get_coeffs_table(hgo_type, coeffs, N);

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
        k = get_coeffs_table(hgo_type, coeffs, N);

        tmp_ind = 2 * (N - 1);
        A = zeros(tmp_ind + 1, tmp_ind + 1);
        for i = 1:1:Int(round((tmp_ind / 2)))
            A[(i - 1) * 2 + 1, (i - 1) * 2 + 1] =
                - k[i][1] * epsilon[i]^-1;
            A[(i - 1) * 2 + 1, (i - 1) * 2 + 2] = 1;
            A[i * 2, (i - 1) * 2 + 1] = - k[i][2] * epsilon[i]^-2;
            if i > 1
                A[(i - 1) * 2 + 1, (i - 1) * 2] =
                    k[i][1] * epsilon[i]^-1;
                A[i * 2, (i - 1) * 2] = k[i][2] * epsilon[i]^-2;
            end
        end
        B = - A[:, 1];
        A[end, end - 1] = epsilon[N]^-1;
        A[end, end] = - epsilon[N]^-1;

        return A, B;
    end
    if hgo_type == UnivEst.CASCADE
        k = get_coeffs_table(hgo_type, coeffs, N);

        tmp_ind = 2 * (N - 1);
        A = zeros(tmp_ind, tmp_ind);
        for i = 1:1:Int(round((tmp_ind / 2)))
            A[(i - 1) * 2 + 1, (i - 1) * 2 + 1] =
                - k[(i - 1) * 2 + 1] * epsilon[1]^-1;
            A[(i - 1) * 2 + 1, (i - 1) * 2 + 2] = 1;
            A[i * 2, (i - 1) * 2 + 1] =
                - k[(i - 1) * 2 + 2] * epsilon[1]^-2;
            if i < Int(round((tmp_ind / 2)))
                A[i * 2, (i - 1) * 2 + 4] = 1;
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

"""
    test_hgo(sys::system_obs, hgo_type::Int, eps::Vector{Float64};
            coeffs::Vector = [])
    test_hgo(sys::system_obs, hgo_type::Int, eps::Vector{Float64},
            d::Function; coeffs::Vector = [])
    test_hgo(sys::system_obs, hgo_type::Int, eps::Vector{Float64},
            d::Function, d_sys::Function, d_sys_u0::Vector{Float64};
            coeffs::Vector = [])
"""
function test_hgo(sys::system_obs, hgo_type::Int, epsilon::Float64,
		phi::Function, p::Vector{Float64}; coeffs::Vector = [], S::Vector = [])
    d(t) = 0.0;
    return test_hgo(sys, hgo_type, epsilon, phi, p, d, coeffs = coeffs,
        S = S);
end
function test_hgo(sys::system_obs, hgo_type::Int, epsilon::Float64,
        phi::Function, p::Vector{Float64}, d::Function; coeffs::Vector = [],
		S::Vector = [])
    if hgo_type == UnivEst.CLASSICALHGO
        sys_m = length(sys.u0);
        hgo = get_hgo_dynamics(hgo_type, sys_m - 1, coeffs, epsilon, phi, p);
        f = get_system_dynamics(sys.phi, sys.u0, sys.p);
        dynamics1(u, p, t) = [
            f(u[1:sys_m], p, t)
            hgo(u[(sys_m + 1):end], u[1] + d(t), t)
        ];

		u0 = [sys.u0; vec(zeros(sys_m, 1))];
		sol = get_sol(dynamics1, u0, sys.p, sys.t0, sys.tf, sys.ts,
			sys.tolerances);
        return sol[1:sys_m, :], sol[(sys_m + 1):end, :];
    end
    if hgo_type == UnivEst.MIN_CASCADE
        sys_m = length(sys.u0);
        hgo = get_hgo_dynamics(hgo_type, sys_m - 1, coeffs, epsilon, S, phi, p);
        f = get_system_dynamics(sys.phi, sys.u0, sys.p);
        dynamics2(u, p, t) = [
            f(u[1:sys_m], p, t)
            hgo(u[(sys_m + 1):end], u[1] + d(t), t)
        ];

		u0 = [sys.u0; vec(zeros(sys_m, 1))];
		sol = get_sol(dynamics2, u0, sys.p, sys.t0, sys.tf, sys.ts,
			sys.tolerances);
        u = sol[1:sys_m, :];
        z = sol[(sys_m + 1):end, :];
        hx = get_mincascade_estimates(sys_m, epsilon^-1, coeffs, S, z);
        return u, z, hx;
    end
end
function test_hgo(sys::system_obs, hgo_type::Int, epsilon::Float64,
        phi::Function, p::Vector{Float64}, d::Function, d_sys::Function,
		d_sys_u0::Vector{Float64}; coeffs::Vector = [],
		S::Vector = [])
    print("Not implemented.")
end

"""
    gain_plot(W::Vector{Float64}, t0::Float64, ts::Float64,
        tf::Float64; gain_type::Int = UnivEst.TIMEVARYING_GAIN,
        get_vals::Int = 0)
"""
function gain_plot(W::Vector{Float64}, t0::Float64, ts::Float64,
        tf::Float64; gain_type::Int = UnivEst.TIMEVARYING_GAIN,
        get_vals::Int = 0)
    g_func = get_gain_func(gain_type);
    g_vals = zeros(Int(round((tf - t0) / ts)), 1);
    j = 1;
    for i = t0:ts:(tf - ts)
        g_vals[j] = g_func(W, i);
        j = j + 1;
    end
    pl = plot(t0:ts:(tf - ts), g_vals);
    if get_vals == 0
        return pl;
    else
        return g_vals;
    end
end

################################################################################
#############################NOT EXPORTED#######################################
################################################################################
"""
	get_hgo_dynamics(hgo_type::Int, n::Int, coeffs::Vector,
        epsilon::Vector{Float64})
    get_hgo_dynamics(hgo_type::Int, n::Int, coeffs::Vector,
        epsilon::Float64, phi::Function, p::Vector{Float64})
	get_hgo_dynamics(hgo_type::Int, n::Int, coeffs::Vector,
        epsilon::Float64, S::Vector, phi::Function,
		p::Vector{Float64})
    get_hgo_dynamics(hgo_type::Int, n::Int, coeffs::Vector, g::Function,
        phi::Function)

Returns a function that implements high-gain observers for integration purposes.
"""
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
        epsilon::Float64, phi::Function, p::Vector{Float64})
    N = n + 1;
    A, B = get_hgo_matrices(hgo_type, n, coeffs, [epsilon]);
    H = zeros(N, 1);
    H[end] = 1;
    hgo(u, y, t) = vec(A * u + B * y + H * phi(u, p, t));
    return hgo;
end
function get_hgo_dynamics(hgo_type::Int, n::Int, coeffs::Vector,
        epsilon::Float64, S::Vector, phi::Function,
		p::Vector{Float64})
    return get_min_cascade_dynamics(n, coeffs, epsilon, S, phi, p);
end
function get_hgo_dynamics(hgo_type::Int, N::Int, coeffs::Vector, g::Function,
        phi::Function)
    k = get_coeffs_table(hgo_type, coeffs, N);

    A1 = zeros(N, 1);
    A2 = zeros(N, N);
    for i = 1:1:N
        A1[i, 1] = - k[i];
        if i + 1 <= N
            A2[i, i + 1] = 1;
        end
    end
    B = - A1[:, 1];
    H = zeros(N, 1);
    H[end] = 1;

    gain_vec(p, t) = [g(p, t).^i for i in [1:N]];
    hgo(u, y, p, t) = vec(A1 .* gain_vec(p, t)[1] * u[1] + A2 * u +
        B .* gain_vec(p, t)[1] * y + H * phi(u, t));

    return hgo
end

"""
    get_coeffs_table(hgo_type::Int, coeffs::Vector, N::Int)

Returns feasible coefficients for the 'k' parameters appearing into high-gain
observers (only up to order 5).
"""
function get_coeffs_table(hgo_type::Int, coeffs::Vector, N::Int)
    if length(coeffs) == 0
        if hgo_type == UnivEst.CLASSICALHGO || hgo_type == UnivEst.MIN_CASCADE
            COEFFS_TABLE = [
                [0.3, 0.02],
                [0.6, 0.11, 0.006],
                [1.0, 0.35, 0.05, 0.0024],
                [1.5, 0.85, 0.225, 0.0274, 0.0012]
            ];
            if N - 1 > length(COEFFS_TABLE)
                print("You need to provide a coefficient vector\n")
                return;
            end
            k = COEFFS_TABLE[N - 1];
        end
        if hgo_type == UnivEst.M_CASCADE
            COEFFS_TABLE = [
                [0.3, 0.02],
                [0.5, 0.06],
                [0.7, 0.12],
                [0.9, 0.2]
            ];
            if N - 1 > length(COEFFS_TABLE)
                print("You need to provide a coefficient vector\n")
                return;
            end
            k = COEFFS_TABLE[1:(N - 1)];
        end
        if hgo_type == UnivEst.CASCADE
            COEFFS_TABLE = [
                [0.3, 0.02],
                [2/5, 7/100, 2/5, 3/175],
                [1/2, 4/25, 1/2, 21/400, 1/2, 3/175],
                [0.6, 0.3, 0.6, 0.111, 0.6, 0.0485, 0.6, 0.0178]
            ];
            if N - 1 > length(COEFFS_TABLE)
                print("You need to provide a coefficient vector\n")
                return;
            end
            k = COEFFS_TABLE[N - 1];
        end
    else
        k = coeffs;
    end

    return k;
end

"""
	get_min_cascade_dynamics(n::Int, coeffs::Vector,
		epsilon::Vector{Float64}, S::Vector{Float64}, phi::Function,
		p::Vector{Float64})
	get_min_cascade_dynamics(n::Int, coeffs::Vector,
		g::Function, S::Vector{Float64}, phi::Function,
		p::Vector{Float64})

Return minimum-order cascade of hgos dynamics function.
"""
function get_min_cascade_dynamics(n::Int, coeffs::Vector,
		epsilon::Float64, S::Vector, phi::Function,
		p::Vector{Float64})
    N = n + 1;
    k = get_coeffs_table(UnivEst.MIN_CASCADE, coeffs, N)

    g = epsilon^-1;

    if N == 2
        hx2 = get_hx(N, g, k, S);
		dz2(z, y, t) = [
			- g * k[1] * (z[1] - y) + hx2(z)[2]
			- g * k[2] / k[1] * (z[2] + hx2(z)[1]) +
                (g * k[2] / k[1])^-1 * phi(hx2(z), p, t)
		];
		return dz2;
    end
    if N == 3
        hx3 = get_hx(N, g, k, S);
		dz3(z, y, t) = [
			- g * k[1] * (z[1] - y) + hx3(z)[2]
			- g * k[2] / k[1] * (z[2] + hx3(z)[1]) +
                (g * k[2] / k[1])^-1 * hx3(z)[3]
			- g * k[3] / k[2] * (z[3] + hx3(z)[2]) +
                (g * k[3] / k[2])^-1 * phi(hx3(z), p, t)
		];
		return dz3;
    end
    if N == 4
        hx4 = get_hx(N, g, k, S);
		dz4(z, y, t) = [
			- g * k[1] * (z[1] - y) + hx4(z)[2]
			- g * k[2] / k[1] * (z[2] + hx4(z)[1]) +
                (g * k[2] / k[1])^-1 * hx4(z)[3]
			- g * k[3] / k[2] * (z[3] + hx4(z)[2]) +
                (g * k[3] / k[2])^-1 * hx4(z)[4]
			- g * k[4] / k[3] * (z[4] + hx4(z)[3]) +
                (g * k[4] / k[3])^-1 * phi(hx4(z), p, t)
		];
		return dz4;
    end
end
function get_min_cascade_dynamics(N::Int, coeffs::Vector,
		g::Function, S::Vector, phi::Function)
    k = get_coeffs_table(UnivEst.MIN_CASCADE, coeffs, N)

    if N == 2
        hx2 = get_hx(N, g, k, S);
		dz2(z, y, p, t) = [
			- g(p, t) * k[1] * (z[1] - y) + hx2(z, p, t)[2]
			- g(p, t) * k[2] / k[1] * (z[2] + hx2(z, p, t)[1]) +
                (g(p, t) * k[2] / k[1])^-1 * phi(hx2(z, p, t), t)
		];
		return dz2;
    end
    if N == 3
        hx3 = get_hx(N, g, k, S);
		dz3(z, y, p, t) = [
			- g(p, t) * k[1] * (z[1] - y) + hx3(z, p, t)[2]
			- g(p, t) * k[2] / k[1] * (z[2] + hx3(z, p, t)[1]) +
                (g(p, t) * k[2] / k[1])^-1 * hx3(z, p, t)[3]
			- g(p, t) * k[3] / k[2] * (z[3] + hx3(z, p, t)[2]) +
                (g(p, t) * k[3] / k[2])^-1 * phi(hx3(z, p, t), t)
		];
		return dz3;
    end
    if N == 4
        hx4 = get_hx(N, g, k, S);
		dz4(z, y, p, t) = [
			- g(p, t) * k[1] * (z[1] - y) + hx4(z, p, t)[2]
			- g(p, t) * k[2] / k[1] * (z[2] + hx4(z, p, t)[1]) +
                (g(p, t) * k[2] / k[1])^-1 * hx4(z, p, t)[3]
			- g(p, t) * k[3] / k[2] * (z[3] + hx4(z, p, t)[2]) +
                (g(p, t) * k[3] / k[2])^-1 * hx4(z, p, t)[4]
			- g(p, t) * k[4] / k[3] * (z[4] + hx4(z, p, t)[3]) +
                (g(p, t) * k[4] / k[3])^-1 * phi(hx4(z, p, t), t)
		];
		return dz4;
    end
end

"""
    get_hx(n::Int, g::Float64, coeffs::Vector{Float64}, S::Vector{Float64})

Returns hat x vector in minimum-order cascade of high-gain observers.
"""
function get_hx(n::Int, g::Float64, coeffs::Vector{Float64}, S::Vector)
	sat(x, v) = sign(x) * v * min(abs(x) / v, 1);
	if length(S) == 0
		S = ones(n) * 1e03;
	end

    if n == 2
        hx12(z) = z[1];
        hx22(z) = sat(g * coeffs[2] / coeffs[1] * (z[2] + hx12(z)), S[1]);
        hx2(z) = [hx12(z); hx22(z)];
		return hx2;
    end
    if n == 3
        hx13(z) = z[1];
        hx23(z) = sat(g * coeffs[2] / coeffs[1] * (z[2] + hx13(z)), S[1]);
        hx33(z) = sat(g * coeffs[3] / coeffs[2] * (z[3] + hx23(z)), S[2]);
        hx3(z) = [hx13(z); hx23(z); hx33(z)];
		return hx3;
    end
    if n == 4
        hx14(z) = z[1];
        hx24(z) = sat(g * coeffs[2] / coeffs[1] * (z[2] + hx14(z)), S[1]);
        hx34(z) = sat(g * coeffs[3] / coeffs[2] * (z[3] + hx24(z)), S[2]);
        hx44(z) = sat(g * coeffs[4] / coeffs[3] * (z[4] + hx34(z)), S[3]);
        hx4(z) = [hx14(z); hx24(z); hx34(z); hx44(z)];
		return hx4;
    end
end
function get_hx(n::Int, g::Function, coeffs::Vector{Float64}, S::Vector)
	sat(x, v) = sign(x) * v * min(abs(x) / v, 1);
	if length(S) == 0
		S = ones(n) * 1e03;
	end

    if n == 2
        hx12(z) = z[1];
        hx22(z, p, t) =
            sat(g(p, t) * coeffs[2] / coeffs[1] * (z[2] + hx12(z)), S[1]);
        hx2(z, p, t) = [hx12(z); hx22(z, p, t)];
		return hx2;
    end
    if n == 3
        hx13(z) = z[1];
        hx23(z, p, t) =
            sat(g(p, t) * coeffs[2] / coeffs[1] * (z[2] + hx13(z)), S[1]);
        hx33(z, p, t) =
            sat(g(p, t) * coeffs[3] / coeffs[2] * (z[3] + hx23(z, p, t)), S[2]);
        hx3(z, p, t) = [hx13(z); hx23(z, p, t); hx33(z, p, t)];
		return hx3;
    end
    if n == 4
        hx14(z) = z[1];
        hx24(z, p, t) =
            sat(g(p, t) * coeffs[2] / coeffs[1] * (z[2] + hx14(z)), S[1]);
        hx34(z, p, t) =
            sat(g(p, t) * coeffs[3] / coeffs[2] * (z[3] + hx24(z, p, t)), S[2]);
        hx44(z, p, t) =
            sat(g(p, t) * coeffs[4] / coeffs[3] * (z[4] + hx34(z, p, t)), S[3]);
        hx4(z, p, t) = [hx14(z); hx24(z, p, t); hx34(z, p, t); hx44(z, p, t)];
		return hx4;
    end
end

"""
    get_mincascade_estimates(n::Int, g::Float64, coeffs::Vector{Float64},
        S::Vector, z::Matrix{Float64})

Extracts hat x vector from the state vector z of minimum-order cascades of
high-gain observers.
"""
function get_mincascade_estimates(n::Int, g::Float64, coeffs::Vector, S::Vector,
        z::Matrix{Float64})
    k = get_coeffs_table(UnivEst.MIN_CASCADE, coeffs, n);
    hx_f = get_hx(n, g, k, S);

    hx = zeros(size(z, 1), size(z, 2));
    for i = 1:1:size(z, 2)
        hx[:, i] = hx_f(z[:, i]);
    end
    return hx;
end
function get_mincascade_estimates(n::Int, g::Float64, coeffs::Vector, S::Vector,
        z::Vector{Float64})
    k = get_coeffs_table(UnivEst.MIN_CASCADE, coeffs, n);
    hx_f = get_hx(n, g, k, S);
    hx = hx_f(z);

    return hx;
end

"""
    get_gain_func(gain_type::Int)
"""
function get_gain_func(gain_type::Int)
    sigma(x) = exp(x) / (1 + exp(x));
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

"""
    test_timevarying_hgo(sys::system_obs, p::Vector{Float64}, d::Function;
        coeffs::Vector = [], gain_type::Int = UnivEst.TIMEVARYING_GAIN,
        hgo_type::Int = UnivEst.CLASSICALHGO, S::Vector = [],
        tfinal::Float64 = 0.0)
    test_timevarying_hgo(sys::system, sysobs::system_obs,
        gain_params::Vector{Float64}, grad_params::Vector{Float64}, d::Function;
        coeffs::Vector = [], gain_type::Int = UnivEst.TIMEVARYING_GAIN,
        hgo_type::Int = UnivEst.CLASSICALHGO, S::Vector = [],
        tfinal::Float64 = 0.0)
"""
function test_timevarying_hgo(sys::system_obs, p::Vector{Float64}, d::Function;
        coeffs::Vector = [], gain_type::Int = UnivEst.TIMEVARYING_GAIN,
        hgo_type::Int = UnivEst.CLASSICALHGO, S::Vector = [],
        tfinal::Float64 = 0.0)

    if tfinal == 0.0
        tfin = sys.tf;
    else
        tfin = tfinal;
    end

    n = length(sys.u0);
    if hgo_type == UnivEst.MIN_CASCADE && length(S) < n
        println("A valid vector of saturations S must be supplied.");
        return;
    end

    f = get_system_dynamics(sys.phi, sys.u0, sys.p);
    gain_func = get_gain_func(gain_type);
    phi(u, t) = sys.phi(u, sys.p, t);
    u0 = vcat(sys.u0, zeros(n, 1));
    if hgo_type == UnivEst.CLASSICALHGO
        hgo = get_hgo_dynamics(hgo_type, n, coeffs, gain_func, phi);
        dynamics1(u, p, t) = [
            f(u[1:n], sys.p, t)
            hgo(u[(n + 1):end], u[1] + d(t), p, t)
        ];
        sol = get_sol(dynamics1, vec(u0), p, sys.t0, tfin, sys.ts,
            sys.tolerances);
        return sol[1:n, :]', sol[(n + 1):(2 * n), :]';
    else
        hgo = get_min_cascade_dynamics(n, coeffs, gain_func, S, phi);
        dynamics2(u, p, t) = [
            f(u[1:n], sys.p, t)
            hgo(u[(n + 1):end], u[1] + d(t), p, t)
        ];
        sol = get_sol(dynamics2, vec(u0), p, sys.t0, tfin, sys.ts,
            sys.tolerances);

        rnum = size(sol, 2);
        hx = zeros(rnum, n);
        for i = 1:1:rnum
            tmp = sol[(n + 1):(2 * n), i];
            hx[i, :] = get_mincascade_estimates(n,
                gain_func(p, (i - 1) * sys.ts), coeffs, S,
                reshape(tmp, size(tmp, 1), size(tmp, 2)));
        end
        return sol[1:n, :]', hx;
    end
end
function test_timevarying_hgo(sys::system, sysobs::system_obs,
        gain_params::Vector{Float64}, grad_params::Vector{Float64}, d::Function;
        coeffs::Vector = [], gain_type::Int = UnivEst.TIMEVARYING_GAIN,
        hgo_type::Int = UnivEst.CLASSICALHGO, S::Vector = [],
        tfinal::Float64 = 0.0)
    if hgo_type != UnivEst.CLASSICALHGO
        println("hgo_type = UnivEst.CLASSICALHGO: not implemented yet");
    end
    if gain_type != UnivEst.TIMEVARYING_GAIN
        println("gain_type = UnivEst.TIMEVARYING_GAIN: not implemented yet");
    end

    if tfinal == 0.0
        tfin = sys.tf;
    else
        tfin = tfinal;
    end

    global SUPPENV
    n = length(sys.u0);
    m = length(sysobs.u0);
    gain_func = get_gain_func(gain_type);
    phi(u, t) = sysobs.phi(u, sysobs.p, t);
    hgo = get_hgo_dynamics(hgo_type, m, coeffs, gain_func, phi);
    grad_alg(x, xi, t) = (sign(t - grad_params[2]) + 1) / 2 *
        grad_params[1] * K(x, t) * (xi - obs_map(x, t));
    dynamics(u, p, t) = [
        sys.f(u[1:n], sys.p, t)
        hgo(u[(n + 1):(n + m)], u[1] + d(t), gain_params, t)
        grad_alg(u[(n + m + 1):end], u[(n + 1):(n + m)], t)
    ];
    sol = get_sol(dynamics, vec(u0), p, sys.t0, tfin, sys.ts,
        sys.tolerances);
    return sol[1:n, :]', sol[(n + 1):(n + m), :]', sol[(n + m + 1):end, :]';
end
