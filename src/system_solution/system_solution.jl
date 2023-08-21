################################################################################
#########################EXPORTED FUNCTIONS#####################################
################################################################################

"""
    get_sys_solution(sys::system_obs)

Returns the solution of the dynamical system associated with sys.
"""
function get_sys_solution(sys::system_obs)
    f = get_system_dynamics(sys.phi, sys.u0, sys.p);
    sol = get_sol(f, sys.u0, sys.p, sys.t0, sys.tf, sys.ts, sys.tolerances);
    output = sol[1, :];

    return sol, output
end

"""
    get_periodical_signal_samples(sig, bias::Float64,
        amps::Vector{Float64}, phases::Vector{Float64}, puls::Vector{Float64})

Returns samples of the periodical signal
    bias + sum_i amps[i] * sin(puls[i] * t + phases[i])
"""
function get_periodical_signal_samples(sig, bias::Float64,
        amps::Vector{Float64}, phases::Vector{Float64}, puls::Vector{Float64})
    times = sig.t0:sig.ts:sig.tf;
    samples = zeros(length(times) - 1, 1);
    for i = 1:1:length(samples)
        samples[i] = sig.s(bias, amps, puls, phases, (i - 1) * sig.ts);
    end

    return samples;
end

################################################################################
#############################NOT EXPORTED#######################################
################################################################################

"""
    get_system_dynamics(phi::Function, u0::Vector{Float64}, p::Vector{Float64})

This function generates a system in observability canonical form of dimension
n = length(u0) and where d^n/dt^n x = phi(x, p, t).
"""
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

"""
    get_sol(f, u0::Vector{Float64}, p::Vector{Float64}, t0::Float64,
            tf::Float64, ts::Float64, tolerances::Vector{Float64})

Integrates f parameterized with p, from the initial condition u0 and from time
t0 to tf. Then, the solution is sampled with ts.
The solver is fixed (Tsit5()). See ?solve.
"""
function get_sol(f, u0::Vector{Float64}, p::Vector{Float64}, t0::Float64,
        tf::Float64, ts::Float64, tolerances::Vector{Float64})
    prob = ODEProblem(f, u0, (t0, tf), saveat = range(t0, tf,
        length = Int(round((tf - t0) / ts))));
    sol = solve(prob, p = p, abstol = tolerances[1], reltol = tolerances[2],
        sensealg = InterpolatingAdjoint(autojacvec =
            ZygoteVJP(allow_nothing = true)));
    return sol[:, :]
end
