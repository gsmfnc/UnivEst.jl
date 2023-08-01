phi(u, p, t) = - u[4] + p[1] * u[3] + p[2] * u[2] + p[3] * (u[1]^2 - 1);
u0 = [-0.9, 0.0, 0.0, 0.0];
p = [-5.2, -2.7, 4.5];
t0 = 0.0;
tf = 35.0;
ts = 1e-02;
opt = ADAM(1e-01);

env = init_env(phi, u0, p, t0, tf, ts, opt; reltol = 1e-06, abstol = 1e-06);

epsilon = [2e-02];
data1 = estimate_time_derivatives(env, UnivEst.CLASSICALHGO, epsilon, 3,
    Ts = 1e-04);

epsilon = [2e-02, 2e-02, 2e-02, 2e-02];
data = estimate_time_derivatives(env, UnivEst.CASCADE1, epsilon, 3,
   Ts = 1e-04);
data2 = extract_estimates(data, UnivEst.CASCADE1);

epsilon = [2e-02];
data = estimate_time_derivatives(env, UnivEst.CASCADE2, epsilon, 3,
   Ts = 1e-04);
data3 = extract_estimates(data, UnivEst.CASCADE2);

env = set_env_parameter(env, "data", data1);
env = set_env_parameter(env, "M", 4);
env = set_env_parameter(env, "d_time", 1.0);
env = set_env_parameter(env, "ode_kwargs", (alg = SSPRK53(), dt = ts,));

tfs = [2.5, 5.0, 10.0, 15.0];
estp = vcat(env.data[:, 1], randn(length(env.p)) * 1e-03);
hp = training_routine(env, tfs, estp, 500)
tfs = [30.0];
env = set_env_parameter(env, "opt", ADAM(1e-03));
hp = training_routine(env, tfs, hp, 500)

using Plots, LaTeXStrings
est_sol = get_sys_solution(env, hp[5:end], hp[1:4]);
times = env.t0:env.ts:(env.tf - env.ts);
plot(times, env.data')
plot!(times, est_sol')
plot!(legendfontsize = 15)
plot!(legend = :topleft)
plot!(grid = :false)
plot!(ylim = (-10, 10))
xlabel!("t")

env = set_env_parameter(env, "M", 0);
env = set_env_parameter(env, "d_time", 0.0);
env = set_env_parameter(env, "opt", ADAM(1e-01));

tfs = [2.5, 5.0, 7.5, 10.0, 12.5, 15.0];
estp = vcat(randn(length(env.u0) + length(env.p)) * 1e-03);
hp = training_routine(env, tfs, estp, 500)
env = set_env_parameter(env, "opt", ADAM(1e-02));
tfs = [20.0, 35.0];
hp = training_routine(env, tfs, hp, 500)

using Plots, LaTeXStrings
est_sol = get_sys_solution(env, hp[5:end], hp[1:4]);
times = env.t0:env.ts:(env.tf - env.ts);
plot(times, env.y_samples, label = L"y(t)")
plot!(times, est_sol[1, :], label = L"\hat y(t)")
plot!(legendfontsize = 15)
plot!(legend = :topleft)
plot!(grid = :false)
#plot!(ylim = (-10, 10))
xlabel!("t")

7-element Vector{Float64}:
 -0.9474142171545663
  0.12374798313228949
 -0.03766775207987246
 -0.33998848902331474
 -5.187059958839606
 -2.6902974641298663
  4.4779940771625135
