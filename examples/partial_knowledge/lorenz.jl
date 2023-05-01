include("lorenz_data.jl")

t0 = 0.0;
tf = 30.0;
ts = 1e-02;

using Plots, LaTeXStrings
plot(t0:ts:(tf - ts), y_samples)
xlabel!("t")
ylabel!("y(t)")
title!("Output samples")
plot!(legend = :false)
plot!(grid = :false)

f(u, p, t) = [
    u[2] - u[1]
    - u[1] * u[3]
    u[1] * u[2] - p[1]
];
y(u, p, t) = [
    u[1]
];
opt = ADAM(1e-02);
n = 3;
l = 1;

env = init_env(f, y, y_samples, t0, tf, ts, opt, n, l);

tfs = [5.0, 10.0];
tfs = [2.0];
estp = randn(n + l) * 1e-03;
#hp = training_routine(env, tfs, estp, 500)
hp = training_routine(env, tfs, estp, 3)

est_sol = get_sys_solution(env, [hp[end]], hp[1:n])
times = env.t0:env.ts:(env.tf - env.ts);
plot(times, y_samples, label = L"y(t)")
plot!(times, est_sol[1, :], label = L"\hat y(t)")
plot!(legendfontsize = 15)
plot!(grid = :false)
xlabel!("t")

env = set_env_parameter(env, "opt", ADAM(1e-03));
tfs = [30.0];
hp = training_routine(env, tfs, hp, 500)

est_sol = get_sys_solution(env, [hp[end]], hp[1:n])
plot(times, y_samples, label = L"y(t)")
plot!(times, est_sol[1, :], label = L"\hat y(t)")
plot!(legendfontsize = 15)
plot!(grid = :false)
xlabel!("t")
