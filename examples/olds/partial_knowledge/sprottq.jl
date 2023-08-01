f(u, p, t) = [
    p[1] - u[2]
    u[3] + p[2]
    u[1] * u[2] - u[3]
];
u0 = [2.0, 0.0, 0.0];
p = [0.9, 0.4];
t0 = 0.0;
tf = 30.0;
ts = 1e-02;
y(u, p, t) = [u[2]];
opt = ADAM(1e-02);

env = init_env(f, y, u0, p, t0, tf, ts, opt);

tfs = [2.5, 5.0, 10.0, 15.0];
estp = randn(length(u0) + length(p)) * 1e-03;
hp = training_routine(env, tfs, estp, 500)

using Plots, LaTeXStrings
est_sol = get_sys_solution(env, hp[(length(u0) + 1):end], hp[1:length(u0)])
times = env.t0:env.ts:(env.tf - env.ts);
plot(times, env.y_samples, label = L"y(t)")
plot!(times, est_sol[2, :], label = L"\hat y(t)")
plot!(legendfontsize = 15)
plot!(grid = :false)
xlabel!("t")
