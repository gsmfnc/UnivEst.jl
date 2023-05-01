phi(u, p, t) = - p[2] * u[3] - u[2] - u[1] * p[2] + p[1] * u[1]^2 - p[1] * u[1];
u0 = [0.3, 0.1, -0.3];
p = [0.5, 0.5];
t0 = 0.0;
tf = 30.0;
ts = 1e-02;
opt = ADAM(1e-02);

env = init_env(phi, u0, p, t0, tf, ts, opt);

tfs = [5.0, 10.0];
estp = randn(length(u0) + length(p)) * 1e-03;
hp = training_routine(env, tfs, estp, 1000)

using Plots, LaTeXStrings
est_sol = get_sys_solution(env, hp[(length(u0) + 1):end], hp[1:length(u0)])
times = env.t0:env.ts:(env.tf - env.ts);
plot(times, env.y_samples, label = L"y(t)")
plot!(times, est_sol[1, :], label = L"\hat y(t)")
plot!(legendfontsize = 15)
plot!(legend = :topleft)
plot!(grid = :false)
xlabel!("t")
