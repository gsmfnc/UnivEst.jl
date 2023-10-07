u0 = [0.0, -0.4];
p = [-1.0, -0.4];
phi(u, p, t) = p[1] * (1 - u[2]) * u[2] + p[2] * u[1] + sin(t);

fq1 = init_system_obs(phi, u0, p = p, t0 = 0.0, tf = 50.0, ts = 1e-02);
fq1_sol, fq1_y = get_sys_solution(fq1);

# Noise signal
noise_samples = randn(length(fq1_y), 1) * 0.05;
samples = fq1_y + noise_samples;
plot(samples)
plot!(noise_samples)

# Training
N = 16;
n = 2;
sigma(x) = tanh(x) + 1;
W(p) = reshape(p[1:N], 1, :);
V(p) = reshape(p[N + 1:(N + N * n)], :, n);
lin(p) = p[(N + N * n + 1):end]';
phi(u, p, t) = (W(p) * sigma.(V(p) * u))[1] + lin(p) * u + sin(t);
p0 = vec(randn(N + N * n + n, 1));
fq1_est = init_system_obs(phi, u0, p = p0, t0 = 0.0, tf = 50.0, ts = 1e-02);

tfs = [2.5, 5.0];
hu0, hp, times, estps = sysobs_training(fq1_est, samples, tfs, 1000,
    save = true, callback = true);

tfs = [7.5, 12.5, 15.0, 20.0, 25.0, 30.0];
hu02, hp2, times, estps2 = sysobs_training(fq1_est, samples, tfs, 300,
    save = true, callback = true, estu0 = hu0, estp0 = hp);

tfs = [35.0, 40.0];
hu03, hp3, times, estps3 = sysobs_training(fq1_est, samples, tfs, 300,
    save = true, callback = true, estu0 = hu02, estp0 = hp2,
    opt = Adam(1e-04));

fq1_hsol, fq1_hy =
    get_sys_solution(fq1_est, u0_arg = hu02, p_arg = hp2);
plot(fq1_y)
plot!(fq1_hy)

writedlm("fq1_hp.csv", hp, ",")
writedlm("fq1_hu0.csv", hu0, ",")
writedlm("fq1_noise.csv", noise_samples, ",")
writedlm("fq1_y.csv", fq1_y, ",")
writedlm("fq1_hy.csv", fq1_hy, ",")
