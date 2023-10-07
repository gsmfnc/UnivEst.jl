# Duffing oscillator (Equation (2.6)) in "Sprott, Julien C. Elegant chaos:
# algebraically simple chaotic flows. World Scientific, 2010."

phi(u, p, t) =
    - p[1] * u[2] - p[2] * u[1] - p[3] * u[1]^3 + p[4] * sin(p[5] * t);
u0 = [-0.9, 4.];
p = [1., -1., 1., 1., 0.8];

duff = init_system_obs(phi, u0, p = p, t0 = 0.0, tf = 50.0, ts = 1e-02);
duff_sol, duff_y = get_sys_solution(duff);

# Noise signal
noise_samples = randn(length(duff_y), 1) * 0.05;
samples = duff_y + noise_samples;
plot(samples)
plot!(noise_samples)

# Training
tfs = [2.5, 5.0, 7.5, 10.0];
hu0, hp, times, estps = sysobs_training(duff, samples, tfs, 1000, save = true,
    callback = true);
estps0 = estps;

tfs = [12.5, 15.0, 20.0, 25.0, 30.0, 35.0];
hu0, hp, times, estps = sysobs_training(duff, samples, tfs, 300, save = true,
    estu0 = estps0[1:2, end], estp0 = estps0[3:end, end], callback = true,
    opt = Adam(1e-03));
estps0 = hcat(estps0, estps);

tfs = [35.0, 40.0, 45.0, 50.0];
hu0, hp, times, estps = sysobs_training(duff, samples, tfs, 300, save = true,
    estu0 = estps0[1:2, end], estp0 = estps0[3:end, end], callback = true,
    opt = Adam(1e-04));
estps0 = hcat(estps0, estps);

tfs = [50.0];
hu0, hp, times, estps = sysobs_training(duff, samples, tfs, 1000, save = true,
    estu0 = estps0[1:2, end], estp0 = estps0[3:end, end], callback = true,
    opt = Adam(1e-04));
estps0 = hcat(estps0, estps);

hu0 = [-0.89692, 3.95929];
hp = [0.989977, -0.9989, 1.00302, -0.992544, -0.800269];

duff_hsol, duff_hy =
    get_sys_solution(duff, u0_arg = hu0, p_arg = hp);
plot(duff_y)
plot!(duff_hy)

estps = estps0;

writedlm("duffing_estps.csv", estps, ",")
writedlm("duffing_noise.csv", noise_samples, ",")
writedlm("duffing_y.csv", duff_y, ",")
writedlm("duffing_hy.csv", duff_hy, ",")
