# Duffing oscillator (Equation (2.6)) in "Sprott, Julien C. Elegant chaos:
# algebraically simple chaotic flows. World Scientific, 2010."

phi(u, p, t) =
    - p[1] * u[2] - p[2] * u[1] - p[3] * u[1]^3 + p[4] * sin(p[5] * t);
u0 = [-0.9, 4.];
p = [1., -1., 1., 1., 0.8];

duff = init_system_obs(phi, u0, p = p, t0 = 0.0, tf = 50.0, ts = 1e-02);
duff_sol, duff_y = get_sys_solution(duff);

# Noise signal
signum(u, p, t) = 20.0 * [
    u[2]
    - sign(u[1]) + 20.0^-1 * (sin(t) + 10.0 * sin(10.0 * t))
];
d(u, p, t) = 1e-02 * (u[1] + u[2] + u[1] * u[2]);
signum_u0 = [1.0, 0.1];
signum_noise = init_system(signum, d, signum_u0, t0 = 0.0, tf = duff.tf,
                            ts = duff.ts);
noise_sol, noise_samples = get_sys_solution(signum_noise);

plot(duff_y)
plot!(noise_samples)
plot!(duff_y + noise_samples)

samples = duff_y + noise_samples;

# Training without time derivatives
tfs = [2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 20.0];
hu0, hp, times, estps = sysobs_training(duff, samples, tfs, 300, save = true,
    callback = true);
estps0 = estps;

tfs = [25.0, 30.0, 35.0];
hu0, hp, times, estps = sysobs_training(duff, samples, tfs, 300, save = true,
    estu0 = estps0[1:2, end], estp0 = estps0[3:end, end], callback = true,
    opt = Adam(1e-03));
estps0 = hcat(estps0, estps);

tfs = [35.0, 40.0, 45.0, 50.0];
hu0, hp, times, estps = sysobs_training(duff, samples, tfs, 300, save = true,
    estu0 = estps0[1:2, end], estp0 = estps0[3:end, end], callback = true,
    opt = Adam(1e-04));
estps0 = hcat(estps0, estps);

duff_hsol, duff_hy =
    get_sys_solution(duff, u0_arg = hu0, p_arg = hp);
plot(duff_y)
plot!(duff_hy)

estps = estps0;

writedlm("duffing_estps.csv", estps, ",")
writedlm("duffing_noise.csv", noise_samples, ",")
writedlm("duffing_y.csv", duff_y, ",")
writedlm("duffing_hy.csv", duff_hy, ",")
