# System CS_1 from Table 6.4 in "Sprott, Julien C. Elegant chaos: algebraically
# simple chaotic flows. World Scientific, 2010."

phi(u, p, t) = - u[4] + p[1] * u[3] + p[2] * u[2] + p[3] * (u[1]^2 - 1);
u0 = [-0.9, 0.0, 0.0, 0.0];
p = [-5.2, -2.7, 4.5];

cs1 = init_system_obs(phi, u0, p = p, t0 = 0.0, tf = 50.0, ts = 1e-02);
cs1_sol, cs1_y = get_sys_solution(cs1);

# Noise signal
signum(u, p, t) = 20.0 * [
    u[2]
    - sign(u[1]) + 20.0^-1 * (sin(t) + 10.0 * sin(10.0 * t))
];
d(u, p, t) = 1e-02 * (u[1] + u[2] + u[1] * u[2]);
signum_u0 = [1.0, 0.1];
signum_noise = init_system(signum, d, signum_u0, t0 = 0.0, tf = cs1.tf,
                            ts = cs1.ts);
noise_sol, noise_samples = get_sys_solution(signum_noise);

plot(cs1_y)
plot!(noise_samples)
plot!(cs1_y + noise_samples)

samples = cs1_y + noise_samples;

tfs = [2.5, 5.0];
hu0, hp, times, estps = sysobs_training(cs1, samples, tfs, 300, save = true,
    callback = true);
estps0 = estps;

tfs = [6.0];
hu0, hp, times, estps = sysobs_training(cs1, samples, tfs, 1000, save = true,
    callback = true, estu0 = hu0, estp0 = hp);
estps1 = hcat(estps0, estps)

tfs = [8.0, 10.0, 12.5];
hu0, hp, times, estps = sysobs_training(cs1, samples, tfs, 300, save = true,
    callback = true, estu0 = hu0, estp0 = hp);
estps2 = hcat(estps1, estps)

tfs = [15.0, 17.5, 20.0];
hu0, hp, times, estps = sysobs_training(cs1, samples, tfs, 1000, save = true,
    callback = true, estu0 = hu0, estp0 = hp);
estps3 = hcat(estps2, estps);

tfs = [25.0, 30.0, 40.0, 50.0];
hu0, hp, times, estps = sysobs_training(cs1, samples, tfs, 300, save = true,
    callback = true, estu0 = hu0, estp0 = hp, opt = Adam(1e-04));
estps4 = hcat(estps3, estps);

cs1_hsol, cs1_hy = get_sys_solution(cs1, u0_arg = hu0, p_arg = hp);
plot((cs1_sol - cs1_hsol)')

writedlm("cs1_estps.csv", estps4, ",")
writedlm("cs1_noise.csv", noise_samples, ",")
writedlm("cs1_y.csv", cs1_y, ",")
writedlm("cs1_hy.csv", cs1_hy, ",")
