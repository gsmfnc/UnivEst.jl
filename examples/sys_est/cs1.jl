# System CS_1 from Table 6.4 in "Sprott, Julien C. Elegant chaos: algebraically
# simple chaotic flows. World Scientific, 2010."

phi(u, p, t) = - u[4] + p[1] * u[3] + p[2] * u[2] + p[3] * (u[1]^2 - 1);
u0 = [-0.9, 0.0, 0.0, 0.0];
p = [-5.2, -2.7, 4.5];

cs1 = init_system_obs(phi, u0, p = p, t0 = 0.0, tf = 50.0, ts = 1e-02);
cs1_sol, cs1_y = get_sys_solution(cs1);

# Noise signal
noise_samples = randn(length(cs1_y), 1) * 0.01;
samples = cs1_y + noise_samples;
plot(samples)
plot!(noise_samples)

tfs = [1.0, 2.0, 3.0, 4.0, 5.0];
hu0, hp, times, estps = sysobs_training(cs1, samples, tfs, 300, save = true,
    callback = true);
estps0 = estps;

tfs = [6.0, 7.0, 8.0, 9.0, 10.0];
hu0, hp, times, estps = sysobs_training(cs1, samples, tfs, 300, save = true,
    callback = true, estu0 = hu0, estp0 = hp, opt = Adam(1e-03));
estps1 = hcat(estps0, estps)

tfs = [11.0, 12.0, 13.0, 14.0];
hu0, hp, times, estps = sysobs_training(cs1, samples, tfs, 300, save = true,
    callback = true, estu0 = hu0, estp0 = hp, opt = Adam(1e-03));
estps2 = hcat(estps1, estps)

tfs = [15.0];
hu0, hp, times, estps = sysobs_training(cs1, samples, tfs, 1000, save = true,
    callback = true, estu0 = hu0, estp0 = hp, opt = Adam(1e-02));
estps3 = hcat(estps2, estps)

tfs = [17.5, 20.0, 25.0, 30.0];
hu0, hp, times, estps = sysobs_training(cs1, samples, tfs, 300, save = true,
    callback = true, estu0 = hu0, estp0 = hp, opt = Adam(1e-03));
estps4 = hcat(estps3, estps);

tfs = [35.0, 40.0, 45.0, 50.0];
hu0, hp, times, estps = sysobs_training(cs1, samples, tfs, 100, save = true,
    callback = true, estu0 = hu0, estp0 = hp, opt = Adam(1e-04));
estps5 = hcat(estps4, estps);

cs1_hsol, cs1_hy = get_sys_solution(cs1, u0_arg = hu0, p_arg = hp);
plot(cs1_y - cs1_hy)

writedlm("cs1_estps.csv", estps5, ",")
writedlm("cs1_noise.csv", noise_samples, ",")
writedlm("cs1_y.csv", cs1_y, ",")
writedlm("cs1_hy.csv", cs1_hy, ",")
