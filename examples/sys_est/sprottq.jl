u0 = [2.0, 0.0, 0.0];
p = [0.9, 0.4];

h(u, p, t) = u[1];
f(u, p, t) = [
    p[1] - u[2]
    u[3] + p[2]
    u[1] * u[2] - u[3]
];
sprott = init_system(f, h, u0, p = p, t0 = 0.0, tf = 50.0, ts = 1e-02);
sprott_sol, sprott_y = get_sys_solution(sprott);

## Training
# Noise signal
signum(u, p, t) = 20.0 * [
    u[2]
    - sign(u[1]) + 20.0^-1 * (sin(t) + 10.0 * sin(10.0 * t))
];
d(u, p, t) = 2.5e-02 * (u[1] + u[2] + u[1] * u[2]);
signum_u0 = [1.0, 0.1];
signum_noise = init_system(signum, d, signum_u0, t0 = 0.0, tf = sprott.tf,
                            ts = sprott.ts);
noise_sol, noise_samples = get_sys_solution(signum_noise);

plot(sprott_y)
plot!(noise_samples)
plot!(sprott_y + noise_samples)

samples = sprott_y + noise_samples;

tfs = [2.5, 5.0, 7.5, 10.0];
hu0, hp, times, estps = sys_training(sprott, samples, tfs, 300, save = true,
    callback = true, estu0 = u0);

tfs = [10.0, 15.0, 20.0];
hu0, hp, times, estps = sys_training(sprott, samples, tfs, 100, save = true,
    callback = true, estu0 = u0, estp0 = hp);

tfs = [25.0, 30.0, 40.0, 50.0];
hu0, hp, times, estps = sys_training(sprott, samples, tfs, 100, save = true,
    callback = true, estu0 = u0, estp0 = hp, opt = Adam(1e-03));

sprott_hsol, sprott_hy =
    get_sys_solution(sprott, u0_arg = hu0, p_arg = hp);
plot(sprott_y)
plot!(samples)
plot!(sprott_hy)

writedlm("sprott_estps.csv", estps, ",")
writedlm("sprott_noise.csv", noise_samples, ",")
writedlm("sprott_y.csv", sprott_y, ",")
writedlm("sprott_hy.csv", sprott_hy, ",")
