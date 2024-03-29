# Diffusionless Lorenz (Equation (3.2)) in "Sprott, Julien C. Elegant chaos:
# algebraically simple chaotic flows. World Scientific, 2010."

u0 = [1.0, 0.0, 1.0];
p = [-1.0];
f(u, p, t) = [
    u[2] - u[1]
    - u[1] * u[3]
    u[1] * u[2] + p[1]
];
h(u, p, t) = u[1];

## Training
lorenz = init_system(f, h, u0, p = p, t0 = 0.0, tf = 50.0, ts = 1e-02);
lor_sol, lor_y = get_sys_solution(lorenz);

# Noise signal
noise_samples = randn(length(lor_y), 1) * 0.05;
samples = lor_y + noise_samples;
plot(samples)
plot!(noise_samples)

tfs = [2.5, 5.0, 7.5, 10.0, 15.0];
hu0, hp, times, estps = sys_training(lorenz, samples, tfs, 300, save = true,
    callback = true);

estps0 = estps;

tfs = [30.0, 40.0];
hu0, hp, times, estps = sys_training(lorenz, samples, tfs, 300, save = true,
    callback = true, estu0 = hu0, estp0 = hp, opt = Adam(1e-04));

estps1 = hcat(estps0, estps)

tfs = [50.0];
hu0, hp, times, estps = sys_training(lorenz, samples, tfs, 300, save = true,
    callback = true, estu0 = hu0, estp0 = hp, opt = Adam(1e-05));

estps2 = hcat(estps1, estps)

lor_hsol, lor_hy =
    get_sys_solution(lorenz, u0_arg = hu0, p_arg = hp);
plot(lor_y)
plot!(samples)
plot!(lor_hy)

writedlm("lorenz_estps.csv", estps2, ",")
writedlm("lorenz_noise.csv", noise_samples, ",")
writedlm("lorenz_y.csv", lor_y, ",")
writedlm("lorenz_hy.csv", lor_hy, ",")
