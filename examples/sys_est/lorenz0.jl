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
sign_a = 1e-03;

## Training without time derivatives
lorenz = init_system(f, h, u0, p = p, t0 = 0.0, tf = 50.0, ts = 1e-02);
lor_sol, lor_y = get_sys_solution(lorenz);

# Noise signal
signum(u, p, t) = 20.0 * [
    u[2]
    - sign(u[1]) + 20.0^-1 * (sin(t) + 10.0 * sin(10.0 * t))
];
d(u, p, t) = sign_a * (u[1] + u[2] + u[1] * u[2]);
signum_u0 = [1.0, 0.1];
signum_noise = init_system(signum, d, signum_u0, t0 = 0.0, tf = lorenz.tf,
                            ts = lorenz.ts);
noise_sol, noise_samples = get_sys_solution(signum_noise);

plot(lor_y)
plot!(noise_samples)
plot!(lor_y + noise_samples)

samples = lor_y + noise_samples;

tfs = [2.5, 5.0, 7.5, 10.0, 15.0];
hu0, hp, times, estps = sys_training(lorenz, samples, tfs, 300, save = true,
    callback = true);

## Training with time derivatives
obs_map(u, p, t) = [
    u[1]
    u[2] - u[1]
    - u[1] * u[3] - u[2] + u[1]
];
lorenz2 =
    init_system(f, h, obs_map, u0, p = p, t0 = 0.0, tf = 50.0, ts = 1e-02);

signum(u, t) = 20.0 * [
    u[2]
    - sign(u[1]) + 20.0^-1 * (sin(t) + 10.0 * sin(10.0 * t))
];
d(u) = sign_a * (u[1] + u[2] + u[1] * u[2]);
signum_u0 = [1.0, 0.1];

eps = [0.01, 0.025, 0.05];
xi, z, ds = estimate_t_derivatives(lorenz2, UnivEst.M_CASCADE, 2, eps, d,
    signum, signum_u0);

real_dy = zeros(3, size(xi, 2));
measured_output = zeros(size(xi, 2), 1);
for i = 1:1:size(xi, 2)
    real_dy[:, i] = obs_map(xi[:, i], lorenz2.p, (i - 1) * lorenz2.ts);
    measured_output[i] = real_dy[1, i] + d(ds[:, i]);
end

p1 = plot(real_dy[1, :]);
p1 = plot!(z[1, :]);
p2 = plot(real_dy[2, :]);
p2 = plot!(z[3, :]);
p3 = plot(real_dy[3, :]);
p3 = plot!(z[5, :]);
plot(p1, p2, p3, layout = (3, 1))

samples_d = hcat(measured_output, z[3, :], z[5, :]);

tfs = [10.0];
hu0, hp, times, estps = sys_training(lorenz2, samples_d, tfs, 300, save = true,
    callback = true, dtime = 2.0);
