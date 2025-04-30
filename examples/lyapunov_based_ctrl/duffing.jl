# Duffing oscillator (Equation (2.6)) in "Sprott, Julien C. Elegant chaos:
# algebraically simple chaotic flows. World Scientific, 2010."

phi(u, p, t) = - u[2] + u[1] - u[1]^3 + sin(0.8 * t);
u0 = [-0.9, 4.];
p = [0.];

duff = init_system_obs(phi, u0, p = p, t0 = 0.0, tf = 50.0, ts = 1e-02);
duff_sol, duff_y = get_sys_solution(duff);

# Noise signal
noise_samples = randn(length(duff_y), 1) * 0.05;
samples = duff_y + noise_samples;
plot(0.0:1e-02:(50.0-1e-02), duff_y, label = L"x_1")
plot!(0.0:1e-02:(50.0-1e-02), noise_samples,
    label = L"N_1", xlabel = L"t\ [s]")

############################ Identification
n = 2;
N = 64;
Np1 = N + N * n;
Np2 = 2 * N + N * n;
sigma(x) = tanh(x) + 1;
W1(p) = reshape(p[1:N], 1, :);
V1(p) = reshape(p[N + 1:(N + N * n)], :, n);
W2(p) = reshape(p[1:(2 * N)], 2, :);
V2(p) = reshape(p[(2 * N + 1):(2 * N + N * n)], :, n);
lin(p) = p[(Np1 + Np2 + 1):end]';

est_phi(u, p, t) = (W1(p[1:Np1]) * sigma.(V1(p[1:Np1]) *
    W2(p[(Np1 + 1):(Np1 + Np2)]) * sigma.(V2(p[(Np1 + 1):(Np1 + Np2)]) *
    u)))[1] + lin(p) * u + sin(0.8 * t);
estp0 = vec(randn(Np1 + Np2 + n, 1));
est_duff = init_system_obs(est_phi, vec(zeros(n, 1)), p = estp0, t0 = 0.0,
    tf = 50.0, ts = 1e-02);

tfs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
hu0, hp, times, estps = sysobs_training(est_duff, samples, tfs, 100,
    save = true, callback = true);
tfs = [13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0];
hu0, hp, times, estps = sysobs_training(est_duff, samples, tfs, 100,
    save = true, callback = true, estu0 = hu0, estp0 = hp, opt = Adam(1e-03));
tfs = [20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0];
hu0, hp, times, estps = sysobs_training(est_duff, samples, tfs, 100,
    save = true, callback = true, estu0 = hu0, estp0 = hp, opt = Adam(1e-03));
tfs = [28.0, 29.0, 30.0];
hu0, hp, times, estps = sysobs_training(est_duff, samples, tfs, 100,
    save = true, callback = true, estu0 = hu0, estp0 = hp, opt = Adam(1e-03));
tfs = [31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0];
hu0, hp, times, estps = sysobs_training(est_duff, samples, tfs, 100,
    save = true, callback = true, estu0 = hu0, estp0 = hp, opt = Adam(1e-03));
tfs = [41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0];
hu0, hp, times, estps = sysobs_training(est_duff, samples, tfs, 100,
    save = true, callback = true, estu0 = hu0, estp0 = hp, opt = Adam(1e-04));

############################ Controller design (with desired dynamics)
n = 2;
N = 64;
Np1 = N + N * n;
Np2 = 2 * N + N * n;
sigma(x) = tanh(x) + 1;
W1(p) = reshape(p[1:N], 1, :);
V1(p) = reshape(p[N + 1:(N + N * n)], :, n);
W2(p) = reshape(p[1:(2 * N)], 2, :);
V2(p) = reshape(p[(2 * N + 1):(2 * N + N * n)], :, n);
lin(p) = p[(Np1 + Np2 + 1):end]';

include("examples/lyapunov_based_ctrl/estimates_duffing.jl")

# parametric controller
u(x, p) = [
    0.0
    p[1] * x[1] + p[2] * x[2] + p[3] * x[1]^2 + p[4] * x[2]^2 +
        p[5] * x[1] * x[2] + p[6] * x[1]^3 + p[7] * x[2]^3 +
        p[8] * x[1]^2 * x[2] + p[9] * x[1] * x[2]^2
];

# system dynamics
est_phi(x, p, t) = (W1(p[1:Np1]) * sigma.(V1(p[1:Np1]) *
    W2(p[(Np1 + 1):(Np1 + Np2)]) * sigma.(V2(p[(Np1 + 1):(Np1 + Np2)]) *
    x)))[1] + lin(p) * x;
dyn_noctrl(x, p) = [
    x[2]
    est_phi(x, hp, 0.0)
];
f(x, p, t) = dyn_noctrl(x, p) + u(x, p);

# initialize controlled system
sys_to_ctrl = init_controlled_system(f, u);

# init desired system
using MatrixEquations
A = [
    0.  1.
    -12. -7.
];
f_des(x, p, t) = A * x;
Q = [
    1.  0.
    0.  1.
];
P = lyapc(A', Q, Q);

# Generate test points
test_points = zeros(n, 2000);
radius = 0.01;
for i = 1:1:size(test_points, 2)
    temp = randn(n);
    nz = 1 / sqrt(sum(temp.^2));
    test_points[:, i] = temp * nz * radius;
    if i % 20 == 0
        radius = radius + 0.05;
    end
end

# random initial guess
estp0 = vec(randn(9));

# training
# estp = ctrl_training(sys_to_ctrl, f_des, estp0, test_points, 1000, P);
estp = [
 -11.21462860180651
  -6.087270696547453
   0.10373871591730613
   0.04419245439221753
   0.014481493373023496
   0.0254276910100563
   0.0017761151937959777
   0.005600921274550955
   0.012832715950337567
];

############################ Numerical simulation

epsln = 0.0001;
sat_val = 40;
# complete control system
full_dyn(x, p) = [
    x[2]                        # duffing
    - x[2] + x[1] - x[1]^3      # duffing
    x[4]                     + epsln^-1 * 0.30 * (x[1] - x[3]) # hgo
    est_phi(x[3:4], hp, 0.0) + epsln^-2 * 0.02 * (x[1] - x[3]) # hgo
];
full_u(x, p) = [
    0.0
    max(-sat_val, min(
        p[1] * x[3] + p[2] * x[4] + p[3] * x[3]^2 + p[4] * x[4]^2 +
        p[5] * x[3] * x[4] + p[6] * x[3]^3 + p[7] * x[4]^3 +
        p[8] * x[3]^2 * x[4] + p[9] * x[3] * x[4]^2,
    sat_val))
    0.0
    0.0
];
full_f(x, p, t) = full_dyn(x, p) + full_u(x, p);

# initialize controlled system
control_sys = init_controlled_system(full_f, full_u);
desired_sys = init_controlled_system(f_des, u);

u0e = vec(randn(n, 1)) * 2.0;
u0hgo = vec(zeros(n, 1));
t_sol, sol, u_vals = get_controlled_sys_solution(control_sys, estp,
    vcat(u0e, u0hgo), tf = 15.0);
t_sol2, sol2, u_vals2 = get_controlled_sys_solution(desired_sys, estp, u0e,
    tf = 15.0);

p1 = plot(t_sol, sol[1, :], label = L"x_1");
p1 = plot!(t_sol, sol2[1, :], label = L"x_{1,des}");
p2 = plot(t_sol, sol[2, :], label = L"x_2");
p2 = plot!(t_sol, sol2[2, :], label = L"x_{2,des}");
p3 = plot(t_sol, u_vals[2, :], label = L"u(\hat x;p_{\star})",
    xlabel = L"t\ [s]");
plot(p1, p2, p3, layout = (3, 1))
